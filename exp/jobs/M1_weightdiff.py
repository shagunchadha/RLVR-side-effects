"""M1: weight-difference spectra between consecutive post-training stages (RLVR-DPO vs DPO-SFT).

Memory and disk (rewritten Oct 28 after the first molab run was reclaimed at ~160/355 tensors):
  - Hub checkpoints are streamed one safetensors file at a time: each file is fetched with hf_hub_download into
    --work-dir (not the HF cache) and deleted as soon as no remaining tensor of this shard needs it. The three 7B
    checkpoints share one tensor->file map, so at most one file per checkpoint (~5 GB each) is on disk at a time.
    The hf-xet chunk cache is disabled (HF_XET_CHUNK_CACHE_SIZE_BYTES=0) unless set by the caller.
  - Tensors are processed one at a time: the three copies are read straight onto --device (safe_open device=...),
    the SVDs run there, and everything is freed (del + torch.cuda.empty_cache()) before the next tensor. No full
    model is ever in CPU RAM.
  - Every finished tensor is appended (flushed, fsynced) as one row to the progress CSV
    ($RESULTS_DIR/M1/M1_progress_shard<i>of<n>.csv, or --progress). A restart skips tensors already in it, so a crash
    costs at most the tensor in flight. A torn last line is dropped on load. exp/m1_sync.py copies the CSV and the
    log off the notebook while the job runs.
  - Every 10 tensors (and around each download / delete) one RES line logs available RAM, process RSS, cgroup
    memory, work-dir size and free disk, and GPU memory. Each tensor logs a START line before it is loaded, so the
    log shows the tensor in flight when a process dies.

Per tensor (name present in all three checkpoints) and comparison c in {RLVR-DPO, DPO-SFT}, with dW = W_later - W_earlier:
  w_norm      ||W_earlier||_F
  dw_norm     ||dW||_F
  rel_dw      ||dW||_F / ||W_earlier||_F
  eff_rank    exp(H(p)), p_i = s_i / sum(s)   (Roy & Vetterli 2007; 2-D tensors only)
  stable_rank ||dW||_F^2 / s_1^2              (2-D only)
  num_rank    number of singular values > 1e-6 * s_1 (2-D only; angles beyond min(num_rank) are not meaningful)
  sv_1..sv_32 top singular values of dW       (2-D only)
Per 2-D tensor (subspace overlap between the two updates):
  cos_left_k_j / cos_right_k_j  cosines of the principal angles between the top-k left (right) singular subspaces of
                                dW(RLVR-DPO) and dW(DPO-SFT), k in {8, 32}, j = 1..k (descending)
  mean_cos2_left_k / _right_k   mean squared cosine (= ||U1^T U2||_F^2 / k); random k-subspaces of R^n give ~k/n
The SVD is exact (torch.linalg.svd on the device; the full spectrum is needed for eff_rank / num_rank / stable_rank).
Only the top-32 singular vectors are kept, and the rest is freed at once.

Sharding: tensors are split round-robin over --shard i/n. When a shard's progress CSV covers all its tensors, the
shard parquet is written to the store. When every shard of the config exists (or with --assemble),
results/sprint/M1/{weightdiff.csv, subspace_angles.csv, heatmap_*.png} are written.
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .. import config as C
from .. import store as S

LADDER = ("sft", "dpo", "rlvr")
COMPARISONS = (("rlvr", "dpo"), ("dpo", "sft"))
K_TOP = 32
K_ANGLES = (8, 32)
_LAYER = re.compile(r"\.layers\.(\d+)\.(.+)$")
PROGRESS_COLS = ["config_hash", "idx", "tensor", "files", "status", "seconds", "metrics_json", "angle_json"]
RES_EVERY = 10
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def add_args(p):
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--assemble", action="store_true", help="only assemble CSV + heatmaps from existing shards")
    p.add_argument("--max-tensors", type=int, default=None, help="debug: process only the first N tensors")
    p.add_argument("--work-dir", default=None, help="where safetensors files are streamed (default $RESULTS_DIR/M1/_dl)")
    p.add_argument("--progress", default=None, help="progress CSV (default $RESULTS_DIR/M1/M1_progress_shard<i>of<n>.csv)")
    p.add_argument("--keep-files", action="store_true", help="do not delete downloaded safetensors files")


# ------------------------------------------------------------------ resources
def _meminfo():
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) * 1024
        return d
    except OSError:
        return {}


def _read_int(path):
    try:
        v = open(path).read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def _rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def resources(work_dir=None, device="cpu"):
    """One-line snapshot: available RAM, RSS, cgroup memory, work-dir bytes, free disk, GPU memory (GB)."""
    gb = lambda x: "na" if x is None else f"{x / 1e9:.1f}"
    mi = _meminfo()
    cg = _read_int("/sys/fs/cgroup/memory.current") or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    cgmax = _read_int("/sys/fs/cgroup/memory.max") or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    parts = [f"ram_avail={gb(mi.get('MemAvailable'))}", f"ram_total={gb(mi.get('MemTotal'))}", f"rss={gb(_rss())}",
             f"cgroup={gb(cg)}/{gb(cgmax)}"]
    if work_dir is not None:
        try:
            parts += [f"workdir={gb(_du(work_dir))}", f"disk_free={gb(shutil.disk_usage(work_dir).free)}"]
        except OSError:
            parts.append("disk=na")
    if str(device).startswith("cuda"):
        import torch
        free, total = torch.cuda.mem_get_info()
        parts += [f"gpu_free={gb(free)}/{gb(total)}", f"gpu_alloc={gb(torch.cuda.memory_allocated())}",
                  f"gpu_peak={gb(torch.cuda.max_memory_allocated())}"]
    return " ".join(parts)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ checkpoint access
class Checkpoint:
    """Access to one checkpoint's safetensors files, one tensor at a time.

    local_dir: files already on disk (never deleted). repo + sha: files are fetched one at a time into work_dir with
    hf_hub_download and removed with release(). get() reads one tensor straight onto the requested device."""

    def __init__(self, repo=None, sha=None, local_dir=None, work_dir=None, key=None):
        self.repo, self.sha, self.local_dir = repo, sha, local_dir
        self.work_dir = Path(work_dir) if work_dir else None
        self.key = key or (repo or str(local_dir)).split("/")[-1]
        self._handles, self._paths = {}, {}
        idx = self._small("model.safetensors.index.json")
        if idx:
            self.weight_map = json.load(open(idx, encoding="utf-8"))["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(self.fetch("model.safetensors"), "pt") as h:
                self.weight_map = {k: "model.safetensors" for k in h.keys()}

    @property
    def streaming(self):
        return self.local_dir is None

    def _dir(self):
        d = (self.work_dir or Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M1" / "_dl") / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _small(self, name):
        """A small metadata file (the index), or None if the repo has none."""
        if not self.streaming:
            p = Path(self.local_dir) / name
            return str(p) if p.exists() else None
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(self.repo, name, revision=self.sha, local_dir=str(self._dir()))
        except Exception:
            return None

    def fetch(self, fn):
        """Path of safetensors file fn, downloading it into the work dir if needed."""
        if fn in self._paths:
            return self._paths[fn]
        if not self.streaming:
            p = str(Path(self.local_dir) / fn)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(self.repo, fn, revision=self.sha, local_dir=str(self._dir()))
        self._paths[fn] = p
        return p

    def release(self, fn, keep=False):
        """Close fn and, for a streamed checkpoint, delete it from disk."""
        self._handles.pop(fn, None)
        p = self._paths.pop(fn, None)
        if p and self.streaming and not keep:
            for q in (Path(p), self._dir() / ".cache" / "huggingface" / "download" / (fn + ".metadata")):
                try:
                    q.unlink()
                except OSError:
                    pass

    def on_disk(self):
        return sorted(self._paths)

    def names(self):
        return set(self.weight_map)

    def get(self, name, device="cpu"):
        from safetensors import safe_open
        fn = self.weight_map[name]
        if fn not in self._handles:
            self._handles[fn] = safe_open(self.fetch(fn), "pt", device=str(device))
        return self._handles[fn].get_tensor(name)


# ------------------------------------------------------------------ metrics
def parse_name(name):
    m = _LAYER.search(name)
    if m:
        return int(m.group(1)), m.group(2).replace(".weight", "")
    return -1, name.replace("model.", "").replace(".weight", "")


def spectrum_metrics(dw, w_earlier):
    import torch
    out = {"w_norm": float(torch.linalg.norm(w_earlier)), "dw_norm": float(torch.linalg.norm(dw))}
    out["rel_dw"] = out["dw_norm"] / out["w_norm"] if out["w_norm"] > 0 else float("nan")
    if dw.ndim != 2:
        return out, None, None
    U, s, Vh = torch.linalg.svd(dw, full_matrices=False)
    k = min(K_TOP, s.shape[0])
    U_top, V_top = U[:, :k].clone(), Vh[:k].T.clone()          # keep only what the angles need
    del U, Vh
    s_np = s.double().cpu().numpy()
    del s
    tot = s_np.sum()
    if tot > 0:
        p = s_np / tot
        p = p[p > 0]
        out["eff_rank"] = float(math.exp(-(p * np.log(p)).sum()))
        # numerical rank (s_i > 1e-6 * s_1): principal angles beyond it involve arbitrary null-space directions
        out["num_rank"] = int((s_np > 1e-6 * s_np[0]).sum())
        out["stable_rank"] = float((s_np ** 2).sum() / s_np[0] ** 2)
    else:
        out["eff_rank"] = out["stable_rank"] = 0.0
        out["num_rank"] = 0
    for j in range(K_TOP):
        out[f"sv_{j + 1}"] = float(s_np[j]) if j < len(s_np) else float("nan")
    return out, U_top, V_top


def principal_cosines(A, B, k):
    """Cosines of the principal angles between span(A[:, :k]) and span(B[:, :k]) (orthonormal columns), descending."""
    import torch
    k = min(k, A.shape[1], B.shape[1])
    return torch.linalg.svdvals(A[:, :k].T @ B[:, :k]).clamp(0, 1).double().cpu().numpy()


def tensor_rows(name, tensors, device):
    """tensors: {ckpt: torch tensor}. Returns (metric rows, angle row or None)."""
    import torch
    layer, module = parse_name(name)
    T = {k: v.to(device=device, dtype=torch.float32) for k, v in tensors.items()}
    rows, bases = [], {}
    for later, earlier in COMPARISONS:
        dw = T[later] - T[earlier]
        m, U, V = spectrum_metrics(dw, T[earlier])
        del dw
        rows.append({"tensor": name, "layer": layer, "module": module, "shape": list(T[later].shape),
                     "comparison": f"{later.upper()}-{earlier.upper()}", **m})
        bases[(later, earlier)] = (U, V)
    angle = None
    (U1, V1), (U2, V2) = bases[COMPARISONS[0]], bases[COMPARISONS[1]]
    if U1 is not None:
        angle = {"tensor": name, "layer": layer, "module": module, "shape": list(T["rlvr"].shape),
                 "n_left": int(U1.shape[0]), "n_right": int(V1.shape[0])}
        for k in K_ANGLES:
            cl, cr = principal_cosines(U1, U2, k), principal_cosines(V1, V2, k)
            angle[f"mean_cos2_left_{k}"] = float((cl ** 2).mean())
            angle[f"mean_cos2_right_{k}"] = float((cr ** 2).mean())
            angle[f"random_baseline_left_{k}"] = k / U1.shape[0]
            angle[f"random_baseline_right_{k}"] = k / V1.shape[0]
            for j, c in enumerate(cl):
                angle[f"cos_left_{k}_{j + 1}"] = float(c)
            for j, c in enumerate(cr):
                angle[f"cos_right_{k}_{j + 1}"] = float(c)
    del T, bases, U1, V1, U2, V2
    return rows, angle


# ------------------------------------------------------------------ progress CSV
def load_progress(path, cfg_hash):
    """{tensor: row} for the completed tensors in the progress CSV. A torn last line (crash mid-write) is dropped and
    the file rewritten without it; a row from a different config aborts."""
    path = Path(path)
    if not path.exists():
        return {}
    done, good, bad = {}, [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                json.loads(r["metrics_json"])
                json.loads(r["angle_json"])
                float(r["seconds"])
            except (TypeError, ValueError, KeyError):
                bad += 1
                continue
            if r["config_hash"] != cfg_hash:
                raise SystemExit(f"{path} holds rows of config {r['config_hash']}, not {cfg_hash}; use another --progress")
            done[r["tensor"]] = r
            good.append(r)
    if bad:
        log(f"progress: dropped {bad} unreadable row(s) (torn write) from {path}")
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PROGRESS_COLS)
            w.writeheader()
            w.writerows(good)
        os.replace(tmp, path)
    return done


def append_progress(path, row):
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------ job
def job_config():
    return {"job": "M1", "ladder": {k: list(C.ckpt(k)) for k in LADDER}, "comparisons": [list(c) for c in COMPARISONS],
            "k_top": K_TOP, "k_angles": list(K_ANGLES), "dtype": "float32", "version": 1}


def ordered_by_file(ckpts, names):
    """names grouped by their (file per checkpoint) tuple, groups in sorted order, names sorted within a group."""
    key = lambda nm: tuple(ckpts[k].weight_map[nm] for k in LADDER)
    return sorted(names, key=lambda nm: (key(nm), nm))


def run_shard(ckpts, names, device, store, i, n, progress, keep_files=False, work_dir=None):
    import torch
    done = load_progress(progress, store.hash)
    order = ordered_by_file(ckpts, names)
    todo = [nm for nm in order if nm not in done]
    log(f"M1 shard {i}/{n}: {len(names)} tensors, {len(done)} already in {progress}, {len(todo)} to do, device {device}")
    log("RES start", resources(work_dir, device))
    # how many remaining tensors still need each (checkpoint, file): a file is deleted when its count reaches 0
    need = {}
    for nm in todo:
        for k in LADDER:
            need[(k, ckpts[k].weight_map[nm])] = need.get((k, ckpts[k].weight_map[nm]), 0) + 1
    for t, name in enumerate(todo):
        idx = order.index(name)
        files = {k: ckpts[k].weight_map[name] for k in LADDER}
        for k in LADDER:
            if files[k] not in ckpts[k].on_disk():
                log(f"FETCH {k} {files[k]} (on disk for {k}: {ckpts[k].on_disk() or 'none'})")
                t0 = time.time()
                ckpts[k].fetch(files[k])
                log(f"FETCHED {k} {files[k]} in {time.time() - t0:.0f}s | RES", resources(work_dir, device))
        log(f"START {idx + 1}/{len(order)} {name}")
        t0 = time.time()
        tensors = {k: ckpts[k].get(name, device) for k in LADDER}
        shapes = {tuple(v.shape) for v in tensors.values()}
        if len(shapes) != 1:
            status, metrics, angle = "shape_mismatch", [{"tensor": name, "layer": parse_name(name)[0],
                                                          "module": parse_name(name)[1], "shape": [],
                                                          "comparison": "SHAPE_MISMATCH", "w_norm": float("nan")}], None
        else:
            status = "ok"
            metrics, angle = tensor_rows(name, tensors, device)
        del tensors
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        append_progress(progress, {"config_hash": store.hash, "idx": idx, "tensor": name, "files": json.dumps(files),
                                   "status": status, "seconds": f"{time.time() - t0:.2f}",
                                   "metrics_json": json.dumps(metrics), "angle_json": json.dumps(angle)})
        for k in LADDER:
            need[(k, files[k])] -= 1
            if need[(k, files[k])] == 0:
                ckpts[k].release(files[k], keep=keep_files)
                log(f"RELEASED {k} {files[k]}{' (kept)' if keep_files else ' (deleted)'}")
        if (t + 1) % RES_EVERY == 0 or t + 1 == len(todo):
            log(f"RES {len(done) + t + 1}/{len(order)} done |", resources(work_dir, device))
    done = load_progress(progress, store.hash)
    missing = [nm for nm in names if nm not in done]
    if missing:
        raise SystemExit(f"progress CSV lacks {len(missing)} tensors of this shard after the loop: {missing[:5]}")
    rows = []
    for nm in order:
        for x in json.loads(done[nm]["metrics_json"]):
            rows.append({"kind": "metric", "tensor": nm, "json": json.dumps(x)})
    for nm in order:
        a = json.loads(done[nm]["angle_json"])
        if a is not None:
            rows.append({"kind": "angle", "tensor": nm, "json": json.dumps(a)})
    return store.write(i, n, rows, model_repo=";".join(C.ckpt(k)[0] for k in LADDER),
                       model_sha=";".join(C.ckpt(k)[1] for k in LADDER), extra_meta={"n_tensors": len(names)})


def assemble(store, n, out_dir):
    metric, angle = [], []
    for i in range(n):
        for r in store.read(i).to_pylist():
            d = json.loads(r["json"])
            d.pop("payload", None)
            (metric if r["kind"] == "metric" else angle).append(d)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(path, rows):
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r.get("layer", -1), r.get("module", ""), r.get("comparison", ""))):
                w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    write(out_dir / "weightdiff.csv", metric)
    write(out_dir / "subspace_angles.csv", angle)
    heatmaps(metric, angle, out_dir)
    (out_dir / "provenance.json").write_text(json.dumps({"config": store.cfg, "config_hash": store.hash,
                                                         "git_sha": C.git_sha(), "n_shards": n,
                                                         "package_versions": C.package_versions()}, indent=1))
    return len(metric), len(angle)


def heatmaps(metric, angle, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = sorted({r["layer"] for r in metric if r["layer"] >= 0})
    mods = sorted({r["module"] for r in metric if r["layer"] >= 0 and r.get("shape") and len(r["shape"]) == 2})
    if not layers or not mods:
        return

    def grid(comp, key, rows):
        g = np.full((len(mods), len(layers)), np.nan)
        for r in rows:
            if r["layer"] >= 0 and r["module"] in mods and r.get("comparison", comp) == comp and key in r:
                g[mods.index(r["module"]), layers.index(r["layer"])] = r[key]
        return g
    panels = [("RLVR-DPO", "rel_dw", metric, "rel ||dW||/||W|| RLVR-DPO"),
              ("DPO-SFT", "rel_dw", metric, "rel ||dW||/||W|| DPO-SFT"),
              ("RLVR-DPO", "eff_rank", metric, "effective rank of dW RLVR-DPO"),
              ("DPO-SFT", "eff_rank", metric, "effective rank of dW DPO-SFT"),
              (None, "mean_cos2_left_32", angle, "mean cos^2, top-32 left subspaces (RLVR-DPO vs DPO-SFT)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(8, len(layers) * 0.35), 2.2 * len(panels) + 1))
    for ax, (comp, key, rows, title) in zip(np.atleast_1d(axes), panels):
        g = grid(comp, key, [r for r in rows if comp is None or r.get("comparison") == comp])
        im = ax.imshow(g, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(mods)), mods, fontsize=6)
        ax.set_xticks(range(len(layers)), layers, fontsize=6)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_M1.png", dpi=150)
    plt.close(fig)
    # ratio panel
    a, b = grid("RLVR-DPO", "rel_dw", [r for r in metric if r.get("comparison") == "RLVR-DPO"]), \
        grid("DPO-SFT", "rel_dw", [r for r in metric if r.get("comparison") == "DPO-SFT"])
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 3))
    im = ax.imshow(np.log10(a / b), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(mods)), mods, fontsize=6)
    ax.set_xticks(range(len(layers)), layers, fontsize=6)
    ax.set_title("log10( rel_dw(RLVR-DPO) / rel_dw(DPO-SFT) )", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_M1_ratio.png", dpi=150)
    plt.close(fig)


def run(args, ckpts=None, out_dir=None):
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")     # no hidden multi-GB chunk cache beside the work dir
    import torch
    ja = args.job_args
    device = ja.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = job_config()
    store = S.Store("M1", "sft-dpo-rlvr", cfg)
    i, n = S.parse_shard(args.shard)
    out_dir = out_dir or (C.RESULTS_SPRINT / "M1")
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M1"
    root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(getattr(ja, "work_dir", None) or root / "_dl")
    progress = Path(getattr(ja, "progress", None) or root / f"M1_progress_shard{i}of{n}.csv")
    if not ja.assemble:
        if store.done(i):
            print(f"M1 shard {i}/{n} exists; skipping")
        else:
            ckpts = ckpts or {k: Checkpoint(*C.ckpt(k), work_dir=work_dir, key=k) for k in LADDER}
            common = sorted(set.intersection(*(c.names() for c in ckpts.values())))
            missing = {k: sorted(c.names() - set(common)) for k, c in ckpts.items()}
            if any(missing.values()):
                print("tensors not present in all three checkpoints (skipped):", {k: v[:5] for k, v in missing.items()})
            names = S.shard_slice(common, i, n)[: ja.max_tensors]
            print("wrote", run_shard(ckpts, names, device, store, i, n, progress,
                                     keep_files=getattr(ja, "keep_files", False), work_dir=work_dir))
    if all(store.done(j) for j in range(n)):
        m, a = assemble(store, n, out_dir)
        print(f"assembled {m} metric rows and {a} angle rows into {out_dir}")
    else:
        print(f"{sum(store.done(j) for j in range(n))}/{n} shards present; assemble after the rest finish")
    return 0
