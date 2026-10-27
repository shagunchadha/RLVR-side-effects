"""M1: weight-difference spectra between consecutive post-training stages (RLVR-DPO vs DPO-SFT).

No full model is ever held in memory: tensors are read one at a time through safetensors' memory-mapped safe_open
from each checkpoint's shards (downloaded to the HF cache on first use; ~15 GB per 7B checkpoint).

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

Sharding: tensors are split round-robin over --shard i/n; each shard is a parquet in the store. When every shard of the
config exists (or with --assemble), results/sprint/M1/{weightdiff.csv, subspace_angles.csv, heatmap_*.png} are written.
"""
import json
import math
import re

import numpy as np

from .. import config as C
from .. import store as S

LADDER = ("sft", "dpo", "rlvr")
COMPARISONS = (("rlvr", "dpo"), ("dpo", "sft"))
K_TOP = 32
K_ANGLES = (8, 32)
_LAYER = re.compile(r"\.layers\.(\d+)\.(.+)$")


def add_args(p):
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--assemble", action="store_true", help="only assemble CSV + heatmaps from existing shards")
    p.add_argument("--max-tensors", type=int, default=None, help="debug: process only the first N tensors")


# ------------------------------------------------------------------ checkpoint access
class Checkpoint:
    """Lazy, memory-mapped access to one checkpoint's safetensors files (local dir or HF repo@sha)."""

    def __init__(self, repo=None, sha=None, local_dir=None):
        self.repo, self.sha, self.local_dir = repo, sha, local_dir
        self._handles = {}
        idx = self._file("model.safetensors.index.json", optional=True)
        if idx:
            self.weight_map = json.load(open(idx, encoding="utf-8"))["weight_map"]
        else:
            from safetensors import safe_open
            f = self._file("model.safetensors")
            with safe_open(f, "pt") as h:
                self.weight_map = {k: "model.safetensors" for k in h.keys()}

    def _file(self, name, optional=False):
        if self.local_dir is not None:
            p = f"{self.local_dir}/{name}"
            import os
            if os.path.exists(p):
                return p
            if optional:
                return None
            raise FileNotFoundError(p)
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(self.repo, name, revision=self.sha)
        except Exception:
            if optional:
                return None
            raise

    def names(self):
        return set(self.weight_map)

    def get(self, name):
        from safetensors import safe_open
        fn = self.weight_map[name]
        if fn not in self._handles:
            self._handles[fn] = safe_open(self._file(fn), "pt")
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
    s_np = s.double().cpu().numpy()
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
    return out, U, Vh.T


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
    return rows, angle


# ------------------------------------------------------------------ job
def job_config():
    return {"job": "M1", "ladder": {k: list(C.ckpt(k)) for k in LADDER}, "comparisons": [list(c) for c in COMPARISONS],
            "k_top": K_TOP, "k_angles": list(K_ANGLES), "dtype": "float32", "version": 1}


def run_shard(ckpts, names, device, store, i, n):
    import torch
    metric_rows, angle_rows = [], []
    for t, name in enumerate(names):
        tensors = {k: ckpts[k].get(name) for k in LADDER}
        shapes = {tuple(v.shape) for v in tensors.values()}
        if len(shapes) != 1:
            metric_rows.append({"tensor": name, "layer": parse_name(name)[0], "module": parse_name(name)[1],
                                "shape": [], "comparison": "SHAPE_MISMATCH", "w_norm": float("nan")})
            continue
        r, a = tensor_rows(name, tensors, device)
        for x in r:
            x["kind"] = "metric"
            x["payload"] = json.dumps(x)
        metric_rows += r
        if a is not None:
            angle_rows.append(a)
        del tensors
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if (t + 1) % 20 == 0:
            print(f"  shard {i}/{n}: {t + 1}/{len(names)} tensors", flush=True)
    rows = [{"kind": "metric", "tensor": x["tensor"], "json": json.dumps(x)} for x in metric_rows]
    rows += [{"kind": "angle", "tensor": x["tensor"], "json": json.dumps(x)} for x in angle_rows]
    return store.write(i, n, rows, model_repo=";".join(C.ckpt(k)[0] for k in LADDER),
                       model_sha=";".join(C.ckpt(k)[1] for k in LADDER), extra_meta={"n_tensors": len(names)})


def assemble(store, n, out_dir):
    import csv
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
    import torch
    ja = args.job_args
    device = ja.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = job_config()
    store = S.Store("M1", "sft-dpo-rlvr", cfg)
    i, n = S.parse_shard(args.shard)
    out_dir = out_dir or (C.RESULTS_SPRINT / "M1")
    if not ja.assemble:
        if store.done(i):
            print(f"M1 shard {i}/{n} exists; skipping")
        else:
            ckpts = ckpts or {k: Checkpoint(*C.ckpt(k)) for k in LADDER}
            common = sorted(set.intersection(*(c.names() for c in ckpts.values())))
            missing = {k: sorted(c.names() - set(common)) for k, c in ckpts.items()}
            if any(missing.values()):
                print("tensors not present in all three checkpoints (skipped):", {k: v[:5] for k, v in missing.items()})
            names = S.shard_slice(common, i, n)[: ja.max_tensors]
            print(f"M1 shard {i}/{n}: {len(names)} of {len(common)} tensors on {device}", flush=True)
            print("wrote", run_shard(ckpts, names, device, store, i, n))
    if all(store.done(j) for j in range(n)):
        m, a = assemble(store, n, out_dir)
        print(f"assembled {m} metric rows and {a} angle rows into {out_dir}")
    else:
        print(f"{sum(store.done(j) for j in range(n))}/{n} shards present; assemble after the rest finish")
    return 0
