"""M1b (exploratory, registration amendment 2): how much of the M1 weight differences is real signal?

    python -m exp.run --job M1b [--shuffles 5] [--device cuda]

Per tensor present in all three checkpoints and comparison c in {RLVR-DPO, DPO-SFT} (W_l later, W_e earlier):
  (1) share_changed      share of elements with W_l != W_e as bf16 values (exact; +0 == -0)
      ulp_*              distribution of |W_l - W_e| in bf16 ulps (distance between the bf16 values in the ordered
                         bit representation): bins 0, 1, 2, 3, 4-7, 8-15, 16-255, >=256; ulp_median_changed
  (2) shuffle null (2-D only), dW = W_l - W_e in float32: the entries of dW are permuted at random --shuffles times
      (seeded per tensor and comparison); null_s1_max = the largest top singular value over the shuffles.
      signal_rank = number of singular values of dW above null_s1_max; signal_energy = share of ||dW||_F^2 in those
      directions; signal_rel = sqrt(sum of their s_i^2) / ||W_e||_F
      Limitation: the null keeps the entries and destroys their arrangement, so it detects structure beyond the
      entry distribution. A very sparse update (few changed elements) has nearly the same spectrum as its shuffles
      and cannot exceed the null even when it is low-rank; read signal_rank together with share_changed.
  (3) cos_dw_w           <dW, W_e> / (||dW|| ||W_e||); rel_dw_orth = ||dW minus its projection on W_e|| / ||W_e||.
      q_proj and k_proj are flagged (qk_norm = True): the model applies RMSNorm to q and k after the projection, so a
      dW parallel to W is (largely) absorbed by the norm.
  (4) is answered in the summary: does the k_proj concentration among the largest RLVR-DPO updates survive ranking by
      signal_rel and by rel_dw_orth instead of rel_dw?
Streams the checkpoints like M1 (one safetensors file per checkpoint on disk, deleted after use), one tensor at a
time on the device, exact SVDs (torch.linalg.svdvals). Per-tensor progress file; pre-flight and RES lines.
Outputs: results/sprint/M1b/{m1b_tensors.csv, summary.md}.
"""
import csv
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np

from .. import config as C
from .. import resources as R
from . import M1_weightdiff as M1

ULP_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 255), (256, None)]
QK = ("self_attn.q_proj", "self_attn.k_proj")


def add_args(p):
    p.add_argument("--device", default=None)
    p.add_argument("--shuffles", type=int, default=5)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--summary-only", action="store_true", help="rebuild the CSV and summary from the progress file")


def job_config(a):
    return {"job": "M1b", "ladder": {k: list(C.ckpt(k)) for k in M1.LADDER},
            "comparisons": [list(c) for c in M1.COMPARISONS], "shuffles": a.shuffles, "ulp_bins": ULP_BINS,
            "svd": "torch.linalg.svdvals float32", "version": 1}


def ordered_bits(w):
    """Float tensor -> int64 keys whose differences are distances in ulps of the stored dtype (bf16 for the Olmo
    checkpoints): monotone in the value, +0 and -0 both map to 0."""
    import torch
    size = w.element_size()
    b = w.contiguous().view({2: torch.int16, 4: torch.int32}[size]).to(torch.int64)
    mag = b & ((1 << (8 * size - 1)) - 1)
    return torch.where(b < 0, -mag, mag)


def compare(wl, we, name, comp, shuffles, device):
    import torch
    layer, module = M1.parse_name(name)
    row = {"tensor": name, "layer": layer, "module": module, "shape": list(we.shape), "comparison": comp,
           "qk_norm": module in QK, "n": we.numel(), "dtype": str(we.dtype).replace("torch.", "")}
    N = we.numel()
    row["share_changed"] = int((wl != we).sum()) / N
    d = (ordered_bits(wl) - ordered_bits(we)).abs()
    for lo, hi in ULP_BINS:
        m = (d >= lo) if hi is None else ((d >= lo) & (d <= hi))
        row[f"ulp_{lo}" + ("+" if hi is None else (f"_{hi}" if hi != lo else ""))] = int(m.sum()) / N
    ch = d[d > 0]
    row["ulp_median_changed"] = float(ch.float().median()) if ch.numel() else 0.0
    del d, ch
    dw, w = wl.float() - we.float(), we.float()
    nw, ndw = float(torch.linalg.norm(w)), float(torch.linalg.norm(dw))
    dot = float((dw * w).sum())
    row["rel_dw"] = ndw / nw if nw else float("nan")
    row["cos_dw_w"] = dot / (ndw * nw) if ndw and nw else 0.0
    row["rel_dw_orth"] = float(torch.linalg.norm(dw - (dot / nw ** 2) * w)) / nw if nw else float("nan")
    if dw.ndim == 2 and ndw > 0:
        s = torch.linalg.svdvals(dw).double().cpu().numpy()
        g = torch.Generator(device=dw.device).manual_seed(zlib.crc32(f"{name}|{comp}".encode()))
        flat, tops = dw.flatten(), []
        for _ in range(shuffles):
            perm = torch.randperm(flat.numel(), generator=g, device=dw.device)
            tops.append(float(torch.linalg.svdvals(flat[perm].view_as(dw))[0]))
            del perm
        thr = max(tops)
        r = int((s > thr).sum())
        row.update({"s1": float(s[0]), "null_s1_max": thr, "null_s1_all": json.dumps([round(x, 8) for x in tops]),
                    "signal_rank": r, "signal_energy": float((s[:r] ** 2).sum() / (s ** 2).sum()),
                    "signal_rel": float(math.sqrt((s[:r] ** 2).sum())) / nw})
    elif dw.ndim == 2:
        row.update({"s1": 0.0, "null_s1_max": 0.0, "null_s1_all": "[]", "signal_rank": 0, "signal_energy": 0.0,
                    "signal_rel": 0.0})
    del dw, w
    return row


def run(args, ckpts=None, out_dir=None):
    import torch
    a = args.job_args
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = job_config(a)
    h = C.config_hash(cfg)
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M1b"
    work = Path(a.work_dir or root / "_dl")
    prog = R.Progress(root / "progress.csv", h)
    out_dir = Path(out_dir or C.RESULTS_SPRINT / "M1b")
    if not a.summary_only:
        ckpts = ckpts or {k: M1.Checkpoint(*C.ckpt(k), work_dir=work, key=k) for k in M1.LADDER}
        names = sorted(set.intersection(*(c.names() for c in ckpts.values())))[: a.max_tensors]
        order = M1.ordered_by_file(ckpts, names)
        todo = [nm for nm in order if nm not in prog.done]
        R.log(f"M1b: {len(order)} tensors, {len(order) - len(todo)} done, {len(todo)} to do, device {device}")
        res = R.ResLog(180, dirs=[work], gpu=device.startswith("cuda"))
        if todo:
            R.preflight(2 * 15, dirs=[work], gpu=device.startswith("cuda"))
        need = {}
        for nm in todo:
            for k in M1.LADDER:
                need[(k, ckpts[k].weight_map[nm])] = need.get((k, ckpts[k].weight_map[nm]), 0) + 1
        for t, name in enumerate(todo):
            files = {k: ckpts[k].weight_map[name] for k in M1.LADDER}
            for k in M1.LADDER:
                if files[k] not in ckpts[k].on_disk():
                    R.log(f"FETCH {k} {files[k]}")
                    ckpts[k].fetch(files[k])
                    res.now(f"fetched {k} {files[k]}")
            R.log(f"START {order.index(name) + 1}/{len(order)} {name}")
            t0 = time.time()
            W = {k: ckpts[k].get(name, device) for k in M1.LADDER}
            rows = [compare(W[l], W[e], name, f"{l.upper()}-{e.upper()}", a.shuffles, device)
                    if W[l].shape == W[e].shape else {"tensor": name, "comparison": "SHAPE_MISMATCH"}
                    for l, e in M1.COMPARISONS]
            del W
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            prog.add(name, rows, time.time() - t0)
            for k in M1.LADDER:
                need[(k, files[k])] -= 1
                if need[(k, files[k])] == 0:
                    ckpts[k].release(files[k])
                    R.log(f"RELEASED {k} {files[k]} (deleted)")
            res.maybe(f"{len(prog.done)}/{len(order)}")
        res.now("done")
    rows = [r for v in prog.done.values() for r in v]
    write_outputs(rows, out_dir, cfg, h)
    R.log(f"M1b: wrote {len(rows)} rows and the summary to {out_dir}")
    return 0


# ------------------------------------------------------------------ outputs
def write_outputs(rows, out_dir, cfg, h):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    rows = sorted(rows, key=lambda r: (r.get("layer", -1), r.get("module", ""), r.get("comparison", "")))
    with open(out_dir / "m1b_tensors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    (out_dir / "summary.md").write_text(summary(rows, cfg, h), encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps({"config": cfg, "config_hash": h, "git_sha": C.git_sha(),
                                                         "package_versions": C.package_versions()}, indent=1))


def summary(rows, cfg, h):
    L = ["# M1b: bf16 resolution, shuffle null and QK-norm check on the M1 weight differences", "",
         f"Exploratory (registration amendment 2). Generated by `exp/jobs/M1b_bf16.py`, config {h}. "
         f"{cfg['shuffles']} shuffles per tensor and comparison; exact float32 SVDs.", ""]
    by = {c: [r for r in rows if r.get("comparison") == c] for c in ("RLVR-DPO", "DPO-SFT")}
    two = {c: [r for r in v if len(r.get("shape", [])) == 2] for c, v in by.items()}
    L += ["## (1) Share of elements changed and size of the changes in bf16 ulps", "",
          "| comparison | tensors | elements changed (element-weighted) | median over tensors | "
          + " | ".join(k for k in rows[0] if k.startswith("ulp_") and k != "ulp_median_changed") + " | median ulps among changed (median over tensors) |",
          "|---|---|---|---|" + "---|" * (len([k for k in rows[0] if k.startswith("ulp_") and k != "ulp_median_changed"]) + 1)]
    ulpk = [k for k in rows[0] if k.startswith("ulp_") and k != "ulp_median_changed"]
    for c, v in by.items():
        n = np.array([r["n"] for r in v], float)
        sh = np.array([r["share_changed"] for r in v])
        pooled = [float((np.array([r[k] for r in v]) * n).sum() / n.sum()) for k in ulpk]
        L.append(f"| {c} | {len(v)} | {(sh * n).sum() / n.sum():.4f} | {np.median(sh):.4f} | "
                 + " | ".join(f"{x:.4f}" for x in pooled) + f" | {np.median([r['ulp_median_changed'] for r in v]):.1f} |")
    L += ["", "ulp columns: share of all elements (element-weighted over tensors) whose change is that many bf16 ulps.", ""]
    L += ["## (2) Shuffle null: singular directions above the largest top singular value of the shuffled dW", "",
          "| comparison | 2-D tensors | with signal_rank > 0 | median signal_rank | median signal_energy | median s1 / null_s1_max |",
          "|---|---|---|---|---|---|"]
    for c, v in two.items():
        L.append(f"| {c} | {len(v)} | {sum(r['signal_rank'] > 0 for r in v)} | {np.median([r['signal_rank'] for r in v]):.0f} | "
                 f"{np.median([r['signal_energy'] for r in v]):.4f} | "
                 f"{np.median([r['s1'] / r['null_s1_max'] for r in v if r['null_s1_max'] > 0]):.2f} |")
    L += ["", "## (3) cos(dW, W) per module (median over layers); q_proj / k_proj have QK-norm", "",
          "| module | RLVR-DPO cos | DPO-SFT cos | RLVR-DPO rel_dw_orth / rel_dw |", "|---|---|---|---|"]
    mods = sorted({r["module"] for r in two["RLVR-DPO"] if r["layer"] >= 0})
    for m in mods:
        a = [r for r in two["RLVR-DPO"] if r["module"] == m]
        b = [r for r in two["DPO-SFT"] if r["module"] == m]
        L.append(f"| {m}{' (QK-norm)' if m in QK else ''} | {np.median([r['cos_dw_w'] for r in a]):+.3f} | "
                 f"{np.median([r['cos_dw_w'] for r in b]):+.3f} | {np.median([r['rel_dw_orth'] / r['rel_dw'] for r in a if r['rel_dw']]):.3f} |")
    L += ["", "## (4) Does the k_proj concentration survive re-ranking?", "",
          "Among the RLVR-DPO 2-D tensors, the number of `self_attn.k_proj` tensors in the top 10 and top 20 under each ranking, "
          "and the M1-H1 statistic recomputed on that measure (share of tensors where RLVR-DPO < DPO-SFT; median ratio).", "",
          "| ranking | k_proj in top 10 | k_proj in top 20 | share RLVR-DPO < DPO-SFT | median ratio RLVR-DPO / DPO-SFT |",
          "|---|---|---|---|---|"]
    ds = {r["tensor"]: r for r in two["DPO-SFT"]}
    for key in ("rel_dw", "signal_rel", "rel_dw_orth"):
        srt = sorted(two["RLVR-DPO"], key=lambda r: -r[key])
        ratio = np.array([r[key] / ds[r["tensor"]][key] for r in two["RLVR-DPO"] if ds[r["tensor"]][key] > 0])
        L.append(f"| {key} | {sum(r['module'] == 'self_attn.k_proj' for r in srt[:10])} | "
                 f"{sum(r['module'] == 'self_attn.k_proj' for r in srt[:20])} | {(ratio < 1).mean():.3f} | {np.median(ratio):.2f} |")
    L += ["", "Top 10 RLVR-DPO by signal_rel:", "", "| # | tensor | signal_rel | signal_rank | signal_energy | rel_dw | cos_dw_w |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(two["RLVR-DPO"], key=lambda r: -r["signal_rel"])[:10], 1):
        L.append(f"| {i} | {r['tensor']} | {r['signal_rel']:.6f} | {r['signal_rank']} | {r['signal_energy']:.4f} | "
                 f"{r['rel_dw']:.5f} | {r['cos_dw_w']:+.3f} |")
    return "\n".join(L) + "\n"
