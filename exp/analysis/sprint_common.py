"""Shared helpers for the sprint summaries (results/sprint/<job>/summary.md).

Shards are fetched from the HF results repo (HF_REPO, default Shagsc/rlvr-se-runs) into results/sprint/_hf/
(git-ignored: results/sprint/**/runs/), reading only the files a summary needs. Statistics as registered ("Fixed
design facts", amendment 3): mean of paired differences, percentile bootstrap over units (10,000 resamples, seed
20261024, generator default_rng([20261024, crc32(label)])), two-sided Wilcoxon.
"""
import os
import sys
from pathlib import Path

import numpy as np

from .m2_summary import DI, ROOT

HF_REPO = os.environ.get("HF_REPO", "Shagsc/rlvr-se-runs")
LOCAL = ROOT / "results" / "sprint" / "_hf"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _files():
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(HF_REPO, repo_type="dataset")


def fetch(prefix, suffix=".parquet"):
    """Download every repo file under `prefix` ending in `suffix`; return their local paths (sorted)."""
    from huggingface_hub import hf_hub_download
    out = []
    for p in sorted(f for f in _files() if f.startswith(prefix) and f.endswith(suffix)):
        out.append(Path(hf_hub_download(HF_REPO, p, repo_type="dataset", local_dir=str(LOCAL))))
    return out


def read(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pylist()


def ci_mean(x, label):
    """Mean with a percentile bootstrap CI over units (x: 1-D array of per-unit values)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    lo, hi = DI.percentile_ci(DI.bootstrap_paired_differences(x, lambda b: b.mean(axis=1), label))
    return float(x.mean()), float(lo), float(hi)


def paired(a, b, label):
    """Paired difference a - b over the shared units of two {unit: value} dicts."""
    us = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in us], dtype=float)
    m, lo, hi = ci_mean(d, label)
    p = DI.wilcoxon_detail(d)["p"] if len(d) >= 2 else float("nan")
    return {"mean": m, "lo": lo, "hi": hi, "n": len(d), "p": p}


def fmt(r, nd=3):
    if r["n"] < 2:
        return f"n = {r['n']}"
    return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}], n = {r['n']}, Wilcoxon p = {r['p']:.3g}"


def verdict(r, direction):
    """'result' if the CI excludes 0 in the registered direction (+1 / -1), 'opposite' if it excludes 0 the other way."""
    if r["n"] < 2 or np.isnan(r["lo"]):
        return "no data"
    if direction > 0:
        return "result" if r["lo"] > 0 else ("opposite" if r["hi"] < 0 else "no result (CI includes 0)")
    return "result" if r["hi"] < 0 else ("opposite" if r["lo"] > 0 else "no result (CI includes 0)")
