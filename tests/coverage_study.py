"""Coverage study of the percentile bootstrap intervals used by scripts/analyze.py (not a pytest module; run by hand).

    python tests/coverage_study.py [--reps 1000] [--boot 2000] [--out coverage.csv]

For each configuration, synthetic paired data with a KNOWN true effect are generated, the same library functions that
analyze.py uses build a 95% percentile interval, and the share of repeats whose interval covers the truth is reported
with its repeat count. Acceptance criterion fixed in advance: every configuration must reach at least 92%.
Uses synthetic data only; it never reads anything under results/.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stats as S  # noqa: E402

MIN_COVERAGE = 0.92


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------------ item-level binary (B, E, C accuracy, D)
def binary_truth(pa, noise_sd=0.15, thr=0.45):
    """P(U + N(0, sd) < thr) for U ~ U(0, 1) by numerical integration."""
    xs = np.linspace(0, 1, 4001)
    cdf = np.array([normal_cdf((thr - x) / noise_sd) for x in xs])
    return pa - float(np.sum((cdf[1:] + cdf[:-1]) / 2 * np.diff(xs)))


def run_binary(n, reps, boot, pa, thr, label, seed):
    rng = np.random.default_rng(seed)
    truth = binary_truth(pa, thr=thr)
    hit = 0
    for r in range(reps):
        u = rng.random(n)
        ya = (u < pa).astype(float)
        yb = (u + rng.normal(0, 0.15, n) < thr).astype(float)
        d = S.paired_bootstrap((ya,), (yb,), S.mean_stat, f"{label}{r}", n_boot=boot)
        lo, hi = S.percentile_ci(d)
        hit += lo <= truth <= hi
    return hit, truth


# ------------------------------------------------------------------ D order-averaged score in {0, 0.5, 1}
def run_dscore(n, reps, boot, label, seed):
    rng = np.random.default_rng(seed)
    truth = 0.1 - 0.005                                   # E[min(1, q + .1)] - E[q] for q ~ U(0, 1)
    hit = 0
    for r in range(reps):
        q = rng.random(n)
        pa = np.minimum(1.0, q + 0.1)
        sa = (rng.random((n, 2)) < pa[:, None]).mean(axis=1)
        sb = (rng.random((n, 2)) < q[:, None]).mean(axis=1)
        d = S.paired_bootstrap((sa,), (sb,), S.mean_stat, f"{label}{r}", n_boot=boot)
        lo, hi = S.percentile_ci(d)
        hit += lo <= truth <= hi
    return hit, truth


# ------------------------------------------------------------------ calibration gap
def gap_draw(rng, n):
    p = rng.uniform(0.6, 0.95, n)
    ca = (rng.random(n) < p).astype(float)
    cb = (rng.random(n) < p - 0.05).astype(float)
    fa = np.clip(p + 0.10 + rng.normal(0, 0.1, n), 0, 1)
    fb = np.clip(p + 0.02 + rng.normal(0, 0.1, n), 0, 1)
    return fa, ca, fb, cb


def run_gap(n, reps, boot, label, seed):
    big = gap_draw(np.random.default_rng(seed + 1), 2_000_000)
    truth = (big[0].mean() - big[1].mean()) - (big[2].mean() - big[3].mean())
    rng = np.random.default_rng(seed)
    hit = 0
    for r in range(reps):
        fa, ca, fb, cb = gap_draw(rng, n)
        d = S.paired_bootstrap((fa, ca), (fb, cb), S.gap_stat, f"{label}{r}", n_boot=boot)
        lo, hi = S.percentile_ci(d)
        hit += lo <= truth <= hi
    return hit, truth


# ------------------------------------------------------------------ length units: median of paired differences
def length_draw(rng, n, delta):
    base = rng.lognormal(np.log(500), 0.5, n)
    return base * np.exp(rng.normal(delta, 0.2, n)), base * np.exp(rng.normal(0, 0.2, n))


def run_length(n, delta, reps, boot, label, seed):
    a, b = length_draw(np.random.default_rng(seed + 1), 2_000_000, delta)
    truth = float(np.median(a - b))
    rng = np.random.default_rng(seed)
    hit = 0
    for r in range(reps):
        xa, xb = length_draw(rng, n, delta)
        d = S.bootstrap_paired_differences(xa - xb, S.median_stat, f"{label}{r}", n_boot=boot)
        lo, hi = S.percentile_ci(d)
        hit += lo <= truth <= hi
    return hit, truth


# ------------------------------------------------------------------ trait: mean of wording means
def run_wordings(reps, boot, label, seed, interaction_sd, p_invalid):
    """Six wordings x three seeds. Truth = 1.0 (the mean of the population of wording effects, with the
    wording-by-checkpoint interaction having mean zero). Invalid cells are dropped as in the analysis."""
    rng = np.random.default_rng(seed)
    hit, used = 0, 0
    for r in range(reps):
        w = rng.normal(0, 0.3, 6)
        wa = w + rng.normal(0, interaction_sd, 6)
        wb = w + rng.normal(0, interaction_sd, 6)
        va = 4.0 + wa[:, None] + rng.normal(0, 0.4, (6, 3))
        vb = 3.0 + wb[:, None] + rng.normal(0, 0.4, (6, 3))
        va[rng.random((6, 3)) < p_invalid] = np.nan
        vb[rng.random((6, 3)) < p_invalid] = np.nan
        d = S.bootstrap_wordings(va, vb, f"{label}{r}", n_boot=boot)
        if not np.isfinite(d).any():
            continue
        lo, hi = S.percentile_ci(d)
        hit += lo <= 1.0 <= hi
        used += 1
    return hit, 1.0, used


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    R, B = args.reps, args.boot
    configs = [
        ("trait wordings n=6, shared wording effects", lambda: run_wordings(R, B, "w1", 11, 0.0, 0.0)),
        ("trait wordings n=6, wording x checkpoint interaction sd 0.3", lambda: run_wordings(R, B, "w2", 12, 0.3, 0.0)),
        ("trait wordings n=6, interaction 0.3 + 15% invalid cells", lambda: run_wordings(R, B, "w3", 13, 0.3, 0.15)),
        ("binary items n=30 (B caving: the 30 wrong-user items), p=.60 vs ~.45", lambda: run_binary(30, R, B, 0.60, 0.45, "b30", 20)),
        ("binary items n=40 (B incl. controls), p=.60 vs ~.45", lambda: run_binary(40, R, B, 0.60, 0.45, "b40", 21)),
        ("binary items n=50 (E MATH), p=.60 vs ~.45", lambda: run_binary(50, R, B, 0.60, 0.45, "b50", 22)),
        ("binary items n=50 (E MATH), ceiling p=.95 vs ~.90", lambda: run_binary(50, R, B, 0.95, 0.90, "b50c", 23)),
        ("binary items n=80 (C accuracy), p=.60 vs ~.45", lambda: run_binary(80, R, B, 0.60, 0.45, "b80", 24)),
        ("binary items n=150, p=.60 vs ~.45", lambda: run_binary(150, R, B, 0.60, 0.45, "b150", 25)),
        ("D order-averaged score n=150 ({0,.5,1})", lambda: run_dscore(150, R, B, "d150", 31)),
        ("calibration gap n=80 (continuous, paired)", lambda: run_gap(80, R, B, "g80", 41)),
        ("length units n=48 (A), median paired diff, x2", lambda: run_length(48, math.log(2), R, B, "l48", 51)),
        ("length units n=48 (A), median paired diff, null", lambda: run_length(48, 0.0, R, B, "l48n", 52)),
        ("length units n=150 (D), median paired diff, null", lambda: run_length(150, 0.0, R, B, "l150n", 53)),
        ("length units n=150 (D), median paired diff, +10%", lambda: run_length(150, 0.1, R, B, "l150", 54)),
    ]
    rows, failed = [], []
    print(f"{'configuration':70s} {'reps':>5s} {'coverage':>9s}  95% CI of coverage  verdict")
    for name, fn in configs:
        out = fn()
        hit, truth = out[0], out[1]
        reps_used = out[2] if len(out) > 2 else R
        cov = hit / reps_used
        lo, hi = S.wilson(hit, reps_used)
        ok = cov >= MIN_COVERAGE
        if not ok:
            failed.append(name)
        rows.append([name, reps_used, B, f"{truth:.4f}", f"{cov:.3f}", f"{lo:.3f}", f"{hi:.3f}", "pass" if ok else "FAIL"])
        print(f"{name:70s} {reps_used:5d} {cov:9.3f}  [{lo:.3f}, {hi:.3f}]  {'pass' if ok else 'FAIL'}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "repeats", "bootstrap_resamples", "true_effect", "coverage", "wilson_lo",
                        "wilson_hi", f"verdict (>= {MIN_COVERAGE})"])
            w.writerows(rows)
    print("\nACCEPTANCE: " + ("all configurations >= 92%" if not failed else "FAILED: " + "; ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
