"""Statistical primitives for the analysis (docs/analysis-plan.md sections 4 and 5).

Pure functions, no I/O. Bootstrap randomness comes only from `make_rng`, seeded with the plan's fixed seed, so every
result is a deterministic function of the data and the label of the call (call order does not matter).
"""
import itertools
import math
import zlib

import numpy as np

SEED = 20261024          # plan section 4
N_BOOT = 10_000          # plan section 4
Z95 = 1.959963984540054


def make_rng(label):
    """Deterministic generator for one bootstrap call: fixed plan seed plus a hash of the call's label."""
    return np.random.default_rng([SEED, zlib.crc32(str(label).encode("utf-8"))])


# ------------------------------------------------------------------ intervals
def wilson(k, n, z=Z95):
    """Wilson score interval for k successes of n. Returns (lo, hi); (nan, nan) when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile_ci(samples, level=0.95):
    """Percentile interval of the finite bootstrap samples; (nan, nan) if none are finite."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    lo, hi = np.percentile(s, [100 * a, 100 * (1 - a)])
    return (float(lo), float(hi))


def boot_p_two_sided(samples, obs):
    """Two-sided p from the proportion of resamples on the opposite side of zero from the observed estimate
    (plan section 5, calibration gap), doubled and capped at 1. A resample exactly at zero counts as opposite."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(obs) or obs == 0:
        return 1.0
    opposite = np.mean(s <= 0) if obs > 0 else np.mean(s >= 0)
    return float(min(1.0, 2 * opposite))


# ------------------------------------------------------------------ bootstrap
def paired_bootstrap(cols_a, cols_b, stat, label, n_boot=N_BOOT):
    """Paired bootstrap of stat(a) - stat(b). `cols_*` are tuples of equal-length arrays (one entry per item); the
    same resampled item indices are applied to both checkpoints. `stat` maps resampled (n_boot, n) columns to
    (n_boot,). Returns the array of resampled differences."""
    n = len(cols_a[0])
    idx = make_rng(label).integers(0, n, size=(n_boot, n))
    return stat(*[np.asarray(c)[idx] for c in cols_a]) - stat(*[np.asarray(c)[idx] for c in cols_b])


def mean_stat(x):
    return x.mean(axis=1)


def median_stat(x):
    return np.median(x, axis=1)


def gap_stat(conf, correct):
    return conf.mean(axis=1) - correct.mean(axis=1)


def paired_wording_arrays(va, vb):
    """Restrict two (wordings x seeds) arrays to the wordings with at least one valid generation in BOTH checkpoints.
    Returns (va, vb, kept_wording_indices)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    keep = [w for w in range(va.shape[0]) if np.isfinite(va[w]).any() and np.isfinite(vb[w]).any()]
    return va[keep], vb[keep], keep


def wording_means(v):
    """Mean of the valid seeds of each wording (NaN where none)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    cnt = ok.sum(axis=1)
    return np.where(cnt > 0, np.where(ok, v, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)


def _compress(v):
    """Left-pack the valid seed values of each wording; returns (padded values, valid counts)."""
    v = np.asarray(v, float)
    counts = np.isfinite(v).sum(axis=1)
    out = np.zeros_like(v)
    for w, row in enumerate(v):
        r = row[np.isfinite(row)]
        out[w, :len(r)] = r
    return out, counts


def bootstrap_wordings(va, vb, label, n_boot=N_BOOT):
    """Component A resampling (plan section 4). `va`, `vb` are (wordings x seeds) arrays, NaN = invalid. Only wordings
    valid in both checkpoints enter. Wordings are resampled jointly; within each resampled wording the valid seeds are
    resampled with replacement, independently per checkpoint and per resampled copy. The statistic is the mean of the
    wording means (equal weight per wording). Returns the resampled differences a - b."""
    va, vb, _ = paired_wording_arrays(va, vb)
    n_w = va.shape[0]
    if n_w == 0:
        return np.full(n_boot, np.nan)
    rng = make_rng(label)
    w = rng.integers(0, n_w, size=(n_boot, n_w))

    def resampled(v):
        pad, cnt = _compress(v)
        m = cnt[w]                                        # (B, W) valid seeds of each resampled wording
        s = pad.shape[1]
        u = rng.random((n_boot, n_w, s))
        pick = np.minimum((u * m[:, :, None]).astype(int), np.maximum(m[:, :, None] - 1, 0))
        x = pad[w[:, :, None], pick]
        use = np.arange(s)[None, None, :] < m[:, :, None]
        return (np.where(use, x, 0.0).sum(axis=2) / m).mean(axis=1)

    return resampled(va) - resampled(vb)


def bootstrap_paired_differences(d, stat, label, n_boot=N_BOOT):
    """Bootstrap of stat(d) where d holds one paired difference per item (items resampled with replacement)."""
    d = np.asarray(d, float)
    if d.size == 0:
        return np.full(n_boot, np.nan)
    idx = make_rng(label).integers(0, d.size, size=(n_boot, d.size))
    return stat(d[idx])


# ------------------------------------------------------------------ tests
def mcnemar_exact(b, c):
    """Exact two-sided McNemar p from the discordant counts b and c (binomial sign test, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def mcnemar_from_pairs(y_a, y_b):
    """McNemar exact p for paired binary outcomes. Returns (p, b, c) with b = a-only successes, c = b-only."""
    y_a, y_b = np.asarray(y_a, bool), np.asarray(y_b, bool)
    b = int(np.sum(y_a & ~y_b))
    c = int(np.sum(~y_a & y_b))
    return mcnemar_exact(b, c), b, c


def _average_ranks(x):
    """Ranks of x (1-based), ties given the average rank. Values equal to 12 decimals are tied."""
    x = np.round(np.asarray(x, float), 12)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


EXACT_MAX_N = 25


def wilcoxon_detail(d):
    """Two-sided Wilcoxon signed-rank test on paired differences `d`, hand-implemented.

    Zero differences are dropped before ranking (Wilcoxon's rule). If none remain, p = 1.
    Branches (reported as `branch`):
      exact                   n <= 25 non-zero differences and no tied |d|: the exact null distribution of W+ is built by
                              dynamic programming over the 2^n sign patterns; p = min(1, 2 * P(W <= min(W+, W-))).
      normal_tie_corrected    any tied |d| (or n > 25 with ties): normal approximation, variance
                              n(n+1)(2n+1)/24 - sum(t^3 - t)/48 over tie groups of size t, no continuity correction.
      normal                  n > 25, no ties: the same approximation (the tie term is zero).
    Returns {'p', 'branch', 'n', 'w_plus'}."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    d = d[np.round(d, 12) != 0]
    n = int(d.size)
    if n == 0:
        return {"p": 1.0, "branch": "all_zero", "n": 0, "w_plus": 0.0}
    absd = np.round(np.abs(d), 12)
    ranks = _average_ranks(absd)
    w_plus = float(ranks[d > 0].sum())
    ties = len(np.unique(absd)) < n
    if n <= EXACT_MAX_N and not ties:
        total = n * (n + 1) // 2
        counts = np.zeros(total + 1)
        counts[0] = 1.0
        for r in range(1, n + 1):
            shifted = np.zeros_like(counts)
            shifted[r:] = counts[:total + 1 - r]
            counts = counts + shifted
        w = int(round(min(w_plus, total - w_plus)))
        p = min(1.0, 2 * counts[:w + 1].sum() / 2.0 ** n)
        return {"p": float(p), "branch": "exact", "n": n, "w_plus": w_plus}
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if ties:
        _, t = np.unique(absd, return_counts=True)
        var -= float(np.sum(t ** 3 - t)) / 48
    if var <= 0:
        return {"p": 1.0, "branch": "normal_tie_corrected" if ties else "normal", "n": n, "w_plus": w_plus}
    z = (w_plus - mean) / math.sqrt(var)
    return {"p": float(math.erfc(abs(z) / math.sqrt(2))), "branch": "normal_tie_corrected" if ties else "normal",
            "n": n, "w_plus": w_plus}


def wilcoxon_signed_rank(d):
    """p-value of `wilcoxon_detail`."""
    return wilcoxon_detail(d)["p"]


def paired_permutation_exact(d):
    """Exact paired permutation (sign-flip) test on the differences `d`, statistic = mean. All 2^n sign patterns are
    enumerated. Two-sided p = share of patterns with |mean| >= |observed mean|. n = 0 gives p = 1."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return 1.0
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    stats = np.abs((signs * d).sum(axis=1)) / n
    obs = abs(d.sum()) / n
    return float(np.mean(stats >= obs - 1e-12))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return [float(x) for x in adj]
