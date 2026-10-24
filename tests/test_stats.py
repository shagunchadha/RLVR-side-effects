"""Tests for scripts/stats.py. Expected values are worked out by hand (arithmetic shown), not taken from the code."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stats as S  # noqa: E402


# ---- McNemar exact: two-sided binomial sign test on the discordant pairs
def test_mcnemar_known_table():
    # b = 1, c = 5: n = 6, P(X <= 1) = (C(6,0)+C(6,1))/2^6 = 7/64, doubled = 14/64
    assert S.mcnemar_exact(1, 5) == pytest.approx(14 / 64)


def test_mcnemar_extreme_and_symmetric():
    # b = 0, c = 10: P(X = 0) = 1/1024, doubled = 2/1024
    assert S.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert S.mcnemar_exact(3, 3) == 1.0          # doubled tail exceeds 1, capped
    assert S.mcnemar_exact(0, 0) == 1.0          # no discordant pairs


def test_mcnemar_from_pairs_counts():
    a = [1, 1, 1, 0, 0, 1]
    b = [1, 0, 0, 1, 0, 0]
    p, only_a, only_b = S.mcnemar_from_pairs(a, b)
    assert (only_a, only_b) == (3, 1)            # items 2,3,6 vs item 4
    assert p == pytest.approx(2 * (1 + 4) / 16)  # n = 4, k = 1: P(X <= 1) = 5/16, doubled = 10/16


# ---- Wilcoxon signed-rank
def test_wilcoxon_all_positive_n5():
    # ranks 1..5 all positive: W- = 0, exact P(W <= 0) = 1/32, two-sided = 2/32
    assert S.wilcoxon_signed_rank([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)


def test_wilcoxon_one_negative():
    # d = -1, 2, 3, 4, 5: W- = 1, subsets with sum <= 1 are {} and {1}: 2/32, doubled = 4/32
    assert S.wilcoxon_signed_rank([-1, 2, 3, 4, 5]) == pytest.approx(4 / 32)


def test_wilcoxon_all_zero_is_one():
    assert S.wilcoxon_signed_rank([0, 0, 0]) == 1.0
    assert S.wilcoxon_detail([0, 0, 0])["branch"] == "all_zero"


def test_wilcoxon_exact_branch_and_zero_dropping():
    d = S.wilcoxon_detail([0, 1, 2, 3, 4, 5])            # the zero is dropped before ranking: same as [1..5]
    assert d["branch"] == "exact" and d["n"] == 5 and d["w_plus"] == 15
    assert d["p"] == pytest.approx(2 / 32)


def test_wilcoxon_tied_case_uses_tie_corrected_normal():
    # d = (1, -1, 2, 3): |d| ranks 1.5, 1.5, 3, 4 (one tie group of size 2); W+ = 1.5 + 3 + 4 = 8.5
    # mean n(n+1)/4 = 5; variance 4*5*9/24 = 7.5 minus (2^3 - 2)/48 = 0.125 -> 7.375; z = 3.5/sqrt(7.375) = 1.2888
    # two-sided p from a normal table: 2 * (1 - 0.90126) = 0.1975 (no continuity correction)
    d = S.wilcoxon_detail([1, -1, 2, 3])
    assert d["branch"] == "normal_tie_corrected"
    assert d["w_plus"] == 8.5
    assert d["p"] == pytest.approx(0.1975, abs=5e-4)


def test_wilcoxon_tie_correction_actually_applied():
    # without the correction the variance would be 7.5 and p = 2*(1 - Phi(1.2780)) = 0.2012: the corrected
    # value must differ from that
    assert abs(S.wilcoxon_signed_rank([1, -1, 2, 3]) - 0.1853) > 0.002


def test_wilcoxon_switches_to_normal_above_25():
    d25 = S.wilcoxon_detail(list(range(1, 26)))
    d26 = S.wilcoxon_detail(list(range(1, 27)))
    assert d25["branch"] == "exact" and d26["branch"] == "normal"
    # n = 26, all positive: W+ = 351, mean 175.5, var 26*27*53/24 = 1550.25, z = 175.5/39.372 = 4.4574, p = 8.3e-6
    assert d26["p"] == pytest.approx(8.3e-6, rel=0.03)


def test_wilcoxon_three_level_scores_all_ties():
    # differences in {-1, -0.5, 0.5, 1}: every |d| is tied (two groups of two) -> normal, tie-corrected branch
    d = S.wilcoxon_detail([1, 1, 0.5, -0.5])
    assert d["branch"] == "normal_tie_corrected" and 0 < d["p"] <= 1


# ---- Holm-Bonferroni
def test_holm_sequence():
    # sorted: .005 (x4 = .02), .01 (x3 = .03), .03 (x2 = .06), .04 (x1 = .04 -> monotone max .06)
    adj = S.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_empty():
    assert S.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert S.holm([]) == []


# ---- exact paired permutation test
def test_permutation_three_values():
    # d = (1,2,3): sign patterns give sums 6,0,2,-4,4,-2,0,-6; |sum| >= 6 in 2 of 8 patterns
    assert S.paired_permutation_exact([1, 2, 3]) == pytest.approx(2 / 8)


def test_permutation_two_values():
    # d = (1,1): sums 2,0,0,-2; |sum| >= 2 in 2 of 4
    assert S.paired_permutation_exact([1, 1]) == pytest.approx(0.5)


def test_permutation_six_wordings_all_same_sign():
    # six equal-sign differences: only the all-plus and all-minus patterns reach the observed |mean|: 2/64
    assert S.paired_permutation_exact([0.5] * 6) == pytest.approx(2 / 64)


# ---- Wilson interval
def test_wilson_zero_successes():
    # k = 0: lower 0, upper z^2 / (n + z^2) = 3.841459 / 13.841459
    lo, hi = S.wilson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(3.841459 / 13.841459, rel=1e-5)


def test_wilson_half():
    # k = 5, n = 10 is symmetric about .5; standard tabulated value 0.2366 to 0.7634
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_empty():
    lo, hi = S.wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# ---- bootstrap machinery
def test_rng_is_deterministic_and_label_dependent():
    a = S.make_rng("x").integers(0, 1000, 5)
    b = S.make_rng("x").integers(0, 1000, 5)
    c = S.make_rng("y").integers(0, 1000, 5)
    assert (a == b).all() and not (a == c).all()
    assert S.SEED == 20261024 and S.N_BOOT == 10_000


def test_paired_bootstrap_deterministic_and_degenerate():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    d1 = S.paired_bootstrap((x,), (x,), S.mean_stat, "same", n_boot=200)
    d2 = S.paired_bootstrap((x,), (x,), S.mean_stat, "same", n_boot=200)
    assert (d1 == d2).all()
    assert np.allclose(d1, 0)                    # identical columns: every resampled difference is exactly zero
    lo, hi = S.percentile_ci(d1)
    assert lo == 0 and hi == 0


def test_percentile_ci_known_samples():
    s = np.arange(1, 101, dtype=float)           # 1..100: 2.5th percentile = 3.475, 97.5th = 97.525 (linear interp.)
    lo, hi = S.percentile_ci(s)
    assert lo == pytest.approx(3.475) and hi == pytest.approx(97.525)


def test_boot_p_two_sided():
    s = np.array([-1.0] * 10 + [1.0] * 90)       # observed positive, 10% of resamples at or below zero -> p = 0.2
    assert S.boot_p_two_sided(s, 0.5) == pytest.approx(0.2)
    assert S.boot_p_two_sided(s, 0.0) == 1.0


def test_bootstrap_wordings_shapes_and_nan_handling():
    va = np.array([[1.0, 2.0, 3.0], [np.nan] * 3])
    vb = np.array([[1.0, 2.0, 3.0], [np.nan] * 3])
    d = S.bootstrap_wordings(va, vb, "w", n_boot=50)
    assert d.shape == (50,) and np.isfinite(d).all()        # the all-invalid wording is dropped, never NaN
    assert np.abs(d).max() < 2.0


def test_paired_wording_arrays_keep_wordings_valid_in_both():
    va = np.array([[1.0, np.nan, np.nan], [np.nan] * 3, [2.0, 2.0, 2.0]])
    vb = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [np.nan] * 3])
    pa, pb, kept = S.paired_wording_arrays(va, vb)
    assert kept == [0] and pa.shape == (1, 3)
    assert S.wording_means(va).tolist()[0] == 1.0 and np.isnan(S.wording_means(va)[1])


def test_bootstrap_wordings_equal_weight_per_wording():
    # wording 0 has one valid seed (value 1), wording 1 has three (value 5): mean of wording means = 3, not the pooled 4
    va = np.array([[1.0, np.nan, np.nan], [5.0, 5.0, 5.0]])
    vb = np.zeros_like(va)
    d = S.bootstrap_wordings(va, vb, "eq", n_boot=2000)
    assert d.mean() == pytest.approx(3.0, abs=0.15)         # a pooled mean would centre on 4.0


def test_bootstrap_paired_differences_median():
    d = np.array([5.0, 5.0, 5.0, 5.0])
    samples = S.bootstrap_paired_differences(d, S.median_stat, "m", n_boot=100)
    assert np.allclose(samples, 5.0)
    assert np.isnan(S.bootstrap_paired_differences([], S.median_stat, "m", n_boot=5)).all()


# ---- coverage: a 95% percentile interval should cover the true paired difference about 95% of the time
def _p_b():
    """P(U + N(0, .15) < .45) for U ~ Uniform(0, 1), by numerical integration of the normal CDF."""
    from math import erf, sqrt
    xs = np.linspace(0, 1, 2001)
    cdf = np.array([0.5 * (1 + erf((0.45 - x) / (0.15 * sqrt(2)))) for x in xs])
    return float(np.sum((cdf[1:] + cdf[:-1]) / 2 * np.diff(xs)))


def test_bootstrap_interval_coverage():
    rng = np.random.default_rng(7)
    n, reps = 100, 300
    truth = 0.60 - _p_b()                        # item a succeeds when U < .60, item b when U + noise < .45
    covered = 0
    for r in range(reps):
        u = rng.random(n)                        # shared latent: paired, positively correlated outcomes
        ya = (u < 0.60).astype(float)
        yb = (u + rng.normal(0, 0.15, n) < 0.45).astype(float)
        d = S.paired_bootstrap((ya,), (yb,), S.mean_stat, f"cov{r}", n_boot=1000)
        lo, hi = S.percentile_ci(d)
        covered += lo <= truth <= hi
    assert 0.88 <= covered / reps <= 0.99
