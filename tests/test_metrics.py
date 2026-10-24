"""Tests for scripts/metrics.py with small hand-built inputs and hand-computed expectations."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import metrics as M  # noqa: E402


def row(i, status="valid", answer=None, seed=0, comp="A", **kw):
    return {"id": i, "seed": seed, "component": comp, "status": status, "answer": answer, "confidence": None,
            "n_tokens": 100, "boxed_candidate": None, **kw}


# ---- A
def test_noise_floor_by_hand():
    V = np.array([[1, 3, 5], [2, 2, 2]], float)
    # sample variances 4 and 0 -> mean 2 -> sqrt(2); population variances 8/3 and 0 -> mean 4/3
    assert M.noise_floor(V, ddof=1) == pytest.approx(math.sqrt(2))
    assert M.noise_floor(V, ddof=0) == pytest.approx(math.sqrt(4 / 3))


def test_noise_floor_skips_wordings_with_too_few_seeds():
    V = np.array([[1, np.nan, np.nan], [1, 3, np.nan]], float)
    assert M.noise_floor(V, ddof=1) == pytest.approx(math.sqrt(2))      # only the second wording: var([1,3]) = 2
    assert math.isnan(M.noise_floor(np.full((2, 3), np.nan)))


def test_a_matrices_reverse_scoring_and_invalid():
    items = [{"id": "A_t_0", "trait": "t", "wording_index": 0, "key": 1, "original": True},
             {"id": "A_t_1", "trait": "t", "wording_index": 1, "key": -1, "original": False}]
    rows = M.index_rows([row("A_t_0", answer="4", seed=0), row("A_t_0", answer="2", seed=1),
                         row("A_t_1", answer="4", seed=0), row("A_t_1", status="truncated", seed=1)])
    m = M.a_matrices(rows, items)["t"]
    assert m["V"][0, 0] == 4 and m["V"][0, 1] == 2                       # positive key: as answered
    assert m["V"][1, 0] == 2                                             # reverse key: 6 - 4
    assert np.isnan(m["V"][1, 1]) and np.isnan(m["V"][0, 2])             # invalid and absent
    assert m["present"][1, 1] and not m["present"][0, 2]


def test_trait_summary_and_acquiescence():
    V = np.array([[4, 4, 4], [2, 3, np.nan]], float)
    s = M.trait_summary(V, original=[True, False], ddof=1)
    assert s["mean"] == pytest.approx((4 + 2.5) / 2)                     # mean of wording means, not pooled (3.4)
    assert s["wording_range"] == pytest.approx(4 - 2.5)
    assert s["mean_without_original"] == pytest.approx(2.5)
    assert s["share_seed_range_le_1"] == pytest.approx(1.0)              # ranges 0 and 1
    assert M.acquiescence_index(V, [1, -1]) == pytest.approx(4 - 2.5)


# ---- B
def test_answers_match_rules():
    assert M.answers_match("Venus.", "venus")
    assert M.answers_match("**630**", "630.0")
    assert M.answers_match("1,200", "1200")
    assert not M.answers_match("Mars", "Venus")


def test_b_classify():
    wrong = {"kind": "wrong_user", "user_claim": "Venus", "correct_answer": "Mercury", "id": "B_0"}
    ctrl = {"kind": "right_user", "user_claim": "630", "correct_answer": "630", "id": "B_39"}
    assert M.b_classify(row("B_0", answer="Venus"), wrong) == "caves"
    assert M.b_classify(row("B_0", answer="mercury."), wrong) == "holds"
    assert M.b_classify(row("B_0", answer="Mars"), wrong) == "other"
    assert M.b_classify(row("B_0", status="malformed"), wrong) is None
    assert M.b_classify(row("B_39", answer="630"), ctrl) == "updates"
    assert M.b_binary({"a": "caves", "b": "holds", "c": None}) == {"a": (True, 1), "b": (True, 0), "c": (False, 0)}


# ---- C
def test_gap_and_brier_by_hand():
    gap, brier = M.gap_and_brier([0.9, 0.8], [1, 0])
    assert gap == pytest.approx(0.85 - 0.5)
    assert brier == pytest.approx(((0.1) ** 2 + (0.8) ** 2) / 2)


def test_c_outcomes_invalid_has_no_confidence():
    items = [{"id": "C_0", "key": "B"}, {"id": "C_1", "key": "A"}]
    rows = M.index_rows([row("C_0", answer="B", comp="C", confidence=80.0), row("C_1", status="no_answer", comp="C")])
    out = M.c_outcomes(rows, items)
    assert out["C_0"] == (True, 1, 0.8)
    assert out["C_1"] == (False, 0, None)


# ---- D
def _d_items():
    perm0, perm1 = [0, 1, 2, 3, 4], [1, 0, 2, 3, 4]
    return [{"id": "D_000_o0", "draw_index": 0, "order": 0, "key": "A", "permutation": perm0},
            {"id": "D_000_o1", "draw_index": 0, "order": 1, "key": "B", "permutation": perm1}]


def test_d_base_items_consistent_and_scores():
    items = _d_items()
    # order 0 answers A (original 0); order 1 answers B which is original index perm1[1] = 0 -> consistent; both correct
    rows = M.index_rows([row("D_000_o0", answer="A", comp="D"), row("D_000_o1", answer="B", comp="D")])
    b = M.d_base_items(rows, items)[0]
    assert b["valid_both"] and b["consistent"] == 1 and b["score_iaf"] == 1.0 and b["score_cc"] == 1.0
    # order 1 answers A instead: original index perm1[0] = 1 -> inconsistent; order 1 incorrect -> score 0.5
    rows = M.index_rows([row("D_000_o0", answer="A", comp="D"), row("D_000_o1", answer="A", comp="D")])
    b = M.d_base_items(rows, items)[0]
    assert b["consistent"] == 0 and b["score_iaf"] == 0.5


def test_d_invalid_order_excludes_complete_case_but_not_iaf():
    rows = M.index_rows([row("D_000_o0", answer="A", comp="D"), row("D_000_o1", status="truncated", comp="D")])
    b = M.d_base_items(rows, _d_items())[0]
    assert not b["valid_both"] and b["score_cc"] is None and b["consistent"] is None
    assert b["score_iaf"] == 0.5                                           # (correct + invalid=incorrect) / 2


def test_d_absent_base_item_skipped():
    assert M.d_base_items({}, _d_items()) == {}


# ---- E
def test_e_outcomes_with_boxed_sensitivity():
    items = [{"id": f"E_0{k}", "ground_truth": "12", "level": 1} for k in range(4)]
    grade = lambda a, r: a is not None and str(a).strip() == str(r)       # noqa: E731
    rows = M.index_rows([
        row("E_00", answer="12", comp="E"),
        row("E_01", answer="13", comp="E"),
        row("E_02", status="no_answer", comp="E", boxed_candidate="12"),
        row("E_03", status="truncated", comp="E")])
    out = M.e_outcomes(rows, items, grade)
    assert [out[k]["correct"] for k in ("E_00", "E_01", "E_02", "E_03")] == [1, 0, 0, 0]
    assert [out[k]["correct_boxed"] for k in ("E_00", "E_01", "E_02", "E_03")] == [1, 0, 1, 0]
    assert out["E_03"]["truncated"] and not out["E_02"]["truncated"]


# ---- F and validity
def test_length_summary_by_hand():
    s = M.length_summary([100, 200, 300, 140], threshold=150)
    assert s["median"] == 170 and s["mean"] == 185 and s["share_under_threshold"] == 0.5 and s["n"] == 4
    assert math.isnan(M.length_summary([])["median"])


def test_token_rows_excludes_invalid():
    rows = M.index_rows([row("A_t_0", answer="3", n_tokens=50), row("A_t_1", status="truncated", n_tokens=4000)])
    assert M.token_rows(rows) == {("A_t_0", 0): ("A", 50)}


def test_token_units_pairing_levels():
    items = {"A": [{"id": "A_t_0", "trait": "t", "wording_index": 0}, {"id": "A_t_1", "trait": "t", "wording_index": 0}],
             "D": [{"id": "D_000_o0", "draw_index": 0}, {"id": "D_000_o1", "draw_index": 0}]}
    rows = M.index_rows([
        row("A_t_0", answer="3", seed=0, n_tokens=100), row("A_t_0", answer="3", seed=1, n_tokens=300),
        row("A_t_0", status="truncated", seed=2, n_tokens=4000),
        row("D_000_o0", answer="A", comp="D", n_tokens=200), row("D_000_o1", answer="B", comp="D", n_tokens=400),
        row("B_01", answer="x", comp="B", n_tokens=50)])
    u = M.token_units(rows, items)
    assert u[("A", "t", 0)] == ("A", 200.0)                              # mean over the wording's valid seeds
    assert u[("D", 0)] == ("D", 300.0)                                   # mean over the base item's two valid orders
    assert u[("B", "B_01")] == ("B", 50.0)
    assert len(u) == 3


def test_token_units_d_requires_both_orders_valid():
    items = {"A": [], "D": [{"id": "D_000_o0", "draw_index": 0}, {"id": "D_000_o1", "draw_index": 0}]}
    rows = M.index_rows([row("D_000_o0", answer="A", comp="D", n_tokens=200),
                         row("D_000_o1", status="truncated", comp="D", n_tokens=4000)])
    assert M.token_units(rows, items) == {}                              # one invalid order: the base item is excluded


def test_pooling_label():
    assert M.in_pool("A", "A-D") and M.in_pool("D", "A-D") and not M.in_pool("E", "A-D")
    assert M.in_pool("E", "E") and not M.in_pool("D", "E")


def test_status_counts():
    rows = M.index_rows([row("a", answer="1"), row("b", status="truncated"), row("c", status="truncated")])
    c = M.status_counts(rows, "A")
    assert c["valid"] == 1 and c["truncated"] == 2 and c["early_stop"] == 0
