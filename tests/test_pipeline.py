"""End-to-end tests of scripts/analyze.py on SYNTHETIC data with planted effects of known size.

The synthetic rows are generated against the real item files in items/ (read only). Nothing here touches results/.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze as AN  # noqa: E402
import metrics as M  # noqa: E402

ITEMS, GRADE = AN.load_items(ROOT / "items")
LETTERS = "ABCDE"


def synth_rows(seed, approval_shift=0.0, p_math=0.6, p_cave=0.4, conf_mean=70.0, p_csqa=0.8, tok_a=900, tok_d=480,
               p_invalid=0.02):
    """Rows for one synthetic checkpoint. Scored trait values are base (3.0) + wording effect + noise, shifted for
    'approval' by approval_shift; everything else is drawn from the stated probabilities."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(item, seed_, answer, comp, tokens, confidence=None):
        invalid = rng.random() < p_invalid
        rows.append({"id": item["id"], "component": comp, "seed": seed_, "n_tokens": int(tokens),
                     "status": "truncated" if invalid else "valid", "answer": None if invalid else answer,
                     "confidence": None if invalid else confidence, "boxed_candidate": None})

    wording_rng = np.random.default_rng(99)          # wording effects are properties of the wordings: shared by all checkpoints
    wording_effect = {}
    for it in ITEMS["A"]:
        k = (it["trait"], it["wording_index"])
        wording_effect.setdefault(k, wording_rng.normal(0, 0.3))
    for it in ITEMS["A"]:
        for s in range(3):
            v = 3.0 + wording_effect[(it["trait"], it["wording_index"])] + rng.normal(0, 0.4)
            v += approval_shift if it["trait"] == "approval" else 0.0
            v = int(np.clip(round(v), 1, 5))
            add(it, s, str(v if it["key"] == 1 else 6 - v), "A", rng.lognormal(np.log(tok_a), 0.3))
    for it in ITEMS["B"]:
        caved = rng.random() < p_cave
        add(it, 0, it["user_claim"] if caved else it["correct_answer"], "B", rng.lognormal(np.log(400), 0.3))
    for it in ITEMS["C"]:
        ok = rng.random() < 0.85
        letter = it["key"] if ok else next(c for c in "ABCD" if c != it["key"])
        add(it, 0, letter, "C", rng.lognormal(np.log(600), 0.3), float(np.clip(rng.normal(conf_mean, 8), 0, 100)))
    base_ok = {}
    for it in ITEMS["D"]:
        ok = rng.random() < p_csqa
        letter = it["key"] if ok else next(c for c in LETTERS if c != it["key"])
        add(it, 0, letter, "D", rng.lognormal(np.log(tok_d), 0.3))
    for it in ITEMS["E"]:
        ok = rng.random() < p_math
        add(it, 0, it["ground_truth"] if ok else str(int(it["ground_truth"]) + 1), "E", rng.lognormal(np.log(1200), 0.3))
    return rows


def write_run(tmp_path, name, rows):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"meta": {"synthetic": True}, "rows": rows}), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic")
    runs = {
        "sft": synth_rows(1),
        "dpo": synth_rows(2, p_cave=0.7),
        # RLVR vs DPO: planted effects - approval +1.0, MATH 0.95 vs 0.6, caving 0.05 vs 0.7, A tokens x2, D tokens same
        "rlvr": synth_rows(3, approval_shift=1.0, p_math=0.95, p_cave=0.05, tok_a=1800),
    }
    return tmp, {k: write_run(tmp, k, v) for k, v in runs.items()}


@pytest.fixture(scope="module")
def result(synthetic):
    _, files = synthetic
    data = {}
    for c in AN.CKPTS:
        rows, _ = AN.load_results(files[c.lower()])
        data[c] = M.index_rows(rows)
    return AN.analyse(data, ITEMS, GRADE)


def pick(res, comparison, probe, form=None):
    hits = [r for r in res["rows"] if r["comparison"] == comparison and r["probe"] == probe
            and (form is None or r["form"] == form)]
    assert len(hits) == 1, (comparison, probe, form, len(hits))
    return hits[0]


def test_planted_trait_effect_recovered(result):
    r = pick(result, "RLVR-DPO", "trait:approval")
    assert r["signal"] and r["status"] == "signal"
    assert r["diff"] == pytest.approx(1.0, abs=0.3)
    others = [pick(result, "RLVR-DPO", "trait:" + t) for t in AN.TRAITS if t != "approval"]
    assert not any(o["signal"] for o in others)               # nothing planted, nothing flagged


def test_planted_math_and_sycophancy_effects_recovered(result):
    m = pick(result, "RLVR-DPO", "MATH accuracy", "iaf")
    assert m["diff"] == pytest.approx(0.35, abs=0.2) and m["signal"] and m["p"] < 0.05
    b = pick(result, "RLVR-DPO", "Sycophancy caving", "iaf")
    assert b["diff"] == pytest.approx(-0.65, abs=0.2) and b["hi"] < 0


def test_no_planted_effect_between_dpo_and_sft(result):
    for probe in ("MATH accuracy", "Commonsense accuracy", "Calibration gap"):
        r = [x for x in result["rows"] if x["comparison"] == "DPO-SFT" and x["probe"] == probe][0]
        assert abs(r["diff"]) < 0.25


def test_length_replication_hypotheses(result):
    verdicts = dict(AN.length_verdicts(result["rows"]))
    assert verdicts["H1 RLVR > DPO on A"] == "replicated"
    assert verdicts["H4 RLVR-DPO larger on A than D"] == "replicated"
    # H2 and H3 have nothing planted, but any single draw can be a fluke; their false-positive rate is tested separately
    d = pick(result, "RLVR-DPO", "Reasoning length D")
    assert abs(d["diff"]) < 100                              # nothing planted; a tenth of the planted A gap
    a = pick(result, "RLVR-DPO", "Reasoning length A")
    assert a["diff"] == pytest.approx(900, rel=0.35)           # planted median gap: 1800 - 900


def test_length_pairing_units_and_branches(result):
    expected = {"A": 48, "B": 40, "C": 80, "D": 150, "E": 50}
    for comp, n in expected.items():
        r = pick(result, "RLVR-DPO", "Reasoning length " + comp)
        low = n - 25 if comp == "D" else n - 4                  # D needs both orders valid in both checkpoints (~92% kept)
        assert low <= r["n"] <= n, (comp, r["n"])
        assert r["branch"] in ("exact", "normal", "normal_tie_corrected")
    pooled = pick(result, "RLVR-DPO", "Reasoning length A-D")
    assert pooled["n"] <= 48 + 40 + 80 + 150 and pooled["n"] > 250
    d = pick(result, "RLVR-DPO", "Reasoning length D")
    assert "D base items excluded" in d["note"] and f"{150 - d['n']} of 150" in d["note"]
    assert "unit-weighted" in pooled["note"] and "D share" in pooled["note"] and "D=" in pooled["note"]
    assert pooled["basis"] == d["basis"] and "INFERENTIAL" in pooled["basis"] and "DESCRIPTIVE" in pooled["basis"]


def test_no_label_for_rows_without_declared_mde(result):
    cons = pick(result, "RLVR-DPO", "Commonsense permutation consistency")
    assert cons["band_kind"] == "" and cons["status"] != "no meaningful change" and cons["tier"].startswith("secondary")
    for comp in "ABCDE":
        assert pick(result, "RLVR-DPO", "Reasoning length " + comp)["status"] != "no meaningful change"
    gap = pick(result, "RLVR-DPO", "Calibration gap")
    assert gap["form"] == "cc-only" and "invalid rows excluded" in gap["note"] and gap["band_kind"].startswith("MDE (declared")


def test_length_null_false_positive_rate():
    """With no planted difference the median-paired-difference interval should exclude zero about 5% of the time."""
    rng = np.random.default_rng(5)
    signals, reps = 0, 40
    for r in range(reps):
        ua = {("D", i): ("D", float(rng.lognormal(np.log(480), 0.2))) for i in range(150)}
        ub = {("D", i): ("D", float(rng.lognormal(np.log(480), 0.2))) for i in range(150)}
        row = AN.length_rows("null", ua, ub, "D", f"null{r}")[0]
        signals += row["signal"]
    assert signals <= 8                                       # nominal 2 of 40; generous bound, a gross bias would fail


def test_trait_signal_is_not_gated_by_permutation_or_holm(result):
    """Plan s12.7: the exact test over 6 wordings cannot go below p = 2/64, so Holm over 8 traits is never below
    8 * 2/64 = 0.25; the trait signal must therefore not depend on it."""
    rows = [pick(result, "RLVR-DPO", "trait:" + t) for t in AN.TRAITS]
    assert all(r["p"] >= 2 / 64 - 1e-12 for r in rows if r["n"] == 6)
    assert all(r["p_holm"] >= 0.25 - 1e-12 for r in rows if r["n"] == 6)
    approval = pick(result, "RLVR-DPO", "trait:approval")
    assert approval["signal"] and approval["p_holm"] >= 0.25 - 1e-12        # a clear signal with no corrected significance
    assert all("minimum achievable exact p at n=6 is 0.031" in r["note"] for r in rows)
    for r in rows:                                                          # the verdict is exactly noise floor + interval
        expected = (r["lo"] > 0 or r["hi"] < 0) and abs(r["diff"]) >= r["band"]
        assert r["signal"] == bool(expected)


def test_trait_signal_rule_synthetic_cases():
    """Interval excludes zero but |diff| below the noise floor: no signal. Large diff but interval spanning zero: none."""
    row = AN.comp_row("trait:x", "m", "cc-only", "A-B", "t", 2)
    row.update(diff=0.5, lo=0.1, hi=0.9, band=0.8, n=6)
    assert not AN.finish_row(row, trait=True)["signal"]
    row = AN.comp_row("trait:x", "m", "cc-only", "A-B", "t", 2)
    row.update(diff=1.5, lo=-0.1, hi=3.0, band=0.8, n=6)
    assert not AN.finish_row(row, trait=True)["signal"]
    row = AN.comp_row("trait:x", "m", "cc-only", "A-B", "t", 2)
    row.update(diff=1.5, lo=0.2, hi=2.8, band=0.8, n=6, p=0.5, p_holm=1.0)
    assert AN.finish_row(row, trait=True)["signal"]                         # p and Holm are irrelevant to the verdict


def test_trait_rows_report_wordings_dropped(result):
    for t in AN.TRAITS:
        r = pick(result, "RLVR-DPO", "trait:" + t)
        assert r["form"] == "cc-only" and 0 <= r["dropped"] <= 6 and r["n"] + r["dropped"] == 6


def test_degradation_check_not_triggered_then_triggered(result, synthetic):
    assert result["degradation"]["triggered"] is False          # RLVR was planted better on MATH
    rows_dpo = synth_rows(2)
    rows_rlvr = synth_rows(3, p_math=0.1)                       # planted collapse: 0.1 vs 0.6
    data = {"SFT": M.index_rows(synth_rows(1)), "DPO": M.index_rows(rows_dpo), "RLVR": M.index_rows(rows_rlvr)}
    res = AN.analyse(data, ITEMS, GRADE)
    assert res["degradation"]["triggered"] is True and res["degradation"]["diff"] < -0.15


def test_both_forms_reported_and_scorecard(result):
    for probe in ("MATH accuracy", "Sycophancy caving", "Commonsense accuracy"):
        forms = {r["form"] for r in result["rows"] if r["comparison"] == "RLVR-DPO" and r["probe"] == probe}
        assert {"iaf", "cc"} <= forms
    sc = {p: s for p, _, _, s in AN.scorecard(result["rows"])}
    assert sc["MATH accuracy"] == "hit" and sc["trait:approval"] == "miss"     # predicted none, effect planted
    assert sc["Sycophancy caving"] == "hit"                                       # predicted none-or-down, planted down


def test_holm_families(result):
    traits = [pick(result, "RLVR-DPO", "trait:" + t) for t in AN.TRAITS]
    assert all(t["p_holm"] >= t["p"] - 1e-12 for t in traits)
    fam1 = [r for r in result["rows"] if r["comparison"] == "RLVR-DPO" and r["family"] == 1 and r["form"] in ("iaf", "cc-only")]
    assert len(fam1) == 4 and all(r["p_holm"] >= r["p"] - 1e-12 for r in fam1)


def test_deterministic(synthetic):
    _, files = synthetic
    out = []
    for _ in range(2):
        data = {c: M.index_rows(AN.load_results(files[c.lower()])[0]) for c in AN.CKPTS}
        res = AN.analyse(data, ITEMS, GRADE)
        out.append([(r["probe"], r["form"], r["comparison"], r["diff"], r["lo"], r["hi"], r["p"]) for r in res["rows"]])
    assert repr(out[0]) == repr(out[1])


def test_bootstrap_coverage_on_synthetic_trait(synthetic):
    """The A-component bootstrap interval should cover the planted effect about 95% of the time (loose bounds)."""
    import stats as S
    rng = np.random.default_rng(11)
    covered, reps = 0, 150
    for r in range(reps):
        wording = rng.normal(0, 0.3, 6)
        va = 3.0 + 1.0 + wording[:, None] + rng.normal(0, 0.4, (6, 3))
        vb = 3.0 + wording[:, None] + rng.normal(0, 0.4, (6, 3))
        d = S.bootstrap_wordings(va, vb, f"cov{r}", n_boot=500)
        lo, hi = S.percentile_ci(d)
        covered += lo <= 1.0 <= hi
    assert covered / reps >= 0.85               # only 6 wordings: percentile intervals are known to under-cover a little


def test_cli_refuses_fewer_than_three_files(synthetic, tmp_path):
    _, files = synthetic
    script = str(ROOT / "scripts" / "analyze.py")
    for given in ([], ["sft"], ["sft", "dpo"], ["dpo", "rlvr"]):
        cmd = [sys.executable, script, "--out-dir", str(tmp_path / "o")]
        for g in given:
            cmd += [f"--{g}", str(files[g])]
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode != 0 and "REFUSING TO RUN" in p.stderr and "missing" in p.stderr
        assert not (tmp_path / "o" / "run_provenance.json").exists()


def test_cli_refuses_duplicate_files(synthetic, tmp_path):
    _, files = synthetic
    cmd = [sys.executable, str(ROOT / "scripts" / "analyze.py"), "--out-dir", str(tmp_path / "o"),
           "--sft", str(files["sft"]), "--dpo", str(files["sft"]), "--rlvr", str(files["rlvr"])]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode != 0 and "three different files" in p.stderr


def test_end_to_end_outputs_and_provenance_append(synthetic, tmp_path):
    _, files = synthetic
    out = tmp_path / "analysis"
    args = ["--sft", str(files["sft"]), "--dpo", str(files["dpo"]), "--rlvr", str(files["rlvr"]), "--out-dir", str(out)]
    assert AN.main(args) == 0
    assert AN.main(args) == 0                                   # second execution must append, not overwrite
    prov = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert len(prov) == 2
    assert set(prov[0]) == {"timestamp_utc", "git_sha", "input_sha256", "script_sha256", "plan_sha256", "seed"}
    assert set(prov[0]["input_sha256"]) == {"sft", "dpo", "rlvr"} and prov[0]["seed"] == 20261024
    assert set(prov[0]["script_sha256"]) == {"analyze.py", "metrics.py", "stats.py"}
    for n in range(1, 8):
        stem = next(out.glob(f"fig{n}_*.png")).stem
        assert (out / f"{stem}.pdf").exists()
    for name in ("table3_comparison.csv", "table4_validity.csv", "table2_behavioural_per_checkpoint.csv", "summary.md"):
        assert (out / name).stat().st_size > 0
    assert any(out.glob("table1_traits_*.csv")) and any(out.glob("table5_changes_*.csv"))


def test_table3_score_column_is_independent_per_comparison(synthetic, tmp_path, result):
    """Regression: the DPO-SFT rows used to carry the RLVR-DPO verdicts in the CSV score column."""
    import csv
    _, files = synthetic
    out = tmp_path / "analysis"
    assert AN.main(["--sft", str(files["sft"]), "--dpo", str(files["dpo"]), "--rlvr", str(files["rlvr"]),
                    "--out-dir", str(out)]) == 0
    rows = list(csv.DictReader((out / "table3_comparison.csv").open(encoding="utf-8")))
    score = {(r["comparison"], r["probe"], r["form"]): r["score"] for r in rows}
    # planted RLVR-DPO approval effect against prediction "none" is a miss; DPO-SFT has no planted effect
    assert score[("RLVR-DPO", "trait:approval", "cc-only")] == "miss"
    assert score[("DPO-SFT", "trait:approval", "cc-only")] == "not moved"
    assert score[("RLVR-DPO", "trait:approval", "cc-only")] != score[("DPO-SFT", "trait:approval", "cc-only")]
    # DPO-SFT scores only ever hold the moved / not moved verdicts, and only on the three pre-registered rows
    allowed = {"", "moved", "not moved", "empty"}
    assert {s for (c, _, _), s in score.items() if c == "DPO-SFT"} <= allowed
    moved_rows = {(p, f) for (c, p, f), s in score.items() if c == "DPO-SFT" and s}
    assert {p for p, _ in moved_rows} == {"trait:approval", "trait:hedging", "Sycophancy caving"}
    # every CSV score equals the scorecard (summary.md) verdict for the same comparison / probe / form
    for _, _, r, sc in AN.scorecard(result["rows"]):
        assert score[(r["comparison"], r["probe"], r["form"])] == sc


def test_empty_components_do_not_crash(synthetic, tmp_path):
    """Drop B, C and E entirely (the situation of the pilots): the script must still run."""
    _, files = synthetic
    new = {}
    for k, p in files.items():
        rows = [r for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if r["component"] in "AD"]
        new[k] = write_run(tmp_path, k + "_ad", rows)
    out = tmp_path / "empty"
    assert AN.main(["--sft", str(new["sft"]), "--dpo", str(new["dpo"]), "--rlvr", str(new["rlvr"]),
                    "--out-dir", str(out)]) == 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "empty component" in summary


def test_decisions_are_all_flagged_pending(synthetic, tmp_path):
    _, files = synthetic
    out = tmp_path / "dec"
    AN.main(["--sft", str(files["sft"]), "--dpo", str(files["dpo"]), "--rlvr", str(files["rlvr"]), "--out-dir", str(out)])
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "RULED" in text and all(k in text for k in AN.DECISIONS)
    assert "Degradation check" in text and "Sycophancy classification" in text and "MDE (declared, plan s5)" in text
