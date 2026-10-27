"""Frozen vs matched-system-prompt SFT: every frozen result that compares DPO with SFT, side by side.

    python -m exp.compare_sft_matched [--matched results/sprint/sft_matched_prompt/full_run_sft.json]
                                      [--out-dir results/sprint/sft_matched_prompt]

Registered in docs/sprint/registration-2026-10-27.md, amendment 1 (C3); the list of results is the one in
docs/sprint/disclosure-sft-system-prompt.md. Run it in the analysis environment (requirements_analysis.txt), the one the
frozen numbers were produced in. The self-checks below compare against the committed frozen outputs.

The frozen analysis code is used unchanged and nothing under results/analysis/ is written:
  - scripts/analyze.py analyse() on {frozen SFT, DPO, RLVR} and on {matched SFT, DPO, RLVR}. These give the DPO-SFT
    rows of table 3, H3 (length_verdicts) and C3 (the per-component DPO-SFT length rows);
  - results/analysis/exploratory_dithering.py report(), with its REPORT path redirected into --out-dir. It gives every
    exploratory DPO-SFT / RLVR-SFT row (measures 1-3, addenda 1-2 = C5, addendum 4 = C8);
  - the post-hoc extreme-answer share (A answers in {1, 5}, pooled over valid generations) and its per-wording
    paired DPO - SFT difference (registration statistics: mean, percentile bootstrap, Wilcoxon).
Self-checks: recomputing from the frozen files must reproduce the committed table 3 rows, the committed
exploratory_dithering.md and the registered extreme-answer shares. A failure is printed and the comparison is still
written, flagged.

Robustness rule (amendment 1): a result is robust to the system prompt if the matched primary statistic has the same
sign and the same verdict (table 3 / H3: signal flag; exploratory rows: CI excludes 0) as the frozen one.

Writes <out-dir>/dpo_sft_frozen_vs_matched.{md,csv} and the two recomputed dithering reports.
"""
import argparse
import collections
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {c: ROOT / "results" / f"full_run_{c}" / f"full_run_{c}.json" for c in ("sft", "dpo", "rlvr")}
DEFAULT_OUT = ROOT / "results" / "sprint" / "sft_matched_prompt"
SFT_CMP = ("DPO-SFT", "RLVR-SFT")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN = load_module("analyze", ROOT / "scripts" / "analyze.py")
DI = load_module("exploratory_dithering", ROOT / "results" / "analysis" / "exploratory_dithering.py")


def num(s):
    try:
        return float(str(s).replace("+", ""))
    except ValueError:
        return float("nan")


def same_sign(a, b):
    return np.sign(round(a, 12)) == np.sign(round(b, 12))


# ------------------------------------------------------------------ table 3, H3, C3 (scripts/analyze.py)
def analyse(files):
    items, grade_fn = AN.load_items(ROOT / "items")
    data = {c: AN.M.index_rows(AN.load_results(files[c.lower()])[0]) for c in AN.CKPTS}
    return AN.analyse(data, items, grade_fn)


def table3_rows(res):
    return {(r["probe"], r["form"]): (r, AN.row_cells(r, res)) for r in res["rows"] if r["comparison"] == "DPO-SFT"}


def committed_table3():
    with open(ROOT / "results" / "analysis" / "table3_comparison.csv", encoding="utf-8") as f:
        return {(r["probe"], r["form"]): [r[h] for h in AN.ROW_HEADER] for r in csv.DictReader(f)
                if r["comparison"] == "DPO-SFT"}


def c3_holds(rows):
    """C3: DPO lengthens B, C and D but not A (DPO-SFT reasoning-length rows)."""
    L = {k[0].split()[-1]: r for k, (r, _) in rows.items() if k[0].startswith("Reasoning length ") and "H4" not in k[0]}
    return all(L[c]["signal"] and L[c]["diff"] > 0 for c in "BCD") and not L["A"]["signal"]


# ------------------------------------------------------------------ extreme-answer share (post-hoc lead)
def extreme(files):
    out, per_w = {}, {}
    for c in ("sft", "dpo"):
        rows = [r for r in json.loads(Path(files[c]).read_text(encoding="utf-8"))["rows"]
                if r["component"] == "A" and r["status"] == "valid"]
        ext = [int(r["answer"]) in (1, 5) for r in rows]
        out[c] = (sum(ext) / len(ext), len(ext))
        acc = collections.defaultdict(list)
        for r, e in zip(rows, ext):
            acc[r["id"]].append(e)
        per_w[c] = {k: float(np.mean(v)) for k, v in acc.items()}
    ids = [k for k in per_w["dpo"] if k in per_w["sft"]]
    d = np.array([per_w["dpo"][k] - per_w["sft"][k] for k in ids])
    ci = DI.percentile_ci(DI.bootstrap_paired_differences(d, lambda x: x.mean(axis=1), "extreme|dpo-sft|mean"))
    return {"share": out, "pooled_diff": out["dpo"][0] - out["sft"][0], "n": len(ids), "mean": float(d.mean()),
            "ci": ci, "p": DI.wilcoxon_detail(d)["p"], "excl0": bool(ci[0] > 0 or ci[1] < 0)}


# ------------------------------------------------------------------ exploratory rows (exploratory_dithering.py)
def dithering_report(files, out_path):
    items = json.load(open(ROOT / "items" / "items_A_selfreport.json", encoding="utf-8"))["items"]
    rows = {c: [r for r in json.loads(Path(files[c]).read_text(encoding="utf-8"))["rows"] if r["component"] == "A"]
            for c in DI.CK}
    gens = DI.per_generation(rows, items)
    DI.REPORT = Path(out_path)
    DI.report(gens, items)
    traits, A = None, {}
    for c in DI.CK:
        traits, A[c] = DI._floor_array(gens, items, c)
    F = {c: DI._floors(A[c]) for c in DI.CK}
    FB, _ = DI._nested_floors(A)
    dist = (FB["dpo"] - FB["sft"]).mean(axis=1)
    c8 = {"obs": float((F["dpo"] - F["sft"]).mean()), "ci": DI.percentile_ci(dist)}
    c8["holds"] = bool(c8["ci"][1] < 0)          # C8: DPO more stable (lower floor) than SFT, nested interval excludes 0
    return Path(out_path).read_text(encoding="utf-8"), c8


def sft_rows(md):
    """[(section, quantity, comparison, cells)] for every table row comparing with SFT, in file order."""
    out, section, quantity = [], "", ""
    lines = md.splitlines()
    for j, line in enumerate(lines):
        if line.startswith("#"):
            section, quantity = line.lstrip("# ").strip(), ""
            continue
        if not line.startswith("|") or line.startswith("|---"):
            quantity = "" if not line.startswith("|") else quantity
            continue
        if j + 1 < len(lines) and lines[j + 1].startswith("|---"):     # header row of a table
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] and cells[0] not in SFT_CMP and cells[0] != "RLVR-DPO":
            quantity = cells[0]
        cmp_ = next((c for c in cells if c in SFT_CMP), None)
        if cmp_:
            out.append((section, quantity, cmp_, cells))
    return out


def primary_and_verdict(cells, cmp_):
    """Primary statistic (first numeric after the comparison, skipping an integer n; the mean if the row says so) and
    the 'CI excludes 0' yes/no cell."""
    rest = cells[cells.index(cmp_) + 1:]
    nums = [num(c.split(" [")[0]) for c in rest if math.isfinite(num(c.split(" [")[0]))]
    if rest and rest[0].isdigit():
        nums = nums[1:]
    v = nums[1] if "mean" in rest and len(nums) > 1 else (nums[0] if nums else float("nan"))
    yn = next((c for c in rest if c in ("yes", "no")), "")
    return v, yn


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--matched", default=str(DEFAULT_OUT / "full_run_sft.json"), help="matched-prompt SFT result file")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.resolve() == (ROOT / "results" / "analysis").resolve():
        sys.exit("refusing to write into results/analysis (frozen)")
    out.mkdir(parents=True, exist_ok=True)
    matched = {**FROZEN, "sft": Path(a.matched)}
    if not matched["sft"].is_file():
        sys.exit(f"matched SFT result file not found: {matched['sft']}")
    meta = json.loads(matched["sft"].read_text(encoding="utf-8"))["meta"]
    side = matched["sft"].with_name(matched["sft"].stem + "_system_prompt.json")
    checks, L, C = [], [], []

    def w(s=""):
        L.append(s)
        print(s)

    # 1. table 3, H3, C3
    rf, rm = analyse(FROZEN), analyse(matched)
    tf, tm = table3_rows(rf), table3_rows(rm)
    committed = committed_table3()
    ok_t3 = {k: [str(x) for x in v[1][:len(AN.ROW_HEADER)]] for k, v in tf.items()} == committed
    checks.append(("table 3 DPO-SFT rows recomputed from the frozen files == committed table3_comparison.csv", ok_t3))
    # 2. extreme answers
    ef, em = extreme(FROZEN), extreme(matched)
    ok_ex = (round(ef["share"]["sft"][0], 3), round(ef["share"]["dpo"][0], 3)) == (0.454, 0.203)
    checks.append(("extreme-answer shares from the frozen files == registered 0.454 (SFT) / 0.203 (DPO)", ok_ex))
    # 3. exploratory
    md_f, c8f = dithering_report(FROZEN, out / "exploratory_dithering_recomputed_frozen.md")
    md_m, c8m = dithering_report(matched, out / "exploratory_dithering_matched_sft.md")
    committed_md = (ROOT / "results" / "analysis" / "exploratory_dithering.md").read_text(encoding="utf-8")
    checks.append(("exploratory_dithering.md recomputed from the frozen files == committed file",
                   md_f.replace("\r\n", "\n") == committed_md.replace("\r\n", "\n")))
    xf, xm = sft_rows(md_f), sft_rows(md_m)
    if [x[:3] for x in xf] != [x[:3] for x in xm]:
        sys.exit("the frozen and matched dithering reports do not have the same SFT rows; cannot pair them")

    w("# DPO - SFT: frozen against matched system prompt")
    w()
    w(f"Matched SFT file: `{matched['sft'].relative_to(ROOT) if matched['sft'].is_relative_to(ROOT) else matched['sft']}` "
      f"(sha {meta.get('sha', '?')[:12]}, env {meta.get('env_hash', '?')}, dry_run {meta.get('dry_run')}). "
      f"Swap record: `{side.name}` {'present' if side.exists() else 'MISSING'}. Frozen DPO and RLVR files unchanged in both columns.")
    if meta.get("dry_run"):
        w("**DRY RUN FILE: plumbing only, not a result.**")
    w()
    w("## Self-checks (frozen recomputation)")
    w()
    for name, ok in checks:
        w(f"- {'PASS' if ok else 'FAIL'}: {name}")
    w()
    w("Robust = same sign of the primary statistic and same verdict (registration amendment 1).")
    w()

    w("## Table 3, DPO-SFT rows (scripts/analyze.py)")
    w()
    w("| probe | form | frozen diff [CI] | frozen status | matched diff [CI] | matched status | robust |")
    w("|---|---|---|---|---|---|---|")
    for k, (r, _) in tf.items():
        m = tm[k][0]
        rob = same_sign(r["diff"], m["diff"]) and r["signal"] == m["signal"]
        f_ = lambda x: f"{AN.fnum(x['diff'], 4)} [{AN.fnum(x['lo'], 4)}, {AN.fnum(x['hi'], 4)}]"
        w(f"| {k[0]} | {k[1]} | {f_(r)} | {r['status']} | {f_(m)} | {m['status']} | {'yes' if rob else 'NO'} |")
        C.append(["table3", k[0], k[1], "DPO-SFT", r["diff"], r["lo"], r["hi"], r["status"], m["diff"], m["lo"], m["hi"],
                  m["status"], rob])
    w()

    w("## Claims")
    w()
    h3f = dict(AN.length_verdicts(rf["rows"]))["H3 SFT ~ DPO overall, A-D pooled (no signal)"]
    h3m = dict(AN.length_verdicts(rm["rows"]))["H3 SFT ~ DPO overall, A-D pooled (no signal)"]
    c3f, c3m = c3_holds(tf), c3_holds(tm)
    w("| result | frozen | matched | robust |")
    w("|---|---|---|---|")
    w(f"| H3 SFT ~ DPO overall, A-D pooled (no signal) | {h3f} | {h3m} | {'yes' if h3f == h3m else 'NO'} |")
    w(f"| C3 DPO lengthens B, C, D but not A | {'holds' if c3f else 'does not hold'} | {'holds' if c3m else 'does not hold'} | "
      f"{'yes' if c3f == c3m else 'NO'} |")
    w(f"| C8 DPO more stable across seeds than SFT (addendum 4b: mean floor DPO-SFT, nested 95% CI) | "
      f"{c8f['obs']:+.3f} [{c8f['ci'][0]:+.3f}, {c8f['ci'][1]:+.3f}], {'holds' if c8f['holds'] else 'does not hold'} | "
      f"{c8m['obs']:+.3f} [{c8m['ci'][0]:+.3f}, {c8m['ci'][1]:+.3f}], {'holds' if c8m['holds'] else 'does not hold'} | "
      f"{'yes' if c8f['holds'] == c8m['holds'] and same_sign(c8f['obs'], c8m['obs']) else 'NO'} |")
    rob_e = same_sign(ef["mean"], em["mean"]) and ef["excl0"] == em["excl0"]
    e_ = lambda e: (f"pooled SFT {e['share']['sft'][0]:.3f} (n {e['share']['sft'][1]}), DPO {e['share']['dpo'][0]:.3f}; "
                    f"per-wording DPO-SFT {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}], p {e['p']:.4f}")
    w(f"| DPO-SFT extreme-answer drop (A answers in {{1, 5}}) | {e_(ef)} | {e_(em)} | {'yes' if rob_e else 'NO'} |")
    w()
    C += [["claim", "H3", "", "DPO-SFT", "", "", "", h3f, "", "", "", h3m, h3f == h3m],
          ["claim", "C3", "", "DPO-SFT", "", "", "", c3f, "", "", "", c3m, c3f == c3m],
          ["claim", "C8", "addendum 4b", "DPO-SFT", c8f["obs"], *c8f["ci"], c8f["holds"], c8m["obs"], *c8m["ci"],
           c8m["holds"], c8f["holds"] == c8m["holds"] and same_sign(c8f["obs"], c8m["obs"])],
          ["claim", "extreme-answer drop", "per-wording mean", "DPO-SFT", ef["mean"], *ef["ci"], ef["excl0"], em["mean"],
           *em["ci"], em["excl0"], rob_e]]

    w("## Exploratory rows comparing with SFT (results/analysis/exploratory_dithering.py; C5 = addenda 1-2, C8 = addendum 4)")
    w()
    w("| section | quantity | comparison | frozen row | matched row | robust |")
    w("|---|---|---|---|---|---|")
    for (sec, q, cmp_, cf), (_, _, _, cm) in zip(xf, xm):
        vf, yf = primary_and_verdict(cf, cmp_)
        vm, ym = primary_and_verdict(cm, cmp_)
        rob = same_sign(vf, vm) and yf == ym
        rest = lambda c: " ; ".join(c[c.index(cmp_) + 1:])
        w(f"| {sec[:60]} | {q[:70]} | {cmp_} | {rest(cf)} | {rest(cm)} | {'yes' if rob else 'NO'} |")
        C.append(["exploratory", f"{sec} :: {q}", "", cmp_, vf, "", "", yf, vm, "", "", ym, rob])
    w()
    (out / "dpo_sft_frozen_vs_matched.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(out / "dpo_sft_frozen_vs_matched.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["source", "result", "form", "comparison", "frozen_value", "frozen_lo", "frozen_hi", "frozen_verdict",
                     "matched_value", "matched_lo", "matched_hi", "matched_verdict", "robust"])
        cw.writerows(C)
    print(f"\nwrote {out / 'dpo_sft_frozen_vs_matched.md'} and .csv")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
