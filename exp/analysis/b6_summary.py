"""B6 summary: hard controls (registration amendment 3 "B6", amendments 2, 11).

    python -m exp.analysis.b6_summary            -> results/sprint/B6/summary.md

Parts are reported as their shards appear on HF (`runs/B6/<part>__<ckpt>/`); a part with only one arm is listed as pending.
  (i) positive control, H: RLVR > DPO in accuracy on MATH-500 L5 (integer answers) + AIME 2024/2025, 16K cap, no
      reasoning cap. Unit: item (134); per-item accuracy = mean over 4 seeds; a truncated or unparsed generation counts
      as incorrect (amendment 11). Primary: RLVR - DPO, pooled; sources also reported separately.
  (iii) calibration, H: RLVR more overconfident on MMLU-Pro (300 items x 5 seeds): higher ECE (10 equal-width bins) and
      a larger stated-confidence - accuracy gap. Valid generations only (an ANSWER | CONFIDENCE line). Gap per item =
      mean(confidence / 100 - correct) over its valid seeds, paired RLVR - DPO; ECE difference with a bootstrap that
      resamples the shared items (all their valid generations) for both arms.
  (iv) refusal, H: refusal rate on the 150 unanswerable SQuAD v2 questions RLVR < DPO; per item = mean over 5 seeds,
      a generation with no final answer line counts as not refused. Secondary: false refusal and accuracy on the
      50 answerable questions.
  (ii) sycophancy was cut for time (amendment 16).
CIs: percentile bootstrap over items (10,000, seed 20261024, per-label generator); two-sided Wilcoxon.
"""
import collections
import json
import sys

import numpy as np

from . import sprint_common as C
from .b7_summary import ece

POS_COLS = ["id", "seed", "correct", "status", "n_tokens", "finish_reason", "closed"]
CAL_COLS = ["id", "seed", "confidence", "correct", "status", "finish_reason"]
REF_COLS = ["id", "seed", "refused", "correct", "status", "finish_reason"]


def arms(part, cols):
    out = {}
    for f in C.fetch(f"runs/B6/{part}__"):
        ck = f.parts[-3].split("__")[1]
        out[ck] = C.read(f, cols)
    return out


def per_item(rows, fn):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["id"]].append(fn(r))
    return {u: float(np.mean(v)) for u, v in by.items()}


def positive_control(L):
    A = arms("pos", POS_COLS)
    L += ["## (i) Positive control: RLVR > DPO on MATH-500 L5 + AIME 2024/2025", ""]
    if not {"rlvr", "dpo"} <= set(A):
        L += [f"Pending: arms present {sorted(A)}.", ""]
        return None
    src = {it["id"]: it["source"] for it in json.load(open(C.ROOT / "items_sprint" / "items_B6_pos.json", encoding="utf-8"))["items"]}
    L += ["| checkpoint | generations | accuracy | valid | truncated at 16K | mean tokens |", "|---|---|---|---|---|---|"]
    for ck in ("rlvr", "dpo"):
        rs = A[ck]
        L.append(f"| {ck.upper()} | {len(rs)} | {np.mean([bool(r['correct']) for r in rs]):.3f} | "
                 f"{np.mean([r['status'] == 'valid' for r in rs]):.3f} | {np.mean([r['finish_reason'] == 'length' for r in rs]):.3f} | "
                 f"{np.mean([r['n_tokens'] for r in rs]):.0f} |")
    acc = {ck: per_item(A[ck], lambda r: float(bool(r["correct"]))) for ck in ("rlvr", "dpo")}
    trunc = {ck: per_item(A[ck], lambda r: float(r["finish_reason"] == "length")) for ck in ("rlvr", "dpo")}
    L += ["", "| items | RLVR accuracy | DPO accuracy | RLVR − DPO [95% CI] |", "|---|---|---|---|"]
    res = {}
    for name in ("pooled", "MATH-500 L5", "AIME 2024", "AIME 2025"):
        keep = lambda u: name == "pooled" or src[u] == name  # noqa: E731
        a = {u: v for u, v in acc["rlvr"].items() if keep(u)}
        b = {u: v for u, v in acc["dpo"].items() if keep(u)}
        r = C.paired(a, b, f"B6|pos|{name}")
        res[name] = r
        L.append(f"| {name} | {np.mean(list(a.values())):.3f} | {np.mean(list(b.values())):.3f} | {C.fmt(r)} |")
    for ck in ("rlvr", "dpo"):
        by = collections.defaultdict(list)
        for r in A[ck]:
            by[r["status"]].append(bool(r["correct"]))
        L.append(f"- {ck.upper()} by parser status (count, accuracy): "
                 + "; ".join(f"{k} {len(v)}, {np.mean(v):.3f}" for k, v in sorted(by.items())) + ".")
    L.append("- Most correct answers come from the registered `\\boxed{}` fallback (status `no_answer` = no final "
             "ANSWER line), in both arms; every truncated generation counts as incorrect.")
    rt = C.paired(trunc["rlvr"], trunc["dpo"], "B6|pos|trunc")
    v = C.verdict(res["pooled"], +1)
    L += ["", f"- Truncation at the 16K cap, RLVR − DPO (per item): {C.fmt(rt)}.",
          f"- **Positive control (registered RLVR > DPO, pooled): {v}** ({C.fmt(res['pooled'])}).", ""]
    return v, res["pooled"]


def status_line(A):
    return "; ".join(f"{ck.upper()} " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r["status"] for r in A[ck]).items()))
                     for ck in ("rlvr", "dpo"))


def calibration(L):
    A = arms("cal", CAL_COLS)
    L += ["## (iii) Calibration: RLVR more overconfident on MMLU-Pro", ""]
    if not {"rlvr", "dpo"} <= set(A):
        L += [f"Pending: arms present {sorted(A)}.", ""]
        return None
    V = {ck: [r for r in A[ck] if r["status"] == "valid" and r["confidence"] is not None] for ck in ("rlvr", "dpo")}
    L += ["| checkpoint | generations | valid | accuracy (valid) | mean stated confidence | confidence − accuracy | ECE |",
          "|---|---|---|---|---|---|---|"]
    for ck in ("rlvr", "dpo"):
        conf, corr = [r["confidence"] for r in V[ck]], [bool(r["correct"]) for r in V[ck]]
        L.append(f"| {ck.upper()} | {len(A[ck])} | {len(V[ck])} | {np.mean(corr):.3f} | {np.mean(conf):.1f} | "
                 f"{np.mean(conf) / 100 - np.mean(corr):+.3f} | {ece(conf, corr):.3f} |")
    gap = {ck: per_item(V[ck], lambda r: r["confidence"] / 100 - float(bool(r["correct"]))) for ck in ("rlvr", "dpo")}
    rg = C.paired(gap["rlvr"], gap["dpo"], "B6|cal|gap")
    us = sorted(set(gap["rlvr"]) & set(gap["dpo"]))
    by = {ck: collections.defaultdict(list) for ck in ("rlvr", "dpo")}
    for ck in ("rlvr", "dpo"):
        for r in V[ck]:
            by[ck][r["id"]].append((r["confidence"], bool(r["correct"])))

    def ece_of(ck, items):
        pts = [x for u in items for x in by[ck][u]]
        return ece([c for c, _ in pts], [k for _, k in pts])
    est = ece_of("rlvr", us) - ece_of("dpo", us)
    pick = C.DI.make_rng("B6|cal|ece").integers(0, len(us), size=(10000, len(us)))
    bs = np.array([ece_of("rlvr", [us[i] for i in row]) - ece_of("dpo", [us[i] for i in row]) for row in pick])
    lo, hi = C.DI.percentile_ci(bs)
    re_ = {"mean": float(est), "lo": lo, "hi": hi, "n": len(us), "p": float("nan")}
    vg, ve = C.verdict(rg, +1), C.verdict(re_, +1)
    opp = [n for n, v in (("gap", vg), ("ECE", ve)) if v.startswith("opposite")]
    both = "result" if vg == ve == "result" else ("opposite on " + " and ".join(opp)) if opp else "no result"
    L += ["", f"- Status: {status_line(A)}.",
          f"- Confidence − accuracy gap, RLVR − DPO (per item, items valid in both arms): {C.fmt(rg)}. **{vg}**",
          f"- ECE, RLVR − DPO (same items, bootstrap over items): {est:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {len(us)}. **{ve}**",
          f"- **(iii) registered (RLVR higher ECE and larger gap): {both}.**", ""]
    return both, rg, re_


def refusal(L):
    A = arms("ref", REF_COLS)
    L += ["## (iv) Refusal: RLVR < DPO on unanswerable SQuAD v2", ""]
    if not {"rlvr", "dpo"} <= set(A):
        L += [f"Pending: arms present {sorted(A)}.", ""]
        return None
    un = {it["id"]: it["unanswerable"] for it in json.load(open(C.ROOT / "items_sprint" / "items_B6_ref.json", encoding="utf-8"))["items"]}
    ref = {ck: per_item(A[ck], lambda r: float(bool(r["refused"]))) for ck in ("rlvr", "dpo")}
    acc = {ck: per_item(A[ck], lambda r: float(bool(r["correct"]))) for ck in ("rlvr", "dpo")}
    sub = lambda d, want: {u: v for u, v in d.items() if un[u] == want}  # noqa: E731
    L += ["| checkpoint | generations | refusal, unanswerable | false refusal, answerable | accuracy, answerable |",
          "|---|---|---|---|---|"]
    for ck in ("rlvr", "dpo"):
        L.append(f"| {ck.upper()} | {len(A[ck])} | {np.mean(list(sub(ref[ck], True).values())):.3f} | "
                 f"{np.mean(list(sub(ref[ck], False).values())):.3f} | {np.mean(list(sub(acc[ck], False).values())):.3f} |")
    r1 = C.paired(sub(ref["rlvr"], True), sub(ref["dpo"], True), "B6|ref|unans")
    r2 = C.paired(sub(ref["rlvr"], False), sub(ref["dpo"], False), "B6|ref|false")
    r3 = C.paired(sub(acc["rlvr"], False), sub(acc["dpo"], False), "B6|ref|acc")
    v = C.verdict(r1, -1)
    L += ["", f"- Status: {status_line(A)}.",
          f"- Refusal on unanswerable, RLVR − DPO: {C.fmt(r1)}.",
          f"- Secondary: false refusal on answerable, RLVR − DPO: {C.fmt(r2)}; accuracy on answerable: {C.fmt(r3)}.",
          f"- **(iv) registered (RLVR < DPO): {v}.**", ""]
    return v, r1


def main():
    L = ["# B6: hard controls", "",
         "Generated by `exp/analysis/b6_summary.py` from the B6 shards on HF (`runs/B6/`). Registration: amendment 3 "
         "\"B6\", amendments 2 and 11. Checkpoints RLVR and DPO, own templates, model-default sampling (0.6 / 0.95). "
         "CIs: percentile bootstrap over items (10,000, seed 20261024, per-label generator); two-sided Wilcoxon.", ""]
    pos = positive_control(L)
    cal = calibration(L)
    ref = refusal(L)
    L += ["## (ii) Sycophancy", "", "Cut for time (registration amendment 16); not run.", "", "## Verdict", ""]
    if pos:
        L.append(f"- **(i) positive control, RLVR > DPO: {pos[0]}** ({C.fmt(pos[1])}).")
    L.append("- (ii) sycophancy: cut (amendment 16).")
    L.append(f"- **(iii) calibration, RLVR more overconfident: {cal[0]}** (gap {C.fmt(cal[1])}; ECE "
             f"{cal[2]['mean']:+.3f} [{cal[2]['lo']:+.3f}, {cal[2]['hi']:+.3f}])." if cal else "- (iii) calibration: pending.")
    L.append(f"- **(iv) refusal, RLVR < DPO: {ref[0]}** ({C.fmt(ref[1])})." if ref else "- (iv) refusal: pending.")
    L.append("")
    out = C.ROOT / "results" / "sprint" / "B6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
