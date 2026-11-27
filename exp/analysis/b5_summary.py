"""B5 summary: dose-response over RL steps (registration amendment 3 "B5", amendments 5, 8, 9).

    python -m exp.analysis.b5_summary            -> results/sprint/B5/summary.md

H (amendment 3): A-item think length and P(extreme) rise with RL step, and D length rises less. Primary (amendments 3,
5, 8): Spearman of A think length, A P(extreme) (sampled and readout) and D think length with log step over the 8
registered steps; the A-vs-D slope difference = OLS slope on log step of the per-unit DPO-standardised A length minus
that of D length. Units: A wording (48), D item (100); per-unit means over seeds. CIs: percentile bootstrap over units
(10,000, seed 20261024, per-label generator); for the slope difference A and D units are resampled independently.
DPO reference values: A = the B1 DPO main arm (same protocol and sampling); D = the frozen full-run DPO generation of
the same item (o0; one seed; think tokens counted with the DPO tokenizer). Main (amendment 9) is reported as a ninth
point next to the 8-step fits, not inside them.
"""
import collections
import json
import sys

import numpy as np

from . import sprint_common as C

STEPS = ("step_0025", "step_0050", "step_0075", "step_0150", "step_0250", "step_0425", "step_0775", "step_1375")
N_BOOT = 10000


def spearman_rows(M):
    """Spearman correlation of the columns of M (boot x steps) with log step, row-wise."""
    from scipy.stats import rankdata
    x = rankdata(np.log([int(s.split("_")[1]) for s in STEPS]))
    R = np.apply_along_axis(rankdata, 1, M)
    xc, Rc = x - x.mean(), R - R.mean(axis=1, keepdims=True)
    return (Rc @ xc) / np.sqrt((Rc ** 2).sum(axis=1) * (xc ** 2).sum())


def slope_rows(M):
    x = np.log([int(s.split("_")[1]) for s in STEPS])
    xc = x - x.mean()
    return ((M - M.mean(axis=1, keepdims=True)) @ xc) / (xc ** 2).sum()


def unit_matrix(per_step, units):
    """units x steps matrix of per-unit means (units missing at any step are dropped)."""
    us = [u for u in units if all(u in per_step[s] for s in STEPS)]
    return us, np.array([[per_step[s][u] for s in STEPS] for u in us], dtype=float)


def boot_stat(X, fn, label):
    """Point estimate and percentile CI of fn(column means of X), resampling rows (units)."""
    est = float(fn(X.mean(axis=0)[None, :])[0])
    idx = C.DI.make_rng(label).integers(0, X.shape[0], size=(N_BOOT, X.shape[0]))
    bs = np.stack([X[i].mean(axis=0) for i in idx])
    lo, hi = C.DI.percentile_ci(fn(bs))
    return est, lo, hi


def a_measures(rows):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["status"] != "valid" or r["answer_strict"] is None:
            continue
        per["len"][r["id"]].append(r["think_tokens"])
        per["ext"][r["id"]].append(int(r["answer_strict"]) in (1, 5))
        if r.get("read_primary") and r["p_read_1"] is not None:
            per["rext"][r["id"]].append(r["p_read_1"] + r["p_read_5"])
    return {k: {u: float(np.mean(v)) for u, v in d.items()} for k, d in per.items()}


def d_measures(rows):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["status"] == "valid" and r["think_tokens"] is not None:
            per["len"][r["id"]].append(r["think_tokens"])
        per["acc"][r["id"]].append(bool(r["correct"]))
        per["valid"][r["id"]].append(r["status"] == "valid")
    return {k: {u: float(np.mean(v)) for u, v in d.items()} for k, d in per.items()}


def dpo_d_reference(ids):
    """Think tokens of the frozen full-run DPO generation of each D item (o0), with the DPO tokenizer."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from ..config import CHECKPOINTS
    repo, sha = CHECKPOINTS["dpo"]
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=sha))
    rows = json.load(open(C.ROOT / "results" / "full_run_dpo" / "full_run_dpo.json", encoding="utf-8"))["rows"]
    out = collections.defaultdict(list)
    for r in rows:
        if r["component"] == "D" and r["id"] in ids and r["status"] == "valid" and "</think>" in r["raw_output"]:
            out[r["id"]].append(len(tok.encode(r["raw_output"].split("</think>")[0], add_special_tokens=False).ids))
    return {u: float(np.mean(v)) for u, v in out.items()}


def main():
    A_cols = ["id", "seed", "status", "answer_strict", "think_tokens", "p_read_1", "p_read_5", "read_primary"]
    D_cols = ["id", "seed", "status", "think_tokens", "correct"]
    A, D = {}, {}
    for f in C.fetch("runs/B5/"):
        A[f.parts[-3].split("@")[1].split("__")[0]] = a_measures(C.read(f, A_cols))
    for f in C.fetch("runs/B5D/"):
        D[f.parts[-3].split("@")[1]] = d_measures(C.read(f, D_cols))
    dpo_f = [f for f in C.fetch("runs/B1/dpo__sys-own/")]
    dpoA = a_measures(C.read(dpo_f[0], A_cols))
    a_units = sorted(dpoA["len"])
    d_units = sorted(D["step_0025"]["len"])
    dpoD = dpo_d_reference(set(d_units))
    L = ["# B5: dose-response over RL steps", "",
         "Generated by `exp/analysis/b5_summary.py` from the B5 shards on HF (`runs/B5/`, `runs/B5D/`). Registration: "
         "amendment 3 (B5), amendments 5, 8, 9. Units: A wording (48, 10 seeds), D item (first 100 base items, o0, 10 "
         "seeds). CIs: percentile bootstrap over units (10,000, seed 20261024, per-label generator). A *result* is a 95% "
         "CI excluding 0 in the registered direction.", "",
         "**Registered H:** A think length and P(extreme) rise with RL step, and D length rises less.", "",
         "## Per step (means over units)", "",
         "| step | A think tokens | A P(extreme) sampled | A readout P(1)+P(5) | D think tokens | D accuracy | D valid |",
         "|---|---|---|---|---|---|---|"]
    for s in STEPS + ("main",):
        a, d = A[s], D[s]
        L.append(f"| {s} | {np.mean(list(a['len'].values())):.0f} | {np.mean(list(a['ext'].values())):.3f} | "
                 f"{np.mean(list(a['rext'].values())):.3f} | {np.mean(list(d['len'].values())):.0f} | "
                 f"{np.mean(list(d['acc'].values())):.3f} | {np.mean(list(d['valid'].values())):.3f} |")
    L.append(f"| DPO (B1 main arm; frozen run for D) | {np.mean(list(dpoA['len'].values())):.0f} | "
             f"{np.mean(list(dpoA['ext'].values())):.3f} | {np.mean(list(dpoA['rext'].values())):.3f} | "
             f"{np.mean(list(dpoD.values())):.0f} ({len(dpoD)} items, 1 seed) | | |")
    L += ["", "## Primary: Spearman with log step (8 registered steps)", "",
          "| metric | Spearman ρ [95% CI] | registered | verdict |", "|---|---|---|---|"]
    res = {}
    for key, src, units, name, direc in [("len", A, a_units, "A think length", +1), ("ext", A, a_units, "A P(extreme), sampled", +1),
                                         ("rext", A, a_units, "A P(extreme), readout", +1), ("len", D, d_units, "D think length", 0)]:
        us, X = unit_matrix({s: src[s][key] for s in STEPS}, units)
        est, lo, hi = boot_stat(X, spearman_rows, f"B5|spearman|{name}")
        r = {"mean": est, "lo": lo, "hi": hi, "n": len(us), "p": float("nan")}
        res[name] = r
        v = C.verdict(r, direc) if direc else "reference (rises less than A: see slope difference)"
        L.append(f"| {name} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {len(us)} units) | {'rises' if direc else '—'} | {v} |")
    au, AX = unit_matrix({s: A[s]["len"] for s in STEPS}, [u for u in a_units if dpoA["len"].get(u)])
    du, DX = unit_matrix({s: D[s]["len"] for s in STEPS}, [u for u in d_units if dpoD.get(u)])
    AXs = AX / np.array([dpoA["len"][u] for u in au])[:, None]
    DXs = DX / np.array([dpoD[u] for u in du])[:, None]
    sa, sd = slope_rows(AXs.mean(axis=0)[None, :])[0], slope_rows(DXs.mean(axis=0)[None, :])[0]
    ra = C.DI.make_rng("B5|slope|A").integers(0, len(au), size=(N_BOOT, len(au)))
    rd = C.DI.make_rng("B5|slope|D").integers(0, len(du), size=(N_BOOT, len(du)))
    bs = slope_rows(np.stack([AXs[i].mean(axis=0) for i in ra])) - slope_rows(np.stack([DXs[i].mean(axis=0) for i in rd]))
    lo, hi = C.DI.percentile_ci(bs)
    slope = {"mean": float(sa - sd), "lo": lo, "hi": hi, "n": len(au), "p": float("nan")}
    sv = C.verdict(slope, +1)
    L += ["", "## Primary: A-vs-D slope difference in length", "",
          f"Per-unit think length divided by the unit's DPO value, then the OLS slope of the mean on log(step).", "",
          f"- A slope: {sa:+.4f} per log step ({len(au)} wordings); D slope: {sd:+.4f} ({len(du)} items).",
          f"- **A − D: {sa - sd:+.4f} [{lo:+.4f}, {hi:+.4f}]** (registered: A > D) → **{sv}**.",
          f"- Sensitivity, unstandardised slopes (tokens per log step): A {slope_rows(AX.mean(axis=0)[None, :])[0]:+.1f}, "
          f"D {slope_rows(DX.mean(axis=0)[None, :])[0]:+.1f}.", ""]
    L += ["## Main (ninth point) vs step_1375", "",
          "`B5/verify_last.json`: 345 of 355 tensors differ, max |diff| 4.9e-4 (1–2 bf16 ulps), relative L2 about 4e-4, "
          "about 7% of elements changed. Main is therefore reported as its own point (table above).", ""]
    for name, k, src, us in [("A think tokens", "len", A, a_units), ("A P(extreme) sampled", "ext", A, a_units),
                             ("D think tokens", "len", D, d_units)]:
        r = C.paired(src["main"][k], src["step_1375"][k], f"B5|main-1375|{k}|{name}")
        L.append(f"- main − step_1375, {name}: {C.fmt(r, 1 if 'tokens' in name else 3)}")
    L += ["", "## Verdict", ""]
    for name in ("A think length", "A P(extreme), sampled", "A P(extreme), readout"):
        r = res[name]
        L.append(f"- **{name} rises with log step: {C.verdict(r, +1)}** (ρ = {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]).")
    L.append(f"- **D length rises less than A (slope difference): {sv}** ({sa - sd:+.4f} [{lo:+.4f}, {hi:+.4f}]); "
             f"D Spearman ρ = {res['D think length']['mean']:+.3f} [{res['D think length']['lo']:+.3f}, {res['D think length']['hi']:+.3f}].")
    L += ["", "**Shape (descriptive, not registered):** the rise is not monotone. A length and P(extreme) dip slightly "
          "between step_0025 and step_0250, staying near the DPO values, and rise steeply from step_0425 on. With 8 points, "
          "the Spearman estimate is coarse (its CI is skewed); the slope difference and the per-step table carry the "
          "dose-response. Main writes about 200 more A think tokens than step_1375, although the weights differ only at "
          "the bf16-rounding level.",
          "", "Caveat: the D reference is a single frozen-run DPO generation per item (the frozen pipeline, not B5's "
          "protocol); it scales the D slope only. The unstandardised slopes are shown for that reason.", ""]
    out = C.ROOT / "results" / "sprint" / "B5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
