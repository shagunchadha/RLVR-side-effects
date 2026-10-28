"""Gate 1 (b2): amendment 4 follow-ups F1 (wrap-up onset), F2 (where KL spikes), F3 (prefix handoff). Exploratory.

    python -m exp.analysis.gate1_b2 [--m2 results/sprint/M2/partial] [--ph results/sprint/PH/partial]
                                    [--out results/sprint/M2/gate1_b2.md]

Inputs are the per-unit progress files (pulled from HF progress/): M2/<rlvr|dpo>_wrapup, M2/pair_rlvr_dpo_kldetail,
PH/<rlvr|dpo>__<hash>/gen_shard0of1.csv. Each section reports 'not available yet' if its input is missing.
Units: A wording, D base item; bootstrap 10,000, seed 20261024 (per-label generator); Wilcoxon two-sided.
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from .m2_summary import DI, load, unit, paired

ROOT = Path(__file__).resolve().parents[2]
ONSET_P = 0.2


def valid(r, comp, tc):
    return r.get("trace_ckpt") == tc and r["component"] == comp and r.get("status") == "valid" and "skipped" not in r


def wrapup_trace(r, no_so=False):
    """(mean P(wrap-up) at non-final sentence starts, onset in % of the think block or None if never >= ONSET_P).
    no_so (amendment 5 robustness): the group without 'So' = P(group) - P('So' first tokens); the first tokens are disjoint."""
    te, n = r.get("think_end_tok"), r.get("n_think_pos")
    pos, p = r.get("sent_next_tok") or [], r.get("p_wrapup_at_sent") or []
    if no_so:
        p = [x - y for x, y in zip(p, r.get("p_wrapup_So_at_sent") or [])]
    pairs = [(k, x) for k, x in zip(pos, p) if k != te]
    if not pairs or not n:
        return None, None
    onset = next((100.0 * k / n for k, x in pairs if x >= ONSET_P), None)
    return float(np.mean([x for _, x in pairs])), onset


def f1(S, L, no_so=False):
    R, D = S.get("rlvr_wrapup"), S.get("dpo_wrapup")
    L += ["## F1" + (" robustness (amendment 5): wrap-up onset WITHOUT 'So' (Therefore / Thus / Final)" if no_so else
                     ". Wrap-up onset: P(conclusion opener) at sentence starts, RLVR vs DPO scorer on the same text"), ""]
    if not R or not D:
        return L + ["Not available yet (M2 `--wrapup`, lane N6).", ""]
    L += [f"Primary group: first tokens of So / Therefore / Thus / Final. Onset = first non-final sentence start with "
          f"P(wrap-up) >= {ONSET_P}, in % of the think block (the realised close is at 100%); traces with no onset are "
          "counted separately.", "",
          "| component | traces | n units | mean P(wrap-up) RLVR | DPO | RLVR - DPO [95% CI] | onset RLVR (% think) | onset DPO | "
          "onset RLVR - DPO [95% CI] | no onset RLVR / DPO (traces) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for comp in "AD":
        for tc in ("rlvr", "dpo"):
            m = {}
            for lab, rows in (("r", R), ("d", D)):
                mw, on, none = collections.defaultdict(list), collections.defaultdict(list), 0
                for r in rows.values():
                    if not valid(r, comp, tc):
                        continue
                    a, o = wrapup_trace(r, no_so)
                    if a is None:
                        continue
                    mw[unit(r)].append(a)
                    if o is None:
                        none += 1
                    else:
                        on[unit(r)].append(o)
                m[lab] = ({u: float(np.mean(v)) for u, v in mw.items()}, {u: float(np.mean(v)) for u, v in on.items()}, none)
            sfx = "|noso" if no_so else ""
            pw = paired(m["r"][0], m["d"][0], f"gate1b2|wrapup|{comp}|{tc}{sfx}")
            po = paired(m["r"][1], m["d"][1], f"gate1b2|onset|{comp}|{tc}{sfx}")
            f = lambda q, nd=4: f"{q['mean']:+.{nd}g} [{q['ci'][0]:+.{nd}g}, {q['ci'][1]:+.{nd}g}] (p={q['p']:.2g})" if q.get("n") else "n/a"  # noqa: E731
            L.append(f"| {comp} | {tc.upper()} | {pw.get('n', 0)} | {pw.get('a', float('nan')):.4f} | {pw.get('b', float('nan')):.4f} | "
                     f"{f(pw)} | {po.get('a', float('nan')):.1f} | {po.get('b', float('nan')):.1f} | {f(po, 3)} | {m['r'][2]} / {m['d'][2]} |")
    return L + ["", "Onset lag = 100 - onset. Registered reading: the stopping decision shows up earlier than `</think>` if "
                "the lag is > 10% of the think block on average; RLVR postpones it if its onset is later than DPO's on the same text.", ""]


def f2(S, L, dec):
    kd = S.get("pair_rlvr_dpo_kldetail")
    L += ["## F2. Where KL(RLVR || DPO) spikes, and the tokens carrying it", ""]
    if not kd:
        return L + ["Not available yet (M2 `--pair --kl-detail`, lane N6).", ""]
    rows = {c: [r for r in kd.values() if valid(r, c, "rlvr") and r.get("kl_pos_bins")] for c in "AD"}
    dvals = [x for r in rows["D"] for x in r["kl_pos_bins"] if x is not None]
    thr = float(np.percentile(dvals, 95))
    L += [f"Threshold = 95th percentile of per-bin mean KL over RLVR-written D traces = {thr:.3f} (20 relative-position bins).", "",
          "| component | traces | mean KL by bin (1 = start ... 20 = end) | median first bin above threshold | traces never above | median bins above |",
          "|---|---|---|---|---|---|"]
    for c in "AD":
        rs = rows[c]
        B = np.array([[np.nan if x is None else x for x in r["kl_pos_bins"]] for r in rs])
        first = [next((j + 1 for j, x in enumerate(r["kl_pos_bins"]) if x is not None and x > thr), None) for r in rs]
        nab = [sum(1 for x in r["kl_pos_bins"] if x is not None and x > thr) for r in rs]
        fb = [x for x in first if x is not None]
        L.append(f"| {c} | {len(rs)} | {', '.join(f'{x:.2f}' for x in np.nanmean(B, axis=0))} | "
                 f"{np.median(fb) if fb else float('nan'):.1f} | {sum(x is None for x in first)} | {np.median(nab):.1f} |")
    L += ["", "Registered reading (A): early = median first bin <= 5 and median bins above <= 3; spread = median bins above >= 6.", ""]
    for c in "AD":
        tot, cnt = collections.Counter(), collections.Counter()
        for r in rows[c]:
            for t, s, k in ((int(t), s, int(k)) for t, s, k in r["kl_by_token_top"]):
                tot[t] += s
                cnt[t] += k
        allkl = sum(r["sum_kl_think"] for r in rows[c])
        L += [f"**Top 30 tokens by total KL, {c} (RLVR-written traces)**", "",
              "| rank | token | total KL | share of think KL | count | KL per occurrence |", "|---|---|---|---|---|---|"]
        for j, (t, s) in enumerate(tot.most_common(30), 1):
            L.append(f"| {j} | {dec(t)} | {s:.1f} | {s / allkl:.3f} | {cnt[t]} | {s / cnt[t]:.3f} |")
        L += ["", "Totals use each trace's top 50 tokens by KL (tails undercounted); the token is the one written at the position.", ""]
    return L


def f3(ph_dir, L):
    L += ["## F3. Prefix handoff: remaining think tokens, prefix source x continuing model", ""]
    rows = []
    shards = sorted((Path(ph_dir).parent / "runs").glob("*/*/shard_000.parquet"))
    if shards:                                    # finished shards (results/sprint/PH/runs/<ckpt>/<hash>/) win
        import pyarrow.parquet as pq
        for f in shards:
            rows += pq.read_table(f).to_pylist()
    else:
        for f in sorted(Path(ph_dir).glob("*/gen_shard0of1.csv")):
            rows += list(DI_read(f).values())
    if not {r["cont_ckpt"] for r in rows} >= {"rlvr", "dpo"}:
        return L + [f"Not available yet (PH, lane N6; continuations so far: {len(rows)}).", ""]
    L += ["L[cont, prefix] = mean over units of the mean remaining think tokens over seeds (cap hits counted at the cap). "
          "Text share = (L[DPO, RLVR] - L[DPO, DPO]) / (L[RLVR, RLVR] - L[DPO, DPO]); bootstrap over units with all four cells.", "",
          "| component | cut | n units | L[RLVR,RLVR] | L[DPO,RLVR] | L[RLVR,DPO] | L[DPO,DPO] | cont effect on RLVR prefixes (RLVR - DPO) | "
          "**text share** [95% CI] | cap hits | answer changed (DPO cont / RLVR cont) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for comp in "AD":
        for cut in (0.25, 0.5, 0.75):
            cell = collections.defaultdict(lambda: collections.defaultdict(list))
            ch = collections.defaultdict(list)
            caps = 0
            sel = [r for r in rows if r["component"] == comp and abs(r["cut"] - cut) < 1e-9]
            for r in sel:
                cell[(r["cont_ckpt"], r["prefix_src"])][unit(r)].append(r["remaining_think_tok"])
                caps += not r["closed"]
                if r["answer_changed"] is not None:
                    ch[r["cont_ckpt"]].append(r["answer_changed"])
            keys = [("rlvr", "rlvr"), ("dpo", "rlvr"), ("rlvr", "dpo"), ("dpo", "dpo")]
            us = sorted(set.intersection(*[set(cell[k]) for k in keys])) if all(cell[k] for k in keys) else []
            if not us:
                L.append(f"| {comp} | {cut:.0%} | 0 | | | | | | | {caps} | |")
                continue
            M = np.array([[np.mean(cell[k][u]) for k in keys] for u in us])        # units x 4
            mean = M.mean(0)
            share = (mean[1] - mean[3]) / (mean[0] - mean[3])
            idx = DI.make_rng(f"gate1b2|ph|{comp}|{cut}").integers(0, len(us), size=(DI.N_BOOT, len(us)))
            Mb = M[idx].mean(1)
            sb = (Mb[:, 1] - Mb[:, 3]) / (Mb[:, 0] - Mb[:, 3])
            ce = paired({u: M[j, 0] for j, u in enumerate(us)}, {u: M[j, 1] for j, u in enumerate(us)}, f"gate1b2|ph-cont|{comp}|{cut}")
            L.append(f"| {comp} | {cut:.0%} | {len(us)} | {mean[0]:.0f} | {mean[1]:.0f} | {mean[2]:.0f} | {mean[3]:.0f} | "
                     f"{ce['mean']:+.0f} [{ce['ci'][0]:+.0f}, {ce['ci'][1]:+.0f}] | **{share:.2f}** [{np.percentile(sb, 2.5):.2f}, "
                     f"{np.percentile(sb, 97.5):.2f}] | {caps}/{len(sel)} | {np.mean(ch['dpo']):.2f} / {np.mean(ch['rlvr']):.2f} |")
    return L + ["", "Registered reading (A, 50% cut): text share >= 0.5 -> the length lives in the text; <= 0.2 -> in RLVR's policy.", ""]


def ph_rows(ph_dir):
    shards = sorted((Path(ph_dir).parent / "runs").glob("*/*/shard_000.parquet"))
    if shards:
        import pyarrow.parquet as pq
        return [r for f in shards for r in pq.read_table(f).to_pylist()]
    return [r for f in sorted(Path(ph_dir).glob("*/gen_shard0of1.csv")) for r in DI_read(f).values()]


def f3b(ph_dir, L):
    """Amendment 9: extremity (A) / accuracy (D) of the PH continuations by prefix source x continuing model."""
    from .. import config as C
    L += ["## F3b. Does extremity follow the continuing model? (PH continuations, final sampled answers, valid only)", ""]
    rows = ph_rows(ph_dir)
    if not {r["cont_ckpt"] for r in rows} >= {"rlvr", "dpo"}:
        return L + ["Not available yet.", ""]
    gold = {it["id"]: it["key"] for it in C.load_items("D")}
    keys = [("rlvr", "rlvr"), ("dpo", "rlvr"), ("rlvr", "dpo"), ("dpo", "dpo")]
    for comp, metrics in (("A", (("ext", "P(extreme)"), ("rating", "mean rating"))),
                          ("D", (("acc", "accuracy"), ("agree", "agreement with the trace's answer")))):
        for mk, mname in metrics:
            L += [f"**{comp}: {mname}**", "",
                  "| cut | n units | [RLVR cont, RLVR prefix] | [DPO, RLVR] | [RLVR, DPO] | [DPO, DPO] | continuer effect on RLVR prefixes | "
                  "continuer effect on DPO prefixes | prefix effect, DPO continuer | prefix effect, RLVR continuer |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for cut in (0.25, 0.5, 0.75):
                cell = collections.defaultdict(lambda: collections.defaultdict(list))
                for r in rows:
                    if r["component"] != comp or abs(r["cut"] - cut) > 1e-9 or r.get("status") != "valid" or r.get("answer") is None:
                        continue
                    a = r["answer"]
                    if mk == "ext":
                        v = float(a in ("1", "5"))
                    elif mk == "rating":
                        v = float(a)
                    elif mk == "acc":
                        v = float(a == gold.get(r["id"]))
                    else:
                        v = float(a == r.get("trace_answer"))
                    cell[(r["cont_ckpt"], r["prefix_src"])][unit(r)].append(v)
                m = {k: {u: float(np.mean(v)) for u, v in cell[k].items()} for k in keys}
                q = lambda x, y, lab: paired(m[x], m[y], f"gate1b2|f3b|{comp}|{mk}|{cut}|{lab}")  # noqa: E731
                f = lambda d: f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d.get("n", 0) >= 2 else "n/a"  # noqa: E731
                n = len(set.intersection(*[set(m[k]) for k in keys])) if all(m[k] for k in keys) else 0
                L.append(f"| {cut:.0%} | {n} | " + " | ".join(f"{np.mean(list(m[k].values())):.3f}" if m[k] else "-" for k in keys)
                         + f" | {f(q(('rlvr', 'rlvr'), ('dpo', 'rlvr'), 'cont_on_r'))} | {f(q(('rlvr', 'dpo'), ('dpo', 'dpo'), 'cont_on_d'))}"
                         + f" | {f(q(('dpo', 'rlvr'), ('dpo', 'dpo'), 'src_dpo'))} | {f(q(('rlvr', 'rlvr'), ('rlvr', 'dpo'), 'src_rlvr'))} |")
            L.append("")
    return L + ["Registered reading (A, 50% cut): extremity follows the continuing model if the continuer effect on P(extreme) "
                "is positive on both prefix sources (interval excludes 0) while the prefix effect for the DPO continuer is not.", ""]


def DI_read(f):
    from .b1_summary import read_progress
    return read_progress(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", default=str(ROOT / "results/sprint/M2/partial"))
    ap.add_argument("--ph", default=str(ROOT / "results/sprint/PH/partial"))
    ap.add_argument("--out", default=str(ROOT / "results/sprint/M2/gate1_b2.md"))
    a = ap.parse_args(argv)
    S = load(a.m2)
    # finished shards (results/sprint/M2/runs/<label>/<hash>/shard_000.parquet) win over mid-run progress snapshots
    for f in sorted((Path(a.m2).parent / "runs").glob("*/*/shard_000.parquet")):
        import pyarrow.parquet as pq
        try:
            rows = pq.read_table(f).to_pylist()
        except Exception as e:                                 # noqa: BLE001 - a torn copy: keep the progress file
            print(f"skipping unreadable shard {f}: {type(e).__name__}")
            continue
        S[f.parts[-3]] = {f"{r['trace_ckpt']}|{r['id']}|{r['seed']}": r for r in rows}
    try:
        from transformers import AutoTokenizer
        from .. import config as C
        tok = AutoTokenizer.from_pretrained(C.ckpt("dpo")[0], revision=C.ckpt("dpo")[1])
        dec = lambda t: repr(tok.decode([t]))  # noqa: E731
    except Exception:                                          # noqa: BLE001 - laptop without transformers
        try:
            from tokenizers import Tokenizer
            from .. import config as C
            tk = Tokenizer.from_pretrained(C.ckpt("dpo")[0], revision=C.ckpt("dpo")[1])
            dec = lambda t: repr(tk.decode([t], skip_special_tokens=False))  # noqa: E731
        except Exception:                                      # noqa: BLE001
            dec = lambda t: f"id {t}"  # noqa: E731
    L = ["# Gate 1 (b2): amendment 4 follow-ups (exploratory)", "",
         "Generated by `exp/analysis/gate1_b2.py`. Registration: amendment 4 (F1-F3).", ""]
    L = f1(S, L)
    L = f1(S, L, no_so=True)
    L = f2(S, L, dec)
    L = f3(a.ph, L)
    L = f3b(a.ph, L)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
