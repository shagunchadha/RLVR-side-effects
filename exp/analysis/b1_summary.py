"""B1 first numbers from the gen/read progress files (works on partial data; counts are always shown).

    python -m exp.analysis.b1_summary [--dirs results/sprint/_lanes/N2/B1 results/sprint/B1/partial] [--out results/sprint/B1]

Arms are the progress directories <ckpt>__sys-<own|dpo_default>__<config hash>; an arm whose seeds are 0..4 is the
frozen-sampling bridge (0.7 / 0.9), 0..19 the main arm (model defaults). Per arm:
  valid share; sampled P(extreme) = share of valid answers in {1, 5}, per wording then mean over wordings (B1-H1 unit);
  mean sampled rating (as written); think-conditioned readout (primary variant): E[rating], P(1) + P(5), entropy;
  agreement = sampled answer == argmax of the readout, over valid generations with a readout, Wilson 95% (B1-H5);
  no-think prior: E[rating], P(1) + P(5) (one per wording).
Paired contrasts (per wording, n <= 48; mean of differences, percentile bootstrap 10,000, seed 20261024, per-label
generator; Wilcoxon two-sided): B1-H1 RLVR - DPO extreme share (main arms); B1-H2 acquiescence (plan s7: mean
scored rating of positive minus reverse-keyed wordings, per trait; RLVR - DPO over the 8 traits, sign count);
SFT(dpo_default) - SFT(own) (system-prompt effect); DPO - SFT(dpo_default) (matched) next to DPO - SFT(own).
"""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from .m2_summary import DI, ROOT

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def read_progress(f):
    out = {}
    if not f.exists():
        return out
    for r in csv.DictReader(open(f, encoding="utf-8")):
        try:
            out[r["key"]] = json.loads(r["row_json"])
        except ValueError:
            pass
    return out


def load(dirs):
    arms = {}
    for d in dirs:
        for a in sorted(Path(d).glob("*__sys-*__*")):
            g = read_progress(a / "gen_shard0of1.csv")
            r = read_progress(a / "read_shard0of1.csv")
            cur = arms.setdefault(a.name, {"gen": {}, "read": {}})
            if len(g) > len(cur["gen"]):
                cur["gen"] = g
            if len(r) > len(cur["read"]):
                cur["read"] = r
    return arms


def arm_label(name, gen):
    ck, sysm = name.split("__")[0], name.split("__")[1].replace("sys-", "")
    seeds = {r["seed"] for r in gen.values()}
    kind = "bridge (0.7/0.9, 5 seeds)" if seeds and max(seeds) < 5 else "main (model defaults)"
    return ck, sysm, kind


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def per_arm(gen, read):
    valid = [r for r in gen.values() if r["status"] == "valid" and r.get("answer_strict") is not None]
    ext = collections.defaultdict(list)
    for r in valid:
        ext[r["id"]].append(int(r["answer_strict"]) in (1, 5))
    rd = [(read.get(f"{r['id']}|{r['seed']}"), r) for r in valid]
    rd = [(x, r) for x, r in rd if x and x.get("read_primary")]
    agree = sum(1 for x, r in rd if int(np.argmax([x[f"p_read_{j}"] for j in range(1, 6)])) + 1 == int(r["answer_strict"]))
    pri = [v for k, v in read.items() if k.startswith("prior|")]
    prior = {k.split("|", 1)[1]: v for k, v in read.items() if k.startswith("prior|")}
    pE = {i: sum(j * v[f"p_nothink_{j}"] for j in range(1, 6)) for i, v in prior.items()}
    w = collections.defaultdict(lambda: collections.defaultdict(list))
    for x, r in rd:
        i = r["id"]
        w["read_ext"][i].append(x["p_read_1"] + x["p_read_5"])
        w["read_E"][i].append(x["read_expected"])
        w["read_H"][i].append(x["read_entropy"])
        w["agree"][i].append(int(np.argmax([x[f"p_read_{j}"] for j in range(1, 6)])) + 1 == int(r["answer_strict"]))
        if i in pE:
            w["shift"][i].append(abs(x["read_expected"] - pE[i]))
    wd = {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in w.items()}
    wd["prior_ext"] = {i: v["p_nothink_1"] + v["p_nothink_5"] for i, v in prior.items()}
    wd["prior_E"] = pE
    return {"w": wd, "n": len(gen), "valid": len(valid), "ext_w": {k: float(np.mean(v)) for k, v in ext.items()},
            "mean_rating": float(np.mean([int(r["answer_strict"]) for r in valid])) if valid else float("nan"),
            "read_E": float(np.mean([x["read_expected"] for x, _ in rd])) if rd else float("nan"),
            "read_ext": float(np.mean([x["p_read_1"] + x["p_read_5"] for x, _ in rd])) if rd else float("nan"),
            "read_H": float(np.mean([x["read_entropy"] for x, _ in rd])) if rd else float("nan"),
            "n_read": len(rd), "agree": agree,
            "prior_E": float(np.mean([sum((j) * p[f"p_nothink_{j}"] for j in range(1, 6)) for p in pri])) if pri else float("nan"),
            "prior_ext": float(np.mean([p["p_nothink_1"] + p["p_nothink_5"] for p in pri])) if pri else float("nan"),
            "n_prior": len(pri), "valid_rows": valid}


def paired(a, b, label):
    us = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in us])
    if len(d) < 2:
        return f"n = {len(d)}"
    ci = DI.percentile_ci(DI.bootstrap_paired_differences(d, lambda x: x.mean(axis=1), label))
    w = DI.wilcoxon_detail(d)
    return f"{d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], n = {len(d)}, Wilcoxon p = {w['p']:.3g}"


def acquiescence(valid, items):
    """{trait: mean scored rating of positive wordings - mean scored rating of reverse-keyed wordings}."""
    meta = {it["id"]: it for it in items}
    acc = collections.defaultdict(lambda: {1: [], -1: []})
    for r in valid:
        it = meta[r["id"]]
        a = int(r["answer_strict"])
        acc[it["trait"]][it["key"]].append(a if it["key"] == 1 else 6 - a)
    return {t: float(np.mean(v[1]) - np.mean(v[-1])) for t, v in acc.items() if v[1] and v[-1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="*", default=[str(ROOT / "results/sprint/_lanes/N2/B1"),
                                                  str(ROOT / "results/sprint/B1/partial")])
    ap.add_argument("--out", default=str(ROOT / "results" / "sprint" / "B1"))
    a = ap.parse_args(argv)
    items = json.load(open(ROOT / "items" / "items_A_selfreport.json", encoding="utf-8"))["items"]
    arms = load(a.dirs)
    A = {}
    for name, v in arms.items():
        A[arm_label(name, v["gen"])] = per_arm(v["gen"], v["read"])
    L = ["# B1 first numbers (Likert resampling with think-conditioned and no-think readouts)", "",
         "Generated by `exp/analysis/b1_summary.py` from the B1 progress files (partial data while B1 runs; counts shown). "
         "Registration: `docs/sprint/registration-2026-10-27.md` (B1; amendment 1 arms). Ratings as written (1-5), not "
         "reverse-scored, except in the acquiescence index.", "",
         "| checkpoint | system prompt | arm | generations | valid | sampled P(extreme) (per wording) | mean sampled rating | "
         "readout E[rating] | readout P(1)+P(5) | readout entropy | sampled = readout argmax [Wilson 95%] | no-think E[rating] | no-think P(1)+P(5) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(A, key=lambda k: (k[2], ["sft", "dpo", "rlvr"].index(k[0]), k[1])):
        s = A[key]
        lo, hi = wilson(s["agree"], s["n_read"])
        ag = f"{s['agree'] / s['n_read']:.3f} [{lo:.3f}, {hi:.3f}] (n {s['n_read']})" if s["n_read"] else "n/a"
        L.append(f"| {key[0].upper()} | {key[1]} | {key[2]} | {s['n']} | {s['valid']} | "
                 f"{np.mean(list(s['ext_w'].values())):.3f} ({len(s['ext_w'])} wordings) | {s['mean_rating']:.2f} | "
                 f"{s['read_E']:.2f} | {s['read_ext']:.3f} | {s['read_H']:.3f} | {ag} | {s['prior_E']:.2f} | "
                 f"{s['prior_ext']:.3f} ({s['n_prior']}) |")
    L += ["", "## Paired contrasts (per wording, main arms unless stated)", ""]
    g = lambda ck, sm="own", kind="main (model defaults)": A.get((ck, sm, kind))  # noqa: E731
    rows = [("B1-H1: RLVR - DPO, sampled P(extreme)", g("rlvr"), g("dpo"), "B1|H1"),
            ("DPO - SFT(own), sampled P(extreme) (confounded)", g("dpo"), g("sft"), "B1|dpo-sft-own"),
            ("DPO - SFT(dpo_default), sampled P(extreme) (matched prompt)", g("dpo"), g("sft", "dpo_default"), "B1|dpo-sft-matched"),
            ("SFT(dpo_default) - SFT(own): system-prompt effect on P(extreme)", g("sft", "dpo_default"), g("sft"), "B1|sft-prompt"),
            ("Bridge: RLVR - DPO, sampled P(extreme) at 0.7/0.9", g("rlvr", kind="bridge (0.7/0.9, 5 seeds)"),
             g("dpo", kind="bridge (0.7/0.9, 5 seeds)"), "B1|H1-bridge")]
    L += ["| contrast | mean difference [95% CI] |", "|---|---|"]
    for name, x, y, lab in rows:
        L.append(f"| {name} | {paired(x['ext_w'], y['ext_w'], lab) if x and y else 'arm(s) not available yet'} |")
    if g("rlvr") and g("dpo"):
        ar, ad = acquiescence(g("rlvr")["valid_rows"], items), acquiescence(g("dpo")["valid_rows"], items)
        ts = sorted(set(ar) & set(ad))
        d = np.array([ar[t] - ad[t] for t in ts])
        if len(d) >= 2:
            ci = DI.percentile_ci(DI.bootstrap_paired_differences(d, lambda x: x.mean(axis=1), "B1|H2"))
            L.append(f"| B1-H2: RLVR - DPO acquiescence index (mean over {len(d)} traits) | {d.mean():+.3f} [{ci[0]:+.3f}, "
                     f"{ci[1]:+.3f}]; RLVR lower on {int((d < 0).sum())} of {len(d)} traits |")
    if g("rlvr") and g("dpo"):
        L += ["", "### Gate 1 (a): RLVR - DPO per wording on the readouts (main arms; exploratory except H3/H4 as registered)", "",
              "| quantity | RLVR - DPO [95% CI] |", "|---|---|"]
        for k, name in [("read_ext", "think-conditioned readout P(1)+P(5)"), ("read_E", "readout E[rating]"),
                        ("read_H", "readout entropy"), ("agree", "agreement sampled = readout argmax"),
                        ("prior_ext", "B1-H3: no-think prior P(1)+P(5)"), ("prior_E", "no-think prior E[rating]"),
                        ("shift", "B1-H4: |E[r | think] - E[r | no-think]| (mean over seeds)")]:
            L.append(f"| {name} | {paired(g('rlvr')['w'].get(k, {}), g('dpo')['w'].get(k, {}), 'B1|gate1|' + k)} |")
        for ck in ("rlvr", "dpo"):
            sh = g(ck)["w"].get("shift", {})
            L.append(f"| {ck.upper()}: mean |shift from prior| (absolute, not a contrast) | "
                     f"{np.mean(list(sh.values())) if sh else float('nan'):.3f} over {len(sh)} wordings |")
    L += ["", "Registered directions: B1-H1 RLVR > DPO; B1-H2 RLVR < DPO (Holm over H1 and H2 when final). "
          "DPO - SFT contrasts are exploratory (amendment 1).", ""]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "b1_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
