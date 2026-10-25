"""EXPLORATORY (plan s7; not pre-registered). Item 11 of exploratory_length.md.

Draws 5 component A wordings at random (numpy default_rng(20261024), from the wordings whose seed-0 generation is valid in
both RLVR and DPO) and writes the full seed-0 generation of RLVR and of DPO for each to
results/analysis/exploratory_A_samples.txt, for qualitative reading. Standalone (no metrics.py / stats.py).
Run from the repo root:  python results/analysis/sample_A_generations.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = {cp: {(r["id"], r["seed"]): r for r in json.load(open(ROOT / "results" / f"full_run_{cp}" / f"full_run_{cp}.json", encoding="utf-8"))["rows"]}
        for cp in ("dpo", "rlvr")}
items = {it["id"]: it for it in json.load(open(ROOT / "items" / "items_A_selfreport.json", encoding="utf-8"))["items"]}
ok = sorted(i for i in items if rows["rlvr"][(i, 0)]["status"] == "valid" and rows["dpo"][(i, 0)]["status"] == "valid")
pick = list(np.random.default_rng(20261024).choice(ok, size=5, replace=False))
out = []
for i in pick:
    out.append("=" * 100)
    out.append(f"{i}  trait={items[i]['trait']}  key={items[i]['key']}  statement: {items[i]['text']}")
    for cp in ("rlvr", "dpo"):
        r = rows[cp][(i, 0)]
        out.append("-" * 100)
        out.append(f"[{cp.upper()}] seed 0, n_tokens {r['n_tokens']}, answer {r['answer']!r}")
        out.append(r["raw_output"])
(ROOT / "results" / "analysis" / "exploratory_A_samples.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("picked", pick)
