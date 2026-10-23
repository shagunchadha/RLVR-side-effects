#!/usr/bin/env python3
"""Component D, step 1 of 2: draw 200 CommonsenseQA validation items for the blind review (spec section 2D).

Needs `datasets` and network (tau/commonsense_qa). Fixed seed 20261021:
    idx = random.Random(SEED).sample(range(len(validation)), 200)
Writes items/csqa_review.csv with columns
    draw_index, id, question, A, B, C, D, E, key, drop, reason
`drop` and `reason` are left EMPTY for the reviewer. No model output appears anywhere in the file.
Review rule (spec): drop an item when two or more options are defensible; write the reason for every drop.
STOP here: build_D_finalize.py is run only after the review is filled in.
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261021
N_DRAW = 200
PILOT_INDICES = range(10)   # probes/reasoning/commonsense_qa.py uses validation items 0-9


def draw():
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="validation")
    idx = random.Random(SEED).sample(range(len(ds)), N_DRAW)
    rows = []
    for n, i in enumerate(idx):
        r = ds[i]
        assert list(r["choices"]["label"]) == list("ABCDE")
        rows.append({"draw_index": n, "id": r["id"], "question": r["question"],
                     **{l: t for l, t in zip("ABCDE", r["choices"]["text"])},
                     "key": r["answerKey"], "drop": "", "reason": ""})
    return idx, len(ds), rows


if __name__ == "__main__":
    idx, n_val, rows = draw()
    cols = ["draw_index", "id", "question", "A", "B", "C", "D", "E", "key", "drop", "reason"]
    path = HERE / "csqa_review.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    overlap = sorted(set(idx) & set(PILOT_INDICES))
    print(f"validation size {n_val}; drew {len(rows)} (seed {SEED}); wrote {path.name}")
    print(f"overlap with the 10 pilot commonsense items (validation indices 0-9): {overlap}")
