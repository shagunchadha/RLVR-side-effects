#!/usr/bin/env python3
"""Component D, extension: append 150 more CommonsenseQA draws to items/csqa_review.csv and mark mechanical drops.

Needs `datasets` and network (tau/commonsense_qa validation). No model output is used.
  1. Reads the existing review CSV (200 rows, draw_index 0-199) and checks it has not been extended already.
  2. Pool = validation items whose `id` is NOT among the 200 ids in the CSV (dataset order);
     new = random.Random(20261021).sample(pool, 150), one fresh draw (not a larger re-sample).
  3. Appends them as draw_index 200-349 with the same columns, drop and reason empty. The first 200 rows are not
     re-drawn, reordered or edited by this step (the bytes up to the old end of file are verified unchanged).
  4. Mechanical rule (no judgement): any row, old or new, whose five option texts contain two that are identical after
     strip() and lower() gets drop = "x" and reason = "identical option text (mechanical)" if its drop cell is empty.
     This is the only cell the script ever fills. It is needed because a duplicated option makes the key and the
     permutation ambiguous by text. Rows marked are listed; clear the cell to override, but finalize refuses
     a duplicate-option row that is not dropped.
Writes items/csqa_review.csv in place (same csv dialect as build_D.py) and prints the list of mechanical drops.
Options --csv exists so the logic can be tested on a scratch copy.
"""
import argparse
import csv
import io
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261021
N_EXTRA = 150
COLS = ["draw_index", "id", "question", "A", "B", "C", "D", "E", "key", "drop", "reason"]
MECH_REASON = "identical option text (mechanical)"


def serialize(rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLS)   # same dialect build_D.py used (\r\n line ends, minimal quoting)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def has_duplicate_options(r):
    texts = [r[l].strip().lower() for l in "ABCDE"]
    return len(set(texts)) < len(texts)


def extend(csv_path, dataset_rows=None):
    raw = Path(csv_path).read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    assert len(rows) == 200, f"expected the 200 original rows, found {len(rows)} (already extended?)"
    assert serialize(rows).encode("utf-8") == raw, "the CSV does not round-trip byte for byte; refusing to rewrite it"
    if dataset_rows is None:
        from datasets import load_dataset
        dataset_rows = list(load_dataset("tau/commonsense_qa", split="validation"))
    used = {r["id"] for r in rows}
    assert len(used) == 200
    pool = [i for i, r in enumerate(dataset_rows) if r["id"] not in used]
    assert len(pool) == len(dataset_rows) - 200
    idx = random.Random(SEED).sample(pool, N_EXTRA)
    new_rows = []
    for n, i in enumerate(idx):
        r = dataset_rows[i]
        assert list(r["choices"]["label"]) == list("ABCDE")
        new_rows.append({"draw_index": 200 + n, "id": r["id"], "question": r["question"],
                         **{l: t for l, t in zip("ABCDE", r["choices"]["text"])},
                         "key": r["answerKey"], "drop": "", "reason": ""})
    allrows = rows + new_rows
    assert len({r["id"] for r in allrows}) == 350
    marked = []
    for r in allrows:
        if has_duplicate_options(r) and not r["drop"].strip():
            r["drop"], r["reason"] = "x", MECH_REASON
            marked.append(int(r["draw_index"]))
    out = serialize(allrows)
    # the part of the file that held the old rows must be unchanged except for the mechanical cells
    changed_old = [m for m in marked if m < 200]
    if not changed_old:
        assert out.encode("utf-8").startswith(raw), "the first 200 rows were altered"
    return out, marked, allrows, pool, idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "csqa_review.csv"))
    a = ap.parse_args()
    out, marked, allrows, pool, idx = extend(a.csv)
    Path(a.csv).write_bytes(out.encode("utf-8"))
    print(f"pool of unused validation items: {len(pool)}; drew {len(idx)} (seed {SEED}); CSV now has {len(allrows)} rows")
    print(f"mechanical drops ({len(marked)}): draw_index {marked}")
    for m in marked:
        r = allrows[m]
        print(f"  {m:3d} {r['id'][:8]} key={r['key']} | {r['question'][:70]} | " + " / ".join(r[l] for l in "ABCDE"))
    print(f"  of which in the first 200 rows: {[m for m in marked if m < 200]}; in the new 150: {[m for m in marked if m >= 200]}")
