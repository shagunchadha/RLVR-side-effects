#!/usr/bin/env python3
"""Apply the author's blind check of the claude-only rows (spec section 2D). No model output is used.

Reads items/csqa_blind_check_reviewed.csv (author's my_pick and ambiguous), items/csqa_blind_check.csv (unfilled
original) and items/csqa_blind_check_key.json (check_id -> draw_index and key).
Validation (stops on any problem): every row has my_pick in A-E and ambiguous y or n; check_id, question and option
columns equal the original file.
Rule written in advance: an item is overturned if my_pick differs from the key OR ambiguous = y; more than 3 of the 29
overturned -> stop, nothing applied.
Applies: drop = x with reason "author blind check: ambiguous" (ambiguous = y, also when the pick differs) or
"author blind check: key mismatch" in items/csqa_review.csv (drop and reason cells only), and the provenance label
"author-blind-drop" in items/csqa_review_provenance.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_D_extend as bde  # noqa: E402
import build_D_finalize as bdf  # noqa: E402

MAX_OVERTURNED = 3


def evaluate(items=HERE):
    rv = bdf.read_rows(items / "csqa_blind_check_reviewed.csv")
    og = bdf.read_rows(items / "csqa_blind_check.csv")
    key = json.loads((items / "csqa_blind_check_key.json").read_text(encoding="utf-8"))["map"]
    if len(rv) != len(og) or list(rv[0]) != list(og[0]):
        sys.exit("reviewed and original blind-check files differ in shape")
    bad = [r["check_id"] for r in rv if r["my_pick"].strip() not in list("ABCDE") or r["ambiguous"].strip() not in ("y", "n")]
    if bad:
        sys.exit(f"my_pick not A-E or ambiguous not y/n on check_id {bad}")
    for a, o in zip(rv, og):
        for c in o:
            if c not in ("my_pick", "ambiguous") and a[c] != o[c]:
                sys.exit(f"check_id {o['check_id']}: column {c} differs from the original")
    over = []
    for r in rv:
        k = key[r["check_id"]]
        mismatch, amb = r["my_pick"].strip() != k["key"], r["ambiguous"].strip() == "y"
        if mismatch or amb:
            over.append({"draw_index": k["draw_index"], "pick_mismatch": mismatch, "ambiguous": amb,
                         "reason": "author blind check: ambiguous" if amb else "author blind check: key mismatch"})
    return sorted(over, key=lambda o: o["draw_index"])


if __name__ == "__main__":
    over = evaluate()
    print(f"overturned {len(over)}: pick-mismatch {sum(o['pick_mismatch'] for o in over)}, "
          f"ambiguous {sum(o['ambiguous'] for o in over)}; draw_index {[o['draw_index'] for o in over]}")
    if len(over) > MAX_OVERTURNED:
        sys.exit(f"more than {MAX_OVERTURNED} overturned; nothing applied")
    path = HERE / "csqa_review.csv"
    rows = bdf.read_rows(path)
    assert bde.serialize(rows) == path.read_bytes().decode("utf-8"), "csqa_review.csv does not round-trip; refusing"
    by = {int(r["draw_index"]): r for r in rows}
    prov_path = HERE / "csqa_review_provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    for o in over:
        r = by[o["draw_index"]]
        assert not r["drop"].strip(), f"draw_index {o['draw_index']} is already dropped"
        r["drop"], r["reason"] = "x", o["reason"]
        prov["per_draw_index"][str(o["draw_index"])] = "author-blind-drop"
    prov["sources"]["author-blind-drop"] = "claude-only row dropped after the author's blind check (pick differs from key or ambiguous)"
    before = {d: bdf.row_hash(r) for d, r in by.items()}
    path.write_bytes(bde.serialize(rows).encode("utf-8"))
    after = {int(r["draw_index"]): bdf.row_hash(r) for r in bdf.read_rows(path)}
    assert before == after, "a non-drop column changed"
    prov_path.write_text(json.dumps(prov, indent=1) + chr(10), encoding="utf-8")
    print("applied")
