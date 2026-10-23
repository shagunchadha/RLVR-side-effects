#!/usr/bin/env python3
"""Component D, step 2 of 2: finalize after the blind review (spec section 2D).

Reads items/csqa_review.csv (350 rows: draw_index 0-349) once the `drop` column (and a `reason` for every drop) is
filled in.
  - drop is any non-empty value other than 0/no/false/n (case-insensitive); every drop needs a reason;
  - rows with two or more identical option texts were marked drop=x by build_D_extend.py ("identical option text
    (mechanical)"); a duplicate-option row that is NOT dropped is an error;
  - the first 150 survivors in draw_index order are kept (needs >= 150 survivors, else exits with an error); the
    draw_index of the 150th survivor and the number of rows after it are printed, so the reviewer can confirm where
    review stopped (rows after that point are not used, whatever their drop cell says);
  - each kept item gets two option orders: o0 = the original order, o1 = one fixed shuffle per item
    (random.Random(SEED + draw_index), re-drawn until it differs from the identity);
  - the key is located BY INDEX (the CSV `key` letter), never by option text, so items with repeated option text
    cannot mislocate the key; each row stores `permutation` (position j holds original option permutation[j]),
    `key_index` (position of the key in this order) and `original_key_index`; permutation consistency must be
    computed on original option indices (items/grading.py original_option_index);
  - 300 rows, ids D_<draw_index>_o0 / _o1, labels A-E by position;
  - the CSV is read as utf-8-sig, so a BOM written by Excel does not rename the first column;
  - every column other than `drop` and `reason` is hashed per row (sha256 of the cell values) and compared with
    items/csqa_review_hashes.json (the saved original 200 rows plus the 150 appended rows, hashed before the review
    started); any changed row stops the script and names its draw_index, to catch accidental edits in Excel.
    `--write-hashes --original items/csqa_review_original.csv` (re)creates that file from the CSV and checks its
    first 200 rows against the saved pre-append CSV; do this only before the review starts.
Writes items/items_D_csqa.json and items/csqa_drops.json (every drop with its reason, plus surviving draws beyond the
150th that were not used). Does not read any model output.
Options --review / --out-dir exist so the logic can be tested on a scratch copy.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import build_messages_D  # noqa: E402

SEED = 20261021
N_KEEP = 150
N_ROWS = 350
FALSY = {"", "0", "no", "false", "n"}
HASH_FILE = HERE / "csqa_review_hashes.json"
EDITABLE = ("drop", "reason")


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_hash(r):
    """sha256 over every column except drop/reason (column names sorted, so column order does not matter)."""
    cols = sorted(c for c in r if c not in EDITABLE)
    blob = json.dumps([[c, r[c]] for c in cols], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_hashes(review_path, original_path, out_path=HASH_FILE):
    """Hash the 350 rows. The first 200 must equal the saved pre-append original (apart from drop/reason)."""
    rows = sorted(read_rows(review_path), key=lambda r: int(r["draw_index"]))
    orig = sorted(read_rows(original_path), key=lambda r: int(r["draw_index"]))
    assert len(rows) == N_ROWS and len(orig) == 200
    for o in orig:
        r = rows[int(o["draw_index"])]
        assert row_hash(o) == row_hash(r), f"draw_index {o['draw_index']} differs from the saved original"
    Path(out_path).write_text(json.dumps({"hash": "sha256 of [column, value] pairs, columns sorted, excluding drop/reason",
                                          "rows": {str(int(r["draw_index"])): row_hash(r) for r in rows}},
                                         indent=1) + chr(10), encoding="utf-8")
    return len(rows)


def check_unchanged(rows, hash_path):
    saved = json.loads(Path(hash_path).read_text(encoding="utf-8"))["rows"]
    for r in rows:
        want = saved.get(str(int(r["draw_index"])))
        if want is None or row_hash(r) != want:
            sys.exit(f"draw_index {r['draw_index']}: a column other than drop/reason differs from the saved original")
    if len(saved) != len(rows):
        sys.exit(f"hash file has {len(saved)} rows, review CSV has {len(rows)}")


def is_drop(v):
    return v.strip().lower() not in FALSY


def has_duplicate_options(r):
    texts = [r[l].strip().lower() for l in "ABCDE"]
    return len(set(texts)) < len(texts)


def make_items(r):
    """The two rows (original order, one fixed shuffle) of one review row. Key located by index."""
    d = int(r["draw_index"])
    opts = [r[l] for l in "ABCDE"]
    key_i = "ABCDE".index(r["key"])
    rng = random.Random(SEED + d)
    perm = list(range(5))
    while perm == list(range(5)):
        rng.shuffle(perm)
    out = []
    for o, order in ((0, list(range(5))), (1, perm)):
        o_opts = [opts[j] for j in order]
        new_key_i = order.index(key_i)
        out.append({
            "id": f"D_{d:03d}_o{o}", "component": "D", "kind": "mc5", "parser_kind": "mc5",
            "draw_index": d, "csqa_id": r["id"], "order": o, "permutation": order,
            "question": r["question"], "options": o_opts, "key": "ABCDE"[new_key_i], "key_index": new_key_i,
            "original_key_index": key_i, "key_text": opts[key_i],
            "messages": build_messages_D(r["question"], o_opts),
        })
    return out


def provisional_items(review_path, n_rows):
    """UNREVIEWED items (no drops applied) from the first draws; for the plumbing dry run only."""
    rows = sorted(read_rows(review_path), key=lambda r: int(r["draw_index"]))
    items = []
    for r in rows[: n_rows // 2 + 1]:
        items += make_items(r)
    return items


def finalize(review_path, out_dir, n_rows=N_ROWS, hash_path=HASH_FILE):
    """hash_path=None skips the unchanged-columns check (synthetic tests only)."""
    rows = read_rows(review_path)
    assert len(rows) == n_rows, f"expected {n_rows} review rows, found {len(rows)}"
    rows.sort(key=lambda r: int(r["draw_index"]))
    assert [int(r["draw_index"]) for r in rows] == list(range(n_rows)), "draw_index must be 0..n-1"
    if hash_path is not None:
        check_unchanged(rows, hash_path)
    drops, survivors = [], []
    for r in rows:
        if is_drop(r["drop"]):
            if not r["reason"].strip():
                sys.exit(f"draw_index {r['draw_index']}: dropped without a reason")
            drops.append({"draw_index": int(r["draw_index"]), "id": r["id"], "question": r["question"],
                          "reason": r["reason"].strip()})
        else:
            if has_duplicate_options(r):
                sys.exit(f"draw_index {r['draw_index']}: identical option texts but not dropped")
            survivors.append(r)
    if len(survivors) < N_KEEP:
        sys.exit(f"only {len(survivors)} survivors, need {N_KEEP}: review more rows")
    kept, unused = survivors[:N_KEEP], survivors[N_KEEP:]
    last = int(kept[-1]["draw_index"])
    past = [r for r in rows if int(r["draw_index"]) > last]
    past_marked = sum(1 for r in past if is_drop(r["drop"]))
    items = []
    for r in kept:
        items += make_items(r)
    assert len(items) == 2 * N_KEEP
    out_dir = Path(out_dir)
    meta = {"component": "D", "n_rows": len(items), "n_items": N_KEEP, "seed": SEED, "orders": 2,
            "last_kept_draw_index": last, "rows_after_last_kept": len(past),
            "source": "tau/commonsense_qa validation, 350 drawn (200 + 150), blind review, first 150 survivors",
            "answer_format": "ANSWER: <letter A-E>"}
    (out_dir / "items_D_csqa.json").write_text(
        json.dumps({"meta": meta, "items": items}, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    (out_dir / "csqa_drops.json").write_text(json.dumps({
        "n_reviewed": len(rows), "n_dropped": len(drops), "n_survivors": len(survivors), "n_kept": N_KEEP,
        "last_kept_draw_index": last, "rows_after_last_kept": len(past),
        "drops": drops,
        "survivors_not_used": [{"draw_index": int(r["draw_index"]), "id": r["id"]} for r in unused],
    }, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    return {"drops": len(drops), "survivors": len(survivors), "rows": len(items), "last_kept_draw_index": last,
            "rows_after_last_kept": len(past), "drop_cells_filled_after_last_kept": past_marked}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=str(HERE / "csqa_review.csv"))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--write-hashes", action="store_true", help="create items/csqa_review_hashes.json, then exit")
    ap.add_argument("--original", help="pre-append 200-row CSV, with --write-hashes")
    a = ap.parse_args()
    if a.write_hashes:
        print(f"hashed {write_hashes(a.review, a.original)} rows -> {HASH_FILE}")
        sys.exit(0)
    s = finalize(a.review, a.out_dir)
    print(f"dropped {s['drops']}, survivors {s['survivors']}, kept 150 -> {s['rows']} rows; "
          f"wrote items_D_csqa.json and csqa_drops.json to {a.out_dir}")
    print(f"150th survivor is draw_index {s['last_kept_draw_index']}; the CSV has {s['rows_after_last_kept']} rows after it "
          f"({s['drop_cells_filled_after_last_kept']} of them have a drop cell filled; none are used)")
