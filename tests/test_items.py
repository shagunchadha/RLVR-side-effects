#!/usr/bin/env python3
"""Checks for the item files, prompt template, grader, D logic and run-order construction.
Run: python tests/test_items.py   (no dependencies, no GPU, no network)
The D checks run on SYNTHETIC reviews in a temp dir; the real csqa_review.csv is never modified.
"""
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "items"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import build_D_extend as bde  # noqa: E402
import build_D_finalize as bdf  # noqa: E402
import full_run  # noqa: E402
from grading import grade_integer, normalise_integer, original_option_index, permutation_consistent  # noqa: E402
from prompts import REASONING_CAP  # noqa: E402

fails = []


def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name} {detail}")


def load(name):
    return json.loads((ROOT / "items" / name).read_text(encoding="utf-8"))


# ---- A ----
A = load("items_A_selfreport.json")["items"]
check("A count", len(A) == 48)
for t in {i["trait"] for i in A}:
    w = [i for i in A if i["trait"] == t]
    check(f"A {t} 6 wordings", len(w) == 6)
    check(f"A {t} keys", sorted(i["key"] for i in w) == [-1, -1, -1, 1, 1, 1])
    check(f"A {t} one original", sum(i["original"] for i in w) == 1)
check("A 8 traits", len({i["trait"] for i in A}) == 8)
from probes import ALL_PROBES, build_chat  # noqa: E402
pilot = {p["id"]: build_chat(p) for p in ALL_PROBES}
for i in A:
    check(f"A {i['id']} cap sentence last", i["messages"][-1]["content"].endswith(REASONING_CAP))
    if i["original"]:
        check(f"A {i['id']} == pilot", i["messages"] == pilot[f"trait_{i['trait']}_selfreport"])

# ---- B ----
B = load("items_B_sycophancy.json")["items"]
check("B count", len(B) == 40)
check("B kinds", sum(i["kind"] == "wrong_user" for i in B) == 30 and sum(i["kind"] == "right_user" for i in B) == 10)
for i in B:
    check(f"B {i['id']} 3 turns", [m["role"] for m in i["messages"]] == ["user", "assistant", "user"])
    check(f"B {i['id']} cap", i["messages"][-1]["content"].endswith(REASONING_CAP))
    check(f"B {i['id']} first answer", (i["correct_answer"] in i["assistant_first"]) == (i["kind"] == "wrong_user"))
    check(f"B {i['id']} not pilot item", "17 * 24" not in i["question"])

# ---- C: structure and exact key balance ----
C = load("items_C_calibration.json")["items"]
check("C count", len(C) == 80)
check("C subsets", sum(i["subset"] == "ARC-Easy" for i in C) == 40 and sum(i["subset"] == "ARC-Challenge" for i in C) == 40)
check("C four options", all(len(i["options"]) == 4 and i["key"] in "ABCD" for i in C))
check("C unique", len({i["arc_id"] for i in C}) == 80)
check("C key letters 20/20/20/20", sorted(sum(i["key"] == l for i in C) for l in "ABCD") == [20, 20, 20, 20])
for i in C:
    p = i["permutation"]
    check(f"C {i['id']} permutation", sorted(p) == [0, 1, 2, 3] and i["options"] == [i["original_options"][j] for j in p])
    check(f"C {i['id']} key follows", i["options"]["ABCD".index(i["key"])] == i["original_options"]["ABCD".index(i["original_key"])])
    check(f"C {i['id']} prompt shows balanced order",
          all(f"{l}. {t}" in i["messages"][0]["content"] for l, t in zip("ABCD", i["options"])))

# ---- E (MATH-500) ----
E = load("items_E_math.json")["items"]
EL = load("items_E_lengthcheck.json")["items"]
check("E count", len(E) == 50)
check("E ids", [i["id"] for i in E] == [f"E_{n:02d}" for n in range(50)])
for lv in range(1, 6):
    check(f"E level {lv}", sum(i["level"] == lv for i in E) == 10)
    check(f"E length-check level {lv}", sum(i["level"] == lv for i in EL) == 2)
check("E integer refs", all(re.fullmatch(r"-?\d+", i["ground_truth"]) for i in E + EL))
check("E ground truth is the normalised reference", all(i["ground_truth"] == normalise_integer(i["reference_raw"]) for i in E + EL))
check("E no [asy]", not any("[asy]" in i["problem"] for i in E + EL))
check("E length-check disjoint", not ({i["unique_id"] for i in E} & {i["unique_id"] for i in EL}))
check("E length-check count", len(EL) == 10 and [i["id"] for i in EL] == [f"EL_{n:02d}" for n in range(10)])
check("E kind math", all(i["parser_kind"] == "math" for i in E + EL))
check("E template", all(i["messages"][0]["content"].endswith("ANSWER: <integer>." + REASONING_CAP) for i in E + EL))
check("E zero-shot", all("Question:" not in i["messages"][0]["content"] for i in E + EL))

# ---- E contamination swap, synthetic MATH-500 rows (no network) ----
import random as _r  # noqa: E402
import build_E as bE  # noqa: E402
import build_contamination_flags as bcf  # noqa: E402
fake = [{"problem": f"p{lv}_{k}", "answer": str(k + 1), "subject": "s", "level": lv, "unique_id": f"test/s/{lv}_{k}.json"}
        for lv in range(1, 6) for k in range(40)]
b_items, b_check, _, _, b_sw = bE.build(fake)
check("swap: no flags -> no swaps", b_sw == [])
victim = b_items[11]
flag_ids = {victim["unique_id"], b_items[3]["unique_id"]}
s_items, s_check, _, _, s_sw = bE.build(fake, flagged=flag_ids)
check("swap: ids kept", [i["id"] for i in s_items] == [i["id"] for i in b_items])
check("swap: only flagged positions change",
      [n for n in range(50) if s_items[n]["unique_id"] != b_items[n]["unique_id"]] == [3, 11])
check("swap: unchanged items identical", all(s_items[n] == b_items[n] for n in range(50) if n not in (3, 11)))
check("swap: length-check set unchanged", s_check == b_check)
check("swap: same level kept", all(s_items[n]["level"] == b_items[n]["level"] for n in (3, 11)))
check("swap: replacement unflagged and unused", not ({s_items[n]["unique_id"] for n in (3, 11)} & flag_ids)
      and len({i["unique_id"] for i in s_items + s_check}) == 60)
pool11 = sorted((x for x in fake if x["level"] == victim["level"] and x["unique_id"] not in flag_ids
                 and x["unique_id"] not in {i["unique_id"] for i in b_items + b_check}), key=lambda x: x["unique_id"])
check("swap: E_11 uses Random(20261021 + 11) on the sorted pool",
      s_items[11]["unique_id"] == _r.Random(20261021 + 11).choice(pool11)["unique_id"])
check("swap: deterministic", bE.build(fake, flagged=flag_ids)[0] == s_items)
try:
    bE.build(fake, flagged={b_check[0]["unique_id"]})
    check("swap: flagged length-check stops", False, "did not exit")
except SystemExit:
    pass
with tempfile.TemporaryDirectory() as _td:
    for st, ids in (("RL", ["a/1"]), ("DPO", []), ("SFT", ["a/1", "b/2"])):
        (Path(_td) / f"flags_{st}.json").write_text(json.dumps({
            "dataset": "d", "stage": st, "rows_scanned": 1, "math500": {}, "rlvr_math": {},
            "math500_flagged": [{"unique_id": u, "overlap_fraction": 0.6, "exact_matches": 0} for u in ids]}), encoding="utf-8")
    fl = bcf.assemble(_td)
    check("flags: union and rows", fl["flagged_unique_ids"] == ["a/1", "b/2"] and len(fl["flags"]) == 3
          and fl["n_flagged_unique_problems"] == 2)
real_flags = bE.load_flagged()
if real_flags is not None:
    check("E (real): no flagged problem in E or length-check", not ({i["unique_id"] for i in E + EL} & real_flags))

# ---- grading ----
for raw, ref, want in [("12", "12", True), (BOX := chr(92) + "boxed{12}", "12", True), ("$1,234$", "1234", True), ("12.0", "12", True),
                       ("12.5", "12", False), ("13", "12", False), ("1/2", "1", False), ("", "0", False), (None, "0", False),
                       ("-7", "-7", True), ("007", "7", True), ("2021", "2021_3", False), ("$12$", "$12$", True)]:
    check(f"grade {raw!r} vs {ref!r}", grade_integer(raw, ref) == want)
check("normalise", normalise_integer(chr(92) + "boxed{1,000}") == "1000")
check("normalise reference with $", normalise_integer("$1,000.0$") == "1000")


# ---- D: key by index, duplicate option text, permutation consistency ----
def row(d, opts, key, **kw):
    return {"draw_index": d, "id": f"id{d}", "question": f"Question {d}?", **dict(zip("ABCDE", opts)),
            "key": key, "drop": kw.get("drop", ""), "reason": kw.get("reason", "")}


dup = row(7, ["same", "same", "y", "z", "w"], "B")
o0, o1 = bdf.make_items(dup)
check("dup: original key index", o0["key_index"] == 1 and o0["key"] == "B")
check("dup: text lookup would be wrong here", o0["options"].index("same") != o0["key_index"])
check("dup: key follows permutation", o1["permutation"][o1["key_index"]] == 1)
check("dup: key letter matches index", o1["key"] == "ABCDE"[o1["key_index"]])
check("dup: consistent when the same original option is chosen",
      permutation_consistent(o0, "B", o1, "ABCDE"[o1["permutation"].index(1)]))
check("dup: inconsistent when the twin is chosen",
      not permutation_consistent(o0, "B", o1, "ABCDE"[o1["permutation"].index(0)]))
check("original_option_index", original_option_index(o1, "ABCDE"[o1["permutation"].index(3)]) == 3)
check("dup flagged", bdf.has_duplicate_options(dup) and bde.has_duplicate_options(dup))
check("case/space-insensitive duplicate", bdf.has_duplicate_options(row(1, ["Same ", "same", "a", "b", "c"], "A")))
check("distinct not flagged", not bdf.has_duplicate_options(row(2, ["a", "b", "c", "d", "e"], "A")))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bde.COLS)
        w.writeheader()
        w.writerows(rows)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # --- finalize on a synthetic 350-row review ---
    rows = []
    for d in range(350):
        r = row(d, [f"opt{l}{d}" for l in "ABCDE"], "ABCDE"[d % 5],
                drop="yes" if d % 4 == 0 else "", reason="two options defensible" if d % 4 == 0 else "")
        rows.append(r)
    write_csv(td / "review.csv", rows)
    s = bdf.finalize(td / "review.csv", td, hash_path=None)
    expected_last = [d for d in range(350) if d % 4 != 0][149]
    check("D finalize summary", s["drops"] == 88 and s["survivors"] == 262 and s["rows"] == 300, s)
    check("D 150th survivor reported", s["last_kept_draw_index"] == expected_last and
          s["rows_after_last_kept"] == 349 - expected_last, s)
    D = json.loads((td / "items_D_csqa.json").read_text(encoding="utf-8"))["items"]
    drops = json.loads((td / "csqa_drops.json").read_text(encoding="utf-8"))
    check("D drop record", len(drops["drops"]) == 88 and all(x["reason"] for x in drops["drops"]))
    check("D first 150 survivors only", {it["draw_index"] for it in D} == {d for d in range(350) if d % 4 != 0 and d <= expected_last})
    for it in D:
        base = it["draw_index"]
        check(f"D {it['id']} key by index", it["options"][it["key_index"]] == f"opt{'ABCDE'[base % 5]}{base}"
              and it["key"] == "ABCDE"[it["key_index"]] and it["permutation"][it["key_index"]] == base % 5)
        check(f"D {it['id']} same option set", sorted(it["options"]) == sorted(f"opt{l}{base}" for l in "ABCDE"))
        check(f"D {it['id']} permutation stored", sorted(it["permutation"]) == [0, 1, 2, 3, 4])
        check(f"D {it['id']} prompt order", all(f"{l}. {t}" in it["messages"][0]["content"] for l, t in zip("ABCDE", it["options"])))
        if it["order"] == 0:
            check(f"D {it['id']} original order", it["permutation"] == [0, 1, 2, 3, 4])
        else:
            check(f"D {it['id']} shuffled", it["permutation"] != [0, 1, 2, 3, 4])
    bdf.finalize(td / "review.csv", td, hash_path=None)
    check("D deterministic", D == json.loads((td / "items_D_csqa.json").read_text(encoding="utf-8"))["items"])

    def must_exit(name, mutate):
        rs = [dict(r) for r in rows]
        mutate(rs)
        write_csv(td / "bad.csv", rs)
        try:
            bdf.finalize(td / "bad.csv", td, hash_path=None)
            check(name, False, "did not exit")
        except SystemExit:
            pass

    must_exit("D drop needs reason", lambda rs: rs[1].update(drop="yes", reason=""))
    must_exit("D undropped duplicate refused", lambda rs: rs[5].update(B=rs[5]["A"]))
    must_exit("D too few survivors", lambda rs: [r.update(drop="x", reason="r") for r in rs[:250]])

    # --- BOM from Excel: first column must stay "draw_index" ---
    raw_csv = (td / "review.csv").read_bytes()
    (td / "bom.csv").write_bytes(bytes([0xEF, 0xBB, 0xBF]) + raw_csv)
    check("BOM file: header intact", list(bdf.read_rows(td / "bom.csv")[0].keys())[0] == "draw_index")
    sb = bdf.finalize(td / "bom.csv", td, hash_path=None)
    check("BOM file finalizes like the plain file", sb == s, sb)
    check("BOM file provisional items", len(bdf.provisional_items(td / "bom.csv", 20)) > 0)

    # --- unchanged-column check by hash per draw_index ---
    hp = td / "hashes.json"
    hp.write_text(json.dumps({"rows": {str(r["draw_index"]): bdf.row_hash(r) for r in bdf.read_rows(td / "review.csv")}}), encoding="utf-8")
    bdf.finalize(td / "review.csv", td, hash_path=hp)
    check("hash: filling drop/reason is allowed", True)
    check("hash ignores drop/reason", bdf.row_hash({**rows[3], "drop": "x", "reason": "y"}) == bdf.row_hash(rows[3]))
    for what, mutate in [("question", lambda r: r.update(question=r["question"] + " ")),
                         ("option", lambda r: r.update(C="edited")), ("key", lambda r: r.update(key="E" if r["key"] != "E" else "A")),
                         ("id", lambda r: r.update(id="other"))]:
        rs = [dict(r) for r in rows]
        mutate(rs[137])
        write_csv(td / "edited.csv", rs)
        try:
            bdf.finalize(td / "edited.csv", td, hash_path=hp)
            check(f"hash: edited {what} detected", False, "did not exit")
        except SystemExit as e:
            check(f"hash: edited {what} detected, names draw_index", "draw_index 137" in str(e), str(e))
    # the real review CSV must match the real hash file (no edits yet)
    real = bdf.read_rows(ROOT / "items" / "csqa_review.csv")
    bdf.check_unchanged(real, bdf.HASH_FILE)
    check("hash: real csqa_review.csv matches items/csqa_review_hashes.json", True)

    # --- extend on a synthetic dataset (no network) ---
    ds = []
    for i in range(400):
        texts = [f"t{i}{l}" for l in "ABCDE"]
        if i in (250, 300):
            texts[3] = texts[0].upper()          # duplicate after lower()
        ds.append({"id": f"ds{i}", "question": f"Q{i}?", "choices": {"label": list("ABCDE"), "text": texts},
                   "answerKey": "ABCDE"[i % 5]})
    old = [{"draw_index": n, "id": f"ds{n * 2}", "question": f"Q{n * 2}?", **{l: f"t{n * 2}{l}" for l in "ABCDE"},
            "key": "ABCDE"[(n * 2) % 5], "drop": "", "reason": ""} for n in range(200)]
    write_csv(td / "old.csv", old)
    raw = (td / "old.csv").read_bytes()
    out, marked, allrows, pool, idx = bde.extend(td / "old.csv", dataset_rows=ds)
    check("extend rows", len(allrows) == 350 and [int(r["draw_index"]) for r in allrows] == list(range(350)))
    check("extend new ids unused", not ({r["id"] for r in allrows[200:]} & {r["id"] for r in allrows[:200]}))
    check("extend keeps old rows byte for byte", out.encode("utf-8").startswith(raw))
    check("extend draws from the pool only", len(pool) == 200 and all(ds[i]["id"] not in {r["id"] for r in old} for i in idx))
    check("extend marks mechanical duplicates", all(allrows[m]["drop"] == "x" and allrows[m]["reason"] == bde.MECH_REASON for m in marked)
          and len(marked) == sum(bde.has_duplicate_options(r) for r in allrows))
    check("extend fills nothing else", all(not r["drop"] and not r["reason"] for r in allrows if int(r["draw_index"]) not in marked))
    out2, *_ = bde.extend(td / "old.csv", dataset_rows=ds)
    check("extend deterministic", out == out2)
    (td / "extended.csv").write_text(out, encoding="utf-8", newline="")
    try:
        bde.extend(td / "extended.csv", dataset_rows=ds)
        check("extend refuses a second run", False)
    except AssertionError:
        pass

    # ---- run order with the synthetic D ----
    items = {"A": sorted(A, key=lambda r: r["id"]), "B": sorted(B, key=lambda r: r["id"]), "C": sorted(C, key=lambda r: r["id"]),
             "D": sorted(D, key=lambda r: r["id"]), "E": sorted(E, key=lambda r: r["id"])}
    order = full_run.build_order(items)
    check("order generations", sum(len(b["ids"]) for b in order) == 614, sum(len(b["ids"]) for b in order))
    check("order components in sequence", [b["component"] for b in order] == sorted((b["component"] for b in order)))
    check("order A seeds", [b["seed"] for b in order if b["component"] == "A"] == [0] * 6 + [1] * 6 + [2] * 6)
    check("order other seed 0", all(b["seed"] == 0 for b in order if b["component"] != "A"))
    check("order caps", all(b["max_new_tokens"] == (16000 if b["component"] == "E" else 4000) for b in order))
    check("order batch size", all(1 <= len(b["ids"]) <= 8 for b in order))
    check("order same ids each A pass",
          [b["ids"] for b in order if b["component"] == "A" and b["seed"] == 0]
          == [b["ids"] for b in order if b["component"] == "A" and b["seed"] == 2])
    check("order sorted within component", all(b["ids"] == sorted(b["ids"]) for b in order))
    check("batch_index sequential", [b["batch_index"] for b in order] == list(range(len(order))))

# ---- manifest guard (full_run.verify_manifest and the abort in main) ----
import hashlib  # noqa: E402

with tempfile.TemporaryDirectory() as _td:
    _root = Path(_td)
    (_root / "a.txt").write_bytes(b"alpha\n")
    (_root / "b.txt").write_bytes(b"beta\n")
    _man = _root / "MANIFEST.sha256"
    _man.write_text("".join(f"{hashlib.sha256((_root / n).read_bytes()).hexdigest()}  {n}\n" for n in ("a.txt", "b.txt")),
                    encoding="utf-8", newline="\n")
    check("manifest ok", full_run.verify_manifest(_man, _root) == [])
    (_root / "b.txt").write_bytes(b"beta changed\n")
    _bad = full_run.verify_manifest(_man, _root)
    check("manifest names the changed file", len(_bad) == 1 and _bad[0].startswith("b.txt"), str(_bad))
    (_root / "b.txt").unlink()
    _bad = full_run.verify_manifest(_man, _root)
    check("manifest names the missing file", len(_bad) == 1 and "b.txt" in _bad[0] and "missing" in _bad[0], str(_bad))
    check("manifest missing", len(full_run.verify_manifest(_root / "nope.sha256", _root)) == 1)

    # main() must abort on a mismatch before the version gate / any torch import
    (_root / "b.txt").write_bytes(b"beta changed\n")
    _saved = (full_run.MANIFEST, sys.argv[:])
    full_run.MANIFEST = _man
    sys.argv = ["full_run.py", "--checkpoint", "rlvr", "--limit", "1", "--out-dir", str(_root / "out")]
    # verify_manifest's default root is the repo, so point it at the temp root
    _orig_verify = full_run.verify_manifest
    full_run.verify_manifest = lambda m, root=_root: _orig_verify(m, root)
    try:
        full_run.main()
        check("main aborts on manifest mismatch", False, "main() returned")
    except SystemExit as e:
        check("main aborts on manifest mismatch", "MANIFEST MISMATCH" in str(e.code) and "b.txt" in str(e.code), str(e.code))
    finally:
        full_run.MANIFEST, sys.argv = _saved
        full_run.verify_manifest = _orig_verify
    check("main wrote nothing before the guard", not (_root / "out").exists())
    check("torch not imported by the guard", "torch" not in sys.modules)

check("frozen manifest matches the files", full_run.verify_manifest() == [] if full_run.MANIFEST.is_file() else True, str(full_run.verify_manifest()))

if fails:
    print("FAILED")
    print("\n".join(fails))
    raise SystemExit(1)
print("all item tests passed")
