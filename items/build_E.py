#!/usr/bin/env python3
"""Component E: MATH capability control from HuggingFaceH4/MATH-500 (held out), 50 integer-answer problems,
10 per level, plus a separate 10-problem length-check set (spec section 2E).

Needs `datasets` and network. No model output is used.
Source: HuggingFaceH4/MATH-500 `test` split (500 problems; columns problem, solution, answer, subject, level, unique_id).
Eligibility, applied before any draw:
  - the reference `answer`, after stripping $, thousands separators (,) and a trailing .0, is an integer (^-?\\d+$);
  - the problem text does not contain [asy] (figure code).
Draw (fixed seed 20261021, one generator, in this order):
  1. for level 1..5: 10 problems from the eligible problems of that level (sorted by unique_id);
  2. the 50 are shuffled and numbered E_00..E_49;
  3. length-check set: 2 further problems per level from the eligible problems NOT in the 50, numbered EL_00..EL_09
     (own file, never part of the final set, used only to check lengths before the freeze).
If any level has fewer than 10 eligible problems (12 for the length-check set) the script stops and prints the counts.
Contamination swap (applied after the draw above, which does not use the flags): items/contamination_flags.json lists
the MATH-500 problems flagged at >= 50% 8-gram overlap in any Olmo 3 stage (RL, DPO, SFT; scripts/contamination_scan.py).
A drawn item whose problem is flagged is replaced under the same id (E_nn) by one problem drawn with
random.Random(SEED + nn) from the eligible, unflagged problems of the same level that are in neither the 50 nor the
length-check set (sorted by unique_id); the other items are left unchanged. A flagged length-check problem stops the
script. Without a flags file nothing is swapped (and the output says so).
Prompt: zero-shot, prompts.build_messages_E (cap sentence, ANSWER: <integer>). parser_kind is "math".
Writes items/items_E_math.json and items/items_E_lengthcheck.json.
"""
import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import build_messages_E  # noqa: E402
from grading import normalise_integer  # noqa: E402

SEED = 20261021
PER_LEVEL = 10
CHECK_PER_LEVEL = 2
INT = re.compile(r"^-?\d+$")


def eligible(row):
    return "[asy]" not in row["problem"] and normalise_integer(row["answer"]) is not None


def load_flagged(path=None):
    """unique_ids flagged in any stage, or None when there is no flags file."""
    path = Path(path) if path else HERE / "contamination_flags.json"
    if not path.exists():
        return None
    return set(json.loads(path.read_text(encoding="utf-8"))["flagged_unique_ids"])


def build(rows=None, flagged=None):
    """flagged: set of unique_ids to swap out (see module docstring), or None. The last return value is the swap log."""
    if rows is None:
        from datasets import load_dataset
        rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    levels = sorted({int(r["level"]) for r in rows})
    assert levels == [1, 2, 3, 4, 5], levels
    stats = {}
    for lv in levels:
        at = [r for r in rows if int(r["level"]) == lv]
        stats[lv] = {"total": len(at), "integer_answer": sum(normalise_integer(r["answer"]) is not None for r in at),
                     "integer_and_no_asy": sum(eligible(r) for r in at),
                     "asy_among_integer": sum("[asy]" in r["problem"] and normalise_integer(r["answer"]) is not None for r in at)}
    short = {lv: s["integer_and_no_asy"] for lv, s in stats.items() if s["integer_and_no_asy"] < PER_LEVEL + CHECK_PER_LEVEL}
    if short:
        return None, None, stats, short, []
    rng = random.Random(SEED)
    chosen = []
    for lv in levels:
        pool = sorted((r for r in rows if int(r["level"]) == lv and eligible(r)), key=lambda r: r["unique_id"])
        chosen += rng.sample(pool, PER_LEVEL)
    rng.shuffle(chosen)
    taken = {r["unique_id"] for r in chosen}
    check = []
    for lv in levels:
        pool = sorted((r for r in rows if int(r["level"]) == lv and eligible(r) and r["unique_id"] not in taken),
                      key=lambda r: r["unique_id"])
        check += rng.sample(pool, CHECK_PER_LEVEL)

    def item(prefix, n, r):
        return {"id": f"{prefix}_{n:02d}", "component": "E", "kind": "math", "parser_kind": "math",
                "level": int(r["level"]), "subject": r["subject"], "unique_id": r["unique_id"],
                "problem": r["problem"], "reference_raw": r["answer"],
                "ground_truth": normalise_integer(r["answer"]), "messages": build_messages_E(r["problem"])}

    swaps = []
    if flagged:
        bad_check = sorted(r["unique_id"] for r in check if r["unique_id"] in flagged)
        if bad_check:
            sys.exit(f"STOP: length-check problems are flagged: {bad_check}")
        used = {r["unique_id"] for r in chosen} | {r["unique_id"] for r in check}
        for n, r in enumerate(chosen):
            if r["unique_id"] not in flagged:
                continue
            pool = sorted((x for x in rows if int(x["level"]) == int(r["level"]) and eligible(x)
                           and x["unique_id"] not in flagged and x["unique_id"] not in used), key=lambda x: x["unique_id"])
            new = random.Random(SEED + n).choice(pool)
            swaps.append({"id": f"E_{n:02d}", "level": int(r["level"]), "old_unique_id": r["unique_id"],
                          "new_unique_id": new["unique_id"], "pool_size": len(pool)})
            chosen[n] = new
            used.add(new["unique_id"])
    items = [item("E", n, r) for n, r in enumerate(chosen)]
    check_items = [item("EL", n, r) for n, r in enumerate(check)]
    return items, check_items, stats, None, swaps


if __name__ == "__main__":
    flagged = load_flagged()
    items, check_items, stats, short, swaps = build(flagged=flagged)
    print("MATH-500 per level (total / integer answer / integer and no [asy] / [asy] among integer-answer):")
    for lv, s in stats.items():
        print(f"  level {lv}: {s['total']:3d} / {s['integer_answer']:3d} / {s['integer_and_no_asy']:3d} / {s['asy_among_integer']:3d}")
    if short:
        sys.exit(f"STOP: levels with too few eligible problems (need {PER_LEVEL + CHECK_PER_LEVEL}): {short}")
    assert len(items) == 50 and len(check_items) == 10
    assert not ({i['unique_id'] for i in items} & {i['unique_id'] for i in check_items})
    print("selected by level:", dict(sorted(collections.Counter(i["level"] for i in items).items())))
    print("length-check by level:", dict(sorted(collections.Counter(i["level"] for i in check_items).items())))
    print("contamination flags:", "none (no flags file, nothing swapped)" if flagged is None else
          f"{len(flagged)} flagged problems; swaps: {swaps if swaps else 'none needed'}")
    assert not ({i["unique_id"] for i in items + check_items} & (flagged or set())), "flagged problem still selected"
    print("selected with [asy]:", sum("[asy]" in i["problem"] for i in items + check_items))
    meta = {"component": "E", "seed": SEED, "per_level": PER_LEVEL, "source": "HuggingFaceH4/MATH-500 test",
            "answer_format": "ANSWER: <integer>", "parser_kind": "math", "max_new_tokens": 16000,
            "grading": "items/grading.py exact integer match", "eligibility_stats": stats,
            "contamination_swaps": swaps,
            "contamination_flags_file": "items/contamination_flags.json" if flagged is not None else None}
    (HERE / "items_E_math.json").write_text(json.dumps({"meta": {**meta, "n_items": 50}, "items": items},
                                                       indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    (HERE / "items_E_lengthcheck.json").write_text(json.dumps(
        {"meta": {**meta, "n_items": 10, "note": "length-check set: 2 per level, not in the 50; never part of the final set"},
         "items": check_items}, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print("wrote items_E_math.json and items_E_lengthcheck.json")
