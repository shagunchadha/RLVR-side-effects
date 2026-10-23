#!/usr/bin/env python3
"""Regression: items/answer_parser.py against the pilot's own validity and extracted answers.

Runs the new parser over rlvr_baseline_v3, rlvr_newenv_pilot, dpo_pilot_v3 and sft_pilot_v3 (36 rows each)
and compares row by row with the pilot's `forced` flag and `extracted_answer`.
  pilot valid = (not forced) and extracted_answer is not None
Rows are parsed on `raw_output` with any appended "[... forced completion ...]" tail removed (that tail is text
the pilot added after generation, not model output). hit_eos is derived as n_generated_tokens < max_new_tokens.
Run: python tests/parser_regression.py
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "items"))
from answer_parser import parse_generation  # noqa: E402

FILES = ["rlvr_baseline_v3", "rlvr_newenv_pilot", "dpo_pilot_v3", "sft_pilot_v3"]
KIND = {"self_report": "likert", "reasoning": "mc5", "behavioral": "free"}
TAIL = re.compile(r"\n\n\[[^\]]*forced completion[^\]]*\]\nANSWER:.*\Z", re.S)


def lead_int(s):
    m = re.match(r"-?\d{1,4}", s or "")
    return m.group(0) if m else None


def main():
    total = collections.Counter()
    disagreements = []
    forced_map = collections.Counter()
    for name in FILES:
        rows = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))["results"]
        c = collections.Counter()
        for r in rows:
            raw = TAIL.sub("", r["raw_output"])
            n, mx = r["n_generated_tokens"], r["max_new_tokens"]
            p = parse_generation(raw, KIND[r["probe_type"]], n, mx, n < mx)
            pilot_valid = (not r["forced"]) and r["extracted_answer"] is not None
            new_valid = p["status"] == "valid"
            new_ans = p["answer"]
            if r["probe_type"] == "behavioral" and new_ans is not None:
                new_cmp = lead_int(str(new_ans))      # the pilot kept only a leading integer for free answers
            else:
                new_cmp = None if new_ans is None else str(new_ans)
            c["rows"] += 1
            c["status_agree"] += pilot_valid == new_valid
            if pilot_valid and new_valid:
                c["both_valid"] += 1
                c["answer_agree"] += new_cmp == r["extracted_answer"]
                if new_cmp != r["extracted_answer"]:
                    disagreements.append((name, r["id"], r["sample"], "answer", r["extracted_answer"], new_ans))
            if pilot_valid != new_valid:
                disagreements.append((name, r["id"], r["sample"], "validity",
                                      f"pilot valid={pilot_valid} forced={r['forced']}", p["status"] + ": " + str(p["reason"])))
            forced_map[(name, "forced" if r["forced"] else "not forced", p["status"])] += 1
        total.update(c)
        print(f"{name:20s} rows {c['rows']}  validity agrees {c['status_agree']}/{c['rows']}  "
              f"both valid {c['both_valid']}, answers agree {c['answer_agree']}/{c['both_valid']}")
    print(f"\nTOTAL rows {total['rows']}  validity agrees {total['status_agree']}/{total['rows']}  "
          f"answers agree {total['answer_agree']}/{total['both_valid']}")
    print("\npilot 'forced' category -> new status (file, pilot, new: count)")
    for k, v in sorted(forced_map.items()):
        print("  ", k, v)
    print(f"\nDISAGREEMENTS: {len(disagreements)}")
    for d in disagreements:
        print("  ", d)


if __name__ == "__main__":
    main()
