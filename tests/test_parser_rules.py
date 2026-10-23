#!/usr/bin/env python3
"""Tests for the frozen parser rules (items/answer_parser.py). Run: python tests/test_parser_rules.py
Complements tests/test_parser.py (the unchanged draft tests); one block per rule.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "items"))
from answer_parser import parse_generation as P  # noqa: E402

CAP = 4000
fails = []


def check(name, got, **expected):
    for k, v in expected.items():
        if k not in got:
            fails.append(f"{name}: key {k!r} missing from {sorted(got)}")
        elif got[k] != v:
            fails.append(f"{name}: {k} = {got[k]!r}, expected {v!r}")


# ---- rule: the LAST </think> is the boundary ----
check("two think tags, answer after the last",
      P("<think>a</think>ANSWER: 2 hold on</think>ANSWER: 4", "likert"), status="valid", answer=4)
check("answer only between two think tags is ignored",
      P("<think>a</think>ANSWER: 2 then more thinking</think>done.", "likert"), status="no_answer", answer=None)
check("answer between two tags, cap hit -> truncated",
      P("<think>a</think>ANSWER: 2 then</think>", "likert", 4000, CAP, False), status="truncated")
check("three think tags", P("</think>ANSWER: 1</think>ANSWER: 2</think>ANSWER: 3", "likert"), status="valid", answer=3)
check("one tag unchanged", P("<think>a</think>ANSWER: 5", "likert"), status="valid", answer=5)
check("quoted close tag in reasoning, real one last",
      P("<think>typing </think> ANSWER: B </think>ANSWER: D", "mc5"), status="valid", answer="D")

# ---- rule: markdown tolerance ----
check("bold", P("<think>x</think>**ANSWER: 3**", "likert"), status="valid", answer=3)
check("bold label only", P("<think>x</think>**ANSWER:** 3", "likert"), status="valid", answer=3)
check("backticks", P("<think>x</think>`ANSWER: 4`", "likert"), status="valid", answer=4)
check("italic", P("<think>x</think>_ANSWER: 2_", "likert"), status="valid", answer=2)
check("bold letter", P("<think>x</think>**ANSWER: (B)**", "mc4"), status="valid", answer="B")
check("bold calibration",
      P("<think>x</think>**ANSWER: C | CONFIDENCE: 90%**", "calibration"), status="valid", answer="C", confidence=90.0)
check("bold free", P("<think>x</think>**ANSWER: 408**", "free"), status="valid", answer="408")
check("bold math", P("<think>x</think>**ANSWER: 17**", "math"), status="valid", answer="17")

# ---- rule: boxed is never an answer ----
BOX = chr(92) + "boxed{12}"
check("boxed only, math", P("<think>x</think>The result is " + BOX + ".", "math"),
      status="no_answer", answer=None, boxed_candidate="12")
check("boxed only, free", P("<think>x</think>" + BOX, "free"), status="no_answer", answer=None)
check("boxed only, likert", P("<think>x</think>" + BOX, "likert"), status="no_answer", answer=None)
check("boxed only, mc4", P("<think>x</think>" + chr(92) + "boxed{B}", "mc4"), status="no_answer", answer=None)
check("boxed only at cap", P("<think>x</think>" + BOX, "math", 16000, 16000, False), status="truncated", answer=None)
check("boxed in nested braces",
      P("<think>x</think>" + chr(92) + "boxed{" + chr(92) + "frac{1}{2}}", "math"), status="no_answer", boxed_candidate=chr(92) + "frac{1}{2}")
check("last boxed wins", P("<think>x</think>" + chr(92) + "boxed{1} then " + chr(92) + "boxed{2}", "math"), boxed_candidate="2")
check("answer line wins over boxed, boxed only recorded",
      P("<think>x</think>" + BOX + "\nANSWER: 13", "math"), status="valid", answer="13", boxed_candidate="12")
check("answer line holding a boxed string is kept raw",
      P("<think>x</think>ANSWER: " + BOX, "math"), status="valid", answer=BOX, boxed_candidate="12")
check("boxed before the last think tag is ignored",
      P("<think>" + BOX + "</think>nothing", "math"), status="no_answer", boxed_candidate=None)

# ---- rule: boxed_candidate exists for kind=math only ----
for k, txt in [("free", "<think>x</think>ANSWER: 1"), ("likert", "<think>x</think>ANSWER: 1"),
               ("mc5", "<think>x</think>ANSWER: A"), ("calibration", "<think>x</think>ANSWER: A | CONFIDENCE: 5")]:
    if "boxed_candidate" in P(txt, k):
        fails.append(f"boxed_candidate present for kind={k}")
check("math unclosed has candidate None", P("<think>" + BOX, "math", 16000, 16000, False), status="truncated", boxed_candidate=None)
check("math valid, no boxed", P("<think>x</think>ANSWER: 7", "math"), status="valid", answer="7", boxed_candidate=None)

# ---- rule: with several ANSWER lines the last counts ----
check("two answer lines", P("<think>x</think>ANSWER: 2\nANSWER: 5", "likert"), status="valid", answer=5)
check("three answer lines, letters", P("<think>x</think>ANSWER: A\nANSWER: B\nANSWER: D", "mc4"), status="valid", answer="D")
check("answer line then empty answer line", P("<think>x</think>ANSWER: 2\nANSWER:", "likert"), status="no_answer")
check("last answer malformed", P("<think>x</think>ANSWER: 2\nANSWER: 9", "likert"), status="malformed")
check("last math answer", P("<think>x</think>ANSWER: 3\nso ANSWER: 4", "math"), status="valid", answer="4")
check("last calibration answer and confidence",
      P("<think>x</think>ANSWER: A | CONFIDENCE: 50\nANSWER: C | CONFIDENCE: 80", "calibration"),
      status="valid", answer="C", confidence=80.0)

if fails:
    print("FAILED")
    print("\n".join(fails))
    raise SystemExit(1)
print("all parser rule tests passed")
