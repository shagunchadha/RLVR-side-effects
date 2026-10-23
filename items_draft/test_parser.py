#!/usr/bin/env python3
"""Adversarial tests for answer_parser.py. Run: python test_answer_parser.py  (no dependencies)."""
from answer_parser import parse_generation as P

CAP = 4000
fails = []


def check(name, got, **expected):
    for k, v in expected.items():
        if got[k] != v:
            fails.append(f"{name}: {k} = {got[k]!r}, expected {v!r}")


# --- the original failure: quoted ANSWER inside an unclosed think block ---
t = "<think>I'd lean towards disagreeing, so ANSWER: 4 might fit, but let me reconsider whether"
check("quoted answer, cap hit", P(t, "likert", 4000, CAP, False), status="truncated", answer=None)
check("quoted answer, early EOS", P(t, "likert", 53, CAP, True), status="early_stop", answer=None)
check("quoted answer, no info", P(t, "likert"), status="unclosed", answer=None)

# --- answer only inside the reasoning, block closed, nothing after ---
t = "<think>so ANSWER: 4 maybe</think>\nI think this is fine."
check("answer inside think only", P(t, "likert", 80, CAP, True), status="no_answer", answer=None)
check("same, but cap hit", P(t, "likert", 4000, CAP, False), status="truncated")

# --- a normal valid generation ---
t = "<think>reasoning ... ANSWER: 2 was my first thought</think>\nANSWER: 4"
check("valid likert", P(t, "likert", 300, CAP, True), status="valid", answer=4)

# --- markdown decoration ---
check("bold", P("<think>x</think>**ANSWER: 3**", "likert"), status="valid", answer=3)
check("backticks", P("<think>x</think>`ANSWER: 5`", "likert"), status="valid", answer=5)

# --- range and format errors ---
check("likert 6", P("<think>x</think>ANSWER: 6", "likert"), status="malformed", answer=None)
check("likert 0", P("<think>x</think>ANSWER: 0", "likert"), status="malformed")
check("likert 3.5", P("<think>x</think>ANSWER: 3.5", "likert"), status="malformed")
check("likert word", P("<think>x</think>ANSWER: four", "likert"), status="no_answer")

# --- several answers after </think>: the last one counts ---
check("two answers", P("<think>x</think>ANSWER: 2\nOn reflection, ANSWER: 3", "likert"), status="valid", answer=3)

# --- multiple choice ---
check("mc5 valid", P("<think>x</think>ANSWER: C", "mc5"), status="valid", answer="C")
check("mc5 lowercase", P("<think>x</think>ANSWER: c", "mc5"), status="valid", answer="C")
check("mc5 parens", P("<think>x</think>ANSWER: (B)", "mc5"), status="valid", answer="B")
check("mc5 out of range", P("<think>x</think>ANSWER: F", "mc5"), status="malformed")
check("mc4 E invalid", P("<think>x</think>ANSWER: E", "mc4"), status="malformed")
check("mc word not letter", P("<think>x</think>ANSWER: The answer", "mc5"), status="no_answer")

# --- calibration: two fields, both required, both after </think> ---
t = "<think>CONFIDENCE: 99 in draft</think>ANSWER: B | CONFIDENCE: 70"
check("calib valid", P(t, "calibration"), status="valid", answer="B", confidence=70.0)
check("calib newline", P("<think>x</think>ANSWER: A\nCONFIDENCE: 85%", "calibration"),
      status="valid", answer="A", confidence=85.0)
check("calib no confidence", P("<think>x</think>ANSWER: B", "calibration"), status="no_answer")
check("calib confidence 120", P("<think>x</think>ANSWER: B | CONFIDENCE: 120", "calibration"), status="malformed")
check("calib confidence in draft only",
      P("<think>CONFIDENCE: 90</think>ANSWER: B", "calibration"), status="no_answer")
check("calib cap hit mid answer", P("<think>x</think>ANSWER: B | ", "calibration", 4000, CAP, False),
      status="truncated")

# --- free-form answers (sycophancy, MATH) ---
check("free number", P("<think>x</think>ANSWER: 408", "free"), status="valid", answer="408")
check("free period", P("<think>x</think>ANSWER: Canberra.", "free"), status="valid", answer="Canberra")
check("free boxed", P("<think>x</think>ANSWER: \\boxed{12}", "free"), status="valid", answer="\\boxed{12}")
check("free empty", P("<think>x</think>ANSWER:   \n", "free"), status="no_answer")

# --- hit_cap logic ---
check("exactly at cap with EOS", P("<think>x</think>ANSWER: 3", "likert", 4000, CAP, True), status="valid")
check("at cap, valid answer present", P("<think>x</think>ANSWER: 3", "likert", 4000, CAP, False), status="valid")

# --- a think-close tag quoted before the real one: first one wins, content between is ignored ---
t = "<think>Typing </think> in my reasoning is odd. ANSWER: 2 </think>ANSWER: 5"
check("quoted close tag", P(t, "likert"), status="valid", answer=5)
# ^ the LAST ANSWER after the first </think> is used; document this behavior if it matters.

if fails:
    print("FAILED")
    print("\n".join(fails))
    raise SystemExit(1)
print("all parser tests passed")
