"""F1 scorers on known examples (amendment 18). Needs requirements_revision.txt; skipped without it.

Ground truth is never the code under test:
  maths  - hand-checked answers, including correct \\boxed{} answers formatted differently from the gold;
  IFEval - Google's own unit tests (instructions_test.py, instructions_util_test.py), and the GPT-4 responses
           Google ships, scored against the accuracies published in the IFEval paper (Zhou et al. 2023, Table 3);
  IFBench - allenai/IFBench's sample outputs, which must reproduce its committed eval_results_{strict,loose}.jsonl
           record for record.
"""
import gzip
import json
import unittest
from pathlib import Path

import pytest

pytest.importorskip("math_verify")
pytest.importorskip("nltk")
pytest.importorskip("langdetect")

from exp.revision import scorers as SC  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "revision"


def _jsonl_gz(name):
    return [json.loads(l) for l in gzip.decompress((FIX / name).read_bytes()).decode("utf-8").splitlines() if l.strip()]


T = "<think>Some reasoning.</think>\n"


# ------------------------------------------------------------------ maths
@pytest.mark.parametrize("response, gold, aime", [
    (r"So the answer is $\boxed{\dfrac{1}{2}}$.", r"\frac{1}{2}", False),           # \dfrac vs \frac
    (r"The answer is $\boxed{0.5}$.", r"\frac{1}{2}", False),                       # decimal vs fraction
    (r"$\boxed{\frac12}$", r"\frac{1}{2}", False),                                  # brace-less \frac
    (r"\boxed{x=5}", "5", False),                                                   # equation form
    (r"The answer is $\boxed{73}$.", "073", True),                                  # AIME gold with a leading zero
    (r"First I thought $\boxed{3}$, but finally $\boxed{\frac{1}{2}}$.", r"\frac{1}{2}", False),  # last \boxed counts
])
def test_math_correct_but_formatted_differently(response, gold, aime):
    assert SC.math_verify_correct(T + response, gold)
    olmes_gold = SC.olmes_aime_gold(gold) if aime else SC.olmes_math_gold(r"... so $\boxed{" + gold + "}$.")
    assert SC.minerva_flex_correct(T + response, olmes_gold, "aime" if aime else "math")


@pytest.mark.parametrize("response, gold", [
    (r"The answer is $\boxed{\frac{1}{3}}$.", r"\frac{1}{2}"),
    (r"Finally $\boxed{\frac{1}{2}}$, no wait: $\boxed{3}$.", r"\frac{1}{2}"),        # last \boxed is wrong
    (r"The answer is one half.", r"\frac{1}{2}"),                                     # no \boxed (primary)
])
def test_math_wrong(response, gold):
    assert not SC.math_verify_correct(T + response, gold)


def test_minerva_flex_left_right_quirk():
    """lm_eval 0.4.3's normalize_final_answer deletes the unit "ft" everywhere, so "\\left(" becomes "\\le(". A correct
    tuple written without \\left ... \\right is then INCORRECT under the developers' EM Flex (OLMES uses this lm_eval),
    while math-verify scores it correct. Kept as is: the flex cross-check must reproduce the developers' scorer."""
    gold = r"\left( 3, \frac{\pi}{2} \right)"                                       # MATH-500 #0
    og = SC.olmes_math_gold(r"So $\boxed{" + gold + "}$.")
    assert og == r"\le(3,\frac{\pi}{2}\right)"
    plain = T + r"Thus $\boxed{(3, \frac{\pi}{2})}$."
    leftright = T + r"Thus $\boxed{\left( 3, \frac{\pi}{2} \right)}$."
    assert SC.math_verify_correct(plain, gold) and SC.math_verify_correct(leftright, gold)
    assert not SC.minerva_flex_correct(plain, og, "math")
    assert SC.minerva_flex_correct(leftright, og, "math")


def test_minerva_flex_leading_zero_quirk():
    """An AIME answer written with a leading zero (\\boxed{073}) normalises to "073", which sympy refuses to parse
    ("leading zeros in decimal integer literals"), so the developers' EM Flex scores it INCORRECT against the gold
    "73" (OLMES strips the gold's leading zeros only). math-verify scores it correct. Kept as is (see above)."""
    resp = T + r"The answer is $\boxed{073}$."
    assert SC.math_verify_correct(resp, "073")
    assert not SC.minerva_flex_correct(resp, SC.olmes_aime_gold("073"), "aime")


def test_math_unclosed_think_is_incorrect():
    text = r"<think>... it must be $\boxed{\frac{1}{2}}$ ... let me check again"         # capped inside the think
    assert not SC.math_verify_correct(text, r"\frac{1}{2}")
    assert not SC.minerva_flex_correct(text, r"\frac{1}{2}", "math")


def test_math_think_block_is_ignored():
    text = r"<think>Maybe $\boxed{\frac{1}{2}}$?</think> The answer is $\boxed{3}$."
    assert not SC.math_verify_correct(text, r"\frac{1}{2}") and SC.math_verify_correct(text, "3")


def test_olmes_gold_extraction():
    assert SC.olmes_math_gold(r"We get $r = 3.$ So $\boxed{\frac{9}{2}}$ units.") == r"\frac{9}{2}"
    assert SC.olmes_math_gold(r"So $\boxed{12\text{ inches}}$.") == "12"          # Minerva drops units
    assert SC.olmes_aime_gold("025") == "25" and SC.olmes_aime_gold(204) == "204"


# ------------------------------------------------------------------ IFEval
@pytest.fixture(scope="module")
def nltk_data(tmp_path_factory):
    SC.ensure_nltk_data(str(tmp_path_factory.mktemp("nltk")))


@pytest.mark.parametrize("mod", ["instructions_test", "instructions_util_test"])
def test_google_ifeval_unit_tests(mod, nltk_data):
    m = __import__(f"exp.vendor.ifeval_google.{mod}", fromlist=["x"])
    res = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(m))
    assert res.wasSuccessful(), [str(f[0]) for f in res.failures + res.errors][:5]


def test_ifeval_gpt4_responses_reproduce_published_accuracies(nltk_data):
    """IFEval paper (arXiv 2311.07911 v1), Table 3, GPT-4: prompt-level strict 76.89 (416/541), instruction-level
    strict 83.57 (697/834), prompt-level loose 79.30 (429/541), instruction-level loose 85.37 (712/834).

    The official checker draws Python `random` values for arguments an item does not give, so the published numbers
    are one random draw: unseeded, repeated runs of the same code moved prompt-level strict between 76.89 and 77.08
    (Nov 27). With the scorers' fixed seed the result is deterministic, and must lie within 2 prompts / 2
    instructions of every published value."""
    published = {"prompt_strict": 416, "inst_strict": 697, "prompt_loose": 429, "inst_loose": 712}
    # The GPT-4 responses are keyed by the 2023 prompts, i.e. the HF google/IFEval version F1 uses (GitHub's
    # input_data.jsonl later corrected key 2785's wording); match them on that version.
    items = {r["prompt"]: r for r in json.loads((Path(__file__).parents[2] / "items_revision" / "F1_ifeval.json")
                                                 .read_text(encoding="utf-8"))["items"]}
    resp = _jsonl_gz("ifeval_gpt4_responses.jsonl.gz")
    def score():
        acc = {"prompt_strict": [], "prompt_loose": [], "inst_strict": [], "inst_loose": []}
        for r in resp:
            s = SC.ifeval_scores(items[r["prompt"]], "<think></think>" + r["response"])
            for k in acc:
                acc[k] += [s[k]] if k.startswith("prompt") else s[k]
        return {k: (sum(v), len(v)) for k, v in acc.items()}
    a, b = score(), score()
    assert len(resp) == 541 and a == b, "seeded scoring must be deterministic"
    assert {k: n for k, (_, n) in a.items()} == {"prompt_strict": 541, "prompt_loose": 541, "inst_strict": 834, "inst_loose": 834}
    off = {k: a[k][0] - published[k] for k in published}
    assert all(abs(d) <= 2 for d in off.values()), off


# ------------------------------------------------------------------ IFBench
def test_ifbench_sample_outputs_reproduce_the_official_results(nltk_data):
    # The sample outputs and expected results carry the prompts of the repo's data/ copy, which lacks the trailing
    # whitespace of keys 13 and 19 in the package/HF copy F1 uses; match on the stripped prompt.
    items = {r["prompt"].strip(): r for r in _jsonl_gz("ifbench_test.jsonl.gz")}
    sample = {r["prompt"].strip(): r["response"] for r in _jsonl_gz("ifbench_sample_output.jsonl.gz")}
    for level in ("strict", "loose"):
        expected = _jsonl_gz(f"ifbench_eval_results_{level}.jsonl.gz")
        # 294 records; exactly one is malformed upstream (its "prompt" is ~7 tab-joined rows), the other 293 must match
        malformed = [e for e in expected if e["prompt"].strip() not in items]
        assert len(expected) == 294 and len(malformed) == 1 and malformed[0]["prompt"].count("\t") == 18
        mism, mism_head = [], []
        for e in expected:
            p = e["prompt"].strip()
            if p not in items:
                continue
            for fn, out in ((SC.ifbench_scores, mism), (SC.ifbench_scores_head, mism_head)):
                s = fn(items[p], "<think></think>" + sample[p])
                if s[f"prompt_{level}"] != e["follow_all_instructions"] or s[f"inst_{level}"] != e["follow_instruction_list"]:
                    out.append(items[p]["key"])
        # F1's main IFBench scorer is IFBench @ c94da020 (2025-10-17), the commit that produced these results: all 293
        # records must match. The pinned head (48891866, reported alongside) differs on exactly these 5, all in
        # checkers fixed after that date (format:list, ratio:overlap, ratio:sentence_type).
        assert not mism, f"{level}: main (c94da020) differs on {mism}"
        assert sorted(mism_head, key=int) == ["82", "85", "148", "163", "214"], f"{level}: head {mism_head}"


def test_if_unclosed_think_follows_nothing():
    item = {"key": 1, "instruction_id_list": ["punctuation:no_comma"], "prompt": "Say hi without commas.", "kwargs": [{}]}
    s = SC.ifeval_scores(item, "<think>hi there")
    assert s == {"prompt_strict": False, "inst_strict": [False], "prompt_loose": False, "inst_loose": [False], "closed": False}
