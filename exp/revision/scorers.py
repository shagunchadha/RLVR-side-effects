"""F1 scorers (registration amendment 18). CPU only; imports the packages of requirements_revision.txt.

All scorers take the model's full generated text and first strip the think block (the developers strip thinking
traces before scoring, Olmo 3 report A.8 / Table 16): the scored response is the text after the first "</think>".
A generation whose think block never closed has no response; it scores as incorrect / not followed, and is
counted as such (amendment 18: a generation still capped at the last stage is unresolved, never dropped).

Maths (MATH-500, AIME):
  primary   math_verify_correct: math-verify 0.9.0 on the LAST \\boxed{...} of the response (lm_eval's
            last_boxed_only_string), against the dataset's answer field. No \\boxed{} -> incorrect.
  cross-check minerva_flex_correct: OLMES "EM Flex" exactly as the developers' harness computes it
            (oe_eval/tasks/oe_eval_tasks/minerva_math.py GenericMATH.extract_answers / process_results for MATH-500
            with cot_style "plain"; aime.py AIME.extract_answers / process_results for AIME), using lm_eval 0.4.3's
            minerva_math and hendrycks_math utilities (vendored). Gold as OLMES builds it: MATH-500 =
            normalize_final_answer(remove_boxed(last_boxed_only_string(solution))); AIME = answer.lstrip("0").
Instruction following:
  ifeval_scores / ifbench_scores: the official checkers (Google's instruction_following_eval; allenai/IFBench),
  strict and loose, per prompt (all instructions followed) and per instruction.
"""
import os
import re
import signal

from exp.vendor.lm_eval_math import hendrycks_utils as HU
from exp.vendor.lm_eval_math import minerva_utils as MU

_POSIX_ALARM = hasattr(signal, "SIGALRM")
if not _POSIX_ALARM:
    # lm_eval's is_equiv guards sympy with a 5 s SIGALRM timeout, which does not exist on Windows. Scoring for the
    # record runs on Linux; on Windows (CPU tests only) the guard becomes a no-op so the same code path runs.
    class _NoTimeout:
        def __init__(self, seconds=1, error_message="Timeout"):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    MU.timeout = _NoTimeout


def strip_think(text):
    """(response after the first '</think>', closed?) - an unclosed think block has no response."""
    if "</think>" not in text:
        return "", False
    return text.split("</think>", 1)[1], True


# ------------------------------------------------------------------ maths: primary (math-verify)
def last_boxed(response):
    return MU.last_boxed_only_string(response)


def math_verify_correct(text, gold):
    from math_verify import LatexExtractionConfig, parse, verify
    resp, _ = strip_think(text)
    boxed = last_boxed(resp)
    if boxed is None:
        return False
    timeout = {} if _POSIX_ALARM else {"parsing_timeout": None}
    pred = parse(boxed, extraction_config=[LatexExtractionConfig()], **timeout)
    ref = parse(f"${gold}$", extraction_config=[LatexExtractionConfig()], **timeout)
    if not pred or not ref:
        return False
    return bool(verify(ref, pred, **({} if _POSIX_ALARM else {"timeout_seconds": None})))


# ------------------------------------------------------------------ maths: cross-check (OLMES EM Flex)
def olmes_math_gold(solution):
    """OLMES GenericMATH._process_doc: answer = normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))."""
    return MU.normalize_final_answer(MU.remove_boxed(MU.last_boxed_only_string(solution)))


def olmes_aime_gold(answer):
    """OLMES AIME._process_doc: answer = doc['answer'].lstrip('0')."""
    return str(answer).lstrip("0")


def _extract_answers(raw_answer, with_minerva_line):
    """OLMES extract_answers (minerva_math.py with use_cot=True; aime.py). MATH-500 under olmo3:adapt uses
    cot_style 'plain', so the 'Final Answer: The final answer is' line is NOT tried for MATH; AIME always tries it."""
    boxed_answer = MU.last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = MU.remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    all_answers = []
    if with_minerva_line:
        minerva_answer = MU.normalize_final_answer(MU.get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
    if boxed_answer is not None:
        all_answers.append(MU.normalize_final_answer(boxed_answer))
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            answer = MU.normalize_final_answer(raw_answer[dollars[-2] + 1: dollars[-1]])
            all_answers.append(answer)
    if len(all_answers) == 0:
        all_answers.append(MU.normalize_final_answer(raw_answer))
    return all_answers


def minerva_flex_correct(text, olmes_gold, kind):
    """OLMES exact_match_flex for one generation: lm_eval's minerva process_results (exact_match), then any of the
    extracted answers equivalent under minerva is_equiv or hendrycks is_equiv. kind: 'math' or 'aime'."""
    resp, closed = strip_think(text)
    if not closed:
        return False
    doc = {"answer": olmes_gold}
    em = MU.process_results(doc, [resp])["exact_match"]
    if em == 1:
        return True
    for answer in _extract_answers(resp, with_minerva_line=(kind == "aime")):
        if MU.is_equiv(answer, olmes_gold) or HU.is_equiv(answer, olmes_gold):
            return True
    return False


# ------------------------------------------------------------------ instruction following
IF_SEED = 0


def _seed_checkers():
    """The official IFEval/IFBench instruction classes draw Python `random` values for any argument the item does not
    give (and langdetect is randomised), so the same response can pass on one run and fail on the next (measured on
    the official example outputs; see docs/eacl_revision_1/known_issues.md). Seeding both before every check makes
    F1's scores reproducible without changing any checker code."""
    import random
    import langdetect
    random.seed(IF_SEED)
    langdetect.DetectorFactory.seed = IF_SEED


def _if_scores(lib, item, text):
    resp, closed = strip_think(text)
    inp = lib.InputExample(key=item["key"], instruction_id_list=list(item["instruction_id_list"]),
                           prompt=item["prompt"], kwargs=[dict(k) for k in item["kwargs"]])
    p2r = {item["prompt"]: resp}
    out = {}
    for level, fn in (("strict", lib.test_instruction_following_strict), ("loose", lib.test_instruction_following_loose)):
        if not resp.strip():                      # unclosed think block / empty response: nothing is followed
            out[f"prompt_{level}"] = False
            out[f"inst_{level}"] = [False] * len(inp.instruction_id_list)
            continue
        _seed_checkers()
        o = fn(inp, p2r)
        out[f"prompt_{level}"] = bool(o.follow_all_instructions)
        out[f"inst_{level}"] = [bool(x) for x in o.follow_instruction_list]
    out["closed"] = closed
    return out


def ifeval_scores(item, text):
    from exp.vendor.ifeval_google import evaluation_lib as lib
    return _if_scores(lib, item, text)


def ifbench_scores(item, text):
    from exp.vendor.ifbench import evaluation_lib as lib
    return _if_scores(lib, item, text)


def ensure_nltk_data(path=None):
    """The IF checkers need nltk 'punkt' (+ 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng' for IFBench).
    Downloaded once into $NLTK_DATA (default /tmp/nltk_data); IFBench's own util downloads them the same way."""
    import nltk
    path = path or os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(path, exist_ok=True)
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)
    for pkg, loc in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                     ("stopwords", "corpora/stopwords"), ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")):
        try:
            nltk.data.find(loc)
        except LookupError:
            nltk.download(pkg, download_dir=path, quiet=True)
