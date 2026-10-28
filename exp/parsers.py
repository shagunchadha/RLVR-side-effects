"""Text parsers shared by the sprint jobs (pure Python, no model dependency).

- split_think: (think text, post-think text) at the FIRST "</think>" (the frozen parser reads after the LAST one;
  for valid rows there is exactly one)
- sentences: the sentence splitter of results/analysis/exploratory_dithering.py (copied, incl. the 'No.' fix)
- strict_answer: the frozen parser (items/answer_parser.py), imported read-only
- lenient_likert / lenient_letter: for the B3 'bare' condition, where no ANSWER line is requested
"""
import re
import sys

from . import config as C

THINK_END = "</think>"


def split_think(text):
    if THINK_END not in text:
        return text, "", False
    a, b = text.split(THINK_END, 1)
    return a, b, True


# ------------------------------------------------------------------ sentences (copied from exploratory_dithering.py)
_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|approx|cf)\.", re.I)


def sentences(text):
    t = text.replace("**", "").replace("`", "")
    t = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u0001"), t)
    t = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", t)
    t = re.sub(r"(?m)^\s*(\d+|[a-zA-Z])[.)](?=\s)", lambda m: m.group(0).replace(".", "\u0001").replace(")", "\u0002"), t)
    segs = []
    for para in re.split(r"\n\s*\n", t):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])[\"'”’)\]]*\s+", para)
        segs += [p.strip() for p in parts if re.search(r"[A-Za-z0-9]", p)]
    return [s.replace("\u0001", ".").replace("\u0002", ")") for s in segs]


def sentence_end_char_offsets(text):
    """Character offsets just after each sentence terminator or blank-line break (for mapping to token positions)."""
    ends = [m.end() for m in re.finditer(r"[.!?][\"'”’)\]]*(?=\s|$)", text)]
    ends += [m.start() for m in re.finditer(r"\n\s*\n", text)]
    return sorted(set(ends))


# ------------------------------------------------------------------ answers
def _frozen_parser():
    if str(C.ITEMS_DIR) not in sys.path:
        sys.path.insert(0, str(C.ITEMS_DIR))
    import answer_parser  # noqa: E402  (frozen file, read-only import)
    return answer_parser


def strict_answer(text, kind, n_tokens=None, max_new_tokens=None, hit_eos=None):
    return _frozen_parser().parse_generation(text, kind, n_tokens, max_new_tokens, hit_eos)


_SCALE = re.compile(r"\b[1-5]\s*(?:to|-|–)\s*[1-5]\b|scale of\s*[1-5]|out of\s*5|\(\s*[1-5]\s*=")


def lenient_likert(text):
    """Rating 1-5 from free text after the think block: an 'ANSWER: d' line wins; otherwise the LAST standalone
    digit 1-5 that is not part of a scale reference ('1 to 5', 'scale of 1', 'out of 5', '(1 = ...').
    Returns (value or None, rule)."""
    _, post, closed = split_think(text)
    if not closed:
        return None, "unclosed"
    m = list(re.finditer(r"ANSWER\s*:\s*\**\s*([1-5])\b", post))
    if m:
        return int(m[-1].group(1)), "answer_line"
    clean = _SCALE.sub(" ", post)
    d = list(re.finditer(r"(?<![\d.])\b([1-5])\b(?![.\d]\d)", clean))
    if d:
        return int(d[-1].group(1)), "last_digit"
    return None, "none"


def lenient_letter(text, letters="ABCDE"):
    """Option letter from free text after the think block: 'ANSWER: X' wins; otherwise the last standalone capital
    letter in `letters` written as an option ('B', 'B)', '(B)', 'B.', '**B**', 'option B', 'answer is B')."""
    _, post, closed = split_think(text)
    if not closed:
        return None, "unclosed"
    m = list(re.finditer(rf"ANSWER\s*:\s*\**\s*\(?([{letters}])\b", post))
    if m:
        return m[-1].group(1), "answer_line"
    # option-shaped capital letters only (case-sensitive letters, so the article 'a' is never read as option A)
    pat = (rf"(?i:option|answer is|answer:|choice)\s*\**\(?([{letters}])\)?\b|\(([{letters}])\)"
           rf"|\*\*([{letters}])\*\*|^\s*([{letters}])[.)]?\s*$")
    hits = [next(g for g in mm.groups() if g) for mm in re.finditer(pat, post, re.M)]
    hits = [h for h in hits if h in letters]
    if hits:
        return hits[-1], "option_pattern"
    return None, "none"


_YESNO = re.compile(r"ANSWER\s*:\s*\**\s*(yes|no)\b", re.I)


def strict_yesno(text, n_tokens=None, max_new_tokens=None, hit_eos=None):
    """B2 self-determinate cell: the frozen parser's status rules (items/answer_parser.py) with a yes/no field.
    valid = think block closed and the last 'ANSWER:' after it is yes or no; answer is 'yes' / 'no'."""
    out = _frozen_parser().parse_generation(text, "likert", n_tokens, max_new_tokens, hit_eos)
    if not out["closed"]:
        return {**out, "answer": None}
    after = text.rsplit("</think>", 1)[1]
    m = list(_YESNO.finditer(after))
    if m:
        return {**out, "status": "valid", "answer": m[-1].group(1).lower(), "reason": None}
    status = "truncated" if out["hit_cap"] else ("malformed" if "ANSWER" in after else "no_answer")
    return {**out, "status": status, "answer": None, "reason": "no yes/no ANSWER after </think>"}
