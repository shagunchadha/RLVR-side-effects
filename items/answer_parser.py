r"""Answer parser for the full run (spec section 1, validity rule).

A generation is valid only if the think block closed AND the required field(s) appear after the
LAST </think> (the rule the pilots used). Invalid rows are reported, never re-sampled, never filled.

Statuses
  valid      closed think block, every required field present and well-formed
  truncated  no valid answer and the generation reached the token cap (hit_cap)
  early_stop no closed think block, ended on an EOS token before the cap
  no_answer  closed think block, a required field is missing (and the cap was not hit)
  malformed  closed think block, a required field is present but out of range / wrong format
  unclosed   no closed think block and the cause is unknown (token info not supplied);
             only arises when n_tokens / max_new_tokens / hit_eos are not passed (tests, ad-hoc use)

Kinds: likert (1-5), mc4 (A-D), mc5 (A-E), calibration (ANSWER letter A-D and CONFIDENCE 0-100),
free (raw answer string; sycophancy), math (like free, plus a recorded `boxed_candidate`; graded elsewhere).

Rules
  - Only text after the LAST </think> is searched (text between two </think> tags is ignored).
  - The LAST "ANSWER:" after it counts (and the last "CONFIDENCE:").
  - A line with the required label but a value that parses and is out of range is malformed;
    a missing label or a non-numeric / non-letter value is no_answer.
  - Markdown decoration (*, `, quotes, brackets) around the value is ignored (e.g. **ANSWER: 3**).
  - \boxed{...} is never accepted as an answer: output with a boxed expression but no ANSWER line is no_answer.
    (An ANSWER line whose value is written as \boxed{12} is still an ANSWER line; the raw string is stored and the
    grader strips the wrapper.)
  - kind=math only: `boxed_candidate` = content of the last \boxed{...} after the last </think>, or None. It is
    recorded for analysis and never affects status or answer (never used in the primary score).
  - hit_cap = (n_tokens >= max_new_tokens) and hit_eos is not True.
"""
import re

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_DECOR = "*`'\"()[]{}<>|,;:._ \t"
_LETTERS = {"mc4": "ABCD", "mc5": "ABCDE", "calibration": "ABCD"}


def _segments(after, label):
    """Text following each occurrence of `label:` up to the next occurrence of the label or newline."""
    parts = re.split(label + r"\s*:", after)[1:]
    return [p.split("\n", 1)[0] for p in parts]


_BOXED = chr(92) + "boxed{"


def _last_boxed(text):
    """Content of the last boxed{...} (with a backslash) in `text` (balanced braces), or None."""
    i = text.rfind(_BOXED)
    if i < 0:
        return None
    j = i + len(_BOXED)
    depth, k = 1, j
    while k < len(text) and depth:
        depth += (text[k] == "{") - (text[k] == "}")
        k += 1
    return text[j:k - 1].strip() if depth == 0 else None


def _first_token(seg):
    toks = seg.lstrip(_DECOR).split()    # leading markup such as the closing ** of **ANSWER:** 3
    return toks[0].strip(_DECOR) if toks else ""


def _parse_likert(seg):
    tok = _first_token(seg)
    if not _NUM.match(tok):
        return "no_answer", None, "non-numeric"
    if re.match(r"^[1-5]$", tok):
        return "valid", int(tok), None
    return "malformed", None, f"likert value {tok!r} not an integer 1-5"


def _parse_letter(seg, letters):
    tok = _first_token(seg)
    if len(tok) == 1 and tok.isalpha():
        if tok.upper() in letters:
            return "valid", tok.upper(), None
        return "malformed", None, f"letter {tok!r} outside {letters[0]}-{letters[-1]}"
    return "no_answer", None, "not a single letter"


def _parse_confidence(after):
    segs = _segments(after, "CONFIDENCE")
    if not segs:
        return "no_answer", None, "no CONFIDENCE field"
    seg = segs[-1].split("|", 1)[0]
    tok = _first_token(seg.replace("%", " "))
    if not _NUM.match(tok):
        return "no_answer", None, "non-numeric confidence"
    v = float(tok)
    if 0 <= v <= 100:
        return "valid", v, ("confidence in (0,1): may be a fraction" if 0 < v < 1 else None)
    return "malformed", None, f"confidence {tok} outside 0-100"


def parse_generation(text, kind, n_tokens=None, max_new_tokens=None, hit_eos=None):
    out = {"status": None, "answer": None, "confidence": None, "reason": None, "closed": False, "hit_cap": False}
    if kind == "math":
        out["boxed_candidate"] = None
    known = n_tokens is not None and max_new_tokens is not None
    out["hit_cap"] = bool(known and n_tokens >= max_new_tokens and hit_eos is not True)
    out["closed"] = "</think>" in text

    if not out["closed"]:
        if out["hit_cap"]:
            out["status"], out["reason"] = "truncated", "token cap reached inside the think block"
        elif hit_eos is True or (known and n_tokens < max_new_tokens):
            out["status"], out["reason"] = "early_stop", "ended on EOS before the cap, think block not closed"
        else:
            out["status"], out["reason"] = "unclosed", "think block not closed; cause unknown (no token info)"
        return out

    after = text.rsplit("</think>", 1)[1]
    if kind == "math":
        out["boxed_candidate"] = _last_boxed(after)
    ans_segs = _segments(after, "ANSWER")

    def finish(status, reason):
        # a missing field at the cap is truncation, a malformed one stays malformed
        if status == "no_answer" and out["hit_cap"]:
            status, reason = "truncated", "token cap reached before the required field(s): " + str(reason)
        out["status"], out["reason"] = status, reason
        return out

    if not ans_segs:
        return finish("no_answer", "no ANSWER after </think>")
    seg = ans_segs[-1]

    if kind == "likert":
        st, val, why = _parse_likert(seg)
    elif kind in ("mc4", "mc5"):
        st, val, why = _parse_letter(seg, _LETTERS[kind])
    elif kind == "calibration":
        st, val, why = _parse_letter(seg, _LETTERS[kind])
        if st != "no_answer":
            out["answer"] = val
        cst, cval, cwhy = _parse_confidence(after)
        if st == "valid" and cst == "valid":
            out["answer"], out["confidence"] = val, cval
            out["status"], out["reason"] = "valid", cwhy
            return out
        if "malformed" in (st, cst):
            out["answer"] = val if st == "valid" else None
            return finish("malformed", why if st == "malformed" else cwhy)
        out["answer"] = val if st == "valid" else None
        return finish("no_answer", why if st == "no_answer" else cwhy)
    elif kind in ("free", "math"):
        val = seg.strip().strip("*`").strip().rstrip(".").strip()
        if not val:
            return finish("no_answer", "empty ANSWER")
        out["answer"], out["status"] = val, "valid"
        return out
    else:
        raise ValueError(f"unknown kind {kind!r}")

    if st == "valid":
        out["answer"], out["status"] = val, "valid"
        return out
    return finish(st, why)
