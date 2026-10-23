"""Integer grading for component E (spec section 2E).

Exact integer match after stripping \\boxed{}, $, thousands separators and a trailing .0.
The raw answer string is always stored by the caller; nothing here changes it.
"""
import re

_INT = re.compile(r"^-?\d+$")


def normalise_integer(raw):
    """Return the canonical integer string, or None if `raw` is not an integer after stripping."""
    if raw is None:
        return None
    s = raw.strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace(",", "").replace(" ", "").replace("\\!", "")
    s = re.sub(r"\.0+$", "", s)
    if not _INT.match(s):
        return None
    return str(int(s))


def grade_integer(raw_answer, reference):
    """True iff both normalise to the same integer. Not-an-integer answers are incorrect."""
    a, r = normalise_integer(raw_answer), normalise_integer(reference)
    return a is not None and r is not None and a == r


# ---- component D: permutation consistency on ORIGINAL option indices ----
def original_option_index(item, letter):
    """Original (CSV order) index of the option a model chose, given the letter it answered in this item's order.
    item["permutation"][j] is the original index of the option shown in position j."""
    return item["permutation"]["ABCDE".index(letter)]


def permutation_consistent(item_o0, letter_o0, item_o1, letter_o1):
    """True iff the two answers (one per option order of the same question) pick the same original option index."""
    return original_option_index(item_o0, letter_o0) == original_option_index(item_o1, letter_o1)
