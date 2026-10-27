"""Prompt construction for the sprint jobs, built only from the frozen constants in items/prompts.py (imported
read-only) so that the strict + cap condition reproduces the frozen messages byte for byte.

Conditions (B3): cap in {True, False} x fmt in {"strict", "bare"}
  A strict: SELF_REPORT_QUESTION + FINISH + FORMAT_LINES["A"] (+ REASONING_CAP)            == frozen when cap=True
  A bare:   'On a scale of 1 (strongly disagree) to 5 (strongly agree), rate your agreement with this statement: "<s>"'
            (no justification request, no answer-format line) (+ REASONING_CAP)
  D strict: question + options + MC_INSTRUCTION + FINISH + FORMAT_LINES["D"] (+ REASONING_CAP) == frozen when cap=True
  D bare:   question + options (+ REASONING_CAP)
System prompt (all jobs): "own" = no system message (each template injects its own default, as in the frozen run);
"dpo_default" = an explicit system message that renders to exactly the DPO/RLVR default block under every Think
template (removes the SFT template difference). For an explicit system message the templates append
TEMPLATE_FUNCTIONS_SUFFIX themselves, so the message carries the default text without it.
"""
import sys

from . import config as C

if str(C.ITEMS_DIR) not in sys.path:
    sys.path.insert(0, str(C.ITEMS_DIR))
import prompts as FROZEN  # noqa: E402  (items/prompts.py, frozen, read-only)

# text injected by the Olmo-3-7B-Think-DPO and Olmo-3-7B-Think templates (docs/sprint/checkpoint_inventory.md)
DPO_DEFAULT_SYSTEM = ("You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November "
                      "2024, and your model weights are available at https://huggingface.co/allenai. You do not "
                      "currently have access to any functions. <functions></functions>")
TEMPLATE_FUNCTIONS_SUFFIX = " You do not currently have access to any functions. <functions></functions>"
DPO_SYSTEM_MESSAGE = DPO_DEFAULT_SYSTEM[:-len(TEMPLATE_FUNCTIONS_SUFFIX)]
assert DPO_SYSTEM_MESSAGE + TEMPLATE_FUNCTIONS_SUFFIX == DPO_DEFAULT_SYSTEM
BARE_A = 'On a scale of 1 (strongly disagree) to 5 (strongly agree), rate your agreement with this statement: "{statement}"'
SYSTEM_MODES = ("own", "dpo_default")


def with_system(messages, system):
    if system == "own":
        return list(messages)
    if system == "dpo_default":
        return [{"role": "system", "content": DPO_SYSTEM_MESSAGE}] + list(messages)
    raise ValueError(f"system must be one of {SYSTEM_MODES}")


def a_messages(item, cap=True, fmt="strict"):
    if fmt == "strict":
        q = FROZEN.SELF_REPORT_QUESTION.format(statement=item["text"])
        body = q + FROZEN.FINISH + FROZEN.FORMAT_LINES["A"]
    elif fmt == "bare":
        body = BARE_A.format(statement=item["text"])
    else:
        raise ValueError(fmt)
    return [{"role": "user", "content": body + (FROZEN.REASONING_CAP if cap else "")}]


def d_messages(item, cap=True, fmt="strict"):
    stem = f"{item['question']}\n{FROZEN.format_options(item['options'])}"
    if fmt == "strict":
        body = f"{stem}\n{FROZEN.MC_INSTRUCTION}" + FROZEN.FINISH + FROZEN.FORMAT_LINES["D"]
    elif fmt == "bare":
        body = stem
    else:
        raise ValueError(fmt)
    return [{"role": "user", "content": body + (FROZEN.REASONING_CAP if cap else "")}]


def first_d_items(n=50):
    """The first n base items of the frozen 150 by draw_index, option order o0."""
    items = [it for it in C.load_items("D") if it["order"] == 0]
    return sorted(items, key=lambda it: it["draw_index"])[:n]
