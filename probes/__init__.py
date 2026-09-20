"""Locked probe set for the RLVR side-effects study.

Self-contained: no dependency on the notebook, a marimo session, or the
`datasets` library. Import the same objects for every checkpoint so prompts
are identical across runs.

    from probes import ALL_PROBES, build_chat
"""
import copy

from .traits.self_report import SELF_REPORT_PROBES
from .behavioral.behavioral_probes import BEHAVIORAL_PROBES
from .reasoning.commonsense_qa import COMMONSENSE_QA_PROBES

TRAIT_PROBES = SELF_REPORT_PROBES + BEHAVIORAL_PROBES
REASONING_PROBES = COMMONSENSE_QA_PROBES
ALL_PROBES = TRAIT_PROBES + REASONING_PROBES

# Appended to the last user message of every probe in the v3 protocol.
REASONING_CAP = " Keep your reasoning to at most 5-6 sentences, then give your final answer."


def build_chat(probe, cap=REASONING_CAP):
    """Return the chat message list for a probe, with the reasoning cap applied."""
    if "messages" in probe:
        chat = copy.deepcopy(probe["messages"])
    else:
        chat = [{"role": "user", "content": probe["prompt"]}]
    chat[-1]["content"] += cap
    return chat
