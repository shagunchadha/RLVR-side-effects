"""CPU tests for scripts/full_run_matched_prompt.py (the SFT matched-system-prompt rerun): the template swap check."""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import full_run_matched_prompt as MP  # noqa: E402

FX = Path(__file__).parent / "fixtures"


def _toks(tiny_tok):
    sft, dpo = copy.deepcopy(tiny_tok), copy.deepcopy(tiny_tok)
    sft.chat_template = (FX / "chat_template_think_sft.jinja").read_text(encoding="utf-8")
    dpo.chat_template = (FX / "chat_template_think_dpo.jinja").read_text(encoding="utf-8")
    return sft, dpo


def test_full_run_is_untouched():
    assert not MP.FR.verify_manifest(), "scripts/full_run.py and the frozen files must match items/MANIFEST.sha256"


def test_swap_reproduces_dpo_render(tiny_tok):
    sft, dpo = _toks(tiny_tok)
    items = MP.FR.load_items(False, None, list(MP.FR.ORDER))
    report, problems = MP.verify_swap(sft, dpo, items)
    assert problems == []
    assert report["n_items_verified"] == sum(len(v) for v in items.values())
    assert "December 2024" in report["own_default_system_block"]
    assert "November 2024" in report["matched_default_system_block"]
    assert all(v["matched"] > v["own"] for v in report["prompt_tokens"].values())


def test_swap_check_catches_a_body_difference(tiny_tok):
    sft, dpo = _toks(tiny_tok)
    dpo.chat_template = dpo.chat_template.replace("<|im_start|>assistant\n<think>", "<|im_start|>assistant\n")
    items = {"A": MP.FR.load_items(False, None, ["A"])["A"][:2]}
    _, problems = MP.verify_swap(sft, dpo, items)
    assert any("after the system block" in p for p in problems)
