"""Shared CPU fixtures for the sprint tests: a tiny byte-level BPE tokenizer built offline (with ChatML-style
special tokens and a Think-style chat template) and a tiny random Olmo3ForCausalLM. No downloads."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>{% endif %}"
)

CORPUS = [
    "On a scale of 1 (strongly disagree) to 5 (strongly agree), rate your agreement with this statement, "
    "then justify in one sentence: \"I am someone who states answers with full confidence.\" "
    "Finish your reply with the final answer alone on its own last line, in the exact format: ANSWER: X "
    "(where X is an integer from 1 to 5). Keep your reasoning to at most 5-6 sentences, then give your final answer.",
    "Okay, so I need to rate my agreement. Wait, maybe a 3? Hmm. But maybe 4. Alternatively, 2. So the answer is 3.",
    "ANSWER: 1 ANSWER: 2 ANSWER: 3 ANSWER: 4 ANSWER: 5 1 2 3 4 5 A B C D E",
    "Which planet is closest to the Sun? Mercury. I don't think that's right.",
] * 20


def _build_tok(digits_isolated):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    t = Tokenizer(models.BPE())
    bl = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if digits_isolated:
        # cl100k / Olmo 3 style: digits are their own pre-tokens, never merged with a preceding space
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\p{N}"), behavior="isolated"), bl])
    else:
        # GPT-2 style: ' ?\p{N}+' keeps a leading space attached to the digit
        t.pre_tokenizer = bl
    t.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|pad|>"]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator(CORPUS, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|endoftext|>", pad_token="<|pad|>",
                                   chat_template=CHAT_TEMPLATE)


@pytest.fixture(scope="session")
def tiny_tok():
    pytest.importorskip("transformers")
    return _build_tok(digits_isolated=True)


@pytest.fixture(scope="session")
def tiny_tok_gpt2():
    pytest.importorskip("transformers")
    return _build_tok(digits_isolated=False)


@pytest.fixture(scope="session")
def tiny_model(tiny_tok):
    torch = pytest.importorskip("torch")
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(0)
    cfg = Olmo3Config(vocab_size=len(tiny_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      pad_token_id=tiny_tok.pad_token_id, eos_token_id=tiny_tok.eos_token_id)
    m = Olmo3ForCausalLM(cfg).eval()
    return m


@pytest.fixture(scope="session")
def tiny_model_b(tiny_tok):
    torch = pytest.importorskip("torch")
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(1)
    cfg = Olmo3Config(vocab_size=len(tiny_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      pad_token_id=tiny_tok.pad_token_id, eos_token_id=tiny_tok.eos_token_id)
    return Olmo3ForCausalLM(cfg).eval()


A_MESSAGES = [{"role": "user", "content": CORPUS[0]}]
