"""CPU tests for exp.harness on a tiny random Olmo3 and an offline tokenizer (skipped if torch/transformers are absent)."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from exp import harness as H  # noqa: E402
from tests.sprint.conftest import A_MESSAGES  # noqa: E402


def test_prompt_ids_match_rendered_text(tiny_tok):
    ids = H.prompt_ids(tiny_tok, A_MESSAGES)
    txt = H.prompt_text(tiny_tok, A_MESSAGES)
    assert txt.endswith("<|im_start|>assistant\n<think>")
    assert ids == H.text_ids(tiny_tok, txt)
    assert ids[-1] == tiny_tok.convert_tokens_to_ids("<think>")


def test_as_id_list_variants():
    assert H._as_id_list([1, 2]) == [1, 2]
    assert H._as_id_list([[1, 2]]) == [1, 2]
    assert H._as_id_list({"input_ids": [[3]]}) == [3]
    assert H._as_id_list(torch.tensor([[4, 5]])) == [4, 5]


def test_digits_are_single_tokens_without_space_merge(tiny_tok):
    for d in "12345":
        assert len(H.text_ids(tiny_tok, d)) == 1
    # a leading space is a separate token, as in the Olmo 3 (cl100k-style) tokenizer
    assert H.text_ids(tiny_tok, "ANSWER: 3")[-1] == H.single_token_id(tiny_tok, "3")


def test_first_token_ids(tiny_tok):
    g = H.first_token_ids(tiny_tok, ["Wait", "Hmm"])
    assert set(g) == {"Wait", "Hmm"} and all(len(v) >= 1 for v in g.values())


def _manual(model, ids, gen_start):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    return lp[gen_start - 1:len(ids) - 1]


def test_score_matches_full_forward(tiny_tok, tiny_model):
    ids = H.prompt_ids(tiny_tok, A_MESSAGES) + H.text_ids(tiny_tok, "Okay, so maybe 3. Wait, 4.</think>\nANSWER: 4")
    gs = len(H.prompt_ids(tiny_tok, A_MESSAGES))
    te = tiny_tok.convert_tokens_to_ids("</think>")
    out = H.score(tiny_model, ids, gs, target_groups={"think_end": [te], "both": [te, 5]}, chunk=7)
    lp = _manual(tiny_model, ids, gs)
    n = len(ids) - gs
    assert len(out["lp_chosen"]) == n == len(out["entropy"]) == len(out["p_think_end"])
    exp_chosen = lp.gather(1, torch.tensor(ids[gs:])[:, None])[:, 0]
    assert torch.allclose(torch.tensor(out["lp_chosen"]), exp_chosen, atol=1e-4)
    assert torch.allclose(torch.tensor(out["p_think_end"]), lp[:, te].exp(), atol=1e-5)
    assert torch.allclose(torch.tensor(out["p_both"]), lp[:, [te, 5]].exp().sum(-1), atol=1e-5)
    ent = -(lp.exp() * lp).sum(-1)
    assert torch.allclose(torch.tensor(out["entropy"]), ent, atol=1e-4)


def test_score_kl(tiny_tok, tiny_model, tiny_model_b):
    ids = H.prompt_ids(tiny_tok, A_MESSAGES) + H.text_ids(tiny_tok, "Hmm. But 2.")
    gs = len(H.prompt_ids(tiny_tok, A_MESSAGES))
    same = H.score(tiny_model, ids, gs, pair_model=tiny_model)
    assert max(same["kl"]) < 1e-6
    diff = H.score(tiny_model, ids, gs, pair_model=tiny_model_b, chunk=3)
    la, lb = _manual(tiny_model, ids, gs), _manual(tiny_model_b, ids, gs)
    kl = (la.exp() * (la - lb)).sum(-1)
    assert torch.allclose(torch.tensor(diff["kl"]), kl, atol=1e-4) and min(diff["kl"]) > 0


def test_likert_readout_variants(tiny_tok, tiny_model):
    ro = H.likert_readout(tiny_model, tiny_tok, A_MESSAGES, think_text="Okay, maybe 3.")
    assert set(ro) == {"space_in_context", "space_in_option"}
    for r in ro.values():
        assert math.isclose(sum(r.probs), 1.0, rel_tol=1e-9) and 0 < r.mass <= 1 + 1e-6
        assert len(r.logp) == 5 and 1 <= r.argmax() <= 5 and 1 <= r.expected() <= 5
        assert 0 <= r.p_extreme() <= 1 and 0 <= r.entropy() <= math.log(5) + 1e-9
    a = ro["space_in_context"]
    assert a.n_option_tokens == [1] * 5 and all(a.prefix_consistent)
    assert H.primary_variant(ro) == "space_in_context"


def test_likert_readout_tracks_logits(tiny_tok, tiny_model):
    """Boost the logit of '3' in a copy of the model: the readout must put most mass on 3."""
    import copy
    m = copy.deepcopy(tiny_model)
    t3 = H.single_token_id(tiny_tok, "3")
    ctx = H.prompt_text(tiny_tok, A_MESSAGES) + "x" + H.ANSWER_SUFFIX
    h = H._final_hidden(m, H.text_ids(tiny_tok, ctx))[-1]
    with torch.no_grad():
        m.lm_head.weight[t3] += 20 * h / h.norm() ** 2       # raises the logit of '3' by 20 at this position
    r = H.likert_readout(m, tiny_tok, A_MESSAGES, think_text="x")["space_in_context"]
    assert r.argmax() == 3 and r.probs[2] > 0.9


def test_nothink_context_has_closed_think(tiny_tok, tiny_model, monkeypatch):
    seen = {}
    orig = H._readout

    def spy(model, tok, ctx, variant):
        seen[variant] = ctx
        return orig(model, tok, ctx, variant)
    monkeypatch.setattr(H, "_readout", spy)
    H.likert_readout(tiny_model, tiny_tok, A_MESSAGES, think_text=None)
    ctx = seen["space_in_context"]
    assert ctx.endswith("<think></think>\n\nANSWER: ") and ctx.count("<think>") == 1
    H.likert_readout(tiny_model, tiny_tok, A_MESSAGES, think_text="reasoning\n")
    assert seen["space_in_context"].endswith("<think>reasoning\n</think>\n\nANSWER: ")


def test_activation_cache(tiny_tok, tiny_model):
    ids = torch.tensor([H.prompt_ids(tiny_tok, A_MESSAGES)])
    with torch.no_grad():
        ref = tiny_model(ids, output_hidden_states=True).hidden_states
    n_hooks = len(tiny_model.model.layers[0]._forward_hooks)       # transformers installs its own capture hooks
    with H.ActivationCache(tiny_model, layers=[0, 1], positions=[-1, 0]) as cache:
        with torch.no_grad():
            tiny_model(ids)
    assert cache.acts[0].shape == (2, 32)
    # decoder layer 0 output == hidden_states[1]
    assert torch.allclose(torch.tensor(cache.acts[0][0]).float(), ref[1][0, -1], atol=1e-2)
    with H.ActivationCache(tiny_model, layers=[1]) as c2:
        with torch.no_grad():
            tiny_model(ids)
    assert c2.acts[1].shape == (ids.shape[1], 32)
    assert len(tiny_model.model.layers[0]._forward_hooks) == n_hooks   # our hooks removed


def test_resolve_sampling_override(monkeypatch):
    monkeypatch.setattr(H, "generation_defaults", lambda r, s: {"temperature": 0.6, "top_p": 0.95, "top_k": None,
                                                                 "do_sample": True, "eos_token_id": 1,
                                                                 "source": "generation_config.json"})
    s = H.resolve_sampling("r", "s")
    assert (s["temperature"], s["top_p"], s["top_k"]) == (0.6, 0.95, -1) and s["fallback_used"]["top_k"]
    s2 = H.resolve_sampling("r", "s", {"temperature": 0.7, "top_p": None})
    assert s2["temperature"] == 0.7 and s2["top_p"] == 0.95 and s2["override"] == {"temperature": 0.7}


def test_readout_with_gpt2_style_tokenizer(tiny_tok_gpt2):
    """When a leading space merges with the digit, 'space_in_context' is prefix-inconsistent and the harness must
    pick 'space_in_option' (options ' 1'..' 5' as single tokens) as the primary variant."""
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(0)
    m = Olmo3ForCausalLM(Olmo3Config(vocab_size=len(tiny_tok_gpt2), hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                     max_position_embeddings=8192)).eval()
    assert len(H.text_ids(tiny_tok_gpt2, " 3")) == 1
    ro = H.likert_readout(m, tiny_tok_gpt2, A_MESSAGES, think_text="maybe")
    assert not all(ro["space_in_context"].prefix_consistent)
    assert ro["space_in_option"].n_option_tokens == [1] * 5 and all(ro["space_in_option"].prefix_consistent)
    assert H.primary_variant(ro) == "space_in_option"


def test_close_probs_matches_brute_force(tiny_model):
    """close_probs (cached, cropped) equals P(v | prefix) * P(second | prefix + v) from independent full forwards."""
    import torch
    from exp import harness as H
    g = torch.Generator().manual_seed(3)
    V = tiny_model.config.vocab_size
    ids = torch.randint(1, V, (40,), generator=g).tolist()
    first, second = [V // 5, V // 3], V // 2
    pos = [12, 25, 39, 31]
    pc, pf = H.close_probs(tiny_model, ids, pos, first, second, min_p=0.0)

    def nxt(prefix):
        with torch.no_grad():
            return torch.softmax(tiny_model(input_ids=torch.tensor([prefix])).logits[0, -1].float(), -1)
    for k, got, gotf in zip(pos, pc, pf):
        p = nxt(ids[:k])
        want = sum(float(p[v]) * float(nxt(ids[:k] + [v])[second]) for v in first)
        assert abs(got - want) < 1e-5 and abs(gotf - float(p[first].sum())) < 1e-5
    pc0, _ = H.close_probs(tiny_model, ids, pos, first, None)
    assert all(abs(a - b) < 1e-6 for a, b in zip(pc0, pf))


def test_close_probs_beyond_sliding_window(tiny_tok):
    """Positions far beyond a (tiny) sliding window: the cropped cache must still match uncached full forwards."""
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM
    from exp import harness as H
    torch.manual_seed(5)
    cfg = Olmo3Config(vocab_size=len(tiny_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512, sliding_window=8,
                      layer_types=["sliding_attention", "full_attention"])
    m = Olmo3ForCausalLM(cfg).eval()
    V = cfg.vocab_size
    ids = torch.randint(1, V, (60,), generator=torch.Generator().manual_seed(6)).tolist()
    first, second = [V // 5, V // 3], V // 2
    pos = [50, 9, 33, 59, 20]

    def nxt(prefix):
        with torch.no_grad():
            return torch.softmax(m(input_ids=torch.tensor([prefix]), use_cache=False).logits[0, -1].float(), -1)
    pc, pf = H.close_probs(m, ids, pos, first, second, min_p=0.0)
    for k, got, gotf in zip(pos, pc, pf):
        p = nxt(ids[:k])
        want = sum(float(p[v]) * float(nxt(ids[:k] + [v])[second]) for v in first)
        assert abs(got - want) < 1e-5 and abs(gotf - float(p[first].sum())) < 1e-5
