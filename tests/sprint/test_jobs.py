"""CPU tests for the prompt builders and for M2 / B1 / B3 end to end (tiny random Olmo3, fake vLLM)."""
import argparse
import json
from pathlib import Path

import pytest

from exp import config as C
from exp import prompts_sprint as PS


# ------------------------------------------------------------------ prompts (no torch needed)
def test_strict_cap_reproduces_frozen_A():
    for it in C.load_items("A"):
        assert PS.a_messages(it, cap=True, fmt="strict") == it["messages"]


def test_strict_cap_reproduces_frozen_D():
    for it in C.load_items("D"):
        assert PS.d_messages(it, cap=True, fmt="strict") == it["messages"]


def test_condition_texts():
    it = C.load_items("A")[0]
    bare_nocap = PS.a_messages(it, cap=False, fmt="bare")[0]["content"]
    assert "justify" not in bare_nocap and "ANSWER" not in bare_nocap and "5-6 sentences" not in bare_nocap
    assert it["text"] in bare_nocap
    strict_nocap = PS.a_messages(it, cap=False, fmt="strict")[0]["content"]
    assert "ANSWER: X" in strict_nocap and "5-6 sentences" not in strict_nocap
    d = PS.first_d_items(50)
    assert len(d) == 50 and all(x["order"] == 0 for x in d)
    assert [x["draw_index"] for x in d] == sorted(x["draw_index"] for x in d)
    assert "ANSWER" not in PS.d_messages(d[0], cap=False, fmt="bare")[0]["content"]


def test_with_system():
    m = [{"role": "user", "content": "q"}]
    assert PS.with_system(m, "own") == m
    s = PS.with_system(m, "dpo_default")
    assert s[0] == {"role": "system", "content": PS.DPO_SYSTEM_MESSAGE} and s[1:] == m
    with pytest.raises(ValueError):
        PS.with_system(m, "x")


def test_dpo_default_renders_the_dpo_default_block(tiny_tok):
    """--system dpo_default must render byte-identically to the DPO template's own default, under the SFT and the DPO
    templates (fixtures: the pinned Think-SFT / Think-DPO chat templates), with <functions></functions> exactly once."""
    fx = Path(__file__).parent / "fixtures"
    sft_t = (fx / "chat_template_think_sft.jinja").read_text(encoding="utf-8")
    dpo_t = (fx / "chat_template_think_dpo.jinja").read_text(encoding="utf-8")
    item = C.load_items("A")[0]

    def render(tmpl, msgs):
        return tiny_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=tmpl)
    ref = render(dpo_t, item["messages"])
    assert ref.startswith("<|im_start|>system\n" + PS.DPO_DEFAULT_SYSTEM + "<|im_end|>\n")
    for tmpl in (sft_t, dpo_t):
        r = render(tmpl, PS.with_system(item["messages"], "dpo_default"))
        assert r == ref and r.count("<functions></functions>") == 1
    assert render(sft_t, item["messages"]) != ref   # the confound itself


# ------------------------------------------------------------------ jobs on the tiny model
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from exp import harness as H  # noqa: E402
from exp import store as S  # noqa: E402
from exp.jobs import B1_likert, B3_instructions, M2_tracescore  # noqa: E402

TRACE = ("Okay, so I need to rate this. Maybe 3. Wait, 4?\n\nHmm. But maybe 2. So the answer is 4."
         "</think>\nIt fits me.\nANSWER: 4")


def test_m2_prepare_alignment(tiny_tok):
    item = C.load_items("A")[0]
    pids = H.prompt_ids(tiny_tok, item["messages"])
    gids = H.text_ids(tiny_tok, TRACE)
    row = {"raw_output": TRACE, "prompt_tokens": len(pids), "n_tokens": len(gids) + 1, "hit_eos": True}
    p = M2_tracescore.prepare(tiny_tok, item, row, tiny_tok.eos_token_id)
    assert p["prompt_tok_match"] and p["gen_tok_match"] and p["gen_ids"][-1] == tiny_tok.eos_token_id
    te = tiny_tok.convert_tokens_to_ids("</think>")
    assert p["gen_ids"][p["think_end_tok"]] == te
    # sentence ends inside the think block: '.', '?', '.', '.', '.' ; the next content token follows each
    assert len(p["sent_next_tok"]) == len(p["sent_end_tok"]) >= 5
    for k in p["sent_next_tok"]:
        piece = tiny_tok.decode([p["gen_ids"][k]])
        assert piece.strip() != ""                            # not pure whitespace
    assert p["sent_next_tok"][-1] == p["think_end_tok"]      # last sentence is followed by </think>
    row2 = dict(row, prompt_tokens=len(pids) + 1)
    assert not M2_tracescore.prepare(tiny_tok, item, row2, tiny_tok.eos_token_id)["prompt_tok_match"]


def _fake_runs(tmp_path, tiny_tok, monkeypatch):
    item = C.load_items("A")[0]
    pids = H.prompt_ids(tiny_tok, item["messages"])
    gids = H.text_ids(tiny_tok, TRACE)
    for cp in ("sft", "dpo", "rlvr"):
        rows = [{"id": item["id"], "component": "A", "seed": s, "status": "valid", "raw_output": TRACE,
                 "prompt_tokens": len(pids), "n_tokens": len(gids) + 1, "hit_eos": True} for s in range(2)]
        p = tmp_path / f"{cp}.json"
        p.write_text(json.dumps({"meta": {}, "rows": rows}))
        monkeypatch.setitem(C.FULL_RUN, cp, p)


def _args(ckpt, shard, **kw):
    return argparse.Namespace(ckpt=ckpt, shard=shard, env_report=None, job_args=argparse.Namespace(**kw))


def test_m2_single_and_pair(tmp_path, tiny_tok, tiny_model, tiny_model_b, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    _fake_runs(tmp_path, tiny_tok, monkeypatch)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    toks = {k: tiny_tok for k in ("sft", "dpo", "rlvr")}
    a = _args(["rlvr"], "0/1", traces=["sft", "dpo", "rlvr"], pair=None, components=list("ABCDE"), chunk=16, limit=None)
    assert M2_tracescore.run(a, models={"rlvr": tiny_model}, toks=toks) == 0
    st = S.Store("M2", "rlvr", M2_tracescore.job_config(a.job_args, "rlvr"))
    t = st.read(0).to_pylist()
    assert len(t) == 6 and all(r["prompt_tok_match"] and r["gen_tok_match"] for r in t)
    r0 = t[0]
    k = len(r0["sent_next_tok"])
    assert len(r0["p_close_at_sent"]) == k and all(0 <= x <= 1 for x in r0["p_close_at_sent"])
    assert k > 0 and len(r0["p_think_end_at_sent"]) == len(r0["p_switch_at_sent"]) == len(r0["p_switch_Wait_at_sent"]) == k
    assert all(0 <= x <= 1 for x in r0["p_switch_at_sent"]) and 0 <= r0["mean_p_think_end_think"] <= 1
    assert r0["n_think_pos"] == r0["think_end_tok"] and r0["sum_lp_gen"] < 0
    assert M2_tracescore.run(a, models={"rlvr": tiny_model}, toks=toks) == 0     # resume: skip
    b = _args(None, "0/1", traces=["rlvr"], pair=["rlvr", "dpo"], components=["A"], chunk=16, limit=None)
    assert M2_tracescore.run(b, models={"rlvr": tiny_model, "dpo": tiny_model_b}, toks=toks) == 0
    pt = S.Store("M2", "pair_rlvr_dpo", M2_tracescore.job_config(b.job_args, "pair_rlvr_dpo")).read(0).to_pylist()
    assert len(pt) == 2 and pt[0]["mean_kl_think"] >= 0 and pt[0]["mean_kl_gen"] >= 0
    assert len(pt[0]["kl_at_sent"]) == len(pt[0]["sent_next_tok"])


def test_m2_resumes_from_progress_after_crash(tmp_path, tiny_tok, tiny_model, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    _fake_runs(tmp_path, tiny_tok, monkeypatch)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    toks = {k: tiny_tok for k in ("sft", "dpo", "rlvr")}
    a = _args(["rlvr"], "0/1", traces=["sft", "dpo", "rlvr"], pair=None, components=list("ABCDE"), chunk=16, limit=None)
    real, calls = H.score, []

    def crash_on_third(*x, **k):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("simulated reclaim")
        return real(*x, **k)
    monkeypatch.setattr(H, "score", crash_on_third)
    with pytest.raises(RuntimeError):
        M2_tracescore.run(a, models={"rlvr": tiny_model}, toks=toks)
    calls.clear()
    monkeypatch.setattr(H, "score", lambda *x, **k: (calls.append(1), real(*x, **k))[1])
    assert M2_tracescore.run(a, models={"rlvr": tiny_model}, toks=toks) == 0
    assert len(calls) == 4, "the two finished traces are not rescored"
    st = S.Store("M2", "rlvr", M2_tracescore.job_config(a.job_args, "rlvr"))
    assert len(st.read(0).to_pylist()) == 6


def _fake_generate(tok):
    def gen(llm, prompt_ids, seeds, sampling, max_tokens=16000):
        outs = []
        for s in seeds:
            text = TRACE if s % 2 == 0 else "Thinking without closing"
            ids = H.text_ids(tok, text)
            outs.append({"text": text, "token_ids": ids, "logprobs": [0.0] * len(ids),
                         "finish_reason": "stop", "n_tokens": len(ids)})
        return outs
    return gen


def test_b1_end_to_end(tmp_path, tiny_tok, tiny_model, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    a = _args(["dpo"], "0/24", seeds=2, system="own", temperature=None, top_p=None, top_k=None, max_tokens=64,
              gpu_mem=0.4, limit=None)
    assert B1_likert.run(a, model=tiny_model, tok=tiny_tok, generate_fn=_fake_generate(tiny_tok)) == 0
    shard = S.Store("B1", "dpo__sys-own", B1_likert.job_config("dpo", a.job_args, {"temperature": 0.6, "top_p": 0.95,
                                                                                   "top_k": -1, "source": "test"}))
    rows = shard.read(0).to_pylist()
    assert len(rows) == 2 * 2                               # 2 wordings (0/24 of 48) x 2 seeds
    ok = [r for r in rows if r["seed"] == 0][0]
    bad = [r for r in rows if r["seed"] == 1][0]
    assert ok["status"] == "valid" and ok["answer_sampled"] == 4 and ok["think_tokens"] > 0
    assert abs(sum(ok[f"p_read_{j}"] for j in range(1, 6)) - 1) < 1e-9 and ok["read_primary"] == "space_in_context"
    assert abs(sum(ok[f"p_nothink_{j}"] for j in range(1, 6)) - 1) < 1e-9
    assert bad["status"] != "valid" and bad["answer_sampled"] is None and bad["read_primary"] is None
    assert bad["p_nothink_1"] == ok["p_nothink_1"]          # prior is per wording, shared across seeds


def test_b3_end_to_end(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    n_units = len(B3_instructions.units(50))
    assert n_units == (48 + 50) * 4
    a = _args(["rlvr"], "0/98", seeds=2, system="own", n_d=50, temperature=None, top_p=None, top_k=None,
              max_tokens=64, gpu_mem=0.8, limit=None)
    assert B3_instructions.run(a, tok=tiny_tok, generate_fn=_fake_generate(tiny_tok)) == 0
    st = S.Store("B3", "rlvr__sys-own", B3_instructions.job_config("rlvr", a.job_args, {"temperature": 0.6, "top_p": 0.95,
                                                                                        "top_k": -1, "source": "test"}))
    rows = st.read(0).to_pylist()
    assert len(rows) == 4 * 2 and {r["component"] for r in rows} <= {"A", "D"}
    r = [x for x in rows if x["seed"] == 0][0]
    assert r["think_sentences"] >= 5 and r["post_sentences"] == 1 and r["format_ok"] is True
    assert r["answer_lenient"] == ("4" if r["component"] == "A" else None)


def test_b3_format_ok():
    assert B3_instructions.format_ok("\nIt fits.\nANSWER: 4", "A")
    assert not B3_instructions.format_ok("\nANSWER: 4", "A")             # no justification line
    assert not B3_instructions.format_ok("\nIt fits.\nANSWER: 4\nthanks", "A")
    assert B3_instructions.format_ok("\nBecause.\nANSWER: C", "D")
