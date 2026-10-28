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


def test_b1_phases_resume_after_crash(tmp_path, tiny_tok, tiny_model, monkeypatch):
    """gen phase crashes after the first chunk; the rerun generates only the rest; the read phase then writes the shard."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    fake, calls = _fake_generate(tiny_tok), []

    def crashing(llm, prompt_ids, seeds, sampling, max_tokens=16000):
        calls.append(len(seeds))
        if len(calls) == 2:
            raise RuntimeError("simulated reclaim")
        return fake(llm, prompt_ids, seeds, sampling, max_tokens)
    kw = dict(seeds=2, system="own", temperature=None, top_p=None, top_k=None, max_tokens=64, gpu_mem=0.85,
              limit=None, chunk=1)
    with pytest.raises(RuntimeError):
        B1_likert.run(_args(["dpo"], "0/16", phase="gen", **kw), tok=tiny_tok, generate_fn=crashing)
    calls.clear()
    assert B1_likert.run(_args(["dpo"], "0/16", phase="gen", **kw), tok=tiny_tok,
                         generate_fn=lambda *x, **k: (calls.append(1), fake(*x, **k))[1]) == 0
    assert len(calls) == 2, "the finished wording is not regenerated (3 wordings, 1 done before the crash)"
    st = S.Store("B1", "dpo__sys-own", B1_likert.job_config("dpo", _args(["dpo"], "0/16", **kw).job_args,
                                                            {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}))
    assert not st.done(0), "gen phase alone does not write the shard"
    assert B1_likert.run(_args(["dpo"], "0/16", phase="read", **kw), model=tiny_model, tok=tiny_tok,
                         generate_fn=fake) == 0
    rows = st.read(0).to_pylist()
    assert len(rows) == 3 * 2 and all(abs(sum(r[f"p_nothink_{j}"] for j in range(1, 6)) - 1) < 1e-9 for r in rows)


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


def test_b3_resumes_from_progress_after_crash(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    fake, calls = _fake_generate(tiny_tok), []

    def crashing(llm, prompt_ids, seeds, sampling, max_tokens=16000):
        calls.append(len(seeds))
        if len(calls) == 2:
            raise RuntimeError("simulated reclaim")
        return fake(llm, prompt_ids, seeds, sampling, max_tokens)
    kw = dict(seeds=2, system="own", n_d=50, temperature=None, top_p=None, top_k=None, max_tokens=64, gpu_mem=0.85,
              limit=None, chunk=1)
    with pytest.raises(RuntimeError):
        B3_instructions.run(_args(["rlvr"], "0/98", **kw), tok=tiny_tok, generate_fn=crashing)
    calls.clear()
    assert B3_instructions.run(_args(["rlvr"], "0/98", **kw), tok=tiny_tok,
                               generate_fn=lambda *x, **k: (calls.append(1), fake(*x, **k))[1]) == 0
    assert len(calls) == 3, "4 units, 1 finished before the crash: only 3 are regenerated"
    st = S.Store("B3", "rlvr__sys-own", B3_instructions.job_config("rlvr", _args(["rlvr"], "0/98", **kw).job_args,
                                                                  {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}))
    assert len(st.read(0).to_pylist()) == 4 * 2


def test_b3_format_ok():
    assert B3_instructions.format_ok("\nIt fits.\nANSWER: 4", "A")
    assert not B3_instructions.format_ok("\nANSWER: 4", "A")             # no justification line
    assert not B3_instructions.format_ok("\nIt fits.\nANSWER: 4\nthanks", "A")
    assert B3_instructions.format_ok("\nBecause.\nANSWER: C", "D")


def test_m2_kl_detail_attributes_think_kl_to_written_tokens():
    from exp.jobs import M2_tracescore as M2
    prep = {"think_end_tok": 4, "gen_ids": [7, 8, 7, 9, 5, 5]}
    d = M2.kl_detail(prep, [0.1, 0.2, 0.3, 0.05, 9.0, 9.0])      # positions >= think_end_tok are ignored
    assert d["sum_kl_think"] == pytest.approx(0.65)
    assert d["kl_by_token_top"][0][0] == 7 and d["kl_by_token_top"][0][1] == pytest.approx(0.4)
    assert d["kl_by_token_top"][0][2] == 2 and {t for t, _, _ in d["kl_by_token_top"]} == {7, 8, 9}
    assert len(d["kl_pos_bins"]) == M2.KL_BINS


def test_strict_yesno():
    from exp import parsers as P
    assert P.strict_yesno("<think>a.</think>\nNo body has.\nANSWER: no", 10, 100, True)["answer"] == "no"
    assert P.strict_yesno("<think>a.</think>\nANSWER: Yes.", 10, 100, True)["status"] == "valid"
    assert P.strict_yesno("<think>a.</think>\nANSWER: maybe", 10, 100, True)["status"] == "malformed"
    assert P.strict_yesno("<think>never closed", 100, 100, False)["status"] == "truncated"


def test_b2_items_frozen_and_end_to_end(tmp_path, tiny_tok, monkeypatch):
    from exp.jobs import B2_indeterminacy as B2
    items, _ = B2.load_items()
    cells = {c: sum(1 for i in items if i["cell"] == c) for c in ("self_indet", "self_det", "other_indet", "other_det")}
    assert cells == {"self_indet": 48, "self_det": 40, "other_indet": 40, "other_det": 40}
    assert sum(1 for i in items if i["cell"] == "self_det" and i["gold"] == "yes") == 20
    A = {it["id"]: it for it in C.load_items("A")}
    assert all(i["messages"] == A[i["source"]]["messages"] for i in items if i["cell"] == "self_indet")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    a = _args(["rlvr"], "0/84", seeds=2, system="own", chunk=8, max_tokens=64, gpu_mem=0.8, limit=None)
    assert B2.run(a, tok=tiny_tok, generate_fn=_fake_generate(tiny_tok)) == 0


def test_ph_handoff_end_to_end(tmp_path, tiny_tok, monkeypatch):
    from exp.jobs import PH_handoff as PH
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    us, _ = PH.units(tiny_tok, 50)
    assert {u["prefix_src"] for u in us} == {"rlvr", "dpo"} and {u["cut"] for u in us} <= set(PH.CUTS)
    assert all(0 < u["prefix_tok"] < u["trace_think_tok"] for u in us)
    a = _args(["dpo"], None, seeds=2, n_d=50, chunk=4, max_tokens=64, gpu_mem=0.8, limit=3)

    def fake(llm, pids, seeds, sampling, max_tokens=None):
        assert isinstance(max_tokens, list) and len(max_tokens) == len(seeds)
        return [{"text": " So done.\n</think>\n\nANSWER: 4", "token_ids": [1, 2], "logprobs": [0.0, 0.0],
                 "finish_reason": "stop", "n_tokens": 2} for _ in seeds]
    assert PH.run(a, tok=tiny_tok, generate_fn=fake) == 0


def test_m3_settle_rule():
    from exp.jobs import M3_settle as M3
    assert M3.settle(["3", "4", "4", "4"], "4") == 1
    assert M3.settle(["4", "3", "4"], "4") == 2
    assert M3.settle(["4", "4", "3"], "4") is None
    assert M3.settle(["E"], "E") == 0


def test_m3_trace_pass_tiny(tiny_tok, tiny_model):
    from exp import harness as H
    from exp.jobs import M3_settle as M3
    it = C.load_items("A")[0]
    r = {"ckpt": "rlvr", "id": it["id"], "seed": 0, "answer_strict": 4,
         "think_text": "I think so. Then again maybe not. Overall I agree."}
    digits = [H.single_token_id(tiny_tok, str(d)) for d in range(1, 6)]
    row, feats = M3.trace_pass(tiny_model, tiny_tok, it, r, digits, H.text_ids(tiny_tok, H.ANSWER_SUFFIX), [0, 1],
                               wrapup=M3.wrapup_groups(tiny_tok))
    assert len(row["p_wrapup_sent"]) == len(row["sent_tok"]) and row["p_wrapup_sent"][-1] is None
    assert all(a >= b for a, b in zip(row["p_wrapup_sent"][:-1], row["p_wrapup_noso_sent"][:-1]))
    assert len(row["p_sent"]) == len(row["sent_tok"]) >= 2 and row["sent_tok"][-1] == row["n_think_tok"]
    assert all(abs(sum(p) - 1) < 1e-3 for p in row["p_sent"]) and 0 <= row["share_after"] <= 1
    assert set(feats) == set(M3.POS) and set(feats["think25"]) == {0, 1}


def test_m3_d_traces_and_letters(tiny_tok, tiny_model):
    from exp import harness as H
    from exp.jobs import M3_settle as M3
    from exp.jobs.M2_tracescore import items_by_id
    ds = M3.d_traces("dpo")
    assert ds and all(r["answer_strict"] in "ABCDE" for r in ds) and all(r["id"].endswith("_o0") for r in ds)
    r = ds[0]
    letters = tuple("ABCDE")
    row, feats = M3.trace_pass(tiny_model, tiny_tok, items_by_id()[r["id"]], r,
                               [H.single_token_id(tiny_tok, x) for x in letters],
                               H.text_ids(tiny_tok, H.ANSWER_SUFFIX), [], letters, M3.wrapup_groups(tiny_tok))
    assert feats == {} and row["component"] == "D" and set(row["argmax_sent"]) <= set(letters) and row["e_last"] is None
