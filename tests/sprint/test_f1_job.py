"""F1 job (amendment 18) on CPU with a fake engine: stage 1 at cap1, continuation of capped generations from their own
token ids to the total context, no-room and unresolved generations kept, resume after a crash in either stage."""
import argparse
import json

import pytest

from exp import harness as H
from exp import store as S
from exp.jobs import F1_capability as F1


def _args(ck, bench, **kw):
    base = dict(bench=bench, gpu_mem=0.92, limit=2, samples=4, cap1=40, cap2=None)
    base.update(kw)
    return argparse.Namespace(ckpt=[ck], shard=None, env_report=None, job_args=argparse.Namespace(**base))


def _fake(tok, calls, crash_at=None, unresolved_seed=None):
    """Stage 1 (int cap): seeds 1 and 3 hit the cap. Stage 2 (list caps): stops, except unresolved_seed (- 1e6)."""
    def gen(llm, pids, seeds, sampling, max_tokens=16000):
        calls.append((len(seeds), isinstance(max_tokens, list)))
        caps = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(seeds)
        for k, (p, s, cap) in enumerate(zip(pids, seeds, caps)):
            if crash_at is not None and len(calls) == crash_at[0] and k == crash_at[1]:
                raise RuntimeError("simulated reclaim")
            stage2 = isinstance(max_tokens, list)
            if not stage2 and s % 2 == 1:                     # capped inside the think block
                ids = (H.text_ids(tok, "<think>Hmm, wait. ") * 40)[:cap]
                yield {"text": tok.decode(ids), "token_ids": ids, "logprobs": [0.0] * len(ids), "finish_reason": "length", "n_tokens": len(ids)}
            elif stage2 and unresolved_seed is not None and s - F1.CONT_SEED == unresolved_seed:
                ids = (H.text_ids(tok, "still thinking ") * 99)[:cap]
                yield {"text": tok.decode(ids), "token_ids": ids, "logprobs": [0.0] * len(ids), "finish_reason": "length", "n_tokens": len(ids)}
            else:
                text = "done.</think>\nThe answer is $\boxed{3}$." if stage2 else "<think>ok</think>\n$\boxed{3}$"
                ids = H.text_ids(tok, text)
                yield {"text": text, "token_ids": ids, "logprobs": [0.0] * len(ids), "finish_reason": "stop", "n_tokens": len(ids)}
    return gen


def _rows(tmp_path, bench, ck):
    import pyarrow.parquet as pq
    f = next((tmp_path / "store").glob(f"runs/eacl_revision_1/F1/{bench}__{ck}/*/shard_000.parquet"))
    t = pq.read_table(f)
    return t.to_pylist(), S.read_metadata(t)


def test_f1_items_frozen_and_counts():
    n = {"aime2024": 30, "aime2025": 30, "math500": 500, "ifeval": 541, "ifbench": 300}
    for b, (name, k) in F1.BENCH.items():
        items, _ = F1.load_items(b)
        assert len(items) == n[b] and all(it["messages"][0]["role"] == "user" for it in items)
    assert F1.BENCH["aime2024"][1] == F1.BENCH["aime2025"][1] == 32 and F1.BENCH["math500"][1] == 3
    aime, _ = F1.load_items("aime2025")
    assert aime[0]["messages"][0]["content"].endswith("\n\nPresent the answer in LaTex format: \\boxed{Your answer}")


def test_f1_continuation_resume_and_unresolved(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    p_len = len(H.prompt_ids(tiny_tok, F1.load_items("aime2024")[0][0]["messages"]))
    total = p_len + 40 + 25                                      # stage 2 room: 25 tokens for item 0
    calls = []
    # crash in stage 1 after 3 generations, then in stage 2 after 1
    with pytest.raises(RuntimeError):
        F1.run(_args("rlvr", "aime2024", cap2=total), tok=tiny_tok, generate_fn=_fake(tiny_tok, calls, crash_at=(1, 3)))
    with pytest.raises(RuntimeError):
        F1.run(_args("rlvr", "aime2024", cap2=total), tok=tiny_tok, generate_fn=_fake(tiny_tok, calls, crash_at=(3, 1)))
    assert F1.run(_args("rlvr", "aime2024", cap2=total), tok=tiny_tok,
                  generate_fn=_fake(tiny_tok, calls, unresolved_seed=3)) == 0
    assert calls[0] == (8, False) and calls[1] == (5, False), "stage 1 resumes from the 3 saved generations"
    assert calls[2][1] and calls[-1][1] and calls[-1][0] < calls[2][0], "stage 2 resumes too"
    rows, meta = _rows(tmp_path, "aime2024", "rlvr")
    assert len(rows) == 2 * 4
    cont = [r for r in rows if r["continued"]]
    assert {r["seed"] for r in cont} == {1, 3} and all(r["n_tokens_32k"] == 40 for r in cont)
    assert all(r["n_tokens"] > 40 for r in cont if not r["no_room"])
    unres = [r for r in rows if r["unresolved"]]
    assert unres and all(r["seed"] == 3 for r in unres) and all(r["finish_reason"] == "length" for r in unres)
    done = [r for r in cont if not r["unresolved"]]
    # stage 1 + stage 2 joined (the tiny tokenizer treats <think> as special, so check the plain text of both stages)
    assert done and all("Hmm, wait." in r["text"] and "\boxed{3}" in r["text"] for r in done)
    assert json.loads(meta["cap_hit_32k"]) == 4 and json.loads(meta["unresolved"]) == len(unres)


def test_f1_no_room_is_kept_and_flagged(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    calls = []
    p_len = min(len(H.prompt_ids(tiny_tok, it["messages"])) for it in F1.load_items("math500")[0][:2])   # no item has room
    assert F1.run(_args("dpo", "math500", cap2=p_len + 40, samples=2), tok=tiny_tok, generate_fn=_fake(tiny_tok, calls)) == 0
    rows, _ = _rows(tmp_path, "math500", "dpo")
    nr = [r for r in rows if r["no_room"]]
    assert nr and all(r["unresolved"] and r["continued"] and r["n_tokens"] == 40 for r in nr)
    assert all(not c[1] for c in calls), "no stage-2 generation when no capped row has room"


def test_f1_engine_is_not_in_the_config_hash():
    a = argparse.Namespace(bench="aime2024", samples=None, cap1=F1.CAP1, cap2=F1.CAP_TOTAL, gpu_mem=0.92)
    b = argparse.Namespace(bench="aime2024", samples=None, cap1=F1.CAP1, cap2=F1.CAP_TOTAL, gpu_mem=0.80)
    s = {"temperature": 0.6, "top_p": 0.95, "top_k": -1}
    assert F1.job_config("rlvr", a, "x", s, 30) == F1.job_config("rlvr", b, "x", s, 30)
    assert "engine" not in F1.job_config("rlvr", a, "x", s, 30)
    assert F1.job_config("rlvr", a, "x", s, 30)["samples"] == 32 and F1.job_config("rlvr", a, "x", s, 30)["cap_total"] == 65536
