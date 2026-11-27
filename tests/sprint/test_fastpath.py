"""F0b (amendment 18): the revision fast generation path, on CPU with a fake vLLM engine.

What is checked: generate_stream returns every row in input order with on_done called once per request as it
finishes, never holds more than `window` requests, and uses the same per-request seeds/caps as generate; B1's fast
path stores exactly the rows the submitted chunked path stores; the submitted B1 config (and hash) is unchanged;
F0b's parts run end to end and resume; f0b_summary applies the registered pass rule."""
import argparse
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from exp import harness as H
from exp import store as S
from exp.jobs import B1_likert, F0b_fastpath as F0b

from .test_jobs import _args, _fake_generate


# ------------------------------------------------------------------ a fake vLLM 0.31 offline engine
class _Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Out:
    def __init__(self, rid, text, ids):
        self.request_id, self.finished = rid, True
        self.outputs = [types.SimpleNamespace(text=text, token_ids=ids, logprobs=None, finish_reason="stop")]


class FakeLLM:
    """LLM.enqueue + llm_engine.step / has_unfinished_requests / get_num_unfinished_requests, as in vLLM 0.31.
    Request i needs (seed % 3) + 1 steps, so requests finish out of submission order."""

    def __init__(self):
        self.llm_engine, self.inflight, self.counter, self.max_inflight, self.params = self, {}, 0, 0, []

    def enqueue(self, prompts, params):
        rids = []
        for p, sp in zip(prompts, params):
            rid = str(self.counter)
            self.counter += 1
            self.inflight[rid] = [sp.seed % 3 + 1, p["prompt_token_ids"], sp]
            self.params.append(sp)
            rids.append(rid)
        self.max_inflight = max(self.max_inflight, len(self.inflight))
        return rids

    def get_num_unfinished_requests(self):
        return len(self.inflight)

    def has_unfinished_requests(self):
        return bool(self.inflight)

    def step(self):
        out = []
        for rid in list(self.inflight):
            self.inflight[rid][0] -= 1
            if self.inflight[rid][0] == 0:
                _, pids, sp = self.inflight.pop(rid)
                out.append(_Out(rid, f"seed {sp.seed} len {len(pids)}", [sp.seed, len(pids)]))
        return out


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = _Params
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    vllm.inputs = inputs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    return FakeLLM()


def test_generate_stream_order_window_and_callbacks(fake_vllm):
    pids = [[1] * (k + 1) for k in range(23)]
    seeds = list(range(23))
    caps = [100 + k for k in range(23)]
    seen = []
    rows = H.generate_stream(fake_vllm, pids, seeds, {"temperature": 0.6, "top_p": 0.95, "top_k": -1}, max_tokens=caps,
                             on_done=lambda i, r: seen.append(i), window=5)
    assert [r["text"] for r in rows] == [f"seed {s} len {len(p)}" for s, p in zip(seeds, pids)]   # input order
    assert sorted(seen) == list(range(23)) and seen != sorted(seen)            # each once, as they finish
    assert fake_vllm.max_inflight <= 5                                         # the window is respected
    assert sorted((p.seed, p.max_tokens) for p in fake_vllm.params) == list(zip(seeds, caps))
    assert all(p.logprobs == 0 and p.top_k == -1 for p in fake_vllm.params)


def test_generate_stream_raises_if_enqueue_misbehaves(fake_vllm, monkeypatch):
    monkeypatch.setattr(fake_vllm, "enqueue", lambda prompts, params: None)
    with pytest.raises(RuntimeError, match="enqueue"):
        H.generate_stream(fake_vllm, [[1]], [0], {"temperature": 0.6, "top_p": 0.95, "top_k": -1}, max_tokens=8)


def test_b1_submitted_config_unchanged_and_fast_config_distinct():
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    kw = dict(seeds=20, system="own", max_tokens=16000, gpu_mem=0.85)
    old = B1_likert.job_config("rlvr", argparse.Namespace(**kw), sampling)                       # no engine attribute
    sub = B1_likert.job_config("rlvr", argparse.Namespace(engine="submitted", **kw), sampling)
    fast = B1_likert.job_config("rlvr", argparse.Namespace(engine="fast", **kw), sampling)
    assert old == sub and "engine" not in sub
    assert fast["engine"]["enable_chunked_prefill"] and fast["engine"]["tensor_parallel_size"] == 1
    assert S.Store("B1", "x", old).hash == S.Store("B1", "x", sub).hash != S.Store("B1", "x", fast).hash


def test_b1_submitted_hash_reproduced_from_the_stored_config():
    """The submitted B1 main-arm config.json (local HF mirror, when present) hashes to its own directory name with
    today's job_config, so the fast-path change cannot have moved a submitted hash."""
    root = Path(__file__).resolve().parents[2] / "results" / "sprint" / "_hf" / "runs" / "B1"
    cfgs = sorted(root.glob("*__sys-own/*/config.json"))
    if not cfgs:
        pytest.skip("no local mirror of the submitted B1 runs")
    for f in cfgs:
        c = json.loads(f.read_text())
        a = argparse.Namespace(system=c["system"], seeds=c["seeds"], max_tokens=c["max_tokens"])
        ck = f.parts[-3].split("__")[0]
        rebuilt = B1_likert.job_config(ck, a, c["sampling"])
        rebuilt["vllm_flashinfer_sampler"] = c["vllm_flashinfer_sampler"]
        assert S.Store("B1", f.parts[-3], rebuilt).hash == f.parts[-2], f


def test_b1_fast_path_stores_the_same_rows(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("STORE", "local")
    kw = dict(seeds=2, system="own", temperature=None, top_p=None, top_k=None, max_tokens=64, gpu_mem=0.85,
              limit=None, chunk=1, phase="gen")
    out = {}
    for eng in ("submitted", "fast"):
        monkeypatch.setenv("RESULTS_DIR", str(tmp_path / eng))
        assert B1_likert.run(_args(["dpo"], "0/16", engine=eng, **kw), tok=tiny_tok,
                             generate_fn=_fake_generate(tiny_tok)) == 0
        f = next((tmp_path / eng / "B1").glob("dpo__sys-own__*/gen_shard0of16.csv"))
        from exp.analysis.b1_summary import read_progress
        out[eng] = read_progress(f)
    assert out["submitted"] == out["fast"] and len(out["fast"]) == 3 * 2


def _f0b(part, ck, **kw):
    return argparse.Namespace(ckpt=[ck], shard=None, env_report=None,
                              job_args=argparse.Namespace(part=part, gpu_mem=0.92, **kw))


def test_f0b_b1_end_to_end_and_summary(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "rev"))
    monkeypatch.setenv("STORE", "local")
    for ck in ("rlvr", "dpo"):
        assert F0b.run(_f0b("b1", ck, limit=None, seeds=6), tok=tiny_tok, generate_fn=_fake_generate(tiny_tok)) == 0
    shards = sorted((tmp_path / "rev").glob("runs/eacl_revision_1/F0b_B1/*__sys-own/*/shard_000.parquet"))
    assert len(shards) == 2
    import pyarrow.parquet as pq
    rows = pq.read_table(shards[0]).to_pylist()
    assert len(rows) == 48 * 6 and any(r["think_tokens"] for r in rows) and "think_text" not in rows[0]
    # submitted stand-in: the same rows under runs/B1 (identical data must pass every registered check)
    sub = tmp_path / "sub"
    for s in shards:
        dst = sub / "runs" / "B1" / s.parts[-3] / s.parts[-2]
        dst.mkdir(parents=True)
        shutil.copy(s, dst / s.name)
    from exp.analysis import f0b_summary as FS
    monkeypatch.setattr(FS, "SUBMITTED_H1", (0.0, -0.01, 0.01))          # the fake data has RLVR - DPO = 0
    assert FS.main(["--submitted", str(sub), "--revision", str(tmp_path / "rev"), "--out", str(tmp_path / "out")]) == 0
    text = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "F0b validation: PASS" in text and "5/5" in text


def test_f0b_bench_resumes_and_reports_throughput(tmp_path, tiny_tok, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "rev"))
    monkeypatch.setenv("STORE", "local")
    fake, calls = _fake_generate(tiny_tok), []

    def crashing(llm, pids, seeds, sampling, max_tokens=16000):
        """First call: 4 generations finish (and are saved), then the sandbox is reclaimed."""
        calls.append(len(seeds))
        rows = fake(llm, pids, seeds, sampling, max_tokens)
        for k, r in enumerate(rows):
            if len(calls) == 1 and k == 4:
                raise RuntimeError("simulated reclaim")
            yield r
    with pytest.raises(RuntimeError):
        F0b.run(_f0b("bench", "rlvr", limit=3, seeds=2), tok=tiny_tok, generate_fn=crashing)
    assert F0b.run(_f0b("bench", "rlvr", limit=3, seeds=2), tok=tiny_tok, generate_fn=crashing) == 0
    assert calls == [6, 2], "the 4 finished generations are not regenerated"
    import pyarrow.parquet as pq
    f = next((tmp_path / "rev").glob("runs/eacl_revision_1/F0b/bench__rlvr/*/shard_000.parquet"))
    t = pq.read_table(f)
    meta = S.read_metadata(t)
    assert t.num_rows == 6 and json.loads(meta["tokens"]) > 0 and json.loads(meta["cap_hit"]) == 0
    assert len(json.loads(meta["segments"])) == 2                          # the crashed segment is kept
    assert {r["source"] for r in t.to_pylist()} <= {"AIME 2024", "AIME 2025"}
    assert len(F0b.bench_items()) == 60


def test_rule2_one_sided_wilson_edges():
    """Amendment 18 note (Nov 27): rule 2 is one-sided. A perfect run passes against DPO's submitted 959/960 (a
    two-sided interval would fail it); RLVR's 960/960 tolerates 3 invalid of 960, not 4; DPO's tolerates 5, not 6."""
    from exp.analysis.b1_summary import wilson
    ok = lambda k_sub, k_fast, n=960: k_fast / n >= wilson(k_sub, n)[0]   # noqa: E731
    assert ok(959, 960) and wilson(959, 960)[1] < 1.0
    assert ok(960, 957) and not ok(960, 956)
    assert ok(959, 955) and not ok(959, 954)
