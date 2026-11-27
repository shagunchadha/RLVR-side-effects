"""Engine variants (Nov 28 diagnosis): 'fast' must stay exactly FAST_ENGINE (no recorded config hash moves);
'fast_nopc' differs only by prefix caching; the probe's start-up-line parsing and summary work on real log text."""
import argparse
import json

from exp import harness as H
from exp import store as S
from exp.jobs import B1_likert
from exp.revision import engine_probe as P


def test_fast_is_unchanged_and_nopc_differs_only_in_prefix_caching():
    assert H.engine_args("fast") == H.FAST_ENGINE and H.engine_args("fast") is not H.FAST_ENGINE
    d = {k for k in H.FAST_ENGINE if H.engine_args("fast_nopc")[k] != H.FAST_ENGINE[k]}
    assert d == {"enable_prefix_caching"} and H.engine_args("fast_nopc")["enable_prefix_caching"] is False


def test_b1_fast_hash_unchanged_and_nopc_distinct():
    s = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    kw = dict(seeds=20, system="own", max_tokens=16000, gpu_mem=0.92)
    fast = B1_likert.job_config("rlvr", argparse.Namespace(engine="fast", **kw), s)
    assert fast["engine"] == {**H.FAST_ENGINE, "gpu_memory_utilization": 0.92, "generator": "generate_stream"}
    nopc = B1_likert.job_config("rlvr", argparse.Namespace(engine="fast_nopc", **kw), s)
    assert S.Store("B1", "x", fast).hash != S.Store("B1", "x", nopc).hash


def test_probe_parses_vllm_startup_lines_and_summarises():
    log = ("INFO 11-27 16:14:09 [base.py:121] Using Transformers modeling backend.\n"
           "INFO 11-27 16:14:11 [cuda.py:526] Using FLASH_ATTN attention backend out of potential backends: [...]\n"
           "INFO 11-27 16:14:38 [gpu_worker.py:692] Available KV cache memory: 72.35 GiB\n"
           "INFO 11-27 16:14:38 [kv_cache_utils.py:2464] GPU KV cache size: 147,970 tokens, Maximum concurrency for 16,182 tokens per request: 9.14x\n"
           "INFO 11-27 16:14:48 [kv_cache_utils.py:772] kv cache group sizes [16, 16, 16, 16]\n")
    got = {k: rx.findall(log)[-1] for k, rx in P.LINES.items()}
    assert got["kv"] == ("147,970", "16,182", "9.14") and got["kv_mem"] == "72.35" and got["attn"] == "FLASH_ATTN"
    assert got["groups"] == "[16, 16, 16, 16]" and got["model_backend"] == "Transformers"
    res = {"variant": "fast", "t0": 0.0, "finished": 3, "samples": [
        {"t": t, "gen_tokens": 1000 * t, "running": 20, "waiting": 0, "kv_usage": 0.5, "preemptions": 0} for t in (0, 60, 120, 300, 600)]}
    r = P.summarise(res, {"kv": list(got["kv"])})
    assert r["steady_tok_per_s"] == 1000 and r["mean_running"] == 20 and r["per_seq_tok_per_s"] == 50
