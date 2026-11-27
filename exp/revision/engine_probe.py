"""Engine probe (F0b/F1 throughput diagnosis, Nov 28): same model, same AIME load, one engine variant per child process.

    python -m exp.revision.engine_probe --ckpt rlvr --variants fast fast_nopc --minutes 10 [--max-model-len 32768]

For each variant (harness.ENGINES) a child process starts vLLM at --max-model-len, then:
  1. records the engine's own start-up lines: "GPU KV cache size: N tokens, Maximum concurrency for L tokens per
     request: Cx", "Available KV cache memory", "kv cache group sizes", the attention backend and the model backend;
  2. enqueues the 60 F1 AIME problems x 8 seeds (480 requests, model-default sampling, per-request seeds, max_tokens up
     to the context) and steps the engine for --minutes;
  3. every 30 s reads vLLM's own metrics (LLM.get_metrics, with log stats on): vllm:generation_tokens (all generated
     tokens, in-flight included), vllm:num_requests_running / waiting, vllm:kv_cache_usage_perc, vllm:num_preemptions.
Throughput is the slope of vllm:generation_tokens over the steady window (after the first 2 minutes). Nothing is
scored or stored as a result: this is engine diagnosis, not an F-job. Output: one JSON per variant, and a summary
table, under --out (default $RESULTS_DIR/eacl_revision_1/engine_probe/).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LINES = {"kv": re.compile(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
         "kv_mem": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
         "groups": re.compile(r"kv cache group sizes (\[[^\]]*\])"),
         "attn": re.compile(r"Using (\w+) attention backend"),
         "model_backend": re.compile(r"Using (Transformers) modeling backend")}
SAMPLES_PER_PROBLEM = 8


def child(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from exp import config as C
    from exp import harness as H
    from exp.jobs import F1_capability as F1
    from vllm.inputs import TokensPrompt
    repo, sha = C.ckpt(a.ckpt)
    tok = H.load_tokenizer(repo, sha)
    items = F1.load_items("aime2024")[0] + F1.load_items("aime2025")[0]
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None})
    llm = H.load_vllm(repo, sha, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                      engine={**H.engine_args(a.variant), "disable_log_stats": False})
    pids = [H.prompt_ids(tok, it["messages"]) for it in items for _ in range(SAMPLES_PER_PROBLEM)]
    seeds = [s for _ in items for s in range(SAMPLES_PER_PROBLEM)]
    llm.enqueue([TokensPrompt(prompt_token_ids=p) for p in pids],
                [H._sampling_params(sampling, s, a.max_model_len - len(p)) for p, s in zip(pids, seeds)])

    def snap():
        m = {x.name: x for x in llm.get_metrics()}
        val = lambda k: (getattr(m[k], "value", None) if k in m else None)  # noqa: E731
        return {"t": time.time(), "gen_tokens": val("vllm:generation_tokens"), "running": val("vllm:num_requests_running"),
                "waiting": val("vllm:num_requests_waiting"), "kv_usage": val("vllm:kv_cache_usage_perc"),
                "preemptions": val("vllm:num_preemptions"), "prompt_tokens": val("vllm:prompt_tokens")}
    eng, t0, samples, finished = llm.llm_engine, time.time(), [], 0
    samples.append(snap())
    while time.time() - t0 < a.minutes * 60 and eng.has_unfinished_requests():
        finished += sum(1 for o in eng.step() if getattr(o, "finished", False))
        if time.time() - samples[-1]["t"] >= 30:
            samples.append(snap())
            s = samples[-1]
            print(f"PROBE {a.variant} t={s['t'] - t0:5.0f}s gen_tokens={s['gen_tokens']} running={s['running']} "
                  f"waiting={s['waiting']} kv={s['kv_usage']} preempt={s['preemptions']} finished={finished}", flush=True)
    samples.append(snap())
    Path(a.out_json).write_text(json.dumps({"variant": a.variant, "engine": H.engine_args(a.variant), "t0": t0,
                                            "samples": samples, "finished": finished, "n_requests": len(pids)}))
    try:                                 # stop the engine core process without draining 480 unfinished requests
        eng.engine_core.shutdown()
    except Exception as e:               # the parent also kills the child's whole process group afterwards
        print(f"PROBE {a.variant}: engine shutdown: {type(e).__name__}: {e}", flush=True)
    os._exit(0)


def summarise(res, startup):
    s = [x for x in res["samples"] if x["gen_tokens"] is not None]
    steady = [x for x in s if x["t"] - res["t0"] >= 120] or s
    a, b = steady[0], steady[-1]
    rate = (b["gen_tokens"] - a["gen_tokens"]) / max(b["t"] - a["t"], 1e-9)
    run = [x["running"] for x in steady if x["running"] is not None]
    mean_run = sum(run) / len(run) if run else float("nan")
    return {"variant": res["variant"], **startup, "steady_tok_per_s": round(rate), "mean_running": round(mean_run, 1),
            "per_seq_tok_per_s": round(rate / mean_run, 1) if run and mean_run else None,
            "max_kv_usage": max((x["kv_usage"] or 0) for x in s), "preemptions": s[-1]["preemptions"],
            "finished_in_window": res["finished"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rlvr")
    ap.add_argument("--variants", nargs="+", default=["fast", "fast_nopc"])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--out", default=None)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-json", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a)
    out = Path(a.out or Path(os.environ.get("RESULTS_DIR", "/tmp/sprint_out")) / "eacl_revision_1" / "engine_probe")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in a.variants:
        log, js = out / f"{a.ckpt}_{v}.log", out / f"{a.ckpt}_{v}.json"
        cmd = [sys.executable, "-m", "exp.revision.engine_probe", "--child", "--variant", v, "--ckpt", a.ckpt,
               "--minutes", str(a.minutes), "--max-model-len", str(a.max_model_len), "--gpu-mem", str(a.gpu_mem),
               "--out-json", str(js)]
        print(f"=== {time.strftime('%T')} probe {v}: {' '.join(cmd[2:])}", flush=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            rc = proc.wait()
        try:                             # nothing of this variant (e.g. vLLM's EngineCore) may keep the GPU
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            pass
        time.sleep(10)
        text = log.read_text(errors="replace")
        startup = {}
        for k, rx in LINES.items():
            m = rx.findall(text)
            if m:
                startup[k] = m[-1] if not isinstance(m[-1], tuple) else list(m[-1])
        if rc != 0 or not js.exists():
            print(f"probe {v} FAILED (rc {rc}); last log lines:\n" + "\n".join(text.splitlines()[-30:]), flush=True)
            rows.append({"variant": v, **startup, "failed": rc})
            continue
        rows.append(summarise(json.loads(js.read_text()), startup))
        print(json.dumps(rows[-1]), flush=True)
    (out / f"{a.ckpt}_summary.json").write_text(json.dumps(rows, indent=1))
    L = ["| variant | KV cache tokens | max concurrency @ len | KV GiB | groups | steady tok/s | mean running | "
         "tok/s per seq | max KV usage | preemptions |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kv = r.get("kv") or ["?", "?", "?"]
        L.append(f"| {r['variant']} | {kv[0]} | {kv[2]}x @ {kv[1]} | {r.get('kv_mem', '?')} | {r.get('groups', '?')} | "
                 f"{r.get('steady_tok_per_s', 'FAILED')} | {r.get('mean_running', '')} | {r.get('per_seq_tok_per_s', '')} | "
                 f"{r.get('max_kv_usage', '')} | {r.get('preemptions', '')} |")
    if len([r for r in rows if "steady_tok_per_s" in r]) >= 2:
        base = next(r for r in rows if r["variant"] == a.variants[0])
        for r in rows[1:]:
            if "steady_tok_per_s" in r and base.get("steady_tok_per_s"):
                L.append(f"\nspeed-up {r['variant']} / {base['variant']}: {r['steady_tok_per_s'] / base['steady_tok_per_s']:.2f}x "
                         f"(registered threshold to act: >= 1.5x)")
    (out / f"{a.ckpt}_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
