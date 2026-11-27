"""F0b (registration amendment 18): validate the revision fast generation path, and measure its throughput.

    python -m exp.run --job F0b --ckpt rlvr --part b1      # then --ckpt dpo --part b1
    python -m exp.run --job F0b --ckpt rlvr --part bench

Parts:
  b1     B1's generation phase (48 A wordings x 20 seeds, own template, model-default sampling, 16,000 cap) through
         the fast path: harness.FAST_ENGINE (prefix caching, chunked prefill, large running batch, TP = DP = 1) and
         harness.generate_stream (engine kept full, progress per finished generation). It is B1's own code
         (exp.jobs.B1_likert.run with --engine fast --phase gen), stored under eacl_revision_1/F0b_B1, so the
         submitted B1 progress and shards are never touched. After the gen phase the rows are written as a shard
         (with think_tokens, as B1 computes them). The HF readouts are not run: B1-H1 uses the sampled answers only.
         Validation (amendment 18): exp.analysis.f0b_summary compares B1-H1 with the submitted +0.181 [+0.131, +0.233].
  bench  Throughput at the revision cap, not scored: RLVR on the 60 AIME 2024/2025 problems of items_B6_pos.json,
         4 samples each (seeds 0-3), max 32,768 new tokens, fast path. Records per generation its tokens and finish
         reason, and per process segment the wall time and tokens, so tokens/s survives a resume.
"""
import argparse
import json
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import resources as R
from .. import store as S
from . import B1_likert as B1

NS = "eacl_revision_1"
B1_JOB = f"{NS}/F0b_B1"
BENCH_JOB = f"{NS}/F0b"
A_ITEMS = "items/items_A_selfreport.json"
BENCH_CAP = 32768
BENCH_SEEDS = 4


def add_args(p):
    p.add_argument("--part", required=True, choices=["b1", "bench"])
    p.add_argument("--gpu-mem", type=float, default=H.FAST_GPU_MEM)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N items")
    p.add_argument("--seeds", type=int, default=None, help="smoke only; defaults are the registered 20 (b1) / 4 (bench)")


def b1_args(a):
    """B1's own argument namespace for the registered main arm, with the fast engine."""
    return argparse.Namespace(seeds=a.seeds or 20, system="own", temperature=None, top_p=None, top_k=None,
                              max_tokens=16000, gpu_mem=a.gpu_mem, phase="gen", chunk=4, limit=a.limit, engine="fast")


def run_b1(args, llm=None, tok=None, generate_fn=None):
    ck = args.ckpt[0]
    ja = b1_args(args.job_args)
    a2 = argparse.Namespace(ckpt=[ck], shard=None, env_report=args.env_report, job_args=ja)
    rc = B1.run(a2, llm=llm, tok=tok, generate_fn=generate_fn, job=B1_JOB, items_desc=A_ITEMS)
    if rc:
        return rc
    # the same Store / progress file B1.run used (B1.run's own arithmetic, repeated here)
    repo, sha = C.ckpt(ck)
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    cfg = B1.job_config(ck, ja, sampling, B1_JOB, A_ITEMS)
    st = S.Store(B1_JOB, f"{ck}__sys-own", cfg)
    if st.done(0):
        print(f"F0b b1 {ck}: shard exists; skipping")
        return 0
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / B1_JOB / f"{ck}__sys-own__{st.hash}"
    gen = R.Progress(root / "gen_shard0of1.csv", st.hash)
    tok = tok or H.load_tokenizer(repo, sha)
    rows = []
    for r in gen.done.values():
        row = {k: v for k, v in r.items() if k not in ("kind", "think_text", "post_text")}
        row["answer_sampled"] = r["answer_strict"] if r["status"] == "valid" else None
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
        row["sampling"] = json.dumps(sampling, default=str)
        rows.append(row)
    n_expected = (ja.limit or 48) * ja.seeds
    if len(rows) < n_expected:
        raise SystemExit(f"F0b b1 {ck}: {len(rows)} of {n_expected} generations; rerun to resume")
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "engine": cfg.get("engine")})
    print(f"wrote {loc}: {len(rows)} rows")
    return 0


def bench_items(limit=None):
    items, _ = __import__("exp.jobs.B6_controls", fromlist=["load"]).load("items_B6_pos.json")
    return [it for it in items if it["source"] in ("AIME 2024", "AIME 2025")][:limit]


def run_bench(args, llm=None, tok=None, generate_fn=None):
    ck = args.ckpt[0]
    a = args.job_args
    seeds = a.seeds or BENCH_SEEDS
    repo, sha = C.ckpt(ck)
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    items = bench_items(a.limit)
    cfg = {"job": BENCH_JOB, "part": "bench", "ckpt": [repo, sha], "items": "items_B6_pos.json: AIME 2024 + 2025",
           "n_items": len(items), "seeds": seeds, "max_tokens": BENCH_CAP, "sampling": sampling,
           "engine": {**H.FAST_ENGINE, "gpu_memory_utilization": a.gpu_mem, "generator": "generate_stream"},
           "version": 1}
    st = S.Store(BENCH_JOB, f"bench__{ck}", cfg)
    if st.done(0):
        print(f"F0b bench {ck}: shard exists; skipping")
        return 0
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / BENCH_JOB / f"bench__{ck}__{st.hash}"
    gen = R.Progress(root / "gen_shard0of1.csv", st.hash)
    seg_f = root / "segments.json"
    segments = json.loads(seg_f.read_text()) if seg_f.exists() else []
    tok = tok or H.load_tokenizer(repo, sha)
    key = lambda iid, s: f"{iid}|{s}"  # noqa: E731
    reqs = [(it, s) for it in items for s in range(seeds) if key(it["id"], s) not in gen.done]
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    if reqs:
        pids = [H.prompt_ids(tok, it["messages"]) for it, _ in reqs]
        if llm is None and generate_fn is None:
            R.preflight(2 * 15, dirs=[cache])
            llm = H.load_vllm(repo, sha, max_model_len=max(len(p) for p in pids) + BENCH_CAP,
                              gpu_memory_utilization=a.gpu_mem, engine=H.FAST_ENGINE)
            res.now("vLLM loaded (engine fast)")
        t0, seg = time.time(), {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "generations": 0, "tokens": 0}

        def done(i, o):
            it, s = reqs[i]
            gen.add(key(it["id"], s), {"id": it["id"], "source": it["source"], "seed": s, "n_prompt_tok": len(pids[i]),
                                       "n_tokens": o["n_tokens"], "finish_reason": o["finish_reason"],
                                       "t_done": round(time.time() - t0, 2)}, 0.0)
            seg["generations"] += 1
            seg["tokens"] += o["n_tokens"]
            seg["wall_s"] = round(time.time() - t0, 2)
            if seg["generations"] % 20 == 0:
                seg_f.write_text(json.dumps(segments + [seg], indent=1))
                R.log(f"F0b bench {len(gen.done)}/{len(items) * seeds}: {seg['tokens'] / max(seg['wall_s'], 1e-9):.0f} tok/s")
            res.maybe(f"bench {len(gen.done)}")

        try:
            H.generate_stream(llm, pids, [s for _, s in reqs], sampling, max_tokens=BENCH_CAP, on_done=done,
                              generate_fn=generate_fn)
        finally:                      # keep this segment's wall time and tokens even if the process dies here
            segments.append(seg)
            seg_f.write_text(json.dumps(segments, indent=1))
    rows = list(gen.done.values())
    tokens, wall = sum(s["tokens"] for s in segments), sum(s.get("wall_s", 0.0) for s in segments)
    meta = {"segments": segments, "tokens": tokens, "wall_s": wall, "tok_per_s": tokens / wall if wall else None,
            "cap_hit": sum(1 for r in rows if r["finish_reason"] == "length"), "env_report": args.env_report or {}}
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha, extra_meta=meta)
    print(f"wrote {loc}: {len(rows)} rows; {meta['tok_per_s'] or 0:.0f} tok/s over {wall / 3600:.2f} h")
    return 0


def run(args, llm=None, tok=None, generate_fn=None):
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("F0b needs exactly one --ckpt")
    if args.job_args.part == "b1":
        if args.ckpt[0] not in ("rlvr", "dpo"):
            raise SystemExit("F0b b1 validates B1-H1: --ckpt rlvr or dpo")
        return run_b1(args, llm, tok, generate_fn)
    return run_bench(args, llm, tok, generate_fn)
