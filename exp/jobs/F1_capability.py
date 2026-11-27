"""F1: capability control (registration amendment 18, F1, and its Nov 27 notes). Generation only; scoring is CPU-side
(exp/analysis/f1_summary.py with exp/revision/scorers.py), so the GPU environment needs no new packages.

    python -m exp.run --job F1 --ckpt rlvr --bench aime2024 [--limit N --samples K --cap1 C1 --cap2 C2]   (smoke only)

One process = one checkpoint x one benchmark. Benchmarks and samples per item (items_revision/F1_*.json):
  aime2024, aime2025: 30 problems x 32 samples (avg@32, the developers' N)
  math500: 500 x 3;  ifeval: 541 x 3;  ifbench: 300 x 3   (= the developers' "mean of three runs")
Sampling: the checkpoint's native chat template (default system prompt), reasoning on, T = 0.6, top-p = 0.95, no
top-k (generation_config.json; the resolved values are in the config), per-request seeds 0..K-1.
Stages (amendment 18, "cap-hit continuation"):
  1. up to 32,768 new tokens;
  2. every generation that hit 32,768 is CONTINUED from its own prompt + generated token ids (not resampled), same
     model, template and sampling, up to a total context of 65,536 tokens: new tokens = 65,536 - prompt - 32,768.
     Its seed is 1,000,000 + the original seed (a different RNG stream; disclosed). A generation with no room left,
     or still capped after stage 2, is unresolved: kept, flagged, never dropped.
Resume: every finished generation of either stage is appended to its progress file at once (generate_stream).
The engine (harness.FAST_ENGINE, gpu_memory_utilization) is recorded per process segment in the shard metadata and
is NOT part of the config hash: engine settings change throughput, not the sampling distribution, so retuning them
resumes from saved progress instead of starting over. Everything that defines the samples is in the hash.
Shard columns: id, seed, n_prompt_tok, n_tokens_32k, finish_32k, continued, no_room, n_tokens, finish_reason,
unresolved, text (stage-1 text, or the decoded stage-1 + stage-2 token ids for continued generations).
"""
import hashlib
import json
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import resources as R
from .. import store as S

NS = "eacl_revision_1/F1"
BENCH = {"aime2024": ("F1_aime2024.json", 32), "aime2025": ("F1_aime2025.json", 32), "math500": ("F1_math500.json", 3),
         "ifeval": ("F1_ifeval.json", 3), "ifbench": ("F1_ifbench.json", 3)}
CAP1, CAP_TOTAL = 32768, 65536
CONT_SEED = 1_000_000


def add_args(p):
    p.add_argument("--bench", required=True, choices=list(BENCH))
    p.add_argument("--gpu-mem", type=float, default=H.FAST_GPU_MEM)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N items")
    p.add_argument("--samples", type=int, default=None, help="smoke only; default = registered samples per item")
    p.add_argument("--cap1", type=int, default=CAP1, help="smoke only")
    p.add_argument("--cap2", type=int, default=CAP_TOTAL, help="smoke only: total context for stage 2")
    p.add_argument("--engine", default=os.environ.get("F1_ENGINE", "fast"), choices=list(H.ENGINES),
                   help="engine variant (harness.ENGINES); recorded per segment, not in the config hash")


def load_items(bench):
    name = BENCH[bench][0]
    d = C.REPO_ROOT / "items_revision"
    f = d / name
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    man = dict(l.split()[::-1] for l in (d / "MANIFEST.sha256").read_text().splitlines() if l.strip())
    if man.get(name) != h:
        raise SystemExit(f"{f} sha256 {h} != manifest {man.get(name)}")
    return json.loads(f.read_text(encoding="utf-8"))["items"], h


def job_config(ck, a, items_sha, sampling, n_items):
    return {"job": NS, "bench": a.bench, "ckpt": list(C.ckpt(ck)), "items_sha256": items_sha, "n_items": n_items,
            "samples": a.samples or BENCH[a.bench][1], "sampling": sampling, "cap1": a.cap1, "cap_total": a.cap2,
            "continuation": {"mode": "continue own token ids", "seed_offset": CONT_SEED}, "template": "native (own)",
            "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}


def run(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1 or args.ckpt[0] not in ("sft", "dpo", "rlvr"):
        raise SystemExit("F1 needs exactly one --ckpt of sft, dpo, rlvr")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    items, items_sha = load_items(a.bench)
    items = items[: a.limit]
    k = a.samples or BENCH[a.bench][1]
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store(NS, f"{a.bench}__{ck}", job_config(ck, a, items_sha, sampling, len(items)))
    if st.done(0):
        print(f"F1 {a.bench} {ck}: shard exists; skipping")
        return 0
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / NS / f"{a.bench}__{ck}__{st.hash}"
    g1 = R.Progress(root / "gen_32k.csv", st.hash)
    g2 = R.Progress(root / "gen_cont.csv", st.hash)
    seg_f = root / "segments.json"
    segments = json.loads(seg_f.read_text()) if seg_f.exists() else []
    tok = tok or H.load_tokenizer(repo, sha)
    key = lambda iid, s: f"{iid}|{s}"  # noqa: E731
    pids = {it["id"]: H.prompt_ids(tok, it["messages"]) for it in items}
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    t0 = time.time()
    seg = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine": {"variant": getattr(a, "engine", "fast"), **H.engine_args(getattr(a, "engine", "fast")),
                      "gpu_memory_utilization": a.gpu_mem},
           "env": {k: os.environ.get(k) for k in ("VLLM_ATTENTION_BACKEND", "VLLM_USE_V1", "CUDA_VISIBLE_DEVICES")},
           "stage1": 0, "stage2": 0, "tokens": 0}

    def engine():
        nonlocal llm
        if llm is None and generate_fn is None:
            R.preflight(2 * 15, dirs=[cache])
            llm = H.load_vllm(repo, sha, max_model_len=a.cap2, gpu_memory_utilization=a.gpu_mem,
                              engine=H.engine_args(getattr(a, "engine", "fast")))
            res.now(f"vLLM loaded (max_model_len {a.cap2}, engine {getattr(a, 'engine', 'fast')})")
        return llm

    def save_seg():
        seg["wall_s"] = round(time.time() - t0, 1)
        seg_f.parent.mkdir(parents=True, exist_ok=True)
        seg_f.write_text(json.dumps(segments + [seg], indent=1))

    R.log(f"F1 {a.bench} {ck}: {len(items)} items x {k} samples; stage 1 done {len(g1.done)}, stage 2 done {len(g2.done)}; "
          f"sampling {sampling}")
    try:
        # ---------------- stage 1: up to cap1 new tokens
        todo = [(it["id"], s) for it in items for s in range(k) if key(it["id"], s) not in g1.done]
        if todo:
            def done1(i, o):
                iid, s = todo[i]
                capped = o["finish_reason"] == "length"
                g1.add(key(iid, s), {"id": iid, "seed": s, "n_prompt_tok": len(pids[iid]), "text": o["text"],
                                     "n_tokens": o["n_tokens"], "finish_reason": o["finish_reason"],
                                     **({"token_ids": o["token_ids"]} if capped else {})}, time.time() - t0)
                seg["stage1"] += 1
                seg["tokens"] += o["n_tokens"]
                if seg["stage1"] % 50 == 0:
                    save_seg()
                    R.log(f"F1 {a.bench} {ck} stage 1: {len(g1.done)}/{len(items) * k}; {seg['tokens'] / max(time.time() - t0, 1):.0f} tok/s")
                res.maybe(f"stage1 {len(g1.done)}")
            H.generate_stream(engine(), [pids[iid] for iid, _ in todo], [s for _, s in todo], sampling,
                              max_tokens=a.cap1, on_done=done1, generate_fn=generate_fn)
        # ---------------- stage 2: continue every capped generation to the total context
        capped = [r for r in g1.done.values() if r["finish_reason"] == "length"]
        room = {key(r["id"], r["seed"]): a.cap2 - r["n_prompt_tok"] - r["n_tokens"] for r in capped}
        todo2 = [r for r in capped if key(r["id"], r["seed"]) not in g2.done and room[key(r["id"], r["seed"])] > 0]
        for r in capped:                                   # no room at all: recorded, nothing to generate
            kk = key(r["id"], r["seed"])
            if room[kk] <= 0 and kk not in g2.done:
                g2.add(kk, {"no_room": True, "token_ids": [], "n_tokens": 0, "finish_reason": "length"}, 0.0)
        if todo2:
            R.log(f"F1 {a.bench} {ck} stage 2: continuing {len(todo2)} capped generations to {a.cap2} total")

            def done2(i, o):
                r = todo2[i]
                g2.add(key(r["id"], r["seed"]), {"no_room": False, "token_ids": o["token_ids"], "n_tokens": o["n_tokens"],
                                                 "finish_reason": o["finish_reason"]}, time.time() - t0)
                seg["stage2"] += 1
                seg["tokens"] += o["n_tokens"]
                save_seg()
            H.generate_stream(engine(), [pids[r["id"]] + r["token_ids"] for r in todo2],
                              [CONT_SEED + r["seed"] for r in todo2], sampling,
                              max_tokens=[room[key(r["id"], r["seed"])] for r in todo2], on_done=done2,
                              generate_fn=generate_fn)
    finally:
        if seg["stage1"] or seg["stage2"]:
            segments.append({**seg, "wall_s": round(time.time() - t0, 1)})
            seg_f.parent.mkdir(parents=True, exist_ok=True)
            seg_f.write_text(json.dumps(segments, indent=1))
    # ---------------- assemble
    rows = []
    for it in items:
        for s in range(k):
            r1 = g1.done[key(it["id"], s)]
            cont = r1["finish_reason"] == "length"
            r2 = g2.done.get(key(it["id"], s)) if cont else None
            if cont and r2 is None:
                raise SystemExit(f"F1 {a.bench} {ck}: {it['id']}|{s} capped but not continued; rerun to resume")
            text = r1["text"] if not cont else tok.decode(r1["token_ids"] + r2["token_ids"], skip_special_tokens=True)
            fin = r1["finish_reason"] if not cont else r2["finish_reason"]
            rows.append({"id": it["id"], "seed": s, "n_prompt_tok": r1["n_prompt_tok"], "n_tokens_32k": r1["n_tokens"],
                         "finish_32k": r1["finish_reason"], "continued": cont, "no_room": bool(r2 and r2["no_room"]),
                         "n_tokens": r1["n_tokens"] + (r2["n_tokens"] if r2 else 0), "finish_reason": fin,
                         "unresolved": fin == "length", "text": text})
    meta = {"segments": segments, "env_report": args.env_report or {},
            "cap_hit_32k": sum(r["continued"] for r in rows), "unresolved": sum(r["unresolved"] for r in rows)}
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha, extra_meta=meta)
    print(f"wrote {loc}: {len(rows)} rows; capped at 32K {meta['cap_hit_32k']}, unresolved {meta['unresolved']}")
    return 0
