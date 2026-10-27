"""B1: Likert resampling (48 A wordings x 20 seeds per checkpoint) with think-conditioned and no-think readouts.

    python -m exp.run --job B1 --ckpt rlvr [--shard i/n] [--seeds 20] [--system own|dpo_default]
                      [--temperature T --top-p P] [--max-tokens 16000] [--gpu-mem 0.45]

Arms (registration amendment 1):
    --ckpt {sft,dpo,rlvr}                                       main arm, own templates, model-default sampling
    --ckpt sft --system dpo_default                             SFT with the DPO/RLVR system prompt (matched DPO - SFT)
    --ckpt {sft,dpo,rlvr} --seeds 5 --temperature 0.7 --top-p 0.9   bridge arm at the frozen sampling (no top-k)

Phases (resource safety; one model per process): --phase gen samples with vLLM alone (gpu_memory_utilization 0.85,
max_model_len = longest prompt + max_tokens, no CPU offload) and appends every finished generation to
$RESULTS_DIR/B1/<arm>/gen_shard<i>of<n>.csv; --phase read loads the HF model alone and appends readouts to
read_shard<i>of<n>.csv; when every generation has its readout the shard parquet is written. Both phases resume from
their progress files. --phase all (default) does both in one process (tests; injected models).

One process = one checkpoint. Shards split the 48 wordings round-robin. Within a shard: vLLM samples every
(wording, seed); then an HF copy of the same checkpoint computes, per generation, the think-conditioned readout
P("1".."5" | prompt + own think text + "\\n</think>\\n\\nANSWER: "), and, once per wording, the no-think prior
P("1".."5" | prompt + "</think>\\n\\nANSWER: "). Both tokenisation variants are stored; 'read_primary' names the variant
with single-token, prefix-consistent options.

Columns: ckpt, system, id, trait, key, original, seed, sampling (json), text, gen_token_ids, lp_chosen, n_tokens,
finish_reason, closed, status, answer_sampled, think_tokens, read_primary, p_read_1..5, read_mass, read_entropy,
read_expected, read_alt_1..5, read_alt_mass, p_nothink_1..5, nothink_mass, nothink_alt_1..5
"""
import json
import math
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import prompts_sprint as PS
from .. import resources as R
from .. import store as S
from ._gen import run_requests


def add_args(p):
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--system", default="own", choices=PS.SYSTEM_MODES)
    p.add_argument("--temperature", type=float, default=None, help="override generation_config.json")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (gen phase; vLLM alone)")
    p.add_argument("--phase", default="all", choices=["gen", "read", "all"])
    p.add_argument("--chunk", type=int, default=4, help="wordings per vLLM call (progress granularity)")
    p.add_argument("--limit", type=int, default=None, help="debug: first N wordings of this shard")


def job_config(ck, a, sampling):
    return {"job": "B1", "ckpt": list(C.ckpt(ck)), "system": a.system, "seeds": a.seeds, "sampling": sampling,
            "max_tokens": a.max_tokens, "items": "items/items_A_selfreport.json", "readout_suffix": H.ANSWER_SUFFIX,
            "nothink_suffix": H.NOTHINK_SUFFIX, "version": 1}


def readout_cols(prefix, ro, alt_prefix):
    prim = H.primary_variant(ro)
    alt = next(v for v in ro if v != prim)
    r, s = ro[prim], ro[alt]
    out = {f"{prefix}_primary": prim, f"{prefix}_mass": r.mass, f"{alt_prefix}_mass": s.mass}
    out.update({f"{prefix}_{j + 1}": r.probs[j] for j in range(5)})
    out.update({f"{alt_prefix}_{j + 1}": s.probs[j] for j in range(5)})
    return out, r


def run(args, llm=None, model=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("B1 needs exactly one --ckpt")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    i, n = S.parse_shard(args.shard)
    sampling = H.resolve_sampling(repo, sha, {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store("B1", f"{ck}__sys-{a.system}", job_config(ck, a, sampling))
    if st.done(i):
        print(f"B1 {ck} shard {i}/{n} exists; skipping")
        return 0
    phase = getattr(a, "phase", "all")
    chunk = getattr(a, "chunk", 4)
    items = S.shard_slice(C.load_items("A"), i, n)[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B1" / f"{ck}__sys-{a.system}__{st.hash}"
    gen = R.Progress(root / f"gen_shard{i}of{n}.csv", st.hash)
    read = R.Progress(root / f"read_shard{i}of{n}.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    msgs = {it["id"]: PS.with_system(it["messages"], a.system) for it in items}
    key = lambda iid, s: f"{iid}|{s}"  # noqa: E731
    reqs = [{"ckpt": ck, "system": a.system, "id": it["id"], "trait": it["trait"], "key": it["key"],
             "original": it["original"], "seed": s, "kind": "likert", "messages": msgs[it["id"]]}
            for it in items for s in range(a.seeds)]
    R.log(f"B1 {ck} sys={a.system} shard {i}/{n} phase={phase}: {len(reqs)} requests; generated {len(gen.done)}, "
          f"read {sum(1 for k in read.done if not k.startswith('prior|'))}; sampling {sampling}")

    # ---------------- generation (vLLM only)
    if phase in ("gen", "all"):
        todo = [r for r in reqs if key(r["id"], r["seed"]) not in gen.done]
        if todo and llm is None and generate_fn is None:
            R.preflight(2 * 15, dirs=[cache])
            mpl = max(len(H.prompt_ids(tok, m)) for m in msgs.values()) + a.max_tokens
            llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
            res.now(f"vLLM loaded (max_model_len {mpl})")
        by_item = {}
        for r in todo:
            by_item.setdefault(r["id"], []).append(r)
        ids = list(by_item)
        for c in range(0, len(ids), chunk):
            batch = [r for iid in ids[c:c + chunk] for r in by_item[iid]]
            t0 = time.time()
            for row in run_requests(llm, tok, batch, sampling, a.max_tokens, generate_fn):
                row["lp_chosen"] = [round(x, 5) for x in row["lp_chosen"]]
                gen.add(key(row["id"], row["seed"]), row, (time.time() - t0) / len(batch))
            R.log(f"B1 gen {len(gen.done)}/{len(reqs)}")
            res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
        if phase == "gen":
            R.log(f"B1 gen phase done: {len(gen.done)}/{len(reqs)}")
            return 0
        del llm

    # ---------------- readouts (HF model only)
    missing = [r for r in reqs if key(r["id"], r["seed"]) not in gen.done]
    if missing:
        raise SystemExit(f"read phase: {len(missing)} requests have no generation yet; run --phase gen first")
    todo_prior = [it for it in items if f"prior|{it['id']}" not in read.done]
    todo_read = [r for r in reqs if key(r["id"], r["seed"]) not in read.done]
    if (todo_prior or todo_read) and model is None:
        R.preflight(2 * 15, dirs=[cache])
        model = H.load_hf(repo, sha)
        res.now("HF model loaded")
    for it in todo_prior:
        ro = H.likert_readout(model, tok, msgs[it["id"]], think_text=None)
        cols, _ = readout_cols("p_nothink", ro, "nothink_alt")
        cols["nothink_primary"] = cols.pop("p_nothink_primary")
        read.add(f"prior|{it['id']}", cols, 0.0)
    for t, rq in enumerate(todo_read):
        t0 = time.time()
        r = gen.done[key(rq["id"], rq["seed"])]
        if r["think_text"] is not None:
            ro = H.likert_readout(model, tok, msgs[r["id"]], think_text=r["think_text"])
            cols, prim = readout_cols("p_read", ro, "read_alt")
            cols["read_primary"] = cols.pop("p_read_primary")
            cols["read_mass"] = cols.pop("p_read_mass")
            cols["read_entropy"], cols["read_expected"] = prim.entropy(), prim.expected()
        else:
            cols = {"read_primary": None, "read_mass": math.nan, "read_entropy": math.nan, "read_expected": math.nan,
                    "read_alt_mass": math.nan, **{f"p_read_{j}": math.nan for j in range(1, 6)},
                    **{f"read_alt_{j}": math.nan for j in range(1, 6)}}
        read.add(key(rq["id"], rq["seed"]), cols, time.time() - t0)
        if (t + 1) % 100 == 0:
            R.log(f"B1 read {t + 1}/{len(todo_read)}")
        res.maybe(f"read {t + 1}/{len(todo_read)}")

    # ---------------- assemble the shard
    out = []
    for rq in reqs:
        r = gen.done[key(rq["id"], rq["seed"])]
        row = {k: v for k, v in r.items() if k not in ("kind", "think_text", "post_text")}
        row["sampling"] = json.dumps(sampling, default=str)
        row["answer_sampled"] = r["answer_strict"] if r["status"] == "valid" else None
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
        row.update(read.done[key(rq["id"], rq["seed"])])
        row.update(read.done[f"prior|{rq['id']}"])
        out.append(row)
    res.now("done")
    loc = st.write(i, n, out, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "n_items": len(items)})
    print(f"wrote {loc}: {len(out)} rows")
    return 0
