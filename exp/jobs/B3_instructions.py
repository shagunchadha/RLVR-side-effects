"""B3: reasoning cap x answer format, on the 48 A wordings + the first 50 D base items (o0), 10 seeds.

    python -m exp.run --job B3 --ckpt rlvr [--shard i/n] [--seeds 10] [--system own|dpo_default]
                      [--temperature T --top-p P] [--max-tokens 16000] [--gpu-mem 0.85]

Conditions: cap in {True, False} x fmt in {strict, bare} (exp/prompts_sprint.py). strict + cap reproduces the frozen
prompt exactly. One process = one checkpoint; shards split the (item, condition) list round-robin.
Resource safety: vLLM alone (gpu_memory_utilization 0.85, max_model_len = longest prompt + max_tokens, no CPU offload);
every finished generation is appended to $RESULTS_DIR/B3/<arm>/gen_shard<i>of<n>.csv and a restart resumes from it;
the shard parquet is written when all generations exist. Pre-flight and RES lines.

Columns: ckpt, system, id, component, cap, fmt, seed, sampling (json), text, gen_token_ids, lp_chosen, n_tokens,
finish_reason, closed, status (frozen parser; meaningful for strict), answer_strict, answer_lenient, lenient_rule,
think_tokens, think_sentences, post_sentences, format_ok, correct (D)
"""
import json
import os
import re
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import parsers as P
from .. import prompts_sprint as PS
from .. import resources as R
from .. import store as S
from ._gen import run_requests

CONDITIONS = [(cap, fmt) for cap in (True, False) for fmt in ("strict", "bare")]
FORMAT_LAST = {"A": re.compile(r"^ANSWER: [1-5]$"), "D": re.compile(r"^ANSWER: [A-E]$")}


def add_args(p):
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--system", default="own", choices=PS.SYSTEM_MODES)
    p.add_argument("--n-d", type=int, default=50)
    p.add_argument("--chunk", type=int, default=8, help="units per vLLM call (progress granularity)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=None)


def units(n_d):
    out = []
    for it in C.load_items("A"):
        for cap, fmt in CONDITIONS:
            out.append({"id": it["id"], "component": "A", "cap": cap, "fmt": fmt, "kind": "likert",
                        "messages": PS.a_messages(it, cap, fmt)})
    for it in PS.first_d_items(n_d):
        for cap, fmt in CONDITIONS:
            out.append({"id": it["id"], "component": "D", "cap": cap, "fmt": fmt, "kind": "mc5", "key_letter": it["key"],
                        "messages": PS.d_messages(it, cap, fmt)})
    return out


def format_ok(post, component):
    lines = [l for l in post.strip().split("\n") if l.strip()]
    answer_lines = [l for l in lines if re.match(r"^\W*ANSWER\b", l, re.I)]
    return bool(lines) and bool(FORMAT_LAST[component].match(lines[-1].strip())) and len(answer_lines) == 1 and len(lines) >= 2


def job_config(ck, a, sampling):
    return {"job": "B3", "ckpt": list(C.ckpt(ck)), "system": a.system, "seeds": a.seeds, "sampling": sampling,
            "max_tokens": a.max_tokens, "n_d": a.n_d, "conditions": CONDITIONS, "bare_A": PS.BARE_A, "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}


def run(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("B3 needs exactly one --ckpt")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    i, n = S.parse_shard(args.shard)
    sampling = H.resolve_sampling(repo, sha, {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store("B3", f"{ck}__sys-{a.system}", job_config(ck, a, sampling))
    if st.done(i):
        print(f"B3 {ck} shard {i}/{n} exists; skipping")
        return 0
    us = S.shard_slice(units(a.n_d), i, n)[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    reqs = [{**u, "ckpt": ck, "system": a.system, "seed": s, "messages": PS.with_system(u["messages"], a.system)}
            for u in us for s in range(a.seeds)]
    key = lambda r: f"{r['id']}|{r['cap']}|{r['fmt']}|{r['seed']}"  # noqa: E731
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B3" / f"{ck}__sys-{a.system}__{st.hash}"
    gen = R.Progress(root / f"gen_shard{i}of{n}.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    R.log(f"B3 {ck} shard {i}/{n}: {len(reqs)} requests, {len(reqs) - len(todo)} done; sampling {sampling}")
    if todo and llm is None and generate_fn is None:
        R.preflight(2 * 15, dirs=[cache])
        mpl = max(len(H.prompt_ids(tok, r["messages"])) for r in reqs) + a.max_tokens
        llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
        res.now(f"vLLM loaded (max_model_len {mpl})")
    chunk = getattr(a, "chunk", 8) * a.seeds
    for c in range(0, len(todo), chunk):
        batch = todo[c:c + chunk]
        t0 = time.time()
        for row in run_requests(llm, tok, batch, sampling, a.max_tokens, generate_fn):
            row["lp_chosen"] = [round(x, 5) for x in row["lp_chosen"]]
            gen.add(key(row), row, (time.time() - t0) / len(batch))
        R.log(f"B3 gen {len(gen.done)}/{len(reqs)}")
        res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
    rows = [gen.done[key(r)] for r in reqs]
    res.now("done")
    out = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("kind", "think_text", "post_text", "key_letter")}
        row["sampling"] = json.dumps(sampling, default=str)
        if r["component"] == "A":
            row["answer_lenient"], row["lenient_rule"] = P.lenient_likert(r["text"])
            row["answer_lenient"] = None if row["answer_lenient"] is None else str(row["answer_lenient"])
            row["correct"] = None
        else:
            row["answer_lenient"], row["lenient_rule"] = P.lenient_letter(r["text"])
            ans = r["answer_strict"] if r["fmt"] == "strict" and r["status"] == "valid" else row["answer_lenient"]
            row["correct"] = None if ans is None else bool(ans == r["key_letter"])
        row["answer_strict"] = None if r["answer_strict"] is None else str(r["answer_strict"])
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
        row["think_sentences"] = len(P.sentences(r["think_text"])) if r["think_text"] is not None else None
        post = r["post_text"]
        if post is not None:
            just = re.sub(r"(?im)^\W*ANSWER\b.*$", "", post.strip())
            row["post_sentences"] = len(P.sentences(just))
            row["format_ok"] = format_ok(post, r["component"])
        else:
            row["post_sentences"], row["format_ok"] = None, None
        out.append(row)
    loc = st.write(i, n, out, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "n_units": len(us)})
    print(f"wrote {loc}: {len(out)} rows")
    return 0
