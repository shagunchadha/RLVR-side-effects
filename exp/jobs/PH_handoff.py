"""PH: prefix handoff (registration amendment 4, F3; exploratory). One continuing model per process.

    python -m exp.run --job PH --ckpt dpo [--seeds 3] [--max-tokens 16000] [--gpu-mem 0.85] [--limit N]

Material: the 48 A wordings and the first 50 D base items (o0); for each, the frozen full-run trace of RLVR and of DPO
(A: first valid seed with a closed think block among 0, 1, 2; D: the o0 trace). Each trace is re-tokenised exactly as
in M2 (M2_tracescore.prepare) and cut at the first sentence start (sent_next_tok) at or after 25 / 50 / 75% of its
think tokens. The continuing model (--ckpt) samples from prompt ids + prefix ids (the RLVR and DPO templates and
default system prompts are byte-identical, so the prompt ids are the same), 3 seeds, model-default sampling, with a
cap of max_tokens - len(prefix) so that prefix + continuation <= 16000 generated tokens.

Per continuation: remaining think tokens (continuation tokens before '</think>'; at the cap without a close it is the
continuation length, flagged closed=False), the final answer of prefix + continuation (frozen parser) and whether it
equals the trace's own answer. Resource safety as B3 (vLLM alone, 0.85, max_model_len = longest prompt+prefix + cap,
no CPU offload; per-continuation progress file $RESULTS_DIR/PH/<arm>/gen_shard0of1.csv; pre-flight and RES lines).

Columns: cont_ckpt, prefix_src, id, component, cut, seed, prefix_tok, trace_think_tok, trace_answer, text, n_tokens,
finish_reason, closed, remaining_think_tok, status, answer, answer_changed
"""
import json
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import parsers as P
from .. import resources as R
from .. import store as S
from . import M2_tracescore as M2

CUTS = (0.25, 0.5, 0.75)
SOURCES = ("rlvr", "dpo")


def add_args(p):
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-d", type=int, default=50)
    p.add_argument("--chunk", type=int, default=16, help="prefixes per vLLM call (progress granularity)")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=None)


def pick_traces(n_d):
    """{(src, unit id): frozen row}: A first valid closed seed per wording; D the o0 trace of the first n_d base items."""
    d_ids = {it["id"] for it in sorted((x for x in C.load_items("D") if x["order"] == 0),
                                       key=lambda x: x["draw_index"])[:n_d]}
    out = {}
    for tc, r in M2.load_traces(SOURCES, ["A", "D"]):
        if r.get("status") != "valid" or "</think>" not in r["raw_output"]:
            continue
        if r["component"] == "D" and r["id"] not in d_ids:
            continue
        k = (tc, r["id"])
        if k not in out or (r["component"] == "A" and r["seed"] < out[k]["seed"]):
            out[k] = r
    return out


def units(tok, n_d):
    items = M2.items_by_id()
    eos = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else 0
    out, skipped = [], []
    for (src, iid), r in sorted(pick_traces(n_d).items()):
        prep = M2.prepare(tok, items[iid], r, eos)
        te = prep["think_end_tok"]
        own = P.strict_answer(r["raw_output"], items[iid]["parser_kind"])
        for cut in CUTS:
            k = next((s for s in prep["sent_next_tok"] if s >= cut * te and s < te), None)
            if k is None:
                skipped.append((src, iid, cut))
                continue
            out.append({"prefix_src": src, "id": iid, "component": r["component"], "kind": items[iid]["parser_kind"],
                        "cut": cut, "trace_seed": r["seed"], "prompt_ids": prep["prompt_ids"],
                        "prefix_ids": prep["gen_ids"][:k], "prefix_tok": k, "trace_think_tok": te,
                        "trace_answer": None if own["answer"] is None else str(own["answer"])})
    return out, skipped


def job_config(ck, a, sampling):
    return {"job": "PH", "cont_ckpt": list(C.ckpt(ck)), "sources": {s: list(C.ckpt(s)) for s in SOURCES},
            "cuts": list(CUTS), "seeds": a.seeds, "n_d": a.n_d, "sampling": sampling, "max_tokens_total": a.max_tokens,
            "trace_files": {s: str(C.FULL_RUN[s].relative_to(C.REPO_ROOT)) for s in SOURCES},
            "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}


def run(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1 or args.ckpt[0] not in SOURCES:
        raise SystemExit("PH needs exactly one --ckpt (rlvr or dpo): the continuing model")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store("PH", ck, job_config(ck, a, sampling))
    if st.done(0):
        print(f"PH {ck} exists; skipping")
        return 0
    tok = tok or H.load_tokenizer(repo, sha)
    us, skipped = units(tok, a.n_d)
    us = us[: a.limit]
    reqs = [{**u, "seed": s} for u in us for s in range(a.seeds)]
    key = lambda r: f"{r['prefix_src']}|{r['id']}|{r['cut']}|{r['seed']}"  # noqa: E731
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "PH" / f"{ck}__{st.hash}"
    gen = R.Progress(root / "gen_shard0of1.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    R.log(f"PH cont={ck}: {len(us)} prefixes ({len(skipped)} cuts without a sentence start skipped), {len(reqs)} "
          f"requests, {len(reqs) - len(todo)} done; sampling {sampling}")
    if todo and llm is None and generate_fn is None:
        R.preflight(2 * 15, dirs=[cache])
        mpl = max(len(u["prompt_ids"]) for u in us) + a.max_tokens
        llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
        res.now(f"vLLM loaded (max_model_len {mpl})")
    gen_fn = generate_fn or H.generate
    chunk = a.chunk * a.seeds
    for c in range(0, len(todo), chunk):
        batch = todo[c:c + chunk]
        t0 = time.time()
        outs = gen_fn(llm, [r["prompt_ids"] + r["prefix_ids"] for r in batch], [r["seed"] for r in batch], sampling,
                      max_tokens=[a.max_tokens - r["prefix_tok"] for r in batch])
        for r, o in zip(batch, outs):
            closed = "</think>" in o["text"]
            rem = len(H.text_ids(tok, o["text"].split("</think>", 1)[0])) if closed else o["n_tokens"]
            full = tok.decode(r["prefix_ids"]) + o["text"]
            ans = P.strict_answer(full, r["kind"], r["prefix_tok"] + o["n_tokens"], a.max_tokens, o["finish_reason"] == "stop")
            answer = None if ans["status"] != "valid" or ans["answer"] is None else str(ans["answer"])
            row = {k: r[k] for k in ("prefix_src", "id", "component", "cut", "seed", "trace_seed", "prefix_tok",
                                     "trace_think_tok", "trace_answer")}
            row.update({"cont_ckpt": ck, "text": o["text"], "n_tokens": o["n_tokens"], "finish_reason": o["finish_reason"],
                        "closed": closed, "remaining_think_tok": rem, "status": ans["status"], "answer": answer,
                        "answer_changed": None if answer is None or r["trace_answer"] is None else answer != r["trace_answer"]})
            gen.add(key(r), row, (time.time() - t0) / len(batch))
        R.log(f"PH gen {len(gen.done)}/{len(reqs)}")
        res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
    res.now("done")
    out = [gen.done[key(r)] for r in reqs]
    loc = st.write(0, 1, out, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "skipped_cuts": json.dumps(skipped)})
    print(f"wrote {loc}: {len(out)} rows")
    return 0
