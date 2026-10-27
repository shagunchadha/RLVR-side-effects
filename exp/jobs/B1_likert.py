"""B1: Likert resampling (48 A wordings x 20 seeds per checkpoint) with think-conditioned and no-think readouts.

    python -m exp.run --job B1 --ckpt rlvr [--shard i/n] [--seeds 20] [--system own|dpo_default]
                      [--temperature T --top-p P] [--max-tokens 16000] [--gpu-mem 0.45]

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

from .. import config as C
from .. import harness as H
from .. import prompts_sprint as PS
from .. import store as S
from ._gen import run_requests


def add_args(p):
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--system", default="own", choices=PS.SYSTEM_MODES)
    p.add_argument("--temperature", type=float, default=None, help="override generation_config.json")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.45, help="vLLM gpu_memory_utilization (HF readout model shares the GPU)")
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
    items = S.shard_slice(C.load_items("A"), i, n)[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    if llm is None and generate_fn is None:
        llm = H.load_vllm(repo, sha, max_model_len=a.max_tokens + 2048, gpu_memory_utilization=a.gpu_mem)
    reqs = [{"ckpt": ck, "system": a.system, "id": it["id"], "trait": it["trait"], "key": it["key"],
             "original": it["original"], "seed": s, "kind": "likert",
             "messages": PS.with_system(it["messages"], a.system)} for it in items for s in range(a.seeds)]
    rows = run_requests(llm, tok, reqs, sampling, a.max_tokens, generate_fn)
    model = model or H.load_hf(repo, sha)
    msgs = {it["id"]: PS.with_system(it["messages"], a.system) for it in items}
    prior = {}
    for it in items:
        ro = H.likert_readout(model, tok, msgs[it["id"]], think_text=None)
        cols, _ = readout_cols("p_nothink", ro, "nothink_alt")
        cols["nothink_primary"] = cols.pop("p_nothink_primary")
        prior[it["id"]] = cols
    out = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("kind", "think_text", "post_text")}
        row["sampling"] = json.dumps(sampling, default=str)
        row["answer_sampled"] = r["answer_strict"] if r["status"] == "valid" else None
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
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
        row.update(cols)
        row.update(prior[r["id"]])
        out.append(row)
    loc = st.write(i, n, out, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "n_items": len(items)})
    print(f"wrote {loc}: {len(out)} rows")
    return 0
