"""B2: indeterminacy vs self-reference (registration amendment 3). 2 x 2 grid from items_sprint/items_B2.json
(self/other x indeterminate/determinate, frozen with sha256 before any run), 10 seeds, one checkpoint per process.

    python -m exp.run --job B2 --ckpt rlvr [--shard i/n] [--seeds 10] [--max-tokens 16000] [--gpu-mem 0.85]

Prompts: frozen strict format with the reasoning cap (Likert cells), the frozen D messages (other_det), and the same
format with a yes/no field (self_det; parsers.strict_yesno). Own templates, model-default sampling.
Resource safety as B3: vLLM alone (gpu_memory_utilization 0.85, max_model_len = longest prompt + max_tokens, no CPU
offload), every finished generation appended to $RESULTS_DIR/B2/<arm>/gen_shard<i>of<n>.csv (restart resumes), the
shard parquet written when all generations exist. Pre-flight and RES lines.

Columns: ckpt, system, id, cell, self, indet, kind, seed, sampling (json), text, gen_token_ids, lp_chosen, n_tokens,
finish_reason, closed, status, answer_strict, gold, correct (determinate cells), think_tokens, think_sentences
"""
import hashlib
import json
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import parsers as P
from .. import prompts_sprint as PS
from .. import resources as R
from .. import store as S
from ._gen import run_requests

ITEMS = C.REPO_ROOT / "items_sprint" / "items_B2.json"


def add_args(p):
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--system", default="own", choices=PS.SYSTEM_MODES)
    p.add_argument("--chunk", type=int, default=8, help="items per vLLM call (progress granularity)")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=None)


def load_items():
    """Items, after checking the file against items_sprint/MANIFEST.sha256 (frozen before any run)."""
    h = hashlib.sha256(ITEMS.read_bytes()).hexdigest()
    man = dict(l.split()[::-1] for l in (ITEMS.parent / "MANIFEST.sha256").read_text().splitlines() if l.strip())
    if man.get(ITEMS.name) != h:
        raise SystemExit(f"{ITEMS} sha256 {h} != manifest {man.get(ITEMS.name)}")
    return json.load(open(ITEMS, encoding="utf-8"))["items"], h


def job_config(ck, a, sampling, items_sha):
    return {"job": "B2", "ckpt": list(C.ckpt(ck)), "system": a.system, "seeds": a.seeds, "sampling": sampling,
            "max_tokens": a.max_tokens, "items": "items_sprint/items_B2.json", "items_sha256": items_sha,
            "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}


def run(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("B2 needs exactly one --ckpt")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    i, n = S.parse_shard(args.shard)
    items, items_sha = load_items()
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store("B2", f"{ck}__sys-{a.system}", job_config(ck, a, sampling, items_sha))
    if st.done(i):
        print(f"B2 {ck} shard {i}/{n} exists; skipping")
        return 0
    us = S.shard_slice(items, i, n)[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    reqs = [{"id": u["id"], "cell": u["cell"], "self": u["self"], "indet": u["indet"], "kind": u["kind"],
             "gold": u.get("gold"), "ckpt": ck, "system": a.system, "seed": s,
             "messages": PS.with_system(u["messages"], a.system)} for u in us for s in range(a.seeds)]
    key = lambda r: f"{r['id']}|{r['seed']}"  # noqa: E731
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B2" / f"{ck}__sys-{a.system}__{st.hash}"
    gen = R.Progress(root / f"gen_shard{i}of{n}.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    R.log(f"B2 {ck} shard {i}/{n}: {len(reqs)} requests, {len(reqs) - len(todo)} done; sampling {sampling}")
    if todo and llm is None and generate_fn is None:
        R.preflight(2 * 15, dirs=[cache])
        mpl = max(len(H.prompt_ids(tok, r["messages"])) for r in reqs) + a.max_tokens
        llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
        res.now(f"vLLM loaded (max_model_len {mpl})")
    chunk = a.chunk * a.seeds
    for c in range(0, len(todo), chunk):
        batch = todo[c:c + chunk]
        t0 = time.time()
        for row in run_requests(llm, tok, batch, sampling, a.max_tokens, generate_fn):
            row["lp_chosen"] = [round(x, 5) for x in row["lp_chosen"]]
            gen.add(key(row), row, (time.time() - t0) / len(batch))
        R.log(f"B2 gen {len(gen.done)}/{len(reqs)}")
        res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
    res.now("done")
    out = []
    for rq in reqs:
        r = gen.done[key(rq)]
        row = {k: v for k, v in r.items() if k not in ("think_text", "post_text")}
        row["sampling"] = json.dumps(sampling, default=str)
        ans = r["answer_strict"] if r["status"] == "valid" else None
        row["answer_strict"] = None if ans is None else str(ans)
        row["correct"] = None if (r.get("gold") is None or ans is None) else bool(str(ans) == r["gold"])
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
        row["think_sentences"] = len(P.sentences(r["think_text"])) if r["think_text"] is not None else None
        out.append(row)
    loc = st.write(i, n, out, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "n_items": len(us)})
    print(f"wrote {loc}: {len(out)} rows")
    return 0
