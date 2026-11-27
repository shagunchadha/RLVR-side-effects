"""B7: self-report vs behaviour (registration amendment 9). Behavioural part: hedging on open factual questions.

    python -m exp.run --job B7 --ckpt rlvr [--system own|dpo_default] [--seeds 5] [--limit N]

Items: items_sprint/items_B7_trivia.json (100 TriviaQA rc.nocontext validation questions, frozen), prompt = question +
' Answer briefly.' + the reasoning cap. Arms (amendment 9): RLVR own, DPO own, SFT with the DPO system prompt.
Per generation: hedge-word rate per 100 words in the final answer (text after </think>) and in the think block, with
the fixed lexicon below (word-boundary, case-insensitive, multi-word phrases counted once each), and correctness
(normalised alias contained in the normalised final answer). Stated confidence comes from B6's MMLU-Pro outputs and
self-reports from B1 / B8 at analysis time; this job only produces the behavioural rows.
"""
import json
import os
import re
import string
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import prompts_sprint as PS
from .. import resources as R
from .. import store as S
from ._gen import run_requests

HEDGES = ("maybe", "perhaps", "probably", "possibly", "likely", "i think", "i believe", "not sure", "uncertain", "might",
          "could be", "approximately", "around")
_HEDGE_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.I)


def add_args(p):
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--system", default="own", choices=PS.SYSTEM_MODES)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)


def hedge_rate(text):
    words = len(re.findall(r"\b\w+\b", text or ""))
    return (100.0 * len(_HEDGE_RE.findall(text or "")) / words) if words else None


def norm(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_items():
    import hashlib
    f = C.REPO_ROOT / "items_sprint" / "items_B7_trivia.json"
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    man = dict(l.split()[::-1] for l in (f.parent / "MANIFEST.sha256").read_text().splitlines() if l.strip())
    if man.get(f.name) != h:
        raise SystemExit(f"{f} sha256 {h} != manifest {man.get(f.name)}")
    return json.load(open(f, encoding="utf-8"))["items"], h


def run(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("B7 needs exactly one --ckpt")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    items, isha = load_items()
    items = items[: a.limit]
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if generate_fn is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    cfg = {"job": "B7", "ckpt": [repo, sha], "system": a.system, "seeds": a.seeds, "sampling": sampling,
           "max_tokens": a.max_tokens, "items_sha256": isha, "hedges": list(HEDGES),
           "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}
    st = S.Store("B7", f"{ck}__sys-{a.system}", cfg)
    if st.done(0):
        print(f"B7 {ck} exists; skipping")
        return 0
    tok = tok or H.load_tokenizer(repo, sha)
    reqs = [{"id": it["id"], "kind": "free", "seed": s, "messages": PS.with_system(it["messages"], a.system)}
            for it in items for s in range(a.seeds)]
    by = {it["id"]: it for it in items}
    key = lambda r: f"{r['id']}|{r['seed']}"  # noqa: E731
    gen = R.Progress(Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B7" / f"{ck}__sys-{a.system}__{st.hash}" /
                     "gen_shard0of1.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    if todo and llm is None and generate_fn is None:
        R.preflight(2 * 15, dirs=[cache])
        mpl = max(len(H.prompt_ids(tok, r["messages"])) for r in reqs) + a.max_tokens
        llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
        res.now(f"vLLM loaded (max_model_len {mpl})")
    for c in range(0, len(todo), a.chunk * a.seeds):
        batch = todo[c:c + a.chunk * a.seeds]
        t0 = time.time()
        for r, row in zip(batch, run_requests(llm, tok, batch, sampling, a.max_tokens, generate_fn)):
            post, think = row["post_text"], row["think_text"]
            final = post if post is not None else ""
            gen.add(key(r), {"ckpt": ck, "system": a.system, "id": r["id"], "seed": r["seed"], "text": row["text"],
                             "closed": row["closed"], "n_tokens": row["n_tokens"],
                             "hedge_final": hedge_rate(final) if post is not None else None,
                             "hedge_think": hedge_rate(think) if think is not None else None,
                             "think_tokens": len(H.text_ids(tok, think)) if think is not None else None,
                             "correct": (any(al and al in norm(final) for al in by[r["id"]]["aliases"])
                                         if post is not None else None)}, (time.time() - t0) / len(batch))
        R.log(f"B7 gen {len(gen.done)}/{len(reqs)}")
        res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
    rows = [gen.done[key(r)] for r in reqs]
    res.now("done")
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha, extra_meta={"env_report": args.env_report or {}})
    print(f"wrote {loc}: {len(rows)} rows")
    return 0
