"""B5: dose-response over RL steps (registration amendments 3, 5; operational details amendment 8).

    python -m exp.run --job B5 --ckpt rlvr@step_0025 --part A [--phase gen|read] [--limit N]   B1 protocol, 10 seeds
    python -m exp.run --job B5 --ckpt rlvr@step_0025 --part D [--limit N]                      100 D items, 10 seeds
    python -m exp.run --job B5 --verify-last                                                   step_1375 vs main

Checkpoints: branches of allenai/Olmo-3-7B-Think, resolved to the commits in the B0 inventory (config.ckpt
'rlvr@<branch>'). Sampling is fixed to main's generation_config (temperature 0.6, top_p 0.95, no top-k) for every
step: the branches carry a different generation_config.json (amendment 8).
  part A: exactly the B1 protocol (exp/jobs/B1_likert.py: vLLM generation, then HF think-conditioned and no-think
          readouts) on the 48 A wordings, 10 seeds; store job 'B5'.
  part D: the first 100 D base items (o0) with their frozen messages (strict format + reasoning cap), 10 seeds,
          vLLM only; think tokens, validity (frozen parser) and accuracy; store job 'B5D'.
  --verify-last: streams model-0000k-of-00003.safetensors of step_1375 and of main one file at a time and compares
          every tensor (exact equality, max |diff|); writes $RESULTS_DIR/B5/verify_last.json.
"""
import json
import os
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import resources as R
from .. import store as S
from . import B1_likert
from ._gen import run_requests

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "main generation_config.json (fixed, amendment 8)"}
STEPS8 = ("step_0025", "step_0050", "step_0075", "step_0150", "step_0250", "step_0425", "step_0775", "step_1375")
STEPS5 = ("step_0025", "step_0075", "step_0175", "step_0500", "step_1375")


def add_args(p):
    B1_likert.add_args(p)
    p.set_defaults(seeds=10, temperature=0.6, top_p=0.95)
    p.add_argument("--part", default="A", choices=["A", "D"])
    p.add_argument("--n-d", type=int, default=100)
    p.add_argument("--verify-last", action="store_true")


def d_items(n):
    return sorted((x for x in C.load_items("D") if x["order"] == 0), key=lambda x: x["draw_index"])[:n]


def verify_last():
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    repo = C.ckpt("rlvr")[0]
    revs = {"main": C.ckpt("rlvr")[1], "step_1375": C.ckpt("rlvr@step_1375")[1]}
    work = Path(os.environ.get("B5_DL", "/tmp/b5_dl"))       # outside $RESULTS_DIR: never mirrored to HF
    out = {"n_tensors": 0, "n_equal": 0, "max_abs_diff": 0.0, "files": {}}
    for k in (1, 2, 3):
        name = f"model-0000{k}-of-00003.safetensors"
        paths = {r: hf_hub_download(repo, name, revision=sha, cache_dir=str(work / r)) for r, sha in revs.items()}
        with safe_open(paths["main"], "pt") as fa, safe_open(paths["step_1375"], "pt") as fb:
            keys = sorted(fa.keys())
            assert keys == sorted(fb.keys()), f"{name}: tensor names differ"
            for t in keys:
                a, b = fa.get_tensor(t), fb.get_tensor(t)
                out["n_tensors"] += 1
                if a.shape == b.shape and bool((a == b).all()):
                    out["n_equal"] += 1
                else:
                    out["max_abs_diff"] = max(out["max_abs_diff"], float((a.float() - b.float()).abs().max()))
        out["files"][name] = "compared"
        for r in revs:
            import shutil
            shutil.rmtree(work / r, ignore_errors=True)
        R.log(f"B5 verify-last: {name} done ({out['n_equal']}/{out['n_tensors']} tensors equal so far)")
    out["identical"] = out["n_equal"] == out["n_tensors"]
    dst = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B5" / "verify_last.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("B5 verify-last:", json.dumps(out))
    return 0


def run_d(args, llm=None, tok=None, generate_fn=None):
    a = args.job_args
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    sampling = SAMPLING if generate_fn is None else {**SAMPLING, "source": "test"}
    cfg = {"job": "B5D", "ckpt": [repo, sha], "branch": ck, "seeds": a.seeds, "sampling": sampling,
           "max_tokens": a.max_tokens, "n_d": a.n_d, "items": "items/items_D_csqa.json o0, first by draw_index",
           "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}
    st = S.Store("B5D", ck, cfg)
    if st.done(0):
        print(f"B5D {ck} exists; skipping")
        return 0
    items = d_items(a.n_d)[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    reqs = [{"id": it["id"], "component": "D", "kind": "mc5", "gold": it["key"], "ckpt": ck, "seed": s,
             "messages": it["messages"]} for it in items for s in range(a.seeds)]
    key = lambda r: f"{r['id']}|{r['seed']}"  # noqa: E731
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B5D" / f"{ck}__{st.hash}"
    gen = R.Progress(root / "gen_shard0of1.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    R.log(f"B5D {ck}: {len(reqs)} requests, {len(reqs) - len(todo)} done; sampling {sampling}")
    if todo and llm is None and generate_fn is None:
        R.preflight(2 * 15, dirs=[cache])
        mpl = max(len(H.prompt_ids(tok, r["messages"])) for r in reqs) + a.max_tokens
        llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
        res.now(f"vLLM loaded (max_model_len {mpl})")
    chunk = 8 * a.seeds
    for c in range(0, len(todo), chunk):
        batch = todo[c:c + chunk]
        t0 = time.time()
        for row in run_requests(llm, tok, batch, sampling, a.max_tokens, generate_fn):
            row["lp_chosen"] = [round(x, 5) for x in row["lp_chosen"]]
            gen.add(key(row), row, (time.time() - t0) / len(batch))
        R.log(f"B5D gen {len(gen.done)}/{len(reqs)}")
        res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
    out = []
    for rq in reqs:
        r = gen.done[key(rq)]
        row = {k: v for k, v in r.items() if k not in ("think_text", "post_text", "gen_token_ids", "lp_chosen")}
        ans = r["answer_strict"] if r["status"] == "valid" else None
        row["answer_strict"] = None if ans is None else str(ans)
        row["correct"] = None if ans is None else bool(str(ans) == r["gold"])
        row["think_tokens"] = len(H.text_ids(tok, r["think_text"])) if r["think_text"] is not None else None
        row["sampling"] = json.dumps(sampling)
        out.append(row)
    res.now("done")
    loc = st.write(0, 1, out, model_repo=repo, model_sha=sha, extra_meta={"env_report": args.env_report or {}})
    print(f"wrote {loc}: {len(out)} rows")
    return 0


def run(args, **kw):
    a = args.job_args
    if a.verify_last:
        return verify_last()
    if not args.ckpt or len(args.ckpt) != 1 or not args.ckpt[0].startswith("rlvr@"):
        raise SystemExit("B5 needs exactly one --ckpt rlvr@<branch>")
    if a.part == "D":
        return run_d(args, **kw)
    if (a.temperature, a.top_p) != (0.6, 0.95):
        raise SystemExit("B5 fixes sampling to 0.6 / 0.95 (amendment 8)")
    return B1_likert.run(args, job="B5", items_desc="items/items_A_selfreport.json", **kw)
