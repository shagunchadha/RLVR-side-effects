"""M4: persona vectors (registration amendments 3, 12). One checkpoint per process, HF model alone; DPO first.

    python -m exp.run --job M4 --ckpt dpo [--layers 4 11 15 19 23 29] [--new-tokens 192] [--limit N]

Directions (every checkpoint, so that their stability can be compared): for each of the 10 traits in
items_sprint/items_M4_persona.json and each of the 40 Alpaca questions, one response under the positive and one under
the negative system prompt (HF generate, model-default sampling, --new-tokens new tokens, seed fixed per trait and
side); direction[trait, layer] = mean residual over response tokens (positive) - the same (negative). Saved as
$RESULTS_DIR/M4/<ckpt>/directions.npy (float32, traits x layers x d) with directions_meta.json.
Projections (every checkpoint, onto DPO's unit directions and onto its own): for the 48 A wordings, the residual at
the last prompt token (own template, no system prompt) and the mean residual over the first 64 think tokens of each
valid B1 main-arm trace of this checkpoint (seeds 0-9). Only summaries and the direction vectors are kept.
Columns: ckpt, id, trait_item, seed (None = prompt end), position, trait, layer, proj_dpo, proj_own
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import config as C
from .. import harness as H
from .. import resources as R
from .. import store as S
from .M3_settle import b1_traces
from .M5_causal import resid_at

LAYERS = (4, 11, 15, 19, 23, 29)


def add_args(p):
    p.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    p.add_argument("--new-tokens", type=int, default=192)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--seeds", type=int, default=10, help="B1 seeds used for the think-64 projections")
    p.add_argument("--limit", type=int, default=None, help="debug: first N questions / wordings")


def material():
    import hashlib
    f = C.REPO_ROOT / "items_sprint" / "items_M4_persona.json"
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    man = dict(l.split()[::-1] for l in (f.parent / "MANIFEST.sha256").read_text().splitlines() if l.strip())
    if man.get(f.name) != h:
        raise SystemExit(f"{f} sha256 {h} != manifest {man.get(f.name)}")
    return json.load(open(f, encoding="utf-8")), h


def mean_response_resid(model, tok, msgs_list, layers, new_tokens, sampling, seed, batch):
    """Generate one response per message list and return {layer: mean over prompts of the mean residual over the
    response tokens}."""
    import torch
    acc = {li: [] for li in layers}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for c in range(0, len(msgs_list), batch):
        chunk = msgs_list[c:c + batch]
        pids = [H.prompt_ids(tok, m) for m in chunk]
        L = max(len(p) for p in pids)
        inp = torch.tensor([[pad] * (L - len(p)) + p for p in pids], device=model.device)
        att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in pids], device=model.device)
        torch.manual_seed(seed * 1000 + c)
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=att, do_sample=True, temperature=sampling["temperature"],
                                 top_p=sampling["top_p"], max_new_tokens=new_tokens, pad_token_id=pad)
        for p, o in zip(pids, out[:, L:].tolist()):
            if tok.eos_token_id in o:
                o = o[:o.index(tok.eos_token_id)]
            if not o:
                continue
            r = resid_at(model, p + o, list(range(len(p), len(p) + len(o))), layers)
            for li in layers:
                acc[li].append(r[li].mean(0))
    return {li: np.mean(acc[li], axis=0) for li in layers}


def job_config(ck, a, mat_sha):
    return {"job": "M4", "ckpt": list(C.ckpt(ck)), "layers": list(a.layers), "new_tokens": a.new_tokens, "b1_seeds": a.seeds,
            "material_sha256": mat_sha, "directions_from": "dpo (projections also onto own)", "version": 1}


def run(args, model=None, tok=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1 or args.ckpt[0] not in C.THINK_LADDER:
        raise SystemExit("M4 needs exactly one --ckpt in sft / dpo / rlvr")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    mat, mat_sha = material()
    st = S.Store("M4", ck, job_config(ck, a, mat_sha))
    if st.done(0):
        print(f"M4 {ck} exists; skipping")
        return 0
    rroot = Path(os.environ.get("RESULTS_DIR", "sprint_out"))
    root = rroot / "M4"
    dpo_dir_f = root / "dpo" / "directions.npy"
    if ck != "dpo" and not dpo_dir_f.exists() and os.environ.get("STORE", "local") == "hf":
        from .. import hfsync
        hfsync.pull(rroot, os.environ.get("HF_REPO", S.DEFAULT_HF_REPO), jobs=["M4"])
    if ck != "dpo" and not dpo_dir_f.exists():
        raise SystemExit("M4 needs DPO's directions first (run --ckpt dpo)")
    tok = tok or H.load_tokenizer(repo, sha)
    sampling = H.resolve_sampling(repo, sha, {"temperature": None, "top_p": None, "top_k": None}) \
        if model is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    if model is None:
        R.preflight(2 * 15, dirs=[cache])
        model = H.load_hf(repo, sha)
        res.now("HF model loaded")
    traits = list(mat["traits"])
    qs = mat["questions"][: a.limit]
    own_f = root / ck / "directions.npy"
    own_f.parent.mkdir(parents=True, exist_ok=True)
    if not own_f.exists():
        D = np.zeros((len(traits), len(a.layers), model.config.hidden_size), dtype=np.float32)
        for ti, t in enumerate(traits):
            t0 = time.time()
            side = {}
            for k, sgn in (("pos", 0), ("neg", 1)):
                msgs = [[{"role": "system", "content": mat["traits"][t][k]}, {"role": "user", "content": q["text"]}] for q in qs]
                side[k] = mean_response_resid(model, tok, msgs, a.layers, a.new_tokens, sampling, 2 * ti + sgn, a.batch)
            for lj, li in enumerate(a.layers):
                D[ti, lj] = side["pos"][li] - side["neg"][li]
            R.log(f"M4 {ck} direction {t}: {time.time() - t0:.0f}s")
            res.maybe(t)
        np.save(own_f, D)
        (own_f.parent / "directions_meta.json").write_text(json.dumps({"traits": traits, "layers": list(a.layers),
                                                                       "n_questions": len(qs), "ckpt": ck}))
    own = np.load(own_f)
    dpo = own if ck == "dpo" else np.load(dpo_dir_f)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-12)  # noqa: E731
    Ud, Uo = unit(dpo), unit(own)
    prog = R.Progress(root / ck / f"proj__{st.hash}.csv", st.hash)
    A = C.load_items("A")[: a.limit]
    by = {it["id"]: it for it in A}
    units = [(it["id"], None) for it in A]
    trs = [r for r in b1_traces(ck) if r["seed"] < a.seeds and r["id"] in by]
    units += [(r["id"], r["seed"]) for r in trs]
    think = {(r["id"], r["seed"]): r["think_text"] for r in trs}
    for iid, seed in units:
        k = f"{iid}|{seed}"
        if k in prog.done:
            continue
        pids = H.prompt_ids(tok, by[iid]["messages"])
        if seed is None:
            r = resid_at(model, pids, [len(pids) - 1], a.layers)
            vec = {li: r[li][0] for li in a.layers}
        else:
            tids = H.text_ids(tok, think[(iid, seed)])[:64]
            r = resid_at(model, pids + tids, list(range(len(pids), len(pids) + len(tids))), a.layers)
            vec = {li: r[li].mean(0) for li in a.layers}
        V = np.stack([vec[li] for li in a.layers])                     # layers x d
        prog.add(k, {"pd": np.einsum("tld,ld->tl", Ud, V).round(5).tolist(),
                     "po": np.einsum("tld,ld->tl", Uo, V).round(5).tolist()}, 0.0)
        res.maybe(k)
    rows = []
    for iid, seed in units:
        x = prog.done[f"{iid}|{seed}"]
        for ti, t in enumerate(traits):
            for lj, li in enumerate(a.layers):
                rows.append({"ckpt": ck, "id": iid, "trait_item": by[iid]["trait"], "seed": seed,
                             "position": "prompt_end" if seed is None else "think64", "trait": t, "layer": li,
                             "proj_dpo": x["pd"][ti][lj], "proj_own": x["po"][ti][lj]})
    res.now("done")
    cos = {t: [float(np.dot(Ud[ti, lj], Uo[ti, lj])) for lj in range(len(a.layers))] for ti, t in enumerate(traits)}
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "cos_own_vs_dpo_direction": cos})
    print(f"wrote {loc}: {len(rows)} rows")
    return 0
