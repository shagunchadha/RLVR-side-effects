"""M6: representation similarity, linear CKA per layer (registration amendments 3, 13). One process, models in turn.

    python -m exp.run --job M6 [--layers 4 11 15 19 23 29] [--think 64] [--limit N]

Inputs (matched token ids across models): the 48 A wordings and B2's self-determinate (40), other-indeterminate (40)
and other-determinate (40, = D) cells. DPO and RLVR use their own (byte-identical) templates; SFT uses the DPO
system prompt (--system dpo_default), so SFT <-> DPO is also compared on identical ids. The first --think think
tokens are DPO's greedy continuation of each prompt (generated once, in the DPO pass), teacher-forced into every model.
Per model: residuals at the last prompt token (n_prompts x d) and at the --think think positions (n_prompts*think x d),
per layer, saved under $M6_WORK (default /tmp/m6_work, outside $RESULTS_DIR).
CKA (linear, centred; Kornblith et al. 2019) per layer and set for DPO<->RLVR and SFT<->DPO; bootstrap over prompts
(1,000 resamples, seed 20261024; tokens resampled with their prompt) for the A - D difference.
Columns: pair, set, position, layer, cka, and for set 'A-D': diff, ci_lo, ci_hi.
"""
import os
from pathlib import Path

import numpy as np

from .. import config as C
from .. import harness as H
from .. import prompts_sprint as PS
from .. import resources as R
from .. import store as S
from .M5_causal import resid_at, _free

LAYERS = (4, 11, 15, 19, 23, 29)
SETS = ("A", "self_det", "other_indet", "other_det")


def add_args(p):
    p.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    p.add_argument("--think", type=int, default=64)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)


def prompts():
    from .B2_indeterminacy import load_items
    out = [("A", it["id"], it["messages"]) for it in C.load_items("A")]
    out += [(it["cell"], it["id"], it["messages"]) for it in load_items()[0] if it["cell"] != "self_indet"]
    return out


def linear_cka(X, Y):
    """Linear CKA from the feature matrices (rows = observations)."""
    return cka_gram(X @ X.T, Y @ Y.T)


def cka_gram(K, L):
    """Linear CKA from Gram matrices (equal to linear_cka on the features); cheap to bootstrap by indexing K and L."""
    c = lambda M: M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()  # noqa: E731  (H M H, O(n^2))
    Kc, Lc = c(K), c(L)
    return float((Kc * Lc).sum() / np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()))


def job_config(a):
    return {"job": "M6", "models": {k: list(C.ckpt(k)) for k in C.THINK_LADDER}, "sft_system": "dpo_default",
            "layers": list(a.layers), "think": a.think, "boot": a.boot, "sets": list(SETS),
            "think_tokens": "DPO greedy continuation, teacher-forced into all models", "version": 1}


def run(args, load=None, tok=None):
    import torch
    a = args.job_args
    st = S.Store("M6", "ladder", job_config(a))
    if st.done(0):
        print("M6 exists; skipping")
        return 0
    load = load or (lambda ck: H.load_hf(*C.ckpt(ck)))
    tok = tok or H.load_tokenizer(*C.ckpt("dpo"))
    ps = prompts()[: a.limit] if a.limit else prompts()
    work = Path(os.environ.get("M6_WORK", "/tmp/m6_work")) / st.hash
    work.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for ck in ("dpo", "rlvr", "sft"):
        f = work / f"{ck}.npz"
        if f.exists():
            continue
        R.preflight(2 * 15, dirs=[cache])
        m = load(ck)
        res.now(f"{ck} loaded")
        if ck == "dpo" and not (work / "think_ids.npy").exists():
            ids = []
            for _, _, msgs in ps:
                p = H.prompt_ids(tok, msgs)
                with torch.no_grad():
                    o = m.generate(input_ids=torch.tensor([p], device=m.device), do_sample=False,
                                   max_new_tokens=a.think, min_new_tokens=a.think, pad_token_id=pad)
                ids.append(o[0, len(p):len(p) + a.think].tolist())
            np.save(work / "think_ids.npy", np.array(ids, dtype=np.int64))
        think = np.load(work / "think_ids.npy")
        pe = {li: [] for li in a.layers}
        th = {li: [] for li in a.layers}
        for (_, _, msgs), t in zip(ps, think):
            p = H.prompt_ids(tok, PS.with_system(msgs, "dpo_default") if ck == "sft" else msgs)
            r = resid_at(m, p + t.tolist(), [len(p) - 1] + list(range(len(p), len(p) + len(t))), a.layers)
            for li in a.layers:
                pe[li].append(r[li][0].astype(np.float16))
                th[li].append(r[li][1:].astype(np.float16))
        np.savez(f, **{f"pe_{li}": np.stack(pe[li]) for li in a.layers}, **{f"th_{li}": np.stack(th[li]) for li in a.layers})
        del m
        _free()
    Z = {ck: np.load(work / f"{ck}.npz") for ck in ("dpo", "rlvr", "sft")}
    sets = np.array([s for s, _, _ in ps])
    rng = np.random.default_rng(20261024)
    rows = []
    for pair in (("dpo", "rlvr"), ("sft", "dpo")):
        for pos in ("pe", "th"):
            for li in a.layers:
                X, Y = Z[pair[0]][f"{pos}_{li}"].astype(np.float64), Z[pair[1]][f"{pos}_{li}"].astype(np.float64)
                grams = {}
                for s in SETS:
                    idx = np.where(sets == s)[0]
                    if len(idx) < 3:
                        continue
                    Xs, Ys = X[idx].reshape(-1, X.shape[-1]), Y[idx].reshape(-1, Y.shape[-1])
                    grams[s] = (Xs @ Xs.T, Ys @ Ys.T, len(idx))
                    rows.append({"pair": "-".join(pair), "set": s, "position": pos, "layer": li,
                                 "cka": cka_gram(grams[s][0], grams[s][1])})
                if "A" in grams and "other_det" in grams:
                    t = 1 if pos == "pe" else X.shape[1]          # tokens per prompt (rows resampled with their prompt)

                    def boot(s):
                        K, L, n = grams[s]
                        pick = rng.integers(0, n, n)
                        rr = (pick[:, None] * t + np.arange(t)[None, :]).reshape(-1)
                        return cka_gram(K[np.ix_(rr, rr)], L[np.ix_(rr, rr)])
                    bs = [boot("A") - boot("other_det") for _ in range(a.boot)]
                    rows.append({"pair": "-".join(pair), "set": "A-D", "position": pos, "layer": li, "cka": None,
                                 "diff": cka_gram(*grams["A"][:2]) - cka_gram(*grams["other_det"][:2]),
                                 "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))})
                res.maybe(f"cka {pair} {pos} L{li}")
    res.now("done")
    loc = st.write(0, 1, rows, model_repo=";".join(C.ckpt(k)[0] for k in C.THINK_LADDER),
                   model_sha=";".join(C.ckpt(k)[1] for k in C.THINK_LADDER), extra_meta={"env_report": args.env_report or {}})
    print(f"wrote {loc}: {len(rows)} rows")
    return 0
