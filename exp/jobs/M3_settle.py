"""M3: when is the Likert answer settled? (registration amendment 3). One checkpoint per process, HF model alone.

    python -m exp.run --job M3 --ckpt rlvr [--layers 4 11 15 19 23 29] [--limit N]

Traces: the valid B1 main-arm generations of this checkpoint (own template, model-default sampling), read from the
B1 progress files under $RESULTS_DIR/B1/ (with STORE=hf they are first pulled from the HF dataset's progress/B1/).
(a) Settle point: the think text is re-tokenised (as in M2) and each think-block sentence end (parsers'
    sentence splitter) is mapped to the token prefix ending at it. One cached forward over prompt + think ids; then,
    per sentence end k (descending), the cache is cropped to the prompt + k think tokens and B1's readout suffix
    ("\\n</think>\\n\\nANSWER: ") is stepped; P(1..5) = the five digit tokens' probabilities, renormalised (mass
    kept). Settle sentence = first sentence end after which the argmax equals the final sampled answer and stays
    there; share_after = think tokens after it / think tokens (0 and settled=False if the last argmax differs).
(b) Probes: residual stream after layers --layers at the last prompt token and at 10/25/50% of the think block, from
    the same forward. At the end of the job: multinomial logistic regression (scikit-learn, standardised, C=1.0)
    predicting the final sampled answer, 5-fold CV grouped by wording (folds by a seeded shuffle of wordings);
    accuracy and the majority-class baseline per (position, layer). Only the accuracies are stored (shard metadata
    and probe.json); activations stay on the notebook
    ($M3_FEATS_DIR, default /tmp/m3_feats, outside $RESULTS_DIR).
Resource safety: pre-flight, RES lines, per-trace progress file $RESULTS_DIR/M3/<arm>/progress_shard0of1.csv
(restart resumes; activations are recomputed for finished traces only if the probe file is missing).

Columns: ckpt, id, seed, trait, key, answer, n_think_tok, sent_tok, p_sent (list of [p1..p5]), mass_sent,
argmax_sent, settle_sent, settled, share_after, e_last (E[rating] of the full-think readout; compared with B1's
readout in the analysis as a fidelity check). M3_B1_ROOT overrides where the B1 progress files are read (smoke runs).
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import config as C
from .. import harness as H
from .. import parsers as P
from .. import resources as R
from .. import store as S

LAYERS = (4, 11, 15, 19, 23, 29)
POS = ("prompt_end", "think10", "think25", "think50")


def add_args(p):
    p.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    p.add_argument("--limit", type=int, default=None)


def b1_traces(ck):
    """Valid B1 main-arm generations (seeds 0..19, own template) of checkpoint ck from the B1 progress files."""
    from ..analysis.b1_summary import read_progress
    root = Path(os.environ.get("M3_B1_ROOT") or os.environ.get("RESULTS_DIR", "sprint_out"))
    if os.environ.get("STORE", "local") == "hf":
        from .. import hfsync
        hfsync.pull(root, os.environ.get("HF_REPO", S.DEFAULT_HF_REPO), jobs=["B1"])
    best = {}
    for d in sorted((root / "B1").glob(f"{ck}__sys-own__*")):
        g = read_progress(d / "gen_shard0of1.csv")
        if g and max(r["seed"] for r in g.values()) >= 5 and len(g) > len(best):   # main arm (20 seeds), not the bridge
            best = g
    rows = [r for r in best.values() if r["status"] == "valid" and r.get("think_text") and r.get("answer_strict") is not None]
    return sorted(rows, key=lambda r: (r["id"], r["seed"]))


def settle(argmax, final):
    """Index of the first sentence end after which argmax == final for good; None if the last one differs."""
    if not argmax or argmax[-1] != final:
        return None
    s = len(argmax) - 1
    while s > 0 and argmax[s - 1] == final:
        s -= 1
    return s


def trace_pass(model, tok, item, r, digits, suffix_ids, layers):
    import torch
    from transformers import DynamicCache
    pids = H.prompt_ids(tok, item["messages"])
    think = r["think_text"].rstrip("\n")
    enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
    tids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    ks = []
    for c in P.sentence_end_char_offsets(think):
        k = sum(1 for a, b in offs if b <= c)
        if 0 < k <= len(tids) and (not ks or k > ks[-1]):
            ks.append(k)
    if not ks or ks[-1] != len(tids):
        ks.append(len(tids))                                     # the full think block is the last readout
    n = len(tids)
    ids = pids + tids
    gs = len(pids)
    want = {"prompt_end": gs - 1, "think10": gs + max(int(0.10 * n), 1) - 1, "think25": gs + max(int(0.25 * n), 1) - 1,
            "think50": gs + max(int(0.50 * n), 1) - 1}
    with torch.no_grad():
        cache = DynamicCache()
        out = model.model(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
        feats = {p: {li: out.hidden_states[li + 1][0, j].float().cpu().numpy().astype(np.float16) for li in layers}
                 for p, j in want.items()}
        del out
        probs, mass = {}, {}
        for k in sorted(ks, reverse=True):
            extra = cache.get_seq_length() - (gs + k)
            if extra > 0:
                cache.crop(-extra)
            o = model(input_ids=torch.tensor([suffix_ids], device=model.device), past_key_values=cache, use_cache=True)
            pv = torch.softmax(o.logits[0, -1].float(), dim=-1)[digits].cpu().numpy()
            mass[k] = float(pv.sum())
            probs[k] = (pv / pv.sum()).tolist()
    p_sent = [probs[k] for k in ks]
    argmax = [int(np.argmax(p)) + 1 for p in p_sent]
    final = int(r["answer_strict"])
    s = settle(argmax, final)
    e_last = float(np.dot(p_sent[-1], np.arange(1, 6)))
    row = {"ckpt": r["ckpt"], "id": r["id"], "seed": r["seed"], "trait": item["trait"], "key": item["key"],
           "answer": final, "n_think_tok": n, "sent_tok": ks, "p_sent": [[round(x, 5) for x in p] for p in p_sent],
           "mass_sent": [round(mass[k], 5) for k in ks], "argmax_sent": argmax, "settle_sent": s, "settled": s is not None,
           "share_after": (n - ks[s]) / n if s is not None else 0.0,
           "e_last": e_last}
    return row, feats


def fit_probes(feats_by_key, rows, layers, seed=20261024):
    """{position: {layer: (cv accuracy, majority baseline)}}; 5-fold CV grouped by wording."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keys = [k for k in (f"{r['id']}|{r['seed']}" for r in rows) if k in feats_by_key]
    y = np.array([int(k2["answer"]) for k2 in rows if f"{k2['id']}|{k2['seed']}" in feats_by_key])
    words = sorted({k.split("|")[0] for k in keys})
    rng = np.random.default_rng(seed)
    fold_of = {w: i % 5 for i, w in enumerate(rng.permutation(words))}
    folds = np.array([fold_of[k.split("|")[0]] for k in keys])
    out = {}
    for p in POS:
        out[p] = {}
        for li in layers:
            X = np.stack([feats_by_key[k][p][li].astype(np.float32) for k in keys])
            correct, base = 0, 0
            for f in range(5):
                tr, te = folds != f, folds == f
                if te.sum() == 0 or len(set(y[tr])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                clf.fit(X[tr], y[tr])
                correct += int((clf.predict(X[te]) == y[te]).sum())
                vals, cnt = np.unique(y[tr], return_counts=True)
                base += int((y[te] == vals[np.argmax(cnt)]).sum())
            out[p][li] = (correct / len(y), base / len(y))
    return out


def job_config(ck, a):
    return {"job": "M3", "ckpt": list(C.ckpt(ck)), "traces": "B1 main arm (own template, model defaults), valid",
            "readout_suffix": H.ANSWER_SUFFIX, "layers": list(a.layers), "positions": list(POS),
            "probe": "sklearn LogisticRegression(C=1.0) on standardised residuals, 5-fold grouped by wording, seed 20261024",
            "version": 1}


def run(args, model=None, tok=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1:
        raise SystemExit("M3 needs exactly one --ckpt")
    ck = args.ckpt[0]
    repo, sha = C.ckpt(ck)
    st = S.Store("M3", ck, job_config(ck, a))
    if st.done(0):
        print(f"M3 {ck} exists; skipping")
        return 0
    rows_b1 = b1_traces(ck)[: a.limit]
    if not rows_b1:
        raise SystemExit(f"M3 {ck}: no valid B1 main-arm generations found under $RESULTS_DIR/B1")
    items = {it["id"]: it for it in C.load_items("A")}
    tok = tok or H.load_tokenizer(repo, sha)
    digits = [H.single_token_id(tok, str(d)) for d in range(1, 6)]
    suffix_ids = H.text_ids(tok, H.ANSWER_SUFFIX)
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M3" / f"{ck}__{st.hash}"
    prog = R.Progress(root / "progress_shard0of1.csv", st.hash)
    # activations (~190 MB per checkpoint) stay outside $RESULTS_DIR so the HF progress mirror never uploads them
    fpath = Path(os.environ.get("M3_FEATS_DIR", "/tmp/m3_feats")) / f"{ck}__{st.hash}.npz"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    feats = {}
    if fpath.exists():
        z = np.load(fpath, allow_pickle=True)
        feats = z["feats"].item()
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    key = lambda r: f"{r['id']}|{r['seed']}"  # noqa: E731
    todo = [r for r in rows_b1 if key(r) not in prog.done or key(r) not in feats]
    R.log(f"M3 {ck}: {len(rows_b1)} valid B1 traces, {len(rows_b1) - len(todo)} done")
    if todo and model is None:
        R.preflight(2 * 15, dirs=[cache])
        model = H.load_hf(repo, sha)
        res.now("HF model loaded")
    for t, r in enumerate(todo):
        t0 = time.time()
        row, f = trace_pass(model, tok, items[r["id"]], r, digits, suffix_ids, a.layers)
        feats[key(r)] = f
        if key(r) not in prog.done:
            prog.add(key(r), row, time.time() - t0)
        if (t + 1) % 50 == 0:
            R.log(f"M3 {ck} {t + 1}/{len(todo)}")
            np.savez(fpath, feats=np.array(feats, dtype=object))
        res.maybe(f"{t + 1}/{len(todo)}")
    np.savez(fpath, feats=np.array(feats, dtype=object))
    rows = [prog.done[key(r)] for r in rows_b1]
    probes = fit_probes(feats, rows, a.layers)
    (root / "probe.json").write_text(json.dumps({p: {str(li): v for li, v in d.items()} for p, d in probes.items()}, indent=1))
    res.now("done")
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "probes": probes})
    print(f"wrote {loc}: {len(rows)} rows; probe accuracy at prompt end: "
          + ", ".join(f"L{li} {v[0]:.2f} (base {v[1]:.2f})" for li, v in probes["prompt_end"].items()))
    return 0
