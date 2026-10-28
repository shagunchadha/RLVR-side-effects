"""M2: teacher-forced scoring of every frozen full-run trace under the three Think checkpoints.

Modes
  single  python -m exp.run --job M2 --ckpt rlvr [--shard i/n] [--traces sft dpo rlvr]
          scores every trace of the chosen trace sets under ONE scorer model -> per-position features
  pair    python -m exp.run --job M2 --pair rlvr dpo [--shard i/n]
          loads both models and stores per-position KL(P_rlvr || P_dpo) on the same token ids

Token ids: prompt = apply_chat_template(item messages) with the TRACE checkpoint's tokenizer/template (exactly as
scripts/full_run.py), checked against the frozen prompt_tokens; generation = tokenizer(raw_output) (+ EOS if the
frozen row ended on EOS). The same ids are fed to every scorer, so scorers are compared on identical tokens. The three
tokenizers must have identical vocabularies (checked at start).

Indexing convention for the per-position lists: element j is the scorer's prediction FOR generation token j (made from
the prefix ending at generation token j-1). sent_next_tok lists, for each sentence end inside the think block, the
index of the first non-whitespace generation token after it; P(</think> next) at that boundary is p_think_end[k] for
k in sent_next_tok (the think-closing token itself counts as a content token).

Stored per trace (changed Oct 28, before any M2 output, so that rows are ~2 KB and the progress file can live off the
notebook; the per-position values are computed exactly as before, only what is kept changed):
  think positions = generation positions before the think-closing token (all positions if the block never closed);
  mean_<f>_think   mean of feature f over think positions (f = lp_chosen, entropy, p_think_end, p_switch, kl);
  <f>_at_sent      f at each think-block sentence end (the prediction for the first content token after it, i.e. the
                   positions in sent_next_tok) - the unit of M2-H1..H3;
  p_close_at_sent  P(</think> starts here) at the same positions. '</think>' is not one token for the Olmo-3 tokenizer
                   (it is '</' 'think' '>...'), so (amendment 2): sum over v in {'</', ' </'} of P(v) * P('think' | v),
                   by harness.close_probs. p_think_end_* is the first-token proxy P('</') + P(' </'), sensitivity only;
  sum_lp_gen       total log-probability of the generation.
Columns (single): trace_ckpt, scorer, id, seed, component, status, n_tokens_frozen, n_prompt_tok, prompt_tok_match,
n_gen_tok, gen_tok_match, think_end_tok, n_think_pos, sent_end_tok, sent_next_tok, mean_*_think, *_at_sent (lp_chosen,
entropy, p_think_end, p_switch, p_switch_<word>), sum_lp_gen
Columns (pair): trace_ckpt, scorer_a, scorer_b, id, seed, component, status, n_gen_tok, think_end_tok, n_think_pos,
sent_next_tok, mean_kl_think, mean_kl_gen, kl_at_sent

Resource safety: pre-flight (RAM, disk >= 2 x model size), a RES line every 3 min, and a per-trace progress file
($RESULTS_DIR/M2/<label>/progress_shard<i>of<n>.csv) that a restart resumes from. Weights load into $HF_HUB_CACHE: the
launcher points it at a per-job dir and deletes it between scorer models.
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

LADDER = C.THINK_LADDER


def add_args(p):
    p.add_argument("--traces", nargs="*", default=list(LADDER), help="trace sets to score (default all three)")
    p.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"), help="KL mode: store KL(P_A || P_B)")
    p.add_argument("--components", nargs="*", default=list("ABCDE"))
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="debug: first N traces of this shard")
    p.add_argument("--wrapup", action="store_true",
                   help="single mode: also read P(conclusion opener) (amendment 4, F1); label <ckpt>_wrapup")
    p.add_argument("--kl-detail", action="store_true",
                   help="pair mode: also store KL attributed to the written token and a 20-bin positional profile")


def load_traces(trace_sets, components):
    out = []
    for tc in trace_sets:
        rows = json.load(open(C.FULL_RUN[tc], encoding="utf-8"))["rows"]
        for r in rows:
            if r["component"] in components:
                out.append((tc, r))
    out.sort(key=lambda x: (x[0], x[1]["component"], x[1]["id"], x[1]["seed"]))
    return out


def items_by_id():
    return {it["id"]: it for comp in "ABCDE" for it in C.load_items(comp)}


def prepare(tok, item, row, eos_id):
    """Token ids and alignment info for one frozen trace (pure tokenizer work; tested on CPU)."""
    pids = H.prompt_ids(tok, item["messages"])
    enc = tok(row["raw_output"], add_special_tokens=False, return_offsets_mapping=True)
    gids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    if row.get("hit_eos"):
        gids.append(eos_id)
        offs.append((len(row["raw_output"]), len(row["raw_output"])))
    text = row["raw_output"]
    think, _, closed = P.split_think(text)
    think_end_char = len(think) if closed else None
    tok_at = lambda c: next((j for j, (a, b) in enumerate(offs) if a <= c < b), None)  # noqa: E731
    think_end_tok = tok_at(think_end_char) if closed else None
    sent_end_tok, sent_next_tok = [], []
    for c in P.sentence_end_char_offsets(think):
        j = tok_at(c - 1) if c > 0 else None
        if j is None:
            continue
        k = j + 1
        while k < len(gids) and offs[k][1] > offs[k][0] and not text[offs[k][0]:offs[k][1]].strip() \
                and (think_end_tok is None or k < think_end_tok):
            k += 1
        if k < len(gids):
            sent_end_tok.append(j)
            sent_next_tok.append(k)
    return {"prompt_ids": pids, "gen_ids": gids, "n_prompt_tok": len(pids),
            "prompt_tok_match": len(pids) == row.get("prompt_tokens"), "n_gen_tok": len(gids),
            "gen_tok_match": len(gids) == row.get("n_tokens"), "think_end_tok": think_end_tok,
            "sent_end_tok": sent_end_tok, "sent_next_tok": sent_next_tok}


MODEL_GB = 15


def summarize(prep, out, pair=False):
    """Per-trace summary of the per-position lists returned by harness.score (see the module docstring)."""
    te = prep["think_end_tok"]
    n = len(out["lp_chosen"] if not pair else out["kl"])
    think = slice(0, te if te is not None else n)
    at = [k for k in prep["sent_next_tok"] if k < n]
    m = lambda xs: float(np.mean(xs)) if len(xs) else None  # noqa: E731
    feats = ["kl"] if pair else [k for k in out if k != "kl"]
    row = {"think_end_tok": te, "n_think_pos": think.stop, "sent_next_tok": at}
    for f in feats:
        v = out[f]
        row[f"mean_{f}_think"] = m(v[think])
        row[f"{f}_at_sent"] = [v[k] for k in at]
    if pair:
        row["mean_kl_gen"] = m(out["kl"])
    else:
        row["sum_lp_gen"] = float(np.sum(out["lp_chosen"]))
    return row


KL_BINS = 20
KL_TOP = 50


def kl_detail(prep, kl):
    """Pair mode, --kl-detail (Gate 1, exploratory): over think positions, KL summed by the token written there
    (the target of the prediction; top KL_TOP per trace as [token_id, sum_kl, count]) and the mean KL in KL_BINS
    equal bins of relative position in the think block."""
    te = prep["think_end_tok"]
    n = te if te is not None else len(kl)
    v, g = np.asarray(kl[:n], dtype=float), np.asarray(prep["gen_ids"][:n])
    by = {}
    for t, x in zip(g.tolist(), v.tolist()):
        a = by.setdefault(t, [0.0, 0])
        a[0] += x
        a[1] += 1
    top = sorted(by.items(), key=lambda kv: -kv[1][0])[:KL_TOP]
    bins = [float(b.mean()) if len(b) else None for b in np.array_split(v, KL_BINS)] if n else []
    return {"sum_kl_think": float(v.sum()), "kl_by_token_top": [[t, round(a[0], 6), a[1]] for t, a in top],
            "kl_pos_bins": bins}


WRAPUP_PRIMARY = ("So", "Therefore", "Thus", "Final")      # amendment 4, F1: primary group
WRAPUP_PROXY = {"illgowith": "I'll go with", "theanswer": "The answer"}   # first-token proxies, reported per word


def target_groups(tok, wrapup=False):
    groups = {"think_end": H.close_ids(tok, C.THINK_END)[0]}          # first-token proxy (sensitivity only)
    sw = H.first_token_ids(tok, C.SWITCH_WORDS)
    groups["switch"] = sorted({t for v in sw.values() for t in v})
    for w, ids in sw.items():
        groups[f"switch_{w}"] = ids
    if wrapup:
        wu = H.first_token_ids(tok, WRAPUP_PRIMARY)
        groups["wrapup"] = sorted({t for v in wu.values() for t in v})
        for w, ids in wu.items():
            groups[f"wrapup_{w}"] = ids
        for k, phrase in WRAPUP_PROXY.items():
            groups[f"wrapup_{k}"] = H.first_token_ids(tok, [phrase])[phrase]
    return groups


def job_config(a, scorer_desc):
    return {"job": "M2", "mode": "pair" if a.pair else "single", "scorer": scorer_desc, "traces": sorted(a.traces),
            "components": sorted(a.components), "trace_files": {t: str(C.FULL_RUN[t].relative_to(C.REPO_ROOT)) for t in a.traces},
            "ladder": {k: list(C.ckpt(k)) for k in LADDER}, "switch_words": list(C.SWITCH_WORDS), "close": "sum_v P(v) P(think | v), v in first tokens of </think>, ' </think>'; min_p 1e-6", "version": 3,
            **({"kl_detail": {"bins": KL_BINS, "top": KL_TOP}} if getattr(a, "kl_detail", False) else {}),
            **({"wrapup": {"primary": list(WRAPUP_PRIMARY), "proxy": WRAPUP_PROXY}} if getattr(a, "wrapup", False) else {})}


def check_same_vocab(toks):
    vs = [t.get_vocab() for t in toks.values()]
    if any(v != vs[0] for v in vs[1:]):
        raise RuntimeError("the three tokenizers differ in vocabulary; cross-scoring on shared ids is invalid")


def run(args, models=None, toks=None):
    a = args.job_args
    i, n = S.parse_shard(args.shard)
    toks = toks or {k: H.load_tokenizer(*C.ckpt(k)) for k in LADDER}
    check_same_vocab(toks)
    if a.pair:
        A, B = a.pair
        label = f"pair_{A}_{B}" + ("_kldetail" if getattr(a, "kl_detail", False) else "")
    else:
        if not args.ckpt or len(args.ckpt) != 1:
            raise SystemExit("single mode needs exactly one --ckpt (the scorer)")
        A, B = args.ckpt[0], None
        label = A + ("_wrapup" if getattr(a, "wrapup", False) else "")
    cfg = job_config(a, label)
    st = S.Store("M2", label, cfg)
    if st.done(i):
        print(f"M2 {label} shard {i}/{n} exists; skipping")
        return 0
    traces = S.shard_slice(load_traces(a.traces, a.components), i, n)[: a.limit]
    by_id = items_by_id()
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M2" / label
    prog = R.Progress(root / f"progress_shard{i}of{n}.csv", st.hash)
    key = lambda tc, r: f"{tc}|{r['id']}|{r['seed']}"  # noqa: E731
    todo = [(tc, r) for tc, r in traces if key(tc, r) not in prog.done]
    R.log(f"M2 {label} shard {i}/{n}: {len(traces)} traces, {len(traces) - len(todo)} already in {prog.path}, "
          f"{len(todo)} to do")
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    if todo and models is None:
        R.preflight(2 * MODEL_GB * (2 if B else 1), dirs=[cache])
        models = {A: H.load_hf(*C.ckpt(A))}
        if B:
            models[B] = H.load_hf(*C.ckpt(B))
        res.now("models loaded")
    groups = target_groups(toks[A], wrapup=getattr(a, "wrapup", False))
    close = H.close_ids(toks[A], C.THINK_END)
    for t, (tc, r) in enumerate(todo):
        t0 = time.time()
        tok = toks[tc]
        prep = prepare(tok, by_id[r["id"]], r, tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else 0)
        ids = prep["prompt_ids"] + prep["gen_ids"]
        gs = len(prep["prompt_ids"])
        base = {"trace_ckpt": tc, "id": r["id"], "seed": r["seed"], "component": r["component"], "status": r["status"],
                "n_tokens_frozen": r.get("n_tokens"), "n_gen_tok": prep["n_gen_tok"]}
        if not prep["gen_ids"]:
            prog.add(key(tc, r), {**base, "skipped": "empty generation"}, 0.0)
            continue
        if B:
            out = H.score(models[A], ids, gs, pair_model=models[B], chunk=a.chunk)
            row = {**base, "scorer_a": A, "scorer_b": B, **summarize(prep, out, pair=True)}
            if getattr(a, "kl_detail", False):
                row.update(kl_detail(prep, out["kl"]))
        else:
            out = H.score(models[A], ids, gs, target_groups=groups, chunk=a.chunk)
            row = {**base, "scorer": A, "n_prompt_tok": prep["n_prompt_tok"], "prompt_tok_match": prep["prompt_tok_match"],
                   "gen_tok_match": prep["gen_tok_match"], "sent_end_tok": prep["sent_end_tok"], **summarize(prep, out)}
            pc, _ = H.close_probs(models[A], ids, [gs + k for k in row["sent_next_tok"]], *close)
            row["p_close_at_sent"] = pc                                   # primary for M2-H1 / H2 (amendment 2)
            row["mean_p_close_at_sent"] = float(np.mean(pc)) if pc else None
        prog.add(key(tc, r), row, time.time() - t0)
        if (t + 1) % 50 == 0:
            R.log(f"M2 {label} shard {i}/{n}: {len(prog.done)}/{len(traces)}")
        res.maybe(f"{len(prog.done)}/{len(traces)}")
    rows = [prog.done[key(tc, r)] for tc, r in traces if key(tc, r) in prog.done]
    n_skipped = sum(1 for x in rows if "skipped" in x)
    rows = [x for x in rows if "skipped" not in x]
    res.now("done")
    repo = ";".join(C.ckpt(k)[0] for k in ([A, B] if B else [A]))
    sha = ";".join(C.ckpt(k)[1] for k in ([A, B] if B else [A]))
    loc = st.write(i, n, rows, model_repo=repo, model_sha=sha,
                   extra_meta={"env_report": args.env_report or {}, "n_traces": len(traces),
                               "n_skipped_empty": n_skipped, "target_groups": groups})
    n_bad = sum(1 for x in rows if x.get("prompt_tok_match") is False)
    print(f"wrote {loc}: {len(rows)} rows" + (f"; prompt-token mismatches: {n_bad}" if not B else ""))
    return 0
