"""M5: causal tests (registration amendments 3, 5; operational details amendment 7). Exploratory interventions.

    python -m exp.run --job M5 [--layer 19] [--patch-layers 4 11 15 19 23 29] [--cap 3000] [--limit N]

One process, one model at a time (weights stay in $HF_HUB_CACHE between phases; the launcher deletes them after):
  phase 1  DPO   residual at the last prompt token of every A and D prompt (layers --patch-layers)
  phase 2  RLVR  the same, plus the full prompt residuals (for patching) and the backtracking direction:
                 mean residual at positions whose NEXT token is a first token of " But"/" maybe"/" Wait" (with or
                 without the space) minus matched controls (same trace, same relative-position decile, same
                 'current token ends a sentence' status), from B1 main-arm RLVR and DPO traces of the TRAIN wordings
                 (seeds 0-4), read in the RLVR model; its held-out quality = AUC of the projection on EVAL wordings.
                 offset vector (per layer) = mean over TRAIN A wordings of (RLVR - DPO) last-prompt-token residual.
                 RLVR conditions: baseline (seeds 0, 1), steer -alpha*offset (alpha 0.5, 1, 2), ablate the backtracking
                 direction (projected out at every position).
  phase 3  DPO   conditions: baseline, steer +alpha*offset (alpha 0.5, 1, 2), and patching: RLVR's prompt residuals
                 replace DPO's at layer L (prompt positions only) for each L in --patch-layers.
Steering/ablation act at --layer (19, fixed in amendment 7) on every position, during generation and readout.
Evaluation set: the 24 EVAL A wordings (wording_index 1, 3, 5) and the last 25 of the first 50 D base items (o0).
Generation: HF generate with hooks, model-default sampling (0.6 / 0.95), cap --cap new tokens, one seed per condition
(torch seed fixed per condition and batch). Per generation: think tokens (to '</think>', cap if unclosed), closed,
answer (frozen parser), and for A the B1 readout P(1..5) on the generated think text under the same intervention.
Columns: condition, model, alpha, layer, id, component, seed, think_tok, closed, n_new, status, answer, p_read_1..5,
read_ext (P(1)+P(5)), text. Shard metadata: direction AUC, offset norms, cosine(offset, backtracking direction).
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

TARGETS = ("But", "maybe", "Wait")
ALPHAS = (0.5, 1.0, 2.0)
SENT_END = (".", "?", "!", "\n")


def add_args(p):
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--patch-layers", type=int, nargs="*", default=[4, 11, 15, 19, 23, 29])
    p.add_argument("--cap", type=int, default=3000)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--limit", type=int, default=None, help="debug: first N evaluation prompts")


def split_items():
    A = C.load_items("A")
    train = [it for it in A if it["wording_index"] in (0, 2, 4)]
    eval_a = [it for it in A if it["wording_index"] in (1, 3, 5)]
    D = sorted((x for x in C.load_items("D") if x["order"] == 0), key=lambda x: x["draw_index"])[:50]
    return train, eval_a, D[25:]


# ------------------------------------------------------------------ hooks
class Intervene:
    """Forward hook on decoder layers: steer (h + vec), ablate (h - (h.u)u), or patch prompt positions."""

    def __init__(self, model):
        self.model, self.handles = model, []

    def layer(self, li):
        return self.model.model.layers[li]

    def steer(self, li, vec):
        import torch
        v = torch.as_tensor(vec, device=self.model.device, dtype=self.model.dtype)

        def f(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + tuple(out[1:])
            return out + v
        self.handles.append(self.layer(li).register_forward_hook(f))

    def ablate(self, li, unit):
        import torch
        u = torch.as_tensor(unit, device=self.model.device, dtype=torch.float32)

        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hf = hf - (hf @ u)[..., None] * u
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self.handles.append(self.layer(li).register_forward_hook(f))

    def patch(self, li, store):
        """store: object with .rows = list of (offset, np.array(plen, d)) for the current batch; applied on full
        forwards (sequence length > 1) to positions offset .. offset+plen of each row."""
        import torch

        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1 or store.rows is None:
                return out
            h = h.clone()
            for b, (off, arr) in enumerate(store.rows):
                n = min(arr.shape[0], h.shape[1] - off)
                h[b, off:off + n] = torch.as_tensor(arr[:n], device=h.device, dtype=h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.handles.append(self.layer(li).register_forward_hook(f))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class PatchRows:
    rows = None


# ------------------------------------------------------------------ activations
def resid_at(model, ids, positions, layers):
    import torch
    with torch.no_grad():
        out = model.model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True)
    return {li: out.hidden_states[li + 1][0, positions].float().cpu().numpy() for li in layers}


def prompt_resids(model, tok, items, layers, full=False):
    last, fullr = {}, {}
    for it in items:
        ids = H.prompt_ids(tok, it["messages"])
        r = resid_at(model, ids, list(range(len(ids))) if full else [len(ids) - 1], layers)
        last[it["id"]] = {li: r[li][-1] for li in layers}
        if full:
            fullr[it["id"]] = {li: r[li].astype(np.float16) for li in layers}
    return last, fullr


def backtrack_direction(model, tok, traces, layer, target_ids, rng):
    """(direction, per-trace (pos_proj, ctl_proj) lists) from traces = list of (item, think_text)."""
    tgt, ctl = [], []
    for it, think in traces:
        pids = H.prompt_ids(tok, it["messages"])
        tids = H.text_ids(tok, think)
        ids = pids + tids
        n = len(tids)
        if n < 20:
            continue
        nxt = tids[1:] + [None]
        toks = tok.convert_ids_to_tokens(tids)
        ends = [any(t.endswith(e) or t.endswith(e.replace("\n", "Ċ")) for e in SENT_END) for t in toks]
        pos = [j for j in range(n - 1) if nxt[j] in target_ids]
        if not pos:
            continue
        pool = {}
        for j in range(n - 1):
            if nxt[j] not in target_ids:
                pool.setdefault((10 * j // n, ends[j]), []).append(j)
        cpos = []
        for j in pos:
            cand = pool.get((10 * j // n, ends[j])) or []
            if cand:
                cpos.append(int(rng.choice(cand)))
        r = resid_at(model, ids, [len(pids) + j for j in pos + cpos], [layer])[layer]
        tgt.append(r[:len(pos)])
        if cpos:
            ctl.append(r[len(pos):])
    T, Cc = np.concatenate(tgt), np.concatenate(ctl)
    return T.mean(0) - Cc.mean(0), T, Cc


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ generation under an intervention
def generate(model, tok, items, sampling, cap, seed, batch, patch_store=None, patch_full=None):
    import torch
    rows = []
    for c in range(0, len(items), batch):
        chunk = items[c:c + batch]
        pids = [H.prompt_ids(tok, it["messages"]) for it in chunk]
        L = max(len(p) for p in pids)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        inp = torch.tensor([[pad] * (L - len(p)) + p for p in pids], device=model.device)
        att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in pids], device=model.device)
        if patch_store is not None:
            patch_store.rows = [(L - len(p), patch_full[it["id"]]) for p, it in zip(pids, chunk)]
        torch.manual_seed(seed * 1000 + c)
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=att, do_sample=True, temperature=sampling["temperature"],
                                 top_p=sampling["top_p"], max_new_tokens=cap, pad_token_id=pad)
        for it, o in zip(chunk, out[:, L:].tolist()):
            if tok.eos_token_id in o:
                o = o[:o.index(tok.eos_token_id) + 1]
            rows.append((it, tok.decode(o, skip_special_tokens=False), len(o)))
    return rows


def measure(model, tok, it, text, n_new, cap, patch_store=None, patch_full=None):
    closed = "</think>" in text
    think = text.split("</think>", 1)[0]
    if think.startswith("<think>"):
        think = think[len("<think>"):]
    ans = P.strict_answer(text, it["parser_kind"], n_new, cap, n_new < cap)
    row = {"think_tok": len(H.text_ids(tok, think)) if closed else n_new, "closed": closed, "n_new": n_new,
           "status": ans["status"], "answer": None if ans["answer"] is None else str(ans["answer"]), "text": text}
    if it["component"] == "A" and closed:
        if patch_store is not None:
            patch_store.rows = [(0, patch_full[it["id"]])]
        ro = H.likert_readout(model, tok, it["messages"], think_text=think)
        pr = ro[H.primary_variant(ro)].probs
        row.update({f"p_read_{j + 1}": float(pr[j]) for j in range(5)})
        row["read_ext"] = float(pr[0] + pr[4])
    return row


def job_config(a, sampling):
    return {"job": "M5", "dpo": list(C.ckpt("dpo")), "rlvr": list(C.ckpt("rlvr")), "layer": a.layer,
            "patch_layers": list(a.patch_layers), "alphas": list(ALPHAS), "targets": list(TARGETS), "cap": a.cap,
            "sampling": sampling, "eval": "A wording_index 1,3,5 (24) + D first-50[25:] (25)",
            "train": "A wording_index 0,2,4; backtracking from B1 main-arm traces seeds 0-4", "version": 1}


def run(args, load=None):
    a = args.job_args
    sampling = H.resolve_sampling(*C.ckpt("dpo"), {"temperature": None, "top_p": None, "top_k": None}) \
        if load is None else {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "test"}
    st = S.Store("M5", "dpo_rlvr", job_config(a, sampling))
    if st.done(0):
        print("M5 exists; skipping")
        return 0
    load = load or (lambda ck: H.load_hf(*C.ckpt(ck)))
    tok = H.load_tokenizer(*C.ckpt("dpo"))
    train, eval_a, eval_d = split_items()
    ev = (eval_a + eval_d)[: a.limit]
    layers = sorted(set(a.patch_layers) | {a.layer})
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "M5" / f"dpo_rlvr__{st.hash}"
    prog = R.Progress(root / "progress_shard0of1.csv", st.hash)
    work = Path(os.environ.get("M5_WORK", "/tmp/m5_work")) / st.hash
    work.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    key = lambda cond, it: f"{cond}|{it['id']}"  # noqa: E731
    meta = {}

    def run_condition(model, cond, mdl, alpha, layer, seed, patch_store=None, patch_full=None):
        todo = [it for it in ev if key(cond, it) not in prog.done]
        if not todo:
            return
        t0 = time.time()
        for it, text, n_new in generate(model, tok, todo, sampling, a.cap, seed, a.batch, patch_store, patch_full):
            row = {"condition": cond, "model": mdl, "alpha": alpha, "layer": layer, "id": it["id"],
                   "component": it["component"], "seed": seed,
                   **measure(model, tok, it, text, n_new, a.cap, patch_store, patch_full)}
            prog.add(key(cond, it), row, (time.time() - t0) / len(todo))
        R.log(f"M5 {cond}: {len(todo)} generations in {time.time() - t0:.0f}s")
        res.maybe(cond)

    # ---- phase 1: DPO prompt residuals
    p1 = work / "dpo_last.npz"
    if not p1.exists():
        R.preflight(2 * 15, dirs=[cache])
        m = load("dpo")
        res.now("phase 1: DPO loaded")
        last, _ = prompt_resids(m, tok, train + eval_a + eval_d, layers)
        np.savez(p1, last=np.array(last, dtype=object))
        del m
        _free()
    dpo_last = np.load(p1, allow_pickle=True)["last"].item()

    # ---- phase 2: RLVR
    p2 = work / "rlvr.npz"
    m = None
    if not p2.exists() or any(key(c, it) not in prog.done for c in rlvr_conditions() for it in ev):
        R.preflight(2 * 15, dirs=[cache])
        m = load("rlvr")
        res.now("phase 2: RLVR loaded")
    if not p2.exists():
        last, full = prompt_resids(m, tok, train + eval_a + eval_d, layers, full=True)
        from .M3_settle import b1_traces
        tr_ids = {it["id"] for it in train}
        ev_ids = {it["id"] for it in eval_a}
        items = {it["id"]: it for it in C.load_items("A")}
        tr, te = [], []
        for ck in ("rlvr", "dpo"):
            for r in b1_traces(ck):
                if r["seed"] < 5:
                    (tr if r["id"] in tr_ids else te if r["id"] in ev_ids else []).append((items[r["id"]], r["think_text"]))
        tids = {t for w in TARGETS for t in H.first_token_ids(tok, [w])[w]}
        rng = np.random.default_rng(20261024)
        d, _, _ = backtrack_direction(m, tok, tr, a.layer, tids, rng)
        unit = d / np.linalg.norm(d)
        _, T, Cc = backtrack_direction(m, tok, te, a.layer, tids, rng)
        offset = {li: np.mean([last[i][li] - dpo_last[i][li] for i in tr_ids], axis=0) for li in layers}
        np.savez(p2, last=np.array(last, dtype=object), full=np.array(full, dtype=object), unit=unit,
                 offset=np.array(offset, dtype=object),
                 meta=np.array({"backtrack_auc_eval": auc(T @ unit, Cc @ unit), "n_target_eval": len(T),
                                "offset_norm": {li: float(np.linalg.norm(v)) for li, v in offset.items()},
                                "cos_offset_backtrack": float(offset[a.layer] @ unit / np.linalg.norm(offset[a.layer]))},
                               dtype=object))
    z = np.load(p2, allow_pickle=True)
    full, unit, offset, meta = z["full"].item(), z["unit"], z["offset"].item(), z["meta"].item()
    R.log(f"M5 backtracking direction: eval AUC {meta['backtrack_auc_eval']:.3f}; cos(offset, direction) "
          f"{meta['cos_offset_backtrack']:.3f}")
    if m is not None:
        iv = Intervene(m)
        run_condition(m, "rlvr_base_s0", "rlvr", 0.0, None, 0)
        run_condition(m, "rlvr_base_s1", "rlvr", 0.0, None, 1)
        for al in ALPHAS:
            iv.steer(a.layer, -al * offset[a.layer])
            run_condition(m, f"rlvr_minus_offset_{al}", "rlvr", -al, a.layer, 0)
            iv.clear()
        iv.ablate(a.layer, unit)
        run_condition(m, "rlvr_ablate_backtrack", "rlvr", 0.0, a.layer, 0)
        iv.clear()
        del m, iv
        _free()

    # ---- phase 3: DPO conditions
    conds3 = ["dpo_base_s0"] + [f"dpo_plus_offset_{al}" for al in ALPHAS] + [f"dpo_patch_L{li}" for li in a.patch_layers]
    if any(key(c, it) not in prog.done for c in conds3 for it in ev):
        R.preflight(2 * 15, dirs=[cache])
        m = load("dpo")
        res.now("phase 3: DPO loaded")
        iv = Intervene(m)
        run_condition(m, "dpo_base_s0", "dpo", 0.0, None, 0)
        for al in ALPHAS:
            iv.steer(a.layer, al * offset[a.layer])
            run_condition(m, f"dpo_plus_offset_{al}", "dpo", al, a.layer, 0)
            iv.clear()
        for li in a.patch_layers:
            ps = PatchRows()
            iv.patch(li, ps)
            run_condition(m, f"dpo_patch_L{li}", "dpo", 0.0, li, 0, ps, {i: v[li] for i, v in full.items()})
            iv.clear()
        del m, iv
        _free()

    allc = rlvr_conditions() + conds3
    rows = [prog.done[key(c, it)] for c in allc for it in ev]
    res.now("done")
    loc = st.write(0, 1, rows, model_repo=";".join(C.ckpt(k)[0] for k in ("dpo", "rlvr")),
                   model_sha=";".join(C.ckpt(k)[1] for k in ("dpo", "rlvr")),
                   extra_meta={"env_report": args.env_report or {}, **meta})
    print(f"wrote {loc}: {len(rows)} rows")
    return 0


def rlvr_conditions():
    return ["rlvr_base_s0", "rlvr_base_s1"] + [f"rlvr_minus_offset_{al}" for al in ALPHAS] + ["rlvr_ablate_backtrack"]


def _free():
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
