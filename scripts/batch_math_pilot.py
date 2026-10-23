# ---------------------------------------------------------------------------
# EXPLORATORY MATH-length pilot (batch size 8) on the RLVR checkpoint, 4000- and 8000-token caps.
# Standalone; run in the pinned batch venv (see scripts/batch_timing_test.py and requirements_pinned.txt):
#   env -u PYTHONPATH /tmp/batch_venv/bin/python batch_math_pilot.py
# Reuses check_versions() and parse-independent generation settings of batch_timing_test.py
# (bf16, default attention, temperature 0.7, top_p 0.9, left padding, explicit pad token, batch seed 0).
# Prompt = the RLVR-MATH 'messages' user turn verbatim (it already contains the few-shot Minerva format);
# NO "keep your reasoning short" cap sentence, since this pilot measures natural length.
# Problems: random.Random(0).sample(range(7500), 10) over allenai/RLVR-MATH train, sorted by prompt length, batches of 8 then 2.
# Caps default to 4000 and 8000; pass caps as arguments to run others (16000 added later).
# Stages are resumable: a stage is skipped if its output file exists.
# ---------------------------------------------------------------------------
import contextlib, gc, io, json, os, random, re, sys, time, traceback
sys.path.insert(0, "/tmp/batch_test")
import batch_timing_test as bt

RES, BS, SEED, N_PROB, PROB_SEED = bt.RES, 8, 0, 10, 0
CAPS = tuple(int(x) for x in sys.argv[1:]) or (4000, 8000)   # e.g. `batch_math_pilot.py 16000`
T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)

BS_ = chr(92)   # backslash

def last_boxed(text):
    i = text.rfind(BS_ + "boxed{")
    if i < 0: return None
    j = i + 7; depth = 1; k = j
    while k < len(text) and depth:
        depth += (text[k] == "{") - (text[k] == "}"); k += 1
    return text[j:k - 1].strip() if depth == 0 else None

def norm(s):
    if s is None: return None
    s = re.sub(re.escape(BS_) + "(dfrac|tfrac)", lambda m: BS_ + "frac", s)
    for t in ("left", "right", "!", ","):
        s = s.replace(BS_ + t, "")
    for t in ("$", " "): s = s.replace(t, "")
    return s.rstrip(".")

def parse_math(text):
    """boxed answer after </think>, else None (same convention as the probe parser)"""
    return last_boxed(text.split("</think>")[-1]) if "</think>" in text else None

def main():
    bt.check_versions(write=False)
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bt.REPO, revision=bt.SHA)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
    mdl = AutoModelForCausalLM.from_pretrained(bt.REPO, revision=bt.SHA, dtype=torch.bfloat16, device_map="cuda").eval()
    EOS = set(mdl.generation_config.eos_token_id)
    log("attn:", mdl.config._attn_implementation, "pad:", tok.pad_token, tok.pad_token_id, "eos:", sorted(EOS),
        "temp/top_p:", bt.TEMPERATURE, bt.TOP_P, "batch size:", BS, "seed:", SEED)

    d = load_dataset("allenai/RLVR-MATH", split="train")
    idx = random.Random(PROB_SEED).sample(range(len(d)), N_PROB)
    rows = []
    for k in idx:
        r = d[k]; content = r["messages"][-1]["content"]
        ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
        rows.append({"idx": k, "gt": r["ground_truth"], "ids": ids, "plen": len(ids)})
    log("selected RLVR-MATH indices (seed 0):", idx)
    order = sorted(rows, key=lambda r: (r["plen"], r["idx"]))
    log("prompt lengths (tokens):", {r["idx"]: r["plen"] for r in order})

    for cap in CAPS:
        fn = f"batch_test_math_{cap}.json"
        if bt.exists(fn): log(f"skip cap {cap}: {fn} exists"); continue
        log(f"=== MATH pilot: cap {cap}, batch size {BS} ===")
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        out_rows, bmeta, wall = [], [], 0.0
        for b in range(0, len(order), BS):
            batch = order[b:b + BS]
            enc = tok.pad({"input_ids": [r["ids"] for r in batch]}, padding=True, return_tensors="pt").to(mdl.device)
            L = enc["input_ids"].shape[1]
            torch.manual_seed(SEED); torch.cuda.synchronize(); t = time.time()
            with torch.no_grad():
                g = mdl.generate(**enc, max_new_tokens=cap, pad_token_id=tok.pad_token_id, do_sample=True, temperature=bt.TEMPERATURE, top_p=bt.TOP_P)
            torch.cuda.synchronize(); w = time.time() - t
            steps = g.shape[1] - L; wall += w
            bm = {"batch_size": len(batch), "padded_prompt_len": L, "steps": steps, "wall": w, "idx": [r["idx"] for r in batch]}
            bmeta.append(bm)
            for j, r in enumerate(batch):
                toks = g[j, L:].tolist(); cut, ended = len(toks), False
                for kk, x in enumerate(toks):
                    if x in EOS: cut, ended = kk + 1, True; break
                toks = toks[:cut]; text = tok.decode(toks, skip_special_tokens=True)
                a = parse_math(text)
                out_rows.append({"idx": r["idx"], "gt": r["gt"], "prompt_tokens": r["plen"], "n_tokens": len(toks), "ended_eos": ended,
                                 "hit_cap": (not ended) and len(toks) >= cap, "think_closed": "</think>" in text,
                                 "answer": a, "parses": a is not None, "matches_gt_normalised": (norm(a) == norm(r["gt"])) if a is not None else False,
                                 "boxed_anywhere": last_boxed(text) is not None, "batch_wall": w, "batch_steps": steps,
                                 "tokens_per_s_batch_share": len(toks) / w, "tokens": toks, "text": text})
            log(f"cap {cap} batch of {len(batch)}: steps={steps} wall={w:.0f}s tokens={sum(o['n_tokens'] for o in out_rows[-len(batch):])}")
            del g, enc
        useful = sum(o["n_tokens"] for o in out_rows); pos = sum(m["batch_size"] * m["steps"] for m in bmeta)
        res = {"cap": cap, "batch_size": BS, "seed": SEED, "indices": idx, "total_tokens": useful, "wall": wall, "tokens_per_s": useful / wall,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "waste_frac": 1 - useful / pos,
               "n_hit_cap": sum(o["hit_cap"] for o in out_rows), "n_parse": sum(o["parses"] for o in out_rows),
               "n_match_gt": sum(o["matches_gt_normalised"] for o in out_rows), "batches": bmeta, "rows": out_rows}
        bt.save(fn, res)
        log(f"cap {cap} DONE tokens={useful} wall={wall:.0f}s tok/s={useful/wall:.1f} hit_cap={res['n_hit_cap']} parse={res['n_parse']} gt={res['n_match_gt']} peak={res['peak_gpu_gb']:.1f}GB")
    print("DONE", flush=True)

if __name__ == "__main__":
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try: bt.check_versions(write=False)
            finally: pass
    except SystemExit as e:
        print(buf.getvalue()); print("FAILED:", e, flush=True); raise
    open(RES + ("/batch_test_math_env.txt" if CAPS == (4000, 8000) else f"/batch_test_math_{'_'.join(map(str, CAPS))}_env.txt"), "w").write("# version check before MATH pilot\n" + buf.getvalue())
    print(buf.getvalue(), flush=True)
    try: main()
    except SystemExit as e: print("FAILED:", e, flush=True); raise
    except BaseException: traceback.print_exc(); print("FAILED: exception", flush=True); sys.exit(1)
