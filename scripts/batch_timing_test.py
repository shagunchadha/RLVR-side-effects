# ---------------------------------------------------------------------------
# EXPLORATORY batched-generation timing test and batched-vs-unbatched check.
# Standalone (no marimo); run in an isolated venv built from requirements_pinned.txt.
# Model: allenai/Olmo-3-7B-Think (RLVR) at the pinned SHA, because unbatched outputs for the
# 36 pilot rows already exist (results/rlvr_newenv_pilot.json).
# Prompt builder, chat template, dtype, attention implementation, sampling settings and the
# 4000-token cap are those of scripts/rlvr_newenv_pilot.py. Only batching (left padding) differs.
#
# Working directory WD (/tmp/batch_test on the sandbox) must contain:
#   define_trait_probes.py, define_reasoning_probes.py, rlvr_newenv_pilot.json, requirements_pinned.txt
# Usage (PYTHONPATH cleared, venv python):
#   env -u PYTHONPATH /tmp/batch_venv/bin/python batch_timing_test.py envcheck   # before any model load
#   env -u PYTHONPATH /tmp/batch_venv/bin/python batch_timing_test.py run        # stages A-E, resumable
# ---------------------------------------------------------------------------
import copy, csv, gc, importlib.metadata as md, json, os, random, re, subprocess, sys, time, traceback

WD = "/tmp/batch_test"
RES = WD + "/results"
REPO = "allenai/Olmo-3-7B-Think"
SHA = "d97e442d7cc678210054dbcc9b440894d62c89a4"   # docs/checkpoint_map.md
MAX_NEW, TEMPERATURE, TOP_P = 4000, 0.7, 0.9
CAP = " Keep your reasoning to at most 5-6 sentences, then give your final answer."
PIN = {"torch": "2.11.0+cu130", "transformers": "5.14.1", "huggingface-hub": "1.33.0", "accelerate": "1.15.0"}
PIN_DRIVER, PIN_GPU = "595.71.05", "NVIDIA RTX PRO 6000 Blackwell Server Edition"
SUBSET_MOD = {0, 1, 2, 3, 4}          # row i is in the 20-row subset iff i % 9 in SUBSET_MOD
C_BATCH_SIZES = [4, 8, 16, 36]
BATCH_SEED, SHUFFLE_SEED = 0, 1234
PLAN_GENS = 96 + 40 + 80 + 300 + 50    # 566
T0 = time.time()

def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

# ------------------------------- version check -------------------------------
def check_versions(write=False):
    import torch, transformers, huggingface_hub
    smi = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").strip()
    gpu, driver = [x.strip() for x in smi.split(",")]
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "huggingface-hub": huggingface_hub.__version__, "accelerate": md.version("accelerate")}
    out = [f"# batch-test environment check, repo={REPO} sha={SHA}",
           f"python        : {sys.version.split()[0]}  exe={sys.executable}",
           f"PYTHONPATH    : {os.environ.get('PYTHONPATH', '')!r}",
           f"GPU           : {gpu}", f"driver        : {driver}",
           f"torch.cuda    : {torch.version.cuda} available={torch.cuda.is_available()}"]
    bad = []
    if os.environ.get("PYTHONPATH", ""): bad.append("PYTHONPATH not cleared")
    if gpu != PIN_GPU: bad.append(f"GPU {gpu!r} != {PIN_GPU!r}")
    if driver != PIN_DRIVER: bad.append(f"driver {driver} != {PIN_DRIVER}")
    for k, v in PIN.items():
        ok = got[k] == v
        out.append(f"{k:16s}: {got[k]}  (pinned {v})  {'OK' if ok else 'MISMATCH'}")
        if not ok: bad.append(f"{k} {got[k]} != {v}")
    try: out.append(f"datasets      : {md.version('datasets')} (system site, as in the pilots; not in the freeze)")
    except Exception: out.append("datasets      : NOT INSTALLED")
    # every other pinned package, informational + counted as mismatch
    for line in open(WD + "/requirements_pinned.txt"):
        line = line.strip()
        if not line or "==" not in line: continue
        n, v = line.split("==")
        if n in PIN: continue
        try: g = md.version(n)
        except md.PackageNotFoundError: g = None
        if g is None or (g != v and g.split("+")[0] != v):
            out.append(f"  pinned {n}=={v}: installed {g}  MISMATCH"); bad.append(f"{n} {g} != {v}")
    out.append("RESULT: " + ("ABORT - " + "; ".join(bad) if bad else "ALL PINNED VERSIONS MATCH"))
    text = "\n".join(out)
    print(text, flush=True)
    if write:
        os.makedirs(RES, exist_ok=True)
        with open(RES + "/batch_test_env.txt", "w") as f:
            f.write(text + "\n# --- uv pip freeze (batch venv, includes system-site packages) ---\n")
            f.write(sh(f"uv pip freeze --python {sys.executable}"))
    if bad: sys.exit("VERSION MISMATCH: " + "; ".join(bad))

# ------------------------------- parsing (verbatim from the pilot) -------------------------------
ANS = re.compile(r"ANSWER:\s*([^\n]{1,20})")
VALID = {"self_report": re.compile(r"^[1-5]$"), "reasoning": re.compile(r"^[A-E]$", re.I)}
def parse(text, ptype):
    if "</think>" not in text: return None
    m = ANS.findall(text.split("</think>")[-1])
    if not m: return None
    t = m[-1].strip().strip("\"'").rstrip(".:").split()
    t = t[0].strip(".,\"'") if t else ""
    if ptype in VALID:
        return (t.upper() if ptype == "reasoning" else t) if VALID[ptype].match(t) else None
    mm = re.match(r"-?\d{1,4}", t)
    return mm.group(0) if mm else None

def agree(a, b):
    """exact agreement (None counts as a value) and agreement among pairs where both parsed"""
    pairs = list(zip(a, b))
    ex = sum(x == y for x, y in pairs)
    both = [(x, y) for x, y in pairs if x is not None and y is not None]
    return {"n": len(pairs), "exact": ex, "exact_frac": ex / len(pairs) if pairs else None,
            "n_both_parsed": len(both), "both_parsed_equal": sum(x == y for x, y in both),
            "both_parsed_frac": (sum(x == y for x, y in both) / len(both)) if both else None}

def save(name, obj):
    with open(f"{RES}/{name}", "w", encoding="utf-8") as f: json.dump(obj, f, indent=1)

def exists(name): return os.path.exists(f"{RES}/{name}")
def load(name): return json.load(open(f"{RES}/{name}", encoding="utf-8"))

# ------------------------------- main -------------------------------
def main():
    check_versions(write=False)           # abort before any model load
    import torch, transformers, huggingface_hub
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(RES, exist_ok=True)

    log("hub confirms revision:", huggingface_hub.HfApi().model_info(REPO, revision=SHA).sha)
    tok = AutoTokenizer.from_pretrained(REPO, revision=SHA)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        log("pad token was unset; set explicitly to eos_token")
    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
    t_load = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(REPO, revision=SHA, dtype=torch.bfloat16, device_map="cuda").eval()
    load_secs = time.time() - t_load
    gc_ = mdl.generation_config
    eos = gc_.eos_token_id if gc_.eos_token_id is not None else tok.eos_token_id
    EOS = set(eos if isinstance(eos, (list, tuple)) else [eos])
    settings = {
        "repo": REPO, "sha": SHA, "dtype": "bfloat16", "attn_implementation": mdl.config._attn_implementation,
        "load_kwargs": "dtype=torch.bfloat16, device_map='cuda' (no attn_implementation passed, as in the pilot)",
        "max_new_tokens": MAX_NEW, "sampled": {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P},
        "greedy": {"do_sample": False}, "cap_instruction": CAP.strip(),
        "chat_template": "tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt', return_dict=True), chat[-1]['content'] += CAP",
        "padding_side": tok.padding_side, "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
        "eos_token_ids": sorted(EOS), "generation_config_defaults": gc_.to_dict(),
        "subset_rule": "i % 9 in {0,1,2,3,4}", "c_batch_sizes": C_BATCH_SIZES,
        "batch_seed": BATCH_SEED, "shuffle_seed": SHUFFLE_SEED,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "huggingface_hub": huggingface_hub.__version__, "accelerate": md.version("accelerate")},
        "model_load_seconds": round(load_secs, 1), "model_gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }
    log("SETTINGS", json.dumps({k: v for k, v in settings.items() if k != "generation_config_defaults"}, indent=1))
    log("generation_config defaults:", gc_.to_dict())
    if not exists("batch_test_settings.json"): save("batch_test_settings.json", settings)

    # ---- rows: the 36 pilot rows in original order, prompts built exactly as in the pilot ----
    ns = {"commonsense_qa": load_dataset("tau/commonsense_qa", split="validation")}
    exec(open(WD + "/define_trait_probes.py").read(), ns)
    exec(open(WD + "/define_reasoning_probes.py").read(), ns)
    ALL = ns["TRAIT_PROBES"] + ns["REASONING_PROBES"]
    jobs = [(p, s) for p in ALL for s in range(1, (3 if p["probe_type"] == "self_report" else 1) + 1)]
    pilot = json.load(open(WD + "/rlvr_newenv_pilot.json", encoding="utf-8"))["results"]
    assert len(jobs) == len(pilot) == 36
    ROWS = []
    for i, (p, s) in enumerate(jobs):
        chat = copy.deepcopy(p["messages"]) if "messages" in p else [{"role": "user", "content": p["prompt"]}]
        chat[-1]["content"] += CAP
        pr = pilot[i]
        assert (pr["id"], pr["sample"]) == (p["id"], s) and pr["prompt_used"] == chat[-1]["content"], f"row {i} mismatch"
        ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
        ROWS.append({"i": i, "key": f"{p['id']}|{s}", "id": p["id"], "sample": s, "ptype": p["probe_type"], "ids": ids,
                     "plen": len(ids), "pilot_answer": pr["extracted_answer"], "pilot_tokens": pr["n_generated_tokens"]})
    SUB = [r for r in ROWS if r["i"] % 9 in SUBSET_MOD]
    assert len(SUB) == 20
    log("20-row subset ids:", [r["key"] for r in SUB])
    log("20-row subset indices:", [r["i"] for r in SUB])

    # ---- generation core ----
    def gen(rowlist, sample, seed=None):
        enc = tok.pad({"input_ids": [r["ids"] for r in rowlist]}, padding=True, return_tensors="pt").to(mdl.device)
        L = enc["input_ids"].shape[1]
        kw = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P) if sample else dict(do_sample=False)
        if seed is not None: torch.manual_seed(seed)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            g = mdl.generate(**enc, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, **kw)
        torch.cuda.synchronize(); wall = time.time() - t
        steps = g.shape[1] - L
        out = []
        for j, r in enumerate(rowlist):
            toks = g[j, L:].tolist(); cut = len(toks); ended = False
            for k, x in enumerate(toks):
                if x in EOS: cut = k + 1; ended = True; break
            toks = toks[:cut]
            text = tok.decode(toks, skip_special_tokens=True)
            out.append({"key": r["key"], "i": r["i"], "ptype": r["ptype"], "tokens": toks, "n": len(toks), "ended_eos": ended,
                        "hit_cap": (not ended) and len(toks) >= MAX_NEW, "think_closed": "</think>" in text,
                        "answer": parse(text, r["ptype"]), "text": text})
        del g, enc
        return out, {"batch_size": len(rowlist), "padded_prompt_len": L, "steps": steps, "wall": wall}

    def safe(fn, *a, **k):
        try: return fn(*a, **k)
        except torch.OutOfMemoryError as e:
            log("OOM:", str(e)[:200]); gc.collect(); torch.cuda.empty_cache(); return None

    # ===== Stage A: unbatched greedy reference =====
    if not exists("batch_test_stageA.json"):
        log("=== Stage A: unbatched greedy, 20 rows one at a time ===")
        rows_a = []
        for n, r in enumerate(SUB, 1):
            o, m = gen([r], sample=False)
            o = o[0]; o["wall"] = round(m["wall"], 2); rows_a.append(o)
            log(f"A {n}/20 {r['key']:38s} tok={o['n']:5d} {o['wall']:6.1f}s ans={o['answer']!r}")
        tt = sum(o["n"] for o in rows_a); tw = sum(o["wall"] for o in rows_a)
        save("batch_test_stageA.json", {"total_tokens": tt, "total_wall": tw, "tokens_per_s": tt / tw, "rows": rows_a})
    A = {o["key"]: o for o in load("batch_test_stageA.json")["rows"]}

    # ===== Stage B: batched greedy (one batch of 20, left padding) =====
    if not exists("batch_test_stageB.json"):
        log("=== Stage B: batched greedy, one batch of 20 ===")
        torch.cuda.reset_peak_memory_stats()
        res = safe(gen, SUB, False)
        if res is None:
            save("batch_test_stageB.json", {"status": "oom"}); sys.exit("Stage B OOM")
        outs, meta = res
        cmp_rows = []
        for o in outs:
            a = A[o["key"]]
            n = min(len(a["tokens"]), len(o["tokens"]))
            fd = next((k for k in range(n) if a["tokens"][k] != o["tokens"][k]), None)
            if fd is None and len(a["tokens"]) != len(o["tokens"]): fd = n
            cmp_rows.append({"key": o["key"], "identical": fd is None, "first_diff_pos": fd,
                             "n_unbatched": a["n"], "n_batched": o["n"],
                             "answer_unbatched": a["answer"], "answer_batched": o["answer"],
                             "answer_same": a["answer"] == o["answer"]})
            log(f"B {o['key']:38s} identical={fd is None} first_diff={fd} ans {a['answer']!r}/{o['answer']!r}")
        save("batch_test_stageB.json", {"meta": meta, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
             "n_identical": sum(c["identical"] for c in cmp_rows), "n_answer_same": sum(c["answer_same"] for c in cmp_rows),
             "compare": cmp_rows, "rows": [{k: v for k, v in o.items()} for o in outs]})

    # ===== Stage C: throughput on the 36 pilot prompts, sampled =====
    pn = "batch_test_stageC_partial.json"
    C = load(pn) if exists(pn) else {}
    if not exists("batch_test_stageC.json"):
        log("=== Stage C: throughput, 36 prompts sampled ===")
        order = sorted(ROWS, key=lambda r: (r["plen"], r["i"]))
        for bs in C_BATCH_SIZES:
            if str(bs) in C: continue
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batches = [order[k:k + bs] for k in range(0, len(order), bs)]
            wall = 0.0; useful = 0; positions = 0; rows_c = []; bmeta = []; oom = False
            for b in batches:
                res = safe(gen, b, True, BATCH_SEED)
                if res is None: oom = True; break
                outs, m = res
                wall += m["wall"]; useful += sum(o["n"] for o in outs); positions += m["batch_size"] * m["steps"]
                bmeta.append(m)
                rows_c += [{k: v for k, v in o.items() if k not in ("tokens", "text")} for o in outs]
                log(f"C bs={bs} batch {len(bmeta)}/{len(batches)} steps={m['steps']} wall={m['wall']:.0f}s")
            if oom:
                C[str(bs)] = {"batch_size": bs, "status": "oom", "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            else:
                C[str(bs)] = {"batch_size": bs, "status": "ok", "wall": wall, "total_tokens": useful,
                              "tokens_per_s": useful / wall, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                              "n_hit_cap": sum(r["hit_cap"] for r in rows_c), "n_ended_eos": sum(r["ended_eos"] for r in rows_c),
                              "generated_positions": positions, "waste_frac": 1 - useful / positions,
                              "batches": bmeta, "rows": rows_c}
                log(f"C bs={bs} DONE wall={wall:.0f}s tok={useful} tok/s={useful/wall:.1f} peak={C[str(bs)]['peak_gpu_gb']:.1f}GB waste={C[str(bs)]['waste_frac']:.2f}")
            save(pn, C)
        save("batch_test_stageC.json", C)
    C = load("batch_test_stageC.json")
    ok = {int(k): v for k, v in C.items() if v["status"] == "ok"}
    best_bs = max(ok, key=lambda k: ok[k]["tokens_per_s"])
    log("best batch size from C:", best_bs, {k: round(v["tokens_per_s"], 1) for k, v in ok.items()})

    # ===== Stage D: sampled agreement with the unbatched v3/pilot answers =====
    if not exists("batch_test_stageD.json"):
        eff = min(best_bs, len(SUB))
        log(f"=== Stage D: sampled, 20 rows, batch size {eff} (best from C = {best_bs}), seed {BATCH_SEED} ===")
        shuf = SUB[:]; random.Random(SHUFFLE_SEED).shuffle(shuf)
        variants = {"original_order": SUB, "reversed_order": SUB[::-1], "shuffled_order": shuf}
        D = {"best_bs_from_C": best_bs, "effective_bs": eff, "batch_seed": BATCH_SEED, "shuffle_seed": SHUFFLE_SEED, "variants": {}}
        pil = {r["key"]: r["pilot_answer"] for r in ROWS}
        for name, order in variants.items():
            outs_all = []
            for k in range(0, len(order), eff):
                res = safe(gen, order[k:k + eff], True, BATCH_SEED)
                if res is None: sys.exit(f"Stage D OOM ({name})")
                outs_all += res[0]
            by = {o["key"]: o for o in outs_all}
            keys = [r["key"] for r in SUB]
            ba = [by[k]["answer"] for k in keys]; ua = [pil[k] for k in keys]
            sr = [i for i, r in enumerate(SUB) if r["ptype"] == "self_report"]
            rs = [i for i, r in enumerate(SUB) if r["ptype"] != "self_report"]
            D["variants"][name] = {"order": [r["key"] for r in order],
                "agree_all": agree(ba, ua), "agree_self_report": agree([ba[i] for i in sr], [ua[i] for i in sr]),
                "agree_other": agree([ba[i] for i in rs], [ua[i] for i in rs]),
                "rows": [{k: v for k, v in by[x].items() if k != "tokens"} | {"pilot_answer": pil[x]} for x in keys]}
            log(f"D {name}: all={D['variants'][name]['agree_all']['exact']}/20 self_report={D['variants'][name]['agree_self_report']}")
        names = list(variants)
        D["batched_vs_batched"] = {f"{a}__{b}": agree([x["answer"] for x in D["variants"][a]["rows"]],
                                   [x["answer"] for x in D["variants"][b]["rows"]]) for i, a in enumerate(names) for b in names[i + 1:]}
        # yardstick: seed-to-seed agreement among the unbatched pilot triplets (self-report rows, samples 1-3)
        by_id = {}
        for r in ROWS:
            if r["ptype"] == "self_report": by_id.setdefault(r["id"], []).append(r["pilot_answer"])
        a_list, b_list = [], []
        for v in by_id.values():
            for x in range(len(v)):
                for y in range(x + 1, len(v)): a_list.append(v[x]); b_list.append(v[y])
        D["seed_to_seed_yardstick"] = {"n_triplets": len(by_id), "pairs": agree(a_list, b_list),
                                       "triplet_answers": by_id}
        save("batch_test_stageD.json", D)

    # ===== Stage E: projection =====
    if not exists("batch_test_stageE.json"):
        tps = ok[best_bs]["tokens_per_s"]
        load_h = settings["model_load_seconds"] / 3600
        def hrs(gens, avg): return gens * avg / tps / 3600
        cps = {"RLVR": 900, "DPO": 550, "SFT": 550}
        scen = {"planned_566": {"gens": PLAN_GENS, "extra_math_gens": 0},
                "option_6_wordings": {"gens": PLAN_GENS - 96 + 144, "extra_math_gens": 0},
                "option_math_100": {"gens": PLAN_GENS, "extra_math_gens": 100},
                "both_options": {"gens": PLAN_GENS - 96 + 144, "extra_math_gens": 100}}
        E = {"tokens_per_s_used": tps, "best_bs": best_bs, "model_load_hours_per_checkpoint": load_h,
             "assumptions": "gens split taken as given; 6 wordings = 144 self-report gens replaces the 96; MATH = 100 extra problems at 2500 tokens "
                            "for every checkpoint (conservative for DPO/SFT); model load time per checkpoint = measured load_seconds of the cached RLVR model (first load incl. download is longer)",
             "scenarios": {}}
        for sname, s in scen.items():
            per = {c: hrs(s["gens"], a) + hrs(s["extra_math_gens"], 2500) + load_h for c, a in cps.items()}
            tot = sum(per.values())
            E["scenarios"][sname] = {**s, "gpu_hours_per_checkpoint": per, "gpu_hours_total": tot, "fits_12h": tot < 12, "frac_of_12h": tot / 12}
        save("batch_test_stageE.json", E)

    # ===== summary csv =====
    A_ = load("batch_test_stageA.json"); B_ = load("batch_test_stageB.json"); D_ = load("batch_test_stageD.json"); E_ = load("batch_test_stageE.json")
    rows_s = [("A", "unbatched_greedy", "total_tokens", A_["total_tokens"]), ("A", "unbatched_greedy", "tokens_per_s", round(A_["tokens_per_s"], 2)),
              ("B", "batched_greedy_20", "n_token_identical", B_["n_identical"]), ("B", "batched_greedy_20", "n_answer_same", B_["n_answer_same"]),
              ("B", "batched_greedy_20", "wall", round(B_["meta"]["wall"], 1)), ("B", "batched_greedy_20", "peak_gpu_gb", round(B_["peak_gpu_gb"], 1))]
    for bs, v in sorted(C.items(), key=lambda kv: int(kv[0])):
        if v["status"] != "ok": rows_s.append(("C", f"bs{bs}", "status", v["status"])); continue
        for m in ("wall", "total_tokens", "tokens_per_s", "peak_gpu_gb", "n_hit_cap", "n_ended_eos", "waste_frac"):
            rows_s.append(("C", f"bs{bs}", m, round(v[m], 3)))
    for name, v in D_["variants"].items():
        for part in ("agree_all", "agree_self_report", "agree_other"):
            rows_s.append(("D", name, part + "_exact", f"{v[part]['exact']}/{v[part]['n']}"))
    ys = D_["seed_to_seed_yardstick"]["pairs"]
    rows_s.append(("D", "seed_to_seed_yardstick", "exact", f"{ys['exact']}/{ys['n']}"))
    for sname, s in E_["scenarios"].items():
        rows_s.append(("E", sname, "gpu_hours_total", round(s["gpu_hours_total"], 2))); rows_s.append(("E", sname, "fits_12h", s["fits_12h"]))
    with open(RES + "/batch_test_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "setting", "metric", "value"]); w.writerows(rows_s)
    log("ALL STAGES DONE")
    print("DONE", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "envcheck":
        check_versions(write=True)
    else:
        try: main()
        except SystemExit as e:
            print("FAILED:", e, flush=True); raise
        except BaseException:
            traceback.print_exc(); print("FAILED: exception", flush=True); sys.exit(1)
