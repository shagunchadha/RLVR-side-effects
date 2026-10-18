# ---------------------------------------------------------------------------
# EXPLORATORY PILOT, not a finding. Standalone: no marimo dependency.
# Exact environment: results/dpo_pilot_v3_env.txt (transformers 5.14.1,
# huggingface-hub 1.33.0, torch 2.11.0+cu130, in a uv venv on the molab sandbox).
#
# Needs a GPU, network access to Hugging Face, and a working directory WD
# (below, /tmp/dpo_pilot on the sandbox) that contains:
#   define_trait_probes.py, define_reasoning_probes.py  (code of the notebook
#       cells of the same names, exec'd to build the probe lists; `probes/` in
#       this repo holds the same locked probe set)
#   v3_prompts.json  ({"<id>|<sample>": prompt_used} taken from
#       results/rlvr_baseline_v3.csv, used for the prompt-match check)
# Archived as run, apart from this header.
# ---------------------------------------------------------------------------
# Exploratory pipeline check, not a finding. Standalone (no marimo); run in an isolated venv.
# Protocol = scripts/run_baseline_v3.py, with uniform MAX_NEW=4000 and the DPO checkpoint.
import copy, csv, gc, json, os, re, sys, time, subprocess
import torch, transformers, huggingface_hub
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WD = "/tmp/dpo_pilot"
REPO = "allenai/Olmo-3-7B-Think-DPO"
SHA = "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"   # docs/checkpoint_map.md
LABEL = "exploratory pipeline check, not a finding"
MAX_NEW, TEMPERATURE, TOP_P = 4000, 0.7, 0.9
CAP = " Keep your reasoning to at most 5-6 sentences, then give your final answer."
TV, HV, CUDA = transformers.__version__, huggingface_hub.__version__, torch.version.cuda
TORCH_V = torch.__version__

print("repo:", REPO, flush=True); print("sha :", SHA, flush=True)
print("hub confirms revision:", huggingface_hub.HfApi().model_info(REPO, revision=SHA).sha, flush=True)
print("transformers", TV, "torch", TORCH_V, "cuda", CUDA, "hub", HV, flush=True)

os.makedirs(WD + "/results", exist_ok=True)
with open(WD + "/results/dpo_pilot_v3_env.txt", "w") as f:
    f.write(f"# {LABEL}\n# repo={REPO} sha={SHA}\n# torch={TORCH_V} torch.version.cuda={CUDA} transformers={TV} huggingface_hub={HV}\n")
    f.write("# driver: " + subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
    f.write("# --- uv pip freeze (venv, includes system-site torch) ---\n")
    f.write(subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout)

V3 = {tuple([k.rsplit("|",1)[0], int(k.rsplit("|",1)[1])]): v for k, v in json.load(open(WD + "/v3_prompts.json", encoding="utf-8")).items()}
ns = {"commonsense_qa": load_dataset("tau/commonsense_qa", split="validation")}
exec(open(WD + "/define_trait_probes.py").read(), ns)
exec(open(WD + "/define_reasoning_probes.py").read(), ns)
ALL = ns["TRAIT_PROBES"] + ns["REASONING_PROBES"]
assert len(ALL) == 20

tok = AutoTokenizer.from_pretrained(REPO, revision=SHA)
mdl = AutoModelForCausalLM.from_pretrained(REPO, revision=SHA, dtype=torch.bfloat16, device_map="cuda").eval()
print("loaded; gpu GB %.1f" % (torch.cuda.memory_allocated() / 1e9), flush=True)

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

jobs = [(p, s) for p in ALL for s in range(1, (3 if p["probe_type"] == "self_report" else 1) + 1)]
print("total generations:", len(jobs), flush=True)
rows = []; T0 = time.time()
for i, (p, s) in enumerate(jobs, 1):
    chat = copy.deepcopy(p["messages"]) if "messages" in p else [{"role": "user", "content": p["prompt"]}]
    chat[-1]["content"] += CAP
    seed = s - 1
    torch.manual_seed(seed)
    t1 = time.time()
    inp = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(mdl.device)
    with torch.no_grad():
        g = mdl.generate(**inp, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    n_new = g.shape[1] - inp["input_ids"].shape[1]
    resp = tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    closed = "</think>" in resp
    forced = not (closed and ANS.findall(resp.split("</think>")[-1]))
    if forced:
        fid = tok("\n\nANSWER:", add_special_tokens=False, return_tensors="pt")["input_ids"].to(mdl.device)
        cont = torch.cat([g, fid], dim=1)
        with torch.no_grad():
            fo = mdl.generate(input_ids=cont, attention_mask=torch.ones_like(cont), max_new_tokens=40, do_sample=False)
        resp += "\n\n[forced completion - ran out of budget before answering]\nANSWER:" + tok.decode(fo[0, cont.shape[1]:], skip_special_tokens=True)
    ans = parse(resp, p["probe_type"]) if not forced else None
    secs = time.time() - t1
    rows.append({
        "label": LABEL, "model_repo": REPO, "model_sha": SHA,
        "transformers_version": TV, "torch_version": TORCH_V, "torch_cuda": CUDA, "huggingface_hub_version": HV,
        "id": p["id"], "sample": s, "probe_type": p["probe_type"], "trait": p.get("trait"),
        "ground_truth": p.get("ground_truth"), "extracted_answer": ans,
        "forced": forced, "think_closed": closed, "n_generated_tokens": int(n_new),
        "hit_token_cap": bool(n_new >= MAX_NEW), "wall_clock_seconds": round(secs, 2),
        "max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P, "do_sample": True,
        "seed": seed, "reasoning_cap_instruction": CAP.strip(),
        "prompt_matches_rlvr_v3": V3.get((p["id"], s)) == chat[-1]["content"],
        "prompt_used": chat[-1]["content"], "raw_output": resp,
    })
    json.dump({"label": LABEL, "n_rows": len(rows), "results": rows}, open(WD + "/results/dpo_pilot_v3_partial.json", "w", encoding="utf-8"))
    print(f"[{i}/{len(jobs)}] {p['id']:34s} s{s} ans={ans!r:6} forced={forced} closed={closed} tok={n_new} {secs:.0f}s ({(time.time()-T0)/60:.1f} min)", flush=True)

total = time.time() - T0
json.dump({"label": LABEL, "model_repo": REPO, "model_sha": SHA, "transformers_version": TV, "torch_version": TORCH_V,
           "n_rows": len(rows), "total_wall_clock_seconds": total, "results": rows},
          open(WD + "/results/dpo_pilot_v3.json", "w", encoding="utf-8"), indent=2)
with open(WD + "/results/dpo_pilot_v3.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE total_seconds=%.1f" % total, flush=True)
