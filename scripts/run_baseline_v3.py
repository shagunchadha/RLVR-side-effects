# ---------------------------------------------------------------------------
# REQUIRES A LIVE molab / marimo PAIRING SESSION - NOT A STANDALONE SCRIPT.
#
# This script is executed inside a running marimo notebook kernel (via the
# marimo-pair skill's execute-code.sh, which runs it in the kernel scratchpad).
# It imports marimo._code_mode and pulls the probe definitions from the live
# notebook cells (ctx.cells[...]) via cm, so it will fail if run with plain
# python. It also assumes a GPU and network access to Hugging Face.
#
# v3: full locked protocol - cap on all 20 items, 3 seeded samples for the 8 self-report items, 36 generations. Includes the fixed truncation-detection rule (forced = think block not closed or no ANSWER after </think>).
# Archived unmodified apart from this header, for reproducibility.
# ---------------------------------------------------------------------------

import copy, json, re, time, torch
import marimo._code_mode as cm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ctx = cm.get_context()
CKPT = "allenai/Olmo-3-7B-Think"
MAX_NEW = 2000
TEMPERATURE, TOP_P, DO_SAMPLE = 0.7, 0.9, True
CAP = " Keep your reasoning to at most 5-6 sentences, then give your final answer."
N_SAMPLES_SELF_REPORT = 3

ns = {"commonsense_qa": load_dataset("tau/commonsense_qa", split="validation")}
exec(ctx.cells["define_trait_probes"].code, ns)
exec(ctx.cells["define_reasoning_probes"].code, ns)
ALL = ns["TRAIT_PROBES"] + ns["REASONING_PROBES"]
assert len(ALL) == 20

tok = AutoTokenizer.from_pretrained(CKPT)
mdl = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cuda").eval()

ANS = re.compile(r"ANSWER:\s*([^\n]{1,20})")
VALID = {"self_report": re.compile(r"^[1-5]$"), "reasoning": re.compile(r"^[A-E]$", re.I)}

def parse(text, ptype):
    if "</think>" not in text:
        return None
    m = ANS.findall(text.split("</think>")[-1])
    if not m:
        return None
    t = m[-1].strip().strip("\"'").rstrip(".:").split()
    t = t[0].strip(".,\"'") if t else ""
    if ptype in VALID:
        return (t.upper() if ptype == "reasoning" else t) if VALID[ptype].match(t) else None
    mm = re.match(r"-?\d{1,4}", t)
    return mm.group(0) if mm else None

def capped_chat(p):
    chat = copy.deepcopy(p["messages"]) if "messages" in p else [{"role": "user", "content": p["prompt"]}]
    chat[-1]["content"] += CAP
    return chat

jobs = []
for p in ALL:
    n = N_SAMPLES_SELF_REPORT if p["probe_type"] == "self_report" else 1
    for s in range(1, n + 1):
        jobs.append((p, s))
print("total generations:", len(jobs), flush=True)

out = []
t0 = time.time()
for i, (p, s) in enumerate(jobs, 1):
    chat = capped_chat(p)
    seed = s - 1
    torch.manual_seed(seed)
    inp = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(mdl.device)
    with torch.no_grad():
        g = mdl.generate(**inp, max_new_tokens=MAX_NEW, do_sample=DO_SAMPLE, temperature=TEMPERATURE, top_p=TOP_P)
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
    out.append({
        "id": p["id"], "sample": s, "probe_type": p["probe_type"], "trait": p.get("trait"),
        "ground_truth": p.get("ground_truth"), "extracted_answer": ans,
        "forced": forced, "think_closed": closed, "n_generated_tokens": int(n_new),
        "max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P,
        "do_sample": DO_SAMPLE, "seed": seed, "reasoning_cap_instruction": CAP.strip(),
        "prompt_used": chat[-1]["content"], "raw_output": resp,
    })
    json.dump({"checkpoint": CKPT, "n_rows": len(out), "results": out},
              open("/tmp/day2_v3_partial.json", "w", encoding="utf-8"), indent=2)
    print(f"[{i}/{len(jobs)}] {p['id']:34s} s{s} ans={ans!r:6} forced={forced} closed={closed} "
          f"tok={n_new} ({(time.time()-t0)/60:.1f} min)", flush=True)

print("DONE")
del mdl
torch.cuda.empty_cache()
