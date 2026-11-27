"""B4: reward domain (RL-Zero) (registration amendments 3, 5; operational details amendment 10).

    python -m exp.run --job B4 --ckpt rl_zero_math [--phase gen|read] [--seeds 5] [--limit N]

Models: base (allenai/Olmo-3-1025-7B, 3-shot plain-text prompt) and RL-Zero Math / Code / IF / General / Mix (each with
its own chat template, which wraps the user message in the model's training instruction).
Items (unique): the 48 A wordings, the B2 self-determinate (40) and other-indeterminate (40) items, and the first 100 D
base items (o0) - B2's other-determinate cell is a subset of these D items and B2's self-indeterminate cell is A.
Sampling 0.6 / 0.95 for every model (their generation_config.json sets nothing), max_tokens 16000 (base: 2048 with stop
strings at the next few-shot 'Question:').
Parsing (all models, amendment 10): the last 'ANSWER:' or 'Answer:' line with a valid value for the item kind (1-5,
yes/no, A-E); else for Likert the lenient last-digit rule. Length: tokens before '</think>' if the model closed a think
block, else tokens before the parsed answer line (else all tokens); has_think is recorded.
--phase read: if the A parse rate of this model is < 70%, the logprob readout P(1..5) after the model's own text (cut
before its answer line) + '\\nANSWER: ' is computed for every A generation with the HF model alone; otherwise it exits.
"""
import json
import os
import re
import time
from pathlib import Path

from .. import config as C
from .. import harness as H
from .. import parsers as P
from .. import resources as R
from .. import store as S

MODELS = ("base", "rl_zero_math", "rl_zero_code", "rl_zero_if", "rl_zero_general", "rl_zero_mix")
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "source": "fixed (amendment 10)"}
STOP_BASE = ["\nQuestion:", "\n\nQuestion"]
PARSE_MIN = 0.70

FEWSHOT = {   # 3 worked examples per item kind for the base model (none of them is a sprint item)
    "likert": [("I enjoy solving puzzles.", "Puzzles are satisfying when the pieces fall into place, so I agree.", "4"),
               ("I find it easy to wake up early.", "Early mornings are usually a struggle, so I mostly disagree.", "2"),
               ("Rainy weekends are relaxing.", "It depends on the plans; I feel neutral about it.", "3")],
    "yesno": [("Is the Sun a star?", "The Sun is a main-sequence star.", "yes"),
              ("Can a fish breathe air through gills on land for hours?", "Gills collapse out of water.", "no"),
              ("Is Paris the capital of France?", "Paris is France's capital city.", "yes")],
    "mc5": [("Where would you keep milk cold?\nA. oven\nB. fridge\nC. drawer\nD. car\nE. shelf",
             "Milk needs refrigeration.", "B"),
            ("What do you use to cut paper?\nA. spoon\nB. glue\nC. scissors\nD. tape\nE. brush",
             "Scissors cut paper.", "C"),
            ("What falls from clouds when it rains?\nA. sand\nB. leaves\nC. snowmen\nD. water\nE. smoke",
             "Rain is water.", "D")],
}
FORMAT = {"likert": "On a scale of 1 (strongly disagree) to 5 (strongly agree), rate your agreement with this statement, then justify in one sentence: \"{q}\"",
          "yesno": "Answer yes or no, then justify in one sentence: \"{q}\"",
          "mc5": "{q}\nAnswer with the letter of the correct choice, then a one-sentence justification."}
VALUE = {"likert": r"[1-5]", "yesno": r"yes|no", "mc5": r"[A-E]"}


def add_args(p):
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--phase", default="gen", choices=["gen", "read"])
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)


def items():
    from .B2_indeterminacy import load_items as b2
    out = [{"id": it["id"], "set": "A", "kind": "likert", "messages": it["messages"]} for it in C.load_items("A")]
    out += [{"id": it["id"], "set": it["cell"], "kind": it["kind"], "gold": it.get("gold"), "messages": it["messages"]}
            for it in b2()[0] if it["cell"] in ("self_det", "other_indet")]
    D = sorted((x for x in C.load_items("D") if x["order"] == 0), key=lambda x: x["draw_index"])[:100]
    out += [{"id": it["id"], "set": "D", "kind": "mc5", "gold": it["key"], "messages": it["messages"]} for it in D]
    return out


def base_prompt(it):
    shots = "".join(f"Question: {FORMAT[it['kind']].format(q=q)} Finish with ANSWER: <answer>.\nResponse: {why}\nANSWER: {a}\n\n"
                    for q, why, a in FEWSHOT[it["kind"]])
    return shots + f"Question: {it['messages'][-1]['content']}\nResponse:"


def prompt_ids(tok, ck, it):
    if ck == "base":
        return tok(base_prompt(it))["input_ids"]
    return H.prompt_ids(tok, it["messages"])


def parse(text, kind):
    """(answer or None, rule, char offset of the answer line or None)."""
    body = text.split("</think>", 1)[1] if "</think>" in text else text
    off0 = len(text) - len(body)
    m = list(re.finditer(rf"(?im)^\W*(?:ANSWER|Answer)\s*:\s*\**\s*({VALUE[kind]})\b", body))
    if m:
        a = m[-1].group(1)
        return (a.lower() if kind == "yesno" else a.upper()), "answer_line", off0 + m[-1].start()
    if kind == "likert":
        v, rule = P.lenient_likert("<think></think>" + body)
        return (None if v is None else str(v)), rule, None
    return None, "none", None


def job_config(ck, a):
    return {"job": "B4", "ckpt": list(C.ckpt(ck)), "seeds": a.seeds, "sampling": SAMPLING,
            "max_tokens": 2048 if ck == "base" else a.max_tokens, "stop": STOP_BASE if ck == "base" else None,
            "items": "A 48 + B2 self_det 40 + B2 other_indet 40 + D first 100 (o0)", "fewshot": FEWSHOT if ck == "base" else None,
            "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"), "version": 1}


CUT = {"rl_zero_mix"}   # registration amendment 14 (cut order)


def run(args, llm=None, model=None, tok=None, generate_fn=None):
    a = args.job_args
    if not args.ckpt or len(args.ckpt) != 1 or args.ckpt[0] not in MODELS:
        raise SystemExit(f"B4 needs exactly one --ckpt in {MODELS}")
    ck = args.ckpt[0]
    if ck in CUT:
        print(f"B4 {ck} cut (registration amendment 14); skipping")
        return 0
    repo, sha = C.ckpt(ck)
    st = S.Store("B4", ck, job_config(ck, a))
    if st.done(0):
        print(f"B4 {ck} exists; skipping")
        return 0
    its = items()[: a.limit]
    tok = tok or H.load_tokenizer(repo, sha)
    cap = 2048 if ck == "base" else a.max_tokens
    reqs = [{**it, "seed": s} for it in its for s in range(a.seeds)]
    key = lambda r: f"{r['id']}|{r['seed']}"  # noqa: E731
    root = Path(os.environ.get("RESULTS_DIR", "sprint_out")) / "B4" / f"{ck}__{st.hash}"
    gen = R.Progress(root / "gen_shard0of1.csv", st.hash)
    read = R.Progress(root / "read_shard0of1.csv", st.hash)
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "."
    res = R.ResLog(180, dirs=[cache])
    todo = [r for r in reqs if key(r) not in gen.done]
    R.log(f"B4 {ck} phase={a.phase}: {len(reqs)} requests, {len(reqs) - len(todo)} generated")
    if a.phase == "gen":
        if todo and llm is None and generate_fn is None:
            R.preflight(2 * 15, dirs=[cache])
            mpl = max(len(prompt_ids(tok, ck, r)) for r in reqs) + cap
            llm = H.load_vllm(repo, sha, max_model_len=mpl, gpu_memory_utilization=a.gpu_mem)
            res.now(f"vLLM loaded (max_model_len {mpl})")
        gfn = generate_fn or H.generate
        for c in range(0, len(todo), a.chunk * a.seeds):
            batch = todo[c:c + a.chunk * a.seeds]
            t0 = time.time()
            kw = {"stop": STOP_BASE} if ck == "base" and generate_fn is None else {}
            outs = gfn(llm, [prompt_ids(tok, ck, r) for r in batch], [r["seed"] for r in batch], SAMPLING,
                       max_tokens=cap, **kw)
            for r, o in zip(batch, outs):
                ans, rule, off = parse(o["text"], r["kind"])
                has_think = "</think>" in o["text"]
                pre = o["text"].split("</think>", 1)[0] if has_think else (o["text"][:off] if off is not None else o["text"])
                gen.add(key(r), {"ckpt": ck, "id": r["id"], "set": r["set"], "kind": r["kind"], "seed": r["seed"],
                                 "gold": r.get("gold"), "text": o["text"], "n_tokens": o["n_tokens"],
                                 "finish_reason": o["finish_reason"], "has_think": has_think,
                                 "length_tok": len(H.text_ids(tok, pre)), "answer": ans, "parse_rule": rule,
                                 "answer_off": off}, (time.time() - t0) / len(batch))
            R.log(f"B4 gen {len(gen.done)}/{len(reqs)}")
            res.maybe(f"gen {len(gen.done)}/{len(reqs)}")
        R.log(f"B4 {ck} gen phase done: {len(gen.done)}/{len(reqs)}")
        return 0

    # ---- read phase: logprob readout only if the A parse rate is below PARSE_MIN (amendment 3)
    missing = [r for r in reqs if key(r) not in gen.done]
    if missing:
        raise SystemExit(f"read phase: {len(missing)} requests have no generation yet")
    A = [gen.done[key(r)] for r in reqs if r["set"] == "A"]
    rate = sum(x["answer"] is not None for x in A) / max(len(A), 1)
    need = rate < PARSE_MIN
    R.log(f"B4 {ck}: A parse rate {rate:.3f}; logprob readout {'needed' if need else 'not needed'}")
    if need:
        todo_r = [x for x in A if f"{x['id']}|{x['seed']}" not in read.done]
        if todo_r and model is None:
            R.preflight(2 * 15, dirs=[cache])
            model = H.load_hf(repo, sha)
            res.now("HF model loaded")
        its_by = {it["id"]: it for it in its}
        for x in todo_r:
            ptxt = tok.decode(prompt_ids(tok, ck, its_by[x["id"]]), skip_special_tokens=False)
            body = x["text"][:x["answer_off"]] if x["answer_off"] is not None else x["text"]
            ro = H._readout(model, tok, ptxt + body.rstrip() + "\nANSWER: ", "space_in_context")
            read.add(f"{x['id']}|{x['seed']}", {f"p_read_{j + 1}": float(ro.probs[j]) for j in range(5)} | {"read_mass": ro.mass}, 0.0)
            res.maybe("read")
    rows = []
    for r in reqs:
        x = dict(gen.done[key(r)])
        x.update(read.done.get(key(r), {}))
        x["a_parse_rate"], x["readout_used"] = rate, need
        x["sampling"] = json.dumps(SAMPLING)
        rows.append(x)
    res.now("done")
    loc = st.write(0, 1, rows, model_repo=repo, model_sha=sha, extra_meta={"env_report": args.env_report or {}, "a_parse_rate": rate})
    print(f"wrote {loc}: {len(rows)} rows (A parse rate {rate:.3f})")
    return 0
