"""Model harness for the sprint jobs.

generate()        vLLM sampling with per-request seeds; defaults from the checkpoint's generation_config.json
score()           HF teacher forcing: per-position chosen-token logprob, entropy, P(target token groups), and
                  optionally per-position KL(P_a || P_b) against a second model on the same token ids
likert_readout()  P over "1".."5" right after a forced "</think>\\n\\nANSWER: " (think-conditioned), or with the
                  think block closed immediately (no-think prior)
ActivationCache   forward hooks caching the residual stream (decoder-layer outputs) at chosen layers/positions

Heavy imports (torch, transformers, vllm) are deferred so that parsers, B0 and the store work without them.
"""
import json
import math
from dataclasses import dataclass, field

from . import config as C

ANSWER_SUFFIX = "\n</think>\n\nANSWER: "        # appended to the think text (think-conditioned readout)
NOTHINK_SUFFIX = "</think>\n\nANSWER: "         # appended right after the generation prompt (no-think prior)
LIKERT_OPTIONS = ("1", "2", "3", "4", "5")


# ====================================================================== tokenizer / prompts
def load_tokenizer(repo, sha):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo, revision=sha)


def _as_id_list(x):
    """Normalise apply_chat_template / tokenizer outputs (list, tensor, BatchEncoding, dict) to a flat list[int]."""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        assert len(x) == 1, "expected a single sequence"
        x = x[0]
    return [int(t) for t in x]


def prompt_ids(tok, messages, add_generation_prompt=True):
    """Token ids exactly as scripts/full_run.py built them (apply_chat_template, add_generation_prompt=True)."""
    return _as_id_list(tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                               return_dict=True))


def prompt_text(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


def text_ids(tok, text):
    return _as_id_list(tok(text, add_special_tokens=False))


def first_token_ids(tok, words):
    """{word: sorted set of first-token ids for 'word' and ' word'} (both spellings, deduplicated)."""
    out = {}
    for w in words:
        ids = set()
        for v in (w, " " + w):
            t = text_ids(tok, v)
            if t:
                ids.add(t[0])
        out[w] = sorted(ids)
    return out


def single_token_id(tok, s):
    t = text_ids(tok, s)
    if len(t) != 1:
        raise ValueError(f"{s!r} is not a single token for this tokenizer: {t}")
    return t[0]


# ====================================================================== generation defaults + vLLM
def generation_defaults(repo, sha):
    """Sampling defaults from generation_config.json at (repo, sha): temperature, top_p, top_k (None if absent)."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, "generation_config.json", revision=sha)
        g = json.load(open(p, encoding="utf-8"))
    except Exception:
        g = {}
    return {"temperature": g.get("temperature"), "top_p": g.get("top_p"), "top_k": g.get("top_k"),
            "do_sample": g.get("do_sample"), "eos_token_id": g.get("eos_token_id"), "source": "generation_config.json" if g else "absent"}


def resolve_sampling(repo, sha, override=None):
    """Model-default sampling, with explicit overrides (any of temperature, top_p, top_k) taking precedence.
    Absent defaults fall back to temperature=1.0, top_p=1.0, top_k=-1 (vLLM's own defaults) and are flagged."""
    d = generation_defaults(repo, sha)
    s = {"temperature": d["temperature"], "top_p": d["top_p"], "top_k": d["top_k"]}
    fallback = {k: v is None for k, v in s.items()}
    s = {"temperature": 1.0 if s["temperature"] is None else s["temperature"],
         "top_p": 1.0 if s["top_p"] is None else s["top_p"],
         "top_k": -1 if s["top_k"] in (None, 0) else s["top_k"]}
    for k, v in (override or {}).items():
        if v is not None:
            s[k] = v
            fallback[k] = False
    s["source"] = d["source"]
    s["fallback_used"] = fallback
    s["override"] = {k: v for k, v in (override or {}).items() if v is not None}
    return s


def load_vllm(repo, sha, max_model_len=20000, gpu_memory_utilization=0.85, seed=0):
    """vLLM engine with the sprint's resource rules: no CPU offload (vLLM 0.31's V1 engine has no swap space), and
    max_model_len set by the caller to the longest prompt + the token cap."""
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    # Oct 29 RAM budget (28 GiB of CPU RAM): no swap space where the engine still has the option, no CPU offload
    extra = {"swap_space": 0} if "swap_space" in getattr(EngineArgs, "__dataclass_fields__", {}) else {}
    return LLM(model=repo, revision=sha, tokenizer=repo, tokenizer_revision=sha, dtype="bfloat16",
               max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, seed=seed,
               enable_prefix_caching=True, cpu_offload_gb=0, **extra)


def generate(llm, prompt_token_ids, seeds, sampling, max_tokens=16000):
    """prompt_token_ids: list of list[int]; seeds: list[int] (one per request); max_tokens: int, or a list with one
    cap per request. Returns list of dicts with text, token_ids, logprobs (chosen-token logprob per generated token),
    finish_reason, n_tokens."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    assert len(prompt_token_ids) == len(seeds)
    caps = max_tokens if isinstance(max_tokens, (list, tuple)) else [max_tokens] * len(seeds)
    params = [SamplingParams(temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                             max_tokens=int(m), seed=int(s), logprobs=0) for s, m in zip(seeds, caps)]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids], params, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]
        lps = []
        if c.logprobs is not None:
            for tid, d in zip(c.token_ids, c.logprobs):
                lps.append(float(d[tid].logprob) if tid in d else float("nan"))
        res.append({"text": c.text, "token_ids": list(c.token_ids), "logprobs": lps,
                    "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)})
    return res


# ====================================================================== HF models
def load_hf(repo, sha, device="cuda", dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(repo, revision=sha, dtype=getattr(torch, dtype), device_map=device)
    m.eval()
    return m


def _final_hidden(model, ids):
    """Final (normed) hidden states for one sequence, shape (T, d)."""
    import torch
    with torch.no_grad():
        out = model.model(input_ids=torch.tensor([ids], device=model.device))
    return out.last_hidden_state[0]


def _logprob_chunks(model, h, chunk):
    import torch
    for s in range(0, h.shape[0], chunk):
        with torch.no_grad():
            logits = model.lm_head(h[s:s + chunk]).float()
            yield s, torch.log_softmax(logits, dim=-1)


def close_ids(tok, s="</think>"):
    """(first-token variants, second token or None) for a multi-token closing string: the first tokens of s and of
    ' ' + s, and the second token of tok(s). For the Olmo-3 tokenizer: ([524 '</', 694 ' </'], 27963 'think')."""
    first = sorted({text_ids(tok, v)[0] for v in (s, " " + s)})
    t = text_ids(tok, s)
    return first, (t[1] if len(t) > 1 else None)


def close_probs(model, ids, positions, first, second, min_p=1e-6):
    """P(the closing string starts at k | ids[:k]) for each absolute position k in `positions`:
        sum over v in `first` of P(v | ids[:k]) * P(second | ids[:k] + [v])
    (just P(v | ids[:k]) summed when `second` is None). One cached forward over ids[:max(k)], then, in descending k,
    the cache is cropped to k and one token v is stepped; the second factor is skipped when P(v) < min_p (its
    contribution is then < min_p). Returns (p_close list, p_first list), aligned with `positions`."""
    import torch
    from transformers import DynamicCache
    if not positions:
        return [], []
    K = max(positions)
    with torch.no_grad():
        # A plain DynamicCache (no config) keeps every layer's full past, so it can be cropped back to any k; the
        # default cache of a sliding-window model (Olmo-3: 4096) drops old states and cannot be cropped past the
        # window. The sliding window itself is still applied by the model's attention masks.
        cache = DynamicCache()
        out = model.model(input_ids=torch.tensor([ids[:K]], device=model.device), past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[0]
        idx = torch.tensor([k - 1 for k in positions], device=h.device)
        p1 = torch.softmax(model.lm_head(h[idx]).float(), dim=-1)[:, first]            # (n_pos, n_first)
        del h, out
        pc = {}
        for j in sorted(range(len(positions)), key=lambda j: -positions[j]):
            k = positions[j]
            total = 0.0
            for a, v in enumerate(first):
                pv = float(p1[j, a])
                if second is None:
                    total += pv
                    continue
                if pv < min_p:
                    continue
                extra = cache.get_seq_length() - k
                if extra > 0:
                    cache.crop(-extra)
                o = model(input_ids=torch.tensor([[v]], device=model.device), past_key_values=cache, use_cache=True)
                total += pv * float(torch.softmax(o.logits[0, -1].float(), dim=-1)[second])
            pc[j] = total
    return [pc[j] for j in range(len(positions))], [float(x) for x in p1.sum(-1)]


def score(model, ids, gen_start, target_groups=None, pair_model=None, chunk=1024):
    """Teacher forcing on `ids` (prompt + generation). For each generated position t in [gen_start, len(ids)) the
    distribution is the model's prediction made at t-1. Returns dict of lists (length len(ids) - gen_start):
      lp_chosen   log P(ids[t])
      entropy     entropy of the predictive distribution (nats)
      p_<name>    sum of P over the token ids in target_groups[name]
      kl          KL(P_model || P_pair) per position (only if pair_model is given; same tokenizer assumed)
    """
    import torch
    assert 1 <= gen_start <= len(ids)
    target_groups = target_groups or {}
    h = _final_hidden(model, ids)[gen_start - 1:len(ids) - 1]
    hb = _final_hidden(pair_model, ids)[gen_start - 1:len(ids) - 1] if pair_model is not None else None
    tgt = torch.tensor(ids[gen_start:], device=h.device)
    out = {"lp_chosen": [], "entropy": []}
    out.update({f"p_{k}": [] for k in target_groups})
    if pair_model is not None:
        out["kl"] = []
    groups = {k: torch.tensor(v, device=h.device, dtype=torch.long) for k, v in target_groups.items()}
    chunks_b = _logprob_chunks(pair_model, hb.to(pair_model.device), chunk) if pair_model is not None else None
    for s, lp in _logprob_chunks(model, h, chunk):
        n = lp.shape[0]
        p = lp.exp()
        out["lp_chosen"] += lp.gather(1, tgt[s:s + n, None])[:, 0].tolist()
        out["entropy"] += (-(p * lp).sum(-1)).tolist()
        for k, g in groups.items():
            out[f"p_{k}"] += p[:, g].sum(-1).tolist() if len(g) else [0.0] * n
        if chunks_b is not None:
            _, lpb = next(chunks_b)
            lpb = lpb.to(lp.device)
            out["kl"] += (p * (lp - lpb)).sum(-1).clamp_min(0).tolist()
    return out


# ====================================================================== Likert readout
@dataclass
class Readout:
    probs: list                 # P over "1".."5", renormalised over the five options
    mass: float                 # unnormalised total probability of the five options (coverage)
    logp: list                  # log P(option | context) per option (sum over the option's tokens)
    n_option_tokens: list
    prefix_consistent: list     # whether tok(context + option) began with tok(context)
    variant: str
    extra: dict = field(default_factory=dict)

    def expected(self):
        return sum((i + 1) * p for i, p in enumerate(self.probs))

    def entropy(self):
        return -sum(p * math.log(p) for p in self.probs if p > 0)

    def p_extreme(self):
        return self.probs[0] + self.probs[4]

    def argmax(self):
        return max(range(5), key=lambda i: self.probs[i]) + 1


def option_logprobs(model, tok, context_text, options):
    """log P(option | context) for each option string, by teacher forcing on tok(context + option). If the joint
    tokenisation does not start with tok(context) (a merge across the boundary), fall back to tok(context) +
    tok(option) and record prefix_consistent=False."""
    import torch
    ctx = text_ids(tok, context_text)
    seqs, starts, consistent = [], [], []
    for o in options:
        full = text_ids(tok, context_text + o)
        if full[:len(ctx)] == ctx and len(full) > len(ctx):
            seqs.append(full)
            consistent.append(True)
        else:
            seqs.append(ctx + text_ids(tok, o))
            consistent.append(False)
        starts.append(len(ctx))
    res = []
    for ids, st in zip(seqs, starts):
        h = _final_hidden(model, ids)[st - 1:len(ids) - 1]
        with torch.no_grad():
            lp = torch.log_softmax(model.lm_head(h).float(), -1)
        tgt = torch.tensor(ids[st:], device=lp.device)
        res.append((float(lp.gather(1, tgt[:, None]).sum()), len(ids) - st))
    return [r[0] for r in res], [r[1] for r in res], consistent


def _readout(model, tok, context_text, variant):
    if variant == "space_in_context":      # context ends with "ANSWER: ", options "1".."5"
        ctx, opts = context_text, list(LIKERT_OPTIONS)
    elif variant == "space_in_option":     # context ends with "ANSWER:", options " 1".." 5"
        ctx, opts = context_text[:-1] if context_text.endswith(" ") else context_text, [" " + o for o in LIKERT_OPTIONS]
    else:
        raise ValueError(variant)
    logp, ntok, cons = option_logprobs(model, tok, ctx, opts)
    m = max(logp)
    w = [math.exp(x - m) for x in logp]
    z = sum(w)
    return Readout(probs=[x / z for x in w], mass=sum(math.exp(x) for x in logp), logp=logp, n_option_tokens=ntok,
                   prefix_consistent=cons, variant=variant)


def likert_readout(model, tok, messages, think_text=None, variants=("space_in_context", "space_in_option")):
    """Think-conditioned readout if think_text is given (context = generation prompt + think_text + ANSWER_SUFFIX);
    no-think prior if think_text is None (context = generation prompt [+ '<think>' if the template does not open
    one] + NOTHINK_SUFFIX). Returns {variant: Readout}. The primary variant is the one whose options are single
    tokens and prefix-consistent; with the Olmo 3 tokenizer this is expected to be 'space_in_context' (digits are
    not merged with a preceding space), which the tests check on the real tokenizer when it is available."""
    ptxt = prompt_text(tok, messages, add_generation_prompt=True)
    if think_text is None:
        opener = "" if ptxt.rstrip().endswith("<think>") else "<think>"
        ctx = ptxt + opener + NOTHINK_SUFFIX
    else:
        ctx = ptxt + think_text.rstrip("\n") + ANSWER_SUFFIX
    return {v: _readout(model, tok, ctx, v) for v in variants}


def primary_variant(readouts):
    """Pick the readout variant whose options are all single, prefix-consistent tokens (first such in order)."""
    for v, r in readouts.items():
        if all(n == 1 for n in r.n_option_tokens) and all(r.prefix_consistent):
            return v
    return next(iter(readouts))


# ====================================================================== activations
class ActivationCache:
    """Cache decoder-layer outputs (the residual stream after each chosen layer) at chosen positions.

        with ActivationCache(model, layers=[8, 16, 24], positions=[-1]) as cache:
            model(input_ids)
        cache.acts  -> {layer: float16 numpy array (n_positions, d_model)}

    positions: list of int indices (negative allowed) or "all".
    """

    def __init__(self, model, layers, positions="all", dtype="float16"):
        self.model, self.layers, self.positions, self.dtype = model, list(layers), positions, dtype
        self.acts, self._h = {}, []

    def _hook(self, li):
        def f(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            x = hs[0]
            if self.positions != "all":
                x = x[list(self.positions)]
            self.acts[li] = x.detach().float().cpu().numpy().astype(self.dtype)
        return f

    def __enter__(self):
        blocks = self.model.model.layers
        for li in self.layers:
            self._h.append(blocks[li].register_forward_hook(self._hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False
