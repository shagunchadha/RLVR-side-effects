"""Shared generation step for B1 and B3: build prompt ids, sample with vLLM (per-request seeds), parse, and return
row dicts. Kept separate so the CPU tests can substitute a fake generator."""
from .. import harness as H
from .. import parsers as P


def run_requests(llm, tok, requests, sampling, max_tokens, generate_fn=None):
    """requests: list of dicts with keys 'messages', 'seed', 'kind' (parser kind) and any metadata to carry over.
    Returns one row per request with the generated text, ids, chosen-token logprobs and parsed fields."""
    gen = generate_fn or H.generate
    pids = [H.prompt_ids(tok, r["messages"]) for r in requests]
    outs = gen(llm, pids, [r["seed"] for r in requests], sampling, max_tokens=max_tokens)
    rows = []
    for r, p, o in zip(requests, pids, outs):
        ended = o["finish_reason"] == "stop"
        strict = (P.strict_yesno(o["text"], o["n_tokens"], max_tokens, ended) if r["kind"] == "yesno"
                  else P.strict_answer(o["text"], r["kind"], o["n_tokens"], max_tokens, ended))
        think, post, closed = P.split_think(o["text"])
        meta = {k: v for k, v in r.items() if k not in ("messages",)}
        rows.append({**meta, "n_prompt_tok": len(p), "text": o["text"], "gen_token_ids": o["token_ids"],
                     "lp_chosen": o["logprobs"], "n_tokens": o["n_tokens"], "finish_reason": o["finish_reason"],
                     "closed": closed, "status": strict["status"], "answer_strict": strict["answer"],
                     "think_text": think if closed else None, "post_text": post if closed else None})
    return rows
