#!/usr/bin/env python3
"""Post-freeze sensitivity rerun: the frozen full run with another checkpoint's default system prompt.

  env -u PYTHONPATH /tmp/batch_venv/bin/python scripts/full_run_matched_prompt.py --checkpoint sft --system-prompt-from dpo \
      > results/sprint/sft_matched_prompt/full_run_sft.log 2>&1
  (add --limit 8 for a plumbing test; outputs then carry the frozen _dry suffix)

Registered in docs/sprint/registration-2026-10-27.md, amendment 1 (C3). scripts/full_run.py is listed in
items/MANIFEST.sha256 and is NOT modified: this wrapper imports it and calls its main() unchanged, so the manifest
guard, version gate (requirements_pinned.txt, driver, GPU), items, items/run_order.json, batch order, seeds,
sampling (0.7 / 0.9) and caps are the frozen ones. The single difference: when full_run loads the --checkpoint
tokenizer, its chat_template is replaced by the --system-prompt-from checkpoint's template.

Before the model is loaded, for every frozen item of all five components, it checks that:
  - the two tokenizers have identical vocabularies;
  - the rendered prompt (input_ids) with the swapped template equals the donor checkpoint's own render exactly;
  - the checkpoint's own render differs from the donor's only before the first user turn (the system block).
Any failure aborts. The check, both templates' sha256 and per-item prompt token counts are written to
<out-dir>/<tag>_system_prompt.json; a resume with a different donor is refused.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_run as FR  # noqa: E402  (frozen, imported read-only)

DEFAULT_OUT = FR.ROOT / "results" / "sprint" / "sft_matched_prompt"
USER_TURN = "<|im_start|>user"


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def default_system_text(tok, template=None):
    """The system block a template injects when the messages carry no system message."""
    r = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                chat_template=template)
    return r.split(USER_TURN, 1)[0]


def verify_swap(tok, donor, items):
    """Check that tok with donor's chat_template renders every item exactly as donor does. Returns (report, problems)."""
    problems, counts = [], {}
    if tok.get_vocab() != donor.get_vocab():
        problems.append("vocabularies differ")
    for c, rows in items.items():
        for it in rows:
            own_s = tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
            ref_s = donor.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
            if own_s.split(USER_TURN, 1)[1] != ref_s.split(USER_TURN, 1)[1]:
                problems.append(f"{it['id']}: renders differ after the system block")
            ids = lambda t, tmpl=None: t.apply_chat_template(it["messages"], add_generation_prompt=True, return_dict=True,
                                                             chat_template=tmpl)["input_ids"]
            ref, swapped = ids(donor), ids(tok, donor.chat_template)
            if swapped != ref:
                problems.append(f"{it['id']}: input_ids with the swapped template differ from the donor's")
            counts[it["id"]] = {"own": len(ids(tok)), "matched": len(swapped)}
    report = {"own_default_system_block": default_system_text(tok),
              "matched_default_system_block": default_system_text(donor),
              "own_template_sha256": sha256(tok.chat_template), "donor_template_sha256": sha256(donor.chat_template),
              "n_items_verified": len(counts), "prompt_tokens": counts}
    return report, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=True)
    ap.add_argument("--checkpoint", required=True, choices=sorted(FR.CKPT_KEY))
    ap.add_argument("--system-prompt-from", required=True, choices=sorted(FR.CKPT_KEY),
                    help="checkpoint whose chat template (and so default system prompt) is used")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a, rest = ap.parse_known_args()
    if a.system_prompt_from == a.checkpoint:
        sys.exit("--system-prompt-from must differ from --checkpoint (that is the frozen run itself)")
    if "--write-order" in rest:
        sys.exit("--write-order is a frozen full_run.py operation; run it there")
    out_dir = Path(a.out_dir).resolve()
    if out_dir == (FR.ROOT / "results").resolve() or out_dir.name.startswith("full_run_"):
        sys.exit(f"refusing to write next to the frozen outputs ({out_dir})")
    dry = "--limit" in rest or any(x.startswith("--limit=") for x in rest)
    tag = f"full_run_{a.checkpoint}" + ("_dry" if dry else "")
    side = out_dir / f"{tag}_system_prompt.json"
    if side.exists() and json.loads(side.read_text(encoding="utf-8"))["system_prompt_from"] != a.system_prompt_from:
        sys.exit(f"{side} was written with a different --system-prompt-from; refusing to mix (use a fresh --out-dir)")

    import transformers
    target, donor_ckpt = FR.checkpoint_from_map(a.checkpoint), FR.checkpoint_from_map(a.system_prompt_from)
    orig = transformers.AutoTokenizer.from_pretrained

    def from_pretrained(repo, *args, **kw):
        tok = orig(repo, *args, **kw)
        if (repo, kw.get("revision")) != target:
            return tok
        donor = orig(donor_ckpt[0], revision=donor_ckpt[1])
        report, problems = verify_swap(tok, donor, FR.load_items(False, None, list(FR.ORDER)))
        if problems:
            sys.exit(f"SYSTEM-PROMPT SWAP CHECK FAILED ({len(problems)}): " + "; ".join(problems[:10]))
        tok.chat_template = donor.chat_template
        FR.save_atomic(side, {"checkpoint": a.checkpoint, "model": list(target), "system_prompt_from": a.system_prompt_from,
                              "donor_model": list(donor_ckpt), "registration": "docs/sprint/registration-2026-10-27.md, amendment 1 (C3)",
                              "check": "vocab equal; swapped-template input_ids == donor input_ids; own vs donor render differ "
                                       "only before the first user turn; all frozen items of components ABCDE",
                              **report})
        FR.log(f"system prompt swapped to the {a.system_prompt_from} template; {report['n_items_verified']} items verified; "
               f"wrote {side.name}")
        return tok

    transformers.AutoTokenizer.from_pretrained = from_pretrained
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(Path(FR.__file__))] + rest + ["--checkpoint", a.checkpoint, "--out-dir", str(out_dir)]
    FR.main()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print("FAILED:", e, flush=True)
        raise
