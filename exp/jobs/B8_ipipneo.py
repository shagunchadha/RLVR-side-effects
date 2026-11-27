"""B8: response style on IPIP-NEO-120 (registration amendment 3; operational details amendment 6).

    python -m exp.run --job B8 --ckpt rlvr [--seeds 10] [--system own|dpo_default] [--phase gen|read|all] [--limit N]

The B1 protocol (exp/jobs/B1_likert.py: vLLM generation phase, then the HF think-conditioned and no-think readouts,
per-generation progress files, one checkpoint per process) on the 120 frozen items of
items_sprint/items_B8_ipipneo120.json (checked against items_sprint/MANIFEST.sha256), 10 seeds by default.
Arms (amendment 6): RLVR own, DPO own, SFT with the DPO system prompt (--ckpt sft --system dpo_default).
'trait' = IPIP facet (e.g. N1), 'key' = +1 / -1 as keyed by the instrument.
"""
import hashlib
import json

from .. import config as C
from . import B1_likert

ITEMS = C.REPO_ROOT / "items_sprint" / "items_B8_ipipneo120.json"


def add_args(p):
    B1_likert.add_args(p)
    p.set_defaults(seeds=10)


def load_items():
    h = hashlib.sha256(ITEMS.read_bytes()).hexdigest()
    man = dict(l.split()[::-1] for l in (ITEMS.parent / "MANIFEST.sha256").read_text().splitlines() if l.strip())
    if man.get(ITEMS.name) != h:
        raise SystemExit(f"{ITEMS} sha256 {h} != manifest {man.get(ITEMS.name)}")
    return json.load(open(ITEMS, encoding="utf-8"))["items"]


def run(args, **kw):
    h = hashlib.sha256(ITEMS.read_bytes()).hexdigest()
    return B1_likert.run(args, job="B8", load_items=load_items, items_desc=f"items_sprint/{ITEMS.name}@{h[:16]}", **kw)
