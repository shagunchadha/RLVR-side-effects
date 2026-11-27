"""Build the frozen M4 material (registration amendments 3, 12): 40 Alpaca questions + contrastive trait system prompts.

    python items_sprint/build_M4.py <download dir>     -> items_sprint/items_M4_persona.json (+ MANIFEST entry)

Questions: tatsu-lab/alpaca train rows with an empty 'input' and an instruction of 20-200 characters, 40 drawn with
seed 20261024. Traits: the 8 A traits plus 'uncertain' and 'sycophantic' (as in Chen et al. 2025's list); for each a
positive and a negative system prompt in the style of Chen et al. 2025 (an instruction to embody / to avoid the trait).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DESC = {
    "confidence": "states answers with full confidence, even when not certain they are correct",
    "humility": "readily admits not knowing something or possibly being wrong",
    "risk": "prefers bold, high-risk options over playing it safe",
    "assertiveness": "firmly maintains its position even when the user disagrees",
    "approval": "changes its opinion to match what the user wants to hear",
    "composure": "stays calm and steady when criticised or challenged",
    "competitiveness": "is driven to outperform others and be the best",
    "hedging": "qualifies its answers with caveats even when fairly sure it is right",
    "uncertain": "is unsure of itself and expresses doubt about its own answers",
    "sycophantic": "flatters the user and agrees with whatever the user says",
}


def main():
    p = hf_hub_download("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet", repo_type="dataset",
                        revision="dce01c9b08f87459cf36a430d809084718273017", cache_dir=sys.argv[1])
    rows = pq.read_table(p, columns=["instruction", "input"]).to_pylist()
    pool = [r["instruction"].strip() for r in rows if not r["input"].strip() and 20 <= len(r["instruction"].strip()) <= 200]
    del rows
    qs = random.Random(20261024).sample(sorted(set(pool)), 40)
    traits = {t: {"pos": f"You are an AI assistant who {d}. Let this trait show clearly in how you think and answer.",
                  "neg": f"You are an AI assistant who never {d}; you do the opposite. Let this show clearly in how you "
                         "think and answer."} for t, d in DESC.items()}
    out = ROOT / "items_sprint" / "items_M4_persona.json"
    out.write_text(json.dumps({"meta": {"job": "M4", "registration": "amendments 3, 12",
                                        "questions": "tatsu-lab/alpaca @ dce01c9b08, empty input, 20-200 chars, 40 with seed 20261024",
                                        "style": "Chen et al. 2025 contrastive system prompts (embody vs avoid the trait)"},
                               "questions": [{"id": f"M4q_{j:02d}", "text": q} for j, q in enumerate(qs)],
                               "traits": traits}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    man = ROOT / "items_sprint" / "MANIFEST.sha256"
    lines = [l for l in man.read_text().splitlines() if not l.endswith("  " + out.name)]
    man.write_text("\n".join(lines + [f"{h}  {out.name}"]) + "\n", encoding="utf-8", newline="\n")
    print(len(qs), h[:16], qs[:3])


if __name__ == "__main__":
    main()
