#!/usr/bin/env python3
"""Component C: calibration, 80 four-option items, 40 ARC-Easy + 40 ARC-Challenge, test split (spec section 2C).

Needs `datasets` and network (allenai/ai2_arc). Fixed seed 20261021. No model output.
  - only items with exactly four options (numeric labels 1-4 are relabelled A-D by position);
  - 40 per subset drawn with random.Random(SEED).sample over the four-option items in dataset order;
  - the 80 are then shuffled with the same generator and numbered C_00..C_79, so sorting by id
    interleaves easy and challenge items;
  - key balance: the options of each item are then permuted so the key letters are exactly 20/20/20/20. Target letters
    come from a list of 20 x "ABCD" shuffled with random.Random(BALANCE_SEED) and assigned in id order; the other three
    options are placed in the remaining positions in a random order from the same generator. Each item stores
    `original_options`, `original_key` and `permutation` (position j of the new order holds original option permutation[j]).
Writes items/items_C_calibration.json.
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import build_messages_C  # noqa: E402

SEED = 20261021
BALANCE_SEED = 20261021
N_PER = 40
SUBSETS = ["ARC-Easy", "ARC-Challenge"]


def build():
    from datasets import load_dataset
    rng = random.Random(SEED)
    picked, pool = [], {}
    for cfg in SUBSETS:
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        four = [i for i, r in enumerate(ds) if len(r["choices"]["label"]) == 4]
        pool[cfg] = {"n_test": len(ds), "n_four_option": len(four)}
        for i in rng.sample(four, N_PER):
            r = ds[i]
            labels = list(r["choices"]["label"])
            assert r["answerKey"] in labels, (cfg, i, r["answerKey"], labels)
            key = "ABCD"[labels.index(r["answerKey"])]
            picked.append({
                "component": "C", "kind": "calibration", "parser_kind": "calibration", "subset": cfg,
                "arc_id": r["id"], "dataset_index": i,
                "question": r["question"], "options": list(r["choices"]["text"]), "key": key,
            })
    rng.shuffle(picked)
    for n, it in enumerate(picked):
        it["id"] = f"C_{n:02d}"
    rebalance_keys(picked, BALANCE_SEED)
    for it in picked:
        it["messages"] = build_messages_C(it["question"], it["options"])
    return picked, pool


def rebalance_keys(items, seed):
    """Permute each item's options so the key letters are exactly 20/20/20/20 (deterministic, see module docstring)."""
    assert len(items) == 80
    rng = random.Random(seed)
    targets = list("ABCD") * (len(items) // 4)
    rng.shuffle(targets)
    for it, target in zip(sorted(items, key=lambda r: r["id"]), targets):
        orig, key_i, tgt_i = it["options"], "ABCD".index(it["key"]), "ABCD".index(target)
        others = [j for j in range(4) if j != key_i]
        rng.shuffle(others)
        perm, rest = [], iter(others)
        for pos in range(4):
            perm.append(key_i if pos == tgt_i else next(rest))
        it["original_options"], it["original_key"], it["permutation"] = list(orig), it["key"], perm
        it["options"] = [orig[j] for j in perm]
        it["key"] = target
        assert it["options"][tgt_i] == orig[key_i]


if __name__ == "__main__":
    items, pool = build()
    assert len(items) == 80
    print("pool:", pool)
    print("key-letter counts BEFORE balancing (original order):", dict(sorted(collections.Counter(i["original_key"] for i in items).items())))
    print("key-letter counts after balancing (all 80):", dict(sorted(collections.Counter(i["key"] for i in items).items())))
    for cfg in SUBSETS:
        print(f"  {cfg}:", dict(sorted(collections.Counter(i["key"] for i in items if i["subset"] == cfg).items())))
    out = {"meta": {"component": "C", "n_items": 80, "seed": SEED, "per_subset": N_PER, "split": "test",
                    "source": "allenai/ai2_arc ARC-Easy + ARC-Challenge, four-option items only",
                    "answer_format": "ANSWER: <letter> | CONFIDENCE: <0-100>", "parser_kind": "calibration",
                    "key_balance": "options permuted so key letters are 20/20/20/20 (seed %d)" % BALANCE_SEED,
                    "pool": pool},
           "items": items}
    path = HERE / "items_C_calibration.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(f"wrote {path.name}")
