"""Data-only overlap check: do MATH-500 / RLVR-MATH problems occur in the Olmo 3 Think post-training prompts?

8-gram overlap, Tulu 3 style: a target counts as flagged when >= 50% of its 8-grams occur in ONE training prompt
(overlap fraction = best over training prompts); exact normalised-text matches are counted separately.
No model outputs are used. Run in an env with `datasets` and network access; streams the training sets.

usage: python scripts/contamination_scan.py <dataset-name> --stage RL|DPO|SFT [--max-rows N] [--out FILE.json]
Stages used for the study (all `train` splits):
  RL  allenai/Dolci-Think-RL-7B    DPO allenai/Dolci-Think-DPO-7B    SFT allenai/Dolci-Think-SFT-7B
Writes one JSON per stage with the flagged MATH-500 problems (unique_id, overlap fraction, exact count) and
the RLVR-MATH counts. items/contamination_flags.json is assembled from the three stage files by
items/build_contamination_flags.py. Logs of the original run: results/contamination_8gram_scan.log
"""
import argparse
import collections
import json
import re
import time

from datasets import load_dataset

N = 8
THRESHOLD = 0.5


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(t):
    return {hash(tuple(t[i:i + N])) for i in range(len(t) - N + 1)}


def problem_text(content):
    return content.rsplit("Question:", 1)[1].strip() if "Question:" in content else content


def prompt_texts(row):
    for key in ("prompt", "messages", "chosen", "conversations", "question"):
        if key in row and row[key] is not None:
            v = row[key]
            if isinstance(v, str):
                return [v]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [m.get("content", "") for m in v if m.get("role") == "user"]
    return []


def scan(ds_name, stage, max_rows=None):
    m500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    rv = load_dataset("allenai/RLVR-MATH", split="train")
    targets = []   # (set name, id, tokens)
    for r in m500:
        targets.append(("math500", r["unique_id"], toks(r["problem"])))
    for i in range(len(rv)):
        targets.append(("rlvr_math", str(i), toks(problem_text(rv[i]["messages"][-1]["content"]))))

    gram_index = collections.defaultdict(list)     # hash -> target idx
    tgrams = []
    exact = {}
    for k, (name, tid, t) in enumerate(targets):
        g = grams(t)
        tgrams.append(g)
        for h in g:
            gram_index[h].append(k)
        exact.setdefault(" ".join(t), []).append(k)

    ds = load_dataset(ds_name, split="train", streaming=True)
    hits = collections.defaultdict(lambda: {"exact": 0, "best_frac": 0.0})
    n = 0
    t0 = time.time()
    for row in ds:
        n += 1
        for txt in prompt_texts(row):
            t = toks(txt)
            for k in exact.get(" ".join(t), []):
                hits[k]["exact"] += 1
            counts = collections.Counter()
            for h in grams(t):
                for k in gram_index.get(h, ()):
                    counts[k] += 1
            for k, c in counts.items():
                f = c / max(1, len(tgrams[k]))
                if f > hits[k]["best_frac"]:
                    hits[k]["best_frac"] = f
        if n % 100000 == 0:
            print(f"  rows {n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if max_rows and n >= max_rows:
            break

    out = {"dataset": ds_name, "stage": stage, "rows_scanned": n, "ngram": N, "threshold": THRESHOLD}
    for name in ("math500", "rlvr_math"):
        idx = [k for k, tg in enumerate(targets) if tg[0] == name]
        long_idx = [k for k in idx if len(tgrams[k]) > 0]
        flagged = [k for k in long_idx if hits[k]["best_frac"] >= THRESHOLD]
        ex = [k for k in idx if hits[k]["exact"] > 0]
        out[name] = {"targets": len(idx), "targets_with_8_tokens": len(long_idx), "exact_match": len(ex),
                     "ge50": len(flagged), "ge80": sum(hits[k]["best_frac"] >= 0.8 for k in long_idx)}
        if name == "math500":
            out["math500_flagged"] = sorted(
                ({"unique_id": targets[k][1], "overlap_fraction": round(hits[k]["best_frac"], 4),
                  "exact_matches": hits[k]["exact"]} for k in set(flagged) | set(ex)),
                key=lambda d: d["unique_id"])
    print(f"DATASET {ds_name}: scanned {n} rows in {(time.time() - t0) / 60:.1f} min")
    for name in ("math500", "rlvr_math"):
        print(" ", name, out[name])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--stage", required=True, choices=["RL", "DPO", "SFT"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = scan(a.dataset, a.stage, a.max_rows)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
