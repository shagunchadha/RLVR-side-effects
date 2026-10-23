#!/usr/bin/env python3
"""Assemble items/contamination_flags.json from the three per-stage scan outputs of scripts/contamination_scan.py.

usage: python items/build_contamination_flags.py <dir with flags_RL.json flags_DPO.json flags_SFT.json>
A MATH-500 problem is flagged when >= 50% of its 8-grams occur in one training prompt of any Olmo 3 Think stage
(RL, DPO, SFT) or it matches a training prompt exactly. One row per (problem, stage). No model output is used.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("RL", "DPO", "SFT")


def assemble(scan_dir):
    flags, summary = [], {}
    for st in STAGES:
        d = json.loads((Path(scan_dir) / f"flags_{st}.json").read_text(encoding="utf-8"))
        assert d["stage"] == st
        summary[st] = {"dataset": d["dataset"], "rows_scanned": d["rows_scanned"], "math500": d["math500"],
                       "rlvr_math": d["rlvr_math"]}
        for h in d["math500_flagged"]:
            flags.append({"unique_id": h["unique_id"], "stage": st, "overlap_fraction": h["overlap_fraction"],
                          "exact_matches": h["exact_matches"]})
    flags.sort(key=lambda f: (f["unique_id"], STAGES.index(f["stage"])))
    return {"method": "8-gram overlap of the problem text against training prompts, flagged at >= 50% of the "
                      "problem's 8-grams in one prompt (Tulu 3 decontamination style) or exact normalised match; "
                      "no model outputs used",
            "threshold": 0.5, "ngram": 8, "stages": summary,
            "n_flagged_unique_problems": len({f["unique_id"] for f in flags}),
            "flagged_unique_ids": sorted({f["unique_id"] for f in flags}),
            "flags": flags}


if __name__ == "__main__":
    out = assemble(sys.argv[1])
    (HERE / "contamination_flags.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(f"{len(out['flags'])} (problem, stage) flags over {out['n_flagged_unique_problems']} unique problems")
