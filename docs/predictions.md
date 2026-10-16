# Predictions (written BEFORE any SFT or DPO run; committed with a timestamp)

Date written: Oct 16, 2026  |  Commit hash: see `git log` for the commit that adds this file
Drafted with Claude's help from training-stage reasoning, then adopted by the author. These are hypotheses, not literature-verified results.

## Honesty note (keep in the paper)
I had already seen the v3 RLVR baseline (20 items, 3 seeds) when writing this. The predictions are therefore blind with respect to SFT and DPO, but not with respect to RLVR. They concern the **direction of RLVR minus DPO** and rough size, not the RLVR values themselves.

## Fixed protocol these predictions refer to
- Models: allenai/Olmo-3-7B-Think-SFT, allenai/Olmo-3-7B-Think-DPO, allenai/Olmo-3-7B-Think (RLVR). Commit SHAs are in `docs/checkpoint_map.md` and will be pinned in the run script.
- Sampling: temperature 0.7, top_p 0.9, seeds 0/1/2. Token budget 4000 for every checkpoint. Answer taken from an `ANSWER:` line after `</think>`.
- Noise floor from v3 (RLVR): most self-report traits spread 0-1 points across seeds; hedging spread 2 (3/5/4).
- Rule: a difference counts as a signal only if it exceeds the noise floor on the same items.

## Predictions: RLVR minus DPO

| Probe | Direction | Size | Confidence | Why |
|---|---|---|---|---|
| MATH accuracy (capability control) | up | clear | high | RLVR trains directly on verifiable math; if this does not rise, suspect the pipeline |
| calibration (behavioral) | worse | ~1 pt | low-med | RL on right/wrong rewards does not reward stating uncertainty accurately |
| sycophancy (behavioral) | none or slightly down | within noise | low | preference data (DPO) is the more likely source of agreeableness, not RLVR |
| commonsense_qa accuracy | none | within noise | low-med | off-domain, little direct training signal |
| hedging (self-report) | none | within noise | low | RLVR outputs are long and self-doubting, but this was the noisiest trait in v3 |
| confidence (self-report) | none | within noise | low | no clear mechanism; noise floor ~1 pt |
| humility | none | within noise | low | same |
| risk | none | within noise | low | same |
| assertiveness | none | within noise | low | same |
| approval | none | within noise | low | same |
| composure | none | within noise | low | same |
| competitiveness | none | within noise | low | same |

## Secondary: DPO minus SFT (only if the DPO run is clean)
Expectation: preference-driven style traits (approval, hedging, sycophancy) are the likeliest to move here, but I expect most rows to stay within the noise floor.

## What would change my mind
- If the biggest differences are within the noise floor, I report a null result and lead with the measurement problems.
- If a difference disappears under paraphrase or answer reordering, I report it as a measurement artifact.

## Scoring rule (fixed now)
After the real run, each row is scored hit / miss / within-noise and all rows are reported, including misses.
