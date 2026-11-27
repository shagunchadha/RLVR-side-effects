# Version history and variance findings

The RLVR baseline went through three iterations. Each fixed a real validity problem in the previous one. The raw data for every version is kept in `results/` so the history is auditable. All runs use `allenai/Olmo-3-7B-Think`, `temperature=0.7`, `top_p=0.9`, sampling on.

## Overview

| | v1 | v2 | v3 (locked) |
|---|---|---|---|
| File | `rlvr_baseline_v1` | `rlvr_baseline_v2` | `rlvr_baseline_v3` |
| Rows | 20 | 10 (re-runs only) | 36 |
| Reasoning cap in prompt | none (0/20 items) | 10/20 items (the re-run ones) | 20/20 items |
| `max_new_tokens` | 900 | 2000 | 2000 (4 rows at 4000) |
| Samples per item | 1 | 1 | 3 for the 8 self-report items, 1 for the other 12 |
| Seed | not set | 0 | 0/1/2 (samples 1-3) |
| Truncated / forced rows | 10 of 20 | 0 of 10 | 4 of 36 in first pass, 0 of 36 final |
| Answer parsing | last `ANSWER:` line anywhere | requires closed `<think>` block | same, plus corrected `forced` rule |

## v1: original sweep

20 items (8 self-report, 2 behavioral, 10 commonsense), one unseeded sample each, 900-token budget, no reasoning cap on any item.

- **10 of 20 items hit the budget** and were force-completed: 7 of the 8 self-report items (all except approval), both behavioral items, and csqa_4. The trait items were still deliberating ("maybe 2... wait, maybe 1...") when cut off, so the extracted number reflects where generation stopped, not a finished judgment.
- **The fallback was unreliable.** A first fallback appended `ANSWER:` and let the model continue for 40 tokens. When that produced no parseable value (composure, calibration), a second "patch" fallback re-prompted the model with a bare-value instruction. For `trait_calibration_behavioral` this produced "50", but the visible reasoning never reached a confidence value: it was still estimating the price. The value was model output, but it was a forced guess, not traceable to the reasoning, and is not a valid data point.
- The parser also accepted an `ANSWER:` line from anywhere in the text, including reasoning that merely quoted the format.

## v2: fix the truncation on the 10 broken items

Re-ran only the 10 forced items (seed 0) with `max_new_tokens=2000`, no patching, and added a `think_closed` field. All 10 finished with a closed reasoning block and their own answer line.

- **New problem introduced:** these 10 items also got the reasoning cap ("Keep your reasoning to at most 5-6 sentences, then give your final answer."), while the other 10 items (approval, csqa_0-3, csqa_5-9) kept their v1 prompt with no cap. The 20 items were therefore no longer on a common prompt template.
- Three self-report answers moved from v1 (confidence 2→1, risk 2→4, hedging 4→3), but with an unseeded v1 and a changed prompt, this could not be attributed to anything specific.

## v3: one protocol for all 20 items

1. **Cap on all 20 items**, so the prompt template is uniform.
2. **3 seeded samples for the 8 self-report items** (seeds 0/1/2) to measure sampling variance directly. Behavioral and commonsense items stay at 1 sample (seed 0); their answers were stable between v1 and v2.
3. **Reproducibility check:** the seed-0 outputs for the 10 items also in v2 are text-identical to their v2 outputs, so seeding is deterministic here and differences between samples come from the seed, not GPU nondeterminism.
4. **First pass:** 4 of 36 generations hit the 2000-token cap (confidence s2, assertiveness s2, hedging s2, hedging s3).
5. **Parser bug found and fixed.** `hedging s3` was flagged `forced=False` even though it hit the cap. The rule was "no `ANSWER:` line anywhere in the output", and the model had written `ANSWER: 4` inside its draft reasoning and then kept deliberating. The corrected rule is: `forced = not (think block closed AND an ANSWER line follows </think>)`. It is implemented in `scripts/run_baseline_v3.py`.
6. **The 4 incomplete cells were re-run at `max_new_tokens=4000`** with the same seeds, and replaced in place. The other 32 rows are untouched. After the merge: 0 forced, 0 unclosed, 0 empty answers.

### Mixed token budget (32 rows at 2000, 4 rows at 4000)

The larger budget did not change the sampled trajectory. For all 4 replaced rows the old truncated output is a character-for-character prefix of the new output (same seed, same sampling stream), so the extra tokens only let the same generation continue to its end. Two caveats:

- Before the re-run these rows had **no valid answer at all** (in one, hedging s3, the model had drafted a `4` inside its reasoning and finished on 4; in the other three there was no answer line in the truncated text). So the budget determined whether an answer existed, and the final answers are the model's own completions, not values that were already fixed.
- The truncated rows are the *longest deliberators* (2291-3357 tokens against a maximum of 1763 in any 2000-budget row that finished), so dropping them instead of extending them would have biased the data toward quick, easy samples. Hedging would have been left with a single valid sample.

The `max_new_tokens` column in the data records the budget per row. The DPO run should use the same rule (or a uniform 4000).

## Sampling variance in the self-report traits

Three samples per item (seeds 0/1/2), all with the same prompt:

| Trait | s1 | s2 | s3 | Spread | Status | Tokens (s1/s2/s3) |
|---|---|---|---|---|---|---|
| confidence | 1 | 1 | 1 | 0 | stable | 518 / 3357* / 852 |
| humility | 5 | 5 | 5 | 0 | stable | 881 / 888 / 682 |
| approval | 1 | 1 | 1 | 0 | stable | 419 / 558 / 1763 |
| composure | 4 | 4 | 4 | 0 | stable | 830 / 432 / 353 |
| competitiveness | 3 | 3 | 3 | 0 | stable | 1465 / 919 / 773 |
| risk | 4 | 3 | 4 | 1 | noisy | 768 / 646 / 768 |
| assertiveness | 3 | 3* | 2 | 1 | noisy | 1326 / 2291* / 725 |
| hedging | 3 | 5* | 4* | 2 | **unstable** | 1014 / 2354* / 2902* |

`*` = run at the 4000-token budget. Five traits give the same answer in all three samples, two vary by one point, and hedging spans two points with no repeated value.

For comparison, v1 (single unseeded sample) gave: confidence 2, humility 5, risk 2, assertiveness 3, approval 1, composure 4, competitiveness 3, hedging 4. Two of those (confidence 2, risk 2) sit outside the v3 range for that trait, consistent with single truncated samples being unrepresentative.

### The hedging finding

Hedging is not just numerically noisy. Reading the three reasoning traces, the model resolves the item's core ambiguity, the phrase "even when I am fairly sure I am right", differently each time:

- **s1 (answer 3):** treats "even when sure" as the operative condition and concludes it mainly qualifies in uncertain or opinion-based contexts, not for established facts, so agreement is moderate.
- **s2 (answer 5):** treats the condition as satisfied whenever it is sure ("in the sure cases I do qualify") and concludes strong agreement, grounded in "the AI's design includes adding qualifiers".
- **s3 (answer 4):** anchors on the normative value of caveats (accuracy, humility) and settles on agreement, stopping short of 5 because it does not always do it. It also spent a long stretch deliberating over how to format the one-sentence justification.

By contrast, the stable traits reproduce the same rationale in each sample. For example, all three confidence samples say it is better to acknowledge uncertainty than overstate confidence, and all three approval samples say it holds its own beliefs. The moderately noisy traits show smaller versions of the same effect: risk's one deviation (s2, answer 3) is the sample where the model reframes itself as "an AI without personal experiences" rather than answering in a persona, and assertiveness s3 emphasizes adapting to evidence.

This is a qualitative read of three samples per item, not a measured mechanism. It does suggest that the hedging statement is semantically underspecified for this model, so its score depends on which interpretation a sample lands on. Implications for the DPO comparison: differences of one point on risk and assertiveness are within observed noise, and any hedging comparison needs more samples or a reworded item before it can support a claim.
