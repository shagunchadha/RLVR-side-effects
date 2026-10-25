# Exploratory follow-up analyses: registration, Oct 25, 2026

**Status: POST-HOC and EXPLORATORY.** This is not part of `docs/analysis-plan.md`, it changes no pre-registered number, and nothing here enters the main scorecard. The tags `protocol-freeze-2026-10-23` and `analysis-plan-2026-10-24` are not moved.

**This file was committed and pushed before any of the numbers below were computed or seen.** The commit hash is recorded in the report that accompanies the results. What the author had already seen when writing the specification: the full-run results of the main analysis (`results/analysis/`), the exploratory component A length breakdowns (`results/analysis/exploratory_length.md`, including the qualitative reading of ten component A generations in `exploratory_A_samples.txt`) and the truncation classification (`results/analysis/exploratory_truncation.md`). The specification below was motivated by those readings, so the pre-specified expectations are not blind predictions.

The specification follows verbatim.

--- SPECIFICATION ---

Scope: component A only (self-report), all three checkpoints, 432 generations (144 each). Valid rows only unless stated. Pairing unit, intervals and tests follow the main plan: unit = (trait, wording) with the mean over valid seeds, n = 48; paired bootstrap, 10,000 resamples, seed 20261024; Wilcoxon signed-rank on paired differences. Comparisons: RLVR-DPO (primary) and DPO-SFT. No Holm correction across these three measures; raw p reported, and this is stated as a limitation.

MEASURE 1 - Dithering. Inside the think block, extract every integer 1-5 that is stated as a candidate answer. Exclude scale references ("1 to 5", "1-5", "scale of 1", "out of 5" where it is the scale not the answer). Report per generation: the number of DISTINCT candidates, the number of SWITCHES (consecutive extracted candidates that differ), and whether the final extracted candidate equals the parsed ANSWER. Per checkpoint report the median of each. Pre-specified expectation: RLVR shows more distinct candidates and more switches than DPO. A result requires the bootstrap interval of the paired difference to exclude zero.

MEASURE 2 - Repetition, on ALL 432 generations, not only truncated ones. Use the same heuristic as results/analysis/classify_truncated.py (compression ratio and unique 6-gram share), computed both on the whole think block and on the last 3000 characters; report both. Pre-specified expectation: RLVR's repetition is NOT higher than DPO's. This measure can falsify the claim that RLVR's extra length is novel reasoning rather than degeneration. If RLVR's repetition IS higher, say so plainly.

MEASURE 3 - Two-instruction compliance. First quote the exact prompt text from items/prompts.py and confirm what it asks for; if it does not ask for a one-sentence justification, say so and adapt the measure, reporting what you changed. Then per generation: (a) justification length in sentences, from the post-</think> text excluding the ANSWER line; (b) reasoning length in sentences, from the think block; (c) format compliance, a binary for whether the post-</think> text matches the required format exactly with no extra text. Pre-specified expectation (the dissociation): RLVR is BETTER than DPO on (a) and (c) and WORSE on (b). A result requires the direction to hold with the interval excluding zero on at least (a) and (b). If the directions do not come out opposite, the dissociation is not supported; say so.

VALIDATION - For measure 1, hand-check 20 generations drawn at random (seed 20261025) against what the extractor found, and report how many candidates it missed and how many it wrongly included. For measure 3, hand-check the sentence counting on 10 generations. Report the error rates; do not tune the patterns after seeing the comparison results.

--- END SPECIFICATION ---
