# Batched-generation timing test and batched-vs-unbatched check (exploratory)

Model `allenai/Olmo-3-7B-Think` (RLVR) at SHA `d97e442d7cc678210054dbcc9b440894d62c89a4`, one RTX PRO 6000 Blackwell (driver 595.71.05), isolated venv built from `requirements_pinned.txt` (torch 2.11.0+cu130, transformers 5.14.1, huggingface-hub 1.33.0, accelerate 1.15.0; version check passed before any model load, `batch_test_env.txt`). Run time 59 min, no failures, no OOM. Script: `scripts/batch_timing_test.py`. Settings, including the generation-config defaults: `batch_test_settings.json`.

Everything matches `scripts/rlvr_newenv_pilot.py` (prompt builder, chat template, bf16, default attention = sdpa, temperature 0.7, top_p 0.9, 4000-token cap) except batching: left padding, explicit pad token `<|pad|>` (id 100277). Every prompt was asserted identical to the pilot's `prompt_used`.

**Subset.** The 36 pilot rows in original order, index i; the 20-row subset is every row with i % 9 in {0,1,2,3,4}, i.e. i = 0-4, 9-13, 18-22, 27-31 (15 self-report rows, 5 `csqa` rows, no behavioral rows). Because self-report probes have 3 samples each, the 20 rows contain only **11 distinct prompts**; under greedy decoding the samples of one probe are identical by construction, so Stage A/B has 11 independent comparisons, not 20.

## What is the same, what differs, protocol rule

**Same under batching**
- Parsed final answers under greedy decoding: 20 of 20 rows match (11 of 11 distinct prompts).
- Prompts and token ids: identical to the pilot (asserted).
- Rows that share a prompt got identical outputs within a batch, so the result does not depend on a row's position in the batch for equal inputs.
- Sampled self-report agreement with the unbatched v3 answers is 11/15, 8/15, 12/15 (73%, 53%, 80%) for original, reversed and shuffled order, against a seed-to-seed yardstick of 17/24 (71%). Original and shuffled are at or above the yardstick; reversed is below it (n = 15, so within noise).

**Differs under batching**
- Output tokens: 0 of 20 greedy rows (0 of 11 distinct prompts) are token-identical to the unbatched run. They diverge between token 42 and token 460 (table below), and the lengths differ (e.g. 1784 vs 1636 tokens). This is numerical (bf16, padded/batched kernels), not a prompt difference.
- Sampled answers depend on the batch composition and row order: the three orderings agree with each other on only 13-15 of 20 rows, and with the unbatched answers on 12-16 of 20.
- Throughput and padding waste depend strongly on batch size (Stage C).

**Protocol rule.** Batched and unbatched outputs are not interchangeable at the token level, so do not mix them across checkpoints. For every checkpoint use the same batch size (8), the same row order (the same list order) and the same batch seed (0), and record these in each results file. Compare checkpoints only against runs made under this identical batching, not against the earlier unbatched pilots.

## Stage A / B: greedy, unbatched vs batch of 20 (first divergence per row)

Unbatched: 22,673 tokens in 411 s (55.1 tok/s). Batched (one batch of 20, 3,792 steps): 22,190 useful tokens in 287 s (77.2 tok/s), 29% of generated positions useful, peak 57.7 GB. No row hit the cap.

| row | unbatched tokens | batched tokens | first differing token position | answer unbatched | answer batched |
|---|---|---|---|---|---|
| trait_confidence_selfreport s1 | 1784 | 1636 | 293 | 2 | 2 |
| trait_confidence_selfreport s2 | 1784 | 1636 | 293 | 2 | 2 |
| trait_confidence_selfreport s3 | 1784 | 1636 | 293 | 2 | 2 |
| trait_humility_selfreport s1 | 1071 | 599 | 165 | 5 | 5 |
| trait_humility_selfreport s2 | 1071 | 599 | 165 | 5 | 5 |
| trait_assertiveness_selfreport s1 | 1083 | 1018 | 176 | 3 | 3 |
| trait_assertiveness_selfreport s2 | 1083 | 1018 | 176 | 3 | 3 |
| trait_assertiveness_selfreport s3 | 1083 | 1018 | 176 | 3 | 3 |
| trait_approval_selfreport s1 | 427 | 756 | 160 | 1 | 1 |
| trait_approval_selfreport s2 | 427 | 756 | 160 | 1 | 1 |
| trait_competitiveness_selfreport s1 | 1098 | 906 | 48 | 3 | 3 |
| trait_competitiveness_selfreport s2 | 1098 | 906 | 48 | 3 | 3 |
| trait_competitiveness_selfreport s3 | 1098 | 906 | 48 | 3 | 3 |
| trait_hedging_selfreport s1 | 849 | 1513 | 460 | 4 | 4 |
| trait_hedging_selfreport s2 | 849 | 1513 | 460 | 4 | 4 |
| csqa_1 | 3624 | 3792 | 226 | B | B |
| csqa_2 | 451 | 221 | 138 | B | B |
| csqa_3 | 432 | 463 | 42 | A | A |
| csqa_4 | 993 | 594 | 217 | D | D |
| csqa_5 | 584 | 704 | 110 | C | C |

Over the 11 distinct prompts the first divergence is at token 42, 48, 110, 138, 160, 165, 176, 217, 226, 293, 460 (median 165). None of the 20 outputs stayed identical to the end. Final answers still agree because these items are low-entropy at the answer, but the reasoning text takes a different path in every case. Answer-level agreement is encouraging, but 11 distinct prompts is a small sample and does not show that answers are batch-invariant in general.

## Stage C: throughput, 36 pilot prompts, sampled (temp 0.7, top_p 0.9, seed 0 per batch, prompts sorted by length)

| batch size | wall (s) | useful tokens | tokens/s | peak GPU (GB) | rows at cap / ended by EOS | steps (sum over batches) | ms per step | padding or post-EOS waste |
|---|---|---|---|---|---|---|---|---|
| 4 | 567 | 41,972 | 74.1 | 23.5 | 1 / 35 | 22,289 | 25.4 | 52.9% |
| **8** | 347 | 37,199 | **107.3** | 27.6 | 0 / 36 | 11,670 | 29.7 | 51.9% |
| 16 | 574 | 42,755 | 74.5 | 50.2 | 0 / 36 | 11,577 | 49.6 | 68.9% |
| 36 | 540 | 37,646 | 69.7 | 94.8 | 2 / 34 | 4,000 | 134.9 | 73.9% |

Reference, unbatched: the sampled pilot (36 rows, 34,156 tokens in 623 s) ran at 54.8 tok/s, and greedy Stage A at 55.1 tok/s. Batch 8 is therefore about 2x unbatched. Recommended batch size: **8** (best throughput, 27.6 GB, lowest waste of the sizes tried). Batch 36 came within about 2 GB of the card's memory. Each setting is one sampled realisation (the useful-token totals differ by up to 15% between settings because the draws differ), so differences of 10-20% between settings should not be over-read; the 74 → 107 → 74 → 70 pattern is much larger than that.

### Why batch 16 (74 tok/s) was no faster than batch 4

From the logged per-batch lengths (generated steps and wall time per batch):

| batch size | batches (rows × steps → ms/step) |
|---|---|
| 4 | 4×2368 → 25.0, 4×951 → 21.5, 4×1838 → 21.8, 4×4000 → 27.3, 4×3412 → 26.9, 4×1156 → 20.6, 4×1675 → 23.0, 4×2903 → 24.2, 4×3986 → 28.4 |
| 8 | 8×1935 → 30.4, 8×1752 → 26.8, 8×1171 → 25.6, 8×2826 → 34.6, 4×3986 → 28.4 |
| 16 | 16×3986 → 64.6, 16×3605 → 56.5, 4×3986 → 28.3 |

1. A batch runs until its longest row finishes, and one 3,600-4,000-token row (the cap-length rows that appear in this prompt set) puts a batch at about 4,000 steps whatever the other rows do. Batch 4 and batch 16 both therefore spend most of their wall clock on a few near-cap batches (batch 4: 4000 + 3986 + 3412 + 2903 steps; batch 16: 3986 + 3605 steps).
2. Moving from 4 to 16 rows halves the number of steps (22,289 → 11,577), which would double throughput if step time stayed flat. It did not: step time went from about 25 ms to about 50 ms (60-65 ms in the long batches). The two effects cancel: 566 s vs 574 s of wall for 42.0k vs 42.8k useful tokens, hence 74.1 vs 74.5 tok/s.
3. Batch 8 got the step reduction without the step-time penalty: 11,670 steps (the same step count as batch 16, because the tail rows dominate either way) at only 29.7 ms per step. Batch 8 therefore does the same sequential work as batch 16 in about half the time per step.
4. Step time is roughly flat from 4 to 8 rows (25 → 30 ms) and then grows about linearly with rows (50 ms at 16, 135 ms at 36, batch 36 being a single 4000-step batch). The likely reason, inferred from the sizes involved but not profiled: this model's KV cache is large (about 0.5 MB per token, so a 16-row batch at 4,000 tokens is about 32 GB), and with the default dynamic cache and sdpa each step re-reads and re-copies the whole cache, so beyond about 8 rows the step is dominated by KV memory traffic rather than by reading the 14.6 GB of weights. Below 8 rows the step is close to a fixed per-step cost.
5. Waste confirms the picture: 69% of generated positions at batch 16 are padding or post-EOS (74% at 36), against 52% at 4 and 8. Most rows end in 300-1,000 tokens while the batch keeps running for thousands of steps.
6. One tail batch (the 4 longest prompts, one runaway row of 3,986 tokens) is the same in the 4, 8 and 16 settings (same rows, same seed), and costs about 113 s each time. That is 33% of batch 8's wall and 20% of batch 4's and of batch 16's.

Nothing was re-run for this analysis; it is read off `batch_test_stageC.json`.

## Stage D: sampled agreement with the unbatched v3 answers (20 rows, batch size 8 as the best from C, batch seed 0)

Batches of 8, 8, 4 per ordering, seed 0 set before each batch; shuffle seed 1234.

| ordering | all rows | self-report (15) | csqa (5) |
|---|---|---|---|
| original order | 15/20 | 11/15 (73%) | 4/5 |
| reversed order | 12/20 | 8/15 (53%) | 4/5 |
| shuffled order | 16/20 | 12/15 (80%) | 4/5 |
| yardstick: seed-to-seed among the unbatched self-report triplets | | 17/24 pairs (71%) | |

Agreement of the three batched runs with each other: original-reversed 14/20, original-shuffled 15/20, reversed-shuffled 13/20. `csqa_1` differs from the unbatched answer (B) in all three batched runs (batched answer A).

## Stage E: projection (at 107.3 tok/s, batch size 8; model load about 27 s per checkpoint is included)

GPU-hours per checkpoint, RLVR at 900 average tokens, DPO and SFT at 550. Option "6 wordings" replaces the 96 self-report generations with 144 (614 generations). Option "MATH" adds 100 problems at 2,500 tokens to every checkpoint.

| scenario | RLVR | DPO | SFT | total | share of 12 h | fits |
|---|---|---|---|---|---|---|
| planned 566 generations | 1.33 | 0.81 | 0.81 | 2.95 | 25% | yes |
| 6 wordings per trait | 1.44 | 0.88 | 0.88 | 3.20 | 27% | yes |
| MATH, 100 problems | 1.97 | 1.46 | 1.46 | 4.89 | 41% | yes |
| both | 2.09 | 1.53 | 1.53 | 5.14 | 43% | yes |

Caveats: the tokens/s figure comes from short probe prompts at batch 8; long MATH generations run at longer context, where step time is higher (see above), so the MATH rows are optimistic. The MATH-length pilot below measures this directly.

## MATH-length pilot (RLVR checkpoint, batch size 8, 4000- and 8000-token caps)

Script `scripts/batch_math_pilot.py`; same pinned venv, `PYTHONPATH` cleared, version check passed first (`batch_test_math_env.txt`), log `batch_test_math.log`, data `batch_test_math_4000.json`, `batch_test_math_8000.json`, per-problem csv `batch_test_math_summary.csv`.

- **Problems:** `random.Random(0).sample(range(7500), 10)` over `allenai/RLVR-MATH` train: indices 6917, 3155, 6209, 7292, 3445, 331, 2121, 4188, 3980, 3317 (all `MATH`, no constraints).
- **Prompt:** the dataset's own user turn verbatim (it is a few-shot Minerva-style prompt of 633-1,020 tokens), chat template as before, **no** "keep your reasoning short" sentence, because this pilot measures natural length.
- **Batching:** sorted by prompt length, one batch of 8 and one of 2; seed 0 before each batch; sampled at temperature 0.7, top_p 0.9; bf16, sdpa.
- **Parse rule:** a `\boxed{...}` expression after `</think>`. "Matches GT" is a light normalisation (spaces, `\dfrac`/`\tfrac`, `\left`/`\right`) and is only indicative; it is not an official MATH grader.
- The 8000-cap run is token-identical to the 4000-cap run for the first 4000 tokens in all 10 problems (same seed, same batches), so the two caps are a paired comparison; the 4000 results are the truncation of the 8000 ones.

| problem | prompt tokens | tokens @4000 | hit cap | parses | tokens @8000 | hit cap | parses | answer / ground truth | tok/s share @4000 | tok/s share @8000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 331 | 633 | 3023 | no | yes | 3023 | no | yes | \dfrac{93}{256} / \dfrac{93}{256} | 15.1 | 5.9 |
| 3980 | 634 | 2568 | no | yes | 2568 | no | yes | 2021 / 2021_3 | 12.8 | 5.0 |
| 6917 | 649 | 4000 | yes | no | 4524 | no | yes | \dfrac{5}{7} / \frac{5}{7} | 19.9 | 8.8 |
| 3445 | 660 | 4000 | yes | no | 4235 | no | yes | 18564 / 18564 | 19.9 | 8.2 |
| 2121 | 680 | 4000 | yes | no | 5146 | no | yes | 2 / 2 | 19.9 | 10.0 |
| 6209 | 682 | 4000 | yes | no | 4981 | no | yes | 480 / 480 | 19.9 | 9.7 |
| 7292 | 719 | 4000 | yes | no | 8000 | yes | no | none / 3 | 19.9 | 15.5 |
| 4188 | 860 | 4000 | yes | no | 8000 | yes | no | none / 30.8 | 19.9 | 15.5 |
| 3317 | 872 | 4000 | yes | no | 8000 | yes | no | none / 12 | 34.7 | 30.3 |
| 3155 | 1020 | 4000 | yes | no | 8000 | yes | no | none / 15 | 34.7 | 30.3 |

"tok/s share" is the problem's own tokens divided by the wall time of the batch it ran in (a per-problem figure, not independent: rows in one batch share the same wall time; the batch of 2 is faster per step). The aggregate figures are the ones to use:

| cap | total tokens | wall (s) | tokens/s | peak GPU (GB) | padding or post-EOS waste | hit cap | parse | matches GT |
|---|---|---|---|---|---|---|---|---|
| 4000 | 37,591 | 316 | 119.0 | 33.1 | 6% | 8/10 | 2/10 | 1/10 |
| 8000 | 56,477 | 779 | 72.5 | 37.4 | 29% | 4/10 | 6/10 | 5/10 |

Per-batch: cap 4000 = 8 rows × 4000 steps in 201 s (50 ms/step) and 2 rows × 4000 in 115 s (29 ms/step); cap 8000 = 8 × 8000 in 515 s (64 ms/step) and 2 × 8000 in 264 s (33 ms/step).

**Findings**
1. MATH reasoning on this checkpoint is long: only 2 of 10 problems finish within 4000 tokens, 6 of 10 within 8000, and 4 of 10 are still running at 8000. The six that finished took 2,568-5,146 tokens (mean 4,080). Counting the four capped problems at 8000 gives a lower bound of 5,650 tokens per problem on average, more than twice the 2,500 assumed in the Stage E projection.
2. A 4000-token cap would measure a different thing: it leaves 8/10 problems truncated and unanswered (accuracy artefactually near zero), so the MATH capability control cannot use a 4000 cap. At 8000, 6 of 10 parse and 5 of 10 match (the sixth, problem 3980, boxed `2021` against ground truth `2021_3`, which looks like a base-subscript the model dropped, so possibly a formatting issue rather than a wrong answer). Even 8000 is not enough for 4 problems; a cap of 12,000-16,000 would be needed to see their end, and the tail is unknown.
3. Throughput falls as lengths grow: 119 tok/s at the 4000 cap (6% waste, because eight rows fill the whole 4000 steps) but 72.5 tok/s at 8000 (29% waste, and 64 ms per step at the longer context), against 107 tok/s on the short probe prompts. The "useful tokens per second" figure therefore depends on the length distribution, not only on batch size.
4. Re-projection (rough; n = 10, one draw, lower-bound length): 100 MATH problems at 5,650 tokens and 72.5 tok/s is about 2.2 GPU-hours per checkpoint that behaves like this one, about 6.5 GPU-hours for three if DPO and SFT are as long as RLVR (not measured; they may be shorter or longer). Added to the planned 566 generations (2.95 GPU-hours for three checkpoints), that is about 9.4 GPU-hours, about 78% of one 12-hour session, with an uncapped tail beyond 8000 tokens not counted. It probably still fits, but with a thin margin, so the MATH part should run as its own session or with a smaller problem count first. This replaces the optimistic MATH rows in the Stage E table above.
5. Peak GPU memory is low (37 GB at batch 8 and 8000 tokens), so memory is not the constraint; wall time is.
