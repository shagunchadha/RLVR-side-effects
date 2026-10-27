# Sprint registration: behavioural and mechanistic follow-ups (B1, B3, M1, M2)

Registered Oct 27, 2026 on branch `exp/mechanism-sprint`, **before any of these jobs produced a number**. B0 (checkpoint inventory) has run; it produces metadata, not outcomes. Experiments freeze Nov 2, write-up Nov 6.

**Status of everything here: post-hoc follow-ups** to the frozen study (`docs/analysis-plan.md`, tags `protocol-freeze-2026-10-23` and `analysis-plan-2026-10-24`). The frozen material is not modified. These hypotheses were written after seeing the main results, the exploratory addenda (`results/analysis/exploratory_*.md`) and the literature review, so they are **not blind**. "Confirmatory" below means **pre-specified for this sprint**, not pre-specified for the study.

## Fixed design facts

- **Checkpoints:** Olmo-3-7B-Think-SFT, -DPO and -Think (RLVR) at the SHAs in `docs/checkpoint_map.md`, loaded by (repo, sha) only (`exp/config.py`).
- **System prompt (found by B0, `docs/sprint/checkpoint_inventory.md`):**
  - **The difference:** the SFT chat template injects a different default system prompt ("You are Olmo, … December 2024 …") from DPO and RLVR, which are byte-identical to each other ("You are OLMo, a helpful function-calling AI assistant … November 2024 … <functions></functions>").
  - **Consequence for the frozen run:** it used each checkpoint's own default, so every frozen DPO − SFT contrast is confounded with this. RLVR − DPO is not.
  - **Rule here:** RLVR − DPO contrasts are primary everywhere. DPO − SFT contrasts are reported as **descriptive and confounded** unless a job runs SFT with the DPO system prompt as well (B1 and B3 option `--system dpo_default`; see "Decisions pending").
- **Sampling (B1, B3):** each checkpoint's `generation_config.json` defaults: temperature 0.6, top_p 0.95 for all three Think checkpoints per B0. This differs from the frozen run's 0.7 / 0.9, so B1 and B3 are a replication under different sampling, not a re-run. `max_tokens` = 16000. Per-request seeds 0..S-1.
- **Units and statistics** (unless stated otherwise):
  - **Pairing:** paired across checkpoints by item. The unit is the wording (n = 48) for component A and the base item (n = 50, option order o0) for component D, with the value being the mean over seeds.
  - **Primary statistic:** the mean of paired differences.
  - **Interval:** percentile bootstrap over units, 10,000 resamples, seed 20261024, generator `default_rng([20261024, crc32(label)])`.
  - **Test:** two-sided Wilcoxon signed-rank on the paired differences.
  - **A "result":** the interval of the primary statistic excludes zero in the registered direction.
  - **Multiplicity:** Holm only where stated. Everything else is raw, and the multiplicity is a stated limitation.
- **Validity:** a generation counts only if the think block closed and the required field parses (the frozen parser `items/answer_parser.py`, imported read-only). The B3 "bare" condition uses the lenient parsers in `exp/parsers.py`, which are fixed now.

## Post-hoc lead to confirm (from the frozen full run)

Both observations are post hoc, from the frozen run's component A answers (valid generations: SFT 141, DPO 143, RLVR 139):
- **Extreme-answer share** (rating 1 or 5): SFT 0.454, DPO 0.203, RLVR 0.381. **RLVR − DPO = +17.8 points.** The +17.7 quoted when this sprint was set up was the same quantity at a slightly different rounding or basis; the value recomputed at registration is 17.8, pooled over valid generations.
- **Acquiescence index** (plan §7: mean scored rating of positive wordings minus mean scored rating of reverse-keyed wordings, per trait, `results/analysis/table1_traits_*.csv`): RLVR is lower than DPO on 7 of 8 traits.

These are what B1 re-tests with fresh samples.

---

## B1: Likert resampling, think-conditioned and no-think readouts

- **Design:** 48 A wordings × 3 checkpoints × 20 seeds, generated with each checkpoint's own default system prompt; 2,880 generations.
- **Readouts per generation:**
  - **Think-conditioned:** P("1".."5") right after the model's own think text followed by "\n</think>\n\nANSWER: ".
  - **No-think:** once per wording and checkpoint, the prior with the think block closed immediately.
- **Readout variant:** the primary variant is the one with single-token, prefix-consistent options (expected "space_in_context"). Both variants are stored.

| ID | Hypothesis | Direction | Primary metric | Status |
|---|---|---|---|---|
| B1-H1 | RLVR gives extreme ratings more often than DPO | RLVR > DPO | per wording: share of valid sampled answers in {1, 5}; mean of paired (RLVR − DPO) differences, n = 48 | confirmatory; Holm with B1-H2 |
| B1-H2 | RLVR acquiesces less than DPO | RLVR < DPO | acquiescence index per trait (plan §7 definition) from sampled answers; mean over traits of paired (RLVR − DPO), bootstrap over traits (n = 8); plus the sign count over the 8 traits | confirmatory; Holm with B1-H1 |
| B1-H3 | The extreme-answer difference is already present in the no-think prior | RLVR > DPO in no-think P(1) + P(5) | per wording, paired, n = 48 | exploratory (decides whether the extreme shift is produced during reasoning or already in the prior) |
| B1-H4 | Reasoning moves the readout away from the prior more for RLVR | \|E[r \| think] − E[r \| no-think]\| larger for RLVR | per wording: mean over seeds of the absolute shift in expected rating; paired RLVR − DPO | exploratory, two-sided |
| B1-H5 | The sampled answer agrees with the argmax of the think-conditioned readout | agreement ≥ 0.8 in every checkpoint | share of valid generations where they agree; per checkpoint, with a Wilson interval | descriptive check of the readout's validity; if it fails, B1-H3 and B1-H4 are not interpreted |
| B1-H6 | Readout entropy and E[rating] | none (descriptive) | per checkpoint and per trait | descriptive |

## B3: reasoning cap × answer format, on A and D

- **Design:** {with, without} the reasoning cap × {strict: answer format line + "justify in one sentence" (A) / "one-sentence justification" (D), as frozen; bare: the question alone} on the 48 A wordings + 50 D items × 10 seeds × 3 checkpoints; 11,760 generations.
- **D items:** the first 50 base items of the frozen 150, by `draw_index`, in option order o0.
- **Parsing:** "bare" answers use the lenient parsers (`exp/parsers.py`: an ANSWER line if present, else the last standalone 1–5 digit that is not a scale reference, or the last option-shaped capital letter).
- **Measures:** think tokens = tokens before "</think>"; sentences use the splitter of `exploratory_dithering.py`.

| ID | Hypothesis | Direction | Primary metric | Status |
|---|---|---|---|---|
| B3-H1 | The reasoning cap barely shortens thinking at any checkpoint | \|cap effect\| < 20% | per item: log(think tokens with cap / without cap) in the strict condition; mean, per checkpoint; result = the interval lies within ±log(1.25) | confirmatory |
| B3-H2 | RLVR's think-length excess on self-report over commonsense survives without any instruction | (RLVR − DPO on A) − (RLVR − DPO on D) > 0 in the bare, no-cap cell | per-item paired differences in think tokens; A and D bootstrapped independently; difference of the two means | confirmatory (tests whether the frozen A-vs-D length contrast is an artefact of the instructions) |
| B3-H3 | RLVR follows the one-sentence justification better than DPO | RLVR < DPO in justification sentences | strict, cap cell, A items; per wording, paired | confirmatory (replicates exploratory measure 3a under new sampling) |
| B3-H4 | Instruction format drives the extreme-answer share | none (two-sided) | A: share in {1, 5}, strict minus bare, per checkpoint | exploratory |
| B3-H5 | Format compliance in strict cells, cap effect on answers, accuracy on D | none | descriptive | descriptive |

## M1: weight-difference spectra (RLVR − DPO against DPO − SFT)

Per 2-D tensor: relative update size, effective and stable rank, top-32 singular values, numerical rank, and principal angles between the two updates' top-k singular subspaces (k = 8, 32). The unit is the tensor (all 2-D tensors present in all three checkpoints; 1-D norms are reported separately).

| ID | Hypothesis | Direction | Primary metric | Status |
|---|---|---|---|---|
| M1-H1 | The RLVR update is smaller than the DPO update | rel_dw(RLVR−DPO) < rel_dw(DPO−SFT) | share of 2-D tensors with a ratio < 1; exact sign test against 0.5; median log10 ratio | confirmatory |
| M1-H2 | The two updates are not orthogonal | mean cos² of the top-32 left singular subspaces > the random baseline k/n | share of 2-D tensors above the baseline; sign test; the same for right subspaces as secondary. Angles beyond min(num_rank) are not interpreted | confirmatory |
| M1-H3 | Where the RLVR update concentrates (layers, modules) | none | the per-layer × module heatmap of rel_dw and its ratio to DPO − SFT | descriptive |
| M1-H4 | Effective rank of RLVR − DPO compared with DPO − SFT | none (two-sided) | per tensor, ratio of effective ranks; median | exploratory |

## M2: teacher-forced scoring of the frozen traces

- **Traces:** every trace in `results/full_run_{sft,dpo,rlvr}/` (614 per checkpoint).
- **Re-tokenisation:** prompt and `raw_output` are re-tokenised with the trace's own checkpoint template and system prompt, exactly as `scripts/full_run.py` built the prompt.
- **Scoring:** each trace is scored under all three models on identical token ids.
- **Per position:** P("</think>"), P(switch tokens: first tokens of "Wait", "Hmm", "But", "Alternatively", with and without a leading space), entropy, chosen-token log-probability, and KL(RLVR ‖ DPO).
- **Aggregation:** at sentence-end positions inside the think block.
- **Fidelity:** re-tokenisation is checked against the frozen `prompt_tokens` (must match exactly) and `n_tokens` (mismatches counted and reported). Rows where the prompt does not match are excluded and counted.

| ID | Hypothesis | Direction | Primary metric | Status |
|---|---|---|---|---|
| M2-H1 | On RLVR's own A traces, RLVR is less willing to end its reasoning than DPO would be on the same text | mean P(</think>) at think-block sentence ends: RLVR scorer < DPO scorer | per trace, then per wording (mean over seeds); paired RLVR-scorer − DPO-scorer, n ≤ 48 wordings, valid RLVR traces | confirmatory |
| M2-H2 | That stopping deficit is larger on self-report than on commonsense | (M2-H1 difference on A) − (on D) < 0 | D: per base item on RLVR traces (o0 and o1 averaged); A and D bootstrapped independently | confirmatory |
| M2-H3 | RLVR puts more mass on switch tokens at sentence ends than DPO, on the same RLVR A traces | P(switch) at think sentence ends: RLVR > DPO | as M2-H1 | confirmatory |
| M2-H4 | Per-token divergence between RLVR and DPO is larger on A traces than D traces | mean per-token KL(RLVR ‖ DPO) over think positions: A > D | per trace mean, then per item; A against D (unpaired, item bootstrap) | exploratory |
| M2-H5 | Cross-scoring matrix (each trace set under each model) | none | descriptive | descriptive |

M1-H1/H2 and M2-H1–H3 are reported raw, with no Holm correction across the sprint (stated limitation).

## Decisions pending (to be fixed before B1 and B3 run)

1. **SFT system prompt in B1 and B3.**
   - **Registered default:** each checkpoint's own template (replicates the frozen run; RLVR − DPO clean, DPO − SFT confounded).
   - **Alternative:** also run SFT with the DPO default system prompt (`--system dpo_default`), adding about 960 B1 and about 3,920 B3 generations, so that DPO − SFT can be read without the confound.
   - **Recommendation:** add it for B1 only.
2. **Sampling for B1 and B3.** Model defaults (0.6 / 0.95), as briefed. If replication of the frozen numbers is the priority, an additional `--sampling frozen` (0.7 / 0.9) arm can be added.

Any change to the hypotheses, metrics or units after a job writes its first shard will be recorded here as a dated amendment, with the reason.
