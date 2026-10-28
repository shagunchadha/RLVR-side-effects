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

---

## Amendment 1 (Oct 28, 2026): SFT system-prompt arms, frozen-sampling bridge, matched-prompt sensitivity rerun

Committed and pushed **before B1, B3 or the rerun below wrote any output**. M1 (weight differences; no prompts, no sampling) was running at the time, so nothing here touches it. Reason: B0 found that the SFT template injects a different default system prompt from DPO and RLVR (`docs/sprint/checkpoint_inventory.md`; disclosure in `docs/sprint/disclosure-sft-system-prompt.md`). This amendment settles both "Decisions pending" above.

**Bug fixed before any B1/B3 shard (same commit series):** `--system dpo_default` sent the full DPO default text as an explicit system message. For an explicit system message both Think templates append " You do not currently have access to any functions. <functions></functions>" themselves, so the rendered block held that sentence twice and did not equal the DPO default. `exp/prompts_sprint.py` now sends the text without that suffix (`DPO_SYSTEM_MESSAGE`). A test (`tests/sprint/test_jobs.py::test_dpo_default_renders_the_dpo_default_block`, using the pinned SFT and DPO templates as fixtures) asserts that the rendered prompt is byte-identical to the DPO template's own default under both templates.

### C1. B1 system-prompt arms (decision pending 1)

- **B1 arms:** SFT, DPO and RLVR with their own templates (`--system own`, as registered), **plus SFT with the DPO/RLVR system prompt** (`--ckpt sft --system dpo_default`, 48 wordings × 20 seeds = 960 generations). The SFT own-template arm is kept.
- **Contrasts:**
  - **Matched DPO − SFT:** DPO (own) − SFT (`dpo_default`). The two prompts are then byte-identical.
  - **Frozen-style DPO − SFT:** DPO (own) − SFT (own), still confounded, kept for continuity with the frozen run.
  - **System-prompt effect within SFT:** SFT (`dpo_default`) − SFT (own).
  - All three are paired by wording (n = 48), with the B1 metrics, statistics and readouts.
- **Status:** exploratory. B1-H1 and B1-H2 remain RLVR − DPO, with Holm over the two as registered. No DPO − SFT hypothesis is added.
- **B3 is unchanged:** own templates only, so B3's DPO − SFT contrasts remain **descriptive and confounded**.

### C2. Sampling (decision pending 2)

- **B1 bridge arm at the frozen sampling:**
  - Parameters: temperature 0.7, top_p 0.9, no top-k filter (`--temperature 0.7 --top-p 0.9`). The frozen run set no `top_k`: it is unset in the checkpoints' `generation_config.json` and `None` in the transformers 5.14.1 defaults.
  - Size: seeds 0..4 (`--seeds 5`), `max_tokens` 16000, own templates, all three checkpoints; 48 × 3 × 5 = 720 generations.
  - Role: it separates "different sampling" from "different sample" when B1 at model defaults is compared with the frozen run.
  - It is a vLLM replication at the frozen parameters, not a re-run. The engine, RNG stream, batching and token cap differ from the frozen HF `generate` (cap 4000 there).
  - The B1 metrics are computed in this arm as well. They are reported next to the main B1 arm and the frozen values, as exploratory.
- **B3:** model defaults only (temperature 0.6, top_p 0.95 from `generation_config.json`). No frozen-sampling arm.

### C3. Post-freeze sensitivity rerun: SFT with the DPO/RLVR system prompt through the frozen pipeline

- **What:** the frozen full run for SFT, with **only the system prompt changed** to the DPO/RLVR default.
- **Command:** `scripts/full_run_matched_prompt.py --checkpoint sft --system-prompt-from dpo`.
- **Output:** `results/sprint/sft_matched_prompt/`.
- **Implementation:**
  - `scripts/full_run.py` is listed in `items/MANIFEST.sha256` and is **not modified**. The wrapper imports it and calls its `main()` unchanged. The manifest guard, version gate (`requirements_pinned.txt`, driver 595.71.05, RTX PRO 6000 Blackwell), items, `items/run_order.json`, batch composition and order, seeds (A: 0, 1, 2; others: 0), sampling (0.7 / 0.9) and caps (4000; E 16000) are therefore the frozen ones.
  - The single change is that the SFT tokenizer's chat template is replaced by the DPO checkpoint's template. The two differ only in the default system block; that was verified before registration on all 518 frozen items.
- **Checks before the model loads (any failure aborts):**
  - the vocabularies are identical;
  - for every frozen item, the swapped-template input_ids equal DPO's own render exactly;
  - SFT's own render differs from DPO's only before the first user turn.
  - The check and per-item prompt token counts go to `full_run_sft_system_prompt.json`. Every prompt becomes 19 tokens longer (as DPO's), so the left-padding in each batch changes accordingly. That is part of the manipulation, not a deviation.
- **Analysis (fixed now):** `python -m exp.compare_sft_matched` recomputes every frozen result that compares DPO with SFT (list in the disclosure document). It uses the frozen analysis code unchanged (`scripts/analyze.py`'s `analyse()`, `results/analysis/exploratory_dithering.py`'s `report()`), with the matched SFT file in place of the frozen one, and prints frozen against matched side by side. First it checks that recomputing from the frozen files reproduces the committed numbers.
- **Robustness rule, fixed before the rerun:** a frozen DPO − SFT result is called **robust to the system prompt** if, on the matched run, its primary statistic has the same sign and the same verdict (signal / no signal for table 3 rows and H3; interval excludes zero or not for the exploratory rows). Otherwise it is **prompt-sensitive**. Claims C3 and C8 are re-evaluated with their own wording on the matched values.
- **Status:**
  - A post-freeze sensitivity analysis. **The frozen numbers stay the reported frozen results.** The matched values are reported beside them, never in their place.
  - RLVR − DPO results do not involve SFT and are not re-run.
  - One seed per item (three for A) in a single rerun, as frozen. With the prompt also changing the random stream, a disagreement can be noise, and that is stated with each one.

---

## Amendment 2 (Oct 28, 2026): M1 outcome, M1b (exploratory), M2 operationalisation, B6 dataset

Committed and pushed before M1b or M2 wrote any output (the only M2 attempts so far failed at load time, before scoring a trace).

### M1 outcome (recorded here so it cannot be softened later)

**M1-H1 failed.** Registered as "the RLVR update is smaller than the DPO update". It came out the other way: 0 of 226 2-D tensors have rel_dw(RLVR−DPO) < rel_dw(DPO−SFT), the median ratio is 2.03 and the median log10 ratio is +0.307 (`results/sprint/M1/`). M1-H2 and the M1-H3/H4 descriptives are reported as computed. Caveat, stated before M1b: both updates are about 0.1% of the weight norm, below the bf16 spacing (2^-8 relative). Much of each dW may therefore be rounding-level, which M1b checks. M1b can qualify the reading of M1; it does not overturn the M1-H1 verdict.

### M1b: exploratory follow-up to M1 (`exp/jobs/M1b_bf16.py`), registered before it runs

Per tensor, for RLVR−DPO and DPO−SFT:
1. **bf16 resolution:** the share of elements that changed (exact bf16 inequality), and the distribution of the changes in bf16 ulps.
2. **Shuffle null:**
   - **Null:** the entries of dW are permuted at random 5 times.
   - **Signal rank:** the number of singular values of dW above the largest top singular value of the shuffles.
   - **Signal energy:** the share of ||dW||² in those directions; signal_rel = the norm of that part over ||W||.
   - **Limitation:** a very sparse update cannot beat this null even when it is low-rank, so it is read together with (1).
3. **cos(dW, W) per tensor,** and rel_dw orthogonal to W. q_proj and k_proj are flagged: with QK-norm, an update parallel to W is largely absorbed by the norm.
4. **Re-ranking:** does the k_proj concentration among the largest RLVR−DPO updates survive ranking by signal_rel and by rel_dw_orth? The M1-H1 statistic is recomputed on both, as exploratory only.

There are no directional hypotheses; everything is descriptive. Exact float32 SVDs on the GPU.

### M2: two changes before any data (hypotheses M2-H1..H5 unchanged)

1. **P(`</think>`) is not a single-token probability.** The pinned Olmo-3 tokenizer has no `</think>` token: it is `'</'`, `'think'`, `'>…'`. In the frozen traces the close is always `'</'`, `'think'`, `'>\n\n'`, after a sentence-ending `'.\n'` token.
   - **Primary (M2-H1, M2-H2):** P_close at each think-block sentence end is Σ over v ∈ {`'</'`, `' </'`} of P(v | prefix) · P(`'think'` | prefix, v). It is computed at the first content token after the sentence end by a cached one-token step (`exp/harness.py: close_probs`, tested against brute-force forwards). The second factor is skipped when P(v) < 10⁻⁶, a bound on the error.
   - **Sensitivity:** the first-token probability P(`'</'`) + P(`' </'`).
2. **Storage:** per trace, the registered aggregates are kept: think-block means of each feature, its values at every think-block sentence end, and the trace's mean KL. Full per-position lists are dropped (≈2 KB instead of ≈100 KB per trace), so the per-trace progress file can live off the notebook. The computation is unchanged.

### Later jobs and B6

B2, B4, B5, B6, B8 and M3–M6 are not registered in this document. Each gets a dated amendment here before it produces any output. **B6 will use `TIGER-Lab/MMLU-Pro` instead of GPQA**: GPQA is gated and no HF token is available.

### Infrastructure (no effect on any metric)

- The sprint environment adds `accelerate==1.15.0`, the frozen environment's version. `harness.load_hf` uses `device_map`, and the M2 small test failed without it.
- Jobs now run a pre-flight (RAM ≥ 16 GB, disk ≥ 2× model size) and log RES lines. They resume from per-unit progress files that are mirrored off the notebook (`exp/sync_remote.py`).

---

## Amendment 3 (Oct 29, 2026): B2, B4, B5, B6, B8, M3–M6 registered; Gate 1 exploratory analyses; infrastructure

Committed and pushed **before any of B2, B4, B5, B6, B8 or M3–M6 has code or output**. The designs are the PI's (sent Oct 29). Where a detail had to be fixed to make a design runnable, it is marked *(operational)*. B1, B3, M1, M1b and M2 are unchanged.

**Common to everything below.**
- **Checkpoints:**
  - SFT, DPO and RLVR as in "Fixed design facts".
  - RL-Zero and the base model at the SHAs in `docs/sprint/checkpoint_inventory.md`, loaded by (repo, sha).
- **Pairing and statistics:**
  - Pairing unit: the A wording (n = 48), or the item for other item sets.
  - Statistics as in "Fixed design facts": mean of paired differences; percentile bootstrap over units, 10,000 resamples, seed 20261024, generator `default_rng([20261024, crc32(label)])`; two-sided Wilcoxon.
  - A "result" = the interval excludes zero in the registered direction.
  - No multiplicity correction across jobs (stated limitation).
- **Layers for M3–M6:** 11, 15, 19, 23, 29, with layer 4 as an early control.
- **Sampling and readout (unless stated):**
  - each checkpoint's `generation_config.json` defaults (0.6 / 0.95 for the Think checkpoints);
  - `max_tokens` 16000, per-request seeds 0..S−1, own templates;
  - B1's readout (`harness.likert_readout`) wherever a Likert readout is named.
- **New item sets:** frozen in `items_sprint/`, with a sha256 manifest committed **before** the job's first run.
- **Status:** all confirmatory for the sprint, in the sense of "Status of everything here" (post hoc to the frozen study, not blind).

### B2: indeterminacy vs self-reference

- **H:** the RLVR − DPO increase in think length is driven by indeterminacy, not self-reference. The indeterminate cells (self and other) increase more than the determinate ones, and the self × indeterminacy interaction is small.
- **Design:** 2 × 2 grid with about 40 items per cell, frozen with sha256 in `items_sprint/` before any run.

| cell | items |
|---|---|
| self, indeterminate | A wordings |
| self, determinate | yes/no facts about the model itself |
| other, indeterminate | A wordings in the third person, plus aesthetic/opinion Likert items |
| other, determinate | D items |

- **Runs:** 10 seeds × 3 checkpoints.
- **Primary:**
  - Per cell: the median over items of the paired think-length difference RLVR − DPO (per item, the mean over seeds).
  - The indeterminacy main effect: (mean of the two indeterminate cells) − (mean of the two determinate cells).
  - The interaction: (self-indet − self-det) − (other-indet − other-det).
  - Each with a bootstrap over items within cells.
- *(operational)* "Small" interaction = its 95% interval lies within ±25% of the indeterminacy main effect.
- *(operational)* Think tokens and validity as in B3.

### B4: reward domain (RL-Zero)

- **H:** lengthening and the extreme-answer shift on self-report appear most in RL-Zero-General/Mix (LM-judge reward) and least in Math/Code.
- **Design:** the base model (few-shot) and RL-Zero × 5 (Math, Code, IF, General, Mix), on A, the B2 items and 100 D items, using the RL-Zero template.
- **Primary:**
  - per domain, (A think length / D think length) relative to base;
  - P(extreme) per domain.
- **Parse rate:** reported. Use the logprob readout for any checkpoint whose parse rate is under 70%.
- *(operational)*
  - 5 seeds.
  - D items = the first 100 base items by `draw_index`, option order o0.
  - For checkpoints without a think block, "length" = generated tokens before the answer line, stated per checkpoint.
- **Cut order (PI), if behind:** {Math, IF, General}.

### B5: dose-response over RL

- **H:** A-item think length and P(extreme) rise with RL step, and D length rises less.
- **Design:**
  - Checkpoints: 8 of the 55 step branches of `allenai/Olmo-3-7B-Think`, log-spaced, including the first and the last. The last must equal main: verify by weights hash and record it.
  - Protocol: B1's, with 10 seeds, plus 100 D items (as in B4).
- **Primary:**
  - The Spearman correlation of each metric (A think length, A P(extreme), D think length) with log step.
  - The A-vs-D slope difference: the OLS slope of A length on log step minus that of D length. Both are per-unit means standardised by the DPO value, with a bootstrap over units.
- **Cut order (PI), if behind:** 5 steps.

### B6: hard controls

- **(i) Positive control:** RLVR > DPO on MATH-500 level 5 + AIME 24/25 (16K cap).
- **(ii) Sycophancy:**
  - Items: MMLU/ARC items that DPO gets 40–70% right.
  - Elicitations, three: plain pushback; "are you sure?"; forced openers as in Xu & Daneshjou. No "double check".
  - Hypothesis: caving (initially correct answers switched to incorrect) > 0, and RLVR ≤ DPO.
- **(iii) Calibration:** on `TIGER-Lab/MMLU-Pro` (not the gated GPQA; amendment 2) with stated confidence. RLVR is more overconfident: higher ECE and a larger confidence − accuracy gap.
- **(iv) Refusal:** on SQuAD v2 unanswerable questions, RLVR < DPO.
- **Pairing:** by item.
- *(operational)* Item selection for (ii), by DPO accuracy in a 10-seed pre-pass, and all item lists are frozen in `items_sprint/` before the RLVR runs.

### B8: response style on a validated instrument

- **H:** extreme-response style RLVR > DPO, and acquiescence RLVR < DPO.
- **Instrument:** IPIP-NEO-120, public domain; the items are committed as JSON. Balanced keys.
- **Runs:** B1 readout, 10 seeds. Also DPO vs SFT with the DPO system prompt.
- **Primary:** three style indices, paired by item:
  - extreme = share in {1, 5};
  - midpoint = share of 3;
  - acquiescence = mean rating of positively keyed − reverse-keyed items, per facet, then averaged.

### M3: when the answer is settled

- **H:** RLVR settles its final Likert answer at a similar or earlier point than DPO, so most of RLVR's extra tokens (> 50%) come after the settle point.
- **Method:**
  - (a) After each think-block sentence, force "</think>ANSWER:" and read P(1..5). The settle point is the first sentence after which the argmax equals the final answer and stays there.
  - (b) Logistic probes (5-fold CV, grouped by wording) at prompt end and at 10/25/50% of the think block, at the M3–M6 layers.
- **Primary:**
  - the share of the trace after the settle point, RLVR − DPO;
  - probe accuracy at prompt end, per checkpoint.
- *(operational)*
  - Traces: valid B1 main-arm generations.
  - The forced suffix is B1's readout suffix ("\n</think>\n\nANSWER: "), so at the trace end it equals B1's readout.

### M4: persona vectors

- **H:**
  - Trait directions are stable across checkpoints (cos > 0.8).
  - RLVR shifts A-prompt projections on at least one trait beyond the across-seed variation, even where the Likert mean is flat.
- **Method:**
  - Contrastive system prompts as in Chen et al. 2025: 8 traits + uncertain + sycophantic, over 40 alpaca questions.
  - Directions from DPO; projections for all 3 checkpoints.
  - Only summaries and direction vectors are saved.
- **Primary:** the projection difference RLVR − DPO per trait, with a CI over wordings.

### M5: causal tests

- **H:**
  - Adding the RLVR − DPO stage-offset vector to DPO lengthens A traces in proportion to alpha, and much more than D traces.
  - Subtracting it from RLVR shortens A traces.
  - Ablating a backtracking direction in RLVR shortens A traces while changing the answer in < 10% of items.
- **Method:**
  - Offset vector = mean(RLVR − DPO) activation on A prompts, evaluated on held-out wordings; alpha ∈ {0.5, 1, 2}.
  - Backtracking direction = mean before "Wait" minus matched positions.
  - Activation patching DPO ↔ RLVR at prompt positions, layer by layer.
- **Primary:**
  - the A/D length ratio at alpha = 1;
  - the answer-change rate under ablation.
- *(operational)* Held-out split: 24/24 wordings, split within trait.

### M6: representation similarity

- **H:** DPO ↔ RLVR CKA is lower on A prompts than on D prompts in middle and late layers, while SFT ↔ DPO shows no such A/D gap.
- **Method:** linear CKA per layer on matched A, B2 and D prompts, at the prompt end and over the first 64 think tokens.

### Gate 1 exploratory analyses (added Oct 29)

All exploratory, on data already collected or in collection:
- `exp/analysis/gate1_m2.py`:
  - P(`</think>`) at sentence ends, RLVR vs DPO scorer, by component × trace source;
  - a stopping-hazard counterfactual for the share of the A length gap.
- M2 `--pair --kl-detail`: KL(RLVR ‖ DPO) attributed to the written token, plus a 20-bin positional profile. It has a new config hash; the registered pair shard is untouched.

### Infrastructure (no effect on any metric)

- **Results store:** private HF dataset `Shagsc/rlvr-se-runs`, the namespace of the available token (`HF_REPO`). Progress files and logs are mirrored by `exp/hfsync.py`.
- **Memory budget:** molab sandboxes were reclaimed without errors, so jobs keep to a 28 GiB RAM budget. The cgroup working set plus the job's expected RAM is checked at pre-flight, and the peak is sampled by `exp/memwatch.py`.
- **Model loading:** weights load straight to the GPU, with one model per process (two only on the GPU for M2 `--pair`).
- **B1 readouts:** still run with the HF model, in a separate process after the vLLM process has exited.

---

## Amendment 4 (Oct 29, 2026): Gate 1 (b) follow-ups (exploratory)

Committed and pushed **before any of these three analyses or jobs produced output**. All three are **exploratory**. They follow the Gate 1 (b) result that a lower stopping hazard explains only 0.2% [0.1%, 0.3%] of the A length gap (`results/sprint/M2/gate1_m2.md`). The statistics are as in "Fixed design facts": the unit is the A wording or the D base item, the bootstrap uses 10,000 resamples with seed 20261024, and the test is two-sided Wilcoxon.

### F1. Wrap-up onset (M2 `--wrapup`, GPU)

**Why a GPU run.** The M2 shards do not store the probability of conclusion openers, so the RLVR and DPO scorers are re-run in single mode with extra target groups.
- Traces: RLVR- and DPO-written. Components: A and D.
- Each run has its own config hash; the registered M2 shards are untouched.

**Openers.** The quantity is P(wrap-up) at think-block sentence starts: the position after each sentence end, i.e. `sent_next_tok`, as in M2. The opener sets:
- **Primary group:** first tokens, with and without a leading space, of "So", "Therefore", "Thus", "Final".
- **Reported separately:** "I'll go with" and "The answer". Their first tokens ("I", "The") are not specific to wrap-up, so these two are reported per word as first-token proxies and are not in the primary group.

**Measures.**
1. Mean P(wrap-up) at non-final sentence starts, RLVR scorer − DPO scorer, paired per unit, by component × trace source.
2. Onset: the position, as % of the think block, of the first sentence start with P(wrap-up) ≥ 0.2 under each scorer on the same trace. Compared with the realised close as a lag (close position − onset, in % of the think block); per unit, RLVR scorer − DPO scorer.

**Reading.** "The stopping decision shows up earlier than `</think>`" if the onset lag under a scorer is > 10% of the think block on average. RLVR "postpones" the decision if its onset is later than DPO's on the same text.

### F2. Where KL spikes (from `--kl-detail`, no extra GPU)

1. **Top 30 tokens** by total KL(RLVR ‖ DPO) over think positions, on RLVR-written A and D traces.
2. **First spike.** Threshold: the 95th percentile of the per-bin mean KL over the D traces, using the 20 equal relative-position bins of each RLVR-written D trace. For each A trace, record the first bin above that threshold and the number of bins above it.

**Reading.**
- **Early (the plan):** median first-spike bin ≤ 5 (first 25% of the trace) and median ≤ 3 bins above threshold.
- **Spread (repeated re-opening):** median ≥ 6 bins above threshold.
- Anything else is unclear.

*(operational)* Positions are relative bins, not sentence indices: sentence positions are not in the `--kl-detail` rows.

### F3. Prefix handoff (job PH, GPU)

**Material.**
- Items: the 48 A wordings and the first 50 D base items (o0).
- Traces: the frozen full-run traces of RLVR and DPO, using the first valid seed whose think block closed. For A: seed 0, else 1, else 2. For D: the o0 trace.

**Cuts.** Each trace is cut at the first sentence start (`sent_next_tok`, as in M2) at or after 25%, 50% and 75% of its think tokens.

**Continuation.**
- Each prefix is continued by **both** RLVR and DPO from the same token ids: prompt + think prefix. Their templates and default system prompts are byte-identical.
- 3 seeds, model-default sampling (0.6 / 0.95).
- Cap: 16,000 generated tokens including the prefix.

**Primary measure:** remaining think tokens, i.e. continuation tokens before `</think>`. A continuation that hits the cap is counted at the cap and flagged; the share of such continuations is reported.

**The 2 × 2.** Prefix source {RLVR, DPO} × continuing model {RLVR, DPO}, per cut, A and D separately. Per unit, the mean over seeds.
- Contrasts: continuing-model effect on RLVR prefixes, (RLVR cont − DPO cont); the same on DPO prefixes; prefix-source effect for each continuer.
- **Text share** = (L[DPO cont, RLVR prefix] − L[DPO cont, DPO prefix]) / (L[RLVR cont, RLVR prefix] − L[DPO cont, DPO prefix]), computed from means over units, with a bootstrap over units.
- **Reading:**
  - text share ≥ 0.5 on A at the 50% cut: the length lives in the text, since DPO goes long on RLVR prefixes;
  - ≤ 0.2: the length lives in RLVR's policy, since DPO wraps up quickly.

**Secondary measures:** the final-answer change rate relative to the trace's own answer, and the closing rate.

**Order.** It runs on N6 right after M2 `--kl-detail` and `--wrapup`, before B3a.

---

## Amendment 5 (Oct 29, 2026): Gate 1 call, lane order, added outputs, F1 robustness

Committed and pushed **before any of the jobs or analyses named here produced output under these specifications.** At commit time:
- B1's SFT and bridge arms, and PH (amendment 4), are running unchanged.
- M3, M5, B8, B5, B4, B2, B6 and B3b have not started.
- B3a has 78 RLVR generations from Oct 28 (unchanged design).

**Gate 1 call (PI).** The headline: RLVR re-opens deliberation (an earlier wrap-up signal, then a diffuse divergence on discourse and hedging tokens, not a stopping deficit), and that reasoning drives the extreme-answer shift. Everything below is planned around testing it.

### Lane order

| lane | order |
|---|---|
| N2 | B1 (finish the SFT arms + bridge) → B8 → B5 → B2 → B4 → B6 → B3b |
| N6 | PH → M3 → M5 → B3a → M4 → M6 |

- **Cut order if behind:** M6, then B3a, then M4, then B4 to {Math, IF, General}, then B5 to 5 steps.
- **B6 must ship at least** the MATH-L5/AIME positive control.
- *(operational)* The running lanes switch to the new order at their next job boundary: `exp/launch/queue.sh` reads `exp/launch/queues/<lane>.txt`. No running job is interrupted.

### Added outputs

**M3**, in addition to amendment 3:
- **(a′) Per-sentence trajectory.** For A (B1 main-arm traces), the trajectory of P(extreme) = P(1) + P(5) and of E[rating] from the per-sentence readout. It is summarised in 10 bins of relative position, RLVR vs DPO (and SFT), paired by wording.
- **D analogue.** On the frozen full-run D traces (RLVR and DPO, valid, o0, the first 50 base items), the same per-sentence readout over the letters A–E. The trajectory is of P(max letter) and P(final letter).
  - "Extremity" is not defined for D. P(max letter) is its decisiveness analogue.
- **(a″) Wrap-up onset inside the M3 pass.** From the same forward, P(wrap-up) is read at each sentence start: the F1 primary group (So / Therefore / Thus / Final), and the same group without "So". The onset is the first sentence start with P ≥ 0.2.
  - **Primary added test:** per trace, mean P(extreme) after the onset minus before it, RLVR vs DPO, paired by wording.
  - **Registered direction (from the Gate 1 call):** extremity grows after the onset more for RLVR.
  - Traces with no onset are reported, not imputed.

**M5**, refining amendment 3:
- **Backtracking direction.** Mean residual at the positions just before " But", " maybe" and " Wait" (each first token, with or without the leading space), minus matched control positions. Controls are other sentence-internal positions in the same traces, matched on relative position (same decile) and on the preceding token being a sentence end vs not. Taken from DPO and RLVR traces at the M3–M6 layers; chosen on held-out wordings as registered.
- **Outcomes.** Ablation (projecting the direction out in RLVR) and steering (adding ±alpha × the stage-offset vector) both report:
  - think length (A and D);
  - answer change;
  - **extremity**: P(extreme) from the B1 readout at the end of the steered or ablated trace, and the sampled-answer extreme share.

**B5 and B4.** P(extreme) is a **primary** outcome alongside length (B4: per domain; B5: Spearman with log step and the A-vs-D slope difference for length).

### F1 robustness (exploratory)

The F1 wrap-up onset analysis (amendment 4) is rerun with the opener group excluding " So": Therefore / Thus / Final. It uses the stored per-word columns of the existing `--wrapup` shards (P(group) − P("So") first tokens; the groups' first tokens are disjoint). No new GPU run is needed.

### Repair of the M2 pair shard (approved by the PI)

`runs/M2/pair_rlvr_dpo/b35466982f5e/shard_000.parquet` on HF is a torn copy (400,000 bytes, no parquet footer).
- **Rebuild:** from the complete progress file (1842 rows), with row-level equality verified, and uploaded to the same path as a new HF commit (history keeps the torn version).
- **Logged:** in `docs/sprint/disclosure-ledger.md`.
- **This is the only overwrite.**
