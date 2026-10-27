# Disclosure: the frozen run gave SFT a different system prompt from DPO and RLVR

Written Oct 28, 2026, on branch `exp/mechanism-sprint`. Found by job B0 (`docs/sprint/checkpoint_inventory.md`, Oct 27), after the study was frozen and analysed. **No frozen file is changed.** This document marks which frozen results are affected. The follow-up is registered in `docs/sprint/registration-2026-10-27.md`, amendment 1.

## What happened

`scripts/full_run.py` built every prompt with `tok.apply_chat_template(messages)`, where `messages` holds only the user turn. When there is no system message, each Think chat template inserts its own default system block. The SFT checkpoint's default differs from the DPO and RLVR defaults; DPO and RLVR are byte-identical to each other. So every frozen generation ran with its checkpoint's own default system prompt.

## The two system prompts, verbatim

Rendered system block (from `<|im_start|>system` to `<|im_end|>`, inclusive), pinned revisions from `docs/checkpoint_map.md`:

**Olmo-3-7B-Think-SFT** (`6ff857587e04`):

```
<|im_start|>system
You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>
```

**Olmo-3-7B-Think-DPO** (`7b18bf927b43`) and **Olmo-3-7B-Think** (RLVR, `d97e442d7cc6`), byte-identical:

```
<|im_start|>system
You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>
```

**Differences:**

| SFT | DPO / RLVR |
|---|---|
| `Olmo` | `OLMo` |
| (absent) | `function-calling` before "AI assistant" |
| `December 2024` | `November 2024` |
| (absent) | ` You do not currently have access to any functions. <functions></functions>` appended |

**Token counts** (pinned tokenizers, transformers 5.17.0):

| | text only (characters / tokens) | whole block with markers (tokens) |
|---|---|---|
| SFT | 157 / 40 | 45 |
| DPO / RLVR | 249 / 59 | 64 |

- Every frozen prompt is therefore exactly **19 tokens longer for DPO and RLVR than for SFT**; checked on all 518 items, min = max = 19. Example: A_confidence_0 has 163 prompt tokens for SFT and 182 for DPO and RLVR.
- The two templates are identical outside this block.
- The SFT and DPO tokenizers have identical vocabularies and give identical ids on every frozen prompt (the `tokenizer.json` files differ in bytes, not in encoding).

## Affected: every frozen result that compares DPO with SFT (confounded)

Each result below compares two checkpoints that received different system prompts. A difference, or the absence of one, cannot be attributed to the DPO stage alone. The prompt difference (identity wording, the "function-calling" framing, the empty function list, 19 tokens) could add to, cancel or create it. This applies to null results as much as to signals. **Status of all of them: confounded.** The frozen values stand as reported; they are not re-scored.

### 1. Table 3, DPO-SFT rows (`results/analysis/table3_comparison.csv`, all 28)

| probe | form | tier | DPO − SFT [95% CI] | frozen status | status now |
|---|---|---|---|---|---|
| trait:approval | cc-only | primary | 0.2500 [-0.4444, 1.0000] | inconclusive | **confounded** |
| trait:assertiveness | cc-only | primary | -0.2778 [-0.6667, 0.1667] | inconclusive | **confounded** |
| trait:competitiveness | cc-only | primary | 0.5000 [0.0000, 1.0000] | inconclusive | **confounded** |
| trait:composure | cc-only | primary | -0.4444 [-0.8889, 0.0000] | no meaningful change | **confounded** |
| trait:confidence | cc-only | primary | 0.3889 [-0.0556, 0.8333] | inconclusive | **confounded** |
| trait:hedging | cc-only | primary | 0.1667 [-0.4444, 0.6667] | no meaningful change | **confounded** |
| trait:humility | cc-only | primary | -0.1667 [-0.8889, 0.6681] | inconclusive | **confounded** |
| trait:risk | cc-only | primary | -0.1111 [-0.6667, 0.5556] | no meaningful change | **confounded** |
| sensitivity: hedging without original wording | cc-only | secondary (sensitivity), uncorrected | 0.2000 [-0.4667, 0.8000] | inconclusive | **confounded** |
| Sycophancy caving | iaf | primary | 0.0000 [0.0000, 0.0000] | no meaningful change | **confounded** |
| Sycophancy caving | cc | primary | 0.0000 [0.0000, 0.0000] | no meaningful change | **confounded** |
| Calibration accuracy | iaf | secondary, uncorrected | -0.0500 [-0.1125, 0.0000] | inconclusive | **confounded** |
| Calibration accuracy | cc | secondary, uncorrected | -0.0256 [-0.0769, 0.0256] | inconclusive | **confounded** |
| Calibration gap | cc-only | primary | -0.0032 [-0.0513, 0.0500] | no meaningful change | **confounded** |
| Calibration Brier | cc-only | secondary, uncorrected | 0.0135 [-0.0273, 0.0547] | inconclusive | **confounded** |
| Commonsense accuracy | iaf | primary | 0.0067 [-0.0267, 0.0433] | no meaningful change | **confounded** |
| Commonsense accuracy | cc | primary | -0.0103 [-0.0377, 0.0205] | no meaningful change | **confounded** |
| Commonsense permutation consistency | cc | secondary, uncorrected, complete-case only, no band | 0.0000 [-0.0479, 0.0479] | inconclusive | **confounded** |
| MATH accuracy | iaf | primary | -0.0400 [-0.1200, 0.0400] | no meaningful change | **confounded** |
| MATH accuracy | cc | primary | 0.0000 [0.0000, 0.0000] | no meaningful change | **confounded** |
| sensitivity: MATH boxed_candidate rescoring | iaf | secondary (sensitivity), uncorrected | 0.0000 [0.0000, 0.0000] | no meaningful change | **confounded** |
| Reasoning length A | cc | primary | -12.0000 [-66.8333, 63.1667] | inconclusive | **confounded** |
| Reasoning length B | cc | primary | 155.0000 [114.5000, 273.5000] | signal | **confounded** |
| Reasoning length C | cc | primary | 145.5000 [75.0000, 171.0000] | signal | **confounded** |
| Reasoning length D | cc | primary | 101.2500 [92.7500, 129.2500] | signal | **confounded** |
| Reasoning length E | cc | primary | 55.5000 [-21.0000, 130.0000] | inconclusive | **confounded** |
| Reasoning length A-D | cc | primary | 103.2500 [91.5000, 128.8333] | signal | **confounded** |
| Reasoning length H4 (A minus D) | cc | primary | -113.2500 [-180.8354, -40.7500] | signal | **confounded** |

The signals among them are reasoning length on B, C, D, A–D pooled, and the H4-style A-minus-D interaction for DPO − SFT.

### 2. Length hypothesis H3 (analysis plan §3F): confounded

"SFT ≈ DPO overall (A–D pooled), no signal". Frozen verdict: **not replicated**. DPO − SFT median paired difference was +103.25 tokens [91.50, 128.83], 312 units (`results/analysis/summary.md`). Both the verdict and the size of the difference are confounded.

### 3. Claim C3: confounded

"DPO lengthens B, C and D but not A; consecutive stages lengthen different content" (`docs/literature-review.md`, `docs/olmo3-primary-check.md`). It rests on the DPO − SFT reasoning-length rows for A (−12.0 [−66.8, 63.2], inconclusive), B (+155.0), C (+145.5) and D (+101.25), all signals except A. The component pattern, which is the residue claimed as new, is part of the confounded comparison.

### 4. Claim C8: confounded

"DPO is more stable across seeds than SFT" (exploratory addendum 4, `results/analysis/exploratory_dithering.md`): the mean noise-floor difference is DPO − SFT −0.296.
- (a) simple: [−0.474, −0.123], Wilcoxon p 0.0234.
- (b) nested, primary: [−0.512, −0.105], bootstrap p 0.0026.
- DPO lower on 7 of 8 traits.

### 5. The DPO − SFT extreme-answer drop: confounded

The share of component A answers that are 1 or 5, pooled over valid generations: **SFT 0.454 (n 141) → DPO 0.203 (n 143)**, a drop of 25.1 points. Per wording (n = 48), the paired DPO − SFT difference is −0.247 [−0.340, −0.149] (recomputed by `exp/compare_sft_matched.py`). The RLVR − DPO part of this lead (+17.8 points) is not affected (see below).

### 6. Also affected, found while compiling this list (not in the original request)

- **Claim C5's SFT parts:** "the per-token switch rate equals SFT's; DPO the low outlier". This covers exploratory addenda 1–2: switches and candidate mentions per 1000 tokens, DPO − SFT and RLVR − SFT rows.
- **Every other exploratory DPO − SFT row:** measures 1–3 in `exploratory_dithering.md`, i.e. dithering, repetition, two-instruction compliance. Its own text already labels DPO − SFT as "descriptive only".
- **The RLVR − SFT rows** of addenda 2 and 4.
- **SFT's per-checkpoint values:** tables 1, 2 and 4, the SFT columns of table 3 and the figures, and addendum 3's within-SFT length–rating coupling. These are not contrasts, but they describe SFT *under its own default prompt*. They are valid as such and must not be read as SFT under the DPO/RLVR prompt.
- **Pilot SFT comparisons** (`docs/COMPARISON_rlvr_dpo_sft_pilot.md`, `results/sft_pilot_v3.*`). `scripts/sft_pilot_v3.py` also renders with the checkpoint's own template, so it is presumably the same confound. Not verified line by line.

## Not affected: RLVR − DPO

DPO and RLVR received byte-identical system blocks; both tokenizers give identical ids on every frozen prompt. Every RLVR − DPO result is unaffected by this issue:
- table 3 RLVR-DPO rows and their Holm families;
- the trait bands, which use max(s(t, RLVR), s(t, DPO));
- H1, H2, H4;
- C1, C2, C4, C6, C7;
- the RLVR − DPO exploratory rows;
- the +17.8-point RLVR − DPO extreme-answer difference;
- the sprint's primary contrasts.

Holm in table 3 is applied per comparison, so no RLVR − DPO p-value is pooled with a DPO − SFT one.

## What is being done (registration amendment 1)

1. **Matched-prompt rerun:** SFT is rerun through the frozen pipeline with only the system prompt changed to the DPO/RLVR default: `scripts/full_run_matched_prompt.py --checkpoint sft --system-prompt-from dpo` → `results/sprint/sft_matched_prompt/`. `scripts/full_run.py` itself is unchanged and still passes the manifest guard.
2. **Recompute:** `python -m exp.compare_sft_matched` recomputes every result in sections 1–6 that comes from the frozen analysis code: table 3, H3, C3, C8, the extreme-answer drop and all exploratory SFT rows. It uses the frozen analysis code unchanged, prints frozen against matched side by side and applies the robustness rule fixed in the amendment. First it checks that the frozen recomputation reproduces the committed numbers; all three checks pass in `requirements_analysis.txt`'s numpy 2.3.5.
3. **B1:** adds an SFT arm with the DPO/RLVR system prompt next to the SFT own-template arm. B3 keeps own templates, so its DPO − SFT contrasts remain confounded.

**Reporting rule:** the frozen DPO − SFT numbers stay as frozen, each marked "confounded (system prompt)". The matched values are reported next to them as a post-freeze sensitivity analysis, never in their place.
