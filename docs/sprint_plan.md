# RLVR Side-Effects Project — Sprint Plan

## 1. Literature review

**The starting claim.** A Feb 2025 paper argued reasoning-optimized models show more instrumental-convergence-like behavior (self-preservation, resource acquisition, goal persistence) than non-reasoning models, using a 76-task benchmark called InstrumentalEval across six behavior categories. Its central comparison confounds training method with model identity — it compares direct-RL-trained models like o1 against RLHF-trained models from different labs, so model family, data, and scale vary alongside the training method it's trying to isolate.

**What's been done since, specifically on Olmo 3.** Because AllenAI released full checkpoints across every post-training stage (base → SFT → DPO → RLVR, and RLVR 3.0 → 3.1), Olmo 3 has become a popular natural experiment for exactly this kind of question. As of this review:

- A direct replication ran Olmo 3 32B Think's SFT, DPO, and RLVR checkpoints on five safety benchmarks (Petri, ODCV-Bench, StrongREJECT, MASK, ImpossibleBench) with LiveCodeBench as a capability control. Result: the checkpoints were statistically indistinguishable on every safety dimension while capability roughly doubled — a clean negative result on "does RLVR increase misaligned behavior."
- An eval-awareness study tracked verbalized eval-awareness (VEA) across the same checkpoint lineage on Olmo-3.1-32B-Think, at ten RLVR checkpoints across five benchmarks. Finding: VEA is near-zero in pretraining, rises substantially at SFT, is sharply suppressed by DPO, and re-emerges strongly during RLVR — a non-monotone, training-objective-dependent trajectory, not a steady increase.
- A CoT-faithfulness study traced hint-following/concealment across the same four stages and also found non-monotone shifts.
- A SPAR project ran 11 GRPO checkpoints (Tülu 3.1 lineage, a close sibling of Olmo) through 24 authored situations, with paraphrase and answer-order controls (6,912 total responses). Average risky-behavior rates stayed flat within a predeclared band — but *permutation consistency* (does the model give the same answer under meaning-preserving rewording) fell from 0.302 to 0.156, and answer-order sensitivity rose sharply late in training. This was interpreted as measurement drift, not a real behavior shift, and is the strongest reason to distrust any single-benchmark "instrumental convergence increases with RLVR" result on this model family.
- Separately, checkpoint-level work on fine-tuning in general shows emergent misalignment appearing *after* the primary task has nearly converged, and that early stopping removes the misalignment while keeping ~93% of task performance — this is the direct answer to "does misalignment only emerge after capability saturation," just not run on InstrumentalEval specifically.
- Independent critiques of InstrumentalEval itself show its outputs shift substantially under short prompt-suffix changes across Qwen3 variants — i.e., the instrument is noisy exactly where you'd need it to be stable to detect a small RLVR-stage effect.

**What's been done on personality/trait drift across training stages (RQ3's territory).** This turns out to be a more active area than the original brief suggested:
- "The Personality Illusion" (2025) traced Big Five/self-regulation trait self-reports across training stages and found instructional alignment (RLHF/DPO/instruction tuning as a bundle) increases openness, agreeableness, and self-regulation while reducing neuroticism, and stabilizes trait expression — but self-reported traits don't reliably predict behavior on downstream tasks. This is a real dissociation finding: self-report ≠ behavior.
- A 2026 psychometric-instrument paper derived a bottom-up five-factor structure from LLM self-descriptions (Responsiveness, Deference, Boldness, Guardedness, Verbosity) and argues this factor structure is best read as an artifact of RLHF-style alignment training rather than a genuine personality analog, and does not map onto human Big Five.
- A separate psychometric study across seven models found RLHF specifically predicts *lower* measured psychopathy scores.
- None of this work isolates the **RLVR stage specifically** (as distinct from RLHF/DPO/instruction-tuning bundled together), and none of it uses the Olmo 3 checkpoint ladder, which is the gap RQ3 sits in.

**What's been done on general-capability side effects of RLVR (relevant to "outside the training domain").** This is the most directly useful literature for RQ3's design:
- Multiple 2025–2026 papers document that RLVR trained narrowly on one domain (math, code) tends to *degrade* performance on other domains — e.g., RLVR training on HumanEval+ produced a measured drop on MATH, and training on History dropped MMLU-Pro performance — interpreted as the model internalizing narrow domain heuristics at the expense of general reasoning.
- A separate line of work documents RLVR increasing hallucination rates and jailbreak vulnerability as side effects of prolonged reasoning-focused RL, motivating replay-based mitigations.
- This gives you a **directly testable, already-validated pattern** (general capability regression off the RL domain) that you can look for *in combination with* trait/behavioral shift, rather than inventing a new claim from scratch.

**Net assessment of the gap.** The instrumental-convergence question (original RQ1/RQ2) is close to answered — multiple negative results, a documented measurement-noise confound, and a construct-validity problem with the benchmark itself. The RQ3 territory — does the RLVR stage specifically (isolated from SFT/DPO) shift personality/general-reasoning behavior, and is that shift a real signal or measurement noise the way the SPAR project found — is genuinely less covered, especially on Olmo 3's exact ladder, and connects cleanly to an established side-effect (cross-domain capability regression) that others have already validated on different models.

---

## 2. What's been done vs. what you'd be doing

**What's been done (you would NOT be redoing this):**
- Confirming RLVR checkpoints don't clearly increase safety-benchmark risk scores on Olmo 3 (done, negative result).
- Confirming eval-awareness and CoT-faithfulness shift non-monotonically across Olmo 3's stages (done).
- Establishing that misalignment from fine-tuning emerges after task convergence and that early stopping removes it (done, different benchmark/model).
- Establishing that RLVR narrows cross-domain capability on other model families (done, not on Olmo 3).
- Establishing that RLHF/DPO bundled together shifts Big Five self-report traits (done, not RLVR-isolated, not on Olmo 3).

**What you would be doing:**
A checkpoint-level study that isolates the **RLVR stage specifically** (not RLHF/DPO/SFT bundled) on the Olmo 3 ladder, and asks two linked questions with a fixed, paraphrase-robustness-checked protocol:
1. Does a fixed personality/trait probe battery and a general-reasoning probe (deliberately chosen *outside* Olmo's RLVR training domains — i.e., not math/code) shift measurably across SFT → DPO → RLVR checkpoints, in a pattern distinguishable from prompt-order/paraphrase noise?
2. Does any such shift track the RLVR-vs-MATH capability curve — i.e., does the "narrow domain heuristic" side effect documented on other models (regression on off-domain tasks) also show up on Olmo 3, and does it correlate with any trait shift, rather than being coincidental?

The contribution is narrow and honest: not "RLVR causes instrumental convergence" (multiple studies already say no on this evidence), but "here is what RLVR specifically — isolated from the earlier stages — does to general-purpose behavior outside its reward domain, measured with the robustness checks the field just learned it needs."

---

## 3. Impact and relevance to AI safety

**Why this matters generally.** As RL-with-verifiable-rewards becomes the dominant post-training paradigm (it's how current frontier reasoning models are built), a first-order safety question is whether optimizing narrowly for checkable correctness produces side effects outside the reward's intended scope — misalignment, capability regression, or personality/value drift the developers didn't intend and might not notice because they're not looking outside the reward domain. This is a "silent side effect" risk category distinct from the more publicized alignment-faking or reward-hacking failure modes: it doesn't require a model to be strategic, just for optimization pressure on a narrow objective to bleed into unrelated behavior.

**Why this specific angle matters more than the crowded ones:**
- The instrumental-convergence angle (original RQ1) has genuine forecasting importance but is already answered in the negative on this exact checkpoint family, with good methodological reasons (construct validity, measurement noise) to expect further attempts to also come back null or uninterpretable. Re-running it adds little.
- The RQ3 angle is scientifically cleaner: it doesn't require a contested benchmark (InstrumentalEval) and it connects to an established, validated phenomenon (cross-domain capability regression from narrow RLVR) rather than a contested one.
- Methodologically, explicitly building in the paraphrase/order-robustness check that the SPAR project's finding demands is itself a small contribution to the field's practice — it directly responds to a documented failure mode (checkpoint studies mistaking measurement drift for behavior drift) rather than repeating it.

**Honest limitations to state in your write-up:** Olmo 3's RLVR stage is small relative to frontier post-training, so results won't generalize to heavily-RL'd frontier models without caveats. A null result here (no detectable off-domain shift) is still informative and easy to write up honestly, especially if you can show your probe *would* have detected a shift by validating it against the known cross-domain capability regression pattern first.

---

## 4. Five-week plan

**Week 1 — Narrow and set up (Unit 1 territory, finishing what you started)**
- Finalize probe design: pick 10–20 personality/trait items (e.g., adapted from Big Five self-report + a couple of behavioral, not just self-report, tasks, since self-report/behavior dissociation is a documented issue) and 10–20 general-reasoning items from a benchmark clearly outside Olmo's RLVR training domains (math/code/instruction-following/chat) — e.g., commonsense reasoning or an ethics-judgment set.
- Decide checkpoint set: start with Olmo 3 **7B** (compute-cheap) across base → SFT → DPO → RLVR (and RLVR 3.0 → 3.1 if time allows), using the Think or RL-Zero pathway checkpoints AllenAI released at each training milestone.
- Write the paraphrase/order-robustness protocol: 3–4 meaning-preserving rewordings per item, shuffled answer order where applicable — modeled directly on the SPAR project's method so your robustness numbers are comparable to theirs.
- Deliverable: a locked probe set + protocol document, in the repo, before any model runs.

**Week 2 — Pilot run and calibration**
- Run the full probe battery (with all paraphrase/order variants) on 2 checkpoints only (say, DPO and final RLVR) to sanity-check the pipeline end-to-end: prompting, response parsing, judge/scoring reliability.
- Compute your own permutation-consistency metric on this pilot and compare it to the SPAR project's 0.302 → 0.156 range as a calibration check — if your numbers look wildly different, debug before scaling up.
- Deliverable: pilot results + a short internal note on judge reliability and any protocol fixes needed.

**Week 3 — Full checkpoint sweep**
- Run the complete probe battery across all chosen checkpoints (SFT, DPO, RLVR early/mid/late steps if step-level checkpoints are available).
- In parallel, run the MATH benchmark (already public, cheap) across the same checkpoints to get your capability-control curve.
- Deliverable: raw results across the full checkpoint ladder for both probes and the capability control.

**Week 4 — Analysis**
- Plot trait/reasoning-probe scores against training stage and against MATH performance.
- Run the robustness check: is any apparent shift stable across paraphrases/order, or does it wash out the way the SPAR project's did? This determines whether your headline finding is "RLVR shifts X" or "apparent shifts are measurement noise, here's the evidence" — both are legitimate, honest results.
- Cross-check against the known cross-domain-regression pattern from the RLVR-narrows-generalization literature: does Olmo 3 show the same off-domain capability drop other models have shown? This validates that your off-domain probe is sensitive enough to detect real effects, which strengthens either a positive or null finding on the trait side.
- Deliverable: analysis notebook + draft figures.

**Week 5 — Write-up and repo polish**
- Write the public write-up: why you chose this (with the crowdedness context above, honestly stated), what you found, what you'd recommend next (e.g., "run this at 32B if compute allows," or "extend the paraphrase-robustness protocol to instrumental-convergence-style benchmarks broadly").
- Clean and document the GitHub repo so someone else can rerun your probe battery on a different checkpoint ladder.
- Update your project log with a retrospective: which predictions held, which didn't, what you'd test first if you had another sprint.

---

## 5. Two-day head-start setup (before sprint applications close)

Goal: have a working, reproducible skeleton so "primary setup already exists" is a visible bonus in your application, without needing to have run real experiments yet.

**Day 1**
1. **Repo scaffold.** Create a GitHub repo with a clear structure: `/probes` (your trait + reasoning item sets, even as placeholders), `/scripts` (model loading, inference, scoring), `/data` (benchmark files), `/results`, `/notebooks`, and a README stating the research question, the crowdedness context, and a link to the sprint doc.
2. **Model access.** Pull the Olmo 3 7B checkpoints you plan to use from Hugging Face (`allenai/Olmo-3-7B-...` — check the Think and RL-Zero repos for stage-labeled revisions, since Olmo names post-training checkpoints by step, e.g. `step_XXXX`). Confirm you can load at least one checkpoint locally or on whatever compute you'll use (`transformers >= 4.57.0` is required for Olmo 3). Don't run full inference yet — just confirm the load works and note VRAM/compute needs.
3. **Dataset pull.** Download the MATH benchmark subset AllenAI uses for RLVR-MATH, and pick + download one off-domain reasoning/commonsense dataset (something small and standard) so you have real data files in `/data`, not placeholders.

**Day 2**
4. **Draft probe items.** Write a first-pass version of 8–10 personality/trait items and 8–10 general-reasoning items directly in `/probes` — even a rough draft is enough to show reviewers you've operationalized RQ3 into something testable, not just an idea.
5. **Minimal inference script.** Write a script that loads one checkpoint, runs one probe item through it, and saves the raw output to `/results`. It doesn't need to be the full pipeline — just prove the mechanical path (model → prompt → logged response) works end to end.
6. **One paragraph in the README** explicitly citing the SPAR permutation-consistency finding and stating that your protocol will replicate that robustness check — this signals to reviewers that you've engaged with the literature seriously, not just picked a trendy topic.

This gets you a repo with real (if early) code, real downloaded data and model weights, and a demonstrably narrowed research question — which is a strong signal above what's asked for, without pretending you've already run the actual study.

---

## Addendum: revised 2-day plan for molab (GPU-backed, via Claude Code pairing)

Since you're running this through a paired Claude Code session connected to a molab notebook (NVIDIA RTX Pro 6000, 96GB VRAM), you can skip the "minimal compute / placeholder model" workaround entirely and work with real Olmo 3 7B checkpoints from the start. Sessions auto-shut down after 12 hours or 90 minutes idle, so structure work in chunks you can checkpoint/save rather than one long uninterrupted run.

**Day 1 (molab)**
1. Repo scaffold as marimo notebook cells: a setup cell (imports, HF auth if needed), a data-loading cell, a model-loading cell, a probe-running cell, a results-saving cell — mirrors the original `/probes /scripts /data /results /notebooks` structure but as sequential notebook sections.
2. List available Olmo 3 7B checkpoint revisions on Hugging Face (`allenai/Olmo-3-7B-...` — check Think and RL-Zero repos; post-training checkpoints are named by step, e.g. `step_XXXX`) and map which revisions correspond to SFT / DPO / RLVR stages.
3. Load the final RLVR checkpoint first (single load, no quantization needed given 96GB VRAM) and confirm generation works with a simple test prompt.
4. Pull the MATH benchmark subset (RLVR-MATH) and one off-domain reasoning/commonsense dataset into the notebook's data cell.

**Day 2 (molab)**
5. Draft the first-pass probe items (8–10 personality/trait items, 8–10 general-reasoning items) directly as a notebook cell/data structure.
6. Write and run a minimal end-to-end pass: load one checkpoint → run one probe item → log the raw response — proving the full pipeline works on real weights, not a placeholder.
7. Save/export the notebook back to your local repo (or keep it mirrored via molab's GitHub sync) so the sprint application shows a working, GPU-verified pipeline, not just local scaffolding.
8. Note session-time usage in your project log — useful for estimating how much of the Week 2–3 full checkpoint sweep will fit in molab's 12-hour session windows.

**One clean prompt to hand the paired session:**
```
We have GPU access via molab (RTX Pro 6000, 96GB VRAM), so skip any placeholder-model 
approach. Set up Day 1 of the RLVR sprint project directly on real Olmo 3 7B checkpoints:
1. List available Olmo-3-7B checkpoint revisions on Hugging Face and map them to 
   SFT/DPO/RLVR stages (check both the Think and RL-Zero repos).
2. Load the final RLVR checkpoint and confirm generation works with a test prompt.
3. Pull the RLVR-MATH dataset and one off-domain reasoning/commonsense dataset.
Structure this as clear notebook cells so I can rerun each step independently, and 
keep me updated on VRAM/session time usage as we go.
```
