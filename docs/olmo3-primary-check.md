# Olmo 3 technical report: primary-text check

Oct 27, 2026. Read-only verification. Source: Team Olmo (Ettinger et al.), "Olmo 3", arXiv:2512.13961v2, the PDF (118 pages), text extracted with pypdf from the arXiv PDF and searched for every occurrence of length, verbosity, token counts, normalisation and DAPO / Dr GRPO.

- **Page numbers:** the report's printed page numbers equal the PDF page numbers (section 4 starts on page 36 in both).
- **Quotes:** all quotes are verbatim from the extracted text. pypdf drops some spaces between words in italic or small-caps runs (for example "Olmo 3 Thinkrecipe"); those spaces are restored in brackets only where needed for readability. No words are changed.
- **Relevant sections read in full:** section 4 Olmo 3 Think (pp. 36–52: 4.1 results, 4.2 Think SFT, 4.3 Delta-learning DPO, 4.4 OlmoRL, 4.5 key findings), section 5.3–5.5 Instruct DPO/RL (pp. 57–62), and appendix A.6.2–A.6.3 and Figure 41 (pp. 105–110).

## (a) Is response, reasoning or generation length across the Think SFT → DPO → RLVR stages reported?

**Not as an evaluation-time comparison across stages.** The report contains no table or figure giving response length per stage (Think SFT vs Think DPO vs Think RLVR) on any benchmark or prompt set, and none on self-report, chat or other non-verifiable prompts.

**What it does report is length during RL training, on the RL training prompts:**

- §4.5, pp. 50–51: "Interestingly, we find that sequence lengths first slightly dip and then slowly increase over time. This is likely due to the reasoning SFT and DPO already training the model to produce long reasoning traces of up to the maximum response length of 32K tokens."
- Appendix A.6.3, p. 106: "The response length becomes longer and stabilizes at a higher level." Figure 41 (p. 110), captioned "Reward, KL, response length, and per-verifier reward over the final RL run for Olmo 3 Think", plots response length on an axis running from about 14000 to 20000 tokens over about 1K steps.
- §4.4.3, p. 48: "For our final models, we generated rollouts with a maximum size of 32K tokens in length, averaging more than 10K tokens (for the reasoner models)." Same page: "With Olmo 3, at a 32K generation length, we see a mean generation length of 14628 and a maximum of 32K".
- §4.5, p. 51: "Using a reasoner SFT or DPO as a starting point stresses RL training to its limits, as the model starts with extremely long average generation lengths."

**What it says about preference tuning and length.** This is a general statement made in the Instruct section, §5.3.1, p. 59, about their preference-data pipelines, including delta-learning pairs:
> "Similarly, we empirically observe that preference pairs made with the delta-learning heuristic also exhibit length bias; larger models generate longer responses (Figure 23). Thus, models often learn this length bias in addition to the intended useful quality signal during preference tuning, after which its generation length per prompt increases significantly. While this increased length is empirically useful for reasoning tasks, excessive verbosity can be undesirable for common real-use settings".

Figure 23 shows chosen-minus-rejected token-length differences for the Instruct pairs (GPT-judged and delta-learning, non-thinking). No equivalent measurement is shown for the Think DPO pairs.

**Consequence for C1 and C3**
- **C1 (RLVR lengthens self-report reasoning).** The developers document that response length increases during Think RL on the training distribution. That covers math, code, precise instruction following and general chat, at about 14–20K tokens, starting from an already long DPO model. So "the RL stage of Olmo 3 Think lengthens generations" is **reported by the developers**, for training prompts. They do **not** report length at evaluation time, on self-report or other non-verifiable prompts, or as a paired comparison against the DPO checkpoint. C1 keeps a narrower residue:
  - it is a paired, per-item measurement at evaluation time, on self-report items, against the released DPO checkpoint;
  - it is large (+577 tokens median paired difference at a 4K cap) even though training-time growth was described as slight ("first slightly dip and then slowly increase").
  
  It should no longer be framed as "the RL stage lengthens outputs", which the report already says.
- **C3 (DPO lengthens B, C, D but not A).** The developers state, as a general expectation, that preference tuning on delta-learning pairs carries a length bias that "increases significantly" generation length. Our finding that DPO lengthens some components is therefore **anticipated by the developers**. Not reported: that it lengthens commonsense, sycophancy and calibration but **not** self-report (−12 tokens). That component specificity is the remaining residue.

**Also relevant: the "RLVR" stage is not purely verifiable.** §4.4, p. 44: "The third stage of post-training is reinforcement learning with a mixture of verifiable and LM-judge rewards across a variety of domains." Table 20 (p. 46) lists 21,387 General Chat prompts (Tulu 3 SFT 7,129; Wildchat-4.8M 7,129; Multi-Subject RLVR 7,129) out of 104,869 Think RL prompts. They are scored by an LM judge. p. 46: "Chat—open-ended: We pass the response to an LM judge and ask the judge to give a score in [0, 1] based on the quality of the response without any reference answer." Footnote 36, p. 46: the judge is "Qwen3 32B ... with thinking mode turned off". The same footnote says: "We additionally experimented with ... length-control (Aggarwal and Welleck, 2025) verifiers, but did not find it useful for downstream performance."

**Implication:** the checkpoint we call "RLVR" was also trained against an LLM-judge reward on open-ended chat. LLM judges are known to prefer longer responses (Dubois et al. 2024, arXiv:2404.04475). An LM-judge length preference is a **candidate mechanism** for lengthening on non-verifiable prompts, separate from the verifiable-reward dynamics. The paper should not describe the effect as caused by "verifiable rewards" alone. "The RLVR stage (verifiable plus LM-judge rewards)" is the accurate description.

## (b) The OlmoRL objective and length bias

**Confirmed, with one attribution correction.** §4.4.1, p. 45, list of improvements over vanilla GRPO:
> "• Token-level loss: We use a token-level loss to normalize the loss by the total number of tokens across the batch (Yu et al., 2025), rather than per-sample to avoid a length bias."

The final objective (Eq. 1, p. 45) carries the batch-token normaliser 1/Σᵢ|yᵢ|. The same list also contains:
> "• No standard deviation normalization: When calculating advantage, we do not normalize by the standard deviation of the group, following Liu et al. (2025b). This removes a difficulty bias, where questions with low standard deviation in their rewards (e.g., too hard or too easy) have their advantages significantly increased by the normalization term."

**Corrections to the secondary summaries:**
- **Attribution:** the batch-token normalisation is attributed to **Yu et al. 2025 (DAPO)**, not to Dr GRPO. Dr GRPO (Liu et al. 2025b) is cited for the removal of standard-deviation normalisation, described as removing a *difficulty* bias. The section opens by saying "we adopt improvements from DAPO (Yu et al., 2025) and Dr GRPO (Liu et al., 2025b)". The phrase "to avoid a length bias" is attached specifically to the token-level loss.
- **Overlong filtering:** footnote 34, p. 45: "We experimented with additional changes (e.g., overlong filtering), but did not find these gave consistent improvements." DAPO-style overlong filtering was **not** used.
- **KL:** "No KL loss" (p. 45): there is no KL penalty to the reference policy, which removes one restraint on drift, including drift in length.

**What this does and does not imply for us.** The objective removes the **per-sample length-normalisation bias**, the GRPO artefact that weakens the per-token penalty on long wrong answers (Liu et al. 2025, arXiv:2503.20783). It does not remove other length incentives:
- the RL-dynamics argument that negative rewards favour longer outputs (Fatemi et al. 2025, arXiv:2504.05185);
- any length preference of the LM judge on chat prompts (above);
- the absence of a KL anchor.

So the accurate statement is: **our observed lengthening occurred despite an explicit countermeasure against one known source of RL length bias (per-sample normalisation), but in a stage that also had LM-judge rewards and no KL penalty.** "Despite an explicit length-bias countermeasure" is true only in that narrow sense and should be written with the qualifier.

## (c) Did DPO apply length control, token cutoffs or a performance-vs-length analysis, and on which line?

**Instruct line only.** All length-control interventions in the report are described for Olmo 3 Instruct DPO:
- §5.3.1, p. 59 ("Controlling length bias"): "We seek to strike a balance by filtering the chat and multi-turn subsets of our preference data to limit the length difference between the chosen and rejected responses to 100 tokens."
- §5.3.3, p. 60: "We further sweep different length-control interventions by creating datasets with differing token cutoffs for length filtering. We select the best-performing checkpoint of each length budget and then select the final Olmo 3 Instruct-DPO checkpoint based on qualitative vibe tests and performance-vs-length analysis."
- §5.5, p. 61: "Applying length control during DPO substantially reduces the model's average generation length, allowing us to trade off some performance for improved conciseness and overall usability."
- §5.4, p. 60 (Instruct RL): "We then choose the final RL checkpoint based on final average performance, length analysis, and vibe test."

**Think DPO (§4.3, pp. 42–44; appendix A.6.2, p. 105; Table 48, p. 106) has no length-control intervention, token cutoff or performance-vs-length analysis.** Its construction is "we simply decode a chosen completion yc from one model (Qwen 3 32B, thinking) and a rejected completion yr from an overall weaker model (Qwen 3 0.6B, thinking)" (p. 43). Training: "We train all models for one epoch ..., sweeping learning rate and dataset size" plus "vibe-tests" (p. 44).

**The one length-related element shared by both lines** is the loss. Appendix A.6.2, p. 105: "we optimize the model policy πθ on a length-normalized DPO loss (Lambert et al., 2024)". The β/|y| normalisation divides each log-ratio by response length. Table 48 gives Think DPO a max sequence length of 16K for 7B.

**Consequence for C3.** Think DPO was trained on thinking-model pairs where, by the developers' own observation about delta-learning pairs, the stronger model tends to write longer, and it had no length control. So DPO lengthening is expected. Our C3 residue is the component pattern: lengthening on commonsense, sycophancy and calibration, but not on self-report. The report neither predicts nor explains that.

## Summary for the claims list

| Claim | Effect of the primary text |
|---|---|
| C1 | **Weakened.** The developers report that RL increases response length during Think RL training (training prompts, about 14–20K tokens). Residue: an evaluation-time, paired, self-report-specific measurement against the released DPO checkpoint, at a different scale. Also, the stage includes LM-judge rewards on chat, a candidate mechanism, so "verifiable rewards" must not be named as the cause. |
| C3 | **Weakened.** The developers state that delta-learning preference tuning carries a length bias that increases generation length, and Think DPO had no length control. Residue: lengthening on B, C and D but not A. |
| OlmoRL "length-bias countermeasure" | **Confirmed but narrow.** Batch-token normalisation "to avoid a length bias" (attributed to DAPO). No overlong filtering, no KL. It does not neutralise all length incentives. |
