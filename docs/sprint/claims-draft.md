# Claims draft (for the writeup)

Draft of Nov 4, 2026, for the PI to edit. Built from `gate2.md` (verdicts, CIs) and `disclosure-ledger.md`. Comparison throughout: Olmo-3-7B-Think RLVR vs its DPO parent (SFT where run). **A** = indeterminate self-report items (48 Likert wordings); **D** = determinate multiple-choice items. A **result** is a 95% CI excluding 0 in the registered direction; no multiplicity correction across jobs.

## Claims we can make

**Strong** (registered, replicated across jobs)

1. **RLVR answers self-report items more extremely than DPO.** B1-H1 +0.181 [+0.131, +0.233] (Holm p 3.5e−7); replicated on a validated instrument, B8 IPIP-NEO-120 +0.095 [+0.055, +0.138].
2. **RLVR deliberates longer, specifically on indeterminate items.** B2 indeterminacy effect +626 tokens [+500, +768]; B3-H2 the A − D excess survives with no instruction, +1149 [+942, +1353]; B5 length rises with RL step, more on A than D (A − D slope +0.165 [+0.126, +0.203]).
3. **The extra length is the policy, not the text.** F3: DPO wraps up RLVR's half-finished traces quickly, RLVR continues DPO's (text share −0.09 [−0.19, 0.01]). M3: RLVR settles its answer ~490 tokens *later*; only 0.36 of its extra tokens come after settling, so it is not padding.
4. **Both effects grow with RL training.** B5: A length ρ +0.55, A P(extreme) ρ +0.57 (sampled) / +0.60 (readout) across 9 checkpoints.

**Moderate** (correlational or exploratory)

5. **Extremity builds up during RLVR's deliberation.** M3: P(extreme) climbs from +0.04 over DPO in the first tenth of the trace to +0.15 in the last, and grows more after the wrap-up onset (+0.043 [+0.023, +0.065]). F3b: the continuing RLVR model adds extremity.
6. **Part of the shift is already present without thinking.** B1-H3 no-think prior +0.065 [+0.039, +0.093], about a third of the full effect (exploratory).
7. **RLVR is not worse across the board.** It is not more overconfident (B6 (iii): gap −0.022 [−0.047, +0.001]; both checkpoints state ≈ 90% confidence at ≈ 65% accuracy), and it refuses unanswerable questions *more* (B6 (iv): +0.108 [+0.068, +0.149]) without more false refusals, with higher answerable accuracy (+0.076 [+0.028, +0.132]).

**Descriptive only**

8. **RLVR's self-report contradicts its behaviour on hedging.** It hedges more in its think block (+0.255 per 100 words [+0.129, +0.380]) yet rates itself lower on hedging (3.26 vs 3.47). Three checkpoints: an ordering, not a test (B7).

## Claims we should not make

- **"Deliberation causes / drives the extreme shift."** The one causal test is against it: ablating the backtracking direction cuts RLVR's A deliberation from 1175 to 506 tokens, and P(extreme) does not fall (M5, +0.13 [−0.12, +0.37]). Every offset raises extremity whatever its sign. Say instead: *the shift shows up in the deliberation; we did not show the deliberation causes it.*
- **"RLVR makes a better reasoner, at the cost of side effects."** The positive control failed: RLVR is not more accurate on MATH-500 L5 + AIME (−0.022 [−0.056, +0.009]). The contrasts compare two checkpoints of similar maths accuracy.
- **"RLVR changes the weights less / more surgically."** M1-H1 failed the opposite way: RLVR's update is larger in 226/226 tensors (median ratio 2.03).
- **"RLVR makes the model more acquiescent / less acquiescent."** B1-H2 unclear (Holm p 0.055), B8 no result.
- **"The general-domain RL data causes it."** B4 (RL-Zero by domain) is the opposite: General shows the *least* extreme shift; Code the most.
- **Any representation-level claim from M4 or M6.** M4's persona directions are unstable across seeds (median cos 0.48). M6 has no intervals, and its point estimates show the same A/D gap in SFT ↔ DPO as in DPO ↔ RLVR, so nothing RLVR-specific.
- **Sycophancy.** Not tested (B6 (ii) cut).

## Disclosures the writeup must carry

- **Post-freeze runs, second account:** B3b (part), B6, B7 and M6 ran after the Nov 2 freeze on a second molab account, with designs unchanged and every job resumed from saved progress (ledger, Nov 3).
- **Cuts and moves:** B6 (ii) sycophancy cut for time (amendment 16); B4's Mix arm cut; B6 parts split across notebooks (amendments 16, 17).
- **M6:** the registered intervals were lost to a shard-writer bug; the job is reported from point estimates only (ledger, Nov 4).
- **M5:** answers re-parsed in analysis after a decoding bug (ledger, Oct 29). One seed, 24 A + 25 D items, a single layer.
- **SFT system prompt:** the frozen run gave SFT a different default system prompt (`disclosure-sft-system-prompt.md`); sprint SFT arms use the DPO prompt where stated.
- **Statistics:** no multiplicity correction across jobs; Holm only where registered.

## Suggested headline

> After RLVR, Olmo-3-7B-Think answers self-report questions more extremely and deliberates longer on questions without a fixed answer; both grow with RL training, and the extremity builds up during the deliberation. But cutting the deliberation does not remove the extremity, and RLVR is no better at hard maths than its DPO parent, no more overconfident, and more willing to say a question is unanswerable.
