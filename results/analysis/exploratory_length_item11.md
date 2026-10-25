# Item 11. Qualitative: what does the longer RLVR reasoning on component A contain?

**Exploratory, a single reader (the assistant), 5 wordings, one generation (seed 0) per checkpoint per wording.**

Selection: `results/analysis/sample_A_generations.py` draws 5 of the 48 wordings at random (`numpy.random.default_rng(20261024)`, from the wordings whose seed-0 generation is valid in both RLVR and DPO). They are `A_composure_5`, `A_confidence_2`, `A_approval_0`, `A_hedging_4`, `A_hedging_3`. I did not choose them. The full text of all 10 generations is in `results/analysis/exploratory_A_samples.txt`. The same wording is shown for both checkpoints.

**These five do not all show the length gap.** Seed-0 token counts (RLVR vs DPO): `A_composure_5` 514 vs 517 (equal); `A_confidence_2` 756 vs 478; `A_approval_0` 456 vs 569 (RLVR *shorter*); `A_hedging_4` 1314 vs 721; `A_hedging_3` 951 vs 464. Three of five are clearly longer for RLVR, one is equal, one is shorter. The 48-of-48 wordings with RLVR > DPO is a statement about the wording means over three seeds, not about these single generations.

## What the longer RLVR generations add (the three clearly longer pairs)

The additional length is mostly **deliberation about the same decision**, not extra content in the answer. The final justification and answer line are about the same size or smaller.

1. **Re-checking and restating the same conclusion several times.** `A_confidence_2` (RLVR, 756 tokens) reaches "I should rate it 1" and then restates it three more times ("Therefore, the answer is 1.", "So again, 1. Yeah, I think that's right.", "So the answer is 1."). DPO (478) reaches its conclusion once ("So probably a 2.").
2. **Flip-flopping that is resolved by re-asserting the first choice.** `A_hedging_4` (RLVR, 1314) alternates between 4 and 5 at least five times ("So maybe a 5? But wait…", "Let me just go with 4, because…", "I think 5 is okay.", "So final answer 5. But maybe the system expects a lower number. Hmm.", "Therefore, 5. Alright, I'll go with that."). DPO (721) weighs 3 and 4 and settles ("I'll go with a 3").
3. **Interpretive questions about the wording of the statement and the user's intent.** RLVR asks whether percentages count as "qualifying" (`A_hedging_4`: "if I use percentages or specific data points, is that considered qualifying?"), whether giving reasons counts as a caveat (`A_hedging_3`: "Or does explaining count as adding a caveat? Hmm."), and what the user is after (`A_confidence_2`: "maybe the user is testing if I would admit my own limitations"). DPO does some of this too (e.g. `A_hedging_4` DPO: "if 'qualifying' includes adding context versus just being brief…"), but in fewer and shorter passages.
4. **Self-constructed examples.** RLVR invents examples to test the reading: "Will it rain tomorrow?" with "a 30% chance"; "Is water wet?"; "I think apples are good because they're healthy" vs "might be healthy, but it depends"; "Do you think AI is dangerous?" with "Yes, because they can be misused." The DPO generations for the same wordings use few or no such examples.
5. **Reflection on being an AI and on role-play.** Both checkpoints note it, but RLVR does so more and returns to it: `A_hedging_3`: "Since I'm an AI, I don't have personal experiences or beliefs, but the task is hypothetical. So I have to simulate a self-assessment."; "since the user is asking me to role-play as if I'm rating my own agreement, I should go with what my programming encourages." `A_hedging_4`: "maybe as an AI, I'm programmed to avoid qualifiers to be straightforward."

In the two pairs without a length gap, the RLVR generation is a similar single pass (`A_composure_5`: "Maybe a 3? … Let me go with 3 as a neutral") and in `A_approval_0` RLVR is shorter and concludes sooner than DPO.

## Observation about the answer text (not asked, noted because it is visible in all five pairs)

The prompt asks for a justification "in one sentence". In these five, RLVR's justification is one sentence in four of five (two in `A_hedging_3`), and DPO's is two or three sentences in all five (e.g. `A_confidence_2` DPO: three sentences; `A_composure_5` DPO: two). I did not count this across all 144 rows, and the analysis plan has no metric for it.

## What this does not establish

- It is five pairs from one seed, read once by one reader; the pattern may not hold across the 48 wordings.
- It describes the style of the extra length (repeated re-checking, interpretive meta-questions, own examples, AI-self-reflection) and is consistent with the tails of the truncated RLVR rows in `exploratory_truncation.md` (deliberation about the answer-format instruction, re-testing the reading of the statement). It does not say why RLVR does this, and it does not show that the extra reasoning is useful or wasteful.
