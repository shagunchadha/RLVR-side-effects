# Gate 2: verdicts per registered hypothesis, and the headline

Written Oct 29, 2026 (about 21:30 IST), from the committed summaries in `results/sprint/<job>/summary.md`, Gate 1 (`gate1-2026-10-29.md`) and `results/sprint/M1*/`. A **result** is a 95% CI excluding 0 (or 1 for ratios) in the registered direction. There is no multiplicity correction across jobs (stated limitation), and Holm is applied only where registered. Still running or not yet started: B6, B7, M6, B3b (SFT arm of B3). B4 updated Oct 31 after General finished.

## Verdicts

**B1** (`results/sprint/B1/b1_summary.md`, Gate 1 a)

| ID | verdict | estimate |
|---|---|---|
| B1-H1 RLVR more extreme than DPO | **result** | +0.181 [+0.131, +0.233], Holm p 3.5e−7 |
| B1-H2 RLVR acquiesces less | **unclear** | −0.295 [−0.529, −0.058], 7/8 traits; Holm p 0.055 |
| B1-H3 extreme difference in the no-think prior | **result** (exploratory) | +0.065 [+0.039, +0.093] |
| B1-H4 reasoning moves the readout more for RLVR | **result** (exploratory) | +0.217 [+0.152, +0.280] |
| B1-H5 sampled = readout argmax ≥ 0.8 | **holds** | ≥ 0.96 |
| B1-H6 | descriptive | |

**B2** (`results/sprint/B2/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| B2-H indeterminacy main effect | **result** | +626 tokens [+500, +768] |
| B2-H interaction small | **not established** | −263 [−481, +49]; bound ±156 |

**B3a**, RLVR and DPO only; SFT (B3b) pending (`results/sprint/B3a/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| B3-H1 cap shortens thinking < 20% | **failed** for RLVR and DPO, A and D | ratios 0.63–0.81 |
| B3-H2 A-vs-D excess survives bare/no cap | **result** | +1149 tokens [+942, +1353] |
| B3-H3 RLVR fewer justification sentences | **result** | −0.94 [−1.03, −0.85] |
| B3-H4, H5 | exploratory / descriptive | |

**B4**, base and Math/Code/IF/General finished, Mix cut (`results/sprint/B4/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| B4-H extreme shift most in General | **opposite** | General has the lowest A P(extreme), 0.137, against base 0.278; Code is highest at 0.763 |
| B4-H lengthening most in General | **not supported** | General's A/D ratio vs base is the lowest, 1.72 [1.08, 2.97] (Code 2.57); IF and Math lengths degenerate, so this rests on General vs Code |

**B5** (`results/sprint/B5/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| B5-H A length rises with step | **result** | ρ +0.55 [+0.52, +0.62] |
| B5-H A P(extreme) rises with step | **result** | ρ +0.57 sampled, +0.60 readout |
| B5-H D length rises less | **result** | A − D slope +0.165 [+0.126, +0.203] |

**B8** (`results/sprint/B8/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| B8-H extreme RLVR > DPO (IPIP-NEO-120) | **result** | +0.095 [+0.055, +0.138] |
| B8-H acquiescence RLVR < DPO | **no result** | −0.193 [−0.397, +0.011], 21 facets |

**M1** (`results/sprint/M1/`)

| ID | verdict | estimate |
|---|---|---|
| M1-H1 RLVR update smaller than DPO's | **failed (opposite)** | 0/226 tensors; median ratio 2.03 |
| M1-H2 updates not orthogonal | **result** | 226/226 tensors above the random baseline (top-32 left; median 6.8× baseline) |
| M1-H3, H4 | descriptive / exploratory | |

**M2** (`results/sprint/M2/m2_summary.md`)

| ID | verdict | estimate |
|---|---|---|
| M2-H1 RLVR less willing to stop on the same text | **result, negligible size** | −8.4e−5 |
| M2-H2 stopping deficit larger on A | **no result** | −3.0e−6 [−1.3e−4, +1.2e−4] |
| M2-H3 RLVR more switch-token mass | **result** | +0.031 [+0.027, +0.035] |
| M2-H4 KL larger on A than D | **result** (exploratory) | +0.068 [+0.059, +0.076] |

**M3** (`results/sprint/M3/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| M3-H RLVR settles similar/earlier; > 50% of extra tokens after settle | **failed (opposite)** | settles +488 tokens later; 0.36 [0.30, 0.42] of extra tokens after settle |
| M3 added test (amendment 5): extremity grows after the wrap-up onset more for RLVR | **result** | +0.043 [+0.023, +0.065]; without "So" +0.080 [+0.042, +0.120] |
| M3 probes at prompt end | descriptive | above baseline only for RLVR (0.39 vs 0.19) |

**M4** (`results/sprint/M4/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| M4-H directions stable (cos > 0.8) | **failed** | 2/40 at L19/23, median 0.48 |
| M4-H RLVR shifts a projection beyond seed SD | **result, but hard to interpret given unstable directions** | lower on sycophantic/approval/confidence/composure, higher on hedging |

**M5**, exploratory (`results/sprint/M5/summary.md`)

| ID | verdict | estimate |
|---|---|---|
| M5-H +offset lengthens DPO A more than D (α = 1) | **no result** | A/D ratio 1.15 [0.83, 1.61] |
| M5-H −offset shortens RLVR A | **no result** | ratio 0.97 [0.61, 1.49]; −30 tokens |
| M5-H ablation shortens A | **result** | −669 tokens [−1002, −352] |
| M5-H ablation changes the answer in < 10% | **failed as registered** | 0.21; but the seed-to-seed rate is 0.19 |
| M5 extremity under ablation (amendment 5) | **not reduced** | +0.13 [−0.12, +0.37] |

**Gate 1 exploratory** (`gate1-2026-10-29.md`)

| ID | verdict |
|---|---|
| F1 | earlier wrap-up only via "So" |
| F2 | unclear |
| F3 | **policy**: text share −0.09 [−0.19, 0.01] |
| F3b | extremity follows **both** the continuing model and the text |

**Pending:** B6 (positive control on N6), B7, M6, B3b.

## Does the headline hold?

> "RLVR keeps deliberating where DPO would stop (F3), and that deliberation drives the extreme-answer shift (F3b, M3, M5)."

**First half: holds.**
- F3 puts the extra length in RLVR's policy, not in the text: DPO wraps up RLVR's half-finished reasoning quickly, and RLVR continues DPO's.
- M3 adds that RLVR does not settle early and pad: it settles about 490 tokens later, and about two thirds of its extra tokens come before its answer settles.
- B2 and B3-H2 show the extra deliberation is specific to indeterminate items and survives without any instruction.
- B5 shows it grows with RL step much more on A than on D.

**Second half: correlational support, causal test against.**
- **For** (accumulation): in M3, P(extreme) climbs through RLVR's trace from near DPO's level (+0.04 in the first tenth) to +0.15 in the last tenth, and grows more after the wrap-up onset.
- **For** (F3b): the continuing RLVR model adds extremity, and it accumulates in the text.
- **For** (B5): length and extremity rise together after step 425.
- **Against** (M5): ablating the backtracking direction cuts RLVR's A deliberation to DPO's length (1175 → 506 tokens) without changing answers beyond seed noise, and **P(extreme) does not fall** (+0.13, CI includes 0).
- **Against** (M5 offset): every offset condition raises P(extreme) whatever its sign, which looks like a generic perturbation effect rather than a deliberation → extremity pathway.
- **Against** (B1-H3): part of the shift is already in the no-think prior (+0.065, about a third of B1-H1).

**Verdict: the headline as worded does not hold.** "Drives" is causal, and the one causal test (M5) dissociates deliberation length from extremity. What the data support:

> **RLVR keeps deliberating where DPO would stop (F3, M3, B2, B5), and its extreme answers build up during that deliberation (M3, F3b, B5). But removing much of the deliberation does not remove the extremity (M5), so the deliberation is where the shift shows, not shown to be what causes it.**

**Caveats on M5:** one seed per condition, 24 A + 25 D items, interventions at a single layer (19), and the answer re-parse is disclosed in the ledger. M6 (CKA on A vs D) and B6 (the positive control) are still to come. Neither tests the causal link directly.
