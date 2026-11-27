# DPO - SFT: frozen against matched system prompt

Matched SFT file: `results\sprint\sft_matched_prompt\full_run_sft.json` (sha 6ff857587e04, env b25083c2e3494bcd, dry_run False). Swap record: `full_run_sft_system_prompt.json` present. Frozen DPO and RLVR files unchanged in both columns.

## Self-checks (frozen recomputation)

- PASS: table 3 DPO-SFT rows recomputed from the frozen files == committed table3_comparison.csv
- PASS: extreme-answer shares from the frozen files == registered 0.454 (SFT) / 0.203 (DPO)
- PASS: exploratory_dithering.md recomputed from the frozen files == committed file

Robust = same sign of the primary statistic and same verdict (registration amendment 1).

## Table 3, DPO-SFT rows (scripts/analyze.py)

| probe | form | frozen diff [CI] | frozen status | matched diff [CI] | matched status | robust |
|---|---|---|---|---|---|---|
| trait:approval | cc-only | 0.2500 [-0.4444, 1.0000] | inconclusive | 0.5000 [-0.0556, 1.1111] | inconclusive | yes |
| trait:assertiveness | cc-only | -0.2778 [-0.6667, 0.1667] | inconclusive | -0.3333 [-0.7778, 0.1111] | inconclusive | yes |
| trait:competitiveness | cc-only | 0.5000 [0.0000, 1.0000] | inconclusive | 0.6667 [-0.0556, 1.3889] | inconclusive | yes |
| trait:composure | cc-only | -0.4444 [-0.8889, 0.0000] | no meaningful change | -0.6111 [-1.1667, -0.1111] | inconclusive | yes |
| trait:confidence | cc-only | 0.3889 [-0.0556, 0.8333] | inconclusive | 0.1111 [-0.5556, 0.6667] | inconclusive | yes |
| trait:hedging | cc-only | 0.1667 [-0.4444, 0.6667] | no meaningful change | -0.2222 [-0.8333, 0.3889] | inconclusive | NO |
| trait:humility | cc-only | -0.1667 [-0.8889, 0.6681] | inconclusive | -0.1111 [-1.1111, 0.7778] | inconclusive | yes |
| trait:risk | cc-only | -0.1111 [-0.6667, 0.5556] | no meaningful change | -0.1111 [-0.7222, 0.5000] | no meaningful change | yes |
| sensitivity: hedging without original wording | cc-only | 0.2000 [-0.4667, 0.8000] | inconclusive | -0.0667 [-0.7333, 0.6000] | no meaningful change | NO |
| Sycophancy caving | iaf | 0.0000 [0.0000, 0.0000] | no meaningful change | 0.0000 [0.0000, 0.0000] | no meaningful change | yes |
| Sycophancy caving | cc | 0.0000 [0.0000, 0.0000] | no meaningful change | 0.0000 [0.0000, 0.0000] | no meaningful change | yes |
| Calibration accuracy | iaf | -0.0500 [-0.1125, 0.0000] | inconclusive | 0.0000 [-0.0625, 0.0625] | inconclusive | NO |
| Calibration accuracy | cc | -0.0256 [-0.0769, 0.0256] | inconclusive | 0.0000 [-0.0400, 0.0400] | inconclusive | NO |
| Calibration gap | cc-only | -0.0032 [-0.0513, 0.0500] | no meaningful change | -0.0253 [-0.0640, 0.0160] | inconclusive | yes |
| Calibration Brier | cc-only | 0.0135 [-0.0273, 0.0547] | inconclusive | -0.0035 [-0.0372, 0.0255] | inconclusive | NO |
| Commonsense accuracy | iaf | 0.0067 [-0.0267, 0.0433] | no meaningful change | 0.0033 [-0.0233, 0.0267] | no meaningful change | yes |
| Commonsense accuracy | cc | -0.0103 [-0.0377, 0.0205] | no meaningful change | 0.0035 [-0.0174, 0.0243] | no meaningful change | NO |
| Commonsense permutation consistency | cc | 0.0000 [-0.0479, 0.0479] | inconclusive | -0.0069 [-0.0556, 0.0347] | inconclusive | NO |
| MATH accuracy | iaf | -0.0400 [-0.1200, 0.0400] | no meaningful change | -0.0600 [-0.1400, 0.0000] | no meaningful change | yes |
| MATH accuracy | cc | 0.0000 [0.0000, 0.0000] | no meaningful change | -0.0222 [-0.0667, 0.0000] | no meaningful change | NO |
| sensitivity: MATH boxed_candidate rescoring | iaf | 0.0000 [0.0000, 0.0000] | no meaningful change | -0.0200 [-0.0600, 0.0000] | no meaningful change | NO |
| Reasoning length A | cc | -12.0000 [-66.8333, 63.1667] | inconclusive | 14.6667 [-65.3333, 105.3333] | inconclusive | NO |
| Reasoning length B | cc | 155.0000 [114.5000, 273.5000] | signal | 190.5000 [143.5000, 279.0000] | signal | yes |
| Reasoning length C | cc | 145.5000 [75.0000, 171.0000] | signal | 122.0000 [89.0000, 156.0000] | signal | yes |
| Reasoning length D | cc | 101.2500 [92.7500, 129.2500] | signal | 103.7500 [87.0000, 132.0000] | signal | yes |
| Reasoning length E | cc | 55.5000 [-21.0000, 130.0000] | inconclusive | 81.0000 [12.0000, 152.0000] | signal | NO |
| Reasoning length A-D | cc | 103.2500 [91.5000, 128.8333] | signal | 114.3333 [97.0000, 132.0000] | signal | yes |
| Reasoning length H4 (A minus D) | cc | -113.2500 [-180.8354, -40.7500] | signal | -89.0833 [-170.0000, 5.0833] | inconclusive | NO |

## Claims

| result | frozen | matched | robust |
|---|---|---|---|
| H3 SFT ~ DPO overall, A-D pooled (no signal) | not replicated | not replicated | yes |
| C3 DPO lengthens B, C, D but not A | holds | holds | yes |
| C8 DPO more stable across seeds than SFT (addendum 4b: mean floor DPO-SFT, nested 95% CI) | -0.296 [-0.512, -0.105], holds | -0.263 [-0.480, -0.080], holds | yes |
| DPO-SFT extreme-answer drop (A answers in {1, 5}) | pooled SFT 0.454 (n 141), DPO 0.203; per-wording DPO-SFT -0.247 [-0.340, -0.149], p 0.0000 | pooled SFT 0.455 (n 143), DPO 0.203; per-wording DPO-SFT -0.247 [-0.351, -0.135], p 0.0002 | yes |

## Exploratory rows comparing with SFT (results/analysis/exploratory_dithering.py; C5 = addenda 1-2, C8 = addendum 4)

| section | quantity | comparison | frozen row | matched row | robust |
|---|---|---|---|---|---|
| Measure 1: dithering | distinct candidates per generation | DPO-SFT | 48 ; 0.000 [0.000, 0.333] ; 0.108 [-0.108, 0.316] ; median ; no ; 0.3005 (normal_tie_corrected) | 48 ; 0.333 [0.000, 0.667] ; 0.236 [0.010, 0.458] ; median ; no ; 0.0367 (normal_tie_corrected) | NO |
| Measure 1: dithering | switches per generation | DPO-SFT | 48 ; -2.000 [-3.333, -1.000] ; -3.618 [-5.417, -2.069] ; median ; yes ; 0.0000 (normal_tie_corrected) | 48 ; -1.000 [-2.833, -0.167] ; -2.462 [-4.146, -1.010] ; median ; yes ; 0.0045 (normal_tie_corrected) | yes |
| Measure 1: dithering | candidate mentions per generation (additional) | DPO-SFT | 48 ; -3.667 [-5.667, -1.667] ; -6.392 [-9.191, -3.990] ; median ; yes ; 0.0000 (normal_tie_corrected) | 48 ; -2.000 [-5.333, -1.000] ; -4.562 [-6.951, -2.542] ; median ; yes ; 0.0001 (normal_tie_corrected) | yes |
| Measure 1: dithering | final extracted candidate = parsed ANSWER (share, generations with >=1 | DPO-SFT | 48 ; 0.000 [0.000, 0.000] ; -0.035 [-0.097, 0.021] ; mean ; no ; 0.2442 (normal_tie_corrected) | 48 ; 0.000 [0.000, 0.000] ; -0.063 [-0.111, -0.021] ; mean ; yes ; 0.0139 (normal_tie_corrected) | NO |
| Measure 2: repetition (lower = more repetitive) | whole think block: compression ratio | DPO-SFT | 48 ; 0.049 [0.034, 0.058] ; 0.051 [0.039, 0.063] ; median ; yes ; 0.0000 (normal) | 48 ; 0.049 [0.029, 0.058] ; 0.047 [0.036, 0.058] ; median ; yes ; 0.0000 (normal) | yes |
| Measure 2: repetition (lower = more repetitive) | whole think block: unique 6-gram share | DPO-SFT | 48 ; 0.006 [0.003, 0.009] ; 0.011 [0.007, 0.016] ; median ; yes ; 0.0000 (normal) | 48 ; 0.006 [0.002, 0.011] ; 0.010 [0.006, 0.014] ; median ; yes ; 0.0000 (normal) | yes |
| Measure 2: repetition (lower = more repetitive) | last 3000 chars: compression ratio | DPO-SFT | 48 ; 0.051 [0.038, 0.055] ; 0.048 [0.038, 0.059] ; median ; yes ; 0.0000 (normal) | 48 ; 0.045 [0.030, 0.058] ; 0.045 [0.036, 0.055] ; median ; yes ; 0.0000 (normal) | yes |
| Measure 2: repetition (lower = more repetitive) | last 400 words: unique 6-gram share | DPO-SFT | 48 ; 0.005 [0.003, 0.008] ; 0.008 [0.005, 0.011] ; median ; yes ; 0.0000 (normal_tie_corrected) | 48 ; 0.004 [0.002, 0.008] ; 0.007 [0.005, 0.009] ; median ; yes ; 0.0000 (normal_tie_corrected) | yes |
| Measure 3: two-instruction compliance | (a) justification sentences | DPO-SFT | 48 ; 1.000 [0.667, 1.000] ; 0.976 [0.809, 1.149] ; median ; yes ; 0.0000 (normal_tie_corrected) | 48 ; 0.833 [0.667, 1.333] ; 0.882 [0.719, 1.049] ; median ; yes ; 0.0000 (normal_tie_corrected) | yes |
| Measure 3: two-instruction compliance | (b) reasoning sentences | DPO-SFT | 48 ; -8.167 [-13.167, -2.833] ; -12.924 [-19.400, -7.222] ; median ; yes ; 0.0001 (normal_tie_corrected) | 48 ; -2.833 [-13.000, 1.000] ; -8.951 [-14.174, -4.201] ; median ; no ; 0.0048 (normal_tie_corrected) | NO |
| Measure 3: two-instruction compliance | (c) format compliance (share) | DPO-SFT | 48 ; 0.000 [0.000, 0.333] ; 0.194 [0.125, 0.264] ; mean ; yes ; 0.0000 (normal_tie_corrected) | 48 ; 0.333 [0.333, 0.333] ; 0.368 [0.292, 0.444] ; mean ; yes ; 0.0000 (normal_tie_corrected) | yes |
| Measure 3: two-instruction compliance | (additional) format ok and exactly one justification sentence (share) | DPO-SFT | 48 ; -0.333 [-0.667, -0.333] ; -0.424 [-0.521, -0.319] ; mean ; yes ; 0.0000 (normal_tie_corrected) | 48 ; -0.333 [-0.333, 0.000] ; -0.188 [-0.285, -0.090] ; mean ; yes ; 0.0014 (normal_tie_corrected) | yes |
| Measure 3: two-instruction compliance | (additional) reasoning <= 6 sentences (share) | DPO-SFT | 48 ; 0.000 [0.000, 0.000] ; 0.000 [0.000, 0.000] ; mean ; no ; 1.0000 (all_zero) | 48 ; 0.000 [0.000, 0.000] ; 0.000 [0.000, 0.000] ; mean ; no ; 1.0000 (all_zero) | yes |
| Addendum (Oct 25, requested after the results above were see | switches per 1000 generated tokens | DPO-SFT | 48 ; -3.599 [-4.892, -1.381] ; -3.026 [-4.187, -1.862] ; yes ; 0.0000 (normal) | 48 ; -1.530 [-3.828, -0.787] ; -2.227 [-3.623, -0.876] ; yes ; 0.0035 (normal) | yes |
| Addendum (Oct 25, requested after the results above were see | candidate mentions per 1000 generated tokens | DPO-SFT | 48 ; -7.505 [-8.341, -4.405] ; -6.298 [-7.601, -4.983] ; yes ; 0.0000 (normal) | 48 ; -5.285 [-7.234, -3.820] ; -5.182 [-6.760, -3.683] ; yes ; 0.0000 (normal) | yes |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | switches per 1000 think-block proxy tokens | DPO-SFT | 48 ; -2.951 [-4.698, -0.960] ; -2.762 [-4.018, -1.551] ; yes ; 0.0002 (normal) | 48 ; -1.656 [-3.328, 0.199] ; -1.862 [-3.284, -0.507] ; no ; 0.0188 (normal) | NO |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | switches per 1000 think-block proxy tokens | RLVR-SFT | 48 ; -0.398 [-1.885, 1.627] ; -0.522 [-1.936, 0.827] ; no ; 0.5182 (normal) | 48 ; -0.033 [-1.147, 2.004] ; 0.377 [-1.310, 2.038] ; no ; 0.7505 (normal) | yes |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | candidate mentions per 1000 think-block proxy tokens | DPO-SFT | 48 ; -6.513 [-8.273, -4.085] ; -5.950 [-7.388, -4.514] ; yes ; 0.0000 (normal) | 48 ; -4.995 [-6.750, -3.344] ; -4.686 [-6.305, -3.085] ; yes ; 0.0000 (normal) | yes |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | candidate mentions per 1000 think-block proxy tokens | RLVR-SFT | 48 ; -1.731 [-3.261, 0.576] ; -1.235 [-2.755, 0.370] ; no ; 0.0664 (normal) | 48 ; 0.071 [-2.126, 2.065] ; 0.029 [-1.746, 1.817] ; no ; 0.9918 (normal) | NO |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | switches per 1000 think-block characters | DPO-SFT | 48 ; -0.949 [-1.256, -0.432] ; -0.842 [-1.134, -0.547] ; yes ; 0.0000 (normal) | 48 ; -0.434 [-0.933, -0.223] ; -0.591 [-0.928, -0.263] ; yes ; 0.0020 (normal) | yes |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | switches per 1000 think-block characters | RLVR-SFT | 48 ; -0.241 [-0.419, 0.320] ; -0.183 [-0.506, 0.145] ; no ; 0.3506 (normal) | 48 ; 0.020 [-0.340, 0.503] ; 0.068 [-0.326, 0.476] ; no ; 0.8056 (normal) | NO |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | candidate mentions per 1000 think-block characters | DPO-SFT | 48 ; -2.003 [-2.291, -1.137] ; -1.699 [-2.057, -1.345] ; yes ; 0.0000 (normal) | 48 ; -1.311 [-1.767, -1.056] ; -1.344 [-1.736, -0.969] ; yes ; 0.0000 (normal) | yes |
| Addendum 2 (Oct 25, requested after addendum 1): sensitivity | candidate mentions per 1000 think-block characters | RLVR-SFT | 48 ; -0.429 [-0.835, 0.064] ; -0.376 [-0.760, 0.024] ; no ; 0.0478 (normal) | 48 ; 0.025 [-0.472, 0.527] ; -0.021 [-0.467, 0.423] ; no ; 0.8939 (normal) | NO |
| (a) Simple: traits as the paired units (n = 8), floors treat |  | DPO-SFT | -0.296 [-0.474, -0.123] ; -0.309 [-0.491, -0.056] ; yes ; 0.0234 (exact, 8) ; DPO lower on 7, SFT lower on 1, tie 0 | -0.263 [-0.442, -0.057] ; -0.332 [-0.471, -0.056] ; yes ; 0.0547 (exact, 8) ; DPO lower on 7, SFT lower on 1, tie 0 | yes |
| (a) Simple: traits as the paired units (n = 8), floors treat |  | RLVR-SFT | -0.125 [-0.270, -0.006] ; -0.111 [-0.149, -0.051] ; yes ; 0.1484 (exact, 8) ; RLVR lower on 7, SFT lower on 1, tie 0 | -0.093 [-0.255, +0.094] ; -0.140 [-0.312, +0.176] ; no ; 0.3828 (exact, 8) ; RLVR lower on 6, SFT lower on 2, tie 0 | NO |
| (b) Propagated: nested bootstrap of the floors (primary) |  | DPO-SFT | -0.296 ; [-0.512, -0.105] ; yes ; 0.0026 ; 0 / 2 of 8 ; [-0.408, -0.083] | -0.263 ; [-0.480, -0.080] ; yes ; 0.0064 ; 0 / 0 of 8 ; [-0.387, -0.064] | yes |
| (b) Propagated: nested bootstrap of the floors (primary) |  | RLVR-SFT | -0.125 ; [-0.373, +0.082] ; no ; 0.2158 ; 0 / 0 of 8 ; [-0.302, +0.067] | -0.093 ; [-0.338, +0.100] ; no ; 0.2994 ; 0 / 0 of 8 ; [-0.273, +0.079] | yes |

