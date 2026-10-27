# M1: weight-difference spectra, RLVR − DPO against DPO − SFT

Run Oct 27, 2026 on molab N6 (one process, 355 tensors, 13.5 min; `partial/M1.log`). Code `exp/jobs/M1_weightdiff.py` at git `8087839`; registration `docs/sprint/registration-2026-10-27.md` (M1).

## The registered prediction failed

**M1-H1 predicted that the RLVR update would be smaller than the DPO update. It is larger.**

- **Primary statistic:** in 0 of 226 2-D tensors is rel_dw(RLVR−DPO) smaller than rel_dw(DPO−SFT). The exact sign test against 0.5 rejects the hypothesis in the opposite direction.
- **Size:** the median ratio is 2.03 (median log10 ratio +0.307). RLVR moved the weights about twice as far, relative to their norm, as DPO did.
- **Where:** almost everywhere, most of all in the MLP gate and up projections of layers 29–31 (about 3–5×), where the DPO update was unusually small.

This is reported as a failed prediction. It is not reinterpreted.

## Caveat (to be checked by M1b, not a rescue of the prediction)

Both updates are about 0.1–0.2% of the weight norm (rel_dw 0.0003–0.0018). The bf16 spacing is about 0.4–0.8% of a value, so many elements either did not change or changed by one ulp. The effective ranks are close to the maximum (about 3,500 for the MLP blocks), which is what noise-like updates look like. Part of both dW may therefore be rounding, not learned structure. M1b (registration amendment 2) measures:
- the share of elements changed and the change sizes in ulps;
- the singular directions that beat a shuffle null;
- cos(dW, W), which matters for the QK-normed q_proj and k_proj.

It also checks whether the k_proj concentration among the largest RLVR updates survives re-ranking.

## Files

- `weightdiff.csv`: per tensor × comparison, w_norm, dw_norm, rel_dw, eff_rank, stable_rank, num_rank, sv_1..32.
- `subspace_angles.csv`: principal-angle cosines between the top-8 and top-32 singular subspaces of the two updates.
- `heatmap_M1.png`: layer × module heatmaps of rel_dw for both comparisons, effective rank, and mean cos² of the top-32 left subspaces.
- `heatmap_M1_ratio.png`: log10 of rel_dw(RLVR−DPO) / rel_dw(DPO−SFT).
- `provenance.json`: config, git SHA, package versions.
