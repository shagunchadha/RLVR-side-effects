# Disclosure ledger (sprint)

This file records every departure from the storage rules and every repair of a stored result, with its date, its reason and who approved it. Related disclosures:
- the SFT system-prompt confound: `disclosure-sft-system-prompt.md`;
- design changes: the dated amendments in `registration-2026-10-27.md`.

| date (UTC) | what | why | approval | verification | location |
|---|---|---|---|---|---|
| 2026-10-29 | Rebuilt and re-uploaded `runs/M2/pair_rlvr_dpo/b35466982f5e/shard_000.parquet` in the HF dataset `Shagsc/rlvr-se-runs`, at the same path, as a new commit. HF history keeps the torn version. This is the only overwrite of a shard in the sprint. | The copy uploaded on Oct 29 was torn: 400,000 bytes and no parquet footer. It came from a pull that was cut while the old N6 sandbox was being reclaimed (Oct 28). | PI, Oct 29 (amendment 5) | Rebuilt from the complete progress file `M2/pair_rlvr_dpo/progress_shard0of1.csv`: 1842 rows, all with config hash b35466982f5e, none skipped, in the job's row order. Row-level equality between the rebuilt shard and the progress file: OK. Independent check against the `--kl-detail` run (same models, same traces, run separately on Oct 29): `mean_kl_think` max \|diff\| = 0 over 1842 traces. Shard metadata carries a `repair` field and `git_sha = unknown (shard rebuilt from progress)`. | HF commit `9ab6117293bd` (its message names this ledger) |

No number reported before the repair depended on the torn file: every M2 analysis reads the progress files.
