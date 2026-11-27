# Revision status (branch `eacl_revision_1`, registration amendment 18)

Each line records what ran, what changed compared with the submitted results, and anything that contradicts a claim in the paper.

| date | job | what ran | change vs submitted | contradicts a paper claim? |
|---|---|---|---|---|
| 2026-11-27 | Step 0 | Amendment 18 registered and approved before any F-job ran (`3e7c363`). | none (registration only) | no |
| 2026-11-27 | F0a | **Done.** `test_b6_items_grading_and_syc_flow` failed because amendment 16's `CUT` made `syc` return 0 with no shard. Fixed by `CUT = set()` (amendment 18, F4), with a regression assertion (`fb54232`). Confirmed on Linux with all pins exact: fails before the fix, passes at it. The full `tests/sprint` run there had 2 failures from the host RAM preflight (15.7 GB < 16 GB), not from code. Unit tests now bypass the preflight through a test-only fixture; real jobs are unchanged (`known_issues.md`). | none (B6 (ii) never ran; no stored result depends on this code) | no |
