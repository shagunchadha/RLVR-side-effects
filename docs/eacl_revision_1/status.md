# Revision status (branch `eacl_revision_1`, registration amendment 18)

Each line records what ran, what changed compared with the submitted results, and anything that contradicts a claim in the paper.

| date | job | what ran | change vs submitted | contradicts a paper claim? |
|---|---|---|---|---|
| 2026-11-27 | Step 0 | Amendment 18 registered and approved before any F-job ran (`3e7c363`). | none (registration only) | no |
| 2026-11-27 | F0a | **In progress.** Cause of the failing `test_b6_items_grading_and_syc_flow` found: amendment 16's `CUT` made `syc` return 0 with no shard (see `known_issues.md`). Fix committed (`CUT = set()`, per amendment 18 F4) with a regression assertion. Linux confirmation in the pinned venv (`exp/launch/F0a_linux_check.sh`) is pending. | none (no stored result depends on B6 (ii), which never ran) | no |
