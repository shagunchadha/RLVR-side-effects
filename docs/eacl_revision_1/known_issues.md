# Known issues (branch `eacl_revision_1`)

## `tests/sprint/test_jobs.py::test_b6_items_grading_and_syc_flow` fails

**Status:** open. To be fixed first in the next task, where the sycophancy test (B6 (ii)) is rerun.

**Observed (Nov 27, 2026).** This is the only failure in `tests/sprint`: 1 failed, 80 passed. It fails identically on `eacl_revision_1` before and after the anonymity commit, so that commit did not cause it.

```
tests\sprint\test_jobs.py:478
exp\store.py: in read -> _get_bytes
FileNotFoundError: ...\store\runs\B6\syc__rlvr\7b9bc789aef3\shard_000.parquet
```

**What we know**
- The test runs the DPO pre-pass (`part="sycpre"`), rewrites `B6/syc_selection.json`, then runs `part="syc"` on RLVR. It then reads shard 0 of `B6/syc__rlvr` under the config it rebuilds.
- The pre-pass works: it writes `runs/B6/sycpre__dpo/<hash>/shard_000.parquet` and `config.json`.
- The `syc` run returns 0 but writes **no** `runs/B6/syc__rlvr/` directory at all, under any config hash. So this is not a config-hash mismatch between the test and the job. The `syc` part appears to finish without producing output. One possibility is that the selection it reads back is empty or filtered out; this has not been checked.
- B6 (ii) sycophancy was cut from the study (amendment 16), so no reported result depends on this code path.

**Environment of the observation:** Windows 10, a CPU-only throwaway venv (Python 3.13, torch 2.13.0+cpu, transformers 5.17.0, pyarrow 26.0.0, pytest 9.1.1), with `HF_REPO` unset. Note that numpy was 2.5.1 from the system site-packages, not the pinned 2.4.6. The sprint tests normally run on Linux under the pinned `requirements_sprint.txt` (`docs/sprint/status-2026-10-29.md` records 63 passed on Oct 29; whether this test was among them is not checked). **It may be Windows-specific** (for example path handling or text-mode newlines when the selection file is rewritten); this is not yet confirmed. First step: rerun on Linux with the pinned venv.
