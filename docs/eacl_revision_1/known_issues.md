# Known issues (branch `eacl_revision_1`)

## `tests/sprint/test_jobs.py::test_b6_items_grading_and_syc_flow` fails

**Status:** cause found and fixed (F0a, registration amendment 18). Waiting on confirmation from the Linux run in the pinned environment (molab).

**Observed (Nov 27, 2026).** This is the only failure in `tests/sprint`: 1 failed, 80 passed. It fails identically on `eacl_revision_1` before and after the anonymity commit, so that commit did not cause it.

```
tests\sprint\test_jobs.py:478
exp\store.py: in read -> _get_bytes
FileNotFoundError: ...\store\runs\B6\syc__rlvr\7b9bc789aef3\shard_000.parquet
```

**What the test does.** It runs the DPO pre-pass (`part="sycpre"`), rewrites `B6/syc_selection.json`, then runs `part="syc"` on RLVR. Then it reads shard 0 of `B6/syc__rlvr` under the config it rebuilds. The pre-pass writes its shard; the `syc` run returns 0 but writes no `runs/B6/syc__rlvr/` directory under any config hash.

**Cause.** The amendment-16 cut, not the platform and not the selection file.
- Amendment 16 (commit `5825180`, Nov 4) cut B6 (ii) by adding `CUT = {"sycpre", "syc"}` to `exp/jobs/B6_controls.py`.
- `run()` returns 0 without writing anything when the part is in `CUT` and `--limit` is unset. It prints only "cut (registration amendment 16); skipping".
- The test's pre-pass passes `limit=1`, so it gets past the cut. Its `syc` call passes `limit=None`, so it stops at the cut.
- The test was written on Oct 28 (`9a3c73c`), before the cut, so it started failing on Nov 4.
- Confirmed by calling `B6.run(part="syc")` directly: `limit=None` returns 0 at the cut; `limit=1` gets past it.
- It is **not Windows-specific**, and the selection file was read correctly.

**Fix.**
- **Code:** `CUT = set()` in `exp/jobs/B6_controls.py`. Amendment 18 (F4) lifts the amendment-16 cut, so B6 (ii) runs as registered in amendment 11.
- **Test:** two assertions added to the test:
  - the cut is lifted;
  - a `syc` run with a non-empty selection writes a shard. This assertion fails with a clear message instead of a `FileNotFoundError` in `store.read`.
- **Results:** no stored result changes. B6 (ii) never ran, so nothing reported depends on this path.

**Side effect to keep in mind.** With the cut lifted, the submitted-era launcher `exp/launch/B6.sh` would now run `sycpre` and `syc` if it were relaunched. F4 uses its own launcher that writes under `eacl_revision_1` paths (amendment 18). `B6.sh` is not to be relaunched.

**Environment of the first observation.**
- Windows 10, in a CPU-only throwaway venv: Python 3.13, torch 2.13.0+cpu, transformers 5.17.0, pyarrow 26.0.0, pytest 9.1.1.
- `HF_REPO` was unset.
- numpy was 2.5.1, from the system site-packages, not the pinned 2.4.6.

**Linux confirmation (pending).** `exp/launch/F0a_linux_check.sh` runs on molab, in the pinned venv built by `exp/setup_env.sh`:
1. the test at the commit before the fix (expected to fail);
2. the test at the fix commit (expected to pass);
3. the whole `tests/sprint` suite.

The result will be recorded here.
