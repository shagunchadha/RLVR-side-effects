# Known issues (branch `eacl_revision_1`)

## `tests/sprint/test_jobs.py::test_b6_items_grading_and_syc_flow` fails

**Status:** fixed and confirmed on Linux (F0a, registration amendment 18; fix commit `fb54232`).

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

**Linux confirmation (Nov 27, 2026).** Run by the PI in a separate Linux container:
- **Environment:** Python 3.13.18, all 220 pins of `requirements_sprint.txt` installed exactly (numpy 2.4.6, torch 2.13.0, vllm 0.31.0), no GPU, `HF_TOKEN` unset.
- **Version check:** `exp/setup_env.sh` reported all versions matching. Its only report was "no CUDA GPU visible", so the test steps of `exp/launch/F0a_linux_check.sh` were run directly.
- **Before the fix (`3e7c363`):** the test **failed** with `FileNotFoundError`, after logging "B6 syc rlvr: cut (registration amendment 16); skipping". This confirms the diagnosis.
- **At the fix (`fb54232`):** the test **passed**.
- **`tests/sprint` at `fb54232`:** 79 passed, 2 failed (`test_m6_cka_gram_matches_features_and_runs`, `test_m1b_known_values`).
  - Both stopped at the host RAM preflight, `exp/resources.py:138`: "PREFLIGHT ABORT: available RAM 15.7 GB < 16 GB". The container had under 16 GB.
  - This is not a code fault, but a unit test should not depend on host RAM. See the next issue.

## Unit tests depended on host RAM and disk (the job preflight)

**Status:** fixed (F0a).

- **Cause:** jobs call `exp.resources.preflight` before loading a model. It aborts when available RAM is under 16 GB or free disk is under the 7B job's need. The CPU unit tests run the jobs with tiny models, but still went through that preflight, so on a host with < 16 GB RAM `test_m6_cka_gram_matches_features_and_runs` and `test_m1b_known_values` aborted.
- **Fix, tests only:**
  - An autouse fixture in `tests/sprint/conftest.py` (`_no_host_preflight`) replaces `exp.resources.preflight` with a no-op for the duration of each test.
  - A test marked `real_preflight` keeps the real one.
  - `tests/sprint/test_preflight.py` checks both sides: the bypass is active in unit tests, and the real preflight still aborts on an impossible disk need.
- **Real jobs are unchanged:** `exp/resources.py` is not modified, and nothing outside the tests is patched.

## F0b smoke failed on both lanes: `generate_stream: 2 requests never finished`

**Status:** fixed (Nov 27, 2026). Found by the PI in the F0b smoke step; nothing was written, and both lanes stopped (SKIPPED after two attempts).

- **Cause, from the pinned vLLM 0.31.0 source (PI):**
  - `LLM.enqueue` → `_add_request` → `LLMEngine.add_request` returns the **internal** request id. `input_processor.assign_request_id` sets it to `f"{external_id}-{random_uuid():.8}"`.
  - The `RequestOutput`s from `llm_engine.step()` carry the **external** id, LLM's plain counter; vLLM's own `_run_engine` sorts by `int(request_id)`.
  - `generate_stream` keyed its pending map by the enqueue ids, so no finished output matched. The outputs were skipped silently, and the loop ended when the engine was empty.
- **Fix in `exp/harness.py`:**
  - `generate_stream` keys the pending map by the external id. `_external_request_id` strips the suffix, and anything not shaped `<external>-<8 hex>` raises, since that would be an API change.
  - A finished output whose id matches no pending request now **raises** instead of being skipped.
- **Why the tests missed it:** the fake engine in `tests/sprint/test_fastpath.py` returned the same id from `enqueue` and in the outputs.
- **Tests now:**
  - The fake engine returns suffixed internal ids from `enqueue` and plain external ids in the outputs, as vLLM 0.31 does.
  - New tests cover an unknown output id, an enqueue id without the suffix, and the id parser.
  - Against the previous `harness.py`, the stream test fails with exactly the smoke error ("2 requests never finished"). With the fix, the CPU suite passes: 93 passed.

## F1 scorers: upstream properties found while testing on known examples (Nov 27–28, 2026)

None of these is fixed in our code. They are recorded so the F1 numbers can be read correctly. Full detail is in the registration note to amendment 18 of Nov 28.

1. **OLMES EM Flex (the developers' maths scorer) turns `\left(` into `\le(`.** lm_eval 0.4.3's `normalize_final_answer` deletes the unit "ft". A correct tuple or interval written without `\left…\right` is therefore scored wrong. math-verify, the primary scorer, scores it right.
2. **OLMES EM Flex scores a leading-zero answer (`\boxed{073}`) wrong:** sympy refuses to parse "073". math-verify scores it right.
3. **The IFEval and IFBench checkers are nondeterministic:** they draw `random` values for arguments an item does not give. Unseeded reruns of Google's code on its own GPT-4 responses gave prompt-strict 416 or 417 of 541. F1 seeds `random` and `langdetect` before every check.
4. **OLMES's `nltk` 3.8.1 cannot load `punkt_tab`.** It rewrites the path to `punkt/PY3_tab` and raises `OSError`, which IFBench does not catch, so IFBench crashes. Pinned to IFBench's own `uv.lock` (`nltk` 3.9.2).
5. **IFBench's committed expected results (Oct 2025) predate later checker fixes.**
   - The Oct-2025 code reproduces all 293 strict records.
   - The pinned head differs on 5 records (`format:list`, `ratio:overlap`, `ratio:sentence_type`).
   - One expected-results record is malformed upstream: about 7 tab-joined rows in one "prompt".
6. **The data copies differ upstream.**
   - IFBench's repo has two copies of the test set; they differ by trailing whitespace on 2 prompts.
   - Google's GitHub IFEval data corrected one prompt (key 2785) after the HF release.
   - F1 uses the HF versions, which are the developers' inputs.

## F1: what to check first if AIME differs from the developers' numbers (PI, Nov 28)

If our AIME 2024 or AIME 2025 score differs from the developers' (DPO→RLVR 74.6→71.6 and 62.7→64.6) by **more than 5 points**, the first thing to check is **OLMES's AIME generation cap**. OLMES's `aime:2024::olmo3:adapt` and `aime:2025::olmo3:adapt` configs set `max_gen_toks` to **16,384**. The report says 32,768, and F1 uses 32,768 plus continuation to 65,536, as registered.

- If the developers' AIME numbers came from 16,384-token generations, a higher cap by itself can raise our accuracy, or lower it if long generations wander.
- The check: rescore F1's AIME generations **truncated at 16,384 new tokens**, using each generation's stage-1 tokens, under the same scorers. Compare that number with the developers'.
- Only then go on through the rest of the amendment's investigation order (prompt template, extractor, dataset revision, system prompt, cap-hit rate).

## F1: which IFBench checker is the main score (PI, Nov 28)

- **Main IFBench score:** IFBench @ `c94da020` (2025-10-17), vendored as `exp/vendor/ifbench_2025_10/`. It is the commit that produced IFBench's committed expected results, and it reproduces all 293 of them at both strict and loose. It is also the closest version in time to the developers' evaluation.
- **Alongside:** the pinned head (`48891866`), which differs on 5 sample records.
- **What that version needs:** spaCy 3.8.7 and `en_core_web_sm` 3.8.0, pinned with a hash in `requirements_revision.txt`.
- **Vendored modifications,** beyond import paths:
  - its import-time `download('en_core_web_sm')` is removed, since the model is pinned instead;
  - during scoring, `nltk.download` is a no-op, because the code calls it inside every check and the resources are fetched once beforehand.
- **Scorer wrapper:** it drops `None`-valued kwargs before calling either version, as the head's own `evaluation_lib` does. Without that, the old version's loose path crashes.
