# Operations runbook (branch `eacl_revision_1`)

Every session reads this file before acting and follows it. It is the only copy of the rules. When the PI sets a new rule, or something breaks and a fix is agreed, the rule goes in here **in the same commit as the fix**.

Rules marked *(added)* come from the sprint docs, the launch scripts, earlier notes, or tonight's fixes. They were not in the PI's list of Nov 28.

## Compute and molab

- **Account and notebooks:**
  - One molab account only. At most 2 GPU notebooks (N2, N6); never open a third, because opening one ends another. No other accounts.
  - *(added)* Sandboxes end after 12 h, and can be reclaimed silently earlier (HTTP 410). Plan relaunches; every job resumes from saved progress.
- **Lane cells:** each lane runs as a blocking foreground cell in its notebook. The cell runs until the lane writes `EXIT=`, so the notebook counts as busy even if the laptop dies. A lane is never a detached process with no running cell.
  - *(added)* The cell is named `lane`, uses only underscore-local names plus `lane_result`, and never overwrites the PI's cells (the HF-token cell and the older launch cells).
  - *(added)* **How to start it.** Start the `lane` cell through the notebook's own run endpoint, as the UI does. Do not run it from a code-mode or scratchpad request: when that request's stream ends (a long call, a proxy timeout), the cell run it started is interrupted. Tonight's first attempts were interrupted this way.
  - *(added)* **Monitoring.** While the lane cell runs, the kernel is busy, and scratchpad calls hang. Status comes from HF: lane logs and progress are pushed every 5 min. When the lane ends, its cell finishes, and the cell can be edited and restarted.
  - *(added)* **Long commands.** A long one-off command in a notebook, such as building the environment, is started as a detached process writing a log file, then polled. A long scratchpad call is cut when the stream ends, and its child processes die with it (the N6 env setup did on Nov 28).
- **Watcher:** one lightweight watcher for both notebooks. It polls every 5 min, reports status only, and never touches notebooks or runs.
  - *(added)* It reads HF, not the kernels: the last push per lane, `lane.log`, job-log events, failures. A lane with no push for more than 15 min is reported as stale.
- **Idle notebooks:** never leave a notebook with no running cell for more than 60 min. Tell the PI when a queue is empty.
- **Sandbox recovery:**
  1. When a sandbox dies, the PI reopens the notebook and gives Claude Code the new URL and token.
  2. Claude Code restores `HF_TOKEN` (see Secrets), checks progress on HF, rebuilds the environment, and relaunches the lane, which resumes from saved progress.
  3. *(added)* Before relaunching, check on HF whether the previous sandbox's lane actually ran: its last push, `logs/<lane>/lane.log`, and any partial progress. Report what was lost.
- **Laptop:** Claude Code runs with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`. Keep watchers and background shells minimal, because low memory has killed them before. Runs on molab do not depend on the laptop.
- *(added)* **Staged launch:** setup → smoke (throwaway directory, `STORE=local`) → check the lane is alive → full lane. Report the peak memory at each stage.
- *(added)* **molab runs under gVisor:**
  - `ps` does not list detached processes, so read `/proc` instead;
  - only `memory.usage_in_bytes` exists in the cgroup;
  - the peak comes from `exp/memwatch.py` (`/tmp/mem_peak`).

## Resource safety (per job)

- **Pre-flight** before every job: free RAM, the memory limit, free disk and GPU memory (`exp/resources.py: preflight`). Abort if free RAM < 16 GB or free disk < 2× the size of the models the job needs.
  - Unit tests bypass the pre-flight (`tests/sprint/conftest.py`); real jobs never do.
  - The engine probe runs it too.
- **One at a time:** one job per notebook at a time. One model per process, except jobs registered to load a pair (M2 `--pair`).
- **Weights:** they go into a per-job directory, deleted when the job ends; `HF_XET_CHUNK_CACHE_SIZE_BYTES=0`.
  - In F1 the job is one checkpoint × benchmark unit, so its weights are deleted after each unit. A failed unit keeps them for the lane's one retry.
- **vLLM:** `max_model_len` = prompt + cap; `swap_space=0`; no CPU offload.
  - **The fast engine uses `gpu_memory_utilization` 0.92** instead of the sprint's 0.85. Why that's safe:
    - it budgets **GPU** memory only, and host RAM is unaffected;
    - each GPU runs one job and one model;
    - in the F0b engine log, weights and non-torch memory took 13.94 GiB, peak activations 1.08 GiB and CUDA graphs 0.16 GiB, against 94.97 GiB on the device. That leaves about 7.6 GiB below the device total beyond the 87.4 GiB budget;
    - 0.92 is vLLM's own default;
    - F0b ran both B1 arms and a 4-hour 32K bench at 0.92 with no out-of-memory error.
  - The sprint's 0.85 dated from jobs that shared the GPU with an HF model. Switch back if a job ever shares the GPU.
- **RES lines:** log a RES line (RAM, RSS, cgroup, disk, GPU) every few minutes in every job, including during long stretches with no finished generation. `harness.generate_stream` takes an `on_step` heartbeat for this.
- **Activation jobs** save only summaries and direction vectors, never raw activation dumps.
- **CPU and RAM:** notebooks have 4 CPUs and a 28 GiB RAM budget, so keep thread pools and compile workers small. `exp/launch/lane.sh` sets `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TOKENIZERS_PARALLELISM=false`, `MAX_JOBS=4` and `TORCHINDUCTOR_COMPILE_THREADS=1`.
- *(added)* **Environment:** keep `PYTHONPATH` and `PYTHONSAFEPATH` unset; molab sets `PYTHONSAFEPATH`, which breaks `python -m exp...`. Generation jobs set `VLLM_USE_FLASHINFER_SAMPLER=0`, as in the submitted runs, and record it in their configs.

## Storage and data

- **Where results go:** `STORE=hf`; `HF_REPO` is set in every notebook. Raw outputs go to the HF dataset `<owner>/rlvr-se-runs` under `runs/eacl_revision_1/...`, and progress files under `progress/eacl_revision_1/...`.
- **Sync:** progress is synced to HF at least every 15 min during a unit. `exp.hfsync run` pushes progress and logs every 5 min, and once more at the end.
- **No overwrites:** shards are never overwritten. Jobs skip shards that exist and resume from progress files.
  - *(added)* A repair of a stored result happens only with the PI's approval, recorded in `docs/sprint/disclosure-ledger.md`.
- **Provenance:** every shard records the git SHA, the model repo + commit SHA, the full config and the package versions (`exp/store.py`).
  - *(added)* What defines the samples goes in the config hash: items, seeds, sampling, caps. Engine settings that change only throughput are recorded per process segment, outside the hash, so retuning them resumes instead of restarting (F1).
- **Pins:** model SHAs come from `docs/checkpoint_map.md`. Pinned package versions are used exactly: `requirements_sprint.txt`, with `requirements_revision.txt` on top for the CPU scorers only, resolved with the sprint file as a constraint.
- **Git:** summaries and docs go to git on `eacl_revision_1`. `main` and the `submitted-2026-11-07` tag are never touched.
- **Frozen material is never modified:** `items/` (manifest guard), `results/full_run_*`, the analysis plan, `results/sprint/`, and the existing tags.
- *(added)* **Lane separation:** two lanes never write the same file. Each unit or part belongs to one lane; units move between lanes only through the queue files, never while running. `exp/hfsync.py` pushes only files this notebook changed since its pull; on Oct 30 a stale copy from one lane overwrote another's progress.
- *(added)* **New data:** item sets come from pinned sources, with each source file's sha256 recorded, and are frozen with a `MANIFEST.sha256`. Item directories are byte-exact in git (`.gitattributes`: `-text`); `*.sh` files are `eol=lf`. Vendored code is fetched at pinned commits with import rewrites only (`exp/vendor/VENDOR.md`).

## Secrets and anonymity

- **HF_TOKEN:**
  - Copied into each sandbox from this machine's environment. Never printed, logged, committed, or written to a cell or file. Cells and tests that don't need it run without it.
  - *(added)* **How:** the code that sets it is piped over stdin into the marimo scratchpad, never written to a file. Only `bool(...)` is printed, plus an HF `whoami` status check.
- **molab sandbox tokens** go only to Claude Code. Claude Code keeps them outside the repository.
- **Committed files** contain no usernames, account names, sandbox IDs or emails. The HF dataset is referred to as `<owner>/rlvr-se-runs` in docs, and code takes it from `$HF_REPO`.
- **Anonymous mirror (PI, Nov 28):** before submission, build the anonymous mirror of the repository with all identifying terms hidden. That includes the two older files that still name the account, `docs/history/README_v1.md` and registration amendment 3, which stay unedited in the repo.

## Process

- **Registered design:** any change is written as a dated amendment note **before** it runs, with the PI's approval. Ask the PI before changing any registered decision.
- **Smoke test** before every full job. Exemption (PI, Nov 28): the engine probe, a 2 × 10-min diagnostic that writes no results, has no smoke step.
- **Engine changes:** any change to the engine or the generation path is validated with the F0b rule before use: B1-H1 within its submitted interval, plus the two per-arm checks (amendment 18 and its Nov 27 note).
- **Tests:** unit tests for every parser and readout (and every scorer). The CPU suite passes before every GPU launch.
- **CLI:** every job is a standalone CLI (`python -m exp.run --job ...`) with no marimo dependency.
- *(added)* **Scoring** runs on CPU after generation, so GPU jobs need no new packages.

## Reporting

- **Per job:** write `results/eacl_revision_1/<job>/summary.md` and add a line to `docs/eacl_revision_1/status.md`.
- **Side by side:** submitted and revised numbers are kept together. Nothing from the sprint is overwritten.
- **Contradictions:** anything that contradicts a paper claim is flagged to the PI, never softened or dropped.
- **Git:** commit and push after each job, with the attribution line.

## Where the code or procedure did not follow these rules (Nov 28)

| rule | finding | status |
|---|---|---|
| lane = blocking foreground cell | N2's PROBE lane was started as a detached process; its `lane` cell only waits on it | the next N2 relaunch uses the full cell form |
| lane cells started reliably | starting the cell from code mode was interrupted when the request stream ended | fixed: started through the notebook's run endpoint |
| `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` | the current session was not started with it (a background shell was killed for low memory on Nov 28) | **open:** needs Claude Code restarted with the variable set; it cannot be set from inside the session |
| `HF_TOKEN` never written to a file | for N6 on Nov 28 the token went through a temporary file in Claude Code's scratchpad, deleted at once | fixed: stdin pipe only |
| weights deleted when the job ends | F1 launchers kept one checkpoint's weights across units | fixed: deleted after each unit |
| RES line every few minutes | F1, the F0b bench and the probe logged RES only when a generation finished | fixed: `on_step` heartbeat |
| pre-flight before every job | the engine probe had none | fixed |
| smoke before every full job | the engine probe (a 2 × 10-min diagnostic that writes no results) has no smoke step | resolved: exempt (PI, Nov 28) |
| `gpu_memory_utilization` 0.85 → 0.92 | not recorded before | recorded above (kept at 0.92) |
| no account names in committed files | `docs/history/README_v1.md` (a clone URL and dataset path with a username) and registration amendment 3 (the HF dataset's account name) | resolved: left as they are (PI, Nov 28); hidden in the anonymous mirror before submission |
