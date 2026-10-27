# RLVR Side Effects on Olmo 3

A checkpoint-level study of what **RLVR (reinforcement learning with verifiable rewards)** does to a language model's behavior *outside* its reward domain, using AllenAI's Olmo 3 7B checkpoint ladder.

## Research question

Olmo 3 7B was released with checkpoints for every post-training stage: base → SFT → DPO → RLVR. That lets us isolate the RLVR stage specifically, instead of bundling it with SFT/DPO/RLHF:

> Does the RLVR stage shift personality/trait self-reports, behavioral tendencies (sycophancy, calibration), or general commonsense reasoning, in a way that is distinguishable from sampling and prompt noise?

The RLVR training domains are math, code and instruction-following. Our probes are deliberately off-domain (trait items, commonsense QA), with a planned MATH run as a capability control.

## Why it matters, and what is new here

The instrumental-convergence question (do RL-trained reasoning models show more self-preservation-style behavior?) has largely been answered with negative or measurement-noise-limited results on this exact checkpoint family. Trait and behavioral drift at the *RLVR stage in isolation* is much less covered: prior personality-drift work bundles RLHF/DPO/instruction tuning together, and none of it uses the Olmo 3 ladder. A second motivation is measurement quality: earlier checkpoint studies found apparent behavior shifts that were really measurement drift, so this project treats sampling variance as something to measure first (see [VARIANCE.md](VARIANCE.md)). Full literature review and rationale: [docs/sprint_plan.md](docs/sprint_plan.md).

## Repo structure

```
.
├── README.md
├── VARIANCE.md                  version history (v1 -> v3) and sampling-variance findings
├── RESULTS.md                   findings so far on the RLVR checkpoint
├── probe_list.md                human-readable reference list of all probe items
├── rlvr_probe_notebook.py       marimo notebook: setup, model loading, original probe cells
├── skills-lock.json             pins the marimo-pair Claude Code skill used for pairing
├── docs/
│   └── sprint_plan.md           literature review, rationale, five-week plan
├── probes/                      the locked probe set, importable, no notebook dependency
│   ├── __init__.py              ALL_PROBES, TRAIT_PROBES, REASONING_PROBES, build_chat(), REASONING_CAP
│   ├── traits/self_report.py    8 Likert self-report items
│   ├── behavioral/behavioral_probes.py   2 behavioral items (sycophancy, calibration)
│   └── reasoning/commonsense_qa.py       10 commonsense_qa items (static copy)
├── results/
│   ├── rlvr_baseline_v1.{json,csv}   first sweep (20 rows), kept raw
│   ├── rlvr_baseline_v2.{json,csv}   re-run of the 10 truncated items (10 rows)
│   └── rlvr_baseline_v3.{json,csv}   LOCKED RLVR baseline (36 rows)
│   ├── dpo_pilot_v3.{json,csv}       DPO pipeline check, 36 rows (exploratory pipeline check, not a finding)
│   └── dpo_pilot_v3_env.txt          exact package versions used for the DPO pilot
└── scripts/                     archived run scripts (need a live marimo session, see below)
    ├── rerun_truncated_v2.py
    ├── run_baseline_v3.py
    ├── dpo_pilot_v3.py
    └── rerun_truncated_v3.py
```

**Probe definitions are real files under `probes/`** (extracted from the notebook). They are the source of truth for future runs. `probe_list.md` is a human-readable reference list alongside them, not the source of truth. `probes/` is verified identical to the notebook's inline definitions and reproduces every `prompt_used` in the v3 data exactly. Import it directly:

```python
from probes import ALL_PROBES, build_chat   # build_chat applies the v3 reasoning cap
```

## Checkpoints and data

| Stage | Hugging Face repo |
|---|---|
| Base | `allenai/Olmo-3-1025-7B` |
| SFT | `allenai/Olmo-3-7B-Think-SFT` |
| DPO | `allenai/Olmo-3-7B-Think-DPO` |
| RLVR (final) | `allenai/Olmo-3-7B-Think` |
| RL directly from base | `allenai/Olmo-3-7B-RL-Zero-{Math,Code,IF,General,Mix}` |

Lineage confirmed from the `base_model` tags on the model cards. Datasets: `tau/commonsense_qa` (validation split, first 10 items, stored statically in `probes/`) and `allenai/RLVR-MATH` (planned capability control; pulled in the notebook, not yet used for evaluation).

## How to reproduce

This work was done by pairing Claude Code with a live [marimo](https://marimo.io) notebook on a GPU-backed molab sandbox (NVIDIA RTX Pro 6000, 96 GB VRAM). Models load in bf16 with no quantization. Requires `transformers >= 4.57.0`. molab sessions end after 12 hours or 90 minutes idle, so work is chunked and results are saved as files.

1. Start a GPU marimo/molab notebook from `rlvr_probe_notebook.py` and run the setup, checkpoint-discovery, model-loading and dataset cells.
2. Install the `marimo-pair` skill for Claude Code (source `marimo-team/marimo-pair`, pinned in `skills-lock.json`) and connect Claude Code to the running notebook.
3. Run the sweep with the v3 protocol. The protocol is:
   - prompts from `probes/`, with the reasoning cap `"Keep your reasoning to at most 5-6 sentences, then give your final answer."` appended to the last user message of every probe;
   - `do_sample=True`, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=2000`;
   - seeds `torch.manual_seed(0/1/2)` for samples 1-3 of the 8 self-report items, seed 0 for the other 12 items (36 generations);
   - an answer is valid only if the `<think>` block closed and an `ANSWER: X` line follows `</think>`; otherwise the row is flagged `forced`.
4. `scripts/run_baseline_v3.py` implements this exactly. **The scripts are not standalone**: they run inside the notebook kernel through `marimo._code_mode` and read the probe definitions from the live notebook cells, so they fail under plain `python`. They are archived unmodified (plus a header) for reproducibility. The next run should import from `probes/` instead; copy or clone `probes/` into the sandbox's working directory to do so.

## DPO pilot (exploratory pipeline check, not a finding)

`results/dpo_pilot_v3.{json,csv}` is a pipeline check of the v3 protocol on `allenai/Olmo-3-7B-Think-DPO`, pinned at commit `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`. It uses the same 36 items and seeds, the same prompt template and cap sentence, `temperature=0.7`, `top_p=0.9`, and a uniform `max_new_tokens=4000` (v3 used 2000). It was run in an isolated venv (transformers 5.14.1, huggingface-hub 1.33.0, torch 2.11.0+cu130; see `results/dpo_pilot_v3_env.txt`), not the versions used for v3, which were never recorded. It is not compared to RLVR and supports no claim about the DPO stage. Script: [`scripts/dpo_pilot_v3.py`](scripts/dpo_pilot_v3.py) (standalone, no marimo; its header lists the inputs it expects).

**Output schema note: `early_stop` vs truncation.** A row can be forced for two different reasons, and they should not be pooled. *Truncation*: generation used the whole budget (`n_generated_tokens >= max_new_tokens`, flagged as `hit_token_cap`). *Early stop* (`early_stop`): generation ended on an EOS token before the cap without a closed think block, i.e. `not think_closed and n_generated_tokens < max_new_tokens`. `early_stop` is defined here but is not a column in the existing DPO pilot files; it is derivable from `think_closed`, `n_generated_tokens` and `max_new_tokens`. The pilot has one such row (`trait_confidence_selfreport` sample 3: sampled `<|endoftext|>` after 53 tokens, mid-sentence; it reproduces exactly under the same seed).

## Known, deferred cleanup

These were deliberately left alone because the notebook session is still live and editing cells would re-run their dependents (the notebook is in autorun mode):

- **`day1_` / `day2_` cell names.** Cells in `rlvr_probe_notebook.py` are still named `day1_status`, `day2_title`, `day2_smoke_test`, `day2_full_sweep`, `day2_fix_unparsed`, `day2_save_results`.
- **Save-path inconsistency.** `day2_save_results` still writes to `probes/results/day2_sweep_results.{json,csv}` (a path relative to the sandbox), whereas the local files were renamed to `results/rlvr_baseline_v{1,2,3}`. Re-running that cell would recreate the old names.
- **Two copies of the probes.** The notebook still contains its own inline probe definitions (`define_trait_probes`, `define_reasoning_probes`). `probes/` is an extracted copy, identical as of this writing. If either changes, they must be reconciled.
- **The notebook's Day 2 markdown** mentions `probes/traits/self_report.py` and similar paths as aspirational; those files now exist under `probes/`.

## Status

- **RLVR baseline (v3): locked and validated.** 36 rows, all with a closed reasoning block and a valid answer. See [RESULTS.md](RESULTS.md) and [VARIANCE.md](VARIANCE.md).
- **DPO comparison: next step, not yet run.** The identical protocol will be applied unchanged to `allenai/Olmo-3-7B-Think-DPO`.
- No paraphrase/order-robustness check and no MATH capability-control run yet.

## Five-week plan and where we stand

Summary of [docs/sprint_plan.md](docs/sprint_plan.md):

| Week | Goal | Status |
|---|---|---|
| Head start | Real 7B checkpoints on GPU, datasets pulled, first probe items, end-to-end pipeline | Done |
| 1 | Locked probe set + protocol; checkpoint set decided; paraphrase/order-robustness protocol | Probe set locked (v3); checkpoint ladder mapped; **paraphrase/order-robustness protocol not yet written** |
| 2 | Pilot on 2 checkpoints (DPO and RLVR), compute permutation consistency, calibrate | RLVR pilot done; **DPO is next** |
| 3 | Full checkpoint sweep + MATH capability control | Not started |
| 4 | Analysis, robustness check against paraphrase/order noise, cross-domain regression check | Not started |
| 5 | Write-up and repo polish | Not started |

We are at the Week 1 to Week 2 boundary: one checkpoint is measured under a locked protocol, and the paraphrase-robustness design and the DPO run are the immediate next items.


## Sprint jobs (branch `exp/mechanism-sprint`)

Behavioural and mechanistic follow-ups to the frozen study. Code lives in `exp/`, docs in `docs/sprint/` (start with
`registration-2026-10-27.md` and `checkpoint_inventory.md`), small summaries in `results/sprint/`. Frozen material
(`items/`, `results/full_run_*`, `docs/analysis-plan.md`, tags) is never modified.

Every job is a standalone CLI with no marimo dependency. From a notebook cell on the molab GPU box:

```bash
cd /tmp && (git clone -b exp/mechanism-sprint https://github.com/shagunchadha/RLVR-side-effects.git rse || git -C rse pull)
cd /tmp/rse && bash exp/setup_env.sh              # pinned venv at /tmp/sprint_venv + version gate (aborts on mismatch)
export HF_HOME=/tmp/hf RESULTS_DIR=/tmp/sprint_out STORE=local     # STORE=hf needs HF_TOKEN
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"   # molab sets PYTHONSAFEPATH=1
$PY -m exp.run --list
```

| job | what | launch | GPU / time (estimate) |
|---|---|---|---|
| B0 | checkpoint inventory (refs, SHAs, sizes, chat template, injected system prompt) -> `docs/sprint/checkpoint_inventory.md` | `$PY -m exp.run --job B0` | none; ~2 min |
| M1 | weight-difference spectra RLVR-DPO vs DPO-SFT -> `results/sprint/M1/` | `$PY -m exp.run --job M1 > /tmp/m1.log 2>&1` (resumable; see below) | GPU recommended; streams one ~5 GB safetensors file per checkpoint at a time (peak ~15 GB on disk) |
| M2 | teacher-forced scoring of the frozen traces | see `exp/jobs/M2_tracescore.py` header | yes |
| B1 | Likert x20 seeds + readouts | see `exp/jobs/B1_likert.py` header | yes |
| B3 | reasoning-cap x format factorial | see `exp/jobs/B3_instructions.py` header | yes |

M1 survives a reclaimed notebook: each finished tensor is appended to `$RESULTS_DIR/M1/M1_progress_shard0of1.csv` and a
restart skips the tensors already there. The durable copy lives on the local machine, through the pairing connection:
`MARIMO_TOKEN=... python exp/m1_sync.py --url <notebook url> push` before relaunching on a fresh notebook (if a local copy exists),
then `... watch` while it runs (copies the CSV every 20 tensors and the log every minute to `results/sprint/M1/partial/`).
The log has a `START i/355 <tensor>` line before every tensor and a `RES` line (RAM, cgroup, disk, GPU) every 10.

Shards go to `$RESULTS_DIR/runs/<job>/<ckpt>/<config_hash>/shard_XXX.parquet` (or the same layout in the private HF
dataset `shagunchadha/rlvr-se-runs` with `STORE=hf`). Each shard records the git SHA, model repo + commit, the full
config and package versions. Re-running a job skips shards that exist and never overwrites. Upload local shards later
with `$PY -m exp.sync --from local --to hf`. CPU tests: `$PY -m pytest tests/sprint -q` (tiny random Olmo3, no downloads).
