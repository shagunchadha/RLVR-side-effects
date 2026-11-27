# Side effects of the RLVR stage in Olmo-3-7B-Think

Olmo-3-7B-Think was post-trained in three released stages (SFT → DPO → RLVR), so the effect of the RLVR stage on its own can be measured by comparing the final checkpoint with its DPO parent on identical inputs. RLVR rewards maths, code and instruction-following. We ask what it changes outside those domains.

After RLVR, the model answers self-report Likert items more extremely than DPO: the share of extreme answers rises by +0.181 [+0.131, +0.233] (Holm p 3.5e−7), and the effect replicates on IPIP-NEO-120 (+0.095 [+0.055, +0.138]). It also deliberates longer on questions with no fixed answer (indeterminacy effect +626 think tokens [+500, +768]), and both effects grow with RL step (ρ +0.55 for length, +0.57 for P(extreme)). Prefix-swap experiments put the extra length in RLVR's policy rather than in the text, and the extremity builds up during the deliberation.

The one causal test points the other way. Ablating the backtracking direction cuts RLVR's deliberation on these items from 1175 to 506 tokens, yet P(extreme) does not fall (+0.13 [−0.12, +0.37]). So the deliberation is where the shift shows, but it is not shown to cause it. The registered positive control also failed: RLVR is no more accurate than DPO on hard maths (−0.022 [−0.056, +0.009]). It is not more overconfident, and it refuses unanswerable questions more often (+0.108 [+0.068, +0.149]). All verdicts are in [`docs/sprint/gate2.md`](docs/sprint/gate2.md), and the claims we do and do not make are in [`docs/sprint/claims-draft.md`](docs/sprint/claims-draft.md). Intervals are 95% bootstrap CIs, and there is no multiplicity correction across jobs.

## Folder structure

The study has two parts:
- **Frozen pilot** (pre-registered; tags `protocol-freeze-2026-10-23` and `analysis-plan-2026-10-24`). It is never modified.
- **Sprint** (behavioural and mechanistic follow-ups, registered Oct 27, 2026 with dated amendments).

```
.
├── README.md
├── requirements_sprint.txt      SPRINT  fully pinned environment for exp/ (uv pip compile)
├── requirements_pinned.txt      FROZEN  environment of the pilot full run
├── requirements_analysis.txt    FROZEN  environment of the pilot analysis
├── rlvr_probe_notebook.py       FROZEN  original marimo notebook (v1-v3 baseline)
├── skills-lock.json                     pins the marimo-pair Claude Code skill
│
├── exp/                         SPRINT  all sprint code; every job is a standalone CLI
│   ├── run.py                           entry point: python -m exp.run --job <ID>
│   ├── jobs/                            one module per job (B0-B8, M1-M6, M1b, PH)
│   ├── analysis/                        summary scripts: raw shards -> results/sprint/<job>/
│   ├── launch/                          notebook queues and lane launch scripts
│   ├── setup_env.sh                     builds the pinned venv and runs the version gate
│   └── harness.py, store.py, ...        model loading, shard store, HF sync, resource checks
├── items_sprint/                SPRINT  item sets added in the sprint (B2, B6, B7, B8, M4) + MANIFEST.sha256
├── tests/sprint/                SPRINT  CPU tests (tiny random Olmo3, no downloads)
├── results/sprint/              SPRINT  per-job summaries, figures and small CSVs (raw outputs are on HF, see below)
│
├── items/                       FROZEN  pilot item sets A-E, builders, review trail, MANIFEST.sha256
├── items_draft/                 FROZEN  earlier item-set drafts
├── probes/                      FROZEN  the v3 probe set (20 items), importable
├── scripts/                     FROZEN  pilot run and analysis scripts (full_run.py, analyze.py, stats.py, ...)
├── tests/                       FROZEN  parser regression and coverage study (tests/sprint/ excepted)
├── results/                     FROZEN  pilot outputs: full_run_{sft,dpo,rlvr}/, analysis/, pilots, batch tests
│
└── docs/
    ├── analysis-plan.md         FROZEN  pre-registered analysis plan of the pilot
    ├── predictions.md, exploratory-registration-2026-10-25.md, COMPARISON_*.md, ...   FROZEN pilot docs
    ├── sprint/                  SPRINT  registration + amendments, disclosure ledger, Gate 1/2, claims draft
    └── history/                         superseded documents (see "History" below)
```

## How to reproduce

**Environment.** Use a Linux x86_64 machine with Python 3.13 and one GPU. We used a 96 GB GPU, and the models run in bf16 without quantization. Build the pinned venv from the repo root:

```bash
bash exp/setup_env.sh        # venv at $VENV (default /tmp/sprint_venv); installs requirements_sprint.txt and
                             # runs exp.versioncheck, which aborts on any version mismatch
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
$PY -m pytest tests/sprint -q   # CPU tests
```

**Running a job.**

```bash
export HF_HOME=/tmp/hf RESULTS_DIR=/tmp/sprint_out STORE=local   # STORE=hf uploads to the HF dataset (needs HF_TOKEN, HF_REPO)
$PY -m exp.run --list            # all jobs with a one-line description
$PY -m exp.run --job B1          # each job module's header documents its options
```

Each job runs a pre-flight check (RAM, disk) before it starts. Finished units are appended to progress files under `$RESULTS_DIR/progress/<job>/`, so a job restarted on a fresh machine skips the units it has already done.

**Where results land.**
- Raw outputs are written as parquet shards to `$RESULTS_DIR/runs/<job>/<checkpoint>/<config_hash>/shard_XXX.parquet`, or to the same layout in the Hugging Face dataset when `STORE=hf`. Each shard records the git SHA, the model repo and commit, the full config and the package versions. Existing shards are never overwritten. To upload local shards later, run `$PY -m exp.sync --from local --to hf`.
- The raw outputs of the reported runs are in a Hugging Face dataset, `<owner>/rlvr-se-runs`. Set `HF_REPO` to point at it.
- Summaries are produced from those shards by `python -m exp.analysis.<job>_summary` (for example `b2_summary`). The scripts download the shards into `results/sprint/_hf/`, a local cache that is not committed, and write `results/sprint/<job>/summary.md`.

**Pilot.** The frozen pilot was run with `scripts/full_run.py` and analysed with `scripts/analyze.py` under `requirements_pinned.txt` and `requirements_analysis.txt`. Its plan is [`docs/analysis-plan.md`](docs/analysis-plan.md), and its outputs are in `results/full_run_*` and `results/analysis/`. The original notebook workflow for the v1-v3 baseline is described in [`docs/history/README_v1.md`](docs/history/README_v1.md).

## Jobs

A is the indeterminate self-report set (48 Likert wordings), and D is the determinate multiple-choice set. The main contrast is RLVR − DPO.

| job | what it tests | summary |
|---|---|---|
| B0 | checkpoint inventory: refs, SHAs, sizes, chat templates, injected system prompts (no GPU) | [`docs/sprint/checkpoint_inventory.md`](docs/sprint/checkpoint_inventory.md) |
| B1 | Likert ×20 seeds plus think/no-think answer readouts: is RLVR more extreme, and does it acquiesce less? | [`results/sprint/B1/b1_summary.md`](results/sprint/B1/b1_summary.md) |
| B2 | 2×2 indeterminacy × self-reference grid: is the extra length driven by indeterminacy? | [`results/sprint/B2/summary.md`](results/sprint/B2/summary.md) |
| B3 | reasoning cap × answer format factorial on A and D (runs B3a and B3b) | [`results/sprint/B3a/summary.md`](results/sprint/B3a/summary.md) |
| B4 | reward domain: base model and RL-Zero Math/Code/IF/General (Mix cut) | [`results/sprint/B4/summary.md`](results/sprint/B4/summary.md) |
| B5 | dose-response over RL step checkpoints (length and P(extreme) vs step) | [`results/sprint/B5/summary.md`](results/sprint/B5/summary.md) |
| B6 | hard controls: MATH-500 L5/AIME positive control, MMLU-Pro calibration, SQuAD v2 refusal (sycophancy cut) | [`results/sprint/B6/summary.md`](results/sprint/B6/summary.md) |
| B7 | self-report vs behaviour: hedging on 100 TriviaQA questions (descriptive) | [`results/sprint/B7/summary.md`](results/sprint/B7/summary.md) |
| B8 | response style on a validated instrument (IPIP-NEO-120) | [`results/sprint/B8/summary.md`](results/sprint/B8/summary.md) |
| M1 | weight-difference spectra, RLVR − DPO against DPO − SFT | [`results/sprint/M1/README.md`](results/sprint/M1/README.md) |
| M1b | M1 follow-up: bf16 changed share, shuffle null, cos(ΔW, W) (exploratory) | [`results/sprint/M1b/summary.md`](results/sprint/M1b/summary.md) |
| M2 | teacher-forced scoring of the frozen traces: stopping, switch tokens, KL | [`results/sprint/M2/m2_summary.md`](results/sprint/M2/m2_summary.md) |
| M3 | when the Likert answer settles in the trace; probes at prompt end | [`results/sprint/M3/summary.md`](results/sprint/M3/summary.md) |
| M4 | persona vectors: trait directions per checkpoint and projections on A prompts | [`results/sprint/M4/summary.md`](results/sprint/M4/summary.md) |
| M5 | causal tests: offset steering, backtracking-direction ablation, prompt patching | [`results/sprint/M5/summary.md`](results/sprint/M5/summary.md) |
| M6 | linear CKA per layer, DPO↔RLVR vs SFT↔DPO on A/B2/D (point estimates only) | [`results/sprint/M6/summary.md`](results/sprint/M6/summary.md) |
| PH | prefix handoff: RLVR/DPO trace prefixes continued by the other model (F3, F3b) | [`results/sprint/M2/gate1_b2.md`](results/sprint/M2/gate1_b2.md), [`docs/sprint/gate1-2026-10-29.md`](docs/sprint/gate1-2026-10-29.md) |

## Registration, amendments and disclosures

- **Sprint registration:** [`docs/sprint/registration-2026-10-27.md`](docs/sprint/registration-2026-10-27.md). Amendments 1-17 are dated sections at the end of the same file.
- **Disclosure ledger:** [`docs/sprint/disclosure-ledger.md`](docs/sprint/disclosure-ledger.md). It records every departure from the storage rules and every repair of a stored result. The SFT system-prompt confound is described in [`docs/sprint/disclosure-sft-system-prompt.md`](docs/sprint/disclosure-sft-system-prompt.md).
- **Gates:** [`docs/sprint/gate1-2026-10-29.md`](docs/sprint/gate1-2026-10-29.md) (Gate 1 and the exploratory F1-F3b analyses) and [`docs/sprint/gate2.md`](docs/sprint/gate2.md) (final verdicts).
- **Pilot pre-registration:** [`docs/analysis-plan.md`](docs/analysis-plan.md) (tag `analysis-plan-2026-10-24`), [`docs/predictions.md`](docs/predictions.md), and the post-hoc [`docs/exploratory-registration-2026-10-25.md`](docs/exploratory-registration-2026-10-25.md).

## Checkpoints

| stage | Hugging Face repo |
|---|---|
| Base | `allenai/Olmo-3-1025-7B` |
| SFT | `allenai/Olmo-3-7B-Think-SFT` |
| DPO | `allenai/Olmo-3-7B-Think-DPO` |
| RLVR (final) | `allenai/Olmo-3-7B-Think` |
| RL directly from base | `allenai/Olmo-3-7B-RL-Zero-{Math,Code,IF,General,Mix}` |

The Hub SHAs at registration are in [`docs/sprint/checkpoint_inventory.md`](docs/sprint/checkpoint_inventory.md). Each output shard records the exact commit it used.

## History

These documents describe only the early single-checkpoint pilot (v1-v3 baseline). They are kept for the record and are superseded by the full run and the sprint.

- [`docs/history/README_v1.md`](docs/history/README_v1.md): the previous README, which describes the v1-v3 baseline and the notebook workflow.
- [`docs/history/RESULTS.md`](docs/history/RESULTS.md): findings on the RLVR checkpoint alone, from before any DPO or SFT comparison.
- [`docs/history/VARIANCE.md`](docs/history/VARIANCE.md): version history (v1 → v3) and the sampling-variance findings of the baseline.
- [`docs/history/probe_list.md`](docs/history/probe_list.md): a readable list of the 20 v3 probe items; the source of truth is `probes/`.
