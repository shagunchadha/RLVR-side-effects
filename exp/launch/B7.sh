#!/usr/bin/env bash
# B7 queue for one GPU notebook (amendment 9): RLVR own, DPO own, SFT with the DPO system prompt. Needs B6 (iii) for the analysis.
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/b2.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B7/_hf"
mkdir -p "$RESULTS_DIR/B7"
step() { echo "=== $(date +%T) B7 $*"; $PY -m exp.run --job B7 "$@" || { echo "FAILED: B7 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/B7/.smoke_ok" ]; then         # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/b7_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 2 --seeds 1 && touch "$RESULTS_DIR/B7/.smoke_ok" || exit 1
fi
HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr || exit 1; rm -rf "$W/rlvr"
HF_HUB_CACHE="$W/dpo" step --ckpt dpo || exit 1; rm -rf "$W/dpo"
HF_HUB_CACHE="$W/sft" step --ckpt sft --system dpo_default || exit 1
rm -rf "$W"; echo "=== $(date +%T) B7 queue done"
