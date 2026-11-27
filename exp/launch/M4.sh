#!/usr/bin/env bash
# M4 queue for one GPU notebook (amendments 3, 12): DPO first (its directions are the reference), then RLVR, SFT.
# Directions are saved per checkpoint and projections have a progress file: re-running after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/m3.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/M4/_hf"
mkdir -p "$RESULTS_DIR/M4"
step() { echo "=== $(date +%T) M4 $*"; $PY -m exp.run --job M4 "$@" || { echo "FAILED: M4 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/M4/.smoke_ok" ]; then         # throwaway dir, always local; B1 traces read from the real dir
  STORE=local M3_B1_ROOT="$RESULTS_DIR" RESULTS_DIR=/tmp/m4_smoke HF_HUB_CACHE="$W/dpo" step --ckpt dpo --limit 2 --new-tokens 8 --seeds 1   && touch "$RESULTS_DIR/M4/.smoke_ok" || exit 1
fi
for ck in dpo rlvr sft; do
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" || exit 1
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
rm -rf "$W"; echo "=== $(date +%T) M4 queue done"
