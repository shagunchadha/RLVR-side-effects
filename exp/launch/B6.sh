#!/usr/bin/env bash
# B6 queue for one GPU notebook (amendments 3, 11). The positive control (part pos, RLVR then DPO) runs first.
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/b2.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B6/_hf"
mkdir -p "$RESULTS_DIR/B6"
step() { echo "=== $(date +%T) B6 $*"; $PY -m exp.run --job B6 "$@" || { echo "FAILED: B6 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/B6/.smoke_ok" ]; then         # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/b6_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part pos --limit 1 --seeds 1 --max-tokens 512   && STORE=local RESULTS_DIR=/tmp/b6_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part cal --limit 2 --seeds 1   && touch "$RESULTS_DIR/B6/.smoke_ok" || exit 1
fi
HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part pos || exit 1                     # (i) positive control first
rm -rf "$W/rlvr"; echo "=== $(date +%T) deleted weights of rlvr"
for part in pos cal ref sycpre syc; do HF_HUB_CACHE="$W/dpo" step --ckpt dpo --part "$part" || exit 1; done
rm -rf "$W/dpo"; echo "=== $(date +%T) deleted weights of dpo"
for part in cal ref syc; do HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part "$part" || exit 1; done
rm -rf "$W"; echo "=== $(date +%T) B6 queue done"
