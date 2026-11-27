#!/usr/bin/env bash
# PH (prefix handoff) queue for one GPU notebook (registration amendment 4, F3); --ckpt = continuing model.
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/ph.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/PH/_hf"
mkdir -p "$RESULTS_DIR/PH"
step() { echo "=== $(date +%T) PH $*"; $PY -m exp.run --job PH "$@" || { echo "FAILED: PH $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/PH/.smoke_ok" ]; then
  STORE=local RESULTS_DIR=/tmp/ph_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 2 --seeds 1 && touch "$RESULTS_DIR/PH/.smoke_ok"
fi
for ck in ${PH_CKPTS:-rlvr dpo}; do
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck"
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
rm -rf "$W"; echo "=== $(date +%T) PH queue done"
