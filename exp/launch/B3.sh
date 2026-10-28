#!/usr/bin/env bash
# B3 queue for one GPU notebook (registration B3; amendment 1: model-default sampling, own templates only).
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   cd /tmp/rse && nohup bash exp/launch/B3.sh > /tmp/b3.log 2>&1; echo EXIT=$? >> /tmp/b3.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B3/_hf"
mkdir -p "$RESULTS_DIR/B3"
step() { echo "=== $(date +%T) B3 $*"; $PY -m exp.run --job B3 "$@" || { echo "FAILED: B3 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/B3/.smoke_ok" ]; then
  RESULTS_DIR=/tmp/b3_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 1 --seeds 2 && touch "$RESULTS_DIR/B3/.smoke_ok"
fi
for ck in ${B3_CKPTS:-rlvr dpo sft}; do      # B3_CKPTS splits B3 across the two GPU lanes
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck"
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
rm -rf "$W"; echo "=== $(date +%T) B3 queue done"
