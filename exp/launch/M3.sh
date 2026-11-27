#!/usr/bin/env bash
# M3 queue for one GPU notebook (registration amendment 3): settle point + probes on B1 main-arm traces; HF model alone.
# Per-trace progress file; re-running this script after a reclaim resumes. Needs B1's main arms (pulled from HF).
#   run by exp/launch/lane.sh (log /tmp/m3.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/M3/_hf"
mkdir -p "$RESULTS_DIR/M3"
step() { echo "=== $(date +%T) M3 $*"; $PY -m exp.run --job M3 "$@" || { echo "FAILED: M3 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/M3/.smoke_ok" ]; then
  STORE=local M3_B1_ROOT="$RESULTS_DIR" RESULTS_DIR=/tmp/m3_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 3 && STORE=local RESULTS_DIR=/tmp/m3_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 2 --d-traces && touch "$RESULTS_DIR/M3/.smoke_ok"
fi
for ck in ${M3_CKPTS:-rlvr dpo sft}; do
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck"                                    # A: B1 main-arm traces (+ probes)
  case "$ck" in rlvr|dpo) HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --d-traces;; esac   # amendment 5: D analogue
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
rm -rf "$W"; echo "=== $(date +%T) M3 queue done"
