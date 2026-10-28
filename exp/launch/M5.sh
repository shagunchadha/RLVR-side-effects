#!/usr/bin/env bash
# M5 queue for one GPU notebook (amendments 3, 5, 7): one process, DPO and RLVR loaded one at a time (HF + hooks).
# Per-condition progress file and saved activations (/tmp/m5_work): re-running after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/b2.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/M5/_hf"
mkdir -p "$RESULTS_DIR/M5"
step() { echo "=== $(date +%T) M5 $*"; $PY -m exp.run --job M5 "$@" || { echo "FAILED: M5 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/M5/.smoke_ok" ]; then         # throwaway dir, always local; B1 traces read from the real dir
  STORE=local M3_B1_ROOT="$RESULTS_DIR" RESULTS_DIR=/tmp/m5_smoke M5_WORK=/tmp/m5_smoke_work HF_HUB_CACHE="$W"     step --limit 2 --cap 64 --patch-layers 19 && touch "$RESULTS_DIR/M5/.smoke_ok"
fi
HF_HUB_CACHE="$W" step
rm -rf "$W"; echo "=== $(date +%T) M5 queue done; weights deleted"
