#!/usr/bin/env bash
# M6 queue for one GPU notebook (amendments 3, 13): one process, DPO, RLVR and SFT loaded one at a time.
# Activations per model are saved under /tmp/m6_work: re-running after a reclaim resumes from the models done.
#   run by exp/launch/lane.sh (log /tmp/b2.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/M6/_hf"
mkdir -p "$RESULTS_DIR/M6"
step() { echo "=== $(date +%T) M6 $*"; $PY -m exp.run --job M6 "$@" || { echo "FAILED: M6 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/M6/.smoke_ok" ]; then         # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/m6_smoke M6_WORK=/tmp/m6_smoke_work HF_HUB_CACHE="$W" step --limit 6 --think 4 --boot 20   && touch "$RESULTS_DIR/M6/.smoke_ok" || exit 1
fi
HF_HUB_CACHE="$W" step
rm -rf "$W"; echo "=== $(date +%T) M6 queue done; weights deleted"
