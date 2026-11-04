#!/usr/bin/env bash
# B6, RLVR (iv) refusal, on N6 (registration amendment 17); RLVR (iii) calibration runs on N2 (B6r.sh).
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   run by exp/launch/queue.sh N6 (log /tmp/b6r6.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
export B6_LANE=N6
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B6/_hf"
mkdir -p "$RESULTS_DIR/B6"
step() { echo "=== $(date +%T) B6 $*"; $PY -m exp.run --job B6 "$@" || { echo "FAILED: B6 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/B6/.smoke_ok_r6" ]; then       # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/b6_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part ref --limit 2 --seeds 1 \
    && touch "$RESULTS_DIR/B6/.smoke_ok_r6" || exit 1
fi
HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --part ref || exit 1
rm -rf "$W"; echo "=== $(date +%T) B6r6 queue done"
