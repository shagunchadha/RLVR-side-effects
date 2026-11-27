#!/usr/bin/env bash
# Engine probe (F0b/F1 throughput diagnosis, Nov 28): RLVR at max_model_len 32768, engine variants "fast" (prefix
# caching on, the current F1 engine) and "fast_nopc" (prefix caching off), 10 minutes of the F1 AIME load each.
# Diagnosis only: writes $RESULTS_DIR/eacl_revision_1/engine_probe/ (mirrored to HF as progress), no F-job output.
#   run by exp/launch/lane.sh (log /tmp/probe.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # as in F0b and F1
export HF_HUB_CACHE="$RESULTS_DIR/eacl_revision_1/_hf/rlvr"
env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python -m exp.revision.engine_probe --ckpt rlvr \
  --variants fast fast_nopc --minutes 10 --max-model-len 32768 || { echo "FAILED: engine probe"; exit 1; }
echo "=== $(date +%T) PROBE done"
