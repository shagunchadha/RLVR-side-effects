#!/usr/bin/env bash
# M2 queue for one GPU notebook. One model per process (the pair step loads exactly two); each model's weights go to
# their own dir under $RESULTS_DIR/M2/_hf (skipped by exp/sync_remote.py) and are deleted before the next model.
# Every step resumes from its progress file, so re-running this script after a reclaim continues where it stopped.
#   cd /tmp/rse && nohup bash exp/launch/M2.sh > /tmp/m2.log 2>&1; echo EXIT=$? >> /tmp/m2.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE=local RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
# molab sets CUDA_HOME to the image's nvcc 13.3; flashinfer JIT must use the venv's pinned 13.4 toolkit (header mismatch otherwise)
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/M2/_hf"
mkdir -p "$RESULTS_DIR/M2"
step() { echo "=== $(date +%T) M2 $*"; $PY -m exp.run --job M2 "$@" || { echo "FAILED: M2 $*"; exit 1; }; }

if [ ! -f "$RESULTS_DIR/M2/.smoke_ok" ]; then        # small test in a throwaway results dir (no real shard written)
  RESULTS_DIR=/tmp/m2_smoke HF_HUB_CACHE="$W/rlvr" step --ckpt rlvr --limit 3 && touch "$RESULTS_DIR/M2/.smoke_ok"
fi
for ck in rlvr dpo sft; do
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck"
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
HF_HUB_CACHE="$W/pair" step --pair rlvr dpo
rm -rf "$W"; echo "=== $(date +%T) M2 queue done; weights deleted"
