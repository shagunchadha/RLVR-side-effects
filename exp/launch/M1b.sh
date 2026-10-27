#!/usr/bin/env bash
# M1b on a GPU notebook: streams the three checkpoints one safetensors file at a time (work dir deleted at the end).
#   cd /tmp/rse && nohup bash exp/launch/M1b.sh > /tmp/m1b.log 2>&1; echo EXIT=$? >> /tmp/m1b.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE=local RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
echo "=== $(date +%T) M1b"
$PY -m exp.run --job M1b || { echo "FAILED: M1b"; exit 1; }
rm -rf "$RESULTS_DIR/M1b/_dl"; echo "=== $(date +%T) M1b done; work dir deleted"
