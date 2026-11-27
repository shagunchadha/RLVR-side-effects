#!/usr/bin/env bash
# F0b on N6 (registration amendment 18): DPO b1 (B1-H1 validation arm).
# Fast path (harness.FAST_ENGINE + generate_stream); every finished generation goes to a progress file, so
# re-running this script after a reclaim resumes. Outputs: eacl_revision_1/F0b_B1/ and eacl_revision_1/F0b/.
#   run by exp/launch/lane.sh (log /tmp/f0b_n6.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # as in the submitted B1 (exp/launch/B1.sh); recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/eacl_revision_1/_hf"
mkdir -p "$RESULTS_DIR/eacl_revision_1"
step() { echo "=== $(date +%T) F0b $*"; $PY -m exp.run --job F0b "$@" || { echo "FAILED: F0b $*"; exit 1; }; }
for cp in dpo:b1; do
  ck="${cp%%:*}"; part="${cp##*:}"
  if [ ! -f "$RESULTS_DIR/eacl_revision_1/.smoke_ok_f0b_${ck}_${part}" ]; then   # throwaway dir, always local
    STORE=local RESULTS_DIR=/tmp/f0b_smoke HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --part "$part" --limit 1 --seeds 2       && touch "$RESULTS_DIR/eacl_revision_1/.smoke_ok_f0b_${ck}_${part}" || exit 1
  fi
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --part "$part" || exit 1
done
rm -rf "$W"; echo "=== $(date +%T) F0b_N6 done"
