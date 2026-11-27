#!/usr/bin/env bash
# F1 on N6 (registration amendment 18): the units in exp/launch/queues/F1_N6.txt, top to bottom.
# The queue is re-read after a git pull before every unit (rebalancing between lanes without a restart). A unit whose
# shard exists returns at once; an unfinished one resumes from its progress files. One smoke per checkpoint (small
# caps, so the continuation path runs too) before that checkpoint's first unit. Weights live in a per-unit directory
# that is deleted when the unit ends (operations.md); a failed unit keeps them for the lane's one retry.
#   run by exp/launch/lane.sh (log /tmp/f1_n6.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # as in the submitted B1 and F0b; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/eacl_revision_1/_hf"; MARK="$RESULTS_DIR/eacl_revision_1/.f1_N6"
mkdir -p "$W" "$MARK"
step() { echo "=== $(date +%T) F1 $*"; $PY -m exp.run --job F1 "$@" || { echo "FAILED: F1 $*"; exit 1; }; }
while :; do
  git pull -q --ff-only || echo "$(date +%T) F1_N6: git pull failed (continuing with the checked-out queue)"
  next=""
  for u in $(grep -v '^#' exp/launch/queues/F1_N6.txt | tr -d '\r'); do [ -f "$MARK/done_${u/:/_}" ] || { next="$u"; break; }; done
  [ -z "$next" ] && break
  ck="${next%%:*}"; bench="${next##*:}"
  for d in "$W"/*; do [ -e "$d" ] && [ "$(basename "$d")" != "$ck" ] && rm -rf "$d"; done   # one model on disk
  if [ ! -f "$MARK/smoke_$ck" ]; then                      # throwaway dir, always local
    rm -rf /tmp/f1_smoke
    STORE=local RESULTS_DIR=/tmp/f1_smoke HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --bench "$bench" --limit 1 --samples 2       --cap1 256 --cap2 2048 && touch "$MARK/smoke_$ck" || exit 1
  fi
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --bench "$bench" && touch "$MARK/done_${next/:/_}" || exit 1
  rm -rf "$W/$ck"                                              # per-job weights dir, deleted when the job ends
done
rm -rf "$W"; echo "=== $(date +%T) F1_N6 queue done"
