#!/usr/bin/env bash
# B1 queue for one GPU notebook (registration B1 + amendment 1). Each arm = gen phase (vLLM alone) then read phase
# (HF alone), separate processes; both resume from their progress files, so re-running this script after a reclaim
# continues. Arms grouped by checkpoint so each model is downloaded once; its weight dir is deleted afterwards.
#   cd /tmp/rse && nohup bash exp/launch/B1.sh > /tmp/b1.log 2>&1; echo EXIT=$? >> /tmp/b1.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE=local RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B1/_hf"
mkdir -p "$RESULTS_DIR/B1"
arm() {   # arm <ckpt> <extra args...>
  for ph in gen read; do
    echo "=== $(date +%T) B1 --ckpt $* --phase $ph"
    $PY -m exp.run --job B1 --ckpt "$@" --phase "$ph" || { echo "FAILED: B1 --ckpt $* --phase $ph"; exit 1; }
  done
}
if [ ! -f "$RESULTS_DIR/B1/.smoke_ok" ]; then         # small test in a throwaway results dir
  RESULTS_DIR=/tmp/b1_smoke HF_HUB_CACHE="$W/rlvr" arm rlvr --limit 1 --seeds 2 && touch "$RESULTS_DIR/B1/.smoke_ok"
fi
BRIDGE="--seeds 5 --temperature 0.7 --top-p 0.9"
HF_HUB_CACHE="$W/rlvr" arm rlvr
HF_HUB_CACHE="$W/rlvr" arm rlvr $BRIDGE
rm -rf "$W/rlvr"; echo "=== $(date +%T) deleted weights of rlvr"
HF_HUB_CACHE="$W/dpo" arm dpo
HF_HUB_CACHE="$W/dpo" arm dpo $BRIDGE
rm -rf "$W/dpo"; echo "=== $(date +%T) deleted weights of dpo"
HF_HUB_CACHE="$W/sft" arm sft
HF_HUB_CACHE="$W/sft" arm sft --system dpo_default
HF_HUB_CACHE="$W/sft" arm sft $BRIDGE
rm -rf "$W"; echo "=== $(date +%T) B1 queue done; weights deleted"
