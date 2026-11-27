#!/usr/bin/env bash
# B4 queue for one GPU notebook (RL-Zero reward domains; amendments 3, 5, 10). B4_CKPTS cuts the list
# (cut order: rl_zero_math rl_zero_if rl_zero_general, plus base). rl_zero_mix is cut (amendment 14).
# vLLM only; every generation is appended to a progress file, so re-running this script after a reclaim resumes.
#   run by exp/launch/lane.sh (log /tmp/b2.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0      # see exp/launch/B1.sh; recorded in the job config
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B4/_hf"
mkdir -p "$RESULTS_DIR/B4"
step() { echo "=== $(date +%T) B4 $*"; $PY -m exp.run --job B4 "$@" || { echo "FAILED: B4 $*"; exit 1; }; }
if [ ! -f "$RESULTS_DIR/B4/.smoke_ok" ]; then         # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/b4_smoke HF_HUB_CACHE="$W/rl_zero_math" step --ckpt rl_zero_math --phase gen --limit 2 --seeds 1   && STORE=local RESULTS_DIR=/tmp/b4_smoke HF_HUB_CACHE="$W/rl_zero_math" step --ckpt rl_zero_math --phase read --limit 2 --seeds 1   && STORE=local RESULTS_DIR=/tmp/b4_smoke HF_HUB_CACHE="$W/base" step --ckpt base --phase gen --limit 2 --seeds 1   && touch "$RESULTS_DIR/B4/.smoke_ok" || exit 1
fi
for ck in ${B4_CKPTS:-base rl_zero_math rl_zero_code rl_zero_if rl_zero_general}; do   # rl_zero_mix cut (amendment 14)
  HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --phase gen && HF_HUB_CACHE="$W/$ck" step --ckpt "$ck" --phase read || exit 1
  rm -rf "$W/$ck"; echo "=== $(date +%T) deleted weights of $ck"
done
rm -rf "$W"; echo "=== $(date +%T) B4 queue done"
