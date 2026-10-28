#!/usr/bin/env bash
# B5 queue for one GPU notebook (dose-response over RLVR step branches; amendments 3, 5, 8).
# B5_STEPS overrides the step list (cut order: 5 steps = step_0025 step_0075 step_0175 step_0500 step_1375). Each arm = gen phase (vLLM alone) then read phase
# (HF alone), separate processes; both resume from their progress files, so re-running this script after a reclaim
# continues. Arms grouped by checkpoint so each model is downloaded once; its weight dir is deleted afterwards.
#   cd /tmp/rse && nohup bash exp/launch/B1.sh > /tmp/b1.log 2>&1; echo EXIT=$? >> /tmp/b1.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export STORE="${STORE:-local}" RESULTS_DIR="${RESULTS_DIR:-/tmp/sprint_out}" HF_HOME=/tmp/hf HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH="/tmp/sprint_venv/bin:$PATH"   # vLLM JIT needs ninja from the venv
# molab sets CUDA_HOME to the image's nvcc 13.3; flashinfer JIT must use the venv's pinned 13.4 toolkit (header mismatch otherwise)
export CUDA_HOME=/tmp/sprint_venv/lib/python3.13/site-packages/nvidia/cu13
# The venv's pinned CUDA pieces (runtime headers 13.0, nvcc 13.4, flashinfer's CCCL) cannot JIT-compile flashinfer's
# sampling kernels; vLLM's PyTorch top-p/top-k sampler is used instead (same sampling parameters; recorded in the config).
export VLLM_USE_FLASHINFER_SAMPLER=0
PY="env -u PYTHONPATH -u PYTHONSAFEPATH /tmp/sprint_venv/bin/python"
W="$RESULTS_DIR/B5/_hf"
mkdir -p "$RESULTS_DIR/B5"
step() { echo "=== $(date +%T) B5 $*"; $PY -m exp.run --job B5 "$@" || { echo "FAILED: B5 $*"; exit 1; }; }
arm() {   # arm <branch>: A gen (vLLM alone), A read (HF alone), D gen (vLLM alone)
  step --ckpt "rlvr@$1" --part A --phase gen && step --ckpt "rlvr@$1" --part A --phase read && step --ckpt "rlvr@$1" --part D
}
if [ ! -f "$RESULTS_DIR/B5/.smoke_ok" ]; then         # throwaway dir, always local
  STORE=local RESULTS_DIR=/tmp/b5_smoke HF_HUB_CACHE="$W/step_0025" step --ckpt rlvr@step_0025 --part A --phase gen --limit 1 --seeds 2   && STORE=local RESULTS_DIR=/tmp/b5_smoke HF_HUB_CACHE="$W/step_0025" step --ckpt rlvr@step_0025 --part A --phase read --limit 1 --seeds 2   && STORE=local RESULTS_DIR=/tmp/b5_smoke HF_HUB_CACHE="$W/step_0025" step --ckpt rlvr@step_0025 --part D --limit 1 --seeds 2   && touch "$RESULTS_DIR/B5/.smoke_ok" || exit 1
fi
[ -f "$RESULTS_DIR/B5/verify_last.json" ] || step --verify-last          # step_1375 vs main, tensor by tensor
for b in ${B5_STEPS:-step_0025 step_0050 step_0075 step_0150 step_0250 step_0425 step_0775 step_1375}; do
  HF_HUB_CACHE="$W/$b" arm "$b" || exit 1
  rm -rf "$W/$b"; echo "=== $(date +%T) deleted weights of $b"
done
rm -rf "$W" /tmp/b5_dl; echo "=== $(date +%T) B5 queue done"
