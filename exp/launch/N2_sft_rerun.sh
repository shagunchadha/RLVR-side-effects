#!/usr/bin/env bash
# N2: SFT matched-system-prompt rerun in the FROZEN environment (registration amendment 1, C3), gated.
#   cd /tmp/rse && nohup bash exp/launch/N2_sft_rerun.sh > /tmp/n2.log 2>&1; echo EXIT=$? >> /tmp/n2.log
# 1. pre-flight; 2. reproduction gate: frozen scripts/full_run.py, DPO, component A, --limit 16 (its batches 0-1 are
#    exactly frozen batches 0-1: seed 0, ids 0-7 and 8-15); every row must match results/full_run_dpo token for token
#    (the four seed-1/2 batches it also produces match frozen batches 6, 7, 12, 13 and are checked too);
# 3. wrapper dry run (prompt tokens = frozen DPO's); 4. the rerun. Any failure stops the script.
set -uo pipefail
cd "$(dirname "$0")/../.."
FPY="env -u PYTHONPATH -u PYTHONSAFEPATH -u VIRTUAL_ENV /tmp/batch_venv/bin/python"
OUT=/tmp/sprint_out/sft_matched_prompt
mkdir -p "$OUT" /tmp/gate /tmp/gate_m
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
sidecar() { $FPY -m exp.reslog --pid "$1" --every 180 --dirs "$HF_HOME" & }

echo "=== $(date +%T) pre-flight"
HF_HOME=/tmp/hf_n2 $FPY -c "from exp import resources as R; R.preflight(30, dirs=['/tmp/hf_n2'], gpu=False)" || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader

echo "=== $(date +%T) gate 1: frozen full_run.py, DPO, A --limit 16"
export HF_HOME=/tmp/hf_n2
$FPY scripts/full_run.py --checkpoint dpo --components A --limit 16 --out-dir /tmp/gate > /tmp/gate.log 2>&1 &
JOB=$!; sidecar $JOB; wait $JOB; RC=$?
grep -E "environment hash|RESULT|MISMATCH|FAILED|ALL BATCHES" /tmp/gate.log
[ $RC -eq 0 ] || { echo "GATE 1 FAILED: full_run.py exited $RC (version gate output above; full log /tmp/gate.log)"; exit 1; }
grep -q "environment hash: b25083c2e3494bcd" /tmp/gate.log || { echo "GATE 1 FAILED: environment hash differs from the frozen b25083c2e3494bcd"; exit 1; }
$FPY - <<'PYG' || exit 1
import json, sys
f = {(r["id"], r["seed"]): r for r in json.load(open("results/full_run_dpo/full_run_dpo.json"))["rows"]}
fb = {(r["id"], r["seed"]): r["batch_index"] for r in f.values()}
g = json.load(open("/tmp/gate/full_run_dpo_dry.json"))["rows"]
bad = [(r["id"], r["seed"], fb[(r["id"], r["seed"])]) for r in g
       if (r["raw_output"], r["n_tokens"], r["hit_eos"]) != (f[(r["id"], r["seed"])]["raw_output"], f[(r["id"], r["seed"])]["n_tokens"], f[(r["id"], r["seed"])]["hit_eos"])]
b01 = [r for r in g if fb[(r["id"], r["seed"])] in (0, 1)]
print(f"gate 1: {len(g)} rows regenerated; frozen batches 0-1: {len(b01)} rows, "
      f"{sum(1 for x in bad if x[2] in (0, 1))} differ; all batches: {len(bad)} differ {bad[:8]}")
sys.exit(1 if bad or len(b01) != 16 else 0)
PYG
echo "=== $(date +%T) GATE 1 PASSED (exact token match)"
rm -rf /tmp/hf_n2

echo "=== $(date +%T) gate 2: wrapper dry run, SFT with the DPO system prompt"
export HF_HOME=/tmp/hf_n2_sft
$FPY scripts/full_run_matched_prompt.py --checkpoint sft --system-prompt-from dpo --components A --limit 8 --out-dir /tmp/gate_m > /tmp/gate_m.log 2>&1 \
  || { echo "GATE 2 FAILED"; tail -20 /tmp/gate_m.log; exit 1; }
$FPY -c "import json,sys;d={(r['id'],r['seed']):r['prompt_tokens'] for r in json.load(open('results/full_run_dpo/full_run_dpo.json'))['rows']};g=json.load(open('/tmp/gate_m/full_run_sft_dry.json'))['rows'];ok=all(d[(r['id'],r['seed'])]==r['prompt_tokens'] for r in g);print('gate 2: prompt_tokens == frozen DPO for all', len(g), 'rows:', ok);sys.exit(0 if ok else 1)" || exit 1
echo "=== $(date +%T) GATE 2 PASSED"

echo "=== $(date +%T) rerun: SFT, DPO/RLVR system prompt, frozen pipeline -> $OUT"
$FPY scripts/full_run_matched_prompt.py --checkpoint sft --system-prompt-from dpo --out-dir "$OUT" > "$OUT/full_run_sft.log" 2>&1 &
JOB=$!; sidecar $JOB; wait $JOB; RC=$?
tail -3 "$OUT/full_run_sft.log"
[ $RC -eq 0 ] || { echo "RERUN FAILED ($RC)"; exit 1; }
rm -rf /tmp/hf_n2_sft; echo "=== $(date +%T) rerun done; weights deleted"
