#!/usr/bin/env bash
# Queue runner for one GPU notebook (amendment 5): one job at a time, in the order of exp/launch/queues/<LANE>.txt,
# re-read (after git pull) before every job, so the order can be changed or jobs cut by a commit.
#   bash exp/launch/queue.sh N2        (normally entered from the old lane via B3b.sh / B3a.sh)
# A job is run by exp/launch/<JOB>.sh with its log at /tmp/<job>.log ending in EXIT=<rc>; retried once (all jobs resume
# from their progress files). Finished / failed-twice jobs are recorded in $RESULTS_DIR/_queue/<LANE>/ (mirrored to HF
# by exp/hfsync.py and pulled back at the start of a relaunched notebook, so a relaunch resumes at the right job).
# Strict order: if the next job has no launcher yet, the runner waits 5 min and re-reads (it never skips ahead); an
# empty queue also waits; it never returns to the old lane. Relaunch: hfsync run ... -- bash exp/launch/queue.sh <LANE>
set -u
cd "$(dirname "$0")/../.."
LANE="$1"
export QUEUE_ACTIVE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=1
ST="${RESULTS_DIR:-/tmp/sprint_out}/_queue/$LANE"; mkdir -p "$ST"
while true; do
  git pull -q || echo "$(date +%T) queue: git pull failed (continuing with the checked-out code)"
  QF="exp/launch/queues/$LANE.txt"
  next=""; pending=0
  for JOB in $(grep -v '^\s*#' "$QF" 2>/dev/null); do
    [ -f "$ST/$JOB.done" ] || [ -f "$ST/$JOB.skipped" ] && continue
    pending=$((pending + 1))
    if [ -f "exp/launch/$JOB.sh" ]; then next="$JOB"; fi
    break                                    # strict order: only the first pending job is a candidate
  done
  if [ -z "$next" ]; then
    # an empty queue does not return to the old lane's stale job list: it waits for jobs added to the queue file
    if [ "$pending" -eq 0 ]; then echo "$(date +%T) queue $LANE: empty; re-reading in 5 min"; sleep 300; continue; fi
    echo "$(date +%T) queue $LANE: next job has no launcher yet; waiting 5 min"; sleep 300; continue
  fi
  LOG="/tmp/$(echo "$next" | tr 'A-Z' 'a-z').log"
  ok=0
  for attempt in 1 2; do
    [ -f "$LOG" ] && mv "$LOG" "$LOG.prev$attempt"
    echo "$(date +%T) queue $LANE: $next attempt $attempt"
    bash "exp/launch/$next.sh" > "$LOG" 2>&1; RC=$?; echo "EXIT=$RC" >> "$LOG"
    if [ $RC -eq 0 ]; then ok=1; break; fi
    echo "$(date +%T) queue $LANE: $next attempt $attempt failed (rc=$RC): $(grep -E 'FAILED|Error' "$LOG" | tail -1 | cut -c1-200)"
    git pull -q || true
  done
  if [ $ok -eq 1 ]; then touch "$ST/$next.done"; echo "$(date +%T) queue $LANE: $next done"
  else touch "$ST/$next.skipped"; echo "$(date +%T) queue $LANE: $next failed twice; SKIPPED"; fi
done
