#!/usr/bin/env bash
# Lane runner for one GPU notebook (never more than one job at a time).
#   nohup bash exp/launch/lane.sh <log-to-wait-for|-> <JOB> [JOB ...] > /tmp/lane.log 2>&1 &
# Waits until <log-to-wait-for> contains a line EXIT=... (the job already running here), then for each JOB runs
# exp/launch/<JOB>.sh with its log at /tmp/<job>.log (lower case) ending in EXIT=<rc>. git pull before every attempt
# picks up fixes. A failed job is retried once (all jobs resume from their progress files); after a second failure it
# is skipped and the lane moves on (logged as SKIPPED).
set -u
cd "$(dirname "$0")/../.."
WAIT="$1"; shift
if [ "$WAIT" != "-" ]; then
  echo "$(date +%T) lane: waiting for EXIT= in $WAIT"
  until grep -q '^EXIT=' "$WAIT" 2>/dev/null; do sleep 30; done
  echo "$(date +%T) lane: $WAIT ended: $(grep '^EXIT=' "$WAIT" | tail -1)"
fi
for JOB in "$@"; do
  LOG="/tmp/$(echo "$JOB" | tr 'A-Z' 'a-z').log"
  for attempt in 1 2; do
    git pull -q || echo "$(date +%T) lane: git pull failed (continuing with the checked-out code)"
    if [ ! -f "exp/launch/$JOB.sh" ]; then
      echo "$(date +%T) lane: $JOB has no launcher (exp/launch/$JOB.sh); SKIPPED"; break
    fi
    [ -f "$LOG" ] && mv "$LOG" "$LOG.attempt$((attempt - 1))"
    echo "$(date +%T) lane: $JOB attempt $attempt"
    bash "exp/launch/$JOB.sh" > "$LOG" 2>&1; RC=$?; echo "EXIT=$RC" >> "$LOG"
    if [ $RC -eq 0 ]; then echo "$(date +%T) lane: $JOB done"; break; fi
    echo "$(date +%T) lane: $JOB attempt $attempt failed (rc=$RC): $(grep -E 'FAILED|Error' "$LOG" | tail -1 | cut -c1-200)"
    [ $attempt -eq 2 ] && echo "$(date +%T) lane: $JOB failed twice; SKIPPED"
  done
done
echo "$(date +%T) lane: queue empty"
echo "EXIT=0"
