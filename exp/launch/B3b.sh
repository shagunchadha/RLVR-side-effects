#!/usr/bin/env bash
# Amendment 5: when the old lane reaches this job, hand over to the queue runner (new lane order, N2).
if [ -z "${QUEUE_ACTIVE:-}" ] && [ -f "$(dirname "$0")/queues/N2.txt" ]; then
  exec bash "$(dirname "$0")/queue.sh" N2
fi
# B3 for SFT (lane N2, after B1); RLVR and DPO run on the other lane (B3a).
B3_CKPTS="sft" exec bash "$(dirname "$0")/B3.sh"
