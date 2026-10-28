#!/usr/bin/env bash
# Amendment 5: when the old lane reaches this job, hand over to the queue runner (new lane order, N6).
if [ -z "${QUEUE_ACTIVE:-}" ] && [ -f "$(dirname "$0")/queues/N6.txt" ]; then
  exec bash "$(dirname "$0")/queue.sh" N6
fi
# B3 for RLVR and DPO (lane N6); the SFT arm runs on the other lane (B3b).
B3_CKPTS="rlvr dpo" exec bash "$(dirname "$0")/B3.sh"
