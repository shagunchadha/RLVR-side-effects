#!/usr/bin/env bash
# B3 for RLVR and DPO (lane N6); the SFT arm runs on the other lane (B3b).
B3_CKPTS="rlvr dpo" exec bash "$(dirname "$0")/B3.sh"
