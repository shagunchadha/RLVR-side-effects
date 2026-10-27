#!/usr/bin/env bash
# B3 for SFT (lane N2, after B1); RLVR and DPO run on the other lane (B3a).
B3_CKPTS="sft" exec bash "$(dirname "$0")/B3.sh"
