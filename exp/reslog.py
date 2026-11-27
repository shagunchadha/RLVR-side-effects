"""Sidecar resource logger for jobs that cannot be edited (the frozen scripts/full_run.py).

    python -m exp.reslog --pid <PID> [--every 180] [--dirs /tmp/hf]

Prints a RES line (available RAM, cgroup memory, the watched process's RSS, dir sizes, free disk, GPU memory via
nvidia-smi) every --every seconds until the process exits. Standard library only, so it runs in any venv.
"""
import argparse
import os
import subprocess
import time

from . import resources as R


def rss(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return f"gpu_used={float(out[0]) / 1e3:.1f}/{float(out[1]) / 1e3:.1f}"
    except Exception:
        return "gpu=na"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--every", type=int, default=180)
    ap.add_argument("--dirs", nargs="*", default=[])
    a = ap.parse_args()
    while os.path.exists(f"/proc/{a.pid}"):
        s = R.snapshot(a.dirs, gpu=False)
        s["job_rss"] = rss(a.pid)
        R.log("RES (sidecar)", R.fmt(s), gpu())
        time.sleep(a.every)


if __name__ == "__main__":
    main()
