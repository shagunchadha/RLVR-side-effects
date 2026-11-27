"""Sandbox memory sampler. molab's gVisor cgroup v1 exposes memory.usage_in_bytes but neither max_usage_in_bytes nor
memory.stat, so the peak is tracked here: usage sampled every --every s, the running peak kept in --peak-file (read by
resources.cgroup_mem as cg_peak) and one line per --log-every s appended to --log.

  python -m exp.memwatch [--every 2] [--log /tmp/memwatch.log] [--peak-file /tmp/mem_peak]      (runs until killed)
  python -m exp.memwatch --report                                                               (print current + peak)
"""
import argparse
import os
import time

USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
PEAK_FILE = os.environ.get("SPRINT_MEM_PEAK_FILE", "/tmp/mem_peak")


def usage():
    try:
        return int(open(USAGE).read())
    except (OSError, ValueError):
        return None


def read_peak(path=PEAK_FILE):
    try:
        return int(open(path).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--log-every", type=float, default=60.0)
    ap.add_argument("--log", default="/tmp/memwatch.log")
    ap.add_argument("--peak-file", default=PEAK_FILE)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    g = lambda x: "na" if x is None else f"{x / 2 ** 30:.2f}"  # noqa: E731
    if a.report:
        print(f"cg_use={g(usage())} cg_peak={g(read_peak(a.peak_file))} GiB")
        return 0
    peak, t_log = read_peak(a.peak_file) or 0, 0.0
    while True:
        u = usage()
        if u is not None and u > peak:
            peak = u
            tmp = a.peak_file + ".tmp"
            with open(tmp, "w") as f:
                f.write(f"{peak} {time.strftime('%H:%M:%S')}\n")
            os.replace(tmp, a.peak_file)
        if time.time() - t_log >= a.log_every:
            with open(a.log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} MEM cg_use={g(u)} cg_peak={g(peak)} GiB\n")
            t_log = time.time()
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
