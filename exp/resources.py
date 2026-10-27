"""Resource safety shared by the sprint jobs: pre-flight check, periodic RES lines, per-item progress files.

  preflight(need_disk_gb, dirs)  print free RAM, cgroup limit, free disk, GPU memory; abort if free disk < need_disk_gb
                                 or available RAM < MIN_RAM_GB. need_disk_gb = 2 x the model size the job downloads.
  ResLog(every_s)                .maybe(note) prints a RES line when every_s seconds have passed (and .now()).
  Progress(path, cfg_hash)       append-only CSV, one row per finished unit (key + JSON payload), fsynced; a torn last
                                 line is dropped on load; rows of another config abort. Jobs skip keys already in it.

On molab (gVisor) the root filesystem reports ~8 EB free, so the disk check cannot see a real limit there; the
cgroup memory counter is reported but does not move with file writes either. Both are logged anyway.
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

MIN_RAM_GB = 16
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def _meminfo():
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) * 1024
        return d
    except OSError:
        return {}


def _read_int(path):
    try:
        v = open(path).read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def _rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):           # the HF cache links snapshots to blobs; count each file once
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def snapshot(dirs=(), gpu=True):
    """dict of resource figures in bytes (None where unavailable)."""
    mi = _meminfo()
    s = {"ram_avail": mi.get("MemAvailable"), "ram_total": mi.get("MemTotal"), "rss": _rss(),
         "cgroup": _read_int("/sys/fs/cgroup/memory.current") or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
         "cgroup_max": _read_int("/sys/fs/cgroup/memory.max") or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")}
    for d in dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        s[f"du:{d.name}"] = du(d)
        s[f"free:{d.name}"] = shutil.disk_usage(d).free
    if gpu:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                s.update({"gpu_free": free, "gpu_total": total, "gpu_alloc": torch.cuda.memory_allocated(),
                          "gpu_peak": torch.cuda.max_memory_allocated()})
        except ImportError:
            pass
    return s


def fmt(s):
    gb = lambda x: "na" if x is None else f"{x / 1e9:.1f}"
    return " ".join(f"{k}={gb(v)}" for k, v in s.items())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def preflight(need_disk_gb, dirs=(), min_ram_gb=MIN_RAM_GB, gpu=True):
    s = snapshot(dirs, gpu)
    log("PREFLIGHT", fmt(s), f"| need disk {need_disk_gb:.0f} GB, RAM {min_ram_gb} GB")
    bad = []
    if s["ram_avail"] is not None and s["ram_avail"] < min_ram_gb * 1e9:
        bad.append(f"available RAM {s['ram_avail'] / 1e9:.1f} GB < {min_ram_gb} GB")
    for k, v in s.items():
        if k.startswith("free:") and v < need_disk_gb * 1e9:
            bad.append(f"free disk at {k[5:]} {v / 1e9:.1f} GB < {need_disk_gb:.0f} GB")
    if bad:
        raise SystemExit("PREFLIGHT ABORT: " + "; ".join(bad))
    return s


class ResLog:
    def __init__(self, every_s=180, dirs=(), gpu=True):
        self.every, self.dirs, self.gpu, self.t = every_s, dirs, gpu, time.time()

    def now(self, note=""):
        log("RES", note, "|", fmt(snapshot(self.dirs, self.gpu)))
        self.t = time.time()

    def maybe(self, note=""):
        if time.time() - self.t >= self.every:
            self.now(note)


class Progress:
    COLS = ["config_hash", "key", "seconds", "row_json"]

    def __init__(self, path, cfg_hash):
        self.path, self.hash = Path(path), cfg_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = self._load()

    def _load(self):
        if not self.path.exists():
            return {}
        done, good, bad = {}, [], 0
        with open(self.path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    row = json.loads(r["row_json"])
                    float(r["seconds"])
                except (TypeError, ValueError, KeyError):
                    bad += 1
                    continue
                if r["config_hash"] != self.hash:
                    raise SystemExit(f"{self.path} holds rows of config {r['config_hash']}, not {self.hash}")
                done[r["key"]] = row
                good.append(r)
        if bad:
            log(f"progress: dropped {bad} unreadable row(s) (torn write) from {self.path}")
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(good)
            os.replace(tmp, self.path)
        return done

    def add(self, key, row, seconds):
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            if new:
                w.writeheader()
            w.writerow({"config_hash": self.hash, "key": key, "seconds": f"{seconds:.2f}", "row_json": json.dumps(row)})
            f.flush()
            os.fsync(f.fileno())
        self.done[key] = row
