"""Resource safety shared by the sprint jobs: pre-flight check, periodic RES lines, per-item progress files.

  preflight(need_disk_gb, dirs)  print free RAM, cgroup limit, free disk, GPU memory; abort if free disk < need_disk_gb
                                 or available RAM < MIN_RAM_GB. need_disk_gb = 2 x the model size the job downloads.
  ResLog(every_s)                .maybe(note) prints a RES line when every_s seconds have passed (and .now()).
  Progress(path, cfg_hash)       append-only CSV, one row per finished unit (key + JSON payload), fsynced; a torn last
                                 line is dropped on load; rows of another config abort. Jobs skip keys already in it.

On molab (gVisor) the root filesystem reports ~8 EB free, so the disk check cannot see a real limit there.

Memory (Oct 29): molab advertises 32 GiB for the GPU machine while the cgroup v1 hard limit is 160 GiB and /proc/meminfo
shows the host. Sandboxes were reclaimed with no error, so the working budget is RAM_BUDGET_GB (28 GiB; env
SPRINT_RAM_BUDGET_GB). RES lines log cgroup v1 memory.usage_in_bytes (cg_use), memory.max_usage_in_bytes (cg_peak,
since the sandbox started), and from memory.stat total_rss, total_cache and the working set (cg_ws = usage -
total_inactive_file, the figure container runtimes evict on). preflight aborts if cg_ws + the job's expected RAM
(need_ram_gb) exceeds the budget.
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

MIN_RAM_GB = 16
RAM_BUDGET_GB = float(os.environ.get("SPRINT_RAM_BUDGET_GB", "28"))
CG1 = "/sys/fs/cgroup/memory/"
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


def _memstat():
    try:
        return {k: int(v) for k, v in (line.split() for line in open(CG1 + "memory.stat"))}
    except (OSError, ValueError):
        return {}


def cgroup_mem():
    """cgroup v1 figures in bytes (None where unavailable); cg_ws = usage - total_inactive_file."""
    use, ms = _read_int(CG1 + "memory.usage_in_bytes"), _memstat()
    ws = use - ms.get("total_inactive_file", 0) if use is not None else None
    return {"cg_use": use, "cg_peak": _read_int(CG1 + "memory.max_usage_in_bytes"), "cg_ws": ws,
            "cg_rss": ms.get("total_rss"), "cg_cache": ms.get("total_cache"),
            "cg_limit": _read_int(CG1 + "memory.limit_in_bytes")}


def snapshot(dirs=(), gpu=True):
    """dict of resource figures in bytes (None where unavailable)."""
    mi = _meminfo()
    s = {"rss": _rss(), **cgroup_mem(), "host_avail": mi.get("MemAvailable")}
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
    gb = lambda x: "na" if x is None else f"{x / 2 ** 30:.1f}"
    return " ".join(f"{k}={gb(v)}" for k, v in s.items()) + " (GiB)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def preflight(need_disk_gb, dirs=(), min_ram_gb=MIN_RAM_GB, gpu=True, need_ram_gb=8):
    """need_ram_gb: CPU RAM the job is expected to add (weights load straight to the GPU; this is the process,
    CUDA context, tokenizer and vLLM engine overhead)."""
    s = snapshot(dirs, gpu)
    log("PREFLIGHT", fmt(s), f"| need disk {need_disk_gb:.0f} GB, job RAM {need_ram_gb} GB, budget {RAM_BUDGET_GB} GiB")
    bad = []
    if s["host_avail"] is not None and s["host_avail"] < min_ram_gb * 1e9:
        bad.append(f"available RAM {s['host_avail'] / 1e9:.1f} GB < {min_ram_gb} GB")
    if s["cg_ws"] is not None and s["cg_ws"] + need_ram_gb * 2 ** 30 > RAM_BUDGET_GB * 2 ** 30:
        bad.append(f"projected RAM {s['cg_ws'] / 2 ** 30:.1f} + {need_ram_gb} GiB > budget {RAM_BUDGET_GB} GiB")
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
