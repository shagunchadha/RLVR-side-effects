"""Mirror job progress files and logs to the private HF dataset ($HF_REPO, default shagunchadha/rlvr-se-runs).

With STORE=hf the shards go to HF directly (exp/store.py); this module covers everything else a resume needs: the
per-unit progress files under $RESULTS_DIR/<job>/ and the job logs. Layout in the dataset:
    progress/<path relative to $RESULTS_DIR>      (weight caches _hf/, runs/ and temp files are never uploaded)
    logs/<lane>/<log file name>

  python -m exp.hfsync pull [--jobs B1 B3 M2]     copy progress/ into $RESULTS_DIR; a local file is replaced only when
                                                  the HF copy is longer (never shortens local progress)
  python -m exp.hfsync push --lane N2 [--log /tmp/lane.log]
  python -m exp.hfsync run --lane N2 --log /tmp/lane.log [--every 300] -- bash exp/launch/lane.sh - B1 B3b
        pull, run the command in the FOREGROUND (stdout+stderr -> --log), push progress + log every --every seconds
        and once more at the end; exits with the command's return code. Push failures are logged, never fatal.

HF_TOKEN is read from the environment by huggingface_hub; it is never printed, logged or written to any file here.
Torn last rows (a laptop copy cut mid-write) are uploaded as they are: resources.Progress drops them on load.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .store import DEFAULT_HF_REPO

IGNORE = ["**/.smoke_ok", "**/_hf/**", "_hf/**", "runs/**", "**/.tmp*", "**/*.tmp", "**/__pycache__/**"]


def _api():
    from huggingface_hub import HfApi
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set")
    return HfApi()


def _log(*a):
    print(time.strftime("%H:%M:%S"), "hfsync:", *a, flush=True)


def pull(root, repo, jobs=None):
    from huggingface_hub import snapshot_download
    pats = [f"progress/{j}/**" for j in jobs] if jobs else ["progress/**"]
    root = Path(root)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=tmp)
        src = Path(tmp) / "progress"
        n_new = n_keep = 0
        for f in (src.rglob("*") if src.exists() else []):
            if not f.is_file():
                continue
            dst = root / f.relative_to(src)
            if dst.exists() and dst.stat().st_size >= f.stat().st_size:
                n_keep += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            n_new += 1
    _log(f"pull: {n_new} file(s) copied into {root}, {n_keep} local copies kept (same size or longer)")


def push(root, repo, lane, logs=()):
    api = _api()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        api.upload_folder(folder_path=str(root), path_in_repo="progress", repo_id=repo, repo_type="dataset",
                          ignore_patterns=IGNORE, commit_message=f"progress {lane} {time.strftime('%H:%M:%S')}")
    for lg in logs:
        if lg and Path(lg).exists():
            api.upload_file(path_or_fileobj=str(lg), path_in_repo=f"logs/{lane}/{Path(lg).name}", repo_id=repo,
                            repo_type="dataset", commit_message=f"log {lane}")


def _safe_push(root, repo, lane, logs):
    try:
        push(root, repo, lane, logs)
        return True
    except Exception as e:                       # network / rate limit: keep the job running, try again next round
        _log(f"push failed ({type(e).__name__}: {str(e)[:200]})")
        return False


def run(root, repo, lane, log, every, cmd):
    pull(root, repo)
    _log(f"run: {' '.join(cmd)} (log {log}; push every {every}s)")
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while True:
            try:
                rc = p.wait(timeout=every)
                break
            except subprocess.TimeoutExpired:
                logs = [log] + sorted(str(q) for q in Path(log).parent.glob("*.log") if q != Path(log))
                ok = _safe_push(root, repo, lane, logs)
                tail = open(log, errors="replace").read().splitlines()[-1:] or [""]
                _log(f"{'pushed' if ok else 'push FAILED'}; job running; last log line: {tail[0][:160]}")
    logs = [log] + sorted(str(q) for q in Path(log).parent.glob("*.log") if q != Path(log))
    for _ in range(3):
        if _safe_push(root, repo, lane, logs):
            break
        time.sleep(30)
    _log(f"run: command exited rc={rc}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push", "run"])
    ap.add_argument("--root", default=os.environ.get("RESULTS_DIR", "sprint_out"))
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO", DEFAULT_HF_REPO))
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--lane", default="laptop")
    ap.add_argument("--log", default=None)
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "pull":
        pull(a.root, a.hf_repo, a.jobs)
    elif a.cmd == "push":
        push(a.root, a.hf_repo, a.lane, [a.log] if a.log else [])
    else:
        cmd = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
        if not cmd or not a.log:
            sys.exit("run needs --log and a command after --")
        return run(a.root, a.hf_repo, a.lane, a.log, a.every, cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
