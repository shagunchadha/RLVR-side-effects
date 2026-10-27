"""Copy the M1 progress CSV (and the job log) between a molab notebook and this repo, over the marimo pairing connection.

Runs on the LOCAL machine (no third-party packages). The notebook's disk is not trusted to survive, so the durable copy
of the progress CSV lives here, and a relaunch on a fresh notebook starts by pushing it back.

  MARIMO_TOKEN=... python exp/m1_sync.py --url https://<sandbox>.molab.run/ pull
  MARIMO_TOKEN=... python exp/m1_sync.py --url ... push                    # before relaunching on a fresh notebook
  MARIMO_TOKEN=... python exp/m1_sync.py --url ... watch [--rows 20] [--every 60]
(MARIMO_EXEC overrides the path of the marimo-pair execute-code.sh, default .claude/skills/marimo-pair/scripts/.)

pull  : copy the remote CSV and log to --local-dir. Incremental: only bytes past the local copy are sent when the
        local copy is a prefix of the remote file; otherwise the whole file.
push  : upload the local CSV to the remote path, if the remote file is missing or shorter (chunked; sha256-verified).
watch : poll every --every s; pull the log every poll (so the START line of the tensor in flight is always here), and
        the CSV whenever it has grown by >= --rows rows or the job ended (EXIT= in the log).
        Stops at EXIT=, or after --max-fail consecutive connection failures, and prints the last log lines it holds.
"""
import argparse
import base64
import gzip
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / ".claude" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
REMOTE_CSV = "/tmp/sprint_out/M1/M1_progress_shard0of1.csv"
REMOTE_LOG = "/tmp/m1.log"
LOCAL_DIR = ROOT / "results" / "sprint" / "M1" / "partial"
PULL_LIMIT = 200_000          # raw bytes per call
PUSH_CHUNK = 9_000            # base64 characters per call (the pairing script passes code on a command line)


class Unreachable(RuntimeError):
    pass


def remote(url, code, timeout=300):
    if not os.environ.get("MARIMO_TOKEN"):
        sys.exit("set MARIMO_TOKEN (the notebook's pairing token)")
    try:
        r = subprocess.run(["bash", os.environ.get("MARIMO_EXEC", str(EXEC)), "--url", url, "-"], input=code, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise Unreachable(f"timeout after {timeout}s") from e
    out = r.stdout + r.stderr
    if "Failed to connect" in out or "did not complete" in out:
        raise Unreachable(out.strip()[-300:])
    return out


def pull_file(url, rpath, lpath):
    """Bring lpath up to date with rpath. Returns the local size, or None if the remote file does not exist."""
    lpath = Path(lpath)
    lpath.parent.mkdir(parents=True, exist_ok=True)
    while True:
        local = lpath.read_bytes() if lpath.exists() else b""
        code = (f"import os,hashlib,base64,gzip\np={rpath!r}\noff={len(local)}\nh={hashlib.sha256(local).hexdigest()!r}\n"
                "if not os.path.exists(p):\n    print('@@NOFILE@@')\nelse:\n    b=open(p,'rb').read()\n"
                "    if off>len(b) or hashlib.sha256(b[:off]).hexdigest()!=h: off=0\n"
                f"    c=b[off:off+{PULL_LIMIT}]\n"
                "    print('@@SYNC@@',len(b),off,base64.b64encode(gzip.compress(c)).decode(),'@@END@@')\n")
        out = remote(url, code)
        if "@@NOFILE@@" in out:
            return None
        m = re.search(r"@@SYNC@@ (\d+) (\d+) (\S*) @@END@@", out)
        if not m:
            raise Unreachable("unexpected reply: " + out[-300:])
        total, off, chunk = int(m.group(1)), int(m.group(2)), gzip.decompress(base64.b64decode(m.group(3)))
        new = (local if off else b"") + chunk
        tmp = lpath.with_suffix(lpath.suffix + ".tmp")
        tmp.write_bytes(new)
        os.replace(tmp, lpath)
        if len(new) >= total:
            return len(new)


def push_file(url, lpath, rpath):
    data = Path(lpath).read_bytes()
    out = remote(url, f"import os\np={rpath!r}\nprint('@@SIZE@@', os.path.getsize(p) if os.path.exists(p) else -1)\n")
    rsize = int(re.search(r"@@SIZE@@ (-?\d+)", out).group(1))
    if rsize >= len(data):
        print(f"remote {rpath} has {rsize} bytes >= local {len(data)}; not pushed")
        return False
    b64 = base64.b64encode(gzip.compress(data)).decode()
    part = rpath + ".part.b64"
    remote(url, f"import os\nos.makedirs(os.path.dirname({rpath!r}), exist_ok=True)\nopen({part!r},'w').close()\n")
    for k in range(0, len(b64), PUSH_CHUNK):
        remote(url, f"open({part!r},'a').write({b64[k:k + PUSH_CHUNK]!r})\n")
    want = hashlib.sha256(data).hexdigest()
    out = remote(url, f"import base64,gzip,hashlib,os\nb=gzip.decompress(base64.b64decode(open({part!r}).read()))\n"
                      f"ok=hashlib.sha256(b).hexdigest()=={want!r}\n"
                      f"if ok:\n    open({rpath!r}+'.tmp','wb').write(b); os.replace({rpath!r}+'.tmp',{rpath!r})\n"
                      f"os.remove({part!r})\nprint('@@PUSH@@', ok, len(b))\n")
    if "@@PUSH@@ True" not in out:
        sys.exit("push failed verification: " + out[-300:])
    print(f"pushed {lpath} -> {rpath} ({len(data)} bytes, sha256 verified)")
    return True


def n_rows(path):
    p = Path(path)
    return max(0, p.read_bytes().count(b"\n") - 1) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pull", "push", "watch"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--remote-csv", default=REMOTE_CSV)
    ap.add_argument("--remote-log", default=REMOTE_LOG)
    ap.add_argument("--local-dir", default=str(LOCAL_DIR))
    ap.add_argument("--rows", type=int, default=20, help="watch: copy after this many new rows")
    ap.add_argument("--every", type=int, default=60, help="watch: poll interval (s)")
    ap.add_argument("--max-fail", type=int, default=5)
    a = ap.parse_args()
    lcsv = Path(a.local_dir) / Path(a.remote_csv).name
    llog = Path(a.local_dir) / "M1.log"
    if a.action == "push":
        push_file(a.url, lcsv, a.remote_csv)
        return
    if a.action == "pull":
        pull_file(a.url, a.remote_csv, lcsv)
        pull_file(a.url, a.remote_log, llog)
        print(f"{lcsv}: {n_rows(lcsv)} rows; {llog}")
        return
    fails, last = 0, n_rows(lcsv)
    while True:
        try:
            out = remote(a.url, f"import os\np={a.remote_csv!r}\nl={a.remote_log!r}\n"
                                "n=max(0,open(p,'rb').read().count(b'\\n')-1) if os.path.exists(p) else 0\n"
                                "e=os.path.exists(l) and 'EXIT=' in open(l,errors='replace').read()[-2000:]\n"
                                "print('@@STAT@@',n,e)\n", timeout=120)
            m = re.search(r"@@STAT@@ (\d+) (True|False)", out)
            rows, ended = int(m.group(1)), m.group(2) == "True"
            fails = 0
            pull_file(a.url, a.remote_log, llog)
            if ended or rows - last >= a.rows:
                pull_file(a.url, a.remote_csv, lcsv)
                last = n_rows(lcsv)
                print(time.strftime("%H:%M:%S"), f"copied: {last} rows -> {lcsv}", flush=True)
            if ended:
                print("job ended (EXIT= in the log)", flush=True)
                break
        except Unreachable as e:
            fails += 1
            print(time.strftime("%H:%M:%S"), f"unreachable ({fails}/{a.max_fail}): {e}", flush=True)
            if fails >= a.max_fail:
                print(f"notebook gone; local copy holds {n_rows(lcsv)} rows. Last log lines:", flush=True)
                if llog.exists():
                    print("\n".join(llog.read_text(errors="replace").splitlines()[-15:]))
                sys.exit(2)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
