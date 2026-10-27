"""Mirror a job's output directory and log between a molab notebook and this repo, over the marimo pairing connection.

Runs on the LOCAL machine (no third-party packages). The notebook's disk is not trusted to survive, so the durable copy
of every progress file lives here, and a relaunch on a fresh notebook starts by pushing it back.

  MARIMO_TOKEN=... python exp/sync_remote.py --url <notebook> --remote-dir /tmp/sprint_out/M2 --remote-log /tmp/m2.log \\
                   --local-dir results/sprint/M2/partial  pull | push | watch [--every 60]

pull  : copy every file under --remote-dir (recursively; weight caches and files > --max-mb skipped) and the log into
        --local-dir. Incremental: only bytes past the local copy are sent when it is a prefix of the remote file.
push  : upload every local file that is missing or shorter on the remote (chunked, sha256-verified). Used before
        relaunching a job on a fresh notebook, so the job resumes from its progress files.
watch : pull every --every s (the polls also keep the notebook active); stops when the log contains EXIT= (pulling
        once more), or after --max-fail consecutive connection failures, printing the last log lines it holds.
(MARIMO_EXEC overrides the path of the marimo-pair execute-code.sh, default .claude/skills/marimo-pair/scripts/.)
From Git Bash on Windows, run it with MSYS_NO_PATHCONV=1: otherwise '/tmp/...' arguments are rewritten to Windows paths
before Python sees them (remote paths that look like Windows paths are refused). watch prints one status line per poll
(remote files, local copies, log size), so a watcher that sees nothing says so.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / ".claude" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
SKIP_DIRS = ("_dl", "_hf", "hf", ".cache")
PULL_LIMIT = 200_000          # raw bytes per call
PUSH_CHUNK = 9_000            # base64 characters per call (the pairing script passes code on a command line)


class Unreachable(RuntimeError):
    pass


def remote(url, code, timeout=300):
    if not os.environ.get("MARIMO_TOKEN"):
        sys.exit("set MARIMO_TOKEN (the notebook's pairing token)")
    try:
        r = subprocess.run(["bash", os.environ.get("MARIMO_EXEC", str(EXEC)), "--url", url, "-"], input=code,
                           capture_output=True, text=True, timeout=timeout)
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


def remote_listing(url, rdir, max_mb):
    """{relative path: size} of the files under rdir, skipping weight caches and large files."""
    code = (f"import os,json\nr={rdir!r}\nout={{}}\nskip={SKIP_DIRS!r}\n"
            "for d,ds,fs in os.walk(r):\n    ds[:]=[x for x in ds if x not in skip]\n"
            "    for f in fs:\n        p=os.path.join(d,f)\n        s=os.path.getsize(p)\n"
            f"        if s<={max_mb}*1e6 and not f.endswith(('.tmp','.part.b64','.lock','.incomplete')): out[os.path.relpath(p,r)]=s\n"
            "print('@@LS@@',json.dumps(out),'@@END@@')\n")
    out = remote(url, code)
    m = re.search(r"@@LS@@ (.*) @@END@@", out)
    if not m:
        raise Unreachable("unexpected reply: " + out[-300:])
    return json.loads(m.group(1))


def pull_all(url, rdir, rlog, ldir, max_mb):
    ldir = Path(ldir)
    n = 0
    for rel, size in remote_listing(url, rdir, max_mb).items():
        lp = ldir / rel
        if not lp.exists() or lp.stat().st_size != size:
            pull_file(url, f"{rdir.rstrip('/')}/{rel}", lp)
            n += 1
    if rlog:
        pull_file(url, rlog, ldir / Path(rlog).name)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pull", "push", "watch"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--remote-dir", required=True)
    ap.add_argument("--remote-log", default=None)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--max-mb", type=float, default=50)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--max-fail", type=int, default=5)
    a = ap.parse_args()
    for nm, v in (("--remote-dir", a.remote_dir), ("--remote-log", a.remote_log)):
        if v and (":" in v or "\\" in v or not v.startswith("/")):
            sys.exit(f"{nm} {v!r} is not a POSIX path on the notebook (Git Bash path conversion? set MSYS_NO_PATHCONV=1)")
    ldir = Path(a.local_dir)
    if a.action == "push":
        for lp in sorted(p for p in ldir.rglob("*") if p.is_file() and p.name != Path(a.remote_log or "").name):
            push_file(a.url, lp, f"{a.remote_dir.rstrip('/')}/{lp.relative_to(ldir).as_posix()}")
        return
    if a.action == "pull":
        print(f"pulled {pull_all(a.url, a.remote_dir, a.remote_log, ldir, a.max_mb)} changed file(s) into {ldir}")
        return
    fails = 0
    while True:
        try:
            n = pull_all(a.url, a.remote_dir, a.remote_log, ldir, a.max_mb)
            fails = 0
            lg = ldir / Path(a.remote_log).name if a.remote_log else None
            nloc = sum(1 for p in ldir.rglob("*") if p.is_file()) if ldir.exists() else 0
            print(time.strftime("%H:%M:%S"), f"poll: pulled {n} changed, {nloc} local file(s) in {ldir}; "
                  f"log {lg.stat().st_size if lg and lg.exists() else 'MISSING'} bytes", flush=True)
            if lg and lg.exists() and "EXIT=" in lg.read_text(errors="replace")[-2000:]:
                pull_all(a.url, a.remote_dir, a.remote_log, ldir, a.max_mb)
                print("job ended (EXIT= in the log)", flush=True)
                break
        except Unreachable as e:
            fails += 1
            print(time.strftime("%H:%M:%S"), f"unreachable ({fails}/{a.max_fail}): {e}", flush=True)
            if fails >= a.max_fail:
                lg = ldir / Path(a.remote_log).name if a.remote_log else None
                print("notebook gone. Last log lines:", flush=True)
                if lg and lg.exists():
                    print("\n".join(lg.read_text(errors="replace").splitlines()[-15:]))
                sys.exit(2)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
