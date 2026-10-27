"""Version gate for the sprint environment.

python -m exp.versioncheck [--no-gpu] [--requirements requirements_sprint.txt]

Prints Python, GPU, driver, CUDA and the installed version of every package pinned in requirements_sprint.txt,
and exits 1 if any pinned package is missing or at a different version, or if PYTHONPATH is set. With --no-gpu the
GPU is reported if present but not required (B0, M1 on CPU, the CPU tests).
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;]+)", line)
        if m:
            pins[canon(m.group(1))] = m.group(3)
    return pins


def gpu_info():
    info = {"cuda_available": False}
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:  # torch missing is reported as a package mismatch below
        info["torch_error"] = repr(e)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            info["nvidia_smi"] = out.stdout.strip()
    except Exception:
        pass
    return info


def installed_versions(pins):
    got = {}
    for name in pins:
        try:
            got[name] = md.version(name)
        except md.PackageNotFoundError:
            got[name] = None
    return got


def env_hash(got, gpu):
    blob = json.dumps({"pkgs": got, "python": platform.python_version(), "gpu": gpu.get("gpu"),
                       "smi": gpu.get("nvidia_smi")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def check(requirements=REPO / "requirements_sprint.txt", need_gpu=True, quiet=False):
    """Return (ok, report dict). Used by the CLI and by exp.run before every job."""
    pins = read_pins(requirements)
    got = installed_versions(pins)
    gpu = gpu_info()
    bad = {k: (pins[k], got[k]) for k in pins if got[k] != pins[k]}
    problems = []
    if os.environ.get("PYTHONPATH"):
        problems.append(f"PYTHONPATH is set ({os.environ['PYTHONPATH']!r}); unset it")
    if bad:
        problems.append(f"{len(bad)} pinned packages missing or at another version")
    if need_gpu and not gpu.get("cuda_available"):
        problems.append("no CUDA GPU visible")
    report = {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
              "env_hash": env_hash(got, gpu), "n_pinned": len(pins), "mismatches": bad, "problems": problems,
              "key_versions": {k: got.get(k) for k in ("torch", "transformers", "vllm", "huggingface-hub",
                                                        "safetensors", "pyarrow", "numpy")}}
    if not quiet:
        print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))
        for k, (want, have) in sorted(bad.items()):
            print(f"MISMATCH {k}: pinned {want}, installed {have}")
        print("RESULT:", "OK" if not problems else "FAIL: " + "; ".join(problems))
    return not problems, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--requirements", default=str(REPO / "requirements_sprint.txt"))
    a = ap.parse_args(argv)
    ok, _ = check(a.requirements, need_gpu=not a.no_gpu)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
