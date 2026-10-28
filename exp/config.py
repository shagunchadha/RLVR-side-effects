"""Pinned checkpoints, repo paths and provenance helpers for the sprint jobs.

Model SHAs are copied from docs/checkpoint_map.md (main-branch commits queried 2026-10-14). Jobs load by
(repo, sha) only, never by a moving branch name.
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ITEMS_DIR = REPO_ROOT / "items"
FULL_RUN = {c: REPO_ROOT / "results" / f"full_run_{c}" / f"full_run_{c}.json" for c in ("sft", "dpo", "rlvr")}
RESULTS_SPRINT = REPO_ROOT / "results" / "sprint"
DOCS_SPRINT = REPO_ROOT / "docs" / "sprint"

# key -> (repo, main commit SHA) ; source: docs/checkpoint_map.md section 2
CHECKPOINTS = {
    "base": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    "sft": ("allenai/Olmo-3-7B-Think-SFT", "6ff857587e040d6d523a3d5f3a56e918f5401d66"),
    "dpo": ("allenai/Olmo-3-7B-Think-DPO", "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"),
    "rlvr": ("allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4"),
    "rl_zero_math": ("allenai/Olmo-3-7B-RL-Zero-Math", "dee26a4074b88215cd1326472e1ab14d0259d892"),
    "rl_zero_code": ("allenai/Olmo-3-7B-RL-Zero-Code", "a961af69d75f9c52b3ae4db7b1664d890229b7ca"),
    "rl_zero_if": ("allenai/Olmo-3-7B-RL-Zero-IF", "3cae007c5324a9a06ee6780f03b848bc7b8fd440"),
    "rl_zero_general": ("allenai/Olmo-3-7B-RL-Zero-General", "24e5814cb36944d3eb36a49679d02c2b3bb46195"),
    "rl_zero_mix": ("allenai/Olmo-3-7B-RL-Zero-Mix", "73b2b1a50d94aec859e22c9b4a524416b5a3a739"),
}
THINK_LADDER = ("sft", "dpo", "rlvr")

# tokens whose per-position probability M2 reads (each is resolved to its first token, with and without a leading space)
THINK_END = "</think>"
SWITCH_WORDS = ("Wait", "Hmm", "But", "Alternatively")


def ckpt(key):
    """(repo, sha) of a checkpoint key; '<key>@<branch>' (e.g. rlvr@step_0025, B5) resolves the branch to the commit
    recorded by B0 in results/sprint/B0_inventory.json, so branch checkpoints are also loaded by (repo, sha) only."""
    if "@" in key:
        base, branch = key.split("@", 1)
        repo = ckpt(base)[0]
        inv = json.load(open(RESULTS_SPRINT / "B0_inventory.json", encoding="utf-8"))
        for r in inv["repos"]:
            if r["repo"] == repo:
                for b in r["branches"]:
                    if b["name"] == branch:
                        return repo, b["sha"]
        raise KeyError(f"branch {branch!r} of {repo} is not in the B0 inventory")
    if key not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {key!r}; known: {sorted(CHECKPOINTS)}")
    return CHECKPOINTS[key]


def config_hash(cfg):
    """12-hex SHA-256 of the canonical JSON of a job config (key order and whitespace do not matter)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:12]


def git_sha():
    env = os.environ.get("GIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "") if sha else "unknown"
    except Exception:
        return "unknown"


_PKGS = ("torch", "transformers", "vllm", "tokenizers", "safetensors", "huggingface-hub", "pyarrow", "numpy",
         "scipy", "pandas", "datasets", "accelerate", "flashinfer-python", "triton")


def package_versions():
    v = {"python": platform.python_version()}
    for p in _PKGS:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def load_items(component):
    name = {"A": "items_A_selfreport.json", "B": "items_B_sycophancy.json", "C": "items_C_calibration.json",
            "D": "items_D_csqa.json", "E": "items_E_math.json"}[component]
    return json.load(open(ITEMS_DIR / name, encoding="utf-8"))["items"]
