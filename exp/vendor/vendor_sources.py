"""Fetch the F1 scorers' upstream source files at pinned commits, apply import-path rewrites only, and record hashes.

    python -m exp.vendor.vendor_sources            # (re)writes exp/vendor/** and tests/sprint/fixtures/revision/**

Registration amendment 18 (F1): maths is scored by math-verify (pinned in requirements_revision.txt) with OLMES's
Minerva "EM Flex" as the cross-check; IFEval and IFBench by their official checkers. The code is vendored rather
than installed because lm_eval 0.4.3 and OLMES pull dependencies that conflict with requirements_sprint.txt.
Every file is fetched from a pinned commit and its upstream sha256 is recorded; the ONLY modifications are the
rewrites listed per file in VENDOR.md (import paths, and for lm_eval the `datasets` / `lm_eval.utils` imports).
The test fixtures are official example outputs with their official expected scores, used by
tests/sprint/test_revision_scorers.py.
"""
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VEND = ROOT / "exp" / "vendor"
FIX = ROOT / "tests" / "sprint" / "fixtures" / "revision"

LMEVAL = ("EleutherAI/lm-evaluation-harness", "3fa4fd725c8a428710109f1d6c14eda37e95baea", "v0.4.3", "MIT")
OLMES = ("allenai/olmes", "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e", None, "Apache-2.0")
GOOGLE = ("google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0", None, "Apache-2.0")
IFBENCH = ("allenai/IFBench", "488918665d53f867fa908a293769ec6e71e74aec", None, "Apache-2.0")
# IFBench as of 2025-10-17: the commit that produced its committed eval_results (F1 main IFBench scorer, PI Nov 28)
IFBENCH_2025 = ("allenai/IFBench", "c94da0204a1552be9e5c206b732e29f7d279f0c7", None, "Apache-2.0")
OLD = "exp.vendor.ifbench_2025_10"
NL = "\n"
IFBENCH_2025_FIX = [("import instructions_util" + NL, f"from {OLD} import instructions_util" + NL),
                    ("import instructions_registry" + NL, f"from {OLD} import instructions_registry" + NL),
                    ("import instructions" + NL, f"from {OLD} import instructions" + NL),
                    ("download('en_core_web_sm')" + NL,
                     "# vendored: no model download at import; en_core_web_sm 3.8.0 is pinned in requirements_revision.txt" + NL)]

LMEVAL_FIX = [  # (old, new): lm_eval's package imports, unavailable here, replaced by stdlib equivalents
    ("import datasets\n", ""),
    ("from lm_eval.utils import eval_logger\n", "import logging\n\neval_logger = logging.getLogger(\"lm_eval\")\n"),
]
FUTURE = "from __future__ import annotations  # vendored: keeps the `datasets.Dataset` annotations unevaluated\n"

FILES = [
    # (repo, upstream path, vendored path, rewrites, prepend)
    (LMEVAL, "lm_eval/tasks/minerva_math/utils.py", "exp/vendor/lm_eval_math/minerva_utils.py", LMEVAL_FIX, FUTURE),
    (LMEVAL, "lm_eval/tasks/hendrycks_math/utils.py", "exp/vendor/lm_eval_math/hendrycks_utils.py", LMEVAL_FIX, FUTURE),
    (LMEVAL, "LICENSE.md", "exp/vendor/lm_eval_math/LICENSE.md", [], ""),
    (OLMES, "oe_eval/tasks/oe_eval_tasks/minerva_math.py", "exp/vendor/olmes_reference/minerva_math.py.txt", [], ""),
    (OLMES, "oe_eval/tasks/oe_eval_tasks/aime.py", "exp/vendor/olmes_reference/aime.py.txt", [], ""),
    (OLMES, "oe_eval/configs/tasks.py", "exp/vendor/olmes_reference/tasks.py.txt", [], ""),
    (OLMES, "LICENSE", "exp/vendor/olmes_reference/LICENSE", [], ""),
] + [
    (GOOGLE, f"instruction_following_eval/{f}", f"exp/vendor/ifeval_google/{f}",
     [("from instruction_following_eval import ", "from exp.vendor.ifeval_google import ")], "")
    for f in ("instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py",
              "instructions_test.py", "instructions_util_test.py")
] + [
    (GOOGLE, "LICENSE", "exp/vendor/ifeval_google/LICENSE", [], ""),
] + [
    (IFBENCH, f"ifbench/{f}", f"exp/vendor/ifbench/{f}",
     [("from ifbench import ", "from exp.vendor.ifbench import "),
      ("from ifbench.instructions import ", "from exp.vendor.ifbench.instructions import ")], "")
    for f in ("instructions.py", "instructions_registry.py", "instructions_util.py", "classic_instructions.py")
] + [
    (IFBENCH, "evaluation_lib.py", "exp/vendor/ifbench/evaluation_lib.py",
     [("from ifbench import ", "from exp.vendor.ifbench import ")], ""),
    (IFBENCH, "LICENSE", "exp/vendor/ifbench/LICENSE", [], ""),
] + [
    (IFBENCH_2025, f, f"exp/vendor/ifbench_2025_10/{f}", IFBENCH_2025_FIX, "")
    for f in ("instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py", "LICENSE")
]
FIXTURES = [  # official example outputs and their official expected results (gzipped, unmodified)
    (IFBENCH, "data/sample_output.jsonl", "ifbench_sample_output.jsonl.gz"),
    (IFBENCH, "eval/eval_results_loose.jsonl", "ifbench_eval_results_loose.jsonl.gz"),
    (IFBENCH, "eval/eval_results_strict.jsonl", "ifbench_eval_results_strict.jsonl.gz"),
    (IFBENCH, "ifbench/data/IFBench_test.jsonl", "ifbench_test.jsonl.gz"),
    (GOOGLE, "instruction_following_eval/data/input_data.jsonl", "ifeval_input_data.jsonl.gz"),
    (GOOGLE, "instruction_following_eval/data/input_response_data_gpt4_20231107_145030.jsonl",
     "ifeval_gpt4_responses.jsonl.gz"),
]


def fetch(repo, path):
    url = f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read(), url


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    rows = []
    for repo, src, dst, rewrites, prepend in FILES:
        raw, url = fetch(repo, src)
        text = raw.decode("utf-8")
        for old, new in rewrites:
            if old not in text and not src.endswith(("LICENSE", "LICENSE.md")):
                pass                                    # not every file has every rewrite (e.g. util modules)
            text = text.replace(old, new)
        out = (prepend + text).encode("utf-8")
        p = ROOT / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(out)
        init = p.parent / "__init__.py"
        if not init.exists() and p.suffix == ".py":
            init.write_text('"""Vendored upstream code; see exp/vendor/VENDOR.md."""\n', encoding="utf-8")
        rows.append({"file": dst, "upstream": url, "license": repo[3], "upstream_sha256": sha(raw), "vendored_sha256": sha(out),
                     "modified": raw != out, "rewrites": [o.strip() for o, _ in rewrites if o in raw.decode("utf-8")]
                     + (["prepended `from __future__ import annotations`"] if prepend else [])})
    FIX.mkdir(parents=True, exist_ok=True)
    for repo, src, name in FIXTURES:
        raw, url = fetch(repo, src)
        (FIX / name).write_bytes(gzip.compress(raw, mtime=0))
        rows.append({"file": f"tests/sprint/fixtures/revision/{name}", "upstream": url, "license": repo[3],
                     "upstream_sha256": sha(raw), "vendored_sha256": "(gzip of upstream bytes)", "modified": False, "rewrites": []})
    (VEND / "__init__.py").write_text('"""Vendored upstream scorer code for F1 (amendment 18); see VENDOR.md."""\n', encoding="utf-8")
    (VEND / "vendor_manifest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    L = ["# Vendored code (F1 scorers, registration amendment 18)", "",
         "Generated by `python -m exp.vendor.vendor_sources`. Each file is fetched from a pinned upstream commit. "
         "The only modifications are the listed rewrites (import paths; for lm_eval also its `datasets` and "
         "`lm_eval.utils` imports, which this environment does not have, replaced by a `logging` logger and lazy "
         "annotations). `olmes_reference/*.txt` are unmodified OLMES sources kept for reference: the OLMES functions "
         "the scorers follow are re-stated in `exp/revision/scorers.py`, each citing its source lines.", "",
         "| file | upstream (pinned) | license | upstream sha256 | modified | rewrites |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['file']}` | {r['upstream']} | {r['license']} | `{r['upstream_sha256'][:16]}…` | "
                 f"{'yes' if r['modified'] else 'no'} | {'; '.join(r['rewrites']) or '—'} |")
    (VEND / "VENDOR.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
