#!/usr/bin/env python3
"""Frozen full run: one parametrized script for SFT, DPO and RLVR (replaces the three pilot scripts).

Usage (venv python, PYTHONPATH cleared, one checkpoint per session):
  env -u PYTHONPATH /tmp/batch_venv/bin/python scripts/full_run.py --checkpoint rlvr          > results/full_run_rlvr.log 2>&1
  env -u PYTHONPATH /tmp/batch_venv/bin/python scripts/full_run.py --checkpoint rlvr --limit 8 > results/full_run_rlvr_dry.log 2>&1   # plumbing test
  python scripts/full_run.py --write-order      # writes/verifies items/run_order.json, no GPU, no torch import

Protocol (docs/item-set-spec-v4.md sections 1 and 5)
  - version gate first: GPU, driver, torch, transformers, huggingface-hub, accelerate and every package in
    requirements_pinned.txt must match; PYTHONPATH must be empty; the report is saved with the results;
  - batch size 8, left padding, temperature 0.7, top_p 0.9, torch.manual_seed(seed) before every generate call
    (the way batch_timing_test.py / batch_math_pilot.py did it);
  - components run in the order A, B, C, D, E; rows sorted by id within a component; self-report (A) is three
    passes over the same order with seeds 0, 1, 2, every other component one pass with seed 0;
  - token cap 4000 for A-D, 16000 for E; one prompt template (items/prompts.py), prompts are stored in the item files;
  - the batch composition is fixed by items/run_order.json (identical for all three checkpoints); resume skips
    finished batches only, so composition never changes; results are saved after every batch.
  - invalid rows are reported, never re-sampled, never filled.
Dry run (--limit N): first N rows per component by id, E = the 10 length-check MATH problems (items_E_lengthcheck.json, not
in the final 50) in the frozen template; D falls
back to unreviewed provisional items if items_D_csqa.json does not exist yet. Output files carry a _dry suffix and
must not be used for anything else.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "items"
sys.path.insert(0, str(ITEMS))
import answer_parser  # noqa: E402

BATCH_SIZE, TEMPERATURE, TOP_P = 8, 0.7, 0.9
ORDER = "ABCDE"
CAPS = {"A": 4000, "B": 4000, "C": 4000, "D": 4000, "E": 16000}
SELF_REPORT_SEEDS = [0, 1, 2]
ITEM_FILES = {"A": "items_A_selfreport.json", "B": "items_B_sycophancy.json", "C": "items_C_calibration.json",
              "D": "items_D_csqa.json", "E": "items_E_math.json"}
DRY_E_FILE = "items_E_lengthcheck.json"
CKPT_KEY = {"sft": "think_sft", "dpo": "think_dpo", "rlvr": "think_rlvr"}
PIN = {"torch": "2.11.0+cu130", "transformers": "5.14.1", "huggingface-hub": "1.33.0", "accelerate": "1.15.0"}
PIN_DRIVER, PIN_GPU = "595.71.05", "NVIDIA RTX PRO 6000 Blackwell Server Edition"
T0 = time.time()


MANIFEST = ITEMS / "MANIFEST.sha256"


def verify_manifest(manifest=MANIFEST, root=ROOT):
    """Check every file listed in the sha256sum-format manifest against its hash; returns a list of problems
    (empty = all match). Paths in the manifest are relative to the repository root."""
    if not Path(manifest).is_file():
        return [f"{manifest} is missing"]
    problems, n = [], 0
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition(" ")
        rel = rel[1:] if rel[:1] in (" ", "*") else rel  # sha256sum text ("  ") and binary (" *") formats
        n += 1
        f = Path(root) / rel
        if not f.is_file():
            problems.append(f"{rel}: listed in the manifest but missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            problems.append(f"{rel}: hash differs from the manifest")
    return problems or ([] if n else [f"{manifest} lists no files"])


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# ------------------------------- checkpoints -------------------------------
def checkpoint_from_map(ckpt):
    """(repo, sha) for think_sft / think_dpo / think_rlvr, parsed from the table in docs/checkpoint_map.md."""
    import re
    text = (ROOT / "docs" / "checkpoint_map.md").read_text(encoding="utf-8")
    m = re.search(r"\|\s*" + CKPT_KEY[ckpt] + r"\s*\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{40})`", text)
    if not m:
        sys.exit(f"{CKPT_KEY[ckpt]} not found in docs/checkpoint_map.md")
    return m.group(1), m.group(2)


# ------------------------------- version gate -------------------------------
def version_gate():
    """Print and return (report text, env hash). Aborts the process on any mismatch."""
    import torch
    import transformers
    import huggingface_hub
    smi = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").strip()
    gpu, driver = [x.strip() for x in smi.split(",")]
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "huggingface-hub": huggingface_hub.__version__, "accelerate": md.version("accelerate")}
    out = ["# full-run environment check",
           f"python        : {sys.version.split()[0]}  exe={sys.executable}",
           f"PYTHONPATH    : {os.environ.get('PYTHONPATH', '')!r}",
           f"GPU           : {gpu}", f"driver        : {driver}",
           f"torch.cuda    : {torch.version.cuda} available={torch.cuda.is_available()}"]
    bad = []
    if os.environ.get("PYTHONPATH", ""):
        bad.append("PYTHONPATH not cleared")
    if gpu != PIN_GPU:
        bad.append(f"GPU {gpu!r} != {PIN_GPU!r}")
    if driver != PIN_DRIVER:
        bad.append(f"driver {driver} != {PIN_DRIVER}")
    for k, v in PIN.items():
        ok = got[k] == v
        out.append(f"{k:16s}: {got[k]}  (pinned {v})  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{k} {got[k]} != {v}")
    try:
        out.append(f"datasets      : {md.version('datasets')} (not required by this script)")
    except Exception:
        out.append("datasets      : not installed (not required by this script)")
    installed = []
    for line in (ROOT / "requirements_pinned.txt").read_text().splitlines():
        line = line.strip()
        if "==" not in line:
            continue
        n, v = line.split("==")
        try:
            g = md.version(n)
        except md.PackageNotFoundError:
            g = None
        installed.append(f"{n}=={g}")
        if g is None or (g != v and g.split("+")[0] != v):
            out.append(f"  pinned {n}=={v}: installed {g}  MISMATCH")
            bad.append(f"{n} {g} != {v}")
    env_hash = hashlib.sha256("\n".join(sorted(installed) + [gpu, driver, sys.version.split()[0]]).encode()).hexdigest()[:16]
    out.append(f"environment hash: {env_hash}  (sha256 prefix of the installed pinned package versions, GPU, driver, python)")
    out.append("RESULT: " + ("ABORT - " + "; ".join(bad) if bad else "ALL PINNED VERSIONS MATCH"))
    text = "\n".join(out)
    print(text, flush=True)
    return text, env_hash, bad


# ------------------------------- items and run order -------------------------------
def load_items(dry, limit, components):
    """{component: [item, ...]} sorted by id. Dry: first `limit` rows per component, E = length-check problems."""
    items = {}
    for c in components:
        if dry and c == "E":
            rows = json.loads((ITEMS / DRY_E_FILE).read_text(encoding="utf-8"))["items"]
        elif dry and c == "D" and not (ITEMS / ITEM_FILES["D"]).exists():
            import build_D_finalize as bdf
            rows = bdf.provisional_items(ITEMS / "csqa_review.csv", limit)
            log("D: items_D_csqa.json does not exist yet; using UNREVIEWED provisional items for the dry run")
        else:
            path = ITEMS / ITEM_FILES[c]
            if not path.exists():
                sys.exit(f"missing {path.name}; a full run needs all five item files (finalize D first)")
            rows = json.loads(path.read_text(encoding="utf-8"))["items"]
        rows = sorted(rows, key=lambda r: r["id"])
        if dry and c != "E":
            rows = rows[:limit]
        items[c] = rows
    return items


def build_order(items):
    """Batches over the whole run, global batch_index; A has three seed passes, the rest one pass with seed 0."""
    batches, n = [], 0
    for c in ORDER:
        if c not in items:
            continue
        ids = [r["id"] for r in items[c]]
        for seed in (SELF_REPORT_SEEDS if c == "A" else [0]):
            for k in range(0, len(ids), BATCH_SIZE):
                batches.append({"batch_index": n, "component": c, "seed": seed, "max_new_tokens": CAPS[c],
                                "ids": ids[k:k + BATCH_SIZE]})
                n += 1
    return batches


def order_document(items, batches):
    sha = {c: hashlib.sha256((ITEMS / ITEM_FILES[c]).read_bytes()).hexdigest() for c in items if (ITEMS / ITEM_FILES[c]).exists()}
    return {"meta": {"batch_size": BATCH_SIZE, "temperature": TEMPERATURE, "top_p": TOP_P, "caps": CAPS,
                     "component_order": ORDER, "self_report_seeds": SELF_REPORT_SEEDS, "other_seed": 0,
                     "rows_sorted_by": "id (string order) within a component",
                     "seeding": "torch.manual_seed(seed) immediately before every generate call",
                     "n_batches": len(batches), "n_generations": sum(len(b["ids"]) for b in batches),
                     "item_file_sha256": sha},
            "batches": batches}


def write_or_verify_order(doc):
    path = ITEMS / "run_order.json"
    text = json.dumps(doc, indent=1) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            sys.exit("items/run_order.json exists and differs from the order derived from the item files; "
                     "the order is part of the frozen protocol and must be identical for all checkpoints")
        log("items/run_order.json verified (identical)")
    else:
        path.write_text(text, encoding="utf-8")
        log("wrote items/run_order.json:", doc["meta"]["n_batches"], "batches,", doc["meta"]["n_generations"], "generations")


# ------------------------------- generation -------------------------------
def save_atomic(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=sorted(CKPT_KEY))
    ap.add_argument("--components", default="ABCDE", help="subset of ABCDE, e.g. AB or A,B")
    ap.add_argument("--limit", type=int, default=None, help="dry run: first N rows per component (E: the 10 length-check problems)")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--write-order", action="store_true", help="write/verify items/run_order.json and exit (no GPU)")
    a = ap.parse_args()
    comps = [c for c in a.components.replace(",", "").upper() if c in ORDER]
    dry = a.limit is not None

    if a.write_order:
        items = load_items(False, None, list(ORDER))
        write_or_verify_order(order_document(items, build_order(items)))
        return
    if not a.checkpoint:
        sys.exit("--checkpoint is required")

    # 0. frozen files must match items/MANIFEST.sha256 before anything else (no model, no torch import)
    bad_files = verify_manifest(MANIFEST)
    if bad_files:
        sys.exit("MANIFEST MISMATCH (items/MANIFEST.sha256): " + "; ".join(bad_files))

    # 1. version gate before anything else
    env_text, env_hash, bad = version_gate()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"full_run_{a.checkpoint}" + ("_dry" if dry else "")
    freeze = sh(f"uv pip freeze --python {sys.executable}")
    (out_dir / f"{tag}_env.txt").write_text(env_text + "\n# --- uv pip freeze (venv, includes system-site packages) ---\n" + freeze, encoding="utf-8")
    if bad:
        sys.exit("VERSION MISMATCH: " + "; ".join(bad))

    # 2. items and order (the order is computed from ALL components, so batch_index is stable under --components)
    all_items = load_items(dry, a.limit, list(ORDER))
    batches = build_order(all_items)
    doc = order_document(all_items, batches)
    if dry:
        save_atomic(out_dir / f"{tag}_run_order.json", doc)
    else:
        write_or_verify_order(doc)
    by_id = {c: {r["id"]: r for r in rows} for c, rows in all_items.items()}
    todo = [b for b in batches if b["component"] in comps]
    log(f"checkpoint {a.checkpoint}  dry={dry}  components {''.join(comps)}  batches {len(todo)}/{len(batches)}")

    # 3. model
    import torch
    import huggingface_hub
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, sha = checkpoint_from_map(a.checkpoint)
    log("repo", repo, "sha", sha, "| hub confirms revision:", huggingface_hub.HfApi().model_info(repo, revision=sha).sha)
    tok = AutoTokenizer.from_pretrained(repo, revision=sha)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
    mdl = AutoModelForCausalLM.from_pretrained(repo, revision=sha, dtype=torch.bfloat16, device_map="cuda").eval()
    eos = mdl.generation_config.eos_token_id if mdl.generation_config.eos_token_id is not None else tok.eos_token_id
    EOS = set(eos if isinstance(eos, (list, tuple)) else [eos])
    log("attn:", mdl.config._attn_implementation, "| pad:", tok.pad_token, tok.pad_token_id, "| eos:", sorted(EOS),
        "| padding_side:", tok.padding_side)

    # 4. resume state
    out_path = out_dir / f"{tag}.json"
    meta = {"checkpoint": a.checkpoint, "model_repo": repo, "sha": sha, "env_hash": env_hash, "dry_run": dry,
            "limit": a.limit, "batch_size": BATCH_SIZE, "temperature": TEMPERATURE, "top_p": TOP_P, "caps": CAPS,
            "order_file_meta": doc["meta"]}
    rows = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if (prev["meta"]["sha"], prev["meta"]["env_hash"], prev["meta"]["dry_run"]) != (sha, env_hash, dry):
            sys.exit("existing results file was produced with a different checkpoint SHA / environment / mode; refusing to mix")
        rows = prev["rows"]
        log(f"resuming: {len(rows)} rows already saved")
    done_batches = {r["batch_index"] for r in rows}
    ids_by_batch = {b["batch_index"]: b["ids"] for b in batches}
    for bi in done_batches:
        saved = [r["id"] for r in rows if r["batch_index"] == bi]
        if bi not in ids_by_batch or sorted(saved) != sorted(ids_by_batch[bi]):
            sys.exit(f"saved rows of batch {bi} do not match the current run order; refusing to resume (use a fresh --out-dir)")

    # 5. generate
    n_gen = sum(len(b["ids"]) for b in todo)
    t_start, n_done_tok = time.time(), 0
    for b in todo:
        if b["batch_index"] in done_batches:
            continue
        its = [by_id[b["component"]][i] for i in b["ids"]]
        id_lists = [tok.apply_chat_template(it["messages"], add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True)["input_ids"][0].tolist() for it in its]
        enc = tok.pad({"input_ids": id_lists}, padding=True, return_tensors="pt").to(mdl.device)
        L = enc["input_ids"].shape[1]
        torch.manual_seed(b["seed"])
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            g = mdl.generate(**enc, max_new_tokens=b["max_new_tokens"], pad_token_id=tok.pad_token_id,
                             do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
        torch.cuda.synchronize()
        wall = time.time() - t
        for j, it in enumerate(its):
            toks = g[j, L:].tolist()
            cut, ended = len(toks), False
            for k, x in enumerate(toks):
                if x in EOS:
                    cut, ended = k + 1, True
                    break
            toks = toks[:cut]
            text = tok.decode(toks, skip_special_tokens=True)
            p = answer_parser.parse_generation(text, it["parser_kind"], len(toks), b["max_new_tokens"], ended)
            assert p["status"] != "unclosed", "token info was supplied; 'unclosed' must not occur"
            rows.append({"id": it["id"], "component": b["component"], "seed": b["seed"], "batch_index": b["batch_index"],
                         "raw_output": text, "n_tokens": len(toks), "hit_eos": ended, "hit_cap": p["hit_cap"],
                         "max_new_tokens": b["max_new_tokens"], "closed": p["closed"], "status": p["status"],
                         "answer": p["answer"], "confidence": p["confidence"], "reason": p["reason"],
                         "boxed_candidate": p.get("boxed_candidate"),   # kind=math only; never used in the primary score
                         "tokens_per_s": len(toks) / wall, "batch_wall_s": wall, "prompt_tokens": len(id_lists[j]),
                         "padded_prompt_len": L, "checkpoint": a.checkpoint, "sha": sha, "env_hash": env_hash})
        del g, enc
        save_atomic(out_path, {"meta": meta, "rows": rows})
        bt = sum(r["n_tokens"] for r in rows if r["batch_index"] == b["batch_index"])
        n_done_tok += bt
        stat = {}
        for r in rows:
            if r["batch_index"] == b["batch_index"]:
                stat[r["status"]] = stat.get(r["status"], 0) + 1
        done = sum(1 for r in rows if r["component"] in comps)
        log(f"batch {b['batch_index']:3d} {b['component']} seed {b['seed']} rows {len(its)} wall {wall:6.0f}s tokens {bt:6d} "
            f"{stat} | {done}/{n_gen} rows | elapsed {(time.time() - t_start) / 60:.1f} min")
    log("ALL BATCHES DONE; rows:", len(rows))
    sm = {}
    for r in rows:
        sm.setdefault(r["component"], {}).setdefault(r["status"], 0)
        sm[r["component"]][r["status"]] += 1
    log("status by component:", json.dumps(sm))
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print("FAILED:", e, flush=True)
        raise
    except BaseException:
        traceback.print_exc()
        print("FAILED: exception", flush=True)
        sys.exit(1)
