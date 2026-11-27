"""Freeze the F1 item sets (registration amendment 18 and its Nov 27 notes) from their sources at pinned revisions.

    python items_revision/build_F1.py        # writes items_revision/F1_*.json and items_revision/MANIFEST.sha256

Sources (each file's sha256 is recorded in the item file's meta and checked on every rebuild):
  MATH-500   HuggingFaceH4/MATH-500 @ 6e4ed1a2 test.jsonl; cross-checked against OpenAI's original split
             (openai/prm800k, prm800k/math_splits/test.jsonl, LFS oid 35dc4108...): identical, 500/500 in order.
  AIME 2024  allenai/aime-2021-2025 @ 73e1eba7 (the developers' OLMES source), year 2024 (30); identical to
             HuggingFaceH4/aime_2024 @ 2fe88a2f.
  AIME 2025  allenai/aime-2021-2025 @ 73e1eba7, year 2025 (30) - the PI's choice of Nov 27 over the registered
             MathArena/aime_2025 (same answers, different wording in 27/30; amendment 18 note).
  IFEval     google/IFEval @ 966cd895 ifeval_input_data.jsonl (= HuggingFaceH4/ifeval, OLMES's source); 541.
  IFBench    allenai/IFBench_test @ ed7c1245 (OLMES's source; = ifbench/data/IFBench_test.jsonl in allenai/IFBench
             @ 48891866); 300.
Prompts, verbatim from OLMES @ 5a51f502 (configs olmo3:adapt; one user turn, the checkpoint's native chat template):
  MATH-500   problem.strip() + "\\n\\nPresent the answer in LaTex format: \\\\boxed{Your answer}"   (cot_style plain)
  AIME       problem          + "\\n\\nPresent the answer in LaTex format: \\\\boxed{Your answer}"   (aime.py _process_doc)
  IFEval / IFBench  the prompt as given.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUFFIX = "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
OLMES = "allenai/olmes@5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
SOURCES = {
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "aime": ("allenai/aime-2021-2025", "73e1eba765ad5847cdb5d1e2e7aaf7b22b585798", "data/train-00000-of-00001.parquet"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl"),
    "ifbench": ("allenai/IFBench_test", "ed7c12452594b7028b15807b86d3938150b663ae", "data/train-00000-of-00001.parquet"),
}
OPENAI_MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def fetch(key):
    from huggingface_hub import hf_hub_download
    repo, rev, f = SOURCES[key]
    p = Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
    b = p.read_bytes()
    meta = {"source": f"hf://datasets/{repo}@{rev}/{f}", "source_sha256": hashlib.sha256(b).hexdigest()}
    if f.endswith(".jsonl"):
        rows = [json.loads(l) for l in b.decode("utf-8").splitlines() if l.strip()]
    else:
        import pyarrow.parquet as pq
        rows = pq.read_table(p).to_pylist()
    return rows, meta


def user(content):
    return [{"role": "user", "content": content}]


def build():
    from exp.revision.scorers import olmes_aime_gold, olmes_math_gold
    out = {}
    rows, meta = fetch("math500")
    out["math500"] = (meta | {"prompt": f"problem.strip() + {SUFFIX!r} ({OLMES}, minerva_math_{{type}}::olmo3:adapt, cot_style plain)",
                              "cross_check": f"openai/prm800k math_splits/test.jsonl (LFS sha256 {OPENAI_MATH500_SHA256}): identical problem/answer/unique_id, 500/500"},
                      [{"id": r["unique_id"], "gold": r["answer"], "olmes_gold": olmes_math_gold(r["solution"]), "level": r["level"],
                        "subject": r["subject"], "kind": "math", "messages": user(r["problem"].strip() + SUFFIX)} for r in rows])
    rows, meta = fetch("aime")
    for yr in (2024, 2025):
        ys = sorted((r for r in rows if int(r["year"]) == yr), key=lambda r: int(r["id"]))
        out[f"aime{yr}"] = (meta | {"filter": f"year == {yr}", "prompt": f"problem + {SUFFIX!r} ({OLMES}, aime.py _process_doc)"},
                            [{"id": f"aime{yr}-{r['id']}", "gold": str(r["answer"]), "olmes_gold": olmes_aime_gold(r["answer"]),
                              "url": r["url"], "kind": "aime", "messages": user(r["problem"] + SUFFIX)} for r in ys])
    for key in ("ifeval", "ifbench"):
        rows, meta = fetch(key)
        items = []
        for r in rows:
            kw = [{k: v for k, v in d.items() if v is not None} for d in r["kwargs"]]   # parquet fills absent keys with None
            items.append({"id": f"{key}-{r['key']}", "key": r["key"], "prompt": r["prompt"],
                          "instruction_id_list": list(r["instruction_id_list"]), "kwargs": kw, "kind": key,
                          "messages": user(r["prompt"])})
        out[key] = (meta | {"prompt": "the prompt as given (one user turn)"}, items)
    n = {"math500": 500, "aime2024": 30, "aime2025": 30, "ifeval": 541, "ifbench": 300}
    man = []
    for k, (meta, items) in out.items():
        assert len(items) == n[k], (k, len(items))
        assert len({i["id"] for i in items}) == len(items), k
        f = HERE / f"F1_{k}.json"
        f.write_text(json.dumps({"meta": meta | {"n": len(items), "registration": "amendment 18 (F1) and its Nov 27 notes"},
                                 "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
        man.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
        print(f.name, len(items), meta["source"], meta["source_sha256"][:16])
    (HERE / "MANIFEST.sha256").write_text("\n".join(man) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))
    build()
