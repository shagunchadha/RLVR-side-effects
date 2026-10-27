"""B0: checkpoint inventory (no GPU, no model weights downloaded).

For every Olmo 3 repo in scope: main SHA, last modified, all refs (branches incl. intermediate RL step branches, tags),
total safetensors size, presence of generation_config.json and its sampling defaults, and the rendered chat
template for one component A item, with a check for an injected system / identity prompt.

Outputs: docs/sprint/checkpoint_inventory.md and results/sprint/B0_inventory.json (overwritten each run; this job
writes documentation, not result shards).
"""
import datetime as dt
import json
import re

from .. import config as C

# families requested in the sprint brief; exact repo names are discovered on the Hub and matched against these
FAMILIES = {
    "7B Think ladder": [r"^allenai/Olmo-3-7B-Think(-SFT|-DPO)?$"],
    "7B Instruct ladder": [r"^allenai/Olmo-3-7B-Instruct(-SFT|-DPO)?$"],
    "32B Think ladder": [r"^allenai/Olmo-3-32B-Think(-SFT|-DPO)?$"],
    "Olmo 3.1 Think": [r"^allenai/Olmo-3\.1-.*Think.*$"],
    "Base": [r"^allenai/Olmo-3-1025-7B$"],
    "7B RL-Zero": [r"^allenai/Olmo-3-7B-RL-Zero-(Math|Code|IF|General|Mix)$"],
}
EXPECTED = ["allenai/Olmo-3-7B-Think", "allenai/Olmo-3-7B-Think-SFT", "allenai/Olmo-3-7B-Think-DPO",
            "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-DPO",
            "allenai/Olmo-3-32B-Think", "allenai/Olmo-3-32B-Think-SFT", "allenai/Olmo-3-32B-Think-DPO",
            "allenai/Olmo-3-1025-7B"] + [f"allenai/Olmo-3-7B-RL-Zero-{x}" for x in ("Math", "Code", "IF", "General", "Mix")]
IDENTITY_PATTERNS = [r"<\|im_start\|>system", r"\bYou are\b", r"\bOlmo\b", r"\bAi2\b", r"\bAllen Institute\b",
                     r"\bknowledge cutoff\b", r"\bdate\b"]


def discover(api):
    names = set(EXPECTED)
    for q in ("Olmo-3-", "Olmo-3.1-"):
        for m in api.list_models(author="allenai", search=q, limit=500):
            names.add(m.id)
    inscope = {}
    for n in sorted(names):
        for fam, pats in FAMILIES.items():
            if any(re.match(p, n) for p in pats):
                inscope[n] = fam
    return inscope


def render_template(repo, sha, messages):
    """Render with transformers if importable (exact), else with jinja2 using the HF chat-template conventions."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo, revision=sha)
        if not getattr(tok, "chat_template", None):
            return None, "no chat_template"
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False), "transformers"
    except ImportError:
        pass
    except Exception as e:      # e.g. a config transformers cannot parse: fall back to rendering the template directly
        fallback_reason = f"transformers failed ({type(e).__name__}); "
    else:
        fallback_reason = ""
    from huggingface_hub import hf_hub_download
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tmpl, cfg = None, {}
    try:
        cfg = json.load(open(hf_hub_download(repo, "tokenizer_config.json", revision=sha), encoding="utf-8"))
        tmpl = cfg.get("chat_template")
    except Exception:
        pass
    if not tmpl:
        try:
            tmpl = open(hf_hub_download(repo, "chat_template.jinja", revision=sha), encoding="utf-8").read()
        except Exception:
            return None, "no chat template found"
    if isinstance(tmpl, list):
        tmpl = next((t["template"] for t in tmpl if t.get("name") == "default"), tmpl[0]["template"])

    def raise_exception(msg):
        raise RuntimeError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda f: dt.datetime.now().strftime(f)
    tok_str = lambda v: v.get("content") if isinstance(v, dict) else v  # noqa: E731
    out = env.from_string(tmpl).render(messages=messages, add_generation_prompt=True, tools=None,
                                       bos_token=tok_str(cfg.get("bos_token")) or "", eos_token=tok_str(cfg.get("eos_token")) or "")
    return out, (locals().get("fallback_reason") or "") + "jinja2"


def identity_check(rendered, user_content):
    if rendered is None:
        return {"injected": None, "matches": []}
    outside = rendered.replace(user_content, "")
    hits = []
    for line in outside.splitlines():
        for p in IDENTITY_PATTERNS:
            if re.search(p, line):
                hits.append(line.strip()[:200])
                break
    has_system = bool(re.search(r"<\|im_start\|>system|\"role\":\s*\"system\"|\bsystem\b\s*\n", outside))
    return {"injected": bool(hits) or has_system, "system_block": has_system, "matches": sorted(set(hits))}


def inventory_repo(api, repo):
    from huggingface_hub import hf_hub_download
    info = api.model_info(repo, files_metadata=True)
    refs = api.list_repo_refs(repo)
    size = sum((s.size or 0) for s in (info.siblings or []) if s.rfilename.endswith(".safetensors"))
    gen = None
    try:
        gen = json.load(open(hf_hub_download(repo, "generation_config.json", revision=info.sha), encoding="utf-8"))
    except Exception:
        pass
    a_item = C.load_items("A")[0]
    rendered, how = render_template(repo, info.sha, a_item["messages"])
    return {
        "repo": repo, "main_sha": info.sha,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "created": info.created_at.isoformat() if getattr(info, "created_at", None) else None,
        "safetensors_bytes": size,
        "branches": sorted([{"name": b.name, "sha": b.target_commit} for b in refs.branches], key=lambda x: x["name"]),
        "tags": sorted([{"name": t.name, "sha": t.target_commit} for t in refs.tags], key=lambda x: x["name"]),
        "base_model_tags": [t for t in (info.tags or []) if t.startswith("base_model:")],
        "generation_config": {k: gen.get(k) for k in ("temperature", "top_p", "top_k", "do_sample", "max_new_tokens",
                                                      "eos_token_id")} if gen else None,
        "chat_template_render": how, "rendered_A_item": rendered,
        "identity_check": identity_check(rendered, a_item["messages"][0]["content"]),
        "pinned_in_sprint": next((k for k, (r, s) in C.CHECKPOINTS.items() if r == repo), None),
        "pinned_sha_matches_main": next((s == info.sha for k, (r, s) in C.CHECKPOINTS.items() if r == repo), None),
    }


def _step_branches(branches):
    return [b for b in branches if re.search(r"step|stage|ckpt|rl", b["name"], re.I) and b["name"] != "main"]


def _system_text(rendered):
    if not rendered or "<|im_start|>system" not in rendered:
        return None
    return rendered.split("<|im_start|>system", 1)[1].split("<|im_end|>", 1)[0].strip()


def write_markdown(inv, path):
    L = ["# Olmo 3 checkpoint inventory (job B0)", "",
         f"Generated {inv['generated_utc']} by `python -m exp.run --job B0` (git {inv['git_sha']}). Read-only Hub "
         "metadata; no weights downloaded. Regenerate rather than edit.", "",
         "## Summary", "",
         "| family | repo | main SHA | last modified | safetensors size (GB) | branches | step-like branches | tags | generation defaults (T / top_p / top_k) | injected system/identity text |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in inv["repos"]:
        g = r["generation_config"] or {}
        gd = f"{g.get('temperature')} / {g.get('top_p')} / {g.get('top_k')}" if g else "absent"
        ic = r["identity_check"]
        L.append(f"| {r['family']} | `{r['repo']}` | `{r['main_sha'][:12]}` | {(r['last_modified'] or '')[:10]} | "
                 f"{r['safetensors_bytes'] / 1e9:.1f} | {len(r['branches'])} | {len(_step_branches(r['branches']))} | "
                 f"{len(r['tags'])} | {gd} | {ic['injected']} |")
    if inv["missing_expected"]:
        L += ["", "**Expected but not found on the Hub:** " + ", ".join(f"`{m}`" for m in inv["missing_expected"])]
    L += ["", "## Default system prompts injected by the chat templates", "",
          "Rendering one component A item with only a user message (as `scripts/full_run.py` did). The template adds the "
          "system block below when no system message is given. Distinct texts and the repos that inject them:", ""]
    groups = {}
    for r in inv["repos"]:
        txt = _system_text(r.get("rendered_A_item"))
        groups.setdefault(txt, []).append(r["repo"])
    for txt, rs in groups.items():
        L += [f"- {', '.join(f'`{x}`' for x in rs)}:", f"  `{txt}`" if txt else "  (no system block)"]
    L += ["", "**Consequence for the frozen study.** `Olmo-3-7B-Think-SFT` injects a different default system prompt from "
          "`Olmo-3-7B-Think-DPO` and `Olmo-3-7B-Think` (which are byte-identical to each other). The frozen full run used each "
          "checkpoint's own template, so the frozen DPO - SFT comparison is confounded with a system-prompt difference "
          "(e.g. A_confidence_0: 163 prompt tokens for SFT vs 182 for DPO and RLVR), while RLVR - DPO is not."]
    L += ["", "## Pinned sprint checkpoints vs Hub main", "",
          "| key | repo | pinned SHA (docs/checkpoint_map.md) | matches current main |", "|---|---|---|---|"]
    for k, (repo, sha) in C.CHECKPOINTS.items():
        m = next((r["pinned_sha_matches_main"] for r in inv["repos"] if r["repo"] == repo), "not in inventory")
        L.append(f"| {k} | `{repo}` | `{sha}` | {m} |")
    L += ["", "## Per repo: refs, defaults, rendered template", ""]
    for r in inv["repos"]:
        L += [f"### `{r['repo']}` ({r['family']})", "",
              f"- main: `{r['main_sha']}`; last modified {r['last_modified']}; created {r['created']}",
              f"- base_model tags: {', '.join(r['base_model_tags']) or 'none'}",
              f"- generation_config.json: `{json.dumps(r['generation_config'])}`",
              f"- branches ({len(r['branches'])}): " + (", ".join(f"`{b['name']}`@`{b['sha'][:10]}`" for b in r["branches"]) or "none"),
              f"- tags ({len(r['tags'])}): " + (", ".join(f"`{t['name']}`@`{t['sha'][:10]}`" for t in r["tags"]) or "none"),
              f"- identity check: injected = {r['identity_check']['injected']}, system block = "
              f"{r['identity_check'].get('system_block')}; matched lines outside the user message: "
              + ("; ".join(f"`{x}`" for x in r["identity_check"]["matches"]) or "none"),
              f"- rendered with: {r['chat_template_render']}", "", "```text",
              (r["rendered_A_item"] or "(no template)").replace("```", "'''"), "```", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def run(args):
    from huggingface_hub import HfApi
    api = HfApi()
    scope = discover(api)
    repos = []
    for repo, fam in scope.items():
        print("inventory", repo, flush=True)
        try:
            r = inventory_repo(api, repo)
        except Exception as e:
            r = {"repo": repo, "error": repr(e), "main_sha": "", "branches": [], "tags": [], "safetensors_bytes": 0,
                 "generation_config": None, "identity_check": {"injected": None, "matches": []}, "last_modified": None,
                 "created": None, "base_model_tags": [], "chat_template_render": "error", "rendered_A_item": None,
                 "pinned_sha_matches_main": None}
        r["family"] = fam
        repos.append(r)
    inv = {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": C.git_sha(),
           "repos": repos, "missing_expected": [e for e in EXPECTED if e not in scope],
           "a_item_id": C.load_items("A")[0]["id"]}
    C.RESULTS_SPRINT.mkdir(parents=True, exist_ok=True)
    C.DOCS_SPRINT.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_SPRINT / "B0_inventory.json").write_text(json.dumps(inv, indent=1), encoding="utf-8")
    write_markdown(inv, C.DOCS_SPRINT / "checkpoint_inventory.md")
    print(f"wrote {C.DOCS_SPRINT / 'checkpoint_inventory.md'} ({len(repos)} repos)")
    return 0
