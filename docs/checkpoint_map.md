# Olmo 3 7B Checkpoint Map

Generated 2026-10-14, read-only (no notebook cells run, no GPU). Source: notebook cell
`MJUe` ("checkpoint_discovery") static source, cross-checked against live Hugging Face
Hub metadata via a local `huggingface_hub` session (not the notebook kernel).

## 1. `OLMO3_7B_PIPELINE` (as defined in notebook cell `MJUe`)

```python
OLMO3_7B_PIPELINE = {
    "base": {
        "repo": "allenai/Olmo-3-1025-7B",
        "stage": "pretrain",
    },
    "think_sft": {
        "repo": "allenai/Olmo-3-7B-Think-SFT",
        "stage": "SFT",
        "base_model": "allenai/Olmo-3-1025-7B",
    },
    "think_dpo": {
        "repo": "allenai/Olmo-3-7B-Think-DPO",
        "stage": "DPO",
        "base_model": "allenai/Olmo-3-7B-Think-SFT",
    },
    "think_rlvr": {
        "repo": "allenai/Olmo-3-7B-Think",
        "stage": "RLVR (final)",
        "base_model": "allenai/Olmo-3-7B-Think-DPO",
        "rl_dataset": "allenai/Dolci-Think-RL-7B",
    },
    "rl_zero_math": {
        "repo": "allenai/Olmo-3-7B-RL-Zero-Math",
        "stage": "RLVR direct from base (RL-Zero)",
        "base_model": "allenai/Olmo-3-1025-7B",
        "rl_dataset": "allenai/Dolci-RLZero-Math-7B",
    },
    "rl_zero_code": {"repo": "allenai/Olmo-3-7B-RL-Zero-Code", "stage": "RLVR direct from base (RL-Zero)", "base_model": "allenai/Olmo-3-1025-7B"},
    "rl_zero_if": {"repo": "allenai/Olmo-3-7B-RL-Zero-IF", "stage": "RLVR direct from base (RL-Zero)", "base_model": "allenai/Olmo-3-1025-7B"},
    "rl_zero_general": {"repo": "allenai/Olmo-3-7B-RL-Zero-General", "stage": "RLVR direct from base (RL-Zero)", "base_model": "allenai/Olmo-3-1025-7B"},
    "rl_zero_mix": {"repo": "allenai/Olmo-3-7B-RL-Zero-Mix", "stage": "RLVR direct from base (RL-Zero)", "base_model": "allenai/Olmo-3-1025-7B"},
}
```

**Revision info in the dict itself: none.** No entry pins a commit SHA or named revision
(e.g. `step_XXXX` / `stageN-stepXXX`) — every repo resolves to its default `main` branch
when loaded. Separately, `MJUe` also calls `hf_api.list_repo_refs(...)` per repo and stores
the discovered intermediate step branches in `_checkpoint_revisions`, but that variable is
never consumed by the model-loading cell (`vblA`), so the step-level checkpoints it finds
are not actually wired into anything that loads a model.

## 2. Live Hugging Face Hub state (queried locally, not via the notebook)

| key | repo | main commit SHA | last modified (UTC) | created (UTC) |
|---|---|---|---|---|
| base | `allenai/Olmo-3-1025-7B` | `a81bae42db3975be1671e27b9c9a56da1a9f980f` | 2026-04-21T17:21:02 | 2025-09-12T19:24:57 |
| think_sft | `allenai/Olmo-3-7B-Think-SFT` | `6ff857587e040d6d523a3d5f3a56e918f5401d66` | 2026-01-05T16:25:46 | 2025-10-14T20:14:36 |
| think_dpo | `allenai/Olmo-3-7B-Think-DPO` | `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38` | 2026-06-25T21:42:31 | 2025-11-18T04:20:27 |
| think_rlvr | `allenai/Olmo-3-7B-Think` | `d97e442d7cc678210054dbcc9b440894d62c89a4` | 2026-06-25T21:41:28 | 2025-11-18T01:22:27 |
| rl_zero_math | `allenai/Olmo-3-7B-RL-Zero-Math` | `dee26a4074b88215cd1326472e1ab14d0259d892` | 2026-06-25T21:42:10 | 2025-11-17T17:36:01 |
| rl_zero_code | `allenai/Olmo-3-7B-RL-Zero-Code` | `a961af69d75f9c52b3ae4db7b1664d890229b7ca` | 2026-06-25T21:41:48 | 2025-11-17T17:36:01 |
| rl_zero_if | `allenai/Olmo-3-7B-RL-Zero-IF` | `3cae007c5324a9a06ee6780f03b848bc7b8fd440` | 2026-06-25T21:42:01 | 2025-11-17T17:36:01 |
| rl_zero_general | `allenai/Olmo-3-7B-RL-Zero-General` | `24e5814cb36944d3eb36a49679d02c2b3bb46195` | 2026-06-25T21:41:55 | 2025-11-17T17:36:01 |
| rl_zero_mix | `allenai/Olmo-3-7B-RL-Zero-Mix` | `73b2b1a50d94aec859e22c9b4a524416b5a3a739` | 2026-06-25T21:42:23 | 2025-12-01T19:14:43 |

(Retrieved via `HfApi().model_info(repo)`, `files_metadata=False`.)

## 3. Does the stage order (base → SFT → DPO → RLVR) and DPO→RLVR lineage hold up?

**Yes — confirmed by both the machine-readable Hub tags and the model card prose.**

The authoritative source is each repo's `base_model` tag (set from the card's YAML
front matter, shown in `HfApi().model_info(repo).tags`):

- `allenai/Olmo-3-7B-Think-SFT` tags include: `'base_model:allenai/Olmo-3-1025-7B'`, `'base_model:finetune:allenai/Olmo-3-1025-7B'`
- `allenai/Olmo-3-7B-Think-DPO` tags include: `'base_model:allenai/Olmo-3-7B-Think-SFT'`, `'base_model:finetune:allenai/Olmo-3-7B-Think-SFT'`
- `allenai/Olmo-3-7B-Think` (RLVR, final) tags include: `'base_model:allenai/Olmo-3-7B-Think-DPO'`, `'base_model:finetune:allenai/Olmo-3-7B-Think-DPO'`

This directly confirms: base → SFT → DPO → **RLVR was fine-tuned from the DPO checkpoint**.

The model card prose (identical table in all four cards) states the same staged pipeline:

> | **Stage** | **Olmo 3 7B Think** | ... |
> | **SFT** | [Olmo-3-7B-Think-SFT](...) | ... |
> | **DPO** | [Olmo-3-7B-Think-DPO](...) | ... |
> | **Final Models (RLVR)** | [Olmo-3-7B-Think](...) | ... |

And the `Olmo-3-7B-Think-SFT` / `-DPO` / `-Think` cards all carry the identical staged
description:

> #### Stage 1: SFT
> - supervised fine-tuning on the Dolci-Think-SFT-7B dataset...
>
> #### Stage 2:DPO
> - direct preference optimization on the Dolci-Think-DPO-7B dataset...
>
> #### Stage 3: RLVR
> - reinforcement learning from verifiable rewards on the Dolci-Think-RL-7B dataset...

**Caveat, stated plainly:** the card *prose* under "Stage 3: RLVR" describes the RL
dataset used, but never spells out in words "RLVR training started from the DPO
checkpoint." That specific claim is only explicit in the `base_model` / `base_model:finetune`
tags, not in a sentence of readable text. If you need a citable sentence rather than a
tag, the tag is the only thing on the card that says it.

## 4. Has `allenai/Olmo-3-7B-Think` changed since 2026-09-28 (the v3 run)?

**No.** `last_modified` for `allenai/Olmo-3-7B-Think` is `2026-06-25T21:41:28Z`, which is
~3 months *before* 2026-09-28. The `main` revision has not changed since the v3 run — the
commit SHA (`d97e442d7cc678210054dbcc9b440894d62c89a4`) is the same one v3 would have
pulled.
