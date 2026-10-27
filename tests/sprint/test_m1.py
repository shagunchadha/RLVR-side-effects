"""CPU tests for M1 on tiny synthetic safetensors checkpoints with known weight differences."""
import argparse
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from exp.jobs import M1_weightdiff as M  # noqa: E402


def _save_ckpt(d, tensors, split=True):
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    files = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
             "model-00002-of-00002.safetensors": names[len(names) // 2:]} if split else {"model.safetensors": names}
    wm = {}
    for fn, ns in files.items():
        save_file({k: tensors[k].contiguous() for k in ns}, str(d / fn))
        wm.update({k: fn for k in ns})
    if split:
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))


@pytest.fixture()
def ladder(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, m = 24, 16
    base = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(n, m, generator=g),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(n, m, generator=g),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(n),
    }
    u = torch.zeros(n); u[0] = 1.0
    v = torch.zeros(m); v[0] = 1.0
    w = torch.zeros(m); w[1] = 1.0
    sft = {k: t.clone() for k, t in base.items()}
    dpo = {k: t.clone() for k, t in base.items()}
    rlvr = {k: t.clone() for k, t in base.items()}
    # down_proj: DPO-SFT = 2 u v^T (rank 1), RLVR-DPO = 0.5 u w^T (same left direction, orthogonal right direction)
    dpo["model.layers.0.mlp.down_proj.weight"] += 2.0 * torch.outer(u, v)
    rlvr["model.layers.0.mlp.down_proj.weight"] = dpo["model.layers.0.mlp.down_proj.weight"] + 0.5 * torch.outer(u, w)
    # q_proj: identical across stages (zero updates)
    # norm: 1-D update
    rlvr["model.layers.0.post_attention_layernorm.weight"] += 0.1
    _save_ckpt(tmp_path / "sft", sft)
    _save_ckpt(tmp_path / "dpo", dpo, split=False)
    _save_ckpt(tmp_path / "rlvr", rlvr)
    return {k: M.Checkpoint(local_dir=str(tmp_path / k)) for k in ("sft", "dpo", "rlvr")}, base


def test_checkpoint_access(ladder):
    ck, base = ladder
    for c in ck.values():
        assert c.names() == set(base)
    assert torch.equal(ck["sft"].get("model.layers.1.self_attn.q_proj.weight"), base["model.layers.1.self_attn.q_proj.weight"])


def test_parse_name():
    assert M.parse_name("model.layers.12.mlp.gate_proj.weight") == (12, "mlp.gate_proj")
    assert M.parse_name("model.embed_tokens.weight") == (-1, "embed_tokens")
    assert M.parse_name("lm_head.weight") == (-1, "lm_head")


def test_rank1_update_metrics(ladder):
    ck, base = ladder
    name = "model.layers.0.mlp.down_proj.weight"
    rows, angle = M.tensor_rows(name, {k: ck[k].get(name) for k in ck}, "cpu")
    r = {x["comparison"]: x for x in rows}
    dpo_sft = r["DPO-SFT"]
    assert math.isclose(dpo_sft["dw_norm"], 2.0, rel_tol=1e-6)
    assert math.isclose(dpo_sft["rel_dw"], 2.0 / float(torch.linalg.norm(base[name])), rel_tol=1e-6)
    assert math.isclose(dpo_sft["eff_rank"], 1.0, abs_tol=1e-4) and math.isclose(dpo_sft["stable_rank"], 1.0, rel_tol=1e-6)
    assert math.isclose(dpo_sft["sv_1"], 2.0, rel_tol=1e-6) and dpo_sft["sv_2"] < 1e-5
    assert math.isclose(r["RLVR-DPO"]["sv_1"], 0.5, rel_tol=1e-6)
    assert dpo_sft["num_rank"] == 1 and r["RLVR-DPO"]["num_rank"] == 1
    # same left direction u: the leading principal cosine is 1 (only the first is meaningful at rank 1)
    assert math.isclose(angle["cos_left_8_1"], 1.0, abs_tol=1e-5)
    assert math.isclose(angle["random_baseline_left_8"], 8 / 24)


def test_zero_update_and_1d(ladder):
    ck, _ = ladder
    rows, _ = M.tensor_rows("model.layers.1.self_attn.q_proj.weight",
                            {k: ck[k].get("model.layers.1.self_attn.q_proj.weight") for k in ck}, "cpu")
    assert all(x["dw_norm"] == 0 and x["eff_rank"] == 0 for x in rows)
    rows, angle = M.tensor_rows("model.layers.0.post_attention_layernorm.weight",
                                {k: ck[k].get("model.layers.0.post_attention_layernorm.weight") for k in ck}, "cpu")
    r = {x["comparison"]: x for x in rows}
    assert angle is None and "eff_rank" not in r["RLVR-DPO"]
    assert math.isclose(r["RLVR-DPO"]["rel_dw"], 0.1, rel_tol=1e-5) and r["DPO-SFT"]["dw_norm"] == 0


def test_job_end_to_end_with_shards(ladder, tmp_path, monkeypatch):
    ck, _ = ladder
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("STORE", "local")
    out = tmp_path / "M1out"
    for i in range(2):
        a = argparse.Namespace(shard=f"{i}/2", job_args=argparse.Namespace(device="cpu", assemble=False, max_tensors=None))
        M.run(a, ckpts=ck, out_dir=out)
    import csv
    rows = list(csv.DictReader(open(out / "weightdiff.csv")))
    assert len(rows) == 6                                      # 3 tensors x 2 comparisons
    ang = list(csv.DictReader(open(out / "subspace_angles.csv")))
    assert len(ang) == 2                                       # two 2-D tensors
    assert (out / "heatmap_M1.png").exists() and (out / "heatmap_M1_ratio.png").exists()
    # resume: re-running a done shard writes nothing new and does not raise
    a = argparse.Namespace(shard="0/2", job_args=argparse.Namespace(device="cpu", assemble=False, max_tensors=None))
    assert M.run(a, ckpts=ck, out_dir=out) == 0
    np.testing.assert_allclose(float(next(r for r in rows if r["comparison"] == "DPO-SFT"
                                          and r["module"] == "mlp.down_proj")["dw_norm"]), 2.0, rtol=1e-6)


def test_principal_cosines_known_subspaces():
    I = torch.eye(10)
    assert np.allclose(M.principal_cosines(I[:, :3], I[:, :3], 3), [1, 1, 1])
    assert np.allclose(M.principal_cosines(I[:, :3], I[:, 3:6], 3), [0, 0, 0], atol=1e-7)
    th = 0.3
    B = I[:, :2].clone(); B[:, 1] = math.cos(th) * I[:, 1] + math.sin(th) * I[:, 5]
    c = M.principal_cosines(I[:, :2], B, 2)
    assert np.allclose(sorted(c), sorted([1.0, math.cos(th)]), atol=1e-6)
