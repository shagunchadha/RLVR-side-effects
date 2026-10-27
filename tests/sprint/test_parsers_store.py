"""CPU tests for exp.parsers, exp.store, exp.config and exp.versioncheck (no model needed)."""
import json

import pytest

from exp import config as C
from exp import parsers as P
from exp import store as S
from exp.versioncheck import read_pins


# ------------------------------------------------------------------ parsers
def test_split_think():
    assert P.split_think("abc</think>\nANSWER: 3") == ("abc", "\nANSWER: 3", True)
    assert P.split_think("no close") == ("no close", "", False)


def test_sentences_counts():
    assert len(P.sentences("Hmm. Wait, no. The answer is 3.")) == 3          # 'no.' ends a sentence
    assert len(P.sentences("Use e.g. this. And 2.5 is a number.")) == 2      # abbreviation and decimal protected
    assert len(P.sentences("One\n\nTwo")) == 2                                # blank line is a boundary
    assert P.sentences("...") == []


@pytest.mark.parametrize("text,val,rule", [
    ("think</think>\nIt fits.\nANSWER: 4", 4, "answer_line"),
    ("think</think>\nOn a scale of 1 to 5 I'd say 2.", 2, "last_digit"),
    ("think</think>\nI pick 5 out of 5.", 5, "last_digit"),      # '5 out of 5' -> the rating 5 survives
    ("think</think>\nRated on a 1-5 scale: no number given.", None, "none"),
    ("think only, no close 3", None, "unclosed"),
    ("t</think>\n**ANSWER: 3**", 3, "answer_line"),
])
def test_lenient_likert(text, val, rule):
    assert P.lenient_likert(text) == (val, rule)


@pytest.mark.parametrize("text,val", [
    ("t</think>\nANSWER: C", "C"),
    ("t</think>\nThe answer is B because...", "B"),
    ("t</think>\nI choose (D).", "D"),
    ("t</think>\n**E**", "E"),
    ("t</think>\nA\n", "A"),
    ("t</think>\nNo letter here.", None),
])
def test_lenient_letter(text, val):
    assert P.lenient_letter(text)[0] == val


def test_strict_answer_uses_frozen_parser():
    r = P.strict_answer("x</think>\nfine\nANSWER: 3", "likert", 10, 4000, True)
    assert r["status"] == "valid" and r["answer"] == 3


def test_sentence_end_offsets():
    t = "One. Two!\n\nThree"
    offs = P.sentence_end_char_offsets(t)
    assert t[:offs[0]] == "One." and 9 in offs


# ------------------------------------------------------------------ config
def test_config_hash_order_invariant():
    assert C.config_hash({"a": 1, "b": [1, 2]}) == C.config_hash({"b": [1, 2], "a": 1})
    assert C.config_hash({"a": 1}) != C.config_hash({"a": 2})


def test_checkpoints_pinned():
    for k, (repo, sha) in C.CHECKPOINTS.items():
        assert repo.startswith("allenai/") and len(sha) == 40
    assert C.ckpt("rlvr")[1] == "d97e442d7cc678210054dbcc9b440894d62c89a4"


def test_read_pins(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("# c\nfoo==1.0\n    # via x\nBar_Baz[extra]==2.3.4 ; python_version>'3'\n")
    assert read_pins(p) == {"foo": "1.0", "bar-baz": "2.3.4"}


# ------------------------------------------------------------------ store
def test_shard_helpers():
    assert S.shard_slice(range(10), 1, 3) == [1, 4, 7]
    assert S.parse_shard("2/5") == (2, 5) and S.parse_shard(None) == (0, 1)
    with pytest.raises(ValueError):
        S.shard_slice(range(3), 3, 3)


def test_local_store_roundtrip_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", "local")
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    cfg = {"job": "T", "x": 1}
    st = S.Store("T", "dpo", cfg, root=tmp_path)
    assert not st.done(0)
    loc = st.write(0, 2, [{"id": "a", "v": [1.0, 2.0]}, {"id": "b", "v": [3.0]}], model_repo="r", model_sha="s")
    assert loc.endswith(f"runs/T/dpo/{C.config_hash(cfg)}/shard_000.parquet")
    assert st.done(0) and st.list_shards() == ["shard_000.parquet"]
    t = st.read(0)
    assert t.column("id").to_pylist() == ["a", "b"]
    meta = S.read_metadata(t)
    assert meta["git_sha"] == "deadbeef" and meta["model_sha"] == "s" and meta["config"] == cfg
    assert meta["shard"] == "0" and meta["n_shards"] == "2" and "python" in meta["package_versions"]
    with pytest.raises(S.ShardExists):
        st.write(0, 2, [{"id": "c", "v": [0.0]}])
    assert st.read(0).num_rows == 2                     # unchanged
    cfgfile = tmp_path / f"runs/T/dpo/{C.config_hash(cfg)}/config.json"
    assert json.loads(cfgfile.read_text()) == cfg
    runs = list(S.iter_local_runs(tmp_path))
    assert {r[3].name for r in runs} == {"config.json", "shard_000.parquet"}


def test_store_rejects_bad_backend(tmp_path):
    with pytest.raises(ValueError):
        S.Store("T", "dpo", {}, backend="s3", root=tmp_path)
