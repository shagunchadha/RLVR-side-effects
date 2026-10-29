"""hfsync: a notebook never pushes files it only pulled (the Oct 30 cross-lane overwrite of B4 rl_zero_general)."""
import os
import time
from pathlib import Path

import huggingface_hub

from exp import hfsync


def _fake_snapshot(files):
    def snap(repo, repo_type=None, allow_patterns=None, local_dir=None):
        for rel, text in files.items():
            p = Path(local_dir) / "progress" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    return snap


def test_pulled_files_are_not_pushed_until_changed(tmp_path, monkeypatch):
    root = tmp_path / "out"
    (root / "B4" / "general").mkdir(parents=True)
    (root / "B4" / "general" / "gen.csv").write_text("a\n" * 5)          # ours, longer than HF's: kept and pushed
    files = {"B4/general/gen.csv": "a\n" * 3, "B6/pos/gen.csv": "b\n" * 4, "_queue/N2/B5.done": ""}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(files))
    hfsync.pull(root, "x/y")
    skip = hfsync.unchanged_since_pull(root)
    assert "B6/pos/gen.csv" in skip and "_queue/N2/B5.done" in skip     # pulled copies of other lanes' files
    assert "B4/general/gen.csv" not in skip                            # local copy was longer: ours
    time.sleep(0.01)
    with open(root / "B6" / "pos" / "gen.csv", "a") as f:               # this notebook now writes to it
        f.write("b\n")
    os.utime(root / "B6" / "pos" / "gen.csv")
    assert "B6/pos/gen.csv" not in hfsync.unchanged_since_pull(root)


def test_push_passes_the_skip_list(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot({"M6/x.csv": "z\n"}))
    hfsync.pull(root, "x/y")
    seen = {}

    class Api:
        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **k):
            seen.update(k)
    monkeypatch.setattr(hfsync, "_api", lambda: Api())
    hfsync.push(root, "x/y", "N2")
    assert "M6/x.csv" in seen["ignore_patterns"] and "**/_hf/**" in seen["ignore_patterns"]
