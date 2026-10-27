"""Pluggable shard storage.

STORE=local (default): shards under $RESULTS_DIR (default ./sprint_out/).
STORE=hf: the same layout in the private dataset repo $HF_REPO (default shagunchadha/rlvr-se-runs); needs HF_TOKEN.

Layout:  runs/<job>/<ckpt>/<config_hash>/shard_XXX.parquet   (+ config.json beside the shards)

Rules: a shard that exists is never overwritten (write() raises); jobs resume by asking exists() first.
Every shard carries file-level metadata: git_sha, model_repo, model_sha, config (JSON), package_versions,
created_utc, shard, n_shards, job, ckpt, config_hash.
"""
import io
import json
import os
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

DEFAULT_HF_REPO = "shagunchadha/rlvr-se-runs"


class ShardExists(RuntimeError):
    pass


def shard_name(i):
    return f"shard_{i:03d}.parquet"


def shard_slice(seq, i, n):
    """Deterministic round-robin split: shard i of n gets seq[i::n]. 0 <= i < n."""
    if not (0 <= i < n):
        raise ValueError(f"shard {i}/{n} out of range")
    return list(seq)[i::n]


def parse_shard(s):
    """'3/8' -> (3, 8); None -> (0, 1)."""
    if not s:
        return 0, 1
    i, n = s.split("/")
    return int(i), int(n)


class Store:
    def __init__(self, job, ckpt, cfg, backend=None, root=None, hf_repo=None):
        self.job, self.ckpt, self.cfg = job, ckpt, cfg
        self.hash = C.config_hash(cfg)
        self.backend = (backend or os.environ.get("STORE", "local")).lower()
        if self.backend not in ("local", "hf"):
            raise ValueError(f"STORE must be local or hf, not {self.backend!r}")
        self.root = Path(root or os.environ.get("RESULTS_DIR", "sprint_out"))
        self.hf_repo = hf_repo or os.environ.get("HF_REPO", DEFAULT_HF_REPO)
        self.rel = f"runs/{job}/{ckpt}/{self.hash}"
        self._api = None

    # ---------------------------------------------------------------- backends
    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=os.environ.get("HF_TOKEN"))
            self._api.create_repo(self.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        return self._api

    def _local_path(self, name):
        return self.root / self.rel / name

    def exists(self, name):
        if self.backend == "local":
            return self._local_path(name).exists()
        return self.api.file_exists(self.hf_repo, f"{self.rel}/{name}", repo_type="dataset")

    def _put_bytes(self, name, data):
        if self.exists(name):
            raise ShardExists(f"{self.backend}:{self.rel}/{name} exists; shards are never overwritten")
        if self.backend == "local":
            p = self._local_path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                os.link(tmp, p)          # atomic create-if-absent; fails if p appeared meanwhile
            except FileExistsError:
                raise ShardExists(f"{p} appeared while writing")
            except OSError:              # filesystems without hard links
                if p.exists():
                    raise ShardExists(f"{p} appeared while writing")
                os.replace(tmp, p)
                return
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        else:
            self.api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=f"{self.rel}/{name}",
                                 repo_id=self.hf_repo, repo_type="dataset",
                                 commit_message=f"{self.job}/{self.ckpt}/{self.hash}/{name}")

    def _get_bytes(self, name):
        if self.backend == "local":
            return self._local_path(name).read_bytes()
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(self.hf_repo, f"{self.rel}/{name}", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        return Path(p).read_bytes()

    # ---------------------------------------------------------------- public API
    def ensure_config(self):
        """Write config.json once; if it exists it must be identical."""
        blob = json.dumps(self.cfg, sort_keys=True, indent=1, default=str).encode()
        if self.exists("config.json"):
            if json.loads(self._get_bytes("config.json")) != json.loads(blob):
                raise RuntimeError(f"config.json at {self.rel} differs from this config (hash collision?)")
            return
        self._put_bytes("config.json", blob)

    def done(self, i):
        return self.exists(shard_name(i))

    def write(self, i, n, rows, model_repo=None, model_sha=None, extra_meta=None):
        """rows: list of dicts (or a pyarrow Table). Raises ShardExists if shard i exists."""
        table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(rows)
        meta = {"git_sha": C.git_sha(), "model_repo": model_repo or "", "model_sha": model_sha or "",
                "config": json.dumps(self.cfg, sort_keys=True, default=str), "config_hash": self.hash,
                "package_versions": json.dumps(C.package_versions(), sort_keys=True), "job": self.job,
                "ckpt": self.ckpt, "shard": str(i), "n_shards": str(n),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if extra_meta:
            meta.update({k: v if isinstance(v, str) else json.dumps(v, default=str) for k, v in extra_meta.items()})
        old = table.schema.metadata or {}
        table = table.replace_schema_metadata({**old, **{k.encode(): str(v).encode() for k, v in meta.items()}})
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        self.ensure_config()
        self._put_bytes(shard_name(i), buf.getvalue())
        return f"{self.backend}:{self.rel}/{shard_name(i)}"

    def read(self, i):
        t = pq.read_table(io.BytesIO(self._get_bytes(shard_name(i))))
        return t

    def list_shards(self):
        if self.backend == "local":
            d = self.root / self.rel
            return sorted(p.name for p in d.glob("shard_*.parquet")) if d.exists() else []
        files = self.api.list_repo_files(self.hf_repo, repo_type="dataset")
        return sorted(f.rsplit("/", 1)[1] for f in files if f.startswith(self.rel + "/shard_") and f.endswith(".parquet"))


def read_metadata(table):
    m = table.schema.metadata or {}
    out = {k.decode(): v.decode() for k, v in m.items() if not k.startswith(b"ARROW")}
    for k in ("config", "package_versions"):
        if k in out:
            out[k] = json.loads(out[k])
    return out


def iter_local_runs(root=None):
    """Yield (job, ckpt, config_hash, path) for every local shard or config file."""
    root = Path(root or os.environ.get("RESULTS_DIR", "sprint_out"))
    for p in sorted((root / "runs").glob("*/*/*/*")):
        if p.is_file() and not p.name.startswith(".tmp_"):
            job, ckpt, h = p.parts[-4], p.parts[-3], p.parts[-2]
            yield job, ckpt, h, p
