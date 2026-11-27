"""Upload existing local shards to the HF dataset repo, skipping files already there (never overwrites).

python -m exp.sync --from local --to hf [--root ./sprint_out] [--dry-run]
"""
import argparse
import os
import sys

from . import store as S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="local", choices=["local"])
    ap.add_argument("--to", dest="dst", default="hf", choices=["hf"])
    ap.add_argument("--root", default=os.environ.get("RESULTS_DIR", "sprint_out"))
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO", S.DEFAULT_HF_REPO))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.dry_run and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is required to upload")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not a.dry_run:
        api.create_repo(a.hf_repo, repo_type="dataset", private=True, exist_ok=True)
    remote = set(api.list_repo_files(a.hf_repo, repo_type="dataset")) if not a.dry_run else set()
    n_up = n_skip = 0
    for job, ck, h, p in S.iter_local_runs(a.root):
        rel = f"runs/{job}/{ck}/{h}/{p.name}"
        if rel in remote:
            n_skip += 1
            continue
        print(("would upload " if a.dry_run else "upload ") + rel)
        if not a.dry_run:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=a.hf_repo, repo_type="dataset",
                            commit_message=f"sync {rel}")
        n_up += 1
    print(f"uploaded {n_up}, already present {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
