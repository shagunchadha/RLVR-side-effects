"""Sprint job dispatcher.

    python -m exp.run --job <ID> [--ckpt KEY ...] [--shard i/n] [job-specific options]
    python -m exp.run --list

Every job except B0 (Hub metadata only) first runs the version gate (exp.versioncheck) and aborts on a mismatch
with requirements_sprint.txt, unless --allow-env-mismatch is given (the mismatch is then recorded in every shard).
"""
import argparse
import importlib
import sys

JOBS = {
    # id: (module, needs_gpu, version_gated, one-line description)
    "B0": ("exp.jobs.B0_inventory", False, False, "checkpoint inventory: refs, SHAs, sizes, chat template (no GPU)"),
    "M1": ("exp.jobs.M1_weightdiff", False, True, "weight-difference spectra RLVR-DPO vs DPO-SFT (GPU optional)"),
    "M2": ("exp.jobs.M2_tracescore", True, True, "teacher-forced scoring of the frozen full-run traces"),
    "B1": ("exp.jobs.B1_likert", True, True, "Likert sampling x20 seeds + think/no-think readouts"),
    "B3": ("exp.jobs.B3_instructions", True, True, "reasoning-cap x answer-format factorial on A + 50 D items"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help="job id")
    ap.add_argument("--list", action="store_true", help="list jobs")
    ap.add_argument("--ckpt", nargs="*", default=None, help="checkpoint keys from exp.config.CHECKPOINTS")
    ap.add_argument("--shard", default=None, help="i/n: run only shard i of n (default 0/1)")
    ap.add_argument("--allow-env-mismatch", action="store_true", help="run despite a failed version gate (recorded)")
    args, rest = ap.parse_known_args(argv)
    if args.list or not args.job:
        for k, (mod, gpu, gated, desc) in JOBS.items():
            print(f"{k:3s} gpu={'yes' if gpu else 'no ':3s} gate={'yes' if gated else 'no ':3s} {desc}")
        return 0
    if args.job not in JOBS:
        sys.exit(f"unknown job {args.job!r}; use --list")
    mod_name, needs_gpu, gated, _ = JOBS[args.job]
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "add_args"):
        jp = argparse.ArgumentParser(prog=f"exp.run --job {args.job}")
        mod.add_args(jp)
        args.job_args = jp.parse_args(rest)
    elif rest:
        sys.exit(f"job {args.job} takes no extra arguments: {rest}")
    else:
        args.job_args = argparse.Namespace()
    args.env_report = None
    if gated:
        from . import versioncheck
        ok, report = versioncheck.check(need_gpu=needs_gpu)
        args.env_report = report
        if not ok and not args.allow_env_mismatch:
            sys.exit("version gate failed (see above); fix the environment or pass --allow-env-mismatch")
    return mod.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
