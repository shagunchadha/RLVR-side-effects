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
    "M1b": ("exp.jobs.M1b_bf16", False, True, "M1 follow-up: bf16 changed share and ulps, shuffle null, cos(dW, W) (exploratory)"),
    "M2": ("exp.jobs.M2_tracescore", True, True, "teacher-forced scoring of the frozen full-run traces"),
    "M3": ("exp.jobs.M3_settle", True, True, "settle point of the Likert answer + probes on B1 traces (amendment 3)"),
    "M4": ("exp.jobs.M4_persona", True, True, "persona vectors: trait directions per checkpoint, A-prompt projections (amendments 3, 12)"),
    "M5": ("exp.jobs.M5_causal", True, True, "causal tests: offset steering, backtracking ablation, prompt patching (amendments 3, 5, 7)"),
    "M6": ("exp.jobs.M6_cka", True, True, "linear CKA per layer DPO<->RLVR and SFT<->DPO on matched A/B2/D inputs (amendments 3, 13)"),
    "B1": ("exp.jobs.B1_likert", True, True, "Likert sampling x20 seeds + think/no-think readouts"),
    "B2": ("exp.jobs.B2_indeterminacy", True, True, "indeterminacy x self-reference 2x2 grid (amendment 3)"),
    "PH": ("exp.jobs.PH_handoff", True, True, "prefix handoff: RLVR/DPO trace prefixes continued by the other model (amendment 4)"),
    "B4": ("exp.jobs.B4_rlzero", True, True, "reward domain: base (few-shot) + RL-Zero x5 on A, B2 and 100 D items (amendments 3, 5, 10)"),
    "B5": ("exp.jobs.B5_dose", True, True, "dose-response over RL step branches: B1 protocol on A + 100 D items (amendments 3, 5, 8)"),
    "B6": ("exp.jobs.B6_controls", True, True, "hard controls: MATH-L5/AIME, sycophancy, MMLU-Pro calibration, SQuAD v2 refusal (amendments 3, 11)"),
    "B7": ("exp.jobs.B7_hedging", True, True, "self-report vs behaviour: hedging on 100 TriviaQA questions (amendment 9)"),
    "B8": ("exp.jobs.B8_ipipneo", True, True, "response style on IPIP-NEO-120 with the B1 protocol (amendments 3, 6)"),
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
