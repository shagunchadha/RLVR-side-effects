#!/usr/bin/env bash
# F0a (registration amendment 18): rerun test_b6_items_grading_and_syc_flow on Linux in the pinned sprint venv,
# at the commit before the fix (expected: FAIL) and at the fix commit (expected: PASS), then the whole tests/sprint
# suite at the branch head. CPU only; no model download, no HF access (the test uses STORE=local).
#   bash exp/launch/F0a_linux_check.sh            # from a clone of eacl_revision_1
# The two older commits are tested in separate git worktrees, so this checkout (and this script) never changes.
set -uo pipefail
unset PYTHONPATH PYTHONSAFEPATH
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export VENV="${VENV:-/tmp/sprint_venv}"
bash exp/setup_env.sh || { echo "F0a: env setup / version check FAILED"; exit 2; }
PY="$VENV/bin/python"
T="tests/sprint/test_jobs.py::test_b6_items_grading_and_syc_flow"
FIX="$(git log -1 --format=%H -S 'CUT = set()' -- exp/jobs/B6_controls.py)"
[ -n "$FIX" ] || { echo "F0a: fix commit not found"; exit 2; }
W="${TMPDIR:-/tmp}/f0a_worktrees"; rm -rf "$W"; git worktree prune
git worktree add -q --detach "$W/before" "$FIX^" && git worktree add -q --detach "$W/fix" "$FIX" || exit 2
run() { (cd "$1" && shift && env -u PYTHONPATH -u PYTHONSAFEPATH "$PY" -m pytest -q -p no:cacheprovider "$@"); }

echo "== F0a before fix ($(git rev-parse --short "$FIX^"))"; run "$W/before" "$T"; BEFORE=$?
echo "== F0a at fix ($(git rev-parse --short "$FIX"))";      run "$W/fix" "$T";    AT=$?
echo "== tests/sprint at head ($(git rev-parse --short HEAD))"; run "$REPO" tests/sprint; SUITE=$?
git worktree remove --force "$W/before"; git worktree remove --force "$W/fix"
echo "F0a SUMMARY: before-fix exit $BEFORE (expect 1), at-fix exit $AT (expect 0), suite exit $SUITE (expect 0)"
[ "$BEFORE" -ne 0 ] && [ "$AT" -eq 0 ] && [ "$SUITE" -eq 0 ]
