#!/usr/bin/env bash
# Build the pinned sprint venv (Linux x86_64, Python 3.13) and run the version check.
#   bash exp/setup_env.sh            # venv at $VENV (default /tmp/sprint_venv)
# PYTHONPATH is cleared for the install and the check, so the host notebook's site-packages cannot leak in.
# requirements_sprint.txt is fully pinned (uv pip compile, see its header); this script installs exactly that file.
set -euo pipefail
unset PYTHONPATH
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/tmp/sprint_venv}"
PY="${PYTHON:-python3.13}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it into the user site" >&2
  "$PY" -m pip install --user uv
fi
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python "$PY"
fi
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements_sprint.txt"
# the repo itself is used from the checkout (python -m exp.run from the repo root); nothing else is installed
cd "$REPO"
env -u PYTHONPATH "$VENV/bin/python" -m exp.versioncheck "$@"
echo "OK: activate with   source $VENV/bin/activate   (and keep PYTHONPATH unset)"
