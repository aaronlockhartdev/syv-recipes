#!/bin/bash
# Apply every patch in patches/ to the vLLM in the repo's venv
# (override the default ./.venv with VENV=/path/to/venv).
#
# Same semantics as the Dockerfile: set -e + patch --batch (never prompts
# on a dead stdin) + a compileall gate -- a failing or already-applied
# patch aborts instead of leaving a half-patched vLLM.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-$DIR/.venv}
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "no python at $PY" >&2
  echo "create the venv first (README: Bare metal) or point VENV at it" >&2
  exit 1
fi

if ! SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null); then
  echo "vllm is not installed in $VENV" >&2
  echo "install it first:  uv pip install --python $PY -r $DIR/requirements.txt" >&2
  exit 1
fi
echo "patching vLLM at $SP"

n=0
for p in "$DIR"/patches/*.patch; do
  echo "== $(basename "$p")"
  if ! patch -p1 -d "$SP" --batch < "$p"; then
    echo >&2
    echo "patch failed: $(basename "$p") (rejects written next to the files in $SP)" >&2
    echo "if it says 'Reversed (or previously applied) patch detected', this venv is" >&2
    echo "already patched -- the clean reset is:  rm -rf $VENV && uv venv $VENV --python 3.12" >&2
    exit 1
  fi
  n=$((n + 1))
done

if ! "$PY" -m compileall -q "$SP"; then
  echo "patches applied but the tree does not compile -- vLLM is broken, do not run it" >&2
  exit 1
fi
echo "done: $n patches applied and the patched tree compiles"
