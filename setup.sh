#!/bin/bash
# Bare-metal one-shot: uv venv + pinned deps + the 19 patches + both models.
#
# Zero arguments. Paths overridable with the same env vars the recipes use,
# so setup and serve can never disagree:  VENV  (default ./.venv),  MODEL,  DRAFT
#
# Idempotent and resumable -- re-running it (after a venv wipe or a
# stale-patch abort) is the recovery path.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-$DIR/.venv}
MODEL=${MODEL:-$DIR/models/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT=${DRAFT:-$DIR/models/Qwen3.8-27B-DFlash2-W4A16}
export VENV   # patch-vllm.sh reads the same name

command -v uv >/dev/null || {
  echo "uv is not installed -- one time:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  echo "== uv venv $VENV --python 3.12"
  uv venv "$VENV" --python 3.12
fi
echo "== uv pip install -r requirements.txt"
uv pip install --python "$VENV/bin/python" -r "$DIR/requirements.txt"

echo "== patch-vllm.sh"
bash "$DIR/patch-vllm.sh"

mkdir -p "$(dirname "$MODEL")" "$(dirname "$DRAFT")"
echo "== prepare/build_fast_model.py $MODEL"
"$VENV/bin/python" "$DIR/prepare/build_fast_model.py" "$MODEL"
echo "== prepare/fetch_dflash2.py $DRAFT"
"$VENV/bin/python" "$DIR/prepare/fetch_dflash2.py" "$DRAFT"

echo "ready:  bash $DIR/recipes/dflash2.sh   or   bash $DIR/recipes/mtp.sh"
