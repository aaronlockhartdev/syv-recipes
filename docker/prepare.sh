#!/bin/bash
# One-time model preparation into /app/models (idempotent, CPU only;
# ~21 GB of downloads on the first run, then HF-cache hits):
#   1. the modified Qwen3.8-27B "fast" model
#   2. the W4A16 DFlash2 drafter
#   3. (optional, DSPARK=1 or DSPARK=/dir; =0 is a no-op) the bf16 DSpark drafter
set -e
cd /app
export PATH=/app/.venv/bin:$PATH
python prepare/build_fast_model.py /app/models/Qwen3.8-27B-W4A16-AutoRound-fast
python prepare/fetch_dflash2.py /app/models/Qwen3.8-27B-DFlash2-W4A16
case "${DSPARK:-}" in 0|false|no) DSPARK= ;; esac
if [ -n "${DSPARK:-}" ]; then
  DSPARK_DIR=/app/models/Qwen3.8-27B-DSpark
  case "$DSPARK" in 1|true|yes) ;; *) DSPARK_DIR=$DSPARK ;; esac
  python prepare/fetch_dspark.py "$DSPARK_DIR"
fi
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  printf '  \033[32m✓\033[0m models ready under /app/models\n'
else
  echo "  ✓ models ready under /app/models"
fi
