#!/bin/bash
# One-time model preparation into /app/models (idempotent, CPU only;
# ~21 GB of downloads on the first run, then HF-cache hits):
#   1. the modified Qwen3.8-27B "fast" model
#   2. the W4A16 DFlash2 drafter
set -e
cd /app
export PATH=/app/venv/bin:$PATH
python prepare/build_fast_model.py /app/models/Qwen3.8-27B-W4A16-AutoRound-fast
python prepare/fetch_dflash2.py /app/models/Qwen3.8-27B-DFlash2-W4A16
echo "prepare: models ready under /app/models"
