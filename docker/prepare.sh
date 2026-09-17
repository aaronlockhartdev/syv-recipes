#!/bin/bash
# One-time model preparation into /app/models (idempotent, CPU only;
# ~25 GB of downloads on the first run, then HF-cache hits):
#   1. the modified Qwen3.8-27B "fast" model
#   2. the W4A16 DFlash2 drafter
#   3. the bf16 DSpark drafter (DSPARK=/path redirects; =0/false/no skips)
#   4. the Swift W4A16 variant, opt-in (SWIFT=1; SWIFT=/path redirects)
set -e
cd /app
export PATH=/app/.venv/bin:$PATH
python prepare/build_fast_model.py /app/models/Qwen3.8-27B-W4A16-AutoRound-fast
python prepare/fetch_dflash2.py /app/models/Qwen3.8-27B-DFlash2-W4A16
# DSPARK=/path redirects the dir, DSPARK=0/false/no skips the fetch
DSPARK_DIR=/app/models/Qwen3.8-27B-DSpark
case "${DSPARK:-}" in
  0|false|no) DSPARK_DIR= ;;
  ""|1|true|yes) ;;
  *) DSPARK_DIR=$DSPARK ;;
esac
if [ -n "$DSPARK_DIR" ]; then
  python prepare/fetch_dspark.py "$DSPARK_DIR"
fi
# SWIFT=1 builds the Swift W4A16 variant (opt-in), SWIFT=/path redirects
SWIFT="${SWIFT//[[:space:]]/}"
SWIFT_DIR=
case "${SWIFT:-}" in
  0|false|no|"") ;;
  1|true|yes) SWIFT_DIR=/app/models/Qwen3.8-27B-Swift-W4A16 ;;
  *) SWIFT_DIR=$SWIFT ;;
esac
if [ -n "$SWIFT_DIR" ]; then
  python prepare/build_swift_model.py "$SWIFT_DIR"
fi
# Some clients (JetBrains AI Assistant) send tool-call arguments as a JSON
# array instead of an object; harden the templates so `|items` does not blow
# up ("Can only get item pairs from a mapping.") once for every prepared
# model. A template that does not match the known pattern warns and is left
# alone; only an unreadable one fails prepare. HARDEN_TEMPLATES=0 skips it.
if [ "${HARDEN_TEMPLATES:-1}" != "0" ]; then
  python prepare/harden_chat_template.py
fi
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  printf '  \033[32m✓\033[0m models ready under /app/models\n'
else
  echo "  ✓ models ready under /app/models"
fi
