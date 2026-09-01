#!/bin/bash
# First argument selects what to run:
#   w4a16-int8-dflash2  recipes/w4a16-int8-dflash2.sh  (default) -- int8 KV, split-KV verify
#   w4a16-int8-mtp      recipes/w4a16-int8-mtp.sh      -- int8 KV, Qwen MTP head, PIECEWISE graphs
#   w4a16-bf16-dflash2  recipes/w4a16-bf16-dflash2.sh  -- unquantized KV, FlashAttention
#   w4a16-int4-dflash2  recipes/w4a16-int4-dflash2.sh  -- int4 KV, ~2x the context capacity
#   w4a8-int8-dflash2   recipes/w4a8-int8-dflash2.sh   -- W4A8 linears, faster prefill
#   prepare             docker/prepare.sh              (download + build the models into /app/models)
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent) unless PREPARE=0.
set -e
cd /app
export PATH=/app/.venv/bin:$PATH
cmd=${1:-w4a16-int8-dflash2}; shift || true
case "$cmd" in
  w4a16-int8-dflash2|w4a16-int8-mtp|w4a16-bf16-dflash2|w4a16-int4-dflash2|w4a8-int8-dflash2)
    if [ "${PREPARE:-1}" != "0" ]; then bash docker/prepare.sh; fi
    exec bash "recipes/$cmd.sh" "$@" ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
