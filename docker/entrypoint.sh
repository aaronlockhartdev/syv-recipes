#!/bin/bash
# First argument selects what to run:
#   dflash2    recipes/dflash2.sh   (default) -- int8 KV, split-KV verify
#   mtp        recipes/mtp.sh       -- int8 KV, Qwen MTP head, PIECEWISE graphs
#   bf16       recipes/bf16.sh      -- unquantized KV, FlashAttention
#   int4       recipes/int4.sh      -- int4 KV, ~2x the context capacity
#   int8_act   recipes/int8_act.sh  -- W4A8 linears, faster prefill
#   prepare  docker/prepare.sh    (download + build the models into /app/models)
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent) unless PREPARE=0.
set -e
cd /app
export PATH=/app/.venv/bin:$PATH
cmd=${1:-dflash2}; shift || true
case "$cmd" in
  dflash2|mtp|bf16|int4|int8_act)
    if [ "${PREPARE:-1}" != "0" ]; then bash docker/prepare.sh; fi
    exec bash "recipes/$cmd.sh" "$@" ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
