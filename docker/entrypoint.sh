#!/bin/bash
# First argument selects what to run:
#   dflash2  recipes/dflash2.sh   (default)
#   mtp      recipes/mtp.sh
#   prepare  docker/prepare.sh    (download + build the models into /app/models)
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent) unless PREPARE=0.
set -e
cd /app
export PATH=/app/.venv/bin:$PATH
cmd=${1:-dflash2}; shift || true
case "$cmd" in
  dflash2|mtp)
    if [ "${PREPARE:-1}" != "0" ]; then bash docker/prepare.sh; fi
    exec bash "recipes/$cmd.sh" "$@" ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
