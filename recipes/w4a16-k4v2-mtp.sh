#!/bin/bash
# w4a16-k4v2-mtp: Qwen's own MTP head (3 drafts, probabilistic) on the KVarN
# 4-bit-key / 2-bit-value KV cache (kvarn_k4v2_g128) -- TP=2, the model's
# full 262,144-token context, 8 seats for the 4-8 concurrent range.
#
# KVarN (Huawei CSL, Apache-2.0) is a native vLLM attention backend ported
# onto the 0.28.0 this repo runs: Hadamard rotation + iterative variance
# normalization + 4/2-bit RTN per 128-token tile, ~840 B/token/layer
# (fp8: 2048 B) -- roughly half the int4 per-token-head cache, so the same
# VRAM holds the full 262,144-token context the model declares. Upstream
# measurements (single 3090; not ours): 262k fits, needle-in-a-haystack
# correct at 4k...240k, perplexity +0.16%, prefill within +/-5% of fp8,
# and a decode tax that runs from ~6% on short prompts to 2.13x at 112k
# context (upstream docs/long-context.md; their MTP lane -- this recipe's
# drafter).
#
# The lane is the upstream single-user launcher's CTX=huge SPEC=mtp profile:
# 3 drafts (its huge default; from C4 up, rejected drafts cost more as the
# verify batch grows, and 3 is ahead of 4) on the KVarN cache. MTP runs on
# the V1 runner, so the dflash2-only V2 graph accounting
# (VLLM_V2_CUDAGRAPH_MEM_MIB) does not apply here.
#
# Three flags are not optional here:
#   --block-size 128: the KVarN tile IS the block (its variance
#     normalization runs per 128-token tile). The hybrid page unification
#     then pads the attention page to the 2048-token Gated-DeltaNet page;
#     the backend reads it in 128-token kernel tiles.
#   --prefix-match-unit 128: the prefix hash unit must equal the tile, so
#     cache hits land on tile boundaries -- a non-multiple of 128 corrupts
#     the pool (upstream single-user launcher). The k4v2 analogue of the
#     int4 recipe's 848.
#   cudagraph_mode=PIECEWISE: the default (FULL_AND_PIECEWISE) has a
#     documented MTP corruption: with a prefix-cache hit, one prompt
#     length in 128 returns "" / "#" or fluent wrong text. Upstream forced
#     PIECEWISE for MTP for correctness; at the served lengths it costs
#     nothing measured.
#
# KVarN + MTP + prefix caching (all three) corrupts prompt_logprobs:
# perplexity ~23% high, NaN 400s on some documents (upstream gotcha 51 /
# #64). Ordinary generation is not implicated -- needle retrieval and
# decode rates are normal on the same server, and every other combination
# (the dflash2 drafter, no speculation, any non-kvarn KV) is clean to four
# decimals. Prefix caching stays on: the shared-prefix win is what this
# recipe is built for. To measure quality, run the battery with prefix
# caching off for the run (EXTRA_ARGS).
#
# Deviations from the upstream single-card launcher (CTX=huge SPEC=mtp):
#   1. TP=2 with gpu-memory-utilization sizing: all the recipes here run
#      2x3090; the launcher's single-card 4.90 GiB KV_MEM pin is dropped
#      at TP>1 (its own rule), as in the other recipes.
#   2. --max-model-len 262144, not the launcher's 200000: the pin above
#      caps the single card (one 262k request needs ~5.05 GiB of its 4.90
#      GiB pool, upstream PR #13 -- the same wall the sibling k4v2
#      recipe's 245760 choice describes), so the launcher settles for
#      200k. On TP=2 the KV of a max-len request shards across both cards
#      and the 0.93-utilization pool on two 24 GB cards is well above it,
#      so the full declared length fits; this box has not measured that
#      boot -- if the engine refuses the length at start-up, lower it via
#      EXTRA_ARGS.
#   3. --max-num-seqs 8, not the launcher's 2: the single-card huge
#      profile is single-user; this recipe targets 4-8 concurrent. The 8
#      is the launcher's MTP value, and MTP's 1+3 state slots per
#      resident are half the dflash2's, so the seats fit where the
#      sibling k4v2 recipe's 2 are needed.
#   4. KVARN_POOL_MEM_FRAC 0.25, not the single-user 0.15: the fp16 tail
#      pool (the first 128 tokens per request, never quantized) must
#      cover the seats; the upstream batch lane uses 0.25 to keep all of
#      its 64 slots, and 8 fits inside that with room to spare.
#   5. --max-num-batched-tokens 2048, not the 8192 the int8-mtp recipe
#      deviates to: under KVarN the activation peak a bigger chunk
#      profiled shrinks exactly the pool this recipe sells, and every
#      quoted number for the kvarn lane was measured at 2048.
#
# The env vars support the patch stack; the vllm line is the complete
# server configuration.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"

# .env at the repo root: fills any variable below that is unset or empty in
# the real environment (which always wins); values may be quoted, whole-line
# # comments only.
if [ -f "$REPO/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in ''|\#*) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    [ "$k" != "$line" ] || continue
    k="${k%"${k##*[![:space:]]}"}"
    case "$k" in ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;; esac
    # never let a .env flip a shell-control variable (GLOBIGNORE would
    # silently disable the /dev/shm cleanup glob below, among others)
    case "$k" in IFS|GLOBIGNORE|CDPATH|BASH_ENV|ENV|SHELLOPTS|PS1|LINENO|PWD|OLDPWD|SECONDS|RANDOM|UID|EUID) continue ;; esac
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    v="${v%$'\r'}"
    v="${v#\"}"; v="${v%\"}"
    if [ "${#v}" -ge 2 ] && [ "${v:0:1}" = "${v: -1}" ]; then
      [ "${v:0:1}" = "'" ] && v="${v:1:${#v}-2}"
    fi
    [ -n "$v" ] || continue
    [ -n "${!k:-}" ] || export "$k=$v"
  done < "$REPO/.env"
fi

VENV=${VENV:-$REPO/.venv}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
PORT=${PORT:-8080}
export PATH="$VENV/bin:$PATH"

[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL -- run: python prepare/build_fast_model.py <dir> (or: docker run ... prepare)" >&2; exit 1; }
if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null; then
  echo "no vllm found -- create the uv venv first (README: Bare metal), or run this in the container" >&2; exit 1
fi

# a dead engine leaves its /dev/shm offload region and the next boot dies on it (upstream #33)
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[w4a16-k4v2-mtp] removing stale offload region $f"; rm -f "$f"; }
  done
fi

# the KVarN backend dequantizes the packed tiles inside its own verify
# path; the split-KV verify kernel (patches/spec-decode-*.patch) reads
# bf16/int8 caches only, so it stays off here
export VLLM_SPEC_DECODE_ATTN=0
# the fp16 tail pool (first 128 tokens per request, never quantized) is
# budgeted as a share of the post-weight VRAM envelope: 0.25 keeps all 8
# seats (upstream batch lane; their single-user huge default 0.15 caps seats)
export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.25}
# vision tower in pinned host RAM by default (upstream default; patches/vision-tower-cpu-offload.patch):
# off the VRAM budget, bit-exact, ~+12% per image forward; =0 keeps it GPU-resident
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
# torch sampler (flashinfer's needs nvcc to JIT)
export VLLM_USE_FLASHINFER_SAMPLER=0
# hold the mamba "align" state snapshots a multi-turn prefix hit resumes
# from alive until the request ends (upstream #52 / vllm#45238;
# patches/mamba-align-checkpoint-order.patch). That patch ships default
# off -- we default it on (deviation); retention is bounded (<=3 blocks per
# request per group). Set VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=0 to opt out.
export VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=${VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS:-1}
# keep DeltaNet's transient workspace from fragmenting the allocator (boot OOM)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# no --language-model-only: the server takes image input.
# Vision: image count is unlimited; each image is capped at 2097152 px
# = 2048 tokens, and that cap sets the encoder's profiled peak in the
# KV pool (at most the 4096-token encoder budget). xxhash: faster
# prefix-cache hashes than sha256. No --attention-backend: the kvarn_k4v2
# cache dtype selects the KVarN backend itself (the kvarn-0.28.0 patch
# registers it in the CUDA priority list, like TurboQuant).
# draft_sample_method is required on 0.28.0 (upstream #73): without it
# the rejection test loses its denominator and acceptance drops ~16%.
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --api-server-count 1 \
  --kv-cache-dtype kvarn_k4v2_g128 \
  --block-size 128 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --prefix-match-unit 128 \
  --mamba-cache-mode align \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"cudagraph_mode":"PIECEWISE","custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${EXTRA_ARGS:-}
