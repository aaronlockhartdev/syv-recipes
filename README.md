# syv-recipes — Qwen3.8-27B serving for a two-GPU box

A fork of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
specialized to one machine shape: **two 24 GB GPUs (tensor-parallel 2)**,
serving Qwen3.8-27B with vLLM 0.28.0 and its custom patch stack. Everything
not needed for serving is removed; each recipe calls `vllm serve` with
explicit flags — no CTX/SPEC-style configuration to construct.

| recipe | KV cache | speculation | reach for it when |
|---|---|---|---|
| [w4a16-int8-dflash2.sh](recipes/w4a16-int8-dflash2.sh) (default) | `int8_per_token_head`, native max len | DFlash2 drafter, 7 drafts in one pass | the all-round pick: fastest decode, double the context pool of bf16 |
| [w4a16-int8-mtp.sh](recipes/w4a16-int8-mtp.sh) | `int8_per_token_head`, native max len | Qwen's own MTP head, 3 drafts, probabilistic | no separate drafter model; the head ships inside the checkpoint |
| [w4a16-bf16-dflash2.sh](recipes/w4a16-bf16-dflash2.sh) | unquantized bf16, FlashAttention | DFlash2, 7 | the quality baseline -- no quantized-KV approximation anywhere |
| [w4a16-int4-dflash2.sh](recipes/w4a16-int4-dflash2.sh) | `int4_per_token_head` | DFlash2, 7 | context capacity: double the context of int8 in the same VRAM |
| [w4a8-int8-dflash2.sh](recipes/w4a8-int8-dflash2.sh) | `int8_per_token_head` + W4A8 linears | DFlash2, 7 | faster prefill at the documented quality cost |

The numbers in the notes below are upstream's, measured on their reference
box; we have not benchmarked these recipes -- run them on your hardware
before quoting a figure. Most predate the 0.28.0 rebase: upstream has not
re-measured its 0.28.0 matrix yet (see Notes).

All recipes take image input (no `--language-model-only`), with no
image-count limit per request; each image is capped at 2097152 px =
2048 tokens, and that cap sets the encoder's profiled peak, which
comes out of the KV pool. The ~0.9 GB vision
tower is offloaded to pinned host RAM by default
(`VLLM_VISION_CPU_OFFLOAD_GB=1`; `=0` keeps it GPU-resident). They also
use `--enable-prefix-caching` with `--prefix-caching-hash-algo xxhash`
(the `xxhash` package is in requirements.txt),
`--gpu-memory-utilization 0.93` (under TP>1 the pool is sized from
utilization), `--mamba-ssm-cache-dtype float16` (halves the GDN state
cost), `--max-num-batched-tokens 4096`, the qwen3 reasoning parser,
qwen3_coder tool parsing, and `--enable-prompt-tokens-details`, so every
response's `usage.prompt_tokens_details.cached_tokens` shows how much of
the prompt the prefix cache served. Port 8080 (`PORT=…`) and a `.venv` at
the repo root (`MODEL`, `DRAFT` overridable the same way). The dflash2-
family recipes run the default FULL_AND_PIECEWISE capture (upstream swept
dflash2 clean across every 128-residue prompt length in FULL mode);
w4a16-int8-mtp.sh forces PIECEWISE -- see its header.

## The full matrix (model-quant x KV x speculation)

The recipe name *is* the cell: `<model-quant>-<kv>-<speculation>.sh`. Both
model rows run the same W4A16-AutoRound checkpoint -- "w4a8" is the
serve-time activation quant (`VLLM_MARLIN_INPUT_DTYPE=int8`), not a
different download.

| model | KV | spec | status | decode | prefill (16k) | quality |
|---|---|---|---|---|---|---|
| w4a16 | int8 | dflash2 | recipe (default) | | | |
| w4a16 | int8 | mtp | recipe (upstream's MTP lane was fp8/FlashInfer; MTP + int8 + split-KV never measured upstream) | | | |
| w4a16 | bf16 | dflash2 | recipe (the quality baseline) | | | |
| w4a16 | int4 | dflash2 | recipe (upstream measured depth quality once, on a single 4090: 96.0% GSM8K, 100k needle retrieved; untested on TP=2) | | | |
| w4a16 | bf16 | mtp | valid, no recipe (the split-KV kernel's native path is bf16) | | | |
| w4a16 | int4 | mtp | untested (does the 3D dispatch cover MTP verify, or is the 2D walk?) | | | |
| w4a8 | int8 | dflash2 | recipe | | | |
| w4a8 | bf16 | dflash2 | valid, no recipe (upstream's single-user INT8 default lane) | | | |
| w4a8 | int4 | dflash2 | untested | | | |
| w4a8 | int8 | mtp | untested (the layer-select patch keeps `mtp.*` on W4A16) | | | |
| w4a8 | bf16 | mtp | untested | | | |
| w4a8 | int4 | mtp | untested | | | |

Fill the last three columns with your own measurements (upstream's
reference-box numbers are in the notes, deliberately not here). Cells that
never become recipes: fp8 KV wedges 3090-class cards (upstream issue #34)
-- also why our mtp recipe is int8 rather than upstream's fp8 lane; a w4a8
layer set of `mlp|linear_attn` crashes at first forward (inductor bug);
every mtp cell runs PIECEWISE capture (FULL_AND_PIECEWISE corrupts output
on prefix-cache hits).

## Layout

```
Dockerfile          uv + pinned vLLM 0.28.0 + every patch in patches/
requirements.txt    the pinned set (vllm pulls torch 2.13 / flashinfer itself)
setup.py            bare-metal one-shot: venv, deps, patches, model prep
recipes/            the five serve configurations
prepare/            build_fast_model.py, fetch_dflash2.py, patch_vllm.py -- model prep + idempotent patching (runs standalone)
patches/            the 28 vLLM patches (below)
docker/             entrypoint.sh, prepare.sh
```

## Docker

```bash
docker build -t syv-recipes .

# optional but recommended: reuse the host's existing HF hub cache -- the
# one-liner resolves to $HF_HUB_CACHE, or $HF_HOME/hub, or the default,
# however the host configures it
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
docker run -d --name qwen --gpus all --ipc=host -p 8080:8080 \
  -v qwen-models:/app/models -v qwen-cache:/cache \
  -v "$HUB":/cache/.cache/huggingface/hub \
  --restart unless-stopped syv-recipes            # entrypoint default: w4a16-int8-dflash2
```

The entrypoint prepares the models on first start (downloads through the
mounted hub cache — seconds when it is warm; `qwen-cache` holds everything
else), then execs the recipe: `syv-recipes w4a16-int8-mtp`,
`w4a16-bf16-dflash2`, `w4a16-int4-dflash2` or `w4a8-int8-dflash2` for the
others, `syv-recipes prepare` for prep only,
`PREPARE=0` to skip it. `qwen-models` receives the assembled dirs:
hard-linked off the cache when both volumes share a filesystem, a second
~21 GB copy when they don't. `VLLM_API_KEY=…` turns on key auth; without
it the server binds 0.0.0.0 and is open. Any of these can be passed from a
`.env` with `--env-file`.

The `-v "$HUB":…` mount is a run-time one (docker build cannot see host
directories, and the cache is consumed at run time by the prep anyway);
the container side is pinned in the Dockerfile. Caveats: the container
runs as root, so files it downloads into your host cache are root-owned;
on Docker Desktop the mount crosses virtiofs, so the prep's hard-links
fall back to a full copy.

## Bare metal (uv)

```bash
./setup.py             # venv + pinned deps + the 28 patches + both models
bash recipes/w4a16-int8-dflash2.sh  # or any of the other four
```

`setup.py` is idempotent — re-run it any time; it is also the recovery
path after a venv wipe. `patch_vllm.py` converges whatever state it
finds: a stamped venv whose vllm version, hash of `patches/`, and hash of the files the patches touch
all still match is a fast "Audited … in place" no-op; a fully patched but
not-yet-stamped tree (patched by an older run) is audited and stamped
without touching a file; a partially patched tree (a crash mid-run) is
completed in build order; a changed patch set, a vllm
version change, or a hand-edited tree is detected and vllm reinstalled
pristine before re-apply (the attestation covers the
hunks: a hand edit that leaves every hunk's before/after content intact
is indistinguishable from a patched tree). Paths are overridable with
env vars of the same names the recipes use (defaults: `.venv`,
`models/Qwen3.8-27B-W4A16-AutoRound-fast`,
`models/Qwen3.8-27B-DFlash2-W4A16`), so a custom layout stays consistent
between setup and serve:

```bash
MODEL=/data/qwen VENV=/data/qwen/.venv ./setup.py
MODEL=/data/qwen bash recipes/w4a16-int8-dflash2.sh
```

The same overrides can live in a `.env` file at the repo root instead:
every recipe and both `setup.py` and `patch_vllm.py` read it, but only for
a variable that is unset or empty in the real environment, which always
wins. Values may be quoted; whole-line `#` comments only. The variables the
scripts and recipes consume are `VENV`, `MODEL`, `DRAFT` and `PORT` (e.g.
`VENV=/data/qwen/.venv`); note the `VLLM_*` env vars each recipe hard-exports
are always set by the recipe itself, so a `.env` cannot change them.
In the container use the native equivalent, `docker run --env-file .env`
(`.env` is gitignored and out of the build context).

What `setup.py` runs, in order, if you'd rather do it by hand (install uv
first, and GNU `patch` if your distro lacks it:
`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare/patch_vllm.py
.venv/bin/python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
.venv/bin/python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
```

The recipes put the venv's `bin` on PATH (`VENV`, defaulting to `.venv`) and
default to port 8080; `MODEL`, `DRAFT` and `PORT` may be overridden with
env vars of the same names.

## The kept patches (all apply to vLLM 0.28.0)

DFlash2 itself is native in vLLM 0.28.0 (upstream PR #52816) — the 0.27.1
`dflash2-backport` is retired and no longer applied. The `dflash2-*`
patches below extend the native implementation instead.

- `dflash2-lookup-drafting.patch` — lookup-augmented drafting (`VLLM_DFLASH2_LOOKUP=1`, off by default): propose continuations of earlier occurrences of the current suffix; on 0.28.0 it also carries the W4A16 draft-checkpoint support (packed-qkv dequant, quantized lm_head sharing) that native DFlash2 needs for this model
- `dflash2-ngram-chains.patch` — drafter-free n-gram chains on top of lookup (`VLLM_DFLASH2_CHAIN=1`, off by default, greedy by default): while a request keeps reproducing its own context, whole verify blocks come from history and the drafter's forward and graph replay are skipped (upstream #38, ported from Dmtrii-tesla's fork)
- `dflash2-prewarm.patch` — pre-compiles the `_prepare_dflash_inputs_kernel` Triton variants at boot, so the first large prefill never JIT-compiles mid-request (#48)
- `dflash2-z-adaptive-emitted.patch` — fixes the emitted-token accounting the lookup's adaptive block-length choice reads
- `hybrid-kv-groups-v2-cudagraph.patch` — KV-group sizing for the drafter's sliding-window layers; explicit CUDA-graph memory accounting (`VLLM_V2_CUDAGRAPH_MEM_MIB`)
- `hybrid-sw-block-promote.patch` — lets a quantized KV cache fit in a hybrid target+drafter (block-size promotion instead of page padding)
- `int4-kv-per-token-head.patch` — boot blockers for int4 per-token-head KV with the drafter
- `marlin-int8-layer-select.patch` -- env-selectable int8-activation layers for the Marlin path (the w4a8-int8-dflash2 recipe)
- `marlin-int8-negative-scales.patch` — correctness fix for negative group scales in W4A8
- `marlin-repack-staged-sm80.patch` — staged Marlin repack (load-time allocation hygiene; the header records #27's corrected history — the old "VMM churn" theory was disproven)
- `marlin-tune-table.patch` — routes `marlin_gemm` through a locally built tunable Marlin extension (`VLLM_MARLIN_TUNE=1`, a no-op without that build): +3-7% on the M≤16 decode/verify GEMMs, +2-20% on W4A8 chunked-prefill GEMMs
- `mamba-align-checkpoint-order.patch` — retention of the mamba state snapshots a conversation's prefix-cache hits resume from; fixes the ~1-in-4-5-turn TTFT spikes (upstream #52 / vllm#45238). Ships default off; all five recipes default it on (`VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1`)
- `mamba-chunked-prefill-align.patch` — correctness fix: GDN/Mamba state loss and NaN during chunked prefill (the state-copy source column, and an uninitialized-memory mask in the flash-linear-attention chunk-o kernel)
- `offload-dflash-eagle-groups.patch` — OffloadingConnector group flagging under dflash
- `qwen3_5-embed-quant.patch` — route the embedding table through the quantized path (the fast model needs it)
- `qwen3_5-mtp-draft-vocab.patch` — vocab-truncated MTP draft head (the fast model's 40k draft head)
- `sampler-small-topk-fast-softmax.patch` — sort-free top-k/top-p, multi-block softmax, truncated drafts
- `spec-decode-attn.patch` — split-KV verify attention (`VLLM_SPEC_DECODE_ATTN`), bf16 path
- `spec-decode-int4-kv-mq3d.patch` — multi-query 3D dispatch for the int4-KV verify (`VLLM_INT4_MQ_3D`; the w4a16-int4-dflash2 recipe)
- `spec-decode-int8-kv.patch` — teaches the split-KV kernel to read the int8 per-token-head cache
- `spec-decode-scratch-token-units.patch` — sizes the int4-3D softmax scratch buffers in query tokens, not sequences (the #46 gate campaign: 9-16-token verify batches had been falling back to 2D silently)
- `spec-decode-scratch-within-budget.patch` — allocates those buffers inside the memory profile, so the reported KV budget is honest (#57 merge review)
- `spec-sampler-prewarm.patch` — compiles the rejection sampler's Triton kernels at boot with a spec-shaped dummy verify, so the first draft-verifying request never JITs mid-request (#48)
- `speed-knobs-envs.patch` — registers the speed knobs as env vars (torch.compile cache key)
- `triton-prefill-attn-int8.patch` — int8-QK Triton prefill attention for the head_dim-256 layers (`VLLM_PREFILL_ATTN=int8`; bf16 KV, single-request prefill chunks only, off by default; 1.27-1.35x on the kernel vs FA2)
- `vision-tower-cpu-offload.patch` — vision tower in pinned host RAM (`VLLM_VISION_CPU_OFFLOAD_GB`)
- `vllm-pr50021-gdn-spec-bounds.patch` — bounds on accepted-token state lookups in the GDN/Mamba spec kernels
- `xgrammar-spec-terminated.patch` — structured output survives tokens accepted past the grammar's end

Removed from the upstream stack: KVarN (4/2-bit KV, including its
V2-runner port) and the whole WSL2 lane. Lookup-augmented drafting and
n-gram chains are in the set (both off by default) — adopted in this sync.

## Notes

- **int8 KV is a trade** (the three int8-KV recipes): double the pool of
  bf16, at the cost of the Triton backend and a per-step unpack;
  single-user spec-decoded quality at depth is unmeasured upstream (batch
  mode: 100k-needle ok, PPL neutral — upstream docs/long-context.md).
  Verify perplexity/GSM8K on your workload before trusting it.
- **w4a16-bf16-dflash2 is the baseline**: upstream measured 96.5% GSM8K on
  this exact shape (dflash2, 4.80 tokens/step). If a quantized recipe's
  output looks off, this is the one to compare against.
- **w4a16-int4-dflash2**: depth quality has been measured once upstream,
  on a single RTX 4090 — 96.0% GSM8K (200 questions, greedy) and a
  100k-token needle retrieved at 90% depth, inside the 95.0-96.5% band the
  other configs read; it has not been measured on this TP=2 shape. The
  3D verify dispatch is vetted: upstream found two defects in the 3D path
  — the silent 2D fallback on 9-16-token batches (its #46 gate campaign)
  and scratch allocated outside the memory budget (the #57 review) — and
  the two `spec-decode-scratch-*` patches fix both. Compare outputs against
  w4a16-bf16-dflash2 before trusting it anyway — it costs ~20% decode vs
  the bf16 path, and its prefix cache only works with the
  `--prefix-match-unit 848` flag the recipe passes (without it the
  drafter's 848-token sliding-window block can never match the 1696-token
  hash unit).
- **w4a8-int8-dflash2 is a quality-for-speed trade**: the default MLP-only
  layer set costs +2.2% PPL for +13-14% prefill; `INT8_LAYERS=all` (the
  recipe expands the upstream shorthand to `mlp|linear_attn|self_attn`) is
  +27-30% at GSM8K 95.0 vs 96.5 and +4.1% PPL. Decode is unchanged either
  way (memory-bound).
- **MTP + split-KV verify**: a configuration upstream never measured (see
  the w4a16-int8-mtp header); check draft acceptance on your workload.
- **The 0.28.0 rebase** (this sync): DFlash2 went native, the old
  backport is retired, and the set above all apply to 0.28.0 (three
  patches keep upstream's older "written against 0.27.1" stamp — they
  apply cleanly to the 0.28.0 tree).
  Upstream has not re-measured its 0.28.0 matrix; the upstream numbers
  quoted in these notes are the 0.27.1 stack's. The one measured delta is
  the #73 fix below: 0.28.0 with `draft_sample_method` reads 3.23 tok/step
  / 121.7 tok/s against 0.27.1's 3.19 / 120.5 (reference 3090, CTX=fast =
  bf16 KV, k=15) — at or above the old base.
- **`draft_sample_method` is required on 0.28.0** (upstream #73): the
  native speculator base allocates the draft-logits buffer only when the
  speculative config asks for it; without it the rejection test loses its
  denominator and acceptance drops ~16%. The dflash2 recipes set it
  (`"probabilistic"`); if you assemble a speculative config by hand, do
  not omit it. The MTP recipe has always set it.
- **Two opt-in DFlash2 extensions ship with the set**, both off by default
  (the recipes do not set them): `VLLM_DFLASH2_LOOKUP=1` (lookup-augmented
  drafting) and, on top of it, `VLLM_DFLASH2_CHAIN=1` (drafter-free n-gram
  chains; greedy requests by default — at temperature the chain's
  point-mass proposals lose to the drafter's). Upstream measures +7% on
  the "reproduce a 25k-token document" cell with chains (256.9 → 276
  tok/s), flat on prose. Both were measured upstream on a single card;
  lookup is untested on TP=2 — check acceptance and output before relying
  on it (drop either into `.env` to try).
- **Multi-turn prefix caching**: `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS`
  (on by default in the recipes — the patch ships it off; set it to 0
  to opt out) keeps the mamba state snapshots a conversation's hits
  resume from alive until the request ends. The failure this fixes:
  upstream #47 saw a ~44 s TTFT spike every 4-5 turns at 44k context —
  the mamba group's hit vanishing while the attention group stayed 100%
  cached; the fix is #52 (vllm#45238), whose own repro measured
  2.1 s → 17-31 s on a shrunk pool.
  Diagnose any prefix-cache miss with
  `usage.prompt_tokens_details.cached_tokens` (upstream gotcha 43: a clean
  0 after a hit means a full recompute).
- **`VLLM_PREFILL_ATTN=int8`** (off by default): int8-QK prefill attention
  for the head_dim-256 layers — 1.27-1.35x on the kernel, +0.4% rising to
  +5.3% end-to-end prefill upstream, but only with bf16 KV, so it applies
  to w4a16-bf16-dflash2 here; on its own it is "within a few percent
  either way" (upstream #62), the gain compounds with the int8-GEMM lane.
- **`VLLM_MARLIN_TUNE=1`** (off by default; we leave it off): routes
  `marlin_gemm` through a separately built tunable Marlin extension;
  a no-op without that build. We are not building it: upstream measured
  +2-20% per GEMM on W4A8 prefill, but ~+0.4% end-to-end at the 250 W
  cap (and the build's source tree is in neither git repo), so stock
  is the right default on this hardware. Uncapped cards would
  additionally gain +3-7% on the M≤16 verify GEMMs.
- **TP=2**: upstream measured +16–35% decode at C1 vs one 3090 (PCIe x8,
  no NVLink); DFlash2 wins at every concurrency on two cards, and the
  15-draft block lost 27% at TP=2 — keep 7.
- **Chat template**: the prep replaces the stock template with
  froggeric's v22.5 (fixed thinking, tool calls, and agentic behavior
  across Qwen 3.5/3.6/3.8); per-request variables — `enable_thinking`,
  `reasoning_effort`, `tool_call_format`, … — go through
  `chat_template_kwargs`.
- First start compiles (torch.compile, CUDA graphs); the caches live in
  `$HOME` (the `/cache` volume in Docker), so it happens once.
- Sampling: Qwen recommends 0.7 / top_p 0.8 for instruct, 1.0 / 0.95 with
  thinking (the default).

## License

Apache-2.0, same as the model; the patch stack and model assets carry the
upstream repo's terms (see its LICENSE and the per-patch headers).
