# syv-recipes — Qwen3.8-27B serving for a two-GPU box

A fork of [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
specialized to one machine shape: **two 24 GB GPUs (tensor-parallel 2)**,
serving Qwen3.8-27B with vLLM 0.27.1 and its custom patch stack.
Everything not needed for the two recipes below is removed; the recipes are
plain bash files that call `vllm serve` with **explicit flags** — there are
no CTX/SPEC-style env-var configurations left to construct.

| recipe | speculation | KV cache | the upstream config it matches |
|---|---|---|---|
| [recipes/dflash2.sh](recipes/dflash2.sh) | DFlash2 block drafter, 7 drafts in one pass | `int8_per_token_head`, prefix caching, native max len, 4 seqs | `SPEC=dflash2 CTX=long PREFIX_CACHE=1`, `--tensor-parallel-size 2` |
| [recipes/mtp.sh](recipes/mtp.sh) | Qwen's MTP head, 3 drafts, probabilistic | `int8_per_token_head`, prefix caching, native max len, 8 seqs | `SPEC=mtp CTX=long PREFIX_CACHE=1`, `--tensor-parallel-size 2`, KV swapped fp8 → int8 |

Both recipes take image input (no `--language-model-only`) with the
original repo's vision arguments: at most one image per request
(`--limit-mm-per-prompt`) and a pixel cap below the processor default
(`--mm-processor-kwargs`) so the encoder's profiling peak
(2097152 px = 2048 image tokens) stays small in the KV pool. The ~0.9 GB
vision tower is offloaded to pinned host RAM by default
(`VLLM_VISION_CPU_OFFLOAD_GB=1`) and is copied to the GPUs for each image
forward — zero resident VRAM, bit-exact output, ~+12% on vision forwards
(measured on a PCIe 4.0 x16 3090); `=0` keeps it GPU-resident (it fits,
and saves the copy). They use
`--enable-prefix-caching` with `--prefix-caching-hash-algo xxhash` (128-bit
xxHash instead of the default sha256 for block hashing — faster, and the
non-cryptographic collision caveat is moot on a single-user box; the
`xxhash` package is in requirements.txt),
`--gpu-memory-utilization 0.93` (the launcher's single-card KV pin does not
apply under TP>1 — the pool is sized from utilization),
`--mamba-ssm-cache-dtype float16` (halves the GDN recurrent-state cost),
`--max-num-batched-tokens 2048`, the qwen3 reasoning parser, and `qwen3_coder`
tool parsing. They default to port 8080 (`PORT=…`) and a `.venv` at the repo
root (`MODEL`, `DRAFT` overridable the same way).

## Layout

```
Dockerfile          uv + pinned vLLM 0.27.1 + every patch in patches/
requirements.txt    the pinned set (vllm pulls torch 2.13 / flashinfer itself)
patch-vllm.sh       applies patches/ to the installed vllm (bare metal)
recipes/            dflash2.sh, mtp.sh — the two ways to serve
prepare/            build_fast_model.py, fetch_dflash2.py — one-time model prep
patches/            the 17 kept vLLM patches (below)
docker/             entrypoint.sh, prepare.sh
```

## Docker

```bash
docker build -t syv-recipes .
docker run -d --name qwen --gpus all --ipc=host -p 8080:8080 \
  -v qwen-models:/app/models -v qwen-cache:/cache \
  --restart unless-stopped syv-recipes            # entrypoint default: dflash2
```

The entrypoint prepares the models on first start (downloading through the
`qwen-cache` HF-cache volume, then seconds on a warm cache) and execs the
recipe. `qwen-models` receives the assembled model dirs: hard-linked off the
cache when both volumes share a filesystem (no extra space), a second
~21 GB copy when they don't. `syv-recipes mtp` runs the other one;
`syv-recipes prepare` runs only the prep; `PREPARE=0` skips the prep when the
models volume already holds them. Set `VLLM_API_KEY=…` as an env var to turn
on key auth — optional; without it the server binds 0.0.0.0 and is open.

## Bare metal (uv)

```bash
# once, if uv is not installed yet (and install patch if your distro lacks it)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# the 17 patches, against the installed vllm (fails loud, never prompts)
bash patch-vllm.sh

.venv/bin/python prepare/build_fast_model.py models/Qwen3.8-27B-W4A16-AutoRound-fast
.venv/bin/python prepare/fetch_dflash2.py        models/Qwen3.8-27B-DFlash2-W4A16
bash recipes/dflash2.sh        # or recipes/mtp.sh
```

The recipes put `./.venv/bin` on PATH and default to port 8080; `MODEL`,
`DRAFT` and `PORT` may be overridden with env vars of the same names.
`VLLM_API_KEY=…` in the environment turns on key auth (optional).

## The kept patches (all written against vLLM 0.27.1)

- `dflash2-backport.patch` — DFlash2 drafter (vLLM PR #52816) on 0.27.1, incl. the V2 model-runner speculator
- `hybrid-kv-groups-v2-cudagraph.patch` — hybrid KV-group sizing for the drafter's sliding-window layers; explicit CUDA-graph memory accounting for the V2 runner (`VLLM_V2_CUDAGRAPH_MEM_MIB`)
- `hybrid-sw-block-promote.patch` — lets a quantized KV cache fit in a hybrid target+drafter (block-size promotion instead of page padding)
- `int4-kv-per-token-head.patch` — boot blockers for int4 per-token-head KV with the drafter
- `marlin-int8-layer-select.patch` — env-selectable int8-activation layers for the Marlin path
- `marlin-int8-negative-scales.patch` — correctness fix for negative group scales in W4A8
- `marlin-repack-staged-sm80.patch` — staged Marlin repack (sm80 VMM-wedge candidate)
- `offload-dflash-eagle-groups.patch` — OffloadingConnector group flagging under dflash
- `qwen3_5-embed-quant.patch` — route the embedding table through the quantized path (the fast model needs it)
- `qwen3_5-mtp-draft-vocab.patch` — vocab-truncated MTP draft head (the fast model's 40k draft head)
- `sampler-small-topk-fast-softmax.patch` — sort-free top-k/top-p, multi-block softmax, truncated drafts
- `spec-decode-attn.patch` — split-KV verify attention (`VLLM_SPEC_DECODE_ATTN`)
- `spec-decode-int8-kv.patch` — the split-KV kernel reads the int8 per-token-head cache (what both recipes run on)
- `speed-knobs-envs.patch` — registers the speed knobs as env vars (torch.compile cache key)
- `vision-tower-cpu-offload.patch` — vision tower in pinned host RAM (on by default in both recipes, `VLLM_VISION_CPU_OFFLOAD_GB=1`; `=0` keeps it GPU-resident)
- `vllm-pr50021-gdn-spec-bounds.patch` — bounds on accepted-token state lookups in the GDN/Mamba spec kernels
- `xgrammar-spec-terminated.patch` — structured output survives tokens accepted past the grammar's end

Removed from the upstream stack: KVarN (4/2-bit KV cache), lookup-augmented
drafting (the `DFLASH_TOKENS>7` lane), and n-gram chains — the DFlash2
checkpoint proposes its trained 7 tokens, and the verify block stays 8.

## Notes

- **int8 KV is a trade**: roughly double the pool of bf16, at the cost of the
  Triton backend and a per-step int8 unpack, and its quality at depth was
  never measured in the upstream repo (their measured int8 numbers cover the
  batch-mode activations path, not this cache). Verify perplexity/GSM8K on
  your workload before trusting it with real data.
- **MTP + split-KV verify**: the mtp recipe runs a configuration upstream
  never measured (int8-KV split-KV verify under MTP, see the recipe header);
  check draft acceptance on your workload before trusting it.
- **TP=2**: upstream measured +16–35% decode at C1 on a 1-vs-2×3090 A/B
  (PCIe x8, no NVLink) and found DFlash2 wins at every concurrency on two
  cards; keep the 7-draft block (the one 15-draft datapoint at TP=2 lost 27%).
- **Chat template**: the prep step replaces the base checkpoint's stock
  template with the Qwen-Sharp v22.4.0 template
  (`peculiar-ragdoll/Qwen-Sharp-Chat-Templates`): token-efficient thinking
  and tool calls. Per-request template variables — `enable_thinking`,
  `reasoning_effort`, `tool_call_format`, … — go through
  `chat_template_kwargs`.
- First start compiles (torch.compile, CUDA graph capture) — the caches live
  in `$HOME` (the `/cache` volume in Docker), so it happens once.
- Sampling: Qwen recommends temperature 0.7 / top_p 0.8 for instruct and
  1.0 / 0.95 with thinking (the default).

## License

Apache-2.0, same as the model; the patch stack and model assets carry the
upstream repo's terms (see its LICENSE and the per-patch headers).
