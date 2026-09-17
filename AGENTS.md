# AGENTS.md

## What this repo is

`syv-recipes` is a restructuring and UX improvement of the work done in
`syv-ai/qwen38-27b-rtx3090` (fetched into this repo's `main` branch): the
same vLLM patch stack and models, re-expressed for a 2xRTX-3090 (TP=2,
2x24 GB) box as explicit single-command recipes with uv-based setup and a
one-layer Docker image.

**Direction of authority.** Upstream is the technical source of truth —
the patch set, its measured numbers, correctness gotchas, and commit
history all live there and are *synced into this repo* (into `patches/`,
into comment claims, into this file). This repo owns the deployment layer:
which recipes exist, their default flag values, naming, scripts, and
packaging. A change here that conflicts with an upstream technical fact
needs upstream evidence for it, or an explicit "deviation from upstream"
disclosure in the recipe header (all deviations are disclosed there).

## The nine recipes

| recipe | stack | one-liner |
|---|---|---|
| `w4a16-int8-dflash2` | TRITON_ATTN, int8 KV, split-KV verify, dflash2 drafter | the fast default |
| `w4a16-int8-mtp` | TRITON_ATTN, int8 KV, MTP head, PIECEWISE graphs | no separate drafter |
| `w4a16-bf16-dflash2` | FLASH_ATTN, bf16 KV, dflash2 | the unquantized quality baseline |
| `w4a16-int4-dflash2` | TRITON_ATTN, int4 KV, dflash2, `--prefix-match-unit 848` | ~2x the context capacity |
| `w4a16-int4-mtp` | TRITON_ATTN, int4 KV, MTP head, 3 drafts, PIECEWISE graphs, `--prefix-match-unit 808`, `VLLM_INT4_MQ_3D=1` | the int4 context capacity with no separate drafter; the int4 x MTP cell, unmeasured upstream |
| `w4a16-k4v2-dflash2` | KVarN backend (kvarn_k4v2_g128), dflash2, `--block-size 128`, `--prefix-match-unit 128` | the full 262k context, ~2x the int4 pool |
| `w4a16-k4v2-mtp` | KVarN backend (kvarn_k4v2_g128), MTP head, 3 drafts, `--block-size 128`, `--prefix-match-unit 128` | the full 262k context with no separate drafter; 8 seats for 4-8 concurrent |
| `w4a8-int8-dflash2` | dflash2 + W4A8 Marlin linears (INT8_LAYERS) | faster prefill, documented quality cost |
| `w4a16-int8-dspark` | TRITON_ATTN, int8 KV, DSpark bf16 community drafter (RadixArk) | upstream-measured slower than the dflash2 head on their shape; unmeasured on ours |

Naming scheme: `[model_quant]-[kv]-[spec_decode_method]`. Do not break it.
Adding a recipe = a new file following the scheme, an arm in
`docker/entrypoint.sh`, a row in the README table (and a matrix row if the
combination is new to the matrix).

## Invariants

- **`patches/` is the synced set (35).** They apply in the order of
  `patches/series` (the upstream file; later patches depend on files and
  hunk context created by earlier ones -- e.g. prefill-attn-int8 carries
  spec-decode-attn's lines as context), not in glob order.
  They are the upstream files verbatim (`upstream/synced` names the
  upstream commit they come from); their headers carry the provenance
  (PR/issue refs and, where upstream stamps one, an "upstream <sha>"
  line). Never hand-edit vLLM; `prepare/patch_vllm.py` is the only thing
  that touches the installed tree, and it attests the result with a
  3-field stamp (version + patch-set fingerprint + tree digest).
- **MTP must keep `cudagraph_mode: PIECEWISE`** — the default (FULL)
  corrupts one prompt length in 128 (residue `k+1`) under prefix-cache hits.
- **int4 KV must keep a `--prefix-match-unit` equal to the drafter's
  sliding-window block size** (upstream docs/wsl2-4090.md: the guard
  returns a permanent clean miss unless the hash unit equals that block
  size, and the hybrid coordinator's min turns one group's miss into zero
  reuse model-wide). The block size moves with the draft count: 848 at
  n=7 (w4a16-int4-dflash2, hash unit 1696), 808 at n=3 (w4a16-int4-mtp,
  hash unit 1616). int8 never needs the flag (its geometry lands 864/864).
- **k4v2 (KVarN) must keep `--block-size 128` and `--prefix-match-unit 128`** — the variance-normalization tile is the block, and the prefix hash unit must equal it, so cache hits land on tile boundaries; a non-multiple of 128 corrupts the pool (upstream single-user launcher). The k4v2 analogue of the int4 848 rule.
- **fp8 KV is excluded** — deterministic Xid-31 on 3090-class (upstream
  issue #34). That is also why our MTP recipe is int8 KV rather than
  upstream's fp8/FlashInfer lane.
- **No WSL2 support** anywhere in this repo (upstream had a whole WSL2
  lane; it is out of scope here), and no env vars for it. KVarN was dropped
  with that lane at the fork and is now adopted (w4a16-k4v2-dflash2, user
  decision): its modules carry `KVARN_*` env vars, all with built-in
  defaults; the recipe sets only `KVARN_POOL_MEM_FRAC`.
- **The lookup/chain envs** (`VLLM_DFLASH2_LOOKUP`, `VLLM_DFLASH2_CHAIN`)
  exist in the patch set — both adopted in the
  0.28.0 sync, both off by default, enabled by `.env` only.
- **DFlash2 is native in vLLM 0.28.0** (upstream PR #52816); the 0.27.1
  `dflash2-backport` is retired. The `dflash2-*` patches extend the native
  implementation, and `dflash2-lookup-drafting` additionally carries the
  W4A16 draft-checkpoint support (packed-qkv dequant, quantized lm_head
  sharing) that serving this model needs — do not trim it on the
  assumption it is just an option.
- **A speculative-config for a drafter must set `draft_sample_method`** (upstream
  #73): on 0.28.0 the native speculator base allocates the draft-logits
  buffer only when the config asks; without it the rejection test loses
  its denominator and acceptance drops ~16% (101.4 vs 121.7 tok/s
  upstream). All nine drafter recipes (the five dflash2, the dspark, and
  the three MTP) set it to probabilistic.
- **Vision is on** (no `--language-model-only`) — two 24 GB cards are not
  VRAM-limited; the tower offloads to pinned host RAM by default
  (`VLLM_VISION_CPU_OFFLOAD_GB=1`) since dflash2 + vision OOMs at graph
  capture on 24 GB without it.
- **DSpark is a community checkpoint, not ours**: RadixArk/Qwen3.8-27B-DSpark
  (bf16, 7 drafts/step) is fetched in every setup (`setup.py`,
  `docker/prepare.sh`; `DSPARK=/path` redirects, `=0/false/no` skips). The
  fetch rewrites the checkpoint's config.json architecture to
  Qwen3DSparkModel (0.28.0 maps the published name to the DeepSeek V4
  class), hard-links the weights, and copies the checkpoint's own
  dspark.py/dflash.py so the installed dir stays self-contained; an
  interrupted (partial) cache is completed, never treated as warm.

- **The Swift variant is opt-in** (`SWIFT=1` in setup/prepare; `SWIFT=/path`
  redirects): `prepare/build_swift_model.py` applies the fast model's
  round-to-nearest operations (int8 group-128 lm_head/embed_tokens/MTP
  module, froggeric template) to jamesbrunet's W4A16-AutoRound quant of
  the ukisai reasoning-efficiency fine-tune (~58% fewer thinking tokens
  at <1% accuracy on their benches). The GPTQ int4 upgrades and the 40k
  MTP draft head are deferred (they need the upstream drafter/ pipeline
  run against the fine-tune): until the re-fit lands, MTP runs the native
  full-vocab head and the dflash2/dspark drafters (base-trained) are
  unmeasured on it. The Swift Open License is restrictive (free up to
  $1M ARR) -- the dir is fetched and built, never committed.
- **Mamba align-snapshot retention is on by default**
  (`VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1` in every recipe — the patch
  ships off; deviation). All recipes run `--mamba-cache-mode align`,
  where retaining the CoW'd snapshots fixes the ~1-in-4-5-turn TTFT
  collapse (upstream #52 / vllm#45238); retention is bounded (<=3
  blocks per request per group) and the patch's A/B measured no
  change at realistic pool sizes. Set the var to 0 to opt out.
- **Empty means unset, everywhere**: bash `${VAR:-default}`, Python
  `os.environ.get(k) or default`, and the `.env` loader all treat an
  empty string as absent. `.env` (repo root, gitignored) only fills
  variables that are unset-or-empty in the real environment, which always
  wins. `EXTRA_ARGS` is a plain `.env` variable like the rest.
- **Recipe scripts are data, not logic**: the whole server configuration is
  inlined in the one `exec vllm serve` line; env vars are only what the
  patch stack needs (and their defaults). No config branching. No comment
  lines inside backslash continuation chains. The one sanctioned escape
  hatch is the trailing unquoted `${EXTRA_ARGS:-}` in each recipe's exec
  line (user-supplied vLLM args, appended last so they can override the
  recipe's own flags).
- **All scripts are idempotent** and print in uv's style via the shared
  `prepare/_ui.py`: verb-led lines (first letter always capitalized),
  `+` / `·` / `×` markers, bold stage headers that group their sub-lines,
  counts, elapsed time, and in-place progress (uv-style bars for measured
  work, spinners for the rest; the bar width adapts to the terminal,
  every redraw erases the whole line) whose finished or failed status
  overwrites the in-progress line; a child tool's own output (uv) is
  re-emitted indented six spaces, as sub-output of its stage; non-TTY
  output degrades to one line per event. TTY + NO_COLOR aware.
- **No unmeasured numbers**: the README matrix's measurement columns
  (decode, prefill 16k, quality) are the user's to fill on this hardware.
  Numbers taken from upstream are single-card or a different shape; they
  must be labeled as upstream measurements, never presented as ours.

## Layout

```
Dockerfile  requirements.txt  setup.py  README.md
docker/  entrypoint.sh, prepare.sh
recipes/ the nine *.sh
prepare/ build_fast_model.py, build_swift_model.py, fetch_dflash2.py, fetch_dspark.py, harden_chat_template.py, patch_vllm.py, _ui.py
patches/ the 35 synced patches + patches/series (the apply order)
```

Defaults: venv `.venv/`, models under `models/`, port 8080, and `Qwen3.8-
27B` is served as `qwen3.8-27b` in every recipe (clients are pinned to that
name).

## Syncing with upstream

`main` tracks the `upstream` remote (`syv-ai/qwen38-27b-rtx3090`,
fast-forward only — keep it clean of local commits). The `recipes` branch
does not merge main; sync is a manual diff:

1. On main: `git fetch upstream && git merge upstream/main`.
2. Read the new upstream commits; the useful artifacts are `patches/*.patch`,
   `single-user/` and `batch/` launchers (the flag/lanes source), `docs/`
   (measured numbers and gotchas), and `prepare/` scripts.
3. Adopt relevant patches into `patches/` and launcher knowledge into the
   recipes; note upstream provenance + measured deltas in comments.
4. Validate: the full set applies in order on a pristine Linux vllm 0.28.0
   and compiles — `prepare/patch_vllm.py` does exactly this and is
   re-run on every setup/serve.

Caveats: on this Mac, the macOS vllm wheel is shorter than the Linux one
and hunks in some patches don't match, so patch validation must happen in
a Linux venv (the Docker build does it). Use `gpatch` on macOS; the
system `patch` is netBSD-derived and incompatible. The upstream OpenCode
client docs (a README section pointing OpenCode at their launcher) are for
their single-user flow, not adopted here (user decision 2026-09-11).

## Before calling a change done

- `prepare/patch_vllm.py` clean on the target venv (it self-heals or
  resets; a stamp is only written after the tree compiles)
- `bash -n` on every recipe; `python3 -m py_compile` on every script
- a reviewer pass on non-trivial staged diffs; P0 findings block commit
- no comments introduced inside backslash continuation chains
