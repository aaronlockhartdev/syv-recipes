---
name: upstream-sync
description: Pull changes from the upstream qwen38-27b-rtx3090 repo (the `upstream` remote) into this repo's recipes layer. Use when asked to sync with upstream, pull upstream changes, check for new upstream patches or flags, or adopt upstream improvements. Applies flag/perf/correctness improvements automatically; presents new features to the user for a decision before adopting anything feature-shaped.
---

# Upstream Sync

Pull changes from the upstream repo (`upstream` remote =
`syv-ai/qwen38-27b-rtx3090`) into the recipes layer. Upstream is the
technical source of truth (patch set, measured numbers, correctness
gotchas); this repo owns the deployment layer (recipes, default flags,
naming, packaging). **Read `AGENTS.md` first** — its Invariants are law
during a sync, and its "Removed from the upstream stack" list is the
record of decisions already made.

## The decision rule

- **New feature → present it to the user and wait.** Nothing
  feature-shaped is applied until the user decides. The outcome
  (adopt/skip) is recorded so the same feature is never presented twice.
- **Flag, performance, or correctness improvement → apply
  automatically**, with upstream provenance noted wherever it lands.
- **Deliberately dropped upstream features are out of scope** (KVarN
  4/2-bit KV, lookup-augmented drafting, n-gram chains, the whole WSL2
  lane — see `AGENTS.md` and the README). Never adopt, never present —
  unless a change touches code both stacks share and alters behavior of
  a path this repo depends on; that is presented as a correctness item.
- **A vLLM version bump is always presented on its own, first.** It
  re-baselines every patch, `requirements.txt`, the Dockerfile, and
  every "written against vLLM X" claim, so the rest of the range cannot
  be classified against the new base until it is decided.

## 1. Preconditions

1. On the `recipes` branch with a clean tree. If not, ask the user to
   commit or stash first — do not auto-stash.
2. `git fetch upstream`.
3. Keep `main` honest (it mirrors upstream, fast-forward only):
   `git checkout main && git merge --ff-only upstream/main && git checkout recipes`.
   If the fast-forward fails, `main` has local commits — a broken
   invariant; stop and surface it.
4. **First run only** — establish the sync marker:
   `git tag upstream/synced $(git merge-base recipes main)`
   (the upstream commit the recipes layer last forked from). From then
   on the tag is the source of truth — `main`'s position is not, because
   this branch never merges `main`.

## 2. Scope the range

```bash
git rev-parse refs/tags/upstream/synced upstream/main
git log --oneline refs/tags/upstream/synced..upstream/main
```

If the two revs are equal: up to date — say so and stop.

Upstream's history carries merge commits labeled `Merge #NN`. Work the
range **one PR at a time, in order** — the range can be large (the first
run spans the whole fork gap), and one pass will not digest it. Move
`upstream/synced` as you go: never past a commit whose diff you have not
examined, and never past a presented feature until the user has decided
and the outcome is applied or recorded.

If `upstream/synced` is not an ancestor of `upstream/main`
(`git merge-base --is-ancestor`), upstream rewrote its history:
re-derive the range as a tree diff between the two endpoints
(`git diff upstream/synced..upstream/main`) instead of a commit list,
and move the tag once the tree diff is fully examined.

## 3. What each upstream area means here

Diff per area (`git diff upstream/synced..upstream/main -- <area>`) and
classify before touching anything:

| upstream area | source of… | lands here as | default |
|---|---|---|---|
| `patches/*.patch` that we keep | fixes/perf to features we ship | copied into `patches/`, header provenance refreshed | auto |
| `patches/*.patch` new | see classification below | `patches/` (+ README "The kept patches") | auto or present |
| `single-user/`, `batch/` launchers | the flag/lane values our recipes copy | the `exec vllm serve` lines in `recipes/*.sh` | auto* |
| `docs/` (gotchas, quality, optimizations, long-context, …) | measured numbers, correctness gotchas | `README.md` Notes + recipe headers | auto |
| `prepare/` (build/fetch/quant scripts) | model-prep mechanics | `prepare/` | auto |
| `Dockerfile`, pinned vLLM | the vLLM base version | same files here | **present** (the version-bump decision) |
| `bench/`, `drafter/`, `kvarn/`, `.github/`, `verify.sh`, `docker-compose.yml` | upstream's own harness/CI | nothing — we validate with `prepare/patch_vllm.py` | skip (note interesting ideas in the digest) |

\* "auto" means *the change*, not the lane: launcher edits that enable a
dropped lane (KVarN, WSL2, lookup/ngram) or that change user-visible
quality or VRAM behavior are presented, not applied.

Classifying a **new patch**: it is a **fix/improvement (auto)** when it
touches files, env vars, or lanes this repo already ships (the dflash2
drafter, int8/int4-KV verify, the w4a8 Marlin linears, the MTP head,
vision offload, the speed-knobs envs — check against our `patches/` and
the recipe headers); it is a **feature (present)** when it introduces a
quantization type, spec-decode method, cache shape, or lane we do not
have. Either way, anything that would violate an `AGENTS.md` invariant
is a present item, not an auto one.

## 4. Apply the automatic changes

- **Patches**: copy into `patches/`. Alphabetical order is build order —
  a later patch may depend on files an earlier one creates; if
  upstream's order would change ours, say so in the commit message.
  Keep a provenance line in each header (`upstream <sha>`, PR, or
  issue).
- **Recipes**: update the `exec vllm serve` lines to the new upstream
  defaults, preserving the invariants — naming scheme, MTP's
  `cudagraph_mode: PIECEWISE`, int4's `--prefix-match-unit 848`, no fp8
  KV, no `--language-model-only`, the vision-offload default,
  empty-means-unset env handling, and "recipes are data" (no config
  branching, no comment lines inside backslash continuation chains).
  A combination upstream now measures well that we have no recipe for
  is a **new recipe — a present item** (new file + `docker/entrypoint.sh`
  arm + README table and matrix row).
- **`README.md`**: refresh "The kept patches" (one-liner + provenance),
  Notes (upstream's numbers stay labeled *upstream measurements*, never
  presented as ours), the matrix, and any version claims.
- **`AGENTS.md`**: patch count, and anything the Invariants section must
  now know.
- **Deviations**: where this repo deliberately differs from upstream's
  new state, the recipe or patch header carries an explicit
  "deviation from upstream" disclosure — keep those true as things
  change.

## 5. Present the features, then stop

One block per feature, in this shape:

```
FEATURE: <name>   (upstream <sha> / PR #NN / issue #NN)
  what:        <1–2 lines, from the upstream commit message or docs>
  measured:    <upstream's numbers on their reference box — always labeled theirs>
  impact here: <which of the five recipes / matrix cells it touches,
               and what it means on 2x3090 TP=2, 24 GB>
  call:        adopt | skip | watch — <recommendation and why>
```

End with an explicit "waiting on your call" list. **Do not apply any
feature change, and do not move `upstream/synced` past those commits,
until the user responds.** On *adopt*: apply per section 4 with
provenance. On *skip*: append it to the README's "Removed from the
upstream stack" paragraph (and to `AGENTS.md` Invariants if it is
durable) — a recorded skip is never presented again.

## 6. Validate before committing

A sync is done only when all of these hold (AGENTS.md "Before calling a
change done"):

1. `prepare/patch_vllm.py` clean **on a Linux venv** — this Mac's vllm
   wheel is shorter than the Linux one and some hunks will not match.
   Run the Docker build (`docker build -t syv-recipes .`, which applies
   the set in order and compileall-gates it) or a Linux box with the
   `.venv`. If the vLLM pin moved, update `requirements.txt`/`Dockerfile`
   first.
2. `bash -n` on every touched script; `python3 -m py_compile` on every
   touched Python file.
3. A `reviewer` pass on the staged diff; P0 findings block the commit
   (fix, re-stage, re-review).

## 7. Close the loop

1. Move the marker:
   `git tag -f upstream/synced <last upstream commit fully examined>`.
2. Commit on `recipes`: `sync: upstream <old>..<new> — <what was adopted;
   what was presented>`, and push.
