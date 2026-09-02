#!/usr/bin/env python3
"""Build the modified Qwen3.8-27B "fast" model into a destination directory.

Assembled on the CPU from two Hub repos, fetched through Hugging Face's
built-in cache (so re-runs never re-download), plus one file from a third:

  dbirks/Qwen3.8-27B-W4A16-AutoRound
      the W4A16 AutoRound checkpoint: hard-linked in (shards 1-5, tokenizer, ...)
  syvai/qwen3.8-27b-3090-fast-variant
      the overlay: int4-GPTQ lm_head (shard 7), int4 MTP module + 40k draft
      head (model_extra_tensors.safetensors), the draft-vocab ids, config, index
  peculiar-ragdoll/Qwen-Sharp-Chat-Templates
      one file: chat_template.jinja (Qwen-Sharp v22.4.0) replaces the base
      checkpoint's stock template (token-efficient thinking and tool calls)

The one local step: embed_tokens (~1.3 GB bf16) is requantized to int8
group-128 symmetric in shard 6, the layout CompressedTensorsEmbeddingWNA16Int
expects (the serving side needs patches/qwen3_5-embed-quant.patch).
~8 GB peak RAM, a few minutes on a desktop CPU.

Usage:  python prepare/build_fast_model.py DEST_DIR

Idempotent and resumable: a complete DEST_DIR is left alone, an interrupted
one is repaired on the next run, and nothing is ever written into the shared
HF cache.
"""

import json
import os
import shutil
import struct
import sys

import torch
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_REPO = "dbirks/Qwen3.8-27B-W4A16-AutoRound"
FAST_REPO = "syvai/qwen3.8-27b-3090-fast-variant"
TEMPLATE_REPO = "peculiar-ragdoll/Qwen-Sharp-Chat-Templates"
TEMPLATE_FILE = "chat_template.jinja"

# files from the fast-variant repo that replace/extend the base dir
OVERLAY = (
    "model-00007-of-00007.safetensors",
    "model_extra_tensors.safetensors",
    "mtp_draft_vocab_ids.pt",
    "config.json",
    "model.safetensors.index.json",
)
# the one shard this script rewrites (copy, never link: in-place edits must
# not write through into the shared HF cache)
EMBED_SHARD = "model-00006-of-00007.safetensors"

GROUP, BITS, QMAX = 128, 8, 127


def _shard_meta(path):
    """(header metadata, file size) of a safetensors file, or None when the
    file is missing, truncated, or corrupt (e.g. a killed mid-write copy)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return None
            (n,) = struct.unpack("<Q", head)
            if 8 + n > size:
                return None
            meta = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None
    return meta, size


def safetensors_keys(path):
    """The tensor names in a safetensors file's header (no tensor loads).
    None if the file is unreadable or truncated."""
    r = _shard_meta(path)
    return set(r[0].keys()) if r is not None else None


def _snapshot(repo):
    # local-first: a warm cache resolves without any network (a blackholed
    # network must not stall a model-ready boot); the first run downloads
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return snapshot_download(repo)


def _template_path():
    # local-first like _snapshot; None when it cannot be fetched (offline,
    # cold cache) -- the build then keeps whatever chat template the base
    # checkpoint shipped
    try:
        return hf_hub_download(TEMPLATE_REPO, TEMPLATE_FILE, local_files_only=True)
    except Exception:
        pass
    try:
        return hf_hub_download(TEMPLATE_REPO, TEMPLATE_FILE)
    except Exception as e:
        print(f"  (Qwen-Sharp template unavailable: {e}; keeping the base chat template)")
        return None


def requant_embed(path, key):
    """In-place int8 group-128 symmetric requantization of one embedding."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    w = tensors.pop(key).to(torch.float32)
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8)
    q = q.reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"  embed round-trip relative error: {err:.4f}")
    assert err < 0.01, "quantization error too high, aborting"
    stem = key[: -len(".weight")]
    tensors[stem + ".weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    # the embedding path creates scales in params_dtype (bf16), unlike the linears
    tensors[stem + ".weight_scale"] = scale.squeeze(-1).to(torch.bfloat16).contiguous()
    tensors[stem + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    # temp file + rename: a crash mid-write must never leave a half-written
    # shard with a valid header behind
    save_file(tensors, path + ".tmp", metadata=meta or {"format": "pt"})
    os.replace(path + ".tmp", path)


def file_ready(dst_path, src_path):
    return os.path.isfile(dst_path) and os.path.getsize(dst_path) == os.path.getsize(src_path)


def template_ready(dst, tsrc):
    """True when DEST_DIR already holds the current Qwen-Sharp template."""
    dst_t = os.path.join(dst, TEMPLATE_FILE)
    if tsrc is None:
        return os.path.isfile(dst_t)  # fetch failed: any template is fine
    return file_ready(dst_t, tsrc)


def install(dst_path, src_path, linkable):
    """Put src at dst: hard-link where possible, copy otherwise. Never over an existing file."""
    if os.path.lexists(dst_path):
        os.remove(dst_path)
    if linkable:
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            pass  # cross-device: copy
    shutil.copy(src_path, dst_path)


def complete(dst, fast_dir):
    """dst holds the final (fast) index and every file it references, with the
    packed embed tensors physically present in their shard."""
    try:
        dst_idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
        fast_idx = json.load(open(os.path.join(fast_dir, "model.safetensors.index.json")))
    except (OSError, ValueError):
        return False
    if dst_idx.get("weight_map") != fast_idx.get("weight_map"):
        return False
    # the draft-vocab ids are not in the index but the MTP draft head reads them
    if not os.path.isfile(os.path.join(dst, "mtp_draft_vocab_ids.pt")):
        return False
    for f in sorted(set(dst_idx["weight_map"].values())):
        if not os.path.isfile(os.path.join(dst, f)):
            return False
    # a killed mid-write copy passes everything above with a truncated shard:
    # check the packed embed tensors are physically inside the file
    packed_key = next(
        k for k in fast_idx["weight_map"] if k.endswith("embed_tokens.weight_packed")
    )
    r = _shard_meta(os.path.join(dst, fast_idx["weight_map"][packed_key]))
    if r is None:
        return False
    meta, size = r
    stem = packed_key[: -len(".weight_packed")]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        t = meta.get(stem + "." + s)
        if t is None or t["data_offsets"][1] > size:
            return False
    return True


def verify_draft_vocab(model_dir):
    """The draft-vocab .pt and the packed head it feeds must agree in size:
    the .pt's id count equals the head's row count (both 40960 in the
    published variant). isfile alone passes a truncated .pt, which then dies
    at server boot when the MTP draft-vocab patch torch.loads it. Compares
    the count only -- the ids have no second source of truth; a same-count
    file with different ids passes here but degrades draft acceptance."""
    try:
        dst_idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        ids = torch.load(
            os.path.join(model_dir, "mtp_draft_vocab_ids.pt"),
            map_location="cpu", weights_only=True,
        )
    except Exception as e:
        sys.exit(f"draft-vocab .pt in {model_dir} is unreadable ({e!r}) -- "
                 "re-fetch the fast-variant overlay")
    packed = "mtp.draft_lm_head.weight_packed"
    path = dst_idx.get("weight_map", {}).get(packed)
    meta = _shard_meta(os.path.join(model_dir, path)) if path else None
    rows = meta[0].get(packed, {}).get("shape", [None])[0] if meta else None
    assert rows == len(ids), (
        f"draft-vocab mismatch: {len(ids)} ids vs {rows} draft-head rows -- "
        "re-fetch the fast-variant overlay"
    )
    print(f"  draft vocab: {len(ids)} ids == {rows}-row draft head")


def main():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("usage: python prepare/build_fast_model.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    print(f"== HF cache: {BASE_REPO}")
    base = _snapshot(BASE_REPO)
    print(f"== HF cache: {FAST_REPO}")
    fast = _snapshot(FAST_REPO)
    tsrc = _template_path()
    # fail fast: a bad download or an unexpected overlay layout costs
    # seconds here instead of a full copy + requant build
    verify_draft_vocab(fast)

    if complete(dst, fast) and template_ready(dst, tsrc):
        # dirs built before this check existed (or hit by in-place corruption
        # after a good run) are only caught here
        verify_draft_vocab(dst)
        print("fast model already complete:", dst)
        return

    base_idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    embed_key = next(k for k in base_idx["weight_map"] if k.endswith("embed_tokens.weight"))

    embed_done = False
    if os.path.isfile(os.path.join(dst, EMBED_SHARD)):
        keys = safetensors_keys(os.path.join(dst, EMBED_SHARD))
        embed_done = bool(keys) and (embed_key[: -len(".weight")] + ".weight_packed" in keys)

    for f in sorted(os.listdir(base)):
        if f in (".gitattributes",) or f in OVERLAY:
            continue
        src = os.path.realpath(os.path.join(base, f))
        dstp = os.path.join(dst, f)
        if f == EMBED_SHARD:
            if embed_done:
                continue
            print(f"== {f}: copy + requantize embed_tokens to int8 (the only local step)")
            install(dstp, src, linkable=False)
            requant_embed(dstp, embed_key)
        elif file_ready(dstp, src):
            continue
        else:
            print(f"== {f}: {'link' if f.endswith(('.safetensors',)) else 'copy'} from cache")
            install(dstp, src, linkable=True)

    for f in OVERLAY:
        src = os.path.realpath(os.path.join(fast, f))
        dstp = os.path.join(dst, f)
        if file_ready(dstp, src):
            continue
        print(f"== {f}: fast-variant overlay")
        install(dstp, src, linkable=True)

    if tsrc is not None:
        dst_t = os.path.join(dst, TEMPLATE_FILE)
        if not file_ready(dst_t, tsrc):
            print(f"== {TEMPLATE_FILE}: Qwen-Sharp template ({TEMPLATE_REPO})")
            install(dst_t, os.path.realpath(tsrc), linkable=False)

    assert complete(dst, fast), "assembly finished but the completeness check still fails"
    verify_draft_vocab(dst)
    print("fast model ready:", dst)
    print("serve with: recipes/w4a16-int8-dflash2.sh (this dir as MODEL)")


if __name__ == "__main__":
    main()
