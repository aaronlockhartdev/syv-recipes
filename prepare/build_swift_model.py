#!/usr/bin/env python3
"""Build the modified Swift-Qwen3.8-27B W4A16 model into a destination dir.

Assembled on the CPU from one Hub repo, fetched through Hugging Face's
built-in cache (so re-runs never re-download):

  jamesbrunet/Swift-Qwen3.8-27b-W4A16-AutoRound
      the W4A16 AutoRound quant of ukisai/Swift-Qwen3.8-27b (a reasoning-
      efficient fine-tune of Qwen3.8-27B, ~58% fewer thinking tokens at
      <1% accuracy cost on ukisai's benches). As published: 4-bit Linear
      layers (group-128, symmetric, compressed-tensors pack-quantized),
      bf16 lm_head / embed_tokens / MTP module, ~20 GB over 67 shards.
      Distributed under UkisAI's Swift Open License v1.0 (restrictive
      terms) -- fetched and built here, never committed.

The local steps are the same operations the fast variant applies to the
base model (upstream's prepare/quant_lm_head.py, quant_embed.py,
quant_mtp.py), in the plain round-to-nearest flavor:

  1. lm_head      bf16 -> int8 group-128 (~1.3 GB freed; fp16 scales,
                  like the checkpoint's linears)
  2. embed_tokens bf16 -> int8 group-128 (~1.3 GB freed; bf16 scales --
                  the embedding path expects them in params_dtype; the
                  serving side needs patches/qwen3_5-embed-quant.patch)
  3. the mtp.* module (bf16, ~850 MB, in model_extra_tensors.safetensors)
     -> int8 group-128: the published config's "Linear" target matches
     these tensors, and as plain bf16 they break any speculative load
     (upstream's scripts fix exactly that). mtp.fc is adaptive: if its
     round-trip error exceeds the threshold it stays bf16 (upstream's
     experimental --keep-fc shape; upstream verified int8 for the module
     as a whole, not fc retention), the config then targets the decoder
     linears only, and fc goes into the ignore list
  4. the froggeric/Qwen-Fixed-Chat-Templates template (v22.5) replaces
     the repo's stock one, like build_fast_model.py does

The GPTQ-calibrated int4 upgrades and the 40k MTP draft head (with its
draft-vocab ids) are deliberately NOT built here: they need the upstream
drafter/ training pipeline (calibration hidden states, Hessians, output
corpus) run against this fine-tune. Until that re-fit lands, the MTP
recipe runs the native full-vocab head, and the dflash2/dspark drafters
-- trained on the base model -- may show reduced acceptance; measure
before trusting them.

~8 GB peak RAM, a few minutes on a desktop CPU.

Usage:  python prepare/build_swift_model.py DEST_DIR

Idempotent and resumable: a complete DEST_DIR is left alone, an
interrupted one is repaired on the next run, and nothing is ever written
into the shared HF cache.
"""
import copy
import json
import os
import shutil
import struct
import sys
import time

import torch
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

import _ui as ui

REPO = "jamesbrunet/Swift-Qwen3.8-27b-W4A16-AutoRound"
TEMPLATE_REPO = "froggeric/Qwen-Fixed-Chat-Templates"
TEMPLATE_FILE = "chat_template.jinja"

# the small files every install needs; the index's shard list is added to
# the require list once the index itself is readable
META = (
    "config.json",
    "model.safetensors.index.json",
    "model_extra_tensors.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
)

GROUP, BITS, QMAX = 128, 8, 127
_BAR_MIN = 8 << 20

# the MTP module's bf16 linears (upstream prepare/quant_mtp.py list); the
# norms and pre_fc norms in the same file stay bf16
MTP_LINEARS = (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
)
LM_KEY = "lm_head.weight"


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


def _snapshot(progress=None, require=None):
    # local-first: a warm cache resolves without any network (a blackholed
    # network must not stall a model-ready boot); the first run downloads,
    # feeding progress
    return ui.snapshot(REPO, progress=progress, require=require)


def _template_path():
    # local-first like _snapshot; None when it cannot be fetched (offline,
    # cold cache) -- the build then keeps whatever chat template the repo
    # shipped
    try:
        return hf_hub_download(TEMPLATE_REPO, TEMPLATE_FILE, local_files_only=True)
    except Exception:
        pass
    try:
        return hf_hub_download(TEMPLATE_REPO, TEMPLATE_FILE)
    except Exception as e:
        ui.note(f"Could not fetch {TEMPLATE_FILE} from {TEMPLATE_REPO} ({e}); keeping the repo's chat template")
        return None


def _index_files(hub_dir):
    """All safetensors the index references (shards + the extra-tensors
    file), or None when the index is unreadable."""
    try:
        idx = json.load(open(os.path.join(hub_dir, "model.safetensors.index.json")))
        return sorted(set(idx["weight_map"].values()))
    except (OSError, ValueError):
        return None


def _rtn(w):
    """(packed, scale, round-trip relative error) of a 2-D weight at
    group-128 int8 symmetric, round-to-nearest."""
    out_f, in_f = w.shape
    if in_f % GROUP:
        ui.fail(f"hidden size {in_f} is not a multiple of group {GROUP}",
                "This script assumes the published shard layout; re-check the upstream quant")
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    return q, scale, err


def _packed_into(tensors, key, q, scale, scale_dtype):
    """The packed trio for one requantized weight into a tensor dict.
    Linear layers use fp16 scales in this checkpoint; the embedding path
    creates scales in params_dtype (bf16) -- the caller picks."""
    stem = key[: -len(".weight")]
    tensors[stem + ".weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    tensors[stem + ".weight_scale"] = scale.squeeze(-1).to(scale_dtype).contiguous()
    tensors[stem + ".weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)


def _commit(tensors, meta, path):
    """Write the tensor dict through temp file + rename: a crash mid-write
    must never leave a half-written file with a valid header behind."""
    save_file(tensors, path + ".tmp", metadata=meta or {"format": "pt"})
    os.replace(path + ".tmp", path)


def requant(path, key, scale_dtype):
    """In-place int8 group-128 symmetric requantization of one weight
    (the round-to-nearest flavor of upstream's quant_* scripts).
    Aborts when the round-trip error is too high."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    q, scale, err = _rtn(tensors.pop(key).to(torch.float32))
    if err >= 0.01:
        ui.fail(f"{key}: round-trip relative error {err:.4f} >= 0.01 -- aborting")
    ui.note(f"{key}: int8 group-{GROUP}, round-trip rel error {err:.4f}")
    _packed_into(tensors, key, q, scale, scale_dtype)
    _commit(tensors, meta, path)

def requant_mtp(path):
    """The whole MTP module in one file, one pass. Returns True when
    mtp.fc was kept bf16: on this checkpoint its round-to-nearest int8
    error exceeds the threshold, and upstream's experimental --keep-fc
    shape (fc unquantized, +~50 MB) is the safe one -- upstream verified
    int8 for the module as a whole, not fc retention. The config then
    targets the decoder linears only, and fc goes into the ignore list
    so no group claims the bf16 tensor."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    keep_fc = False
    q, scale, err = _rtn(tensors["mtp.fc.weight"].to(torch.float32))
    if err >= 0.01:
        keep_fc = True
        ui.note(f"mtp.fc.weight: int8 round-trip rel error {err:.4f} >= 0.01 -- kept bf16 (upstream's experimental --keep-fc shape)")
    else:
        ui.note(f"mtp.fc.weight: int8 group-{GROUP}, round-trip rel error {err:.4f}")
        tensors.pop("mtp.fc.weight")
        _packed_into(tensors, "mtp.fc.weight", q, scale, torch.float16)
    for m in MTP_LINEARS:
        if m == "mtp.fc":
            continue
        key = m + ".weight"
        q, scale, err = _rtn(tensors.pop(key).to(torch.float32))
        if err >= 0.01:
            ui.fail(f"{key}: round-trip relative error {err:.4f} >= 0.01 -- aborting")
        ui.note(f"{key}: int8 group-{GROUP}, round-trip rel error {err:.4f}")
        _packed_into(tensors, key, q, scale, torch.float16)
    _commit(tensors, meta, path)
    return keep_fc


def file_ready(dst_path, src_path):
    return os.path.isfile(dst_path) and os.path.getsize(dst_path) == os.path.getsize(src_path)


def template_ready(dst, tsrc):
    """True when DEST_DIR already holds the current froggeric template."""
    dst_t = os.path.join(dst, TEMPLATE_FILE)
    if tsrc is None:
        return os.path.isfile(dst_t)  # fetch failed: any template is fine
    return file_ready(dst_t, tsrc)


def _hardlink(src, dst):
    try:
        os.link(src, dst)
        return True
    except OSError:
        return False  # cross-device: copy


def _copy_progressed(src, dst, progress):
    """A chunked copy feeding progress (1 MiB at a time), copystat at the
    end. Replaces shutil.copy for the files big enough to bar."""
    with open(src, "rb") as a, open(dst, "wb") as b:
        while True:
            chunk = a.read(1 << 20)
            if not chunk:
                break
            b.write(chunk)
            progress.tick(len(chunk))
    shutil.copystat(src, dst)


def packed_ok(path, stem):
    """The packed trio of one requantized tensor is physically inside the
    file (a killed mid-write copy has a valid header but no tail)."""
    r = _shard_meta(path)
    if r is None:
        return False
    meta, size = r
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        t = meta.get(stem + "." + s)
        if t is None or t["data_offsets"][1] > size:
            return False
    return True


def complete(dst):
    """dst holds the final index and every file it references, the three
    requantized tensors are packed, and the config carries the four
    groups."""
    try:
        idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
        cfg = json.load(open(os.path.join(dst, "config.json")))
    except (OSError, ValueError):
        return False
    wm = idx.get("weight_map") or {}
    qc = (cfg.get("quantization_config") or {})
    groups = qc.get("config_groups") or {}
    if set(groups) != {"group_0", "group_1", "group_2", "group_3"}:
        return False
    if "lm_head" in (qc.get("ignore") or []):
        return False
    if not os.path.isfile(os.path.join(dst, "model_extra_tensors.safetensors")):
        return False
    for f in sorted(set(wm.values())):
        if not os.path.isfile(os.path.join(dst, f)):
            return False
    # the final index only holds the packed keys (assembly replaced the
    # plain .weight keys), so the guard checks the packed trio directly
    # -- like build_fast_model, a truncated shard must fail here
    stems = {
        "lm_head.weight_packed": "lm_head",
    }
    for m in MTP_LINEARS[1:]:
        stems[m + ".weight_packed"] = m
    embed_packed = next((k for k in wm if k.endswith("embed_tokens.weight_packed")), None)
    if embed_packed:
        stems[embed_packed] = embed_packed[: -len(".weight_packed")]
    for key, stem in stems.items():
        shard = wm.get(key)
        if not shard or not packed_ok(os.path.join(dst, shard), stem):
            return False
    # mtp.fc: packed when group_3 covers all of mtp.; plain bf16 + ignore
    # when it was kept (upstream's --keep-fc shape)
    g3t = ((groups.get("group_3") or {}).get("targets") or [None])[0]
    r = _shard_meta(os.path.join(dst, "model_extra_tensors.safetensors"))
    if not r:
        return False
    fc_packed = packed_ok(os.path.join(dst, "model_extra_tensors.safetensors"), "mtp.fc")
    fc_plain = r[0].get("mtp.fc.weight")
    fc_plain = bool(fc_plain) and fc_plain["data_offsets"][1] <= r[1]
    fc_ignored = "mtp.fc" in (qc.get("ignore") or [])
    if g3t == r"re:^mtp\..*":
        return fc_packed and not fc_ignored
    if g3t == r"re:^mtp\.layers\..*":
        return fc_plain and not fc_packed and fc_ignored
    return False


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    t0 = time.monotonic()

    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("Usage: python prepare/build_swift_model.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    ui.stage("Fetching the Swift W4A16 quant")
    t_r = time.monotonic()
    p = ui.Progress(f"Fetching {REPO}")
    # require = the meta files, plus every shard the index references once
    # the index is readable (a partially downloaded repo must fall through
    # to the network pass and resume, never masquerade as warm)
    require = list(META)
    try:
        idx_file = hf_hub_download(REPO, "model.safetensors.index.json")
        require += [
            f for f in _index_files(os.path.dirname(idx_file))
            if f not in META
        ]
    except Exception:
        pass  # offline + cold cache: the snapshot below raises its own error
    try:
        hub = _snapshot(progress=p, require=require)
    except Exception as e:
        p.finish(False, f"Fetching {REPO} failed ({e!r})",
                 "check the network and Hugging Face reachability; once cached, re-runs are offline",
                 fatal=True)
    p.finish(True, f"Fetched {REPO} in {ui.dur(time.monotonic() - t_r)}")
    tsrc = _template_path()

    hub_idx = json.load(open(os.path.join(hub, "model.safetensors.index.json")))
    wm = hub_idx["weight_map"]
    embed_key = next(k for k in wm if k.endswith("embed_tokens.weight"))
    mtp_files = {wm[m + ".weight"] for m in MTP_LINEARS}
    if len(mtp_files) != 1:
        ui.fail(f"the mtp linears span several files: {sorted(mtp_files)}",
                "This script assumes the published layout (one extra-tensors file); check the repo")
    mtp_file = mtp_files.pop()

    # the files this script rewrites get copied (never linked: in-place
    # edits must not write through into the shared HF cache), the rest
    # hard-links -- including index and config, which the steps below
    # rewrite (open(dst, "w") would otherwise truncate the cache's copy
    # through the link)
    rewrite = {wm[embed_key], wm[LM_KEY], mtp_file,
               "model.safetensors.index.json", "config.json"}

    if complete(dst) and template_ready(dst, tsrc):
        ui.done(f"Swift W4A16 already complete: {dst} ({ui.dur(time.monotonic() - t0)})")
        return

    ui.stage(f"Assembling {dst}")
    for f in sorted(os.listdir(hub)):
        if f in (".gitattributes",) or (f == TEMPLATE_FILE and tsrc is not None):
            continue
        src = os.path.realpath(os.path.join(hub, f))
        if f in rewrite:
            # copy, never link: in-place edits must not write through into
            # the shared HF cache; and always re-copy -- a previous output
            # may be a half-edited file, and requant must start from the
            # cache's bytes (build_fast_model's force=True, same reason)
            dstp = os.path.join(dst, f)
            if os.path.lexists(dstp):
                os.remove(dstp)
            place = ui.Progress(f"Preparing {f} ({ui.human(os.path.getsize(src))})",
                                total=os.path.getsize(src))
            _copy_progressed(src, dstp, place)
            place.finish(True, f"Placed {f}")
        else:
            dstp = os.path.join(dst, f)
            if file_ready(dstp, src):
                continue
            if os.path.lexists(dstp):
                os.remove(dstp)
            size = os.path.getsize(src)
            if _hardlink(src, dstp):
                ui.ok(f"Hard-linked {f} ({ui.human(size)})")
            elif size > _BAR_MIN:
                pr = ui.Progress(f"Copying {f} ({ui.human(size)})", total=size)
                _copy_progressed(src, dstp, pr)
                pr.finish(True, f"Copied {f}")
            else:
                shutil.copy(src, dstp)
                ui.ok(f"Copied {f} ({ui.human(size)})")

    # the three local steps (round-to-nearest int8, in the copied files)
    t_q = time.monotonic()
    requant(os.path.join(dst, wm[LM_KEY]), LM_KEY, torch.float16)
    requant(os.path.join(dst, wm[embed_key]), embed_key, torch.bfloat16)
    keep_fc = requant_mtp(os.path.join(dst, mtp_file))
    ui.ok(f"Requantized lm_head, embed_tokens, mtp module in {ui.dur(time.monotonic() - t_q)}"
          + (" (mtp.fc kept bf16)" if keep_fc else ""))
    # index: the .weight keys become the packed trio, in the same files
    # (a kept-bf16 mtp.fc keeps its plain index entry)
    mtp_keys = [m + ".weight" for m in MTP_LINEARS
                if not (m == "mtp.fc" and keep_fc)]
    for key in [LM_KEY, embed_key] + mtp_keys:
        shard = wm[key]
        stem = key[: -len(".weight")]
        del wm[key]
        for s in ("weight_packed", "weight_scale", "weight_shape"):
            wm[stem + "." + s] = shard
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(hub_idx, f, indent=2)

    # config: clone group_0 (4-bit Linear) into the three specific groups
    # (8-bit), and drop lm_head from the ignore list (upstream's scripts).
    # A kept-bf16 mtp.fc goes into the ignore list, and group_3 targets
    # the decoder linears only (upstream's --keep-fc shape).
    cfg = json.load(open(os.path.join(dst, "config.json")))
    qc = cfg["quantization_config"]
    qc["ignore"] = [i for i in qc.get("ignore") or [] if i != "lm_head"]
    if keep_fc and "mtp.fc" not in qc["ignore"]:
        qc["ignore"] = qc["ignore"] + ["mtp.fc"]
    for name, target in (
        ("group_1", "re:.*lm_head$"),
        ("group_2", "re:.*embed_tokens$"),
        ("group_3", r"re:^mtp\.layers\..*" if keep_fc else r"re:^mtp\..*"),
    ):
        g = copy.deepcopy(qc["config_groups"]["group_0"])
        g["targets"] = [target]
        g["weights"]["num_bits"] = BITS
        qc["config_groups"][name] = g
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if tsrc is not None:
        dst_t = os.path.join(dst, TEMPLATE_FILE)
        if not file_ready(dst_t, tsrc):
            if os.path.lexists(dst_t):
                os.remove(dst_t)
            shutil.copy(tsrc, dst_t)
            ui.ok(f"Installed the {TEMPLATE_REPO} template")

    assert complete(dst), "assembly finished but the completeness check still fails"
    ui.done(f"Swift W4A16 ready: {dst} ({ui.dur(time.monotonic() - t0)})")
    ui.note("MTP: native full-vocab head (the 40k draft head is a later re-fit, upstream drafter/)")
    ui.note("Serve with:  MODEL=<this dir> bash recipes/w4a16-int8-mtp.sh")
    ui.note("The dflash2/dspark recipes load it too, but their drafters are base-trained -- measure acceptance first")


if __name__ == "__main__":
    main()
