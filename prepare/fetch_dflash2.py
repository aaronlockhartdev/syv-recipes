#!/usr/bin/env python3
"""Fetch the W4A16 DFlash2 block drafter into a destination directory.

The int4-GPTQ requantization of incoai/Qwen3.8-27B-DFlash2 (1.2 GB vs the
3.85 GB bf16 original), prebuilt on the Hub. Downloaded through Hugging
Face's built-in cache, then linked into the destination dir.

Usage:  python prepare/fetch_dflash2.py DEST_DIR

Idempotent.
"""
import os
import shutil
import sys
import time

import _ui as ui

REPO = "syvai/Qwen3.8-27B-DFlash2-W4A16"
FILES = ("config.json", "model.safetensors")
_BAR_MIN = 8 << 20  # copies above this size run as a progress bar


def _copy(src, dstp, size):
    """Copy one file into the destination; the big ones run as a bar."""
    f = os.path.basename(src)
    if size > _BAR_MIN:
        p = ui.Progress(f"Copying {f} ({ui.human(size)})", total=size)
        with open(src, "rb") as a, open(dstp, "wb") as b:
            while True:
                chunk = a.read(1 << 20)
                if not chunk:
                    break
                b.write(chunk)
                p.tick(len(chunk))
        p.finish(True, f"Copied {f} ({ui.human(size)})")
    else:
        shutil.copy(src, dstp)
        ui.ok(f"Copied {f} ({ui.human(size)})")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    t0 = time.monotonic()

    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("Usage: python prepare/fetch_dflash2.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    ui.stage(f"Fetching {REPO}")
    t_r = time.monotonic()
    p = ui.Progress(f"Fetching {REPO}")
    try:
        hub = ui.snapshot(REPO, progress=p)
    except Exception as e:
        p.finish(False, f"Fetching {REPO} failed ({e!r})",
                 "check the network and Hugging Face reachability; once cached, re-runs are offline",
                 fatal=True)
    p.finish(True, f"Fetched {REPO} in {ui.dur(time.monotonic() - t_r)}")

    ui.stage(f"Installing {dst}")
    for f in FILES:
        src = os.path.realpath(os.path.join(hub, f))
        dstp = os.path.join(dst, f)
        if os.path.isfile(dstp) and os.path.getsize(dstp) == os.path.getsize(src):
            continue
        if os.path.lexists(dstp):
            os.remove(dstp)
        size = os.path.getsize(src)
        try:
            os.link(src, dstp)
        except OSError:
            _copy(src, dstp, size)
            continue
        ui.ok(f"Hard-linked {f} ({ui.human(size)})")

    ui.done(f"DFlash2 drafter ready: {dst} ({ui.dur(time.monotonic() - t0)})")
    ui.note("Serve with: the dflash2-family recipes (this dir as DRAFT)")


if __name__ == "__main__":
    main()
