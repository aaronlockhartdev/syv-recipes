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

from huggingface_hub import snapshot_download

REPO = "syvai/Qwen3.8-27B-DFlash2-W4A16"
FILES = ("config.json", "model.safetensors")

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def dim(s):
    print(c("2", s))


def ok(s):
    print(c("32", f"+ {s}"))


def done(s):
    print(c("32", f"\u2713 {s}"))


def fail(msg, *hints):
    print(c("31", f"\u00d7 {msg}"))
    for h in hints:
        print(c("2", f"  \u2570\u2500> {h}"))
    sys.exit(1)


def dur(t0):
    d = int((time.monotonic() - t0) * 1000)
    return f"{d}ms" if d < 1000 else f"{d / 1000:.1f}s"


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _snapshot(repo):
    # local-first: a warm cache resolves without any network (a blackholed
    # network must not stall a model-ready boot); the first run downloads
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return snapshot_download(repo)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    t0 = time.monotonic()

    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("usage: python prepare/fetch_dflash2.py DEST_DIR")
    dst = os.path.abspath(sys.argv[1])
    os.makedirs(dst, exist_ok=True)

    dim(f"\u00b7 fetching {REPO}")
    try:
        hub = _snapshot(REPO)
    except Exception as e:
        fail(f"cannot fetch {REPO} ({e!r})",
             "check the network and Hugging Face reachability; once cached, re-runs are offline")
    ok(f"{REPO} ({dur(t0)})")
    for f in FILES:
        src = os.path.realpath(os.path.join(hub, f))
        dstp = os.path.join(dst, f)
        if os.path.isfile(dstp) and os.path.getsize(dstp) == os.path.getsize(src):
            continue
        if os.path.lexists(dstp):
            os.remove(dstp)
        if f.endswith(".safetensors"):
            size = human(os.path.getsize(src))
            dim(f"\u00b7 copying {f} ({size})")
        try:
            os.link(src, dstp)
            method = "hard-linked"
        except OSError:
            shutil.copy(src, dstp)  # cross-device
            method = "copied"
        ok(f"{f}: {method}" + (f" ({size})" if f.endswith(".safetensors") else ""))
    done(f"DFlash2 drafter ready: {dst} ({dur(t0)})")
    dim("  serve with: the dflash2-family recipes (this dir as DRAFT)")


if __name__ == "__main__":
    main()
