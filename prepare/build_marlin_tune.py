#!/usr/bin/env python3
"""Build and install the tuned Marlin extension (VLLM_MARLIN_TUNE) into the
venv.

The extension is a standalone build of vLLM's Marlin GEMM source with a
runtime (size_n, size_k, key, m_max) -> tile-config table measured on the
target GPU; patches/marlin-tune-table.patch routes marlin_gemm through it
when VLLM_MARLIN_TUNE=1. The source tree is local build state -- upstream's
.dockerignore excludes it, so it is in neither git repo -- and this script
takes its path: as an argument, via MARLIN_TUNE_SRC, or from a probe
location (the repo root as marlin-tune/, or ~/marlin-tune). The tree must
contain src/build.sh; it carries its own CUDA 13 toolchain and builds in
place.

Steps (idempotent; stamp at $VENV/.marlin-tune-stamp = source digest +
vllm/torch versions):
  --rebench  first regenerate the tile table on this GPU (the tree's
             bench_marlin.py --grid; run it on an idle GPU)
             run src/build.sh, install the tree's src as a .pth in the
             venv's site-packages (so the patch's plain imports resolve),
             verify the import, stamp.
A re-run with an unchanged tree and a working import is a no-op; a changed
tree, a broken import, or a missing .pth triggers a rebuild.

The recipes turn VLLM_MARLIN_TUNE on automatically when the extension is
importable; VLLM_MARLIN_TUNE=0 keeps it off. Upstream ships it off for a
reason: at the 250 W cap the per-GEMM gains (+2-20% on W4A8 prefill,
+3-7% on the M<=16 verify GEMMs) wash out to ~+0.4% end-to-end -- this
mainly pays on uncapped cards.

Usage:  python prepare/build_marlin_tune.py [SRC_DIR] [--rebench]
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import _ui as ui

REPO = Path(__file__).resolve().parent.parent
VENV = Path(os.environ.get("VENV") or REPO / ".venv")
PY = VENV / "bin" / "python"
STAMP = VENV / ".marlin-tune-stamp"
PROBES = (REPO / "marlin-tune", Path.home() / "marlin-tune")
# build outputs excluded from the source digest (they change on every build)
_SKIP_DIRS = {"build", "cuda13-home", "__pycache__"}
_SKIP_SUFFIXES = (".so", ".o", ".pyc", ".pyd")


def _load_dotenv():
    """Fill unset-or-empty env vars from REPO/.env (same contract as
    patch_vllm.py and the recipes): the real environment always wins."""
    p = REPO / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not k.isidentifier() or k[0].isdigit():
            continue
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if v and not os.environ.get(k):
            os.environ[k] = v


def _src_digest(root: Path):
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(_SKIP_SUFFIXES):
                continue
            p = Path(dirpath) / name
            h.update(str(p.relative_to(root)).encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def _versions():
    """(vllm, torch) versions from package metadata (no imports)."""
    out = subprocess.run(
        [str(PY), "-c",
         "import importlib.metadata as m; print(m.version('vllm'), "
         "m.version('torch'))"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return out.stdout.split()


def _import_ok():
    out = subprocess.run(
        [str(PY), "-c", "import marlin_best, marlin_tune_ext"],
        capture_output=True, text=True)
    return out.returncode == 0, (out.stdout or "") + (out.stderr or "")


def _site_packages():
    out = subprocess.run(
        [str(PY), "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True)
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def main():
    _load_dotenv()
    argv = sys.argv[1:]
    rebench = "--rebench" in argv
    args = [a for a in argv if a != "--rebench"]
    if len(args) > 1:
        ui.fail("Usage: build_marlin_tune.py [SRC_DIR] [--rebench]")
    if not (VENV / "bin").is_dir():
        ui.fail(f"No venv at {VENV}",
                f"Create it:  uv venv {VENV} --python 3.12 && "
                f"uv pip install --python {PY} -r {REPO / 'requirements.txt'}")

    src = None
    if args:
        cands = [Path(a).expanduser().resolve() for a in args]
    elif os.environ.get("MARLIN_TUNE_SRC"):
        cands = [Path(os.environ["MARLIN_TUNE_SRC"]).expanduser().resolve()]
    else:
        cands = list(PROBES)
    for cand in cands:
        if (cand / "src" / "build.sh").is_file():
            src = cand
            break
    if src is None:
        if args:
            ui.fail(f"No Marlin-tune source tree at {args[0]}",
                    "The tree must contain src/build.sh (upstream keeps it "
                    "out of git via .dockerignore -- get a copy from that "
                    "repo's author)")
        else:
            ui.fail("No Marlin-tune source tree found",
                    "It is a local build tree with src/build.sh (upstream's "
                    ".dockerignore excludes it, so neither git repo holds "
                    "it) -- pass its path, set MARLIN_TUNE_SRC, or keep it "
                    f"at {' or '.join(str(p) for p in PROBES)}")

    versions = _versions()
    if versions is None:
        ui.fail(f"The venv has no readable vllm/torch metadata ({PY})",
                "Install the stack first:  "
                f"uv pip install --python {PY} -r {REPO / 'requirements.txt'}")
    vllm_ver, torch_ver = versions

    if rebench:
        bench = next((p for p in (src / "bench_marlin.py",
                                  src / "src" / "bench_marlin.py")
                      if p.is_file()), None)
        if bench is None:
            ui.fail(f"No bench_marlin.py in {src}",
                    "Run the table sweep by hand, then re-invoke without --rebench")
        ui.stage("Regenerating the Marlin tile table on this GPU")
        ui.note(f"{bench.name} --grid (keep the GPU idle; this takes a while)")
        os.chdir(bench.parent)  # the sweep resolves its files relative to itself
        if ui.run_indented([str(PY), bench.name, "--grid"]) != 0:
            ui.fail("The table sweep failed",
                    "See the output above; no build or install was done")

    digest = _src_digest(src)
    stamp_ok = False
    if STAMP.is_file():
        try:
            old = json.loads(STAMP.read_text())
            stamp_ok = (old.get("digest") == digest
                        and old.get("vllm") == vllm_ver
                        and old.get("torch") == torch_ver
                        and old.get("src") == str(src))
        except (json.JSONDecodeError, OSError):
            pass

    sp = _site_packages()
    if sp is None:
        ui.fail(f"Cannot locate the site-packages of {PY}")
    pth = sp / "marlin_tune.pth"
    target = str(src / "src")
    if (not pth.is_file()) or pth.read_text().strip() != target:
        pth.write_text(target + "\n")
        ui.ok(f"Linked the extension into the venv ({pth.name} -> {target})")

    import_ok, _ = _import_ok()
    if stamp_ok and import_ok:
        ui.ok(f"tuned Marlin build already in {VENV} "
              f"(vllm {vllm_ver}, torch {torch_ver})")
        return

    ui.stage(f"Building the tuned Marlin extension (vllm {vllm_ver})")
    os.chdir(src / "src")  # the build resolves its files relative to itself
    if ui.run_indented(["bash", "build.sh"]) != 0:
        ui.fail("The extension build failed (src/build.sh)",
                "The tree carries its own CUDA 13 toolchain (cuda13-home); "
                "see the output above",
                "If the tree expects a specific environment, run src/build.sh "
                "directly and re-invoke")

    import_ok, out = _import_ok()
    if not import_ok:
        tail = [l for l in out.splitlines() if l.strip()][-8:] or ["(no output)"]
        ui.fail("The built extension does not import in this venv",
                *tail,
                "Run src/build.sh directly for the full context, then re-invoke")

    STAMP.write_text(json.dumps({
        "digest": digest, "vllm": vllm_ver, "torch": torch_ver,
        "src": str(src), "built_by": "prepare/build_marlin_tune.py",
    }))
    ui.ok(f"tuned Marlin build in {VENV} "
          f"(vllm {vllm_ver}, torch {torch_ver}) -- the recipes enable "
          "VLLM_MARLIN_TUNE for it automatically; =0 keeps it off")


if __name__ == "__main__":
    main()
