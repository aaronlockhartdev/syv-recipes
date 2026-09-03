#!/usr/bin/env python3
"""Idempotently apply the repo's patch set to the vLLM in a venv.

Can be run standalone: invoked under any other interpreter it
re-execs into the venv's own (the one that can find vllm):

    ./prepare/patch_vllm.py      # or: .venv/bin/python prepare/patch_vllm.py

    VENV=/path overrides the default ./.venv

After a successful run the venv is stamped (.venv/.vllm-patch-stamp)
with the vllm version, a hash of patches/, and a hash of the tree the
patches touch. A re-run whose stamp still matches all three is a fast
no-op:

    Audited vllm 0.27.1: 19 patches in place in 0.03s

When the hash no longer matches -- you pulled or edited patches, the
venv's vllm changed, or the tree was edited after the audit -- the old
patch set cannot be reversed (its bytes are gone), so the script
reinstalls vllm pristine (uv, from its cache) and re-applies the
current set. The same reset recovers a venv interrupted mid-patch. A
venv that is already fully patched but not yet stamped (first run of
this script on an old venv) is just stamped.

How it decides: a hunk is judged by the file's content, not by patch's
own verdicts. Each hunk's old image (context plus the removed lines)
and new image (context plus the added lines) are searched for in the
target file, within 256 lines of the hunk's nominal position. A hunk is
APPLIED when only its new image is in place, FRESH when only its old
one is, CONFLICT when neither -- or both -- are. A match found farther
away than that does not count: at that distance it is not this hunk (a
repeated block elsewhere in the file), and acting on it would patch
the wrong location or bless a corrupted one. No verdict depends on GNU
patch's phrasing or on this particular patch set: any unified-diff
patches audit the same way, and every ambiguity resolves to the loud
reset, never to a silent guess. A FRESH patch additionally gets a
forward dry-run (the one place a real `patch` is consulted) before
anything is applied for real.

The states, then, tested in order on a hardlinked mirror of the tree
(only files that change get materialized):
  1. forward chain: every patch is FRESH (apply it, keep going -- this
     handles a fresh venv and a crash between patches, since a later
     patch that needs a file created by an earlier one is judged after
     the earlier one is in) or APPLIED (skip);
  2. reverse cascade: every patch is APPLIED and reverse-applies,
     walked last to first -- the exact inverse of the build, the only
     order that undoes patches that overlap in the same file (a fully
     patched tree: just stamp it, nothing is touched);
  3. neither: the tree is inconsistent -- reset (reinstall pristine,
     re-apply, stamp).

The Dockerfile builds its image with the same sequence: apply in
alphabetical order, then a compileall gate. GNU patch is required for
the actual applying (Ubuntu ships it; on macOS use Homebrew's gpatch,
which is preferred when present). A lock file in the venv keeps two
runs from patching the same venv at once.
"""

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import _ui as ui
from pathlib import Path

try:
    import fcntl as _flock
except ImportError:  # non-POSIX: skip the run lock rather than crash
    _flock = None

REPO = Path(__file__).resolve().parent.parent  # this script lives in prepare/


_SHELL_SPECIAL = frozenset(
    {"IFS", "GLOBIGNORE", "CDPATH", "BASH_ENV", "ENV", "SHELLOPTS", "PS1",
     "LINENO", "PWD", "OLDPWD", "SECONDS", "RANDOM", "UID", "EUID"})


def _dotenv():
    """Fill unset-or-empty env vars from REPO/.env, with the same contract
    the recipes use: the real environment always wins (a set, non-empty value
    is never touched), values may be quoted, whole-line # comments only, and
    a bare KEY= is skipped (empty means unset, like bash's :-)."""
    p = REPO / ".env"
    try:
        if not p.is_file():
            return
        lines = p.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if (
            not k
            or k[0].isdigit()
            or not k.replace("_", "").isalnum()
            or not k.isascii()
            or k in _SHELL_SPECIAL
        ):
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if v and not os.environ.get(k):
            os.environ[k] = v


_dotenv()
VENV = Path(os.environ.get("VENV") or REPO / ".venv")
PY = VENV / "bin" / "python"
STAMP = VENV / ".vllm-patch-stamp"

FRESH, APPLIED, CONFLICT = "fresh", "applied", "conflict"
# GNU patch is required (macOS ships a netbsd-derived one with different
# --batch/-R semantics); gpatch is Homebrew's GNU build
PATCH = shutil.which("gpatch") or "patch"


def fingerprint(patches):
    h = hashlib.sha256()
    for p in patches:
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def tree_digest(sp, patches):
    """Fingerprint of the tree state the patch set cares about: a hash
    over the union of the files the patches touch (a missing file
    hashes as empty). This makes the stamp an attestation of one
    specific tree, not just of the patch set: edit any patched file
    after an audit and the next run notices instead of reporting
    'in place'."""
    names = sorted({n for p in patches for (n, *_rest) in parse_patch(p)})
    h = hashlib.sha256()
    for name in names:
        f = sp / name
        h.update(name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes() if f.is_file() else b"").digest())
    return h.hexdigest()


def _strip_prefix(path):
    # a/ b/ worktree prefixes; the caller already cut at the tab
    for pre in ("a/", "b/"):
        if path.startswith(pre):
            path = path[len(pre):]
    return path


def _path_field(raw):
    """The path on a --- / +++ header line. A tab-separated timestamp or
    mode may follow the path; everything up to that tab is the path
    (a space inside a path is part of it -- git writes raw, unquoted
    spaces in diff headers; only C-quoted forms, which diff headers
    never use, would come back quoted and then fail loud downstream)."""
    return _strip_prefix(raw.split("\t")[0])


def _is_file_header(lines, i, n):
    # a "--- " line ends a file section only when a "+++ " line follows
    # it; a deleted line whose content starts with "-- " renders "--- "
    # in the diff but is hunk content, not a header
    return lines[i].startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ ")


def parse_patch(p):
    """The patch as [(name, is_create, is_delete, hunks)] in patch order,
    where hunks is [(pre, post, old_start, no_newline)]: pre the old-side
    image (context plus '-' lines), post the new-side image (context
    plus '+' lines), old_start the hunk's nominal line in the old file.
    Works for /dev/null and git-style headers alike."""
    sections = []
    lines = p.read_text().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            src = _path_field(lines[i][4:])
            dst = _path_field(lines[i + 1][4:])
            is_create = src == "/dev/null"
            is_delete = dst == "/dev/null"
            name = src if is_delete else dst
            hunks = []
            j = i + 2
            while j < n and not _is_file_header(lines, j, n):
                if lines[j].startswith("@@ "):
                    tok = lines[j].split()
                    old_spec = tok[1][1:] if len(tok) > 1 and tok[1].startswith("-") else "0,0"
                    old_start = int(old_spec.split(",")[0])
                    pre, post = [], []
                    no_newline = False
                    k = j + 1
                    while k < n and not lines[k].startswith("@@ ") and not _is_file_header(lines, k, n):
                        ch = lines[k][:1]
                        if ch == "-":
                            pre.append(lines[k][1:])
                        elif ch == "+":
                            post.append(lines[k][1:])
                        elif ch in (" ", ""):  # a context line (empty ones may lack the space)
                            pre.append(lines[k][1:])
                            post.append(lines[k][1:])
                        elif ch == "\\":
                            no_newline = True
                        k += 1
                    if is_create or old_start == 0:
                        is_create = True
                    hunks.append((pre, post, old_start, no_newline))
                    j = k
                    continue
                j += 1
            if name:
                sections.append((name, is_create, is_delete, hunks))
            i = j
            continue
        i += 1
    return sections


def _line_offsets(text):
    """Character offset where each line starts."""
    offs, i = [0], 0
    while True:
        i = text.find("\n", i)
        if i < 0:
            break
        offs.append(i + 1)
        i += 1
    return offs


def _find_near(text, offsets, block, nominal):
    """Character offset of block within 256 lines of the hunk's nominal
    position (the window starts tight and expands: 0, then 64, then
    256 lines). A match found only farther away does not count: at
    that distance it is not this hunk -- a repeated block elsewhere in
    the file -- and acting on it would patch the wrong location or
    bless a corrupted one. None when absent."""
    if not block:
        return None
    blines = block.count("\n") + 1
    for margin in (0, 64, 256):
        lo = max(0, nominal - 1 - margin)
        hi = min(len(offsets), nominal - 1 + blines + margin + 1)
        if lo >= len(offsets) or hi <= lo:
            continue
        end = offsets[hi] if hi < len(offsets) else len(text)
        at = text.find(block, offsets[lo], end)
        if at >= 0:
            return at
    return None


def _hunk_state(text, offsets, pre, post, nominal):
    """'fresh' | 'applied' | 'conflict' | None (no information) for one
    hunk against the file's text. Images are searched near the hunk's
    nominal line only (see _find_near). A pure-context hunk (pre ==
    post) carries no state; a context-less insertion (no pre) reduces
    to whether the added lines are in place."""
    if not pre and not post:
        return None
    pre_b = "\n".join(pre)
    post_b = "\n".join(post)
    if pre_b == post_b:
        return None  # context-only hunk: carries no state
    if not pre:
        i = _find_near(text, offsets, post_b, nominal)
        return APPLIED if i is not None else FRESH
    i_pre = _find_near(text, offsets, pre_b, nominal)
    i_post = _find_near(text, offsets, post_b, nominal)
    if i_pre is None:
        return APPLIED if i_post is not None else CONFLICT
    if i_post is None:
        return FRESH
    # both images present: the file holds the version whose image contains
    # the other (an inserted block nests its context, a removed one is
    # nested in the old block); two disjoint matches mean both versions
    # coexist in the file
    if i_pre == i_post:
        if len(pre_b) <= len(post_b):
            return APPLIED  # the old pattern is the start of the new one
        return FRESH
    pre_span = (i_pre, i_pre + len(pre_b))
    post_span = (i_post, i_post + len(post_b))
    if pre_span[0] >= post_span[0] and pre_span[1] <= post_span[1]:
        return APPLIED
    if post_span[0] >= pre_span[0] and post_span[1] <= pre_span[1]:
        return FRESH
    return CONFLICT


def _same_content(actual, expected):
    # trailing-newline differences are not state differences
    return actual.rstrip("\n") == expected.rstrip("\n")


def classify(p, sp):
    """One patch vs the tree, by content: FRESH (every hunk shows only
    its old image), APPLIED (every hunk shows only its new one), or
    CONFLICT (anything else -- a missing file, both images coexisting,
    a hand-edited tree, a vllm from another release). A file whose
    hunks are all pure context verifies nothing and counts as
    APPLIED."""
    sections = parse_patch(p)
    if not sections:
        return CONFLICT
    states = []
    for name, is_create, is_delete, hunks in sections:
        target = sp / name
        if not target.is_file():
            states.append(FRESH if is_create else (APPLIED if is_delete else CONFLICT))
            continue
        try:
            text = target.read_text()
        except (OSError, UnicodeDecodeError):
            states.append(CONFLICT)
            continue
        if is_create:
            expected = "\n".join(h for _, post, _, _ in hunks for h in post)
            states.append(APPLIED if _same_content(text, expected) else CONFLICT)
            continue
        offsets = _line_offsets(text)
        hstates = [h for h in (_hunk_state(text, offsets, pre, post, start)
                               for pre, post, start, _ in hunks) if h is not None]
        if not hstates:
            states.append(APPLIED)  # nothing informative to verify
        elif CONFLICT in hstates:
            states.append(CONFLICT)
        elif FRESH in hstates and APPLIED in hstates:
            states.append(CONFLICT)  # some hunks in, some out: interrupted mid-patch
        elif APPLIED in hstates:
            states.append(APPLIED)
        else:
            states.append(FRESH)
    if CONFLICT in states:
        return CONFLICT
    if FRESH in states and APPLIED in states:
        return CONFLICT  # some files of the patch in, some out: mid-patch crash
    return APPLIED if APPLIED in states else FRESH


def dryrun(p, sp):
    try:
        with open(p, "rb") as fh:
            return subprocess.run([PATCH, "-p1", "-d", str(sp), "--batch", "--dry-run"],
                                  stdin=fh, capture_output=True, text=True)
    except FileNotFoundError:
        ui.fail("The patch tool is not on PATH", "install it:  apt install patch")


def apply_real(p, sp, reverse=False):
    """One real apply, captured: the script's own lines report progress,
    not patch's. The combined output is the failure evidence."""
    cmd = [PATCH, "-p1", "-d", str(sp), "--batch"]
    if reverse:
        cmd.append("-R")
    try:
        with open(p, "rb") as fh:
            return subprocess.run(cmd, stdin=fh, capture_output=True, text=True)
    except FileNotFoundError:
        ui.fail("The patch tool is not on PATH", "install it:  apt install patch")


def _output_tail(r):
    """The last lines of a failed patch run -- the hunk evidence."""
    out = (r.stdout or "") + (r.stderr or "")
    return [l.strip() for l in out.splitlines() if l.strip()][-6:]


def make_mirror(sp):
    """Hardlinked mirror of sp (only files that change get materialized).
    Returns (mirror, root) -- clean up with rmtree(root)."""
    root = Path(tempfile.mkdtemp(prefix="patch-vllm-mirror-"))
    try:
        shutil.copytree(sp, root / sp.name, symlinks=True, copy_function=_copy_link)
    except OSError as e:
        shutil.rmtree(root, ignore_errors=True)
        ui.fail("Could not build a work copy of the vllm tree",
                f"{e} -- free space in {tempfile.gettempdir()} (or point TMPDIR elsewhere) and re-run")
    return root / sp.name, root


def clean_junk(sp):
    """Remove .rej/.orig byproducts. .rej appears whenever a hunk is
    rejected; .orig only with -b (unused here) or left behind by older
    patch invocations -- both are swept either way."""
    for junk in list(sp.rglob("*.rej")) + list(sp.rglob("*.orig")):
        junk.unlink(missing_ok=True)


def _copy_link(a, b):
    try:
        os.link(a, b)
    except OSError:  # cross-device: fall back to a real copy
        shutil.copy2(a, b)


def forward_chain(patches, sp, apply_to=None, tick=None, failures=None):
    """Classify in build order. With apply_to, FRESH patches are applied
    to it as the pass goes, so a patch that only applies after an earlier
    patch in the set is judged against the state it will actually see.
    Returns (verdicts, ok): ok when the chain completes -- every patch FRESH
    (and applied, if apply_to) or APPLIED (skipped), with no CONFLICT.
    failures collects the hunk evidence of each CONFLICT."""
    verdicts = {}
    ok = True
    for p in patches:
        v = classify(p, sp)
        if tick:
            tick()
        if v == FRESH:
            # the content test says the hunk is absent; the dry-run says
            # the patch machinery will actually take it
            r = dryrun(p, sp)
            if r.returncode != 0:
                v = CONFLICT
                if failures is not None:
                    failures[p.name] = _output_tail(r)
        if v == FRESH and apply_to is not None:
            r = apply_real(p, apply_to)
            if r.returncode != 0:
                v = CONFLICT
                if failures is not None:
                    failures[p.name] = _output_tail(r)
        if v == CONFLICT:
            ok = False
        verdicts[p.name] = v
    return verdicts, ok


def reverse_cascade(patches, sp, tick=None):
    """True when the tree is exactly pristine + every patch applied in
    build order: the only order that undoes patches overlapping in the
    same file is the exact inverse of the build, so walk last to first.
    Runs on a mirror; the tree is never touched."""
    m, root = make_mirror(sp)
    try:
        for p in reversed(patches):
            if classify(p, m) != APPLIED:
                return False
            if apply_real(p, m, reverse=True).returncode != 0:
                return False
            if tick:
                tick()
        return True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def reinstall(version):
    if version == "unknown":
        ui.fail("The vllm version is not readable, so it cannot be reinstalled",
                f"Recreate the venv:  rm -rf {VENV} && uv venv {VENV} --python 3.12 && uv pip install --python {PY} -r {REPO / 'requirements.txt'}")
    if shutil.which("uv") is None:
        ui.fail("The uv tool is not on PATH (needed to reinstall vllm)",
                "one time:  curl -LsSf https://astral.sh/uv/install.sh | sh")
    ui.note(f"Reinstalling vllm {version} pristine (uv, from its cache)")
    # uv resolves the pin against its index (PyPI by default): this venv
    # was created from requirements.txt (by setup.py or the Dockerfile),
    # so that is the same distribution the venv already holds
    pin = f"vllm=={version.split('+')[0]}"  # drop the local tag: uv re-resolves the platform wheel
    rc = ui.run_indented(["uv", "pip", "install", "--python", str(PY),
                          "--force-reinstall", "--no-deps", pin])
    if rc != 0:
        ui.fail(f"The uv reinstall of vllm=={version} failed",
                f"The venv may now hold a broken vllm -- recreate it:  rm -rf {VENV} && uv venv {VENV} --python 3.12 && uv pip install --python {PY} -r {REPO / 'requirements.txt'}",
                f"Re-run:  {PY} {REPO / 'prepare' / 'patch_vllm.py'}")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # self-bootstrap: a bare ./prepare/patch_vllm.py arrives via the shebang's
    # generic interpreter; re-exec under the venv's own python (the one
    # with vllm installed). sys.prefix equals the venv's only when the
    # process was actually started through it. Bounded to one hop by the
    # env marker, so a VENV that is not a real venv cannot loop
    if (
        os.environ.get("VENV_PV_REEXEC") != "1"
        and PY.is_file()
        and os.path.realpath(sys.prefix) != os.path.realpath(VENV)
    ):
        os.environ["VENV_PV_REEXEC"] = "1"
        try:
            os.execv(str(PY), [str(PY), os.path.abspath(__file__)] + sys.argv[1:])
        except OSError as e:
            ui.fail(f"cannot exec the venv's python at {PY}",
                    f"it exists but {e.strerror or 'the exec failed'} -- check its permissions, or recreate the venv")

    t0 = time.monotonic()

    if not PY.is_file():
        ui.fail(f"No python at {PY}",
                f"create the venv first:  uv venv {VENV} --python 3.12",
                "or point VENV at an existing one")

    # one run per venv: a concurrent run (or a human with patch) racing
    # the final applies would otherwise let --batch auto-reverse our
    # work, or get it applied twice
    try:
        lock = open(VENV / ".vllm-patch-lock", "a")
    except OSError:
        lock = None
        ui.note(f"Could not create {VENV / '.vllm-patch-lock'} -- running without the run lock")
    if _flock is not None and lock is not None:
        try:
            _flock.flock(lock, _flock.LOCK_EX | _flock.LOCK_NB)
        except OSError:
            ui.fail("Another patch_vllm run is in progress on this venv",
                    "wait for it to finish, then re-run")

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        ui.fail(f"The vllm package is not installed in {VENV}",
                f"install it first:  uv pip install --python {PY} -r {REPO / 'requirements.txt'}")
    sp = Path(spec.submodule_search_locations[0])
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    version = version or "unknown"  # None: the dist exists but its metadata lacks a version
    ui.note(f"Found vllm {version} at {sp}")

    patches = sorted((REPO / "patches").glob("*.patch"))
    if not patches:
        ui.fail(f"No .patch files in {REPO / 'patches'}", "the repo looks incomplete")
    fp = fingerprint(patches)
    current = f"{version} {fp} {tree_digest(sp, patches)}"

    old = None
    try:
        old = STAMP.read_text().strip()
    except OSError:
        pass
    if old == current:
        ui.done(f"Audited vllm {version}: {len(patches)} patches in place in {ui.dur(time.monotonic() - t0)}")
        return

    # --- decide what state the tree is in --------------------------------
    ui.stage("Auditing the tree")
    reset = False
    verdicts = None
    parts = old.split() if old else []
    old_v = parts[0] if len(parts) >= 1 else None
    old_fp = parts[1] if len(parts) >= 2 else None
    old_tree = parts[2] if len(parts) >= 3 else None
    if old and len(parts) not in (2, 3):
        # a corrupted or foreign stamp says nothing we can act on:
        # re-audit the tree instead of resetting on its word
        ui.note(f"The stamp is unreadable ({old[:40]!r} -- this script writes 3 fields) -- re-auditing")
        old_v = old_fp = old_tree = None
    if old_v is not None and (old_v != version or old_fp != fp):
        why = []
        if old_v != version:
            why.append(f"vllm {old_v} -> {version}")
        if old_fp != fp:
            why.append("the patch set changed")
        ui.note(f"The stamp no longer matches ({', '.join(why) or 'unknown reason'})")
        reset = True
    elif old_v is not None and old_tree is None:
        ui.note("The stamp has no tree hash (an older version wrote it) -- re-auditing")
    elif old_v is not None:
        ui.note("The vllm tree changed since the last audit -- re-auditing")
    if not reset:
        p = ui.Progress(f"Auditing {len(patches)} patches", total=2 * len(patches))
        mirror, mroot = make_mirror(sp)
        try:
            verdicts, forward_ok = forward_chain(patches, mirror, apply_to=mirror, tick=p.tick)
        finally:
            shutil.rmtree(mroot, ignore_errors=True)
        if forward_ok:
            # fresh tree (or a crash between patches): apply the FRESH set below
            n_fresh = sum(1 for q in patches if verdicts[q.name] == FRESH)
            p.finish(True, f"Found {n_fresh} fresh, {len(patches) - n_fresh} already applied")
        elif reverse_cascade(patches, sp, tick=p.tick):
            # fully patched: the per-patch verdicts above are unreliable
            # (overlapping patches cannot be judged individually against
            # the final state) -- the cascade proves the whole set is in
            verdicts = {q.name: APPLIED for q in patches}
            p.finish(True, f"Found all {len(patches)} patches already applied")
        else:
            p.finish(False, "The tree is neither fully patched nor cleanly forward-applicable (interrupted mid-patch, or hand-edited)")
            reset = True

    if reset:
        ui.stage(f"Resetting vllm {version}")
        t_r = time.monotonic()
        reinstall(version)
        clean_junk(sp)
        ui.ok(f"Pristine vllm {version} in place in {ui.dur(time.monotonic() - t_r)}")
        p = ui.Progress(f"Verifying {len(patches)} patches against the pristine tree", total=len(patches))
        failures = {}
        mirror, mroot = make_mirror(sp)  # mirror the freshly reinstalled tree
        try:
            verdicts, forward_ok = forward_chain(patches, mirror, apply_to=mirror, tick=p.tick, failures=failures)
        finally:
            shutil.rmtree(mroot, ignore_errors=True)
        if not forward_ok:
            bad = next((q for q in patches if verdicts[q.name] == CONFLICT), patches[-1])
            p.finish(False, f"{bad.stem} fails even against pristine vllm {version}",
                     *failures.get(bad.name, []),
                     "The patch does not match this vllm release -- rebase it against the pinned one",
                     "The venv now holds pristine, unpatched vllm: it will not serve until that is fixed",
                     f"Re-run:  {PY} {REPO / 'prepare' / 'patch_vllm.py'}",
                     fatal=True)

    # --- apply (or confirm) ---------------------------------------------
    n_fresh = sum(1 for q in patches if verdicts[q.name] == FRESH)
    if n_fresh:
        ui.stage(f"Applying {n_fresh} of {len(patches)} patches")
    n_applied = 0
    for q in patches:
        v = verdicts[q.name]
        if v == APPLIED:
            ui.note(f"{q.stem} (already applied)")
            continue
        if v != FRESH:
            ui.fail(f"{q.stem} could not be applied (no clean direction found)",
                    "re-run, or recreate the venv if it keeps happening")
        # the verdicts came from a mirror; confirm against the real tree
        # so a change made since the audit (by a second run, a human, or
        # another tool) cannot be written over -- the next run converges
        if classify(q, sp) != FRESH:
            ui.fail(f"{q.stem}: the tree changed under us since the audit",
                    "re-run -- it converges from the current state")
        r = apply_real(q, sp)
        if r.returncode != 0:
            ui.fail(f"{q.stem} failed on the real apply after a clean dry-run",
                    *_output_tail(r),
                    "the tree changed under us -- re-run",
                    f"or recreate the venv:  rm -rf {VENV} && uv venv {VENV} --python 3.12 && uv pip install --python {PY} -r {REPO / 'requirements.txt'}")
        ui.ok(q.stem)
        n_applied += 1

    clean_junk(sp)
    p = ui.Progress("Compiling the patched tree")
    t_c = time.monotonic()
    # captured, not streamed: the spinner owns the live line, and the
    # output is only wanted as evidence on failure
    r = subprocess.run([str(PY), "-m", "compileall", "-q", str(sp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = [l.rstrip() for l in ((r.stdout or "") + (r.stderr or "")).splitlines()
                if l.strip()][-8:] or ["(no output from compileall)"]
        p.finish(False, "The patched tree does not compile -- vLLM is broken, do not run it",
                 *tail,
                 f"Reset vllm and re-run:  uv pip install --python {PY} --force-reinstall --no-deps vllm=={version.split('+')[0]} && {PY} {REPO / 'prepare' / 'patch_vllm.py'}",
                 f"Or recreate the venv:  rm -rf {VENV} && uv venv {VENV} --python 3.12 && uv pip install --python {PY} -r {REPO / 'requirements.txt'}",
                 fatal=True)
    ui.ok(f"Compiled the patched tree in {ui.dur(time.monotonic() - t_c)}")

    # stamp the state as it is now, not the start-of-run digest (which
    # predates this run's own applies on a fresh venv)
    stamp = f"{version} {fp} {tree_digest(sp, patches)}"
    try:
        STAMP.write_text(stamp + "\n")
    except OSError as e:
        # the tree is patched and compiled; only the attestation is
        # missing, so the next run re-audits instead of failing
        ui.fail(f"Could not write {STAMP} ({e})",
               "The venv is patched and compiles; fix the venv permissions and re-run to stamp it")
    elapsed = ui.dur(time.monotonic() - t0)
    if reset:
        ui.done(f"Reinstalled vllm {version}; applied {n_applied} of {len(patches)} patches in {elapsed}")
    elif n_applied:
        ui.done(f"Applied {n_applied} patches in {elapsed}")
    else:
        ui.done(f"Audited vllm {version}: {len(patches)} patches in place in {elapsed}")


if __name__ == "__main__":
    main()
