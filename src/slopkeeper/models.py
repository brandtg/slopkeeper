# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Shared plumbing: JSON envelope, typed errors, path safety, fingerprinting.

Design rule: the tool NEVER talks to an LLM and NEVER touches the network.
The calling agent is the brain; this tool is the hands and the seatbelt.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from . import VERSION

TOOL = "slopkeeper"
PLAN_SCHEMA = "slopkeeper/plan@1"

# The current fingerprint walk contract. Bump when the walk set changes:
# the prefix version-separates stored manifests/plans from the current
# rules (see corpus_fingerprint and status's stale-contract hint).
# fp5: non-regular files (devices, FIFOs, sockets — sandbox deny-list
# mounts) are pruned from the walk, so a sandboxed and an unsandboxed
# run of the same corpus now fingerprint identically.
CURRENT_FP_PREFIX = "fp5"

# The tool/infra layer — directories that are never content. Dot-dirs are
# CONTENT as of fp4: `.claude/skills`, `.hermes` memories and cron prompts
# are the files most likely to cite corpus paths, and the walk must see
# them. Per-corpus noise (`.cache`, `.claude/projects` transcript dumps)
# is the `.slopkeeperignore` file's job, with doctor suggesting the lines
# (DESIGN.md §6.6). Legacy state dirs (.janitor, .scaffolder) stay listed:
# pre-launch corpora carrying them must never have them walked as content.
HIDDEN_SKIP_DIRS = {
    ".git",
    ".slopkeeper",
    ".janitor",
    ".scaffolder",
    ".venv",
    "node_modules",
    "__pycache__",
    ".obsidian",
    ".DS_Store",
}

# The per-corpus state dir (journals, manifest, vectors, tend.lock).
STATE_DIR = ".slopkeeper"

# The manifest filename inside STATE_DIR. Canonical here so the walk
# contract (ignore.py) can recognize a nested corpus — a subtree with
# its own .slopkeeper/manifest.json — without importing status.py
# (which imports from here).
MANIFEST_NAME = "manifest.json"

MANIFEST_SCHEMA = "slopkeeper/manifest@2"


class SlopKeeperError(Exception):
    """An error with a machine-readable code and a recovery hint for the agent."""

    def __init__(self, code: str, message: str, recovery: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "recovery": self.recovery}


# --- envelope ---------------------------------------------------------------


def envelope(command: str, data=None, next_action: dict | None = None) -> dict:
    out = {
        "tool": TOOL,
        "version": VERSION,
        "command": command,
        "ok": True,
        "data": data,
    }
    if next_action:
        out["next"] = next_action
    return out


def next_step(action: str, command: str, hint: str) -> dict:
    """Tell the calling agent what to do next. Every success path sets this."""
    return {"action": action, "command": command, "hint": hint}


def emit_json(obj: dict) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


# --- path safety ------------------------------------------------------------


def safe_relpath(raw: str | Path, root: Path) -> Path:
    """Normalize `raw` to a relative path strictly inside `root`.

    Accepts absolute paths under root, rejects traversal outside root,
    null bytes, and paths that normalize to the corpus root itself.
    '~' and backslashes are legal POSIX filename characters and are
    accepted literally (no expanduser, no Windows separator semantics).
    Returns a relative Path (no leading '/'), not yet resolved to disk.
    """
    raw_str = str(raw)
    if "\0" in raw_str:
        raise SlopKeeperError(
            "path_escape",
            "null byte in path",
            "Strip the NUL byte; it cannot appear in a filename.",
        )
    p = Path(raw_str)
    root_res = root.resolve()
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(root_res)
        except ValueError:
            raise SlopKeeperError(
                "path_escape",
                f"path {raw} is outside corpus root {root_res}",
                "Use paths relative to the corpus root.",
            )
    elif ".." in p.parts:
        raise SlopKeeperError(
            "path_escape",
            f"path {raw} escapes the corpus root",
            "Use paths relative to the corpus root.",
        )
    if not p.parts:
        # '.' or '' — the corpus root itself is not a movable file.
        raise SlopKeeperError(
            "path_escape",
            f"path {raw!r} is the corpus root itself",
            "Name a file inside the corpus root.",
        )
    # Absolute-form containment check for symlink shenanigans.
    abs_form = root_res / p
    if abs_form.is_symlink() and not abs_form.resolve().is_relative_to(root_res):
        raise SlopKeeperError(
            "symlink_escape",
            f"path {raw} resolves outside the corpus root via symlink",
            "Remove the symlink or move its target explicitly.",
        )
    return p


def corpus_path(root: Path, rel: Path) -> Path:
    return root / rel


# --- integrity --------------------------------------------------------------


def sha256_file(path: Path, cap_bytes: int = 64 * 1024 * 1024) -> str | None:
    try:
        st = path.stat()
        if st.st_size > cap_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_file_exact(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """sha256 of a file's EXACT bytes, streamed — no size cap (DESIGN §5.6).

    sha256_file's 64 MiB cap is a corpus-scan courtesy; it must never
    turn a plan's provenance hash into None. Provenance reuses the same
    streaming pattern minus the cap, so the recorded hash is bound to
    the plan file's real bytes however large they are.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_fingerprint(root: Path) -> str:
    """Stable digest of content per non-skipped file — staleness is
    content-based (DESIGN.md §5.2, §6.6).

    Each entry hashes (relpath, size, sha256) with the content hash
    from sha256_file, streamed in chunks — files are never slurped
    whole. Files past sha256_file's 64 MiB cap (or unreadable ones)
    yield None and contribute size+mtime only — one limit, reused, so
    the fingerprint, manifest and journal agree on what "hashable"
    means; mtime stays their identity because there is no hash.

    The walk is corpus_walk (HIDDEN_SKIP_DIRS + the `.slopkeeperignore`
    set — dot-directories are CONTENT as of fp4, only skip-list members
    and ignored paths are invisible): ignored paths are invisible to the
    gate. The ignore file itself is never walked either — it is tool
    layer, not content — and is not hashed: only the effective path set
    counts, so a comment-only ignore edit never moves the fingerprint.

    Used as a staleness gate: a plan computed against one fingerprint is
    refused against a changed corpus unless --force. The `fp4:` prefix
    version-separates the format (fp4 = the walk without the blanket
    dot-dir prune — dot-dirs are content): plans authored against
    fp1/fp2/fp3 fail cleanly with `corpus_changed` instead of
    colliding. A bare `touch` (new mtime, same content) does NOT change
    a fingerprint; a real content edit does.
    """
    # Lazy import: ignore.py imports HIDDEN_SKIP_DIRS from this module,
    # so a top-level import here would be a circular-import trap.
    from .ignore import corpus_walk

    h = hashlib.sha256()
    count = 0
    for dirpath, _dirnames, filenames in corpus_walk(root):
        dirpath = Path(dirpath)
        for fn in filenames:
            fp = dirpath / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            digest = sha256_file(fp)
            h.update(str(fp.relative_to(root)).encode("utf-8", "replace"))
            h.update(b"\0")
            h.update(str(st.st_size).encode())
            h.update(b":")
            if digest is not None:
                h.update(digest.encode())
            else:
                h.update(str(st.st_mtime_ns).encode())
            h.update(b"\0")
            count += 1
    return f"{CURRENT_FP_PREFIX}:{h.hexdigest()[:32]}:{count}"


# --- state dir ---------------------------------------------------------------


def state_dir(root: Path) -> Path:
    """Resolve the corpus state dir (`.slopkeeper/`)."""
    return Path(root) / STATE_DIR


# --- tending lock (DESIGN.md §5.5) ------------------------------------------

TEND_LOCK_NAME = "tend.lock"


@contextmanager
def tend_lock(root: Path):
    """Advisory exclusive flock on `.slopkeeper/tend.lock`.

    Serializes mutating tends (apply/scaffold/undo --execute and
    status --record) across concurrent agents: journal appends, moves/
    edits and manifest writes cannot interleave. Non-blocking by
    design — a second writer fails immediately with `tending_in_progress`
    instead of queueing, because an agent must decide (wait for the
    other tend, or check for a dead agent's session), never hang.

    Platform guard: fcntl is stdlib on POSIX only. On platforms without
    it (Windows) the attempt fails SOFT — we proceed WITHOUT the lock
    rather than refuse the tend (zero-dep: no msvcrt pip fallbacks; the
    corpus tool is Unix-first, and the single-writer case is the common
    one).

    The OS releases the lock on process death, so there is no stale-
    lock logic and no cleanup ritual: the lock file may remain, an
    empty dotfile inside .slopkeeper/ that every walk and the
    fingerprint already skip.

    A fresh fd is opened per acquisition, so sequential tends (acquire
    → release → acquire) never self-deadlock. flock conflicts are per
    open-file-description, so callers must acquire exactly once around
    the whole critical span — nesting (e.g. locking again inside
    write_manifest when apply already holds it) would conflict with
    itself.
    """
    try:
        import fcntl  # POSIX-only; imported here so Windows never sees it
    except ImportError:
        # No fcntl (Windows): fail soft per §5.5 — proceed unlocked.
        # Covered on POSIX via the None-in-sys.modules import blocker.
        yield
        return
    lock_path = state_dir(root) / TEND_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise SlopKeeperError(
                "tending_in_progress",
                f"another tend is running (lock: {lock_path}): {e}",
                "another tend is running — wait for it or check for a "
                "dead agent's session",
            ) from None
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass  # already released (process death does it too)
        os.close(fd)


# --- plan chain resolution --------------------------------------------------


def resolve_chains(moves: list[dict]) -> dict[str, str]:
    """Resolve move chains to src -> final-dst pairs, LOSSLESSLY.

    File→file chains keep their hops: a→b, b→c resolves to {a: b, b: c}.
    Rewriting a→c made apply() move two files onto one path and silently
    destroy one; the executor orders the hops instead (b→c before a→b).
    A chain is followed through an intermediate only when that
    intermediate is a DIRECTORY move: a file whose source lives under a
    moved directory follows the renamed parent (proj→archive/proj turns
    proj/notes.md→proj/renamed.md into archive/proj/renamed.md), and
    dir→dir hops still collapse. Raises move_cycle on any cycle.
    Returns {src_rel_str: final_dst_rel_str}.
    """
    direct = {m["from"]: m["to"] for m in moves}
    # Sorted src set, built ONCE: dir-key detection asks "does any src
    # live under this prefix?" per move, and a rule-expanded plan can
    # carry tens of thousands of moves — a linear scan per move is
    # O(n²) there, a bisect is O(n log n) overall. Same answer.
    srcs_sorted = sorted(direct)

    # Cycle detection over the full hop graph, file and dir hops alike.
    for start in direct:
        cur = start
        seen = {start}
        while cur in direct:
            cur = direct[cur]
            if cur in seen:
                raise SlopKeeperError(
                    "move_cycle",
                    f"move cycle involving {start}",
                    "Remove the circular moves from the plan.",
                )
            seen.add(cur)

    def is_dir_key(src: str) -> bool:
        return src.endswith("/") or _looks_like_dir_key(src, srcs_sorted)

    dir_moves = []
    file_moves = []
    for src, dst in direct.items():
        if is_dir_key(src):
            dir_moves.append((src, dst))
        else:
            file_moves.append((src, dst))

    def follow_dir_hops(src: str) -> str:
        """Walk src's chain, hopping on only through DIRECTORY-move
        intermediates; a plain file→file hop ends the walk (the pair is
        kept as-is)."""
        cur = direct[src]
        seen = {src, cur}
        while cur in direct:
            nxt = direct[cur]
            if not is_dir_key(nxt) or nxt in seen:
                break
            cur = nxt
            seen.add(cur)
        return cur

    resolved: dict[str, str] = {}
    for src, _ in file_moves:
        # Keep the file's own hop as-is, then rewrite through any moved
        # ancestor directory (the rename survives the dir move).
        final = follow_dir_hops(src)
        for dsrc, ddst in dir_moves:
            dsrc_n = dsrc.rstrip("/") + "/"
            if src.startswith(dsrc_n) and dsrc != src:
                final = ddst.rstrip("/") + "/" + final[len(dsrc_n) :]
                break
        resolved[src] = final
    for dsrc, _ in dir_moves:
        resolved[dsrc] = follow_dir_hops(dsrc)
    return resolved


def _looks_like_dir_key(src: str, srcs_sorted: list[str]) -> bool:
    """A src is treated as a directory if any other move's src lives under it.

    `srcs_sorted` is the plan's move sources, sorted once by the caller
    (resolve_chains): the first entry >= `<src>/` is under the prefix iff
    it starts with it — a bisect answers in O(log n) instead of scanning
    every move per src (a 61k-move rule expansion would otherwise crawl).
    """
    prefix = src.rstrip("/") + "/"
    i = bisect.bisect_left(srcs_sorted, prefix)
    return i < len(srcs_sorted) and srcs_sorted[i].startswith(prefix)
