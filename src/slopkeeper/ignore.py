# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`.slopkeeperignore` — the per-corpus invisible set.

One rule everywhere: an ignored path is a path the tool pretends does
not exist. It is not inventoried, not fingerprinted, not in new_files,
not draftable, invisible to reference scans, and refused as a move
source or edit target at validate (which apply re-runs). Ignored
directories remain legal move DESTINATIONS — "keep filing into
archive/, never reorganize it" is the common shape.

The file lives at the corpus ROOT, dot-prefixed like the other tool
layer: the fingerprint hashes every file outside the tool-layer skip
list and the ignored set — dot-DIRECTORIES are corpus content since
fp4 — so the config that shapes the fingerprint is itself
fingerprinted, and plan coverage already excludes root dot-entries.

Syntax: a small gitignore subset. `#` comments, blank lines skipped.
A trailing `/` means directory-only (the whole subtree, at any depth).
A pattern with an interior or leading slash is anchored to the corpus
root; otherwise it matches basenames at any depth (`*.log` works
everywhere). `*`/`?` do not cross `/`; a `**` SEGMENT spans ZERO OR MORE
directories (gitignore's reading: `a/**/b` matches `a/b` as well as
`a/x/b` — 0.2.7 fixed the compile that required an intervening
directory, which silently left `a/b` visible to a plan). Not in v1:
negation, nested ignore files, a global ignore — hit a wall first.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath

# Re-exported: these are the walk contract's constants and `corpus_walk`
# documents them, so `from .ignore import HIDDEN_SKIP_DIRS` (doctor,
# references) keeps working even though the traversal body now lives in
# scan.py. `os` left with it — the walk is not this module's job twice.
from .models import (  # noqa: F401
    HIDDEN_SKIP_DIRS,
    MANIFEST_NAME,
    STATE_DIR,
)

IGNORE_NAME = ".slopkeeperignore"


def glob_to_regex(pattern: str) -> re.Pattern:
    """A `/`-aware glob → anchored regex.

    `*`/`?` stay within one path segment; a whole-segment `**` spans
    directories. This is the ONE glob grammar — CorpusIgnore's anchored
    rules and plan `rules[].match` (Phase 8) share it, so a pattern means
    the same thing in an ignore file and in a plan rule. (stdlib
    fnmatch.translate is NOT `-`-aware: its `*` crosses `/`.)

    A `**` SEGMENT matches ZERO or more directories, as gitignore says
    (audit A6): `a/**/b` matches `a/b` as well as `a/x/b`. The pre-0.2.7
    compile emitted `a/.*/b`, which required an intervening directory —
    so a rule written believing gitignore semantics silently failed to
    ignore `a/b`, and SlopKeeper planned to organize files the owner
    meant excluded. A trailing `**` (`a/**`) still means everything
    BELOW a, never `a` itself.
    """
    rx = ""
    # True when the last emitted piece carries its own trailing
    # separator, so the next segment must NOT add another `/`.
    joined = False
    segs = pattern.split("/")
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg == "**" and not last:
            rx += ("/" if rx and not joined else "") + "(?:[^/]*/)*"
            joined = True
            continue
        if rx and not joined:
            rx += "/"
        if seg == "**":  # trailing: everything below, never the dir itself
            rx += ".*"
        else:
            rx += "".join(
                "[^/]*" if ch == "*" else "[^/]" if ch == "?" else re.escape(ch)
                for ch in seg
            )
        joined = False
    return re.compile("^" + rx + "$")


class CorpusIgnore:
    """Compiled patterns from one `.slopkeeperignore` file."""

    def __init__(self, patterns: list[str]):
        # Raw lines are kept only for reporting (doctor's rule count).
        self.patterns = list(patterns)
        self.rule_count = 0
        # (fnmatch basename pattern, dir_only) — matches at any depth.
        self._basename_rules: list[tuple[str, bool]] = []
        # (compiled full-path regex, dir_only) — anchored to the root.
        self._anchored_rules: list[tuple[re.Pattern, bool]] = []
        for raw in self.patterns:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            dir_only = line.endswith("/")
            line = line.rstrip("/")
            if line.startswith("/"):
                anchored = True
                line = line.lstrip("/")
            else:
                anchored = "/" in line
            if not line:
                continue
            self.rule_count += 1
            if anchored:
                self._anchored_rules.append((self._compile(line), dir_only))
            else:
                self._basename_rules.append((line, dir_only))

    @staticmethod
    def _compile(pattern: str) -> re.Pattern:
        """Anchored pattern → full-path regex (the shared glob grammar)."""
        return glob_to_regex(pattern)

    def _match_one(self, rel: str, is_dir: bool) -> bool:
        name = rel.rsplit("/", 1)[-1]
        for pat, dir_only in self._basename_rules:
            if dir_only and not is_dir:
                # A dir-only pattern still ignores files INSIDE the dir —
                # that is is_ignored's ancestor walk, not this check.
                continue
            if fnmatch.fnmatchcase(name, pat):
                return True
        for rx, dir_only in self._anchored_rules:
            if dir_only and not is_dir:
                continue
            if rx.match(rel):
                return True
        return False

    def is_ignored(self, rel: str, is_dir: bool = False) -> bool:
        """True when `rel` (posix, root-relative) is ignored — directly,
        or because an ancestor directory is."""
        rel = rel.strip("/")
        if not rel:
            return False
        parts = PurePosixPath(rel).parts
        for i in range(1, len(parts)):
            if self._match_one("/".join(parts[:i]), True):
                return True
        return self._match_one(rel, is_dir)

    def active(self) -> bool:
        return self.rule_count > 0


def load_ignore(root: Path) -> CorpusIgnore:
    """Read `<root>/.slopkeeperignore` fresh per command. Missing or
    unreadable file = empty ignore = the tool sees everything (the safe
    default), never a crash."""
    try:
        text = (Path(root) / IGNORE_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return CorpusIgnore([])
    return CorpusIgnore(text.splitlines())


def corpus_walk(root: Path, nested_out: list[str] | None = None):
    """os.walk over the corpus with the full skip contract applied.

    Yields (dirpath, dirnames, filenames): dirnames are pruned and
    sorted (HIDDEN_SKIP_DIRS and ignored — ignored subtrees are never
    descended); filenames are pruned (ignored) and sorted. Dot-dirs are
    CONTENT as of fp4 — `.claude/skills` and friends are walked like any
    other directory; only skip-list members and ignored paths are
    invisible. Root-level dot-FILES are walked too (they always were —
    the fingerprint hashes them); plan coverage is the only place that
    treats them specially. This is the ONE walk every corpus-side
    command shares — the six copy-pasted dirnames filters it replaces
    could not have grown the ignore rules without drifting apart.
    Expects an absolute root, as every caller resolves before walking.

    Since 0.2.8 the traversal itself lives in `scan.walk_visits` (the
    single pass every read surface folds from) and this is the
    names-only projection over it: one implementation of the contract,
    two granularities. `want_*` flags there are what an item in this
    tuple can never express, which is why the fingerprint and inventory
    consume visits directly rather than names.

    Nested corpora (reference-rewrites PRD Phase 7): any directory
    below the root carrying its own `.slopkeeper/manifest.json` is a
    separate tended corpus — pruned from the walk exactly like a
    skip-list dir (its files are invisible to the parent's inventory,
    status, fingerprint and rewrites), but NAMED: when `nested_out` is
    a list, the subtree's root-relative name is appended to it. Pass a
    list (or use nested_corpora()) to surface what was skipped — never
    a silently empty diff.

    Non-regular files (fp5): character devices, FIFOs and sockets are
    not files — sandbox deny-list mounts show up as `/dev/null`-style
    char devices that would otherwise be inventoried, fingerprinted
    and decode-failed (the field report's 639-vs-630 count and the
    `not_utf8` row for a `loop.md` device). They are pruned here so
    every consumer — inventory, fingerprint, reference scans, rewrites
    — sees the same regular-file set; doctor names what was pruned.

    Symlinked directories (fp6, audit A2): reported in `dirnames` but
    NEVER descended (os.walk does not follow links) — their contents
    stay invisible to every consumer. They are still ENTRIES: the
    fingerprint digests name+target and inventory names them in
    `stats.symlinked_dirs`, because plan coverage validation counts
    top-level entries via iterdir and always has. Moves through or onto
    them are refused by default (models.safe_move_path); --allow-symlinks
    is the deliberate bypass.
    """
    # Lazy import: scan.py imports this module (the contract's rules and
    # helpers live here), so a top-level import would close the loop.
    from .scan import WalkReport, walk_visits

    report = WalkReport(nested=nested_out)
    for visit in walk_visits(root, report=report):
        yield str(visit.path), list(visit.subdirs), [e.name for e in visit.files]


def nested_corpora(root: Path) -> list[str]:
    """Root-relative names of nested corpora under `root`, sorted.

    One shared walk (corpus_walk with a collector) fills the list as it
    prunes — the same rule everywhere, no second implementation to
    drift. Names are what status/inventory/doctor report; the subtrees
    themselves stay invisible to the parent exactly like a skip-list
    dir.
    """
    out: list[str] = []
    for _dirpath, _dirnames, _filenames in corpus_walk(root, nested_out=out):
        pass  # the walk fills `out` as it prunes
    return sorted(set(out))
