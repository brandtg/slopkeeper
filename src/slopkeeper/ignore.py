# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`.slopkeeperignore` — the per-corpus invisible set (DESIGN.md §6.6).

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
everywhere). `*`/`?` do not cross `/`; a `**` SEGMENT does. Not in v1:
negation, nested ignore files, a global ignore — hit a wall first.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath

from .models import HIDDEN_SKIP_DIRS, MANIFEST_NAME, STATE_DIR

IGNORE_NAME = ".slopkeeperignore"


def glob_to_regex(pattern: str) -> re.Pattern:
    """A `/`-aware glob → anchored regex.

    `*`/`?` stay within one path segment; a whole-segment `**` spans
    directories. This is the ONE glob grammar — CorpusIgnore's anchored
    rules and plan `rules[].match` (Phase 8) share it, so a pattern means
    the same thing in an ignore file and in a plan rule. (stdlib
    fnmatch.translate is NOT `-`-aware: its `*` crosses `/`.)
    """
    parts = []
    for seg in pattern.split("/"):
        if seg == "**":
            parts.append(".*")
            continue
        buf = []
        for ch in seg:
            if ch == "*":
                buf.append("[^/]*")
            elif ch == "?":
                buf.append("[^/]")
            else:
                buf.append(re.escape(ch))
        parts.append("".join(buf))
    return re.compile("^" + "/".join(parts) + "$")


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
    """
    root = Path(root)
    ignore = load_ignore(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        kept_dirs = []
        for d in dirnames:
            if d in HIDDEN_SKIP_DIRS:
                continue
            rel = (rel_dir / d).as_posix() if rel_dir.parts else d
            if ignore.is_ignored(rel, is_dir=True):
                continue
            if (Path(dirpath) / d / STATE_DIR / MANIFEST_NAME).exists():
                # A nested corpus: `d` is always a proper descendant
                # here (≠ root — the root's own .slopkeeper is caught
                # by the skip-list above), so its own manifest makes it
                # a separate tended tree. Invisible to the parent,
                # named in nested_out — the parent's reports say what
                # they skip.
                if nested_out is not None:
                    nested_out.append(rel)
                continue
            kept_dirs.append(d)
        dirnames[:] = sorted(kept_dirs)
        kept_files = []
        for fn in filenames:
            # The ignore file itself is tool layer, not content — it
            # shapes every walk and must never appear in one (else
            # writing a comment would shift the fingerprint).
            if fn == IGNORE_NAME and not rel_dir.parts:
                continue
            rel = (rel_dir / fn).as_posix() if rel_dir.parts else fn
            if not ignore.is_ignored(rel, is_dir=False):
                # fp5: only regular files are content. Devices, FIFOs
                # and sockets (sandbox mounts) are pruned — reading one
                # can block forever, and stat-counting one poisoned
                # fingerprint comparability across environments.
                if not (Path(dirpath) / fn).is_file():
                    continue
                kept_files.append(fn)
        yield dirpath, dirnames, sorted(kept_files)


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
