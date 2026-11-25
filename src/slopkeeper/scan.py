# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""One traversal of the corpus, consumed by every read surface (P3/R6).

Before 0.2.8 a single `slopkeeper doctor` walked the tree roughly seven
times and opened most files three or four times: inventory had its own
loop, the fingerprint another, `nested_corpora` another, the visibility
survey another, the nested-repo survey another, and the anchor scan
re-listed the tree to build its own path index. `inventory` was the
worse offender in a different way — it LISTED one walk and FINGERPRINTED
a second one, so the digest a plan bound to described the corpus as it
was milliseconds later than the file list the agent had just read. Two
instants, presented as one (the audit's B5).

`walk_visits` is now the only traversal in the package: one `os.walk`
that applies the corpus skip contract (whose rules `ignore.py` documents)
and yields a `DirVisit` per directory holding `FileEntry` records — the
stat, and on request the content hash, the head bytes and the kind.
`ignore.corpus_walk` is a projection over it (the names-only view every
existing consumer wants), `corpus_fingerprint_detail` is a fold over it,
and `inventory.collect`/`doctor` read the SAME materialized scan, so:

- the fingerprint inventory reports is the fingerprint OF the walk it
  listed — there is no second walk to disagree with;
- one stat per file instead of three, and one content read instead of
  two or four (the hash streams on from the same open that took the
  head bytes);
- doctor's extra surveys (nested corpora, symlinked dirs, ignored files,
  non-regular entries, nested repos) ride along in a `WalkReport` the
  walk fills while it prunes — facts it already learned and used to
  throw away.

The flags exist because the consumers differ: the fingerprint wants
hashes and nothing else, `refs` wants names, inventory wants heads and
kinds. Ask for less, do less.

One deliberate limit on "the text bytes for downstream consumers": the
record carries the HEAD bytes, not the whole file. Every surface that
wants more (the citation scan, the rewrite pass, the anchor scan) reads
the file itself under its own 512 KiB cap. Holding full text per entry
would put the corpus in RAM to serve commands that stream it one file at
a time, and the 60k-file tree R6 exists for is exactly the case where
that choice decides whether the command finishes.
"""

from __future__ import annotations

import hashlib
import os
import stat as statmod
from pathlib import Path

from . import kinds
from .ignore import (
    HIDDEN_SKIP_DIRS,
    IGNORE_NAME,
    MANIFEST_NAME,
    STATE_DIR,
    load_ignore,
)
from .models import SHA256_CAP_BYTES, UNSTATABLE_NAMED

# The head read budget. inventory's `_head` slices this many bytes for
# title/heading work; keeping it on the record means that work costs no
# second open. The sniff reads deeper (kinds.SNIFF_BYTES) and truncates
# to this, so the bytes an entry carries are exactly the bytes inventory
# used to read for itself.
HEAD_BYTES = 4096
_CHUNK = 1 << 20

# Directories whose presence doctor reports as a trap (nested repos,
# dependency trees). Named here because the walk is the only place that
# still SEES them — HIDDEN_SKIP_DIRS prunes them out of every listing.
_WATCHED_HIDDEN = (".git", "node_modules")


class FileEntry:
    """One corpus file as the walk saw it.

    `rel`/`name`/`size`/`mtime`/`mtime_ns`/`mode` are always present;
    `sha256`, `head` and `kind` are None unless the walk was asked for
    them — or the file could not be read, which `read_error` records so
    a missing hash is never silently ambiguous for a caller that asked
    for one.
    """

    __slots__ = (
        "rel",
        "path",
        "name",
        "size",
        "mtime",
        "mtime_ns",
        "mode",
        "sha256",
        "hashed",
        "head",
        "kind",
        "read_error",
    )

    def __init__(self, rel: str, path: Path, name: str, st):
        self.rel = rel
        self.path = path
        self.name = name
        self.size = st.st_size
        self.mtime = st.st_mtime
        self.mtime_ns = st.st_mtime_ns
        self.mode = st.st_mode
        self.sha256: str | None = None
        # False until a real content digest lands. An over-cap or
        # unreadable file stays False, which is how the fingerprint
        # decides to fall back to mtime_ns: one rule, one flag.
        self.hashed = False
        self.head: bytes | None = None
        self.kind: str | None = None
        self.read_error = False


class DirVisit:
    """One directory as the walk saw it: kept children, sorted.

    `rel` is the root-relative posix path ("." for the corpus root),
    `subdirs` the KEPT directory names (skip-list, ignore file and
    nested corpora already pruned), `files` the KEPT regular-file
    entries, `symlinked` the kept dirs that are links as (rel, target).
    `symlinked` is always filled — the fingerprint digest depends on it
    — while the pruned-away facts live in `WalkReport`, where only the
    callers that ask pay for them.
    """

    __slots__ = ("rel", "path", "subdirs", "files", "symlinked")

    def __init__(self, rel: str, path: Path):
        self.rel = rel
        self.path = path
        self.subdirs: list[str] = []
        self.files: list[FileEntry] = []
        self.symlinked: list[tuple[str, str]] = []


class WalkReport:
    """What the walk saw beyond the entries it yields.

    Each field is a fact the pruning loop already learned and used to
    drop: which subtree was skipped as a nested corpus, which directory
    entry was a symlink, which names the ignore file hid, which entries
    were devices or FIFOs, which directories held a `.git` or a
    `node_modules`. `doctor` asks for all of them, which is how its five
    old surveys became this one walk; nobody else pays for the ones it
    does not request. `nested` and `unstatable` are always filled — the
    first because `corpus_walk` has named nested corpora since Phase 7,
    the second because a silently vanished file is what poisoned the
    drift gate (B6).
    """

    __slots__ = (
        "want_non_regular",
        "want_ignored",
        "want_hidden_children",
        "nested",
        "unstatable",
        "ignored_files",
        "non_regular",
        "hidden_children",
        "ignore_rules",
    )

    def __init__(
        self,
        *,
        nested: list[str] | None = None,
        non_regular: bool = False,
        ignored_files: bool = False,
        hidden_children: bool = False,
    ):
        self.want_non_regular = non_regular
        self.want_ignored = ignored_files
        self.want_hidden_children = hidden_children
        self.nested: list[str] = nested if nested is not None else []
        # B6: paths the walk listed and then could not stat. Capped at
        # UNSTATABLE_NAMED — the COUNT is the contract (apply refuses,
        # inventory and doctor warn), the names are diagnostic sugar.
        self.unstatable: list[str] = []
        # Root-relative posix names, REGULAR files only: a FIFO inside an
        # ignored dir is not a strandable file, and doctor's old
        # visibility survey made that distinction, so the walk keeps it.
        self.ignored_files: list[str] = []
        self.non_regular: list[dict] = []
        # rel dir posix -> which of _WATCHED_HIDDEN it holds.
        self.hidden_children: dict[str, list[str]] = {}
        # How many rules the walk's `.slopkeeperignore` carried — the
        # inventory reports the count, and the walk is the only place
        # that has already read the file.
        self.ignore_rules = 0

    def note_unstatable(self, rel: str) -> None:
        if len(self.unstatable) < UNSTATABLE_NAMED:
            self.unstatable.append(rel)


def _was_regular_file(fp: Path) -> bool:
    """True when a path that failed stat() still looks like content.

    The B6 distinction. An entry the walk just listed that cannot be
    stat'd is either churn (lstat still sees a regular file, or the name
    is simply gone — vanishing mid-walk is exactly what B6 counts) or a
    link to nothing, which today's `Path.is_file()` also dropped,
    silently and correctly. Only the first is worth refusing a tend over.
    """
    try:
        lst = fp.lstat()
    except OSError:
        return True  # vanished between the listing and the stat: churn
    return not statmod.S_ISLNK(lst.st_mode)


def _read_entry(
    entry: FileEntry,
    *,
    want_sha: bool,
    want_head: bool,
    need_sniff: bool,
    kind: str | None,
) -> bytes | None:
    """One open per file: the head bytes and/or the content digest.

    The head is read first and its buffer seeds the hash, so the
    fingerprint and the inventory head cost one sequential read instead
    of two opens of the same file. Returns the raw buffer (possibly
    longer than HEAD_BYTES, when the sniff asked for more) so the caller
    can classify without a second open; None means nothing was read.
    """
    hashable = want_sha and entry.size <= SHA256_CAP_BYTES
    must_read = need_sniff or hashable or (want_head and kind != "binary")
    if not must_read:
        # An over-cap file is hashed by mtime (the cap is the contract),
        # and a binary with no head or kind requested needs no open at
        # all — which is how inventory avoided reading 40 MB of video.
        return None
    buf: bytes | None = None
    budget = kinds.SNIFF_BYTES if need_sniff else HEAD_BYTES
    try:
        with entry.path.open("rb") as f:
            if need_sniff or want_head:
                buf = f.read(budget)
            if hashable:
                h = hashlib.sha256()
                if buf:
                    h.update(buf)
                while True:
                    chunk = f.read(_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                entry.sha256 = h.hexdigest()
                entry.hashed = True
    except OSError:
        # Unreadable: no head, no hash. The caller falls back to mtime
        # for the digest and inventory shows `title: null`; the 0.2.7
        # pair of swallowed OSErrors (sha256_file, then _head) lands in
        # the same place, with one flag to say so.
        entry.read_error = True
    return buf


def _scope_under_scope(prefix: str, scope: str) -> bool:
    """Is directory `prefix` ("." for the root) the scope or below it?"""
    if prefix == "." or not prefix:
        return False
    return prefix == scope or prefix.startswith(scope + "/")


def _scope_ancestor(prefix: str, scope: str) -> bool:
    """Is `prefix` a strict ancestor of `scope` (the path we must descend)?"""
    if prefix == "." or not prefix:
        return True
    return scope.startswith(prefix + "/")


def walk_visits(
    root: Path,
    *,
    want_sha: bool = False,
    want_head: bool = False,
    want_kind: bool = False,
    report: WalkReport | None = None,
    scope: str | None = None,
):
    """The corpus traversal: one `os.walk`, one stat per file, `DirVisit`s.

    The skip contract — and the reason for each rule — is documented on
    `ignore.corpus_walk`, whose body is now a projection over this
    generator. `root` is expected absolute, as every caller resolves
    before walking.

    Flags: `want_sha` fills `FileEntry.sha256`/`hashed` (the 64 MiB cap
    applies — over-cap files digest by mtime, see models), `want_head`
    fills the first HEAD_BYTES, `want_kind` fills the `kinds.classify`
    answer. `report` collects the pruned-away facts (see WalkReport);
    pass None and the walk keeps them to a throwaway report.

    `scope` (P4, 2) bounds the walk to one corpus-relative directory and
    its subtree: only that directory and its descendants are visited, so
    a bounded tend of a 60k tree pays for one subtree, not the whole
    corpus. The walk still prunes skip-list / ignored / nested-corpus
    members exactly as it does unsafely; the only change is the descent
    is narrowed to the scope path.
    """
    root = Path(root)
    ignore = load_ignore(root)
    if report is None:
        report = WalkReport()
    report.ignore_rules = ignore.rule_count
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        rel_dir = dpath.relative_to(root)
        prefix = rel_dir.as_posix() if rel_dir.parts else ""

        if scope is not None:
            if not _scope_under_scope(prefix, scope) and not _scope_ancestor(
                prefix, scope
            ):
                # A sibling subtree: never entered when scoped.
                dirnames[:] = []
                continue

        kept_dirs: list[str] = []
        symlinked: list[tuple[str, str]] = []
        for d in dirnames:
            if d in HIDDEN_SKIP_DIRS:
                if (
                    report.want_hidden_children
                    and rel_dir.parts
                    and d in _WATCHED_HIDDEN
                ):
                    report.hidden_children.setdefault(prefix, []).append(d)
                continue
            rel = f"{prefix}/{d}" if prefix else d
            if ignore.is_ignored(rel, is_dir=True):
                continue
            if (dpath / d / STATE_DIR / MANIFEST_NAME).exists():
                # A nested corpus: pruned exactly like a skip-list dir,
                # but NAMED — a silently empty diff is how the Phase 7
                # field report started.
                report.nested.append(rel)
                continue
            kept_dirs.append(d)
        if (
            scope is not None
            and _scope_ancestor(prefix, scope)
            and not _scope_under_scope(prefix, scope)
        ):
            # An ancestor on the way to the scope: follow only the one
            # child that leads in, and yield none of this directory's own
            # files (they are siblings of the scope, not inside it).
            next_part = scope[len(prefix) + 1 if prefix else 0 :].split("/")[0]
            dirnames[:] = [d for d in kept_dirs if d == next_part]
            continue
        kept_dirs.sort()
        # os.walk recurses through THIS list, so pruning and sorting it
        # is not cosmetic: it is what keeps a skip-list directory, an
        # ignored subtree and a nested corpus un-descended, and what
        # makes the entry order (and therefore the fingerprint digest)
        # deterministic across filesystems.
        dirnames[:] = kept_dirs
        # fp6 (A2): a symlinked directory is an ENTRY the plan can be
        # held to, so name+target always travel — the digest needs them
        # and a partial digest would be worse than an extra lstat per
        # directory. os.walk never descends a link, so this is all the
        # walk ever sees behind one.
        for d in kept_dirs:
            dp = dpath / d
            try:
                if not dp.is_symlink():
                    continue
                target = os.readlink(dp)
            except OSError:
                report.note_unstatable(str(dp.relative_to(root)))
                continue
            symlinked.append((f"{prefix}/{d}" if prefix else d, target))

        entries: list[FileEntry] = []
        for fn in filenames:
            # The ignore file itself is tool layer, never content: it
            # shapes every walk and must not appear in one.
            if fn == IGNORE_NAME and not rel_dir.parts:
                continue
            rel = f"{prefix}/{fn}" if prefix else fn
            fp = dpath / fn
            ignored = ignore.is_ignored(rel, is_dir=False)
            if ignored and not report.want_ignored:
                continue
            try:
                st = fp.stat()
            except OSError:
                if not _was_regular_file(fp):
                    continue
                report.note_unstatable(str(fp.relative_to(root)))
                continue
            if not statmod.S_ISREG(st.st_mode):
                # fp5: devices, FIFOs and sockets are pruned from every
                # surface (reading one can block forever); doctor NAMES
                # what was pruned, which is why the entry is recorded
                # before the drop.
                if report.want_non_regular:
                    report.non_regular.append(
                        {"path": rel, "mode": statmod.S_IFMT(st.st_mode)}
                    )
                continue
            if ignored:
                # An ignored REGULAR file: counted for doctor's stranding
                # survey, then dropped — invisible to inventory, the
                # fingerprint and every scan, exactly as before.
                report.ignored_files.append(rel)
                continue
            entry = FileEntry(rel, fp, fn, st)
            kind = kinds.kind_of(fp)
            buf = _read_entry(
                entry,
                want_sha=want_sha,
                want_head=want_head,
                need_sniff=kind is None and (want_kind or want_head),
                kind=kind,
            )
            if kind is None and (want_kind or want_head):
                kind = kinds.classify(fp, head=buf)
                if entry.read_error:
                    # Nothing could be read: inventory's `_kind` called
                    # that binary first, and the same answer keeps the
                    # head path and the citation scan consistent.
                    kind = "binary"
                entry.kind = kind if want_kind else None
            elif want_kind:
                entry.kind = kind
            if want_head and kind != "binary" and buf is not None:
                entry.head = buf[:HEAD_BYTES]
            entries.append(entry)
        entries.sort(key=lambda e: e.name)

        visit = DirVisit(prefix or ".", dpath)
        # A copy, not the live list: os.walk recurses through `dirnames`,
        # and a consumer editing what it was handed would silently edit
        # the traversal. The 0.2.7 inventory did exactly that to prune
        # its own depth; the scan takes the cut as an argument instead.
        visit.subdirs = list(kept_dirs)
        visit.files = entries
        visit.symlinked = symlinked
        yield visit


# --- the materialized scan ------------------------------------------------


class CorpusScan:
    """Every entry of one traversal, held.

    The read surfaces (inventory, doctor, clusters) want the walk more
    than once — a flat file list, the directory tree, the fingerprint —
    and re-walking is how those views drifted apart. Materialize once,
    fold many: the memory is one small record per file, which is less
    than the inventory document already held.
    """

    __slots__ = ("root", "visits", "files", "by_rel", "report", "_rel_paths")

    def __init__(self, root: Path, report: WalkReport | None = None):
        self.root = root
        self.visits: list[DirVisit] = []
        self.files: list[FileEntry] = []
        self.by_rel: dict[str, FileEntry] = {}
        self.report = report if report is not None else WalkReport()
        self._rel_paths: list[Path] | None = None

    # -- views ---------------------------------------------------------

    def rel_paths(self) -> list[Path]:
        """The corpus as a list of root-relative Paths, walk order.

        Built once and cached: `CitationIndex`, the anchor scan and the
        reference scan each used to re-walk the tree to get this list.
        """
        if self._rel_paths is None:
            self._rel_paths = [Path(e.rel) for e in self.files]
        return self._rel_paths

    def sha_of(self, rel: str) -> str | None:
        """The content digest the walk took, by root-relative path.

        Same value — and the same None-for-over-cap rule — as
        `models.sha256_file`, which is what the vector cache keys on, so
        `clusters` reuses the inventory's hashes instead of hashing the
        corpus a second time.
        """
        entry = self.by_rel.get(rel)
        if entry is None or not entry.hashed:
            return None
        return entry.sha256

    def health(self) -> dict:
        """The B6 payload: what this walk could not see, counted."""
        out = {"skipped_unstatable": len(self.report.unstatable)}
        if self.report.unstatable:
            out["skipped_unstatable_paths"] = list(self.report.unstatable)
        return out

    def fingerprint_health(self) -> tuple[str, dict]:
        """(fingerprint, health) of THIS walk — `corpus_fingerprint_detail`
        answered from the records already in hand, no second traversal.
        The health half is the walk's own blind spot (B6), so inventory
        and doctor echo the same numbers the digest was built beside."""
        return fingerprint_from_visits(self.visits, self.report)

    def fingerprint(self) -> str:
        """The digest of THIS walk — the number the plan binds to."""
        return self.fingerprint_health()[0]


def scan_corpus(
    root: Path,
    *,
    want_sha: bool = False,
    want_head: bool = False,
    want_kind: bool = False,
    report: WalkReport | None = None,
    scope: str | None = None,
) -> CorpusScan:
    """Run the one traversal and hold every entry (see CorpusScan).

    `scope` (P4, 2) restricts the walk to one corpus-relative directory
    subtree; every fold over the result (fingerprint, the inventory
    document, the manifest) then describes that scope and nothing else.
    """
    root = Path(root)
    scan = CorpusScan(root, report)
    for visit in walk_visits(
        root,
        want_sha=want_sha,
        want_head=want_head,
        want_kind=want_kind,
        report=scan.report,
        scope=scope,
    ):
        scan.visits.append(visit)
        for e in visit.files:
            scan.files.append(e)
            scan.by_rel[e.rel] = e
    return scan


def prepare(root: Path, scope: str | None = None) -> CorpusScan:
    """The full-facility scan: hashes, heads, kinds, every report facet.

    `doctor` — and `clusters`, which wants the hashes for its vector
    cache — calls this once and answers every survey it has from the
    result. `inventory.collect` builds its own narrower scan when it is
    handed none. `scope` (P4, 2) bounds the whole command to one subtree.
    """
    report = WalkReport(non_regular=True, ignored_files=True, hidden_children=True)
    return scan_corpus(
        root,
        want_sha=True,
        want_head=True,
        want_kind=True,
        report=report,
        scope=scope,
    )


# --- the folds ------------------------------------------------------------


def fingerprint_from_visits(
    visits, report: WalkReport | None = None
) -> tuple[str, dict]:
    """(fingerprint, health) folded over visits the caller already holds.

    The digest format is unchanged from 0.2.7 — per regular entry
    `relpath\\0size:digest\\0`, per symlinked directory
    `relpath\\0symlink:target\\0`, `CURRENT_FP_PREFIX` and the entry
    count framing the hex — this is the same fold with its input lifted
    out of the walk so inventory can report the digest of the walk it
    actually listed. `visits` may be the lazy generator from
    `walk_visits` (nothing materialized) or a list; `report` is the
    walk's own, whose `unstatable` names the blind spot (B6).
    """
    from .models import CURRENT_FP_PREFIX

    h = hashlib.sha256()
    count = 0
    for visit in visits:
        for rel, target in visit.symlinked:
            h.update(rel.encode("utf-8", "replace"))
            h.update(b"\0symlink:")
            h.update(target.encode("utf-8", "replace"))
            h.update(b"\0")
            count += 1
        for entry in visit.files:
            h.update(entry.rel.encode("utf-8", "replace"))
            h.update(b"\0")
            h.update(str(entry.size).encode())
            h.update(b":")
            if entry.hashed:
                h.update(entry.sha256.encode())
            else:
                h.update(str(entry.mtime_ns).encode())
            h.update(b"\0")
            count += 1
    unstatable = list(report.unstatable) if report is not None else []
    health: dict = {"skipped_unstatable": len(unstatable)}
    if unstatable:
        health["skipped_unstatable_paths"] = unstatable
    return f"{CURRENT_FP_PREFIX}:{h.hexdigest()[:32]}:{count}", health


def fingerprint_scan(root: Path, scope: str | None = None) -> tuple[str, dict]:
    """(fingerprint, health) — the streaming fold, nothing materialized.

    `scope` (P4, 2) digests only one corpus-relative subtree: a bounded
    tend binds to the part of the corpus it actually governs, so a change
    in a sibling scope is not drift against this plan.
    """
    report = WalkReport()
    return fingerprint_from_visits(
        walk_visits(root, want_sha=True, report=report, scope=scope), report
    )


# --- the names-only views -------------------------------------------------


def iter_entries(root: Path):
    """Every kept `FileEntry`, flat, walk order (nothing else asked for)."""
    for visit in walk_visits(root):
        for e in visit.files:
            yield e


def iter_rel_paths(root: Path):
    """`iter_entries` as root-relative Paths, lazily."""
    for entry in iter_entries(root):
        yield Path(entry.rel)


def corpus_rel_paths(root: Path) -> list[Path]:
    """Every corpus file as a root-relative Path, walk order."""
    return [Path(e.rel) for e in iter_entries(root)]
