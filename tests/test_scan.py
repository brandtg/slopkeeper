# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""scan.py — the single pass (P3/R6 acceptance).

Three promises, each tested from the outside so a refactor cannot keep
them by accident:

1. ONE traversal. `doctor` walked the corpus about seven times and
   `inventory` twice; every corpus-side command now walks exactly once,
   which the `os.walk` counter below pins as a NUMBER, not a hope.
2. ONE instant. The fingerprint `inventory` reports is the digest of the
   walk it actually listed (audit B5: two walks, two instants, presented
   as one document). The churn test proves it by moving the corpus
   between the scan and the document.
3. THE SAME ANSWERS. The fold is a refactor of the 0.2.7 walk, so the
   entry set (skip list, ignore file, nested corpora, non-regular
   pruning, symlinked-dir entries) and the digest are unchanged — the
   parity tests compare against an independent re-implementation of the
   old rules rather than trusting the new one to grade itself.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from helpers import TempCase, write

from slopkeeper import scan as scan_mod
from slopkeeper.doctor import doctor
from slopkeeper.ignore import IGNORE_NAME, corpus_walk
from slopkeeper.inventory import collect
from slopkeeper.models import (
    CURRENT_FP_PREFIX,
    corpus_fingerprint,
    corpus_fingerprint_detail,
)
from slopkeeper.scan import WalkReport, fingerprint_from_visits, prepare, scan_corpus


def rel_files(root: Path) -> list[str]:
    """Every corpus file path, via the names-only projection (corpus_walk)."""
    out = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        d = Path(dirpath)
        for fn in filenames:
            out.append((d / fn).relative_to(root).as_posix())
    return sorted(out)


class WalkCounter:
    """Count `os.walk` calls for the duration of a block."""

    def __init__(self):
        self.n = 0
        self._real = os.walk

    def __enter__(self):
        real = self._real

        def counting(*args, **kwargs):
            self.n += 1
            return real(*args, **kwargs)

        self._patch = mock.patch.object(os, "walk", counting)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class TestOneTraversal(TempCase):
    """R6's number: one materialized pass per command."""

    def test_inventory_walks_once(self):
        with WalkCounter() as w:
            collect(self.root, fast=True)
        self.assertEqual(w.n, 1, "inventory re-walked the corpus")

    def test_doctor_walks_once(self):
        # 0.2.7 ran about seven: inventory, the fingerprint, the nested
        # corpora, the visibility survey, the nested-repo survey, the
        # anchor scan's path list and the dangling-citation scan's list.
        with WalkCounter() as w:
            doctor(self.root)
        self.assertEqual(w.n, 1, "doctor re-walked the corpus")

    def test_fingerprint_walks_once(self):
        with WalkCounter() as w:
            corpus_fingerprint(self.root)
        self.assertEqual(w.n, 1)

    def test_refs_scan_walks_once(self):
        from slopkeeper.references import scan_references_stream

        with WalkCounter() as w:
            scan_references_stream(self.root)
        self.assertEqual(w.n, 1)

    def test_anchors_walk_once(self):
        from slopkeeper.anchors import anchored_dirs

        with WalkCounter() as w:
            anchored_dirs(self.root)
        self.assertEqual(w.n, 1)

    def test_status_walks_once(self):
        from slopkeeper.status import status

        with WalkCounter() as w:
            status(self.root)
        self.assertEqual(w.n, 1)

    def test_a_prepared_scan_is_not_rewalked_by_its_consumers(self):
        # doctor hands one scan to inventory AND to the anchor scan; the
        # counter across the whole command stays at one.
        with WalkCounter() as w:
            scan = prepare(self.root)
            collect(self.root, fast=True, scan=scan)
            from slopkeeper.anchors import anchored_dirs
            from slopkeeper.references import CitationIndex

            anchored_dirs(
                self.root,
                rel_paths=scan.rel_paths(),
                index=CitationIndex(scan.rel_paths()),
            )
        self.assertEqual(w.n, 1)

    def test_collect_refuses_a_scan_missing_its_facilities(self):
        # A pass built for the fingerprint alone has no kinds and no
        # heads; folding an inventory out of it would ship `kind: null`
        # rows and an mtime-based digest. collect walks for itself
        # instead of trusting a scan it can tell is thin.
        thin = scan_corpus(self.root, want_sha=True)
        full = collect(self.root, fast=True)
        with WalkCounter() as w:
            doc = collect(self.root, fast=True, scan=thin)
        self.assertEqual(w.n, 1, "collect folded a scan it should have replaced")
        self.assertEqual(doc["fingerprint"], full["fingerprint"])
        self.assertEqual(
            [(f["kind"], f["path"]) for f in doc["files"]],
            [(f["kind"], f["path"]) for f in full["files"]],
        )
        # ...and a complete scan really is reused: no walk inside collect.
        ready = prepare(self.root)
        with WalkCounter() as w:
            collect(self.root, fast=True, scan=ready)
        self.assertEqual(w.n, 0)


class TestOneInstant(TempCase):
    """The fingerprint is the walk's, not a second walk's (B5)."""

    def test_inventory_fingerprint_is_the_digest_of_the_walk_it_listed(self):
        scan = prepare(self.root)
        listed = sorted(
            Path(f["path"]) for f in collect(self.root, fast=True, scan=scan)["files"]
        )
        # The corpus moves AFTER the walk: the document must still
        # describe the instant it walked, because that is the instant a
        # plan would bind to.
        write(self.root / "latecomer.md", "# late\n")
        (self.root / "acme_dossier.md").write_text("# mutated\n", encoding="utf-8")
        inv = collect(self.root, fast=True, scan=scan)
        self.assertEqual(inv["fingerprint"], scan.fingerprint())
        self.assertNotIn("latecomer.md", [f["path"] for f in inv["files"]])
        self.assertEqual(sorted(Path(f["path"]) for f in inv["files"]), listed)
        # ...and a FRESH look does see the mutation, so this is a stale
        # scan being honest about itself, not a frozen constant.
        self.assertNotEqual(
            collect(self.root, fast=True)["fingerprint"], inv["fingerprint"]
        )

    def test_inventory_fingerprint_matches_the_standalone_fold(self):
        self.assertEqual(
            collect(self.root, fast=True)["fingerprint"], corpus_fingerprint(self.root)
        )

    def test_fold_and_walk_agree_on_a_hostile_corpus(self):
        # An independent re-implementation of the 0.2.7 rules (its own
        # os.walk, its own filters) must produce the same digest and the
        # same health — the refactor cannot grade itself.
        root = self.root
        (root / "link-dir-target").mkdir()
        write(root / "link-dir-target" / "x.md", "# x\n")
        os.symlink(root / "link-dir-target", root / "linked")
        (root / "sub" / ".slopkeeper").mkdir(parents=True)
        write(root / "sub" / ".slopkeeper" / "manifest.json", "{}")
        write(root / "sub" / "hidden-from-parent.md", "# hidden\n")
        (root / IGNORE_NAME).write_text("ignored/\n", encoding="utf-8")
        (root / "ignored").mkdir()
        write(root / "ignored" / "secret.md", "# s\n")
        os.mkfifo(root / "pipe")

        from slopkeeper.ignore import (
            HIDDEN_SKIP_DIRS,
            MANIFEST_NAME,
            STATE_DIR,
            load_ignore,
        )

        ignore = load_ignore(root)
        h = __import__("hashlib").sha256()
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            rel_dir = d.relative_to(root)
            kept = []
            for name in dirnames:
                if name in HIDDEN_SKIP_DIRS:
                    continue
                rel = (rel_dir / name).as_posix() if rel_dir.parts else name
                if ignore.is_ignored(rel, is_dir=True):
                    continue
                if (d / name / STATE_DIR / MANIFEST_NAME).exists():
                    continue
                kept.append(name)
            dirnames[:] = sorted(kept)
            for name in dirnames:
                dp = d / name
                if not dp.is_symlink():
                    continue
                h.update(str(dp.relative_to(root)).encode("utf-8", "replace"))
                h.update(b"\0symlink:")
                h.update(os.readlink(dp).encode("utf-8", "replace"))
                h.update(b"\0")
                count += 1
            for fn in sorted(filenames):
                if fn == IGNORE_NAME and not rel_dir.parts:
                    continue
                rel = (rel_dir / fn).as_posix() if rel_dir.parts else fn
                if ignore.is_ignored(rel, is_dir=False):
                    continue
                fp = d / fn
                if not fp.is_file():
                    continue
                st = fp.stat()
                h.update(str(fp.relative_to(root)).encode("utf-8", "replace"))
                h.update(b"\0")
                h.update(str(st.st_size).encode())
                h.update(b":")
                from slopkeeper.models import sha256_file

                digest = sha256_file(fp)
                h.update(
                    digest.encode()
                    if digest is not None
                    else str(st.st_mtime_ns).encode()
                )
                h.update(b"\0")
                count += 1
        want = f"{CURRENT_FP_PREFIX}:{h.hexdigest()[:32]}:{count}"
        got, health = corpus_fingerprint_detail(root)
        self.assertEqual(got, want)
        self.assertEqual(health, {"skipped_unstatable": 0})


class TestFlagsAndRecords(TempCase):
    """Ask for less, do less — and record what the pruning dropped."""

    def test_default_walk_asks_nothing_of_the_disk(self):
        got = [
            (e.rel, e.sha256, e.head, e.kind) for e in scan_mod.iter_entries(self.root)
        ]
        self.assertTrue(got)
        for _rel, sha, head, kind in got:
            self.assertIsNone(sha)
            self.assertIsNone(head)
            self.assertIsNone(kind)

    def test_want_fills_exactly_what_was_asked(self):
        s = scan_corpus(self.root, want_sha=True)
        self.assertTrue(all(e.hashed for e in s.files))
        self.assertTrue(all(e.sha256 for e in s.files))
        self.assertTrue(all(e.head is None and e.kind is None for e in s.files))
        s = scan_corpus(self.root, want_head=True)
        self.assertTrue(all(e.head is not None for e in s.files))
        self.assertTrue(all(len(e.head) <= scan_mod.HEAD_BYTES for e in s.files))
        self.assertTrue(all(not e.hashed and e.kind is None for e in s.files))

    def test_sha_of_matches_sha256_file_and_flags_over_cap(self):
        from slopkeeper.models import sha256_file

        s = scan_corpus(self.root, want_sha=True)
        for e in s.files:
            self.assertEqual(s.sha_of(e.rel), sha256_file(e.path))
        # Over-cap files are hashed by mtime, so `sha_of` says None and
        # the fold knows to reach for mtime_ns — one rule, one flag.
        with mock.patch.object(scan_mod, "SHA256_CAP_BYTES", 8):
            capped = scan_corpus(self.root, want_sha=True)
        big = [e for e in capped.files if e.size > 8]
        self.assertTrue(big, "fixture has no over-cap file")
        for e in big:
            self.assertFalse(e.hashed)
            self.assertIsNone(capped.sha_of(e.rel))

    def test_each_file_is_opened_once_per_pass(self):
        # The head bytes and the content digest come from ONE open: the
        # read helper runs once per file even with every facility asked
        # for (which used to be two opens, and three with the old
        # fingerprint walk alongside the inventory walk).
        s = scan_corpus(self.root, want_sha=True, want_head=True, want_kind=True)
        real = scan_mod._read_entry
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(scan_mod, "_read_entry", counting):
            scan_corpus(self.root, want_sha=True, want_head=True, want_kind=True)
        self.assertEqual(len(calls), len(s.files))

    def test_binary_files_are_not_read_for_a_head(self):
        (self.root / "blob.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        s = scan_corpus(self.root, want_head=True, want_kind=True)
        entry = next(e for e in s.files if e.rel == "blob.png")
        self.assertEqual(entry.kind, "binary")
        self.assertIsNone(entry.head)

    def test_report_names_nested_corpora_symlinks_and_pruned_entries(self):
        root = self.root
        (root / "outside").mkdir()
        write(root / "outside" / "t.md", "# t\n")
        os.symlink(root / "outside", root / "linked")
        (root / "sub" / ".slopkeeper").mkdir(parents=True)
        write(root / "sub" / ".slopkeeper" / "manifest.json", "{}")
        write(root / "sub" / "inner.md", "# inner\n")
        (root / IGNORE_NAME).write_text("secrets/\n*.tmp\n", encoding="utf-8")
        (root / "secrets").mkdir()
        write(root / "secrets" / "k.md", "# k\n")
        write(root / "scratch.tmp", "scratch\n")
        (root / "repo" / ".git").mkdir(parents=True)
        write(root / "repo" / ".git" / "HEAD", "ref: refs/heads/main\n")
        (root / "app" / "node_modules").mkdir(parents=True)
        write(root / "app" / "node_modules" / "dep.js", "//x\n")
        os.mkfifo(root / "pipe")

        report = WalkReport(non_regular=True, ignored_files=True, hidden_children=True)
        s = scan_corpus(root, report=report)
        self.assertEqual(sorted(set(report.nested)), ["sub"])
        syms = {(rel, target) for v in s.visits for rel, target in v.symlinked}
        self.assertEqual(len(syms), 1)
        self.assertEqual(syms.pop()[0], "linked")
        # A FILE-level ignore is recorded (the stranding case: half a
        # folder hidden); a DIRECTORY-level ignore prunes the descent, so
        # nothing inside it is ever a strand — the rule the pre-0.2.8
        # visibility survey had, kept verbatim.
        self.assertEqual(sorted(report.ignored_files), ["scratch.tmp"])
        self.assertEqual([r["path"] for r in report.non_regular], ["pipe"])
        self.assertEqual(
            report.hidden_children, {"repo": [".git"], "app": ["node_modules"]}
        )
        # The pruned facts never leak INTO the walk: an ignored file and
        # a nested corpus's contents are not entries.
        rels = {e.rel for e in s.files}
        self.assertNotIn("secrets/k.md", rels)
        self.assertNotIn("scratch.tmp", rels)
        self.assertNotIn("sub/inner.md", rels)
        self.assertNotIn("repo/.git/HEAD", rels)

    def test_unstatable_is_counted_once_and_named(self):
        # The B6 race: the walk lists the file, then stat() fails. (The
        # existing churn tests exercise this through apply; the scan
        # contract is asserted here so the number in the envelope has a
        # source the reader can see.)
        target = self.root / "churny.md"
        write(target, "here then gone\n")
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if Path(self) == target:
                raise OSError(2, "No such file or directory")
            return real_stat(self, *args, **kwargs)

        with mock.patch.object(Path, "stat", stat):
            s = scan_corpus(self.root, want_sha=True)
        self.assertEqual(s.health()["skipped_unstatable"], 1)
        self.assertEqual(s.health()["skipped_unstatable_paths"], ["churny.md"])
        self.assertNotIn("churny.md", {e.rel for e in s.files})
        # The digest sees the churn too, so the gate cannot bless it.
        self.assertNotEqual(s.fingerprint(), corpus_fingerprint(self.root))

    def test_dangling_symlink_is_pruned_not_counted_as_churn(self):
        # A link to nothing is not a corpus being churned under the
        # scan: counting it would make `apply` refuse a tend (B6's
        # corpus_unstable) over a file that was never content.
        os.symlink("does-not-exist.md", self.root / "dangling.md")
        s = scan_corpus(self.root, want_sha=True)
        self.assertEqual(s.health()["skipped_unstatable"], 0)
        self.assertNotIn("dangling.md", {e.rel for e in s.files})


class TestViews(TempCase):
    def test_rel_paths_is_the_walk_order_list(self):
        s = scan_corpus(self.root)
        self.assertEqual(
            sorted(p.as_posix() for p in s.rel_paths()),
            rel_files(self.root),
        )
        # Walk order, not sort order: the root's own files come before
        # any subtree, which is what makes the digest deterministic.
        self.assertEqual([p.as_posix() for p in s.rel_paths()][0], "WATCHLIST.md")
        # cached: the same list object, so consumers cannot disagree by
        # rebuilding it at different instants
        self.assertIs(s.rel_paths(), s.rel_paths())

    def test_by_rel_indexes_every_entry(self):
        s = scan_corpus(self.root, want_sha=True)
        for e in s.files:
            self.assertIs(s.by_rel[e.rel], e)
        self.assertIsNone(s.sha_of("nope.md"))

    def test_visits_are_deterministic(self):
        first = [
            (v.rel, [e.rel for e in v.files]) for v in scan_corpus(self.root).visits
        ]
        second = [
            (v.rel, [e.rel for e in v.files]) for v in scan_corpus(self.root).visits
        ]
        self.assertEqual(first, second)

    def test_fold_over_a_list_equals_the_streaming_fold(self):
        visits = list(scan_mod.walk_visits(self.root, want_sha=True))
        report = WalkReport()
        list_fp = fingerprint_from_visits(visits, report)
        stream_report = WalkReport()
        stream_fp = fingerprint_from_visits(
            scan_mod.walk_visits(self.root, want_sha=True, report=stream_report),
            stream_report,
        )
        self.assertEqual(list_fp, stream_fp)


if __name__ == "__main__":
    unittest.main()
