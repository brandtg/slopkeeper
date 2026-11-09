# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""models.py — safe paths, fingerprint, move chains."""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

from helpers import TempCase, write
from slopkeeper.models import (
    SlopKeeperError,
    corpus_fingerprint,
    resolve_chains,
    safe_relpath,
    sha256_file,
)


class TestModels(unittest.TestCase):
    def test_safe_relpath_ok(self):
        root = Path("/tmp/x")
        self.assertEqual(safe_relpath("a/b.md", root), Path("a/b.md"))
        self.assertEqual(safe_relpath("/tmp/x/a.md", root), Path("a.md"))

    def test_safe_relpath_rejects_escape(self):
        with self.assertRaises(SlopKeeperError):
            safe_relpath("../etc/passwd", Path("/tmp/x"))

    def test_chain_resolution(self):
        # Flipped (src round, data-loss fix): file→file chains keep
        # their hops. The old rewrite (a→c, b→c) made apply() move two
        # files onto one path and silently destroy one; the executor
        # orders the hops instead.
        moves = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]
        self.assertEqual(resolve_chains(moves), {"a": "b", "b": "c"})

    def test_chain_three_hops_kept(self):
        moves = [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "d"},
        ]
        self.assertEqual(resolve_chains(moves), {"a": "b", "b": "c", "c": "d"})

    def test_chain_and_dir_child_combined(self):
        # Both mechanisms in one plan: file→file hops kept as-is AND a
        # dir-child destination rewritten onto the renamed parent.
        moves = [
            {"from": "a.md", "to": "b.md"},
            {"from": "b.md", "to": "c.md"},
            {"from": "d", "to": "e"},
            {"from": "d/f.txt", "to": "d/g.txt"},
        ]
        self.assertEqual(
            resolve_chains(moves),
            {
                "a.md": "b.md",
                "b.md": "c.md",
                "d": "e",
                "d/f.txt": "e/g.txt",
            },
        )

    def test_chain_cycle_raises(self):
        moves = [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}]
        with self.assertRaises(SlopKeeperError):
            resolve_chains(moves)

    def test_dir_move_rewrites_child(self):
        moves = [
            {"from": "proj", "to": "archive/proj"},
            {"from": "proj/notes.md", "to": "proj/renamed.md"},
        ]
        out = resolve_chains(moves)
        self.assertEqual(out["proj"], "archive/proj")
        self.assertEqual(out["proj/notes.md"], "archive/proj/renamed.md")


class TestFingerprint(TempCase):
    def test_changes_on_content(self):
        fp1 = corpus_fingerprint(self.root)
        (self.root / "WATCHLIST.md").write_text("changed\n")
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))

    def test_fingerprint_is_version_prefixed_fp5(self):
        # v4 format: plans authored against older fingerprints (different
        # prefix) can never collide with a current one — they fail
        # cleanly with corpus_changed instead (DESIGN.md §5.2, §6.6).
        self.assertTrue(corpus_fingerprint(self.root).startswith("fp5:"))

    def test_dot_dir_content_shifts_fingerprint(self):
        # fp5 walk contract: a file inside a dot-directory is content —
        # adding or editing it moves the fingerprint (Phase 1 of the
        # reference-rewrites PRD).
        fp1 = corpus_fingerprint(self.root)
        write(self.root / ".claude" / "skills" / "s.md", "# skill\n")
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))

    def test_touch_does_not_change_fingerprint(self):
        # The documented-behavior pin, FLIPPED by §5.2 (was: touch
        # invalidates). Staleness is content-based: same size, same
        # content, new mtime → identical fingerprint. mtime is still
        # hashed, but a bare touch can't move it past the content.
        p = self.root / "acme_dossier.md"
        os.utime(p, (1_000_000_000, 1_000_000_000))
        fp1 = corpus_fingerprint(self.root)
        os.utime(p, (1_500_000_000, 1_500_000_000))
        self.assertEqual(fp1, corpus_fingerprint(self.root))

    def test_same_size_content_edit_changes_fingerprint(self):
        # The false-negative hole §5.2 closes: a byte-for-byte swap of
        # equal length (same size, different content) must change the
        # fingerprint even though the old mtime-only gate would too —
        # here it changes because of the CONTENT, not the timestamp.
        p = self.root / "lead-gen" / "README.md"
        fp1 = corpus_fingerprint(self.root)
        p.write_bytes(p.read_bytes().replace(b"pipeline", b"pipezinb"))
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))


class TestSafeRelpathHostile(TempCase):
    """Hostile-input table for safe_relpath (TEST_PLAN models row).

    Designed behavior (src round): absolute paths outside root, '..'
    traversal, null bytes, and paths that normalize to the corpus root
    itself ('.' or empty) raise SlopKeeperError('path_escape'). '~' and
    backslashes are ACCEPTED: POSIX treats both as legal filename
    characters, and safe_relpath deliberately does no expanduser() and
    no Windows separator semantics.
    """

    def test_rejects_absolute_paths_outside_root(self):
        for raw in ["/etc/passwd", "/tmp/../etc/passwd"]:
            with self.assertRaises(SlopKeeperError) as cm:
                safe_relpath(raw, self.root)
            self.assertEqual(cm.exception.code, "path_escape")

    def test_rejects_dotdot_traversal(self):
        for raw in ["..", "a/../../b", "sub/../../etc"]:
            with self.assertRaises(SlopKeeperError) as cm:
                safe_relpath(raw, self.root)
            self.assertEqual(cm.exception.code, "path_escape")

    def test_rejects_null_bytes(self):
        # A NUL byte would crash later OS calls with an opaque
        # ValueError; it is rejected here with the path_escape code.
        with self.assertRaises(SlopKeeperError) as cm:
            safe_relpath("bad\0name.md", self.root)
        self.assertEqual(cm.exception.code, "path_escape")
        self.assertIn("null byte", cm.exception.message)

    def test_rejects_paths_that_normalize_to_the_root(self):
        # '.', '' (and any spelling that collapses to them) name the
        # corpus root itself — not a movable file — so they are rejected.
        for raw in [".", "", "./", "sub/.."]:
            with self.assertRaises(SlopKeeperError) as cm:
                safe_relpath(raw, self.root)
            self.assertEqual(cm.exception.code, "path_escape")

    def test_accepts_tilde_as_literal_name(self):
        # No expanduser() in safe_relpath: '~' is a legal POSIX filename
        # character and stays a literal name (deliberate, src round).
        self.assertEqual(safe_relpath("~/secrets.md", self.root), Path("~/secrets.md"))

    def test_accepts_backslash_as_literal_name(self):
        # POSIX filenames may contain '\'; Windows-style separators are
        # not special here (deliberate, src round).
        self.assertEqual(safe_relpath("odd\\name.md", self.root), Path("odd\\name.md"))


class TestFingerprintGate(TempCase):
    """corpus_fingerprint invariants: add/remove/stability/hidden dirs."""

    def test_changes_when_file_added(self):
        fp1 = corpus_fingerprint(self.root)
        write(self.root / "new_arrival.md", "# new\n")
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))

    def test_changes_when_file_removed(self):
        fp1 = corpus_fingerprint(self.root)
        (self.root / "acme_dossier.md").unlink()
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))

    def test_stable_on_unchanged_tree(self):
        self.assertEqual(corpus_fingerprint(self.root), corpus_fingerprint(self.root))

    def test_hidden_dirs_do_not_invalidate_gate(self):
        # '.slopkeeper/' holds the journal and manifest; it must never
        # flip the fingerprint, or every apply would staleness-refuse
        # its own writes.
        fp1 = corpus_fingerprint(self.root)
        write(self.root / ".slopkeeper/journal.jsonl", '{"op": 1}\n')
        write(self.root / ".obsidian/appearance.json", "{}\n")
        write(self.root / ".git/config", "[core]\n")
        self.assertEqual(fp1, corpus_fingerprint(self.root))

    def test_dotfiles_are_fingerprinted_pin(self):
        # Actual behavior: only hidden DIRS are skipped (models.py
        # filters os.walk's dirnames); a root-level dotfile IS hashed.
        # TEST_PLAN's "skips hidden dirs/dotfiles" wording does not
        # match the code — pinned here, flagged in the summary.
        fp1 = corpus_fingerprint(self.root)
        write(self.root / ".env", "SECRET=1\n")
        self.assertNotEqual(fp1, corpus_fingerprint(self.root))


class TestSha256File(TempCase):
    def test_returns_none_past_cap(self):
        # Default cap is 64MiB — too big to materialize in a fast suite,
        # so exercise the cap logic with a small explicit limit.
        f = write(self.root / "blob.bin", "12345")  # 5 bytes
        self.assertEqual(
            sha256_file(f, cap_bytes=10), hashlib.sha256(b"12345").hexdigest()
        )
        self.assertIsNotNone(sha256_file(f, cap_bytes=5))  # at cap still hashes
        self.assertIsNone(sha256_file(f, cap_bytes=4))  # strictly past cap


if __name__ == "__main__":
    unittest.main()
