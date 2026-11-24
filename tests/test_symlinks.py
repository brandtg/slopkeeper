# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""A2 (0.2.6): the corpus's first symlink fixtures — the containment
contract for moves (safe_move_path), the fp6 fingerprint entries, and
the inventory side of the reconciliation.

The audit found `safe_relpath` symlink-checked only the FINAL path
component: with `corpus/notes -> /tmp/opencode/outside`, `apply
--execute` moved files OUT of the corpus and `plan validate` said
`valid: true`. These fixtures pin the refusal at every surface
(validate, dry-run, execute — including under --force), the deliberate
--allow-symlinks bypass, hardlink-into-corpus and symlink-at-root
cases, and the fingerprint/inventory contract: a symlinked directory
is an ENTRY (name + target bound, contents invisible).
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import (
    TempCase,
    expect_code,
    make_symlink,
    write,
    write_plan,
)
from slopkeeper.apply import apply_plan
from slopkeeper.inventory import collect
from slopkeeper.models import corpus_fingerprint
from slopkeeper.plan import load_plan, validate_plan
from test_cli import run_cli

# Everything DEFAULT_CORPUS puts at top level except what a test moves.
FULL_KEEP = [
    "WATCHLIST.md",
    "docs",
    "lead-gen",
    "spark-1.2.3",
    "globex_dossier.md",
    "handoff-tp2.md",
    "ledger_wb.html",
]


def _move_plan(root: Path, src: str, dst: str, keep: list[str]):
    """A bound one-move plan (partial: coverage beyond the move is not
    the point of these fixtures)."""
    return write_plan(
        root,
        moves=[{"from": src, "to": dst, "reason": "symlink fixture"}],
        keep=keep,
        folders=[],
        edits=[],
        unresolved=[],
        partial=True,
    )


class TestMoveContainment(TempCase):
    """safe_move_path: through, onto, and the deliberate bypass."""

    def _escaped_notes(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        make_symlink(outside, self.root / "notes")
        return outside

    def test_validate_refuses_move_through_escaping_symlinked_dir(self):
        # The audit repro: corpus/notes -> tmp/outside.
        self._escaped_notes()
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "symlink_escape",
        )

    def test_move_not_crossing_the_link_is_allowed(self):
        # The link EXISTING as a top-level entry must not poison legal
        # moves: this destination never crosses it (the refusal above is
        # about the path, not about the entry existing).
        self._escaped_notes()
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes-copy.md", FULL_KEEP + ["notes"]
        )
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(out["moves"]), 1)

    def test_execute_refuses_and_force_does_not_help(self):
        # --force escapes corpus_changed, never a containment refusal.
        outside = self._escaped_notes()
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "symlink_escape",
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True, force=True),
            "symlink_escape",
        )
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertEqual(list(outside.iterdir()), [])

    def test_dry_run_refuses_too(self):
        self._escaped_notes()
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root),
            "symlink_escape",
        )

    def test_allow_symlinks_executes_the_escape_deliberately(self):
        # The documented bypass: the audit repro MOVES with the flag.
        outside = self._escaped_notes()
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        out = apply_plan(
            plan_path,
            self.root,
            execute=True,
            allow_symlinks=True,
        )
        self.assertEqual(out["moved_count"], 1)
        self.assertFalse((self.root / "acme_dossier.md").exists())
        self.assertTrue((outside / "acme_dossier.md").exists())

    def test_move_onto_internal_symlink_refused(self):
        # Never move ONTO a symlink — even one that lands in the corpus.
        make_symlink(self.root / "docs" / "canonical.md", self.root / "link.md")
        plan_path = _move_plan(self.root, "globex_dossier.md", "link.md", FULL_KEEP)
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "symlink_escape",
        )

    def test_move_onto_symlink_never_clobbers_even_with_flag(self):
        # Two layers, both refuse: the A2 policy names it precisely
        # (symlink_escape, above), and even with --allow-symlinks the
        # shared destination pre-flight (plan_dst_exists) stands — the
        # flag permits TRAVERSAL to a fresh path, never CLOBBER of an
        # existing one (shutil.move onto a symlink would write through
        # the link).
        make_symlink(self.root / "docs" / "canonical.md", self.root / "link.md")
        plan_path = _move_plan(self.root, "globex_dossier.md", "link.md", FULL_KEEP)
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "symlink_escape",
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True, allow_symlinks=True),
            "plan_dst_exists",
        )
        self.assertTrue((self.root / "link.md").is_symlink())

    def test_symlink_pointing_at_the_root_itself_refused(self):
        # Resolves INSIDE the root — still refused: it is a symlinked
        # component, and the policy is default-deny on traversal.
        make_symlink(self.root, self.root / "self")
        plan_path = _move_plan(self.root, "acme_dossier.md", "self/newname.md", [])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "symlink_escape",
        )

    def test_source_under_symlinked_dir_refused(self):
        # Hand-authored source below a symlinked dir: the walk can never
        # list it, so validate refuses the crossing outright.
        outside = self.tmp / "outside"
        outside.mkdir()
        write(outside / "x.txt", "x\n")
        make_symlink(outside, self.root / "mirror")
        plan_path = _move_plan(self.root, "mirror/x.txt", "imported/x.txt", [])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "symlink_escape",
        )

    def test_hardlinked_pair_moves_cleanly(self):
        # A hardlink is not a symlink: no traversal, no refusal — the
        # outside sibling keeps the bytes (same inode, rename-safe).
        hard = self.root / "hardcopy.md"
        hard.hardlink_to(self.root / "acme_dossier.md")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=FULL_KEEP + ["hardcopy.md"],
            folders=[],
            edits=[],
            unresolved=[],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 1)
        self.assertEqual(
            (self.root / "hardcopy.md").read_text(), "# Acme dossier\n\nflows.\n"
        )
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )


class TestCliSymlinkFlags(TempCase):
    def test_apply_cli_refuses_then_executes_with_flag(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        make_symlink(outside, self.root / "notes")
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        r = run_cli("apply", str(plan_path), str(self.root), "--execute")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(json.loads(r.stdout)["error"]["code"], "symlink_escape")
        r = run_cli(
            "apply", str(plan_path), str(self.root), "--execute", "--allow-symlinks"
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((outside / "acme_dossier.md").exists())

    def test_plan_validate_cli_takes_the_flag(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        make_symlink(outside, self.root / "notes")
        plan_path = _move_plan(
            self.root, "acme_dossier.md", "notes/acme_dossier.md", FULL_KEEP
        )
        r = run_cli("plan", "validate", str(plan_path), str(self.root))
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(json.loads(r.stdout)["error"]["code"], "symlink_escape")
        r = run_cli(
            "plan", "validate", str(plan_path), str(self.root), "--allow-symlinks"
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestFingerprintBindsSymlinkedDirs(TempCase):
    """fp6: name + target are content; what is BEHIND the link is not."""

    def test_appearance_and_retargeting_move_the_fingerprint(self):
        before = corpus_fingerprint(self.root)
        outside = self.tmp / "outside"
        outside.mkdir()  # a DANGLING link sorts as a non-file and is
        # pruned by the fp5 regular-file rule — only links to real
        # directories are walk entries.
        make_symlink(outside, self.root / "notes")
        after_add = corpus_fingerprint(self.root)
        self.assertNotEqual(before, after_add)
        other = self.tmp / "elsewhere"
        other.mkdir()
        make_symlink(other, self.root / "notes")  # re-target
        self.assertNotEqual(after_add, corpus_fingerprint(self.root))
        (self.root / "notes").unlink()
        self.assertEqual(corpus_fingerprint(self.root), before)

    def test_contents_behind_the_link_stay_invisible(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        make_symlink(outside, self.root / "notes")
        before = corpus_fingerprint(self.root)
        write(outside / "sneaky.md", "# changed behind the link\n")
        self.assertEqual(corpus_fingerprint(self.root), before)

    def test_prefix_is_fp6(self):
        self.assertTrue(corpus_fingerprint(self.root).startswith("fp6:"))


class TestInventoryNamesSymlinkedDirs(TempCase):
    def _with_link(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        write(outside / "hidden.md", "# not content\n")
        make_symlink(outside, self.root / "notes")

    def test_entry_named_never_descended(self):
        self._with_link()
        inv = collect(self.root, fast=True)
        self.assertEqual(inv["stats"]["symlinked_dirs"], ["notes"])
        self.assertIn("notes", [d["path"] for d in inv["dirs"]])
        self.assertNotIn("notes/hidden.md", [f["path"] for f in inv["files"]])

    def test_plain_corpus_reports_nothing(self):
        inv = collect(self.root, fast=True)
        self.assertEqual(inv["stats"]["symlinked_dirs"], [])
