# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""apply.py — undo_journal: reversal incl. edits and dir pruning."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from helpers import (
    TempCase,
    prune_slopkeeper,
    snapshot_tree,
    write_edit_plan,
    write_plan,
)
from slopkeeper.apply import _prune_empty_dirs, apply_plan, undo_journal
from slopkeeper.models import corpus_fingerprint
from slopkeeper.status import load_manifest, manifest_path


class TestApply(TempCase):
    def test_undo_restores_and_prunes(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        statuses = {o["status"] for o in undone["operations"]}
        # Phase 4: the journal also carries the citation rewrites
        # (WATCHLIST.md cites two moved files) — undo reverts those too.
        self.assertEqual(statuses, {"restored", "reverted"})
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())  # pruned


class TestEdits(TempCase):
    def test_undo_reverts_edit(self):
        plan_path = write_edit_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        statuses = [o.get("status") for o in undone["operations"]]
        self.assertIn("reverted", statuses)
        self.assertIn("acme_dossier.md", (self.root / "WATCHLIST.md").read_text())


class TestUndoStatuses(TempCase):
    def test_blocked_dst_occupied_when_src_exists_again(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "acme_dossier.md").write_text("recreated by hand\n")
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        ops = {o["src"]: o["status"] for o in undone["operations"] if "src" in o}
        self.assertEqual(ops["acme_dossier.md"], "blocked_dst_occupied")
        # The hand-recreated file is never overwritten, dst untouched.
        self.assertEqual(
            (self.root / "acme_dossier.md").read_text(), "recreated by hand\n"
        )
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )

    def test_already_absent_when_dst_gone(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "research" / "dossiers" / "acme_dossier.md").unlink()
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        ops = {o["src"]: o["status"] for o in undone["operations"] if "src" in o}
        self.assertEqual(ops["acme_dossier.md"], "already_absent")

    def test_already_absent_when_edit_file_gone(self):
        plan_path = write_edit_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "WATCHLIST.md").unlink()
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        edit_ops = [o for o in undone["operations"] if "file" in o]
        self.assertEqual(
            edit_ops, [{"file": "WATCHLIST.md", "status": "already_absent"}]
        )

    def test_already_changed_edit_op(self):
        plan_path = write_edit_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "WATCHLIST.md").write_text(
            "# Watchlist\n\nhand-edited since apply\n"
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        edit_ops = [o for o in undone["operations"] if "file" in o]
        self.assertEqual(
            edit_ops, [{"file": "WATCHLIST.md", "status": "already_changed"}]
        )
        # The hand edit is never touched.
        self.assertEqual(
            (self.root / "WATCHLIST.md").read_text(),
            "# Watchlist\n\nhand-edited since apply\n",
        )

    def test_already_changed_rewrite_op(self):
        # Phase 4 rewrite entries: a hand edit that drops the journaled
        # `new` citation must refuse the reversal whole (already_changed)
        # — never a partial or guessed revert — and leave the hand edit.
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        journal = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertIn(
            "WATCHLIST.md",
            {e["file"] for e in journal if e.get("op") == "rewrite"},
            "fixture should produce a rewrite on WATCHLIST.md",
        )
        (self.root / "WATCHLIST.md").write_text(
            "# Watchlist\n\nhand-edited since apply\n"
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        rewrite_ops = [o for o in undone["operations"] if "file" in o]
        self.assertEqual(
            rewrite_ops, [{"file": "WATCHLIST.md", "status": "already_changed"}]
        )
        # The hand edit is never touched.
        self.assertEqual(
            (self.root / "WATCHLIST.md").read_text(),
            "# Watchlist\n\nhand-edited since apply\n",
        )

    def test_undo_of_undo_restores_original_state(self):
        plan_path = write_edit_plan(self.root)
        pristine = prune_slopkeeper(snapshot_tree(self.root))
        out = apply_plan(plan_path, self.root, execute=True)
        first = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertTrue(
            all(
                o.get("status") in {"restored", "reverted"} for o in first["operations"]
            )
        )
        second = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(
            {o.get("status") for o in second["operations"]},
            {"already_absent", "already_changed"},
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), pristine)


class TestPrune(TempCase):
    def test_respects_max_up(self):
        chain = self.root / "keeper" / "a" / "b" / "c"
        chain.mkdir(parents=True)
        _prune_empty_dirs(chain, self.root, max_up=2)
        self.assertFalse(chain.exists())
        self.assertFalse((self.root / "keeper" / "a" / "b").exists())
        self.assertTrue((self.root / "keeper" / "a").exists())  # climb capped
        self.assertTrue((self.root / "keeper").exists())

    def test_stops_at_non_empty_dir(self):
        keeper = self.root / "keeper"
        (keeper / "x" / "y").mkdir(parents=True)
        (keeper / "note.md").write_text("keep me\n")
        _prune_empty_dirs(keeper / "x" / "y", self.root, max_up=4)
        self.assertFalse((keeper / "x" / "y").exists())
        self.assertTrue((keeper / "note.md").exists())
        self.assertEqual((keeper / "note.md").read_text(), "keep me\n")

    def test_never_removes_corpus_root(self):
        # Climb reaches the corpus root: pruned dirs go, the root survives.
        (self.root / "a" / "b").mkdir(parents=True)
        _prune_empty_dirs(self.root / "a" / "b", self.root, max_up=4)
        self.assertFalse((self.root / "a").exists())
        self.assertTrue(self.root.exists())
        # Isolated guard proof: an empty root is never rmdir'ed — the
        # d == root check fires before any removal.
        top = self.tmp / "only-a-dir"
        (top / "a").mkdir(parents=True)
        _prune_empty_dirs(top / "a", top, max_up=4)
        self.assertFalse((top / "a").exists())
        self.assertTrue(top.exists())


class TestManifestRefresh(TempCase):
    """§5.4: `undo --execute` rewrites the manifest; dry-run never does."""

    def test_undo_execute_rewrites_manifest_unresolved_survive(self):
        plan_path = write_plan(
            self.root,
            unresolved=[{"path": "ledger_wb.html", "reason": "scraps without a home"}],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(load_manifest(self.root)["last_plan"], str(plan_path))
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        # The manifest exists and describes the POST-undo disk.
        self.assertTrue(manifest_path(self.root).exists())
        self.assertEqual(undone["manifest"], "updated")
        m = load_manifest(self.root)
        self.assertEqual(m["fingerprint"], corpus_fingerprint(self.root))
        self.assertIn("acme_dossier.md", m["files"])
        # Human decisions are not reverted by tooling...
        self.assertEqual(
            m["unresolved"],
            [{"path": "ledger_wb.html", "reason": "scraps without a home"}],
        )
        # ...but the plan did not happen, so it is no longer last_plan.
        self.assertIsNone(m["last_plan"])
        # Phase 4: rewrite journal entries revert alongside the moves.
        self.assertEqual(
            {o["status"] for o in undone["operations"]}, {"restored", "reverted"}
        )

    def test_undo_without_prior_manifest_writes_fresh(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        manifest_path(self.root).unlink()
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undone["manifest"], "rebuilt")
        m = load_manifest(self.root)
        self.assertIsNone(m["last_plan"])
        self.assertEqual(m["unresolved"], [])
        self.assertEqual(m["fingerprint"], corpus_fingerprint(self.root))
        # Rebuilding the manifest changed nothing about the reversal.
        # Phase 4: rewrite journal entries revert alongside the moves.
        self.assertEqual(
            {o["status"] for o in undone["operations"]}, {"restored", "reverted"}
        )

    def test_dry_run_undo_writes_no_manifest(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        manifest_path(self.root).unlink()
        undone = undo_journal(Path(out["journal"]), self.root)
        self.assertEqual(undone["mode"], "dry-run")
        self.assertNotIn("manifest", undone)
        self.assertFalse(manifest_path(self.root).exists())

    def test_corrupt_manifest_undo_succeeds_and_rebuilds(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        manifest_path(self.root).write_text("{not json", encoding="utf-8")
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undone["manifest"], "rebuilt")
        # Phase 4: rewrite journal entries revert alongside the moves.
        self.assertEqual(
            {o["status"] for o in undone["operations"]}, {"restored", "reverted"}
        )
        m = load_manifest(self.root)  # valid manifest again
        self.assertIsNone(m["last_plan"])
        self.assertEqual(m["unresolved"], [])

    def test_bad_schema_manifest_undo_rebuilds(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        stale = load_manifest(self.root)
        stale["schema"] = "slopkeeper/manifest@1"
        manifest_path(self.root).write_text(json.dumps(stale), encoding="utf-8")
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undone["manifest"], "rebuilt")
        self.assertEqual(load_manifest(self.root)["schema"], "slopkeeper/manifest@2")


if __name__ == "__main__":
    unittest.main()
