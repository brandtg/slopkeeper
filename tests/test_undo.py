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
    write,
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
    """`undo --execute` rewrites the manifest; dry-run never does."""

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


class TestReversalPrecision(TempCase):
    """P6: the auto-undo promise is BYTE-IDENTITY, which needs exact reversal.

    Counting occurrences of `new` (the pre-0.3.0 reversal check) breaks on the
    ordinary sentence "moved acme_dossier.md to research/acme_dossier.md": the
    line already contained the destination, so the check refused, the revert
    was skipped, and the file stayed citing a path undo had just unmoved. The
    journal now carries where each substitution landed, and these fixtures are
    that case.
    """

    def _plan(self, **over) -> Path:
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(self.root),
            "strategy": "hybrid",
            "based_on": {"corpus_fingerprint": corpus_fingerprint(self.root)},
            "moves": [
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "reference pile",
                }
            ],
            "keep": [
                "NOTES.md",
                "WATCHLIST.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "spark-1.2.3",
                "docs",
                "lead-gen",
            ],
            "unresolved": [],
            "folders": [],
        }
        plan.update(over)
        p = self.root.parent / "plan.json"
        p.write_text(json.dumps(plan), encoding="utf-8")
        return p

    def test_rewrite_reverts_a_line_that_already_mentioned_the_destination(self):
        write(
            self.root / "acme_dossier.md",
            "# Acme\n\nflows.\n",
        )
        # The URL is urlish-excluded, so it is NOT a citation and nothing is
        # journaled for it — but it does put the destination string on the
        # line before the tend. Counting occurrences of `new` then sees two
        # where the rewrite introduced one, the old reversal refused, and the
        # file stayed pointing at research/ after undo had unmoved the file.
        note = (
            "See acme_dossier.md, mirrored at "
            "https://x.example/research/acme_dossier.md for the copy.\n"
        )
        write(self.root / "NOTES.md", note)
        before = prune_slopkeeper(snapshot_tree(self.root))
        out = apply_plan(self._plan(), self.root, execute=True)
        self.assertIn(
            "See research/acme_dossier.md,",
            (self.root / "NOTES.md").read_text(),
            "fixture must actually produce the rewrite",
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(
            [o["status"] for o in undone["operations"] if o.get("file") == "NOTES.md"],
            ["reverted"],
            "the rewrite must revert, not be skipped as already_changed",
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)

    def test_edit_reverts_the_occurrence_it_wrote(self):
        # `alpha` already appears in the file BEFORE the edit, so a reversal
        # that takes the first occurrence of the replaced text corrupts the
        # wrong word.
        write(self.root / "acme_dossier.md", "# Acme\n\nflows.\n")
        note = "alpha was here, and the config points at TODO-x\n"
        write(self.root / "NOTES.md", note)
        plan = self._plan(
            edits=[
                {
                    "file": "NOTES.md",
                    "find": "TODO-x",
                    "replace": "alpha",
                    "reason": "anchor edit",
                }
            ]
        )
        out = apply_plan(plan, self.root, execute=True)
        self.assertEqual(
            (self.root / "NOTES.md").read_text(),
            "alpha was here, and the config points at alpha\n",
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(
            [o["status"] for o in undone["operations"] if o.get("file") == "NOTES.md"],
            ["reverted"],
        )
        self.assertEqual((self.root / "NOTES.md").read_text(), note)

    def test_journal_without_offsets_still_reverts_the_old_way(self):
        # Backward compatibility: a 0.2.x journal has no `col`/`at`, and the
        # occurrence-counting reversal is all there is. Unambiguous lines must
        # still come back exactly.
        write(self.root / "acme_dossier.md", "# Acme\n\nflows.\n")
        write(self.root / "NOTES.md", "See acme_dossier.md for flows.\n")
        before = prune_slopkeeper(snapshot_tree(self.root))
        out = apply_plan(self._plan(), self.root, execute=True)
        jpath = Path(out["journal"])
        lines = [json.loads(x) for x in jpath.read_text().splitlines() if x.strip()]
        for entry in lines:
            for rep in entry.get("replacements") or []:
                rep.pop("col", None)
            entry.pop("at", None)
        jpath.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
        undone = undo_journal(jpath, self.root, execute=True)
        self.assertEqual(
            [o["status"] for o in undone["operations"] if o.get("file") == "NOTES.md"],
            ["reverted"],
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)

    def test_offsets_are_what_make_the_ambiguous_line_revertible(self):
        # Same collision as above, WITHOUT the offsets: the counting check is
        # conservative and refuses (already_changed) rather than revert the
        # wrong occurrence. This is the hole the `col` field closes, and the
        # reason a refused revert is the only safe answer when the journal
        # cannot say where the substitution landed.
        write(self.root / "acme_dossier.md", "# Acme\n\nflows.\n")
        write(
            self.root / "NOTES.md",
            "See acme_dossier.md, mirrored at "
            "https://x.example/research/acme_dossier.md for the copy.\n",
        )
        out = apply_plan(self._plan(), self.root, execute=True)
        jpath = Path(out["journal"])
        lines = [json.loads(x) for x in jpath.read_text().splitlines() if x.strip()]
        for entry in lines:
            for rep in entry.get("replacements") or []:
                rep.pop("col", None)
        jpath.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
        undone = undo_journal(jpath, self.root, execute=True)
        self.assertEqual(
            [o["status"] for o in undone["operations"] if o.get("file") == "NOTES.md"],
            ["already_changed"],
        )
        self.assertIn(
            "https://x.example/research/acme_dossier.md",
            (self.root / "NOTES.md").read_text(),
            "a refused revert must leave the file exactly as it was",
        )


if __name__ == "__main__":
    unittest.main()
