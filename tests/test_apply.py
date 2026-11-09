# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""apply.py — execution, journal, atomicity (incl. in-plan edits)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import unittest
import unittest.mock
from pathlib import Path

from helpers import (
    TempCase,
    build_corpus,
    expect_code,
    prune_slopkeeper,
    snapshot_tree,
    write,
    write_edit_plan,
    write_plan,
)
from slopkeeper import cli
from slopkeeper.apply import _execution_order, apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError, corpus_fingerprint
from slopkeeper.plan import load_plan, validate_plan


class TestChainExecutionOrder(unittest.TestCase):
    """_execution_order: hops ordered src-side first, ties keep sorted."""

    def test_two_hop_chain_runs_dst_free_move_first(self):
        # a→b must wait for b→c: b→c frees the path a→b lands on.
        # Rewriting the chain instead is what destroyed b.md's content.
        order = _execution_order({"a.md": "b.md", "b.md": "c.md"})
        self.assertEqual(order, [("b.md", "c.md"), ("a.md", "b.md")])

    def test_three_hop_chain_fully_ordered(self):
        order = _execution_order({"a.md": "b.md", "b.md": "c.md", "c.md": "d.md"})
        self.assertEqual(order, [("c.md", "d.md"), ("b.md", "c.md"), ("a.md", "b.md")])

    def test_unconstrained_moves_keep_sorted_order(self):
        order = _execution_order({"a.md": "b.md", "b.md": "c.md", "x.md": "y.md"})
        # b→c and x→y are ready in the first pass (sorted order kept);
        # a→b waits one pass for b→c.
        self.assertEqual(order, [("b.md", "c.md"), ("x.md", "y.md"), ("a.md", "b.md")])


class TestApply(TempCase):
    def test_dry_run_default_touches_nothing(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root)
        self.assertEqual(out["mode"], "dry-run")
        self.assertTrue((self.root / "acme_dossier.md").exists())
        # B2: a covered file is not a risk — the rewrite pass rewrites
        # WATCHLIST.md's citations, so the dossier moves carry no
        # uncovered reference and the report is empty.
        self.assertEqual(out["reference_risks"], {})
        # ... and the covering rewrites are visible, line by line (B1).
        watch = next(r for r in out["would_rewrite"] if r["file"] == "WATCHLIST.md")
        self.assertEqual(watch["count"], 2)
        self.assertIn(
            {
                "line": 3,
                "old": "acme_dossier.md",
                "new": "research/dossiers/acme_dossier.md",
            },
            watch["lines"],
        )

    def test_execute_moves_and_journals(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 4)
        self.assertFalse((self.root / "acme_dossier.md").exists())
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )
        self.assertTrue(Path(out["journal"]).exists())

    def test_stale_fingerprint_refused(self):
        plan_path = write_plan(self.root)
        (self.root / "WATCHLIST.md").write_text("mutated\n")
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "corpus_changed",
        )
        # force overrides
        out = apply_plan(plan_path, self.root, execute=True, force=True)
        self.assertEqual(out["moved_count"], 4)

    def test_existing_destination_refused(self):
        (self.root / "archive" / "scraps").mkdir(parents=True)
        (self.root / "archive" / "scraps" / "ledger_wb.html").write_text("x")
        plan_path = write_plan(self.root)  # fingerprint computed AFTER mutation
        expect_code(
            self, lambda: apply_plan(plan_path, self.root, execute=True), "dst_exists"
        )
        # nothing moved (all-or-nothing pre-flight)
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_strict_references_aborts(self):
        # rewrites mode "off": with the default pass on, the WATCHLIST
        # citation would be rewritten and the risk satisfied
        # (test_strict_references_satisfied_by_rewrites) — opting out
        # preserves the bare abort contract.
        plan_path = write_plan(self.root, rewrites={"mode": "off"})
        expect_code(
            self,
            lambda: apply_plan(
                plan_path, self.root, execute=True, strict_references=True
            ),
            "referenced_moves",
        )

    def test_dir_move_excludes_descendant_citers(self):
        # Phase 2: a file under a moved directory that cites a sibling
        # under the same dir is rewritten by this apply — not breakage.
        # (Before the path-keyed scan, dir descendants weren't in the
        # moved set and every intra-dir citation looked like a risk.)
        write(self.root / "reports/_sync/a.md", "# a\n")
        write(self.root / "reports/_sync/b.md", "see a.md for the spec\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "reports/_sync/",
                    "to": "reports/sync/",
                    "reason": "rename",
                }
            ],
            keep=[
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
            ],
            unresolved=[],
        )
        out = apply_plan(plan_path, self.root, execute=False)
        self.assertEqual(out["reference_risks"], {})


class TestEdits(TempCase):
    def test_edit_rewrites_after_move(self):
        plan_path = write_edit_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        body = (self.root / "WATCHLIST.md").read_text()
        self.assertIn("research/acme_dossier.md", body)
        self.assertNotIn(
            " acme_dossier.md", body.replace("research/acme_dossier.md", "")
        )
        # Risks resolved by the in-plan edit: none reported.
        self.assertEqual(out["reference_risks"], {})

    def test_edit_stale_rejected(self):
        plan_path = write_edit_plan(self.root)
        # Break the find text: corpus drifts after plan authoring.
        (self.root / "WATCHLIST.md").write_text("# Watchlist\n\nno refs now\n")
        # force=True: the drift is ours and expected; the edit check still runs.
        expect_code(
            self, lambda: apply_plan(plan_path, self.root, force=True), "edit_stale"
        )

    def test_edit_ambiguous_rejected(self):
        (self.root / "WATCHLIST.md").write_text(
            "# Watchlist\n\nSee acme_dossier.md and acme_dossier.md.\n"
        )
        plan_path = write_edit_plan(self.root)
        expect_code(
            self, lambda: apply_plan(plan_path, self.root, force=True), "edit_ambiguous"
        )

    def _citation_edit_plan(self, name: str, encoding: str) -> Path:
        # A citing file at notes/<name> whose bytes are NOT utf-8 when
        # encoding="latin-1" (é is a lone 0xE9); the find text is ASCII
        # and present exactly once, so the encoding is the only thing
        # that can fail. The plan is authored AFTER the file exists so
        # the fingerprint is fresh and the edit pre-flight is the sole
        # gate under test.
        target = self.root / "notes" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes("Café notes: see acme_dossier.md\n".encode(encoding))
        return write_edit_plan(
            self.root,
            keep=[
                "WATCHLIST.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "spark-1.2.3",
                "docs",
                "lead-gen",
            ],
            edits=[
                {
                    "file": f"notes/{name}",
                    "find": "acme_dossier.md",
                    "replace": "research/acme_dossier.md",
                    "reason": "fix citation",
                }
            ],
        )

    def test_edit_not_utf8_refused_dry_run(self):
        # The old errors="replace" read silently U+FFFD'd every
        # undecodable byte and would have re-encoded the whole file.
        plan_path = self._citation_edit_plan("legacy_iso.md", "latin-1")
        before = prune_slopkeeper(snapshot_tree(self.root))
        expect_code(self, lambda: apply_plan(plan_path, self.root), "edit_not_utf8")
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)

    def test_edit_not_utf8_refused_execute(self):
        # All-or-nothing: the refusal happens in the pre-flight, BEFORE
        # any move — the disk stays byte-identical.
        plan_path = self._citation_edit_plan("legacy_iso.md", "latin-1")
        before = prune_slopkeeper(snapshot_tree(self.root))
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "edit_not_utf8",
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

    def test_utf8_edit_still_applies(self):
        # Control: the same edit on a UTF-8 twin applies normally.
        plan_path = self._citation_edit_plan("legacy_utf8.md", "utf-8")
        target = self.root / "notes" / "legacy_utf8.md"
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        body = target.read_text(encoding="utf-8")
        self.assertIn("research/acme_dossier.md", body)
        self.assertIn("Café notes", body)


class TestAtomicity(TempCase):
    """All-or-nothing: a refused plan leaves the disk byte-identical."""

    def test_bad_edit_among_good_moves_moves_nothing(self):
        # Drift the citing file first, THEN author the plan, so the
        # fingerprint is fresh and only the edit pre-flight can fail.
        (self.root / "WATCHLIST.md").write_text("# Watchlist\n\nno refs now\n")
        plan_path = write_edit_plan(self.root)
        before = prune_slopkeeper(snapshot_tree(self.root))
        expect_code(
            self, lambda: apply_plan(plan_path, self.root, execute=True), "edit_stale"
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        # The good move's file is still at its source; no dst dirs appeared.
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

    def test_stale_src_moves_nothing(self):
        plan_path = write_plan(self.root)
        (self.root / "ledger_wb.html").unlink()  # a move src vanishes
        before = prune_slopkeeper(snapshot_tree(self.root))
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "plan_stale_src",
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())


class TestForceGate(TempCase):
    def test_force_overrides_stale_fingerprint_and_executes(self):
        plan_path = write_plan(self.root)
        (self.root / "spark-1.2.3" / "README.md").write_text("# runbook v2\n")
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "corpus_changed",
        )
        out = apply_plan(plan_path, self.root, execute=True, force=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["moved_count"], 4)
        self.assertFalse((self.root / "acme_dossier.md").exists())
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )


class TestContentHashGate(TempCase):
    """§5.2: staleness is content-based, not timestamp-based."""

    def test_touch_does_not_invalidate_plan(self):
        # Flipped pin (was: touch → corpus_changed). A bare mtime bump
        # on unchanged content must not force a re-plan.
        plan_path = write_plan(self.root)
        p = self.root / "spark-1.2.3" / "README.md"
        os.utime(p, (1_000_000_000, 1_000_000_000))
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["moved_count"], 4)

    def test_same_size_edit_raises_corpus_changed(self):
        # The false-negative hole, now closed: a same-size content edit
        # to a file the plan was authored against is refused.
        plan_path = write_plan(self.root)
        p = self.root / "spark-1.2.3" / "README.md"
        p.write_bytes(p.read_bytes().replace(b"runbook", b"runbuok"))
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "corpus_changed",
        )

    def test_v1_fingerprint_fails_cleanly(self):
        # A plan authored against a v1 fingerprint (no fp2: prefix) can
        # never collide with the v2 format: it is refused, not applied.
        plan_path = write_plan(
            self.root,
            based_on={"corpus_fingerprint": "0123456789abcdef01234567:8"},
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "corpus_changed",
        )


class TestEditTargets(TempCase):
    def test_edit_on_moving_file_preflights_at_source_applies_at_dst(self):
        # Flipped pin (Fix 1, src round). edit.file is the move-SOURCE
        # path (AGENT_GUIDE.md), so the pre-flight reads the file at its
        # PRE-move location; the edit then applies post-move at the
        # final destination resolved through the chains. Journaled and
        # undo-able end to end.
        plan_path = write_edit_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            edits=[
                {
                    "file": "acme_dossier.md",
                    "find": "flows",
                    "replace": "STREAMS",
                    "reason": "edit a moved file",
                }
            ],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "spark-1.2.3",
                "docs",
                "lead-gen",
                "WATCHLIST.md",
            ],
        )
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertIn("flows", (self.root / "acme_dossier.md").read_text())
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        # (a) Pre-flighted at the source: the apply succeeded ...
        self.assertFalse((self.root / "acme_dossier.md").exists())
        # (b) ... and the edit landed at the final path.
        dst = self.root / "research" / "acme_dossier.md"
        self.assertTrue(dst.exists())
        self.assertIn("STREAMS", dst.read_text())
        self.assertNotIn("flows", dst.read_text())
        # (c) Journaled with the final path, and undo restores the source.
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        edit_ops = [e for e in lines if e.get("op") == "edit"]
        self.assertEqual(len(edit_ops), 1)
        self.assertEqual(edit_ops[0]["file"], "research/acme_dossier.md")
        self.assertEqual(edit_ops[0]["find"], "flows")
        self.assertEqual(edit_ops[0]["replace"], "STREAMS")
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undo["mode"], "executed")
        reverted = [op for op in undo["operations"] if "file" in op]
        # Phase 4: the WATCHLIST.md citation of the moved file was also
        # rewritten (and journaled), so undo reverts it alongside the edit.
        self.assertEqual(
            reverted,
            [
                {"file": "research/acme_dossier.md", "status": "reverted"},
                {"file": "WATCHLIST.md", "status": "reverted"},
            ],
        )
        self.assertIn("flows", (self.root / "acme_dossier.md").read_text())

    def test_stale_edit_on_moving_file_moves_nothing(self):
        # Atomicity holds when the moving edit target's find text is
        # gone: the pre-flight (at the source) fails and NOTHING moves.
        (self.root / "acme_dossier.md").write_text("# drifted\n")
        plan_path = write_edit_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            edits=[
                {
                    "file": "acme_dossier.md",
                    "find": "flows",
                    "replace": "STREAMS",
                    "reason": "edit a moved file",
                }
            ],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "spark-1.2.3",
                "docs",
                "lead-gen",
                "WATCHLIST.md",
            ],
        )
        before = prune_slopkeeper(snapshot_tree(self.root))
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "edit_stale")
        # The find text is missing at the SOURCE — not a missing file.
        self.assertIn("no longer contains", cm.exception.message)
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

    def test_edit_on_stay_put_file_applies_at_current_path(self):
        plan_path = write_plan(
            self.root,
            edits=[
                {
                    "file": "lead-gen/README.md",
                    "find": "pipeline",
                    "replace": "funnel",
                    "reason": "in-place rewrite",
                }
            ],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 4)
        stay = self.root / "lead-gen" / "README.md"
        self.assertTrue(stay.exists())  # never moved
        self.assertIn("funnel", stay.read_text())
        self.assertNotIn("pipeline", stay.read_text())


class TestPhase1EditResolution(TempCase):
    """D9: edit targets resolve through directory moves — one helper
    (_edit_final_path) behind validate, the dry run and execute."""

    def _dir_move_edit_plan(self, **overrides):
        write(
            self.root / "misc/nvidia/2026-08-24-dan-nvidia-reply.md",
            "# Dan reply\n\nSee the nvidia reply thread.\n",
        )
        base = dict(
            moves=[
                {
                    "from": "misc/nvidia",
                    "to": "projects/nvidia",
                    "reason": "restructure",
                }
            ],
            edits=[
                {
                    "file": "misc/nvidia/2026-08-24-dan-nvidia-reply.md",
                    "find": "nvidia reply thread",
                    "replace": "nvidia response thread",
                    "reason": "edit a dir-move descendant",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
            ],
            unresolved=[],
            folders=[],
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_edit_inside_moved_dir_applies_at_destination(self):
        # D9's exact shape: the edit's file is the move-SOURCE path of a
        # file INSIDE a moved directory. Validate and the dry run always
        # passed it; execute used to read the stale pre-move path after
        # the dir moved (edit_unreadable, partial: true). Now the edit
        # resolves through the directory-move prefix and lands at the
        # destination.
        plan_path = self._dir_move_edit_plan()
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertFalse(out["partial"])
        self.assertEqual(out["errors"], [])
        dst = self.root / "projects" / "nvidia" / "2026-08-24-dan-nvidia-reply.md"
        self.assertTrue(dst.exists())
        self.assertIn("nvidia response thread", dst.read_text())
        self.assertNotIn("nvidia reply thread", dst.read_text())
        # Journaled under the final path, so undo replays it exactly.
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        edit_ops = [e for e in lines if e.get("op") == "edit"]
        self.assertEqual(
            edit_ops[0]["file"], "projects/nvidia/2026-08-24-dan-nvidia-reply.md"
        )
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        reverted = [op for op in undo["operations"] if op.get("status") == "reverted"]
        self.assertEqual(
            reverted,
            [
                {
                    "file": "projects/nvidia/2026-08-24-dan-nvidia-reply.md",
                    "status": "reverted",
                }
            ],
        )

    def test_edit_resolution_parity(self):
        # Chain keys and directory-move descendants resolve identically,
        # and the dry run names the same target execute applies to.
        from slopkeeper.apply import _edit_final_path

        chains = {
            "acme_dossier.md": "research/acme_dossier.md",
            "misc/nvidia": "projects/nvidia",
        }
        self.assertEqual(
            _edit_final_path("acme_dossier.md", chains), "research/acme_dossier.md"
        )
        self.assertEqual(
            _edit_final_path("misc/nvidia/2026-08-24-dan-nvidia-reply.md", chains),
            "projects/nvidia/2026-08-24-dan-nvidia-reply.md",
        )
        # A file nowhere in the plan is returned unchanged.
        self.assertEqual(_edit_final_path("WATCHLIST.md", chains), "WATCHLIST.md")
        # Dry run and execute agree on the target for the D9 shape.
        plan_path = self._dir_move_edit_plan()
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(
            dry["would_edit"][0]["file"],
            "projects/nvidia/2026-08-24-dan-nvidia-reply.md",
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertFalse(out["partial"])
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        edit_ops = [e for e in lines if e.get("op") == "edit"]
        self.assertEqual(
            edit_ops[0]["file"],
            dry["would_edit"][0]["file"],
            "execute must apply the edit at the same path the dry run names",
        )


class TestPhase3WouldEdit(TempCase):
    """D8: would_edit rows carry the resolved path, the owned line and
    the before/after text of the edited line."""

    def _dir_edit_plan(self, **overrides):
        write(
            self.root / "misc/nvidia/reply.md",
            "intro line\nsee the nvidia reply thread today\n",
        )
        base = dict(
            moves=[
                {
                    "from": "misc/nvidia",
                    "to": "projects/nvidia",
                    "reason": "restructure",
                }
            ],
            edits=[
                {
                    "file": "misc/nvidia/reply.md",
                    "find": "reply thread",
                    "replace": "response thread",
                    "reason": "d8 detail",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
            ],
            unresolved=[],
            folders=[],
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_would_edit_carries_line_before_after(self):
        dry = apply_plan(self._dir_edit_plan(), self.root)
        row = dry["would_edit"][0]
        self.assertEqual(row["file"], "projects/nvidia/reply.md")
        self.assertEqual(row["line"], 2)
        self.assertIn("nvidia reply thread", row["before"])
        self.assertNotIn("response thread", row["before"])
        self.assertIn("nvidia response thread", row["after"])
        self.assertNotIn("nvidia reply thread", row["after"])
        self.assertEqual(row["reason"], "d8 detail")

    def test_would_edit_before_is_the_post_rewrite_view(self):
        # The edit's find exists only in POST-rewrite text (C4's second
        # form): `before` must show the line as the edit sees it — the
        # pass's own replacement applied — not the raw disk text.
        write(
            self.root / "notes/x.md", "cites reports/_org/conv.md here\nsecond line\n"
        )
        write(self.root / "reports/_org/keep.md", "keep\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ],
            edits=[
                {
                    "file": "notes/x.md",
                    "find": "reports/org/conv.md",
                    "replace": "reports/org/renamed.md",
                    "reason": "authored post-rewrite",
                }
            ],
            keep=[
                "lead-gen",
                "spark-1.2.3",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "docs",
                "notes",
                "reports",
            ],
            unresolved=[],
            folders=[],
        )
        dry = apply_plan(plan_path, self.root)
        row = next(r for r in dry["would_edit"] if r["file"] == "notes/x.md")
        self.assertEqual(row["line"], 1)
        self.assertIn("reports/org/conv.md", row["before"])
        self.assertNotIn("reports/_org/", row["before"])
        self.assertIn("reports/org/renamed.md", row["after"])

    def test_would_edit_keeps_reason_key(self):
        dry = apply_plan(self._dir_edit_plan(), self.root)
        for row in dry["would_edit"]:
            self.assertIn("reason", row)


class TestPhase5Expansion(TempCase):
    """D2/D11: glob and brace tokens expand against the corpus — a
    rewrite only when every match moves into one destination; a skip
    row otherwise; moved-dir prefixes rewrite whatever follows."""

    def _glob_plan(
        self,
        dst_b="projects/vision-pipeline/2026-09-17_two-stage-pipeline-b.md",
        prefix_rewrites=None,
    ):
        write(self.root / "reports/_org/2026-09-17_two-stage-pipeline-a.md", "# a\n")
        write(self.root / "reports/_org/2026-09-17_two-stage-pipeline-b.md", "# b\n")
        write(self.root / "reports/_org/other.md", "# other\n")
        write(
            self.root / "notes/team-sync.md",
            "see reports/_org/2026-09-17_two-stage-pipeline*.md\n",
        )
        base = dict(
            moves=[
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-a.md",
                    "to": "projects/vision-pipeline/2026-09-17_two-stage-pipeline-a.md",
                    "reason": "vision",
                },
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-b.md",
                    "to": dst_b,
                    "reason": "vision",
                },
            ],
            prefix_rewrites=prefix_rewrites
            or [{"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
                "reports",
            ],
            unresolved=[],
            folders=[],
        )
        return write_plan(self.root, **base)

    def test_d2_glob_expands_to_single_destination(self):
        dry = apply_plan(self._glob_plan(), self.root)
        per_file = {p["file"]: p for p in dry["would_rewrite"]}
        self.assertIn("notes/team-sync.md", per_file)
        new_texts = [r["new"] for r in per_file["notes/team-sync.md"]["lines"]]
        self.assertIn(
            "projects/vision-pipeline/2026-09-17_two-stage-pipeline*.md",
            new_texts,
            "the glob must follow its matches' single destination dir",
        )

    def test_d2_glob_split_skips_with_row(self):
        # The entry is consciously acknowledged (validate's dead-glob
        # gate would refuse it: the glob's matches split); the apply-time
        # pass still refuses the rewrite loudly with a hand-edit row.
        dry = apply_plan(
            self._glob_plan(
                dst_b="other/place/2026-09-17_two-stage-pipeline-b.md",
                prefix_rewrites=[
                    {
                        "from": "reports/_org/",
                        "to": "reports/org/",
                        "reason": "renamed",
                        "acknowledge_broken": True,
                    }
                ],
            ),
            self.root,
        )
        rows = dry.get("skip_rewrites") or []
        self.assertTrue(
            any(
                r["token"] == "reports/_org/2026-09-17_two-stage-pipeline*.md"
                and r["reason"] == "glob_split"
                for r in rows
            ),
            f"split glob must skip with a row, got {rows}",
        )
        per_file = {p["file"]: p for p in dry.get("would_rewrite", [])}
        self.assertNotIn(
            "notes/team-sync.md",
            per_file,
            "a split glob must never be rewritten",
        )

    def test_d11_brace_token_under_moved_dir_rewrites(self):
        for n in (1, 2, 3):
            write(
                self.root / f"misc/actions-offsite/convo/actions-offsite-a-{n}.txt",
                "x\n",
            )
        write(
            self.root / "notes/agenda.md",
            "listen to misc/actions-offsite/convo/actions-offsite-a-{1,2,3}.txt\n",
        )
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "misc/actions-offsite",
                    "to": "projects/actions-offsite",
                    "reason": "restructure",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
            ],
            unresolved=[],
            folders=[],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        agenda = (self.root / "notes" / "agenda.md").read_text()
        self.assertIn(
            "projects/actions-offsite/convo/actions-offsite-a-{1,2,3}.txt", agenda
        )

    def test_d11_shorthand_prefix_token_rewrites(self):
        write(self.root / "mcp-offering/epic/whatever.md", "# epic\n")
        write(
            self.root / "notes/agenda.md",
            "tracked in mcp-offering/epic/external-00/01/02\n",
        )
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "mcp-offering",
                    "to": "projects/mcp-offering",
                    "reason": "restructure",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
            ],
            unresolved=[],
            folders=[],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        agenda = (self.root / "notes" / "agenda.md").read_text()
        self.assertIn("projects/mcp-offering/epic/external-00/01/02", agenda)

    def test_no_dead_glob_rewrites(self):
        # Across both glob fixtures, no rewritten line names a pattern
        # whose directory holds none of the glob's post-move files.
        dry = apply_plan(self._glob_plan(), self.root)
        for p in dry.get("would_rewrite", []):
            for r in p["lines"]:
                if "*" in r["new"]:
                    new_dir = r["new"].rsplit("/", 1)[0]
                    self.assertEqual(new_dir, "projects/vision-pipeline")


class TestPhase7CitingDirResolution(TempCase):
    """D4: relative tokens resolve from the citing file's folder first —
    no more ambiguous_basename floods for intra-dir relative links."""

    def _proj_corpus(self):
        write(self.root / "proj/notes.md", "see tokens/colors.css here\n")
        write(self.root / "proj/tokens/colors.css", "# colors\n")

    def _proj_plan(self, **overrides):
        base = dict(
            moves=[{"from": "proj", "to": "projects/proj", "reason": "restructure"}],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
            ],
            unresolved=[],
            folders=[],
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_citing_dir_candidate_wins(self):
        # Both files move with the dir: the citation is no_change —
        # no ambiguous rows, no rewrite, no risk.
        self._proj_corpus()
        plan_path = self._proj_plan()
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(dry.get("skip_rewrites"), [])
        self.assertEqual(dry["reference_risks"], {})
        self.assertNotIn(
            "notes.md",
            [p["file"] for p in dry.get("would_rewrite", [])],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        # No churn: the intra-dir link still names its (co-moved) file.
        self.assertIn(
            "tokens/colors.css",
            (self.root / "projects/proj/notes.md").read_text(),
        )

    def test_citing_dir_stay_put_produces_no_rows(self):
        # The sibling exists beside the citer and nothing moves: the
        # citation resolves (rel_path) and produces no rows at all.
        self._proj_corpus()
        plan_path = self._proj_plan(
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "unrelated",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "proj",
            ],
        )
        dry = apply_plan(plan_path, self.root)
        skipped_tokens = [r["token"] for r in dry.get("rewrite_skipped", [])]
        self.assertNotIn("tokens/colors.css", skipped_tokens)
        per_file = {p["file"]: p for p in dry.get("would_rewrite", [])}
        self.assertNotIn("proj/notes.md", per_file)

    def test_ambiguous_without_citing_dir_candidate(self):
        # No sibling beside the citer: the basename fallback still
        # reports the mention ambiguous (never attributed).
        for d in ("a", "b", "c", "d"):
            write(self.root / f"{d}/tokens/colors.css", "# colors\n")
        write(self.root / "citer.md", "see tokens/colors.css here\n")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "a", "to": "projects/a", "reason": "restructure"}],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "citer.md",
                "b",
                "c",
                "d",
            ],
            unresolved=[],
            folders=[],
        )
        dry = apply_plan(plan_path, self.root)
        rows = [
            r
            for r in dry.get("rewrite_skipped", [])
            if r["token"] == "tokens/colors.css"
        ]
        self.assertTrue(
            any(r["reason"] == "ambiguous_basename" for r in rows),
            f"no citing-dir candidate: must stay ambiguous, got {rows}",
        )

    def test_relative_form_unchanged(self):
        # An explicit ./ form resolves from the citing dir first and
        # keeps its form when both ends move together (C2: no churn).
        write(self.root / "proj/notes.md", "see ./tokens/colors.css here\n")
        write(self.root / "proj/tokens/colors.css", "# colors\n")
        plan_path = self._proj_plan()
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        self.assertIn(
            "./tokens/colors.css",
            (self.root / "projects/proj/notes.md").read_text(),
        )


class TestPhase8BackticksAndNotRecorded(TempCase):
    """D6: backtick spans with spaces resolve and rewrite; not_recorded
    risk rows carry the line (and token) instead of line: null."""

    def test_spaced_filename_shape_rewrites(self):
        write(
            self.root / "misc/notes/Sample Transcript - Owner's Phone .md", "# t\n"
        )
        write(
            self.root / "reports/1-1/peer/2026-09-29-agenda.md",
            "# agenda\ntranscript: `notes/Sample Transcript - Owner's Phone .md`\n",
        )
        plan_path = write_plan(
            self.root,
            moves=[{"from": "misc", "to": "projects", "reason": "restructure"}],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "reports",
            ],
            unresolved=[],
            folders=[],
        )
        dry = apply_plan(plan_path, self.root)
        per_file = {p["file"]: p for p in dry["would_rewrite"]}
        self.assertIn("reports/1-1/peer/2026-09-29-agenda.md", per_file)
        new_texts = [
            r["new"]
            for r in per_file["reports/1-1/peer/2026-09-29-agenda.md"]["lines"]
        ]
        self.assertIn(
            "projects/notes/Sample Transcript - Owner's Phone .md", new_texts
        )
        # Not a skip: the citation is recorded and fixed.
        self.assertEqual(dry.get("skip_rewrites"), [])

    def test_not_recorded_rows_carry_line(self):
        write(self.root / "docs/target.md", "# target\n")
        write(
            self.root / "WATCHLIST.md",
            "# Watchlist\n\nSee docs/target.md for the flows.\n",
        )
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "docs/target.md",
                    "to": "archive/target.md",
                    "reason": "archive",
                }
            ],
            rewrites={"mode": "off"},
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
            ],
            unresolved=[],
            folders=[],
        )
        dry = apply_plan(plan_path, self.root)
        rows = (dry["reference_risks"].get("docs/target.md") or {}).get("citations", [])
        nr = [c for c in rows if c.get("status") == "not_recorded"]
        self.assertTrue(nr, f"mode off must yield not_recorded rows, got {rows}")
        self.assertEqual(nr[0]["line"], 3)
        self.assertEqual(nr[0]["token"], "docs/target.md")


class TestPhase9OwnedLines(TempCase):
    """D1: a citation on an edit-owned line is edit_covered only when
    its span overlaps the find's first occurrence; everything else on
    the line is edit_owned_uncovered — loud, never silently dropped."""

    def _plan(self):
        write(self.root / "target.md", "# target\n")
        write(
            self.root / "citer.md",
            "see target.md and target.md again\nunrelated line\n",
        )
        return write_plan(
            self.root,
            moves=[
                {
                    "from": "target.md",
                    "to": "deep/target.md",
                    "reason": "cited file moves",
                }
            ],
            edits=[
                {
                    "file": "citer.md",
                    "find": "see target.md",
                    "replace": "see deep/target.md",
                    "reason": "covers the first citation only",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "citer.md",
            ],
            unresolved=[],
            folders=[],
        )

    def test_owned_line_second_citation_reported(self):
        dry = apply_plan(self._plan(), self.root)
        rows = [
            r
            for r in dry.get("skip_rewrites", [])
            if r.get("reason") == "edit_owned_uncovered"
        ]
        self.assertEqual(len(rows), 1, f"one uncovered citation, got {rows}")
        self.assertEqual(rows[0]["file"], "citer.md")
        self.assertEqual(rows[0]["line"], 1)
        self.assertEqual(rows[0]["token"], "target.md")

    def test_owned_line_first_occurrence_covered(self):
        dry = apply_plan(self._plan(), self.root)
        citations = (dry["reference_risks"].get("target.md") or {}).get("citations", [])
        statuses = {c["status"] for c in citations if c.get("file") == "citer.md"}
        self.assertEqual(
            statuses,
            {"edit_covered", "edit_owned_uncovered"},
            f"first occurrence covered, second reported: {citations}",
        )

    def test_owned_line_uncovered_surfaces_as_risk(self):
        dry = apply_plan(self._plan(), self.root)
        risk = dry["reference_risks"].get("target.md") or {}
        self.assertEqual(risk.get("uncovered_references"), 1)
        # And the render's hand-edit list shows it.
        self.assertIn("edit_owned_uncovered", json.dumps(dry))


class TestPhase10SkipList(TempCase):
    """D3: skip_rewrites is the hand-edit list — bare ambiguous mentions
    stay in `skipped` only; everything actionable stays listed."""

    def _mention_plan(self, **overrides):
        write(self.root / "docs/AGENTS.md", "# one\n")
        write(self.root / "misc/AGENTS.md", "# two\n")
        write(self.root / "tools/AGENTS.md", "# three\n")
        write(self.root / "citer.md", "see AGENTS.md everywhere\n")
        base = dict(
            moves=[{"from": "misc", "to": "projects/misc", "reason": "restructure"}],
            keep=[
                "docs",
                "tools",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "citer.md",
            ],
            unresolved=[],
            folders=[],
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_skip_rewrites_excludes_stay_put_basenames(self):
        dry = apply_plan(self._mention_plan(), self.root)
        rows = [r for r in dry.get("skip_rewrites", []) if r["token"] == "AGENTS.md"]
        self.assertEqual(
            rows,
            [],
            f"a bare ambiguous mention is not hand-editable, got {rows}",
        )
        # The audit trail keeps it: skipped carries token and line.
        skipped = [
            r
            for r in dry.get("rewrite_skipped", [])
            if r["token"] == "AGENTS.md" and r["file"] == "citer.md"
        ]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["line"], 1)
        self.assertEqual(skipped[0]["reason"], "ambiguous_basename")

    def test_skip_rewrites_excludes_resolvable_stay_put_links(self):
        # A relative link that resolves (from the citing dir) to a
        # stay-put file produces no rows at all (D4 + D3).
        write(self.root / "proj/notes.md", "see ./skills/x/SKILL.md today\n")
        write(self.root / "proj/skills/x/SKILL.md", "# skill\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "unrelated",
                }
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "proj",
            ],
            unresolved=[],
            folders=[],
        )
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(dry.get("skip_rewrites"), [])
        per_file = {p["file"]: p for p in dry.get("would_rewrite", [])}
        self.assertNotIn(
            "proj/notes.md",
            per_file,
            "a stay-put relative link produces no rewrite and no row",
        )

    def test_prefix_conflict_still_listed(self):
        # A basename mention inside a longer token that resolves to
        # nothing (lib/api.md/x — lib does not move): two live readings
        # — the span is skipped AND listed for hand-editing.
        write(self.root / "docs/api.md", "# api\n")
        write(self.root / "w.md", "see lib/api.md/x and docs/api.md today\n")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            unresolved=[],
            folders=[],
            partial=True,
        )
        dry = apply_plan(plan_path, self.root)
        rows = [
            r for r in dry.get("skip_rewrites", []) if r["reason"] == "prefix_conflict"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], "w.md")
        self.assertEqual(rows[0]["token"], "api.md")

    def test_stale_paths_still_listed(self):
        # The 0.1.9 shape: bare mentions drop out, a real stale path
        # (unresolved_path) stays in the hand-edit list. The stale path
        # sits outside the moved prefix — a token UNDER the moved dir
        # follows the rename instead (D11).
        write(self.root / "misc/email.md", "# email\n")
        write(
            self.root / "w.md",
            "see AGENTS.md everywhere\nthe thread: notes/email.md (acme, 8/26)\n",
        )
        plan_path = self._mention_plan(
            keep=[
                "docs",
                "tools",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "citer.md",
                "w.md",
            ],
        )
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(
            [r["reason"] for r in dry.get("skip_rewrites", [])],
            ["unresolved_path"],
            f"only the actionable stale path is listed, got {dry.get('skip_rewrites')}",
        )
        rows = [r for r in dry.get("skip_rewrites", [])]
        self.assertEqual(rows[0]["token"], "notes/email.md")
        self.assertEqual(rows[0]["line"], 2)


class TestPhase11Dedup(TempCase):
    """D5: one row per (file, line, span, token) — the per-member
    duplicates collapse; disjoint spans stay distinct."""

    def _two_member_plan(self, citer_line="see d/readme.md and note-a.md\n"):
        write(self.root / "d/sub1/note-a.md", "# one\n")
        write(self.root / "d/sub2/note-a.md", "# two\n")
        write(self.root / "d/readme.md", "# readme\n")
        write(self.root / "w.md", citer_line)
        return write_plan(
            self.root,
            moves=[{"from": "d", "to": "e", "reason": "restructure"}],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "w.md",
            ],
            unresolved=[],
            folders=[],
        )

    def test_rewrite_off_ambiguous_mention_yields_one_row(self):
        dry = apply_plan(self._two_member_plan(), self.root)
        risk = dry["reference_risks"]["d"]
        rows = [
            c
            for c in risk["citations"]
            if c["file"] == "w.md" and c.get("token") == "note-a.md"
        ]
        self.assertEqual(
            len(rows),
            1,
            f"K members under one src yield ONE row, got {rows}",
        )
        self.assertEqual(risk["uncovered_references"], 1)

    def test_uncovered_references_count_unique(self):
        dry = apply_plan(
            self._two_member_plan(
                "see d/readme.md and note-a.md and note-a.md again\n"
            ),
            self.root,
        )
        risk = dry["reference_risks"]["d"]
        rows = [
            c
            for c in risk["citations"]
            if c["file"] == "w.md" and c.get("token") == "note-a.md"
        ]
        self.assertEqual(len(rows), 2, "two disjoint spans stay two rows")
        self.assertEqual(risk["uncovered_references"], 2)

    def test_render_dedup(self):
        # The same citation under two srcs is listed once in the
        # render's hand-edit list.
        from slopkeeper.plan import render_markdown

        ref_report = {
            "x.md": {
                "external_references": 1,
                "referenced_by": ["w.md"],
                "citations": [
                    {
                        "file": "w.md",
                        "line": 3,
                        "token": "shared.md",
                        "status": "ambiguous_basename",
                        "why": "ambiguous",
                    }
                ],
            },
            "y.md": {
                "external_references": 1,
                "referenced_by": ["w.md"],
                "citations": [
                    {
                        "file": "w.md",
                        "line": 3,
                        "token": "shared.md",
                        "status": "ambiguous_basename",
                        "why": "ambiguous",
                    }
                ],
            },
        }
        md = render_markdown(
            {
                "moves": [
                    {"from": "x.md", "to": "X/x.md", "reason": "r"},
                    {"from": "y.md", "to": "Y/y.md", "reason": "r"},
                ],
                "edits": [],
            },
            ref_report,
            rewrite_summary={
                "mode": "auto",
                "per_file": [],
                "count": 0,
                "skipped": [],
                "targets": {},
            },
        )
        self.assertEqual(md.count("`w.md:3`"), 1)


class TestPhase14RemoveEmptied(TempCase):
    """D12: remove_emptied removes directories the plan emptied —
    journaled, undoable, never with content, never on a partial run."""

    def _plan(self, **overrides):
        write(self.root / "misc/a.md", "# a\n")
        write(self.root / "misc/b.md", "# b\n")
        write(self.root / "misc/sub/c.md", "# c\n")
        (self.root / "misc/.claude/.cc-writes").mkdir(parents=True, exist_ok=True)
        base = dict(
            moves=[
                {
                    "from": "misc/a.md",
                    "to": "projects/misc/a.md",
                    "reason": "filed",
                },
                {
                    "from": "misc/b.md",
                    "to": "projects/misc/b.md",
                    "reason": "filed",
                },
                {
                    "from": "misc/sub/c.md",
                    "to": "projects/misc/sub/c.md",
                    "reason": "filed",
                },
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "projects",
            ],
            unresolved=[],
            folders=[],
            remove_emptied=True,
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_validate_lists_removals_before_the_gate(self):
        from slopkeeper.plan import load_plan as _load, render_markdown as _render
        from slopkeeper.plan import validate_plan as _validate

        plan = _validate(_load(self._plan()), self.root)
        self.assertEqual(plan["remove_emptied_dirs"], ["misc/sub", "misc"])
        md = _render(plan)
        self.assertIn("Directories this plan will remove", md)
        self.assertIn("`misc/`", md)

    def test_remove_emptied_removes_qualifying_dirs(self):
        out = apply_plan(self._plan(), self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertFalse(out["partial"])
        self.assertFalse((self.root / "misc").exists())
        # Deepest first: the nested dir the per-file moves emptied too.
        self.assertEqual(out.get("removed_dirs"), ["misc/sub", "misc"])
        # Empty ignored subdirs (.claude/.cc-writes) ride along; the
        # journaled op is one per dir, undoable.
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rmdirs = [e for e in lines if e.get("op") == "rmdir"]
        self.assertEqual([e["dir"] for e in rmdirs], ["misc/sub", "misc"])
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertTrue((self.root / "misc").is_dir())
        # The removed tree's empty ignored subdirs ride in the journal
        # op and are recreated in the exact shape.
        self.assertTrue((self.root / "misc/.claude/.cc-writes").is_dir())
        statuses = {op["dir"]: op["status"] for op in undo["operations"] if "dir" in op}
        self.assertEqual(statuses.get("misc"), "recreated")
        self.assertEqual(statuses.get("misc/sub"), "recreated")

    def test_remove_emptied_never_removes_visible_content(self):
        from slopkeeper.plan import load_plan as _load, validate_plan as _validate

        write(self.root / "misc/stray.txt", "data\n")
        plan_path = self._plan()
        # Validate refuses nothing (the dir no longer qualifies) but
        # the removal list is empty; apply must not remove it.
        plan = _validate(_load(plan_path), self.root)
        # misc holds a stray file; misc/sub still qualifies and is
        # removed — misc itself is never touched.
        self.assertEqual(plan.get("remove_emptied_dirs"), ["misc/sub"])
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        self.assertTrue((self.root / "misc").exists())
        self.assertTrue((self.root / "misc/stray.txt").exists())
        self.assertEqual(out.get("removed_dirs"), ["misc/sub"])

    def test_remove_emptied_never_removes_with_files(self):
        # The re-check that matters: an IGNORED file inside the dir is
        # invisible to the validate-time walk (which prunes ignored
        # paths) but is data — the removal refuses loudly instead of
        # deleting it.
        from slopkeeper.plan import load_plan as _load, validate_plan as _validate

        write(self.root / ".slopkeeperignore", "misc/cache.txt\n")
        write(self.root / "misc/cache.txt", "cached junk\n")
        plan_path = self._plan()
        plan = _validate(_load(plan_path), self.root)
        # validate qualifies misc (the ignored file is invisible to it)
        self.assertEqual(plan.get("remove_emptied_dirs"), ["misc/sub", "misc"])
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(out["partial"])
        self.assertTrue((self.root / "misc").exists())
        self.assertTrue((self.root / "misc/cache.txt").exists())
        self.assertTrue(
            any("remove_emptied_refused" in e.get("error", "") for e in out["errors"])
        )

    def test_partial_apply_skips_removal(self):
        # A mid-run move failure leaves the run partial; the removal
        # never fires (loud failure first, T5's precedent).
        from slopkeeper import cli as _cli

        plan_path = self._plan()
        real_move = shutil.move
        state = {"hops": 0}

        def failing_move(src, dst):
            state["hops"] += 1
            if state["hops"] >= 3:
                raise OSError(13, "Permission denied")
            return real_move(src, dst)

        buf = io.StringIO()
        with unittest.mock.patch("shutil.move", side_effect=failing_move):
            with contextlib.redirect_stdout(buf):
                code = _cli.main(["apply", str(plan_path), str(self.root), "--execute"])
        out = json.loads(buf.getvalue())
        self.assertEqual(code, 1)
        self.assertTrue(out["data"]["partial"])
        self.assertNotIn("removed_dirs", out["data"])
        self.assertTrue((self.root / "misc").exists())

    def test_old_journal_without_rmdir_undoes_cleanly(self):
        # A journal with only legacy op kinds (plus a foreign one)
        # undoes via the documented skip path.
        plan_path = self._plan(remove_emptied=False)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        journal = Path(out["journal"])
        with journal.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps({"op": "hologram", "dir": "misc"}) + "\n")
        undo = undo_journal(journal, self.root, execute=True)
        self.assertEqual(undo["mode"], "executed")
        self.assertTrue((self.root / "misc").exists())


class TestPhase16MergeMoves(TempCase):
    """E1: a directory move into a folder created by another move of
    the same plan MERGES file by file — never nests; undo restores a
    byte-identical tree."""

    def _merge_plan(self, **overrides):
        write(self.root / "a/x.png", "png-a\n")
        write(self.root / "c/inner.md", "# inner\n")
        write(self.root / "c/sub/deep.md", "# deep\n")
        base = dict(
            moves=[
                {
                    "from": "a/x.png",
                    "to": "b/x.png",
                    "reason": "creates the destination",
                },
                {"from": "c", "to": "b", "reason": "dir into the created dir"},
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "b",
            ],
            unresolved=[],
            folders=[],
        )
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_merge_into_created_dir_is_flat(self):
        plan_path = self._merge_plan()
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertFalse(out["partial"])
        # Flat: the dir's children sit beside the earlier file move.
        self.assertTrue((self.root / "b/x.png").exists())
        self.assertTrue((self.root / "b/inner.md").exists())
        self.assertTrue((self.root / "b/sub/deep.md").exists())
        self.assertFalse((self.root / "b/c").exists())
        self.assertFalse((self.root / "b/intake").exists())
        self.assertFalse((self.root / "c").exists())
        # One journaled move op per merged child (undo's recipe).
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        merge_ops = [
            e for e in lines if e.get("op") == "move" and e["src"].startswith("c/")
        ]
        self.assertEqual(
            sorted(e["src"] for e in merge_ops),
            ["c/inner.md", "c/sub"],
        )

    def test_merge_leaves_empty_parents_alone(self):
        # Point 1: the merge removes the (now-empty) source dir itself —
        # matching the wholesale move, which IS the rename — but never
        # prunes empty parents. Only remove_emptied removes those.
        write(self.root / "a/x.png", "png-a\n")
        write(self.root / "nested/merge-src/inner.md", "# inner\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "a/x.png", "to": "b/x.png", "reason": "creates b"},
                {"from": "nested/merge-src", "to": "b", "reason": "merge"},
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "b",
                "nested",
                "a",
            ],
            unresolved=[],
            folders=[],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        self.assertFalse((self.root / "nested/merge-src").exists())
        self.assertTrue((self.root / "nested").is_dir())
        self.assertTrue((self.root / "b/inner.md").exists())
        self.assertTrue((self.root / "b/x.png").exists())

    def test_merge_undo_restores_byte_identical_tree(self):
        plan_path = self._merge_plan()
        before = prune_slopkeeper(snapshot_tree(self.root))
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undo["mode"], "executed")
        after = prune_slopkeeper(snapshot_tree(self.root))
        self.assertEqual(after, before, "undo must restore the tree byte-identically")
        # No nested residue from the 0.1.9/0.2.0 E1 bug.
        self.assertFalse((self.root / "b/c").exists())
        self.assertFalse((self.root / "c/c").exists())

    def test_merge_collision_refused_at_validate(self):
        # The dir move's child collides with another move's destination:
        # refused at validate, naming both moves — never a silent
        # overwrite or nest at execute.
        write(self.root / "c/x.png", "png-c\n")
        plan_path = self._merge_plan()
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "merge_collision",
        )

    def test_existing_dst_dir_is_dst_exists(self):
        # A destination that exists on disk before the plan is the
        # dst_exists pre-flight's business — never a merge, never a
        # nest (validate passes it; apply refuses before any move).
        write(self.root / "b/inner.md", "existing\n")
        plan_path = self._merge_plan()
        validate_plan(load_plan(plan_path), self.root)
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "dst_exists",
        )

    def test_merge_into_existing_dir_still_refused(self):
        # A destination that exists BEFORE the plan is a plain
        # dst_exists collision (never a merge, never a nest) — refused
        # before any move, in the dry run too.
        write(self.root / "b/existing.md", "# b\n")
        plan_path = self._merge_plan()
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root),
            "dst_exists",
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "dst_exists",
        )


class TestReferenceScan(TempCase):
    def test_edit_covered_citing_file_not_a_reference_risk(self):
        write(self.root / "target.md", "# Target\n")
        write(self.root / "notes.md", "points at target.md here\n")
        plan_path = write_edit_plan(
            self.root,
            moves=[
                {
                    "from": "target.md",
                    "to": "deep/note-store/target.md",
                    "reason": "cited file moves",
                }
            ],
            edits=[
                {
                    "file": "notes.md",
                    "find": "target.md",
                    "replace": "deep/note-store/target.md",
                    "reason": "rewrite the citation",
                }
            ],
        )
        out = apply_plan(plan_path, self.root, execute=True, allow_partial=True)
        self.assertEqual(out["mode"], "executed")
        # The only citer is resolved by the in-plan edit: no risk reported.
        self.assertEqual(out["reference_risks"], {})
        self.assertIn("deep/note-store/target.md", (self.root / "notes.md").read_text())

    def test_reference_scan_reports_counts(self):
        # Phase 3: the risk scan is un-capped and loud — reference_scan
        # carries status/files_scanned/elapsed_ms, and the old
        # skipped_corpus_too_big skip is dead (never emitted, in either
        # mode). The O(n^2) machinery keeps ITS caps (test_caps.py).
        plan_path = write_plan(self.root)
        dry = apply_plan(plan_path, self.root)
        scan = dry["reference_scan"]
        self.assertEqual(scan["status"], "ok")
        self.assertIsInstance(scan["files_scanned"], int)
        self.assertGreaterEqual(scan["files_scanned"], 1)
        self.assertIsInstance(scan["elapsed_ms"], int)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["reference_scan"]["status"], "ok")
        self.assertGreaterEqual(out["reference_scan"]["files_scanned"], 1)
        self.assertNotIn("skipped_corpus_too_big", json.dumps(dry))
        self.assertNotIn("skipped_corpus_too_big", json.dumps(out))


class TestRewritePass(TempCase):
    """Phase 4: the citation rewrite pass — default on, journaled, undo-exact."""

    def _rewrite_corpus(self, **over):
        # A fresh corpus (not the shared default): docs/ moves to ref/,
        # three files cite paths under it in three token forms — exact
        # path, relative suffix, convention string. b.md is CRLF to pin
        # byte-exact endings.
        root = self.tmp / "rwcorpus"
        write(root / "docs/api.md", "# api\n")
        write(root / "docs/guide.md", "# guide\n")
        write(root / "a.md", "see docs/api.md for the spec\n")
        write(root / "b.md", "read ../docs/guide.md first\r\nsecond line\r\n")
        write(root / "sub/c.md", "convention: docs/<date>_note.md wins\n")
        kw = {
            "moves": [{"from": "docs", "to": "ref", "reason": "docs pile"}],
            "keep": ["a.md", "b.md", "sub"],
            "folders": [],
        }
        kw.update(over)
        return write_plan(root, **kw), root

    def test_rewrite_pass_default_on(self):
        # One dir move rewrites citations in 3 files; the result summary
        # and the journal agree on files and counts.
        plan_path, root = self._rewrite_corpus()
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["rewrites"]["files"], 3)
        self.assertEqual(out["rewrites"]["count"], 3)
        per = {r["file"]: r["count"] for r in out["rewrites"]["per_file"]}
        self.assertEqual(per, {"a.md": 1, "b.md": 1, "sub/c.md": 1})
        self.assertEqual((root / "a.md").read_bytes(), b"see ref/api.md for the spec\n")
        self.assertEqual(
            (root / "b.md").read_bytes(),
            b"read ../ref/guide.md first\r\nsecond line\r\n",
        )
        self.assertEqual(
            (root / "sub/c.md").read_bytes(),
            b"convention: ref/<date>_note.md wins\n",
        )
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rw_ops = [e for e in lines if e.get("op") == "rewrite"]
        self.assertEqual(len(rw_ops), 3)
        self.assertEqual({e["file"] for e in rw_ops}, set(per))
        for e in rw_ops:
            self.assertEqual(e["count"], len(e["replacements"]))

    def test_rewrite_journal_undo_bytes(self):
        # Undo restores every rewritten file byte-for-byte (hash
        # compare) — including the CRLF file's endings.
        plan_path, root = self._rewrite_corpus()
        cites = [root / "a.md", root / "b.md", root / "sub/c.md"]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cites}
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["rewrites"]["count"], 3)
        undone = undo_journal(Path(out["journal"]), root, execute=True)
        self.assertIn("reverted", {o["status"] for o in undone["operations"]})
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
        # And the move itself reversed (docs back in place, ref gone).
        self.assertTrue((root / "docs/api.md").exists())
        self.assertFalse((root / "ref").exists())

    def test_rewrites_off_is_today(self):
        # mode "off" is exactly the pre-PRD apply: no rewrite ops in the
        # journal, no summary fields, citations untouched.
        plan_path, root = self._rewrite_corpus(rewrites={"mode": "off"})
        dry = apply_plan(plan_path, root)
        self.assertNotIn("would_rewrite", dry)
        self.assertNotIn("rewrite_skipped", dry)
        out = apply_plan(plan_path, root, execute=True)
        self.assertNotIn("rewrites", out)
        self.assertNotIn("rewrite_skipped", out)
        self.assertEqual(
            (root / "a.md").read_bytes(), b"see docs/api.md for the spec\n"
        )
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual([e.get("op") for e in lines], ["plan", "move"])

    def test_rewrite_leaves_edit_owned_lines_to_the_edit(self):
        # a.md is named in edits: the LINES its edits own are the
        # edit's (C3 — the edit runs after the pass and does that
        # work; a whole-file skip would strand the file's other
        # citations). a.md is single-line here, so the pass produces
        # nothing for it; the other two citers still rewrite.
        plan_path, root = self._rewrite_corpus(
            edits=[
                {
                    "file": "a.md",
                    "find": "docs/api.md",
                    "replace": "ref/api.md",
                    "reason": "hand-written rewrite",
                }
            ]
        )
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")  # the edit found its text
        self.assertEqual(
            [(r["file"], r["count"]) for r in out["rewrites"]["per_file"]],
            [("b.md", 1), ("sub/c.md", 1)],
        )
        self.assertEqual((root / "a.md").read_bytes(), b"see ref/api.md for the spec\n")
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            [e.get("op") for e in lines if e.get("op") != "plan"],
            ["move", "rewrite", "rewrite", "edit"],
        )

    def test_rewrite_prefix_conflict_skipped(self):
        # A line citing the moved dir AND a path under it: the two new
        # texts nest ("ref" inside "ref/api.md"), which the journal's
        # per-line undo count check could not replay — the line is left
        # untouched and reported, never guessed.
        root = self.tmp / "conflict"
        write(root / "docs/api.md", "# api\n")
        # The dir citation needs a slashed form ("docs/") — a bare word
        # is never a citation (Phase 2). Both tokens rewrite from the
        # same dir rule; the new texts nest, so the line is skipped.
        write(root / "w.md", "see docs/ and docs/api.md today\n")
        plan_path = write_plan(
            root,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(
            dry["rewrite_skipped"],
            [
                {
                    "file": "w.md",
                    "line": 1,
                    "reason": "prefix_conflict",
                    "token": "",
                    "text": "see docs/ and docs/api.md today",
                }
            ],
        )
        self.assertEqual(dry["would_rewrite"], [])
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(), b"see docs/ and docs/api.md today\n"
        )

    def test_rewrite_dup_basename_skipped(self):
        # A bare mention of a basename shared by two files where exactly
        # one member moves: the mention cannot be attributed — skipped.
        root = self.tmp / "dupbase"
        write(root / "dir1/notes.md", "# one\n")
        write(root / "dir2/notes.md", "# two\n")
        write(root / "w.md", "see notes.md here\n")
        plan_path = write_plan(
            root,
            moves=[{"from": "dir1", "to": "one", "reason": "pile"}],
            keep=["dir2", "w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(
            dry["rewrite_skipped"],
            [
                {
                    "file": "w.md",
                    "line": 1,
                    "reason": "ambiguous_basename",
                    "token": "notes.md",
                    "text": "see notes.md here",
                }
            ],
        )
        apply_plan(plan_path, root, execute=True)
        self.assertEqual((root / "w.md").read_bytes(), b"see notes.md here\n")

    def test_strict_references_passes_when_every_citation_is_covered(self):
        # B2: coverage is per CITATION now — a move whose every citing
        # line is rewritten passes strict mode, even if some OTHER move
        # has an uncovered citation.
        plan_path, root = self._rewrite_corpus()
        out = apply_plan(plan_path, root, execute=True, strict_references=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["rewrites"]["count"], 3)

    def test_rewrite_dry_run_pure(self):
        # The dry run reports would-be rewrites per file WITH line
        # detail (B1) and writes NOTHING — every mtime is unchanged.
        plan_path, root = self._rewrite_corpus()
        mtimes = {p: os.stat(p).st_mtime_ns for p in root.rglob("*") if p.is_file()}
        dry = apply_plan(plan_path, root)
        self.assertEqual(
            [(r["file"], r["count"]) for r in dry["would_rewrite"]],
            [("a.md", 1), ("b.md", 1), ("sub/c.md", 1)],
        )
        self.assertEqual(
            dry["would_rewrite"][0]["lines"],
            [{"line": 1, "old": "docs/api.md", "new": "ref/api.md"}],
        )
        self.assertEqual(
            {p: os.stat(p).st_mtime_ns for p in root.rglob("*") if p.is_file()},
            mtimes,
        )


class TestScanPathRewrites(TempCase):
    """Phase 6: scan_paths — ignored subtrees join the rewrite pass.

    Declared via the plan (absolute dir), the pass rewrites citations
    from files the corpus walk never sees, journals them under the
    absolute path, and undo reverses them. Undeclared, the file is
    invisible (edit_target_ignored pins that in test_plan_validate).
    """

    def _rewrite_corpus(self):
        write(self.root / ".cache" / "notes.md", "cite: docs/api.md\n")
        write(self.root / "docs/api.md", "# api\n")
        write(self.root / "docs/guide.md", "# guide\n")
        (self.root / ".slopkeeperignore").write_text(".cache/\n", encoding="utf-8")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            scan_paths=[str(self.root / ".cache")],
            partial=True,
        )
        return plan_path

    def test_scan_path_rewrite_journaled_and_undone(self):
        plan_path = self._rewrite_corpus()
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        # Reported under the ABSOLUTE path (it is outside the walk).
        self.assertEqual(
            {r["file"]: r["count"] for r in out["rewrites"]["per_file"]},
            {str(self.root / ".cache" / "notes.md"): 1},
        )
        self.assertEqual(
            (self.root / ".cache" / "notes.md").read_text(), "cite: ref/api.md\n"
        )
        journal_ops = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rw = [e for e in journal_ops if e.get("op") == "rewrite"]
        self.assertEqual(
            [e["file"] for e in rw], [str(self.root / ".cache" / "notes.md")]
        )
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertIn("reverted", {o["status"] for o in undo["operations"]})
        self.assertEqual(
            (self.root / ".cache" / "notes.md").read_bytes(), b"cite: docs/api.md\n"
        )

    def test_scan_path_dry_run_lists_would_rewrite(self):
        plan_path = self._rewrite_corpus()
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(
            [(r["file"], r["count"]) for r in dry["would_rewrite"]],
            [(str(self.root / ".cache" / "notes.md"), 1)],
        )
        # Zero writes: the ignored file is byte-identical.
        self.assertEqual(
            (self.root / ".cache" / "notes.md").read_bytes(), b"cite: docs/api.md\n"
        )

    def test_undeclared_ignored_file_is_invisible_to_the_pass(self):
        # Without the declaration the rewrite pass never reads the
        # ignored file — the citation is silently NOT fixed, which is
        # exactly why the declaration exists (and refs flags the hit).
        write(self.root / ".cache" / "notes.md", "cite: docs/api.md\n")
        write(self.root / "docs/api.md", "# api\n")
        write(self.root / "docs/guide.md", "# guide\n")
        (self.root / ".slopkeeperignore").write_text(".cache/\n", encoding="utf-8")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            partial=True,
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(
            (self.root / ".cache" / "notes.md").read_text(), "cite: docs/api.md\n"
        )


class TestLinkRecompute(TempCase):
    """Phase 5: link targets recompute; fragments/aliases/titles survive."""

    def test_link_recompute_matrix(self):
        # citing-moves × cited-moves: all four cases produce the exact
        # right relative text (posix relpath between post-move dirs),
        # and the neither-moves case leaves the link untouched.
        cases = [
            (False, False, b"[spec](docs/target.md)\n"),
            (False, True, b"[spec](ref/target.md)\n"),
            (True, False, b"[spec](../docs/target.md)\n"),
            (True, True, b"[spec](../ref/target.md)\n"),
        ]
        for citing_moves, cited_moves, expected in cases:
            root = self.tmp / f"m{int(citing_moves)}{int(cited_moves)}"
            write(root / "docs/target.md", "# t\n")
            write(root / "page.md", "[spec](docs/target.md)\n")
            write(root / "anchor.md", "# anchor\n")
            moves = [
                {"from": "anchor.md", "to": "kept/anchor.md", "reason": "unrelated"}
            ]
            keep = []
            if cited_moves:
                moves.append({"from": "docs", "to": "ref", "reason": "docs move"})
            else:
                keep.append("docs")
            if citing_moves:
                moves.append(
                    {"from": "page.md", "to": "deep/page.md", "reason": "citing move"}
                )
            else:
                keep.append("page.md")
            plan_path = write_plan(root, moves=moves, keep=sorted(keep), folders=[])
            out = apply_plan(plan_path, root, execute=True)
            self.assertEqual(out["mode"], "executed")
            final = "deep/page.md" if citing_moves else "page.md"
            self.assertEqual((root / final).read_bytes(), expected)

    def test_fragment_alias_title_preserved(self):
        # [[old/dir/note#sec|alias]] keeps #sec|alias (the replacement
        # span is the target only); the markdown title and ?query stay.
        root = self.tmp / "frag"
        write(root / "old/dir/note.md", "# note\n")
        write(root / "wiki.md", "see [[old/dir/note#sec|alias]] now\n")
        write(
            root / "md.md",
            '[t](old/dir/note.md#frag "the title") and [q](old/dir/note.md?x=1)\n',
        )
        plan_path = write_plan(
            root,
            moves=[{"from": "old", "to": "new", "reason": "pile"}],
            keep=["wiki.md", "md.md"],
            folders=[],
        )
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(
            (root / "wiki.md").read_bytes(),
            b"see [[new/dir/note.md#sec|alias]] now\n",
        )
        self.assertEqual(
            (root / "md.md").read_bytes(),
            b'[t](new/dir/note.md#frag "the title") and [q](new/dir/note.md?x=1)\n',
        )

    def test_plain_token_fallback(self):
        # A non-link mention of a moved path takes the literal token
        # swap — recomputation is for link sites only.
        root = self.tmp / "plain"
        write(root / "docs/api.md", "# api\n")
        write(root / "w.md", "mention docs/api.md in prose, not a link\n")
        plan_path = write_plan(
            root,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
        )
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"mention ref/api.md in prose, not a link\n",
        )


class TestJournal(TempCase):
    def test_move_lines_carry_real_content_sha256(self):
        plan_path = write_plan(self.root)
        known = "handoff-tp2.md"
        expected = hashlib.sha256((self.root / known).read_bytes()).hexdigest()
        out = apply_plan(plan_path, self.root, execute=True)
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        moves = [entry for entry in lines if entry.get("op") == "move"]
        self.assertEqual(len(moves), 4)
        for entry in moves:
            content = (self.root / entry["dst"]).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual({e["src"]: e["sha256"] for e in moves}[known], expected)


# --- error paths: unreadable edit targets, empty journals ---


class TestErrorPaths(TempCase):
    def test_edit_unreadable(self):
        (self.root / "WATCHLIST.md").chmod(0o000)
        plan_path = write_edit_plan(self.root)
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "edit_unreadable",
        )
        (self.root / "WATCHLIST.md").chmod(0o644)

    def test_undo_empty_journal_is_journal_empty(self):
        j = self.root / ".slopkeeper" / "journal-empty.jsonl"
        j.parent.mkdir(parents=True)
        j.write_text("", encoding="utf-8")
        expect_code(self, lambda: undo_journal(j, self.root), "journal_empty")


class TestPartialApply(TempCase):
    """T5: a mid-run OSError is journaled-partial and loudly reported.

    All-or-nothing covers pre-flight-checked failures only. A hop that
    dies mid-run (EACCES, disk full) must not surface as a bare
    `internal` envelope: earlier hops stay journaled and undoable, the
    envelope says `partial`, `next` points at the exact undo command,
    and the process exits non-zero. A *successful* apply whose errors
    list is non-empty (vanished mid-apply) exits non-zero too.
    """

    @staticmethod
    def _run_main(argv) -> tuple[int, dict]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(argv)
        return code, json.loads(buf.getvalue())

    def _three_move_plan(self) -> Path:
        return write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier research",
                },
                {
                    "from": "globex_dossier.md",
                    "to": "research/dossiers/globex_dossier.md",
                    "reason": "dossier research",
                },
                {
                    "from": "handoff-tp2.md",
                    "to": "spark-1.2.3/handoff-tp2.md",
                    "reason": "spark handoff",
                },
            ],
            # ledger_wb.html loses its move in this 3-move plan.
            keep=[
                "WATCHLIST.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "ledger_wb.html",
            ],
        )

    def test_second_hop_oserror_is_partial_exit_1_and_points_at_undo(self):
        # The ticket's mock: shutil.move raises OSError from the second
        # hop on. Hop 1 lands (and is journaled); hops 2-3 fail mid-run.
        real_move = shutil.move
        state = {"hops": 0}

        def failing_move(src, dst):
            state["hops"] += 1
            if state["hops"] >= 2:
                raise OSError(13, "Permission denied")
            return real_move(src, dst)

        plan_path = self._three_move_plan()
        with unittest.mock.patch("shutil.move", side_effect=failing_move):
            code, out = self._run_main(
                ["apply", str(plan_path), str(self.root), "--execute"]
            )

        self.assertEqual(code, 1)  # EXIT_ERROR — loudly non-zero
        self.assertEqual(out["command"], "apply")
        self.assertTrue(out["ok"])  # the apply ran; not an error envelope
        self.assertTrue(out["partial"])  # the envelope says PARTIAL
        data = out["data"]
        self.assertTrue(data["partial"])
        self.assertEqual(data["mode"], "executed")
        self.assertEqual(data["moved_count"], 1)
        self.assertEqual(
            [e["src"] for e in data["errors"]],
            ["globex_dossier.md", "handoff-tp2.md"],
        )
        # `next` points at the exact undo command for THIS journal.
        nxt = out["next"]
        self.assertEqual(nxt["action"], "undo")
        self.assertTrue(nxt["command"].startswith("slopkeeper undo "))
        self.assertIn(data["journal"], nxt["command"])

        # Hop 1 is on disk; the failed hops stayed put.
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        self.assertTrue((self.root / "globex_dossier.md").exists())
        self.assertTrue((self.root / "handoff-tp2.md").exists())

        # The journal contains hop 1 — and the promised undo dry-run
        # shows would_restore for it. (The rewrite pass also journals
        # its WATCHLIST.md citation fix for the hop that DID move.)
        lines = [
            json.loads(line)
            for line in Path(data["journal"]).read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual([e.get("op") for e in lines], ["plan", "move", "rewrite"])
        self.assertEqual(lines[1]["src"], "acme_dossier.md")
        ucode, uout = self._run_main(["undo", data["journal"], str(self.root)])
        self.assertEqual(ucode, 0)
        self.assertIn(
            {"src": "acme_dossier.md", "status": "would_restore"},
            uout["data"]["operations"],
        )

    def test_vanished_mid_apply_exits_nonzero_with_partial(self):
        # The other half of the bug: a *successful* apply whose errors
        # list is non-empty (a source vanished mid-apply) used to exit 0
        # — an agent keying off ok:true missed it. Pre-flight (and
        # plan_stale_src) cannot see a vanish that happens DURING the
        # loop, so hop 1's execution deletes hop 2's source.
        real_move = shutil.move

        def vanish_globex_then_move(src, dst):
            g = self.root / "globex_dossier.md"
            if g.exists():
                g.unlink()
            return real_move(src, dst)

        plan_path = self._three_move_plan()
        with unittest.mock.patch("shutil.move", side_effect=vanish_globex_then_move):
            code, out = self._run_main(
                ["apply", str(plan_path), str(self.root), "--execute"]
            )

        self.assertEqual(code, 1)
        self.assertTrue(out["partial"])
        data = out["data"]
        self.assertEqual(data["moved_count"], 2)  # hop 1 and hop 3 landed
        self.assertEqual(
            data["errors"],
            [{"src": "globex_dossier.md", "error": "vanished mid-apply"}],
        )

    def test_clean_apply_stays_exit_0_without_partial(self):
        # Control: a fully-successful apply is unchanged — exit 0 and no
        # partial flag on the envelope.
        plan_path = self._three_move_plan()
        code, out = self._run_main(
            ["apply", str(plan_path), str(self.root), "--execute"]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("partial", out)
        self.assertFalse(out["data"]["partial"])
        self.assertEqual(out["data"]["moved_count"], 3)
        self.assertEqual(out["data"]["errors"], [])


class TestRulePlanApply(TempCase):
    """Phase 8: an expanded plan journals exactly like the hand-written one."""

    @staticmethod
    def _build(base: Path) -> Path:
        return build_corpus(
            base / "corpus",
            {"reports/_org/a.md": "a\n", "reports/_org/sub/b.md": "b\n"},
        )

    @staticmethod
    def _move_entries(result: dict) -> list[tuple[str, str, str]]:
        journal = Path(result["journal"])
        return sorted(
            (e["src"], e["dst"], e["sha256"])
            for e in (
                json.loads(line)
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if e.get("op") == "move"
        )

    def test_rule_plan_equals_explicit_plan(self):
        # Two identical corpora: the one-entry rule plan and its explicit
        # equivalent must produce the same journal move entries (src, dst,
        # sha). The journal header's plan sha256 differs — different plan
        # bytes is the entire point of rules.
        rule_root = self._build(self.tmp / "r1")
        explicit_root = self._build(self.tmp / "r2")
        rule_result = apply_plan(
            write_plan(
                rule_root,
                moves=[],
                partial=True,
                rules=[{"match": "reports/_org/", "to": "reports/org"}],
            ),
            rule_root,
            execute=True,
        )
        explicit_result = apply_plan(
            write_plan(
                explicit_root,
                partial=True,
                moves=[
                    {
                        "from": "reports/_org/a.md",
                        "to": "reports/org/a.md",
                        "reason": "rule",
                    },
                    {
                        "from": "reports/_org/sub/b.md",
                        "to": "reports/org/b.md",
                        "reason": "rule",
                    },
                ],
            ),
            explicit_root,
            execute=True,
        )
        self.assertEqual(rule_result["moved_count"], 2)
        self.assertEqual(explicit_result["moved_count"], 2)
        self.assertEqual(rule_result["errors"], [])
        self.assertEqual(explicit_result["errors"], [])
        self.assertEqual(
            self._move_entries(rule_result), self._move_entries(explicit_result)
        )
        # same end state on disk, byte for byte
        self.assertEqual(
            prune_slopkeeper(snapshot_tree(rule_root)),
            prune_slopkeeper(snapshot_tree(explicit_root)),
        )


class TestRewriteOffMoves(TempCase):
    """B1: a move may opt its subtree out of the rewrite pass."""

    def _corpus(self):
        root = self.tmp / "vendored"
        write(root / "proto/index.html", '<a href="uploads/pic.png">p</a>\n')
        (root / "proto/uploads").mkdir(parents=True)
        (root / "proto/uploads/pic.png").write_bytes(b"\x89PNG fake")
        write(root / "outside.md", "proto lives in proto/\n")
        (root / "misc").mkdir(parents=True)
        (root / "misc/pic.png").write_bytes(b"\x89PNG other")
        return root

    def test_rewrite_false_skips_the_subtree_scan(self):
        root = self._corpus()
        # A PNG with the same basename exists outside the subtree, and
        # moves: without the opt-out the pass would rewrite the
        # prototype's internal link to it.
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "proto",
                    "to": "vendored/proto",
                    "reason": "vendored",
                    "rewrite": False,
                },
                {"from": "misc/pic.png", "to": "assets/pic.png", "reason": "asset"},
            ],
        )
        dry = apply_plan(plan_path, root)
        # The vendored subtree produced no rewrites; only the outside
        # citation of the moved dir is rewritten.
        self.assertEqual([r["file"] for r in dry["would_rewrite"]], ["outside.md"])
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "vendored/proto/index.html").read_text(),
            '<a href="uploads/pic.png">p</a>\n',
        )

    def test_rewrite_false_outbound_citations_are_flagged(self):
        # The opt-out's own trap: a vendored subtree that cites a
        # DIFFERENT moving path is not self-covered — its outbound
        # citation will not be rewritten, and the gate must say so.
        root = self._corpus()
        write(root / "outside.md", "see misc/pic.png too\n")
        write(root / "proto/links.md", "see misc/pic.png from the prototype\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "proto",
                    "to": "vendored/proto",
                    "reason": "vendored",
                    "rewrite": False,
                },
                {"from": "misc/pic.png", "to": "assets/pic.png", "reason": "asset"},
            ],
        )
        dry = apply_plan(plan_path, root)
        risk = dry["reference_risks"]["misc/pic.png"]
        self.assertEqual(risk["uncovered_references"], 1)
        by_file = {c["file"]: c for c in risk["citations"]}
        self.assertEqual(by_file["outside.md"]["status"], "rewrite")
        self.assertEqual(by_file["proto/links.md"]["status"], "rewrite_off")
        self.assertIn("move with it", by_file["proto/links.md"]["why"])

    def test_rewrite_false_flag_survives_validation_and_renders(self):
        from slopkeeper.plan import load_plan, render_markdown, validate_plan

        root = self._corpus()
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "proto",
                    "to": "vendored/proto",
                    "reason": "vendored",
                    "rewrite": False,
                },
            ],
        )
        plan = load_plan(plan_path)
        out = validate_plan(plan, root)
        move = next(m for m in out["moves"] if m["from"] == "proto")
        self.assertIs(move.get("rewrite"), False)
        md = render_markdown(out)
        self.assertIn("[rewrite: off]", md)

    def test_non_bool_rewrite_refused(self):
        from slopkeeper.plan import load_plan, validate_plan

        root = self._corpus()
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "proto",
                    "to": "vendored/proto",
                    "reason": "r",
                    "rewrite": "no",
                },
            ],
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), root),
            "plan_bad_move",
        )


class TestRewriteSites(TempCase):
    """B2/B9/B10: sites, tokens, and explained no-rewrites."""

    def test_path_qualified_mention_of_stay_put_file_is_not_ambiguous(self):
        # The 0.1.6 report's 720 noisy rows: `slack/AGENTS.md` names an
        # existing stay-put file — it is not an ambiguous bare mention
        # of AGENTS.md just because another AGENTS.md moves.
        root = self.tmp / "noise"
        write(root / "dir1/AGENTS.md", "# one\n")
        write(root / "dir2/AGENTS.md", "# two\n")
        write(root / "slack/AGENTS.md", "# slack\n")
        write(root / "w.md", "config lives in slack/AGENTS.md\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "dir1/AGENTS.md", "to": "moved/AGENTS.md", "reason": "r"}],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(dry["rewrite_skipped"], [])
        self.assertEqual(dry["would_rewrite"], [])

    def test_plan_file_inside_corpus_is_never_rewritten(self):
        # The tool's own artifacts are excluded from the pass: a plan
        # JSON living inside the corpus cites moved paths by
        # definition — rewriting it mid-apply would corrupt the audit
        # trail (the smoke fixture caught this). (The plan itself
        # still cannot apply from inside the corpus: writing it shifts
        # the fingerprint — the draft_out_inside_root trap.)
        from slopkeeper.apply import _run_artifacts, _run_rewrite_pass
        from slopkeeper.models import resolve_chains
        from slopkeeper.plan import load_plan, validate_plan

        root = self.tmp / "inside"
        write(root / "docs/api.md", "# api\n")
        write(root / "w.md", "see docs/api.md\n")
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(root),
            "strategy": "hybrid",
            "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
            "moves": [
                {
                    "from": "docs/api.md",
                    "to": "ref/api.md",
                    "reason": "cite me",
                },
                {"from": "w.md", "to": "kept/w.md", "reason": "keep dir"},
            ],
            "edits": [],
            "keep": [],
            "unresolved": [],
            "folders": [],
            "partial": True,
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        plan = validate_plan(load_plan(plan_path), root, allow_partial=True)
        chains = resolve_chains(plan["moves"])
        rel_paths = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        exclude = _run_artifacts(root, str(plan_path))
        result = _run_rewrite_pass(root, plan, chains, rel_paths, exclude)
        self.assertNotIn("plan.json", result["rewrites"])
        self.assertIn("kept/w.md", result["rewrites"])
        # the plan file is a scan CANDIDATE (it cites docs/api.md) —
        # only the explicit artifact exclusion keeps it out
        self.assertIn("plan.json", _run_artifacts(root, str(plan_path)))

    def test_cited_but_no_change_explained_in_sites(self):
        # Both ends move: the relative form recomputes to the same
        # text, so there is no rewrite — and the site record says why
        # (B10: no more unexplained "cited but 0 rewrites").
        from slopkeeper.apply import _run_rewrite_pass
        from slopkeeper.plan import load_plan, validate_plan
        from slopkeeper.models import resolve_chains

        root = self.tmp / "nochange"
        write(root / "a/index.html", '<img src="pic.png">\n')
        (root / "a/pic.png").write_bytes(b"\x89PNG")
        write(root / "other.md", "x\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "a", "to": "b", "reason": "whole dir"}],
        )
        plan = validate_plan(load_plan(plan_path), root)
        chains = resolve_chains(plan["moves"])
        rel_paths = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        result = _run_rewrite_pass(root, plan, chains, rel_paths)
        sites = result["sites"].get("a/pic.png", [])
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["status"], "no_change")
        self.assertEqual(sites[0]["token"], "pic.png")
        self.assertEqual(result["rewrites"], {})

    def test_enriched_risk_rows_carry_citations(self):
        # A citation the pass fixes is not a risk; a not_utf8 citer is,
        # with the why spelled out (B2).
        from slopkeeper.apply import build_gate_report
        from slopkeeper.plan import load_plan, validate_plan

        root = self.tmp / "utf8"
        write(root / "docs/api.md", "# api\n")
        write(root / "good.md", "see docs/api.md\n")
        (root / "bad.md").write_bytes(b"see docs/api.md \xff\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["good.md", "bad.md"],
        )
        plan = validate_plan(load_plan(plan_path), root)
        ref_report, summary = build_gate_report(plan, root)
        risk = ref_report["docs"]
        self.assertEqual(risk["uncovered_references"], 1)
        by_file = {c["file"]: c for c in risk["citations"]}
        self.assertEqual(by_file["good.md"]["status"], "rewrite")
        self.assertEqual(by_file["bad.md"]["status"], "not_utf8")
        self.assertIn("never re-encodes", by_file["bad.md"]["why"])
        # the fixable citation still shows up as a rewrite
        self.assertIn("good.md", [r["file"] for r in summary["per_file"]])


class TestPrefixRewrites(TempCase):
    """B8: declared renames for dirs emptied file by file."""

    def _plan(self, root, **overrides):
        kwargs = dict(
            partial=True,
            moves=[
                {
                    "from": "reports/_org/2026-01-01_a.md",
                    "to": "reports/org/2026-01-01_a.md",
                    "reason": "one by one",
                }
            ],
            keep=[
                "README.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
            ],
            prefix_rewrites=[{"from": "reports/_org/", "to": "reports/org/"}],
        )
        kwargs.update(overrides)
        return write_plan(root, **kwargs)

    def test_prefix_rewrite_swaps_convention_strings(self):
        root = self.tmp / "pr"
        write(root / "reports/_org/2026-01-01_a.md", "# a\n")
        write(
            root / "README.md",
            "logs: reports/_org/<date>_<topic>.md\nsee reports/_org/missing.md too\n",
        )
        plan_path = self._plan(root)
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual((root / "reports/org/2026-01-01_a.md").exists(), True)
        text = (root / "README.md").read_text()
        self.assertIn("reports/org/<date>_<topic>.md", text)
        # the unresolvable exact path under the declared prefix follows too
        self.assertIn("reports/org/missing.md", text)
        # ...and the rewrite is journaled and undoable
        undo_journal(Path(out["journal"]), root, execute=True)
        self.assertIn(
            "reports/_org/<date>_<topic>.md", (root / "README.md").read_text()
        )

    def test_prefix_rewrite_shows_in_render(self):
        from slopkeeper.plan import load_plan, render_markdown, validate_plan

        root = self.tmp / "pr2"
        write(root / "reports/_org/2026-01-01_a.md", "# a\n")
        write(root / "README.md", "logs: reports/_org/<date>.md\n")
        plan_path = self._plan(root)
        plan = validate_plan(load_plan(plan_path), root)
        md = render_markdown(plan, emptied_dirs=plan["emptied_dir_suggestions"])
        self.assertIn("`reports/_org` → `reports/org/`", md)
        # the emptied-dir suggestion section carries the evidence
        self.assertIn("Directories emptied file by file", md)
        self.assertIn("`reports/_org/`", md)

    def test_prefix_rewrites_do_not_touch_other_paths(self):
        root = self.tmp / "pr3"
        write(root / "reports/_org/2026-01-01_a.md", "# a\n")
        write(root / "README.md", "unrelated reports/_other/x stays\n")
        plan_path = self._plan(root)
        apply_plan(plan_path, root, execute=True)
        self.assertIn("reports/_other/x", (root / "README.md").read_text())


class TestWritabilityPreflight(TempCase):
    """B7: an unwritable rewrite/edit target refuses before ANY move."""

    def test_unwritable_edit_target_refuses_whole_apply(self):
        plan_path = write_edit_plan(self.root)  # edits kept WATCHLIST.md
        real_access = os.access

        def fake_access(path, mode):
            if str(path).endswith("WATCHLIST.md"):
                return False
            return real_access(path, mode)

        with unittest.mock.patch("slopkeeper.apply.os.access", side_effect=fake_access):
            with expect_code_cm(self, "target_unwritable"):
                apply_plan(plan_path, self.root, execute=True)
        # nothing moved
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

    def test_unwritable_rewrite_target_refuses_whole_apply(self):
        write(self.root / "watch2.md", "see docs/canonical.md\n")
        plan_path = write_plan(
            self.root,
            keep=[
                "WATCHLIST.md",
                "lead-gen",
                "spark-1.2.3",
                "watch2.md",
                "globex_dossier.md",
                "ledger_wb.html",
                "handoff-tp2.md",
            ],
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            unresolved=[{"path": "acme_dossier.md", "reason": "human decides"}],
            folders=[],
        )
        real_access = os.access

        def fake_access(path, mode):
            if str(path).endswith("watch2.md"):
                return False
            return real_access(path, mode)

        with unittest.mock.patch("slopkeeper.apply.os.access", side_effect=fake_access):
            with expect_code_cm(self, "target_unwritable"):
                apply_plan(plan_path, self.root, execute=True)
        self.assertTrue((self.root / "docs/canonical.md").exists())
        self.assertFalse((self.root / "ref").exists())


@contextlib.contextmanager
def expect_code_cm(testcase, code):
    """Context-manager form of expect_code (for with-blocks)."""
    from slopkeeper.models import SlopKeeperError

    try:
        yield
    except SlopKeeperError as e:
        testcase.assertEqual(
            e.code, code, f"expected {code}, got {e.code}: {e.message}"
        )
    else:
        testcase.fail(f"expected SlopKeeperError {code}")


if __name__ == "__main__":
    unittest.main()


class TestFieldReport018(TempCase):
    """The 0.1.8 field report's blockers, each as a pinned behavior.

    C1 rule stamps, C2 form preservation (URL/hosts, stale paths,
    suffix → root-path-out, relative churn), C3 edits and rewrites in
    one file, C4 one find-matching rule in validate and apply, C5
    per-token skips, C6 rewrite_off detail.
    """

    def test_c1_rule_moves_take_the_default_rewrite_pass(self):
        # A rule without "rewrite" expands to moves WITHOUT the stamp,
        # and the pass rewrites citations of them (0.1.8 C1: every
        # expanded move was stamped rewrite:false and got 0 rewrites).
        root = self.tmp / "c1"
        write(root / "reports/_org/a.md", "# a\n")
        write(root / "reports/_org/b.md", "# b\n")
        write(root / "w.md", "see reports/_org/a.md and reports/_org/b.md\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[],
            rules=[{"match": "reports/_org/", "to": "reports/org"}],
            keep=["w.md"],
            folders=[],
        )
        from slopkeeper.plan import load_plan, validate_plan

        plan = validate_plan(load_plan(plan_path), root)
        expanded = [m for m in plan["moves"] if m["from"].startswith("reports/_org/")]
        self.assertEqual(len(expanded), 2)
        self.assertTrue(all("rewrite" not in m for m in expanded))
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["count"] for r in dry["would_rewrite"]}
        self.assertEqual(per, {"w.md": 2})
        # A rule that DOES opt out stamps every move it expands.
        plan_path2 = write_plan(
            root,
            partial=True,
            moves=[],
            rules=[
                {
                    "match": "reports/_org/",
                    "to": "reports/org",
                    "rewrite": False,
                }
            ],
            keep=["w.md"],
            folders=[],
        )
        plan2 = validate_plan(load_plan(plan_path2), root)
        expanded2 = [m for m in plan2["moves"] if m["from"].startswith("reports/_org/")]
        self.assertTrue(all(m.get("rewrite") is False for m in expanded2))
        # The opt-out silences the SUBTREE's own citations; citations
        # from outside into the moved files are still rewritten (B1).
        dry2 = apply_plan(plan_path2, root)
        self.assertEqual(
            [(r["file"], r["count"]) for r in dry2["would_rewrite"]],
            [("w.md", 2)],
        )

    def test_c2_url_token_is_never_a_citation(self):
        # A URL whose path tail ends in a moved basename: never
        # resolved, never rewritten (the tail would mangle the URL into
        # a bare filename).
        root = self.tmp / "c2url"
        write(root / "docs/api.md", "# api\n")
        write(
            root / "w.md",
            "recipes: https://docs.acme.example/nightly/api.md\n",
        )
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(dry["would_rewrite"], [])
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"recipes: https://docs.acme.example/nightly/api.md\n",
        )
        self.assertEqual(out["rewrites"]["count"], 0)

    def test_c2_stale_path_skipped_and_reported_never_barenamed(self):
        # `misc/email.md` is stale (the file lives at misc/acme/) —
        # resolving it by basename and writing the bare new basename
        # was the mangle; now the token is skipped and reported.
        root = self.tmp / "c2stale"
        write(root / "misc/acme/email.md", "# email\n")
        write(root / "w.md", "the thread: misc/email.md (acme, 8/26)\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "misc/acme/email.md",
                    "to": "projects/acme/emails/email.md",
                    "reason": "filed",
                }
            ],
            keep=["w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(dry["would_rewrite"], [])
        self.assertEqual(
            [s for s in dry["skip_rewrites"] if s["token"] == "misc/email.md"],
            [
                {
                    "file": "w.md",
                    "line": 1,
                    "token": "misc/email.md",
                    "reason": "unresolved_path",
                }
            ],
        )
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"the thread: misc/email.md (acme, 8/26)\n",
        )

    def test_c2_suffix_citation_gets_root_path_out(self):
        # `_org/x.md` is a unique suffix of the moved path: resolved,
        # and written as the new ROOT path — never a bare basename.
        root = self.tmp / "c2suffix"
        write(root / "reports/_org/x.md", "# x\n")
        write(root / "reports/sync/f.md", "see _org/x.md and _sync/x.md\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "reports/_org/x.md",
                    "to": "projects/actions/x.md",
                    "reason": "filed",
                }
            ],
            keep=["reports/sync"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["lines"] for r in dry["would_rewrite"]}
        self.assertEqual(
            per["reports/sync/f.md"],
            [
                {
                    "line": 1,
                    "old": "_org/x.md",
                    "new": "projects/actions/x.md",
                }
            ],
        )

    def test_c2_relative_link_untouched_when_both_ends_move_together(self):
        # `./b.md` from a co-moved dir still names the target after the
        # move: NO rewrite (the 0.1.8 style churn `./x.md` → `x.md`).
        root = self.tmp / "c2churn"
        write(root / "team/a.md", "[x](./b.md) and [y](b.md)\n")
        write(root / "team/b.md", "# b\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "team", "to": "proj", "reason": "whole"}],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        self.assertEqual(dry["would_rewrite"], [])
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "proj/a.md").read_bytes(), b"[x](./b.md) and [y](b.md)\n"
        )

    def test_c2_root_path_citation_stays_a_root_path(self):
        # A root-path-form citation (no ./ ../) is rewritten to the new
        # ROOT path, not recomputed into a relative link.
        root = self.tmp / "c2root"
        write(root / "team/b.md", "# b\n")
        write(root / "w.md", "[b](team/b.md) and `team/b.md` in prose\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "team/b.md", "to": "proj/b.md", "reason": "filed"}],
            keep=["w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["lines"] for r in dry["would_rewrite"]}
        self.assertEqual(
            per["w.md"],
            [
                {"line": 1, "old": "team/b.md", "new": "proj/b.md"},
                {"line": 1, "old": "team/b.md", "new": "proj/b.md"},
            ],
        )
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"[b](proj/b.md) and `proj/b.md` in prose\n",
        )

    def test_c2_bare_basename_suffix_of_moved_path_is_not_degraded(self):
        # `sub/x.md` where the token is a proper suffix of the moved
        # path gets the full new path (root path in, root path out).
        root = self.tmp / "c2suffix2"
        write(root / "misc/intake/client.md", "# k\n")
        write(root / "w.md", "client notes: intake/client.md\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "misc/intake/client.md",
                    "to": "projects/intake/client.md",
                    "reason": "filed",
                }
            ],
            keep=["w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["lines"] for r in dry["would_rewrite"]}
        self.assertEqual(
            per["w.md"],
            [
                {
                    "line": 1,
                    "old": "intake/client.md",
                    "new": "projects/intake/client.md",
                }
            ],
        )

    def test_c3_edits_and_rewrites_coexist_in_one_file(self):
        # The pass rewrites the file's lines the edits do not own; the
        # edit lands on its own line after (0.1.8 C3: any edit entry
        # silenced ALL of the file's automatic rewrites).
        root = self.tmp / "c3"
        write(root / "docs/api.md", "# api\n")
        write(root / "docs/guide.md", "# guide\n")
        write(
            root / "w.md",
            "see docs/api.md for the spec\nsee docs/guide.md for the tour\n",
        )
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
            edits=[
                {
                    "file": "w.md",
                    "find": "see docs/guide.md for the tour",
                    "replace": "see ref/guide.md for the tour",
                    "reason": "line the pass cannot take",
                }
            ],
        )
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["lines"] for r in dry["would_rewrite"]}
        # Only the non-owned line is rewritten by the pass.
        self.assertEqual(
            per["w.md"], [{"line": 1, "old": "docs/api.md", "new": "ref/api.md"}]
        )
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"see ref/api.md for the spec\nsee ref/guide.md for the tour\n",
        )
        self.assertEqual(out["mode"], "executed")

    def test_c4_validate_accepts_post_rewrite_find_and_apply_agrees(self):
        # One rule for both surfaces: a find authored from POST-rewrite
        # text passes validate (the pass produces it) and applies after
        # it — and a pre-rewrite find on the same kind of line is
        # owned by its edit, so it never goes stale mid-apply.
        root = self.tmp / "c4"
        write(root / "docs/api.md", "# api\n")
        write(root / "w.md", "see docs/api.md for the spec\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
            edits=[
                {
                    "file": "w.md",
                    "find": "see ref/api.md for the spec",
                    "replace": "see ref/api.md for the spec, updated",
                    "reason": "extends the pass's rewrite",
                }
            ],
        )
        from slopkeeper.plan import load_plan, validate_plan

        validate_plan(load_plan(plan_path), root)  # no raise
        dry = apply_plan(plan_path, root)
        self.assertEqual(dry["mode"], "dry-run")
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertIn(
            "see ref/api.md for the spec, updated", (root / "w.md").read_text()
        )

    def test_c4_validate_refuses_find_the_pass_neither_leaves_nor_makes(self):
        root = self.tmp / "c4stale"
        write(root / "docs/api.md", "# api\n")
        write(root / "w.md", "see docs/api.md for the spec\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[{"from": "docs", "to": "ref", "reason": "pile"}],
            keep=["w.md"],
            folders=[],
            edits=[
                {
                    "file": "w.md",
                    "find": "see elsewhere/api.md for the spec",
                    "replace": "x",
                    "reason": "typo'd find",
                }
            ],
        )
        from slopkeeper.plan import load_plan, validate_plan

        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), root),
            "edit_stale",
        )
        # Nothing moved, nothing written.
        self.assertEqual(
            (root / "w.md").read_bytes(), b"see docs/api.md for the spec\n"
        )

    def test_c5_ambiguous_token_skips_only_itself(self):
        # One ambiguous basename on a line no longer blocks the line's
        # other rewrites (0.1.8 C5).
        root = self.tmp / "c5"
        write(root / "docs/api.md", "# api\n")
        write(root / "dir1/CLAUDE.md", "# one\n")
        write(root / "dir2/CLAUDE.md", "# two\n")
        write(root / "w.md", "local only (see CLAUDE.md); api at docs/api.md\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {"from": "docs", "to": "ref", "reason": "pile"},
                {"from": "dir1", "to": "one", "reason": "pile"},
            ],
            keep=["dir2", "w.md"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        per = {r["file"]: r["lines"] for r in dry["would_rewrite"]}
        self.assertEqual(
            per["w.md"], [{"line": 1, "old": "docs/api.md", "new": "ref/api.md"}]
        )
        skipped = [s for s in dry["rewrite_skipped"] if s["file"] == "w.md"]
        self.assertEqual(
            skipped,
            [
                {
                    "file": "w.md",
                    "line": 1,
                    "reason": "ambiguous_basename",
                    "token": "CLAUDE.md",
                    "text": "local only (see CLAUDE.md); api at docs/api.md",
                }
            ],
        )
        apply_plan(plan_path, root, execute=True)
        self.assertEqual(
            (root / "w.md").read_bytes(),
            b"local only (see CLAUDE.md); api at ref/api.md\n",
        )

    def test_c6_rewrite_off_rows_carry_site_detail(self):
        # A "rewrite": false subtree's citations are shown with
        # file/line/text/token (0.1.8 C6: line/text/token were null).
        root = self.tmp / "c6"
        write(root / "proto/links.md", "see misc/pic.png from the prototype\n")
        (root / "misc").mkdir(parents=True, exist_ok=True)
        (root / "misc/pic.png").write_bytes(b"\x89PNG\r\n")
        plan_path = write_plan(
            root,
            partial=True,
            moves=[
                {
                    "from": "proto",
                    "to": "vendored/proto",
                    "reason": "vendored",
                    "rewrite": False,
                },
                {"from": "misc/pic.png", "to": "assets/pic.png", "reason": "asset"},
            ],
            keep=["misc"],
            folders=[],
        )
        dry = apply_plan(plan_path, root)
        risk = dry["reference_risks"]["misc/pic.png"]
        rows = [c for c in risk["citations"] if c["file"] == "proto/links.md"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "rewrite_off")
        self.assertEqual(row["line"], 1)
        self.assertEqual(row["token"], "misc/pic.png")
        self.assertEqual(row["text"], "see misc/pic.png from the prototype")
