# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""plan.py — the validator: every SlopKeeperError code."""

from __future__ import annotations

import json
import time
import unittest
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
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError
from slopkeeper.plan import load_plan, render_markdown, straggler_report, validate_plan


class TestPlanValidation(TempCase):
    def test_valid_plan_passes(self):
        plan = load_plan(write_plan(self.root))
        out = validate_plan(plan, self.root)
        self.assertEqual(len(out["moves"]), 4)

    def test_bad_rewrites_mode(self):
        # Phase 4: "rewrites" is optional; mode is auto|off; anything
        # else (bad value, extra keys, wrong type) is plan_bad_rewrites.
        for bad in ({"mode": "banana"}, {"mode": "auto", "aggressive": True}, "off"):
            plan = load_plan(write_plan(self.root, rewrites=bad))
            expect_code(
                self, lambda p=plan: validate_plan(p, self.root), "plan_bad_rewrites"
            )
        # Both valid forms pass, and absence means auto.
        for ok in ({"mode": "auto"}, {"mode": "off"}):
            plan = load_plan(write_plan(self.root, rewrites=ok))
            self.assertEqual(validate_plan(plan, self.root)["rewrites"], ok)
        plain = validate_plan(load_plan(write_plan(self.root)), self.root)
        self.assertNotIn("rewrites", plain)

    def test_bad_schema_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["schema"] = "nope"
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_schema")

    def test_stale_source_rejected(self):
        plan = load_plan(write_plan(self.root))
        (self.root / "ledger_wb.html").unlink()
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_stale_src")

    def test_duplicate_destination_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["moves"][1]["to"] = plan["moves"][0]["to"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_dup_dst")

    def test_duplicate_destination_dir_recovery_names_missing_filename(self):
        # Field bug: two moves onto an existing directory mean the
        # filename was omitted from 'to' — the recovery must say "full
        # file path" (and suggest the corrected path), not just "resolve
        # the collision", or the agent burns a round-trip.
        (self.root / "research").mkdir(exist_ok=True)
        plan = load_plan(write_plan(self.root))
        plan["moves"][0]["to"] = "research"
        plan["moves"][1]["to"] = "research"
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "plan_dup_dst")
        self.assertIn("full file path", cm.exception.recovery)
        # The suggested path uses the colliding move's own filename
        # (moves[1] raises: globex_dossier.md).
        self.assertIn("research/globex_dossier.md", cm.exception.recovery)

    def test_duplicate_destination_trailing_slash_gets_dir_recovery(self):
        # Same mistake, destination not yet on disk: a trailing slash
        # still names a directory, not a file path.
        plan = load_plan(write_plan(self.root))
        plan["moves"][0]["to"] = "research/pile/"
        plan["moves"][1]["to"] = "research/pile/"
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "plan_dup_dst")
        self.assertIn("full file path", cm.exception.recovery)

    def test_duplicate_destination_genuine_collision_recovery(self):
        # A real collision (no directory involved) still gets a recovery:
        # each move needs a distinct destination file path.
        plan = load_plan(write_plan(self.root))
        plan["moves"][1]["to"] = plan["moves"][0]["to"]
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "plan_dup_dst")
        self.assertIn("distinct destination", cm.exception.recovery)

    def test_path_escape_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["moves"][0]["to"] = "../outside.md"
        expect_code(self, lambda: validate_plan(plan, self.root), "path_escape")

    def test_incomplete_coverage_rejected(self):
        plan = load_plan(write_plan(self.root))
        del plan["moves"][3]  # ledger_wb.html no longer covered
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_incomplete")

    def test_root_dot_file_exempt_but_dot_dir_is_content(self):
        # fp4 contract (reference-rewrites PRD Phase 1): a root-level
        # dot-FILE (.env) is config/secrets, not sweepable — exempt from
        # coverage. A content dot-DIRECTORY (.claude) is ordinary content
        # and MUST be covered like any top-level entry.
        write(self.root / ".env", "SECRET=1\n")
        plan = load_plan(write_plan(self.root))
        validate_plan(plan, self.root)  # .env exempt: no raise
        write(self.root / ".claude" / "x.md", "# agent content\n")
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_incomplete")
        plan["keep"] = list(plan.get("keep", [])) + [".claude"]
        validate_plan(plan, self.root)  # covering it satisfies the gate

    def test_unresolved_and_keep_satisfy_coverage(self):
        plan = load_plan(write_plan(self.root))
        del plan["moves"][3]
        plan["unresolved"] = [{"path": "ledger_wb.html", "reason": "no signal"}]
        validate_plan(plan, self.root)  # should not raise

    def test_unresolved_null_path_rejected(self):
        # T3(a): apply stores u.get("path") — a null path lands in the
        # manifest and TypeErrors every later `status` walk (root / None).
        # The gate refuses instead of poisoning the state file.
        plan = load_plan(write_plan(self.root))
        plan["unresolved"] = [{"path": None, "reason": "x"}]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_unresolved")

    def test_unresolved_missing_or_blank_path_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["unresolved"] = [{"reason": "no path key at all"}]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_unresolved")
        plan = load_plan(write_plan(self.root))
        plan["unresolved"] = [{"path": "   ", "reason": "blank path"}]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_unresolved")

    def test_unresolved_non_dict_entry_rejected(self):
        # A bare string would slip past isinstance(u, dict) filters and
        # crash downstream shapers — one code for every malformed entry.
        plan = load_plan(write_plan(self.root))
        plan["unresolved"] = ["ledger_wb.html"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_unresolved")

    def test_unresolved_not_a_list_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["unresolved"] = "ledger_wb.html"
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_unresolved")

    def test_nested_moves_rejected(self):
        plan = load_plan(write_plan(self.root))
        # Moving the dir whole is legal; moving dir + contents separately is not.
        plan["moves"].append({"from": "lead-gen", "to": "archive/lead-gen"})
        plan["moves"].append(
            {"from": "lead-gen/README.md", "to": "elsewhere/README.md"}
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_nested_moves")

    def test_dir_move_whole_is_legal(self):
        plan = load_plan(write_plan(self.root))
        plan["moves"].append({"from": "lead-gen", "to": "archive/lead-gen"})
        out = validate_plan(plan, self.root)
        chains = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(chains["lead-gen"], "archive/lead-gen")

    def test_chain_moves_validate_and_apply(self):
        # Flipped pin (src round, data-loss fix): a destination equal to
        # another move's source is a CHAIN (a->b, b->c), and file chains
        # are LOSSLESS — resolve_chains keeps the hops and the executor
        # orders them (b->c before a->b), so no content is overwritten.
        # The intermediate must exist on disk (plan_stale_src gate), so
        # the corpus gains plain files a.md and b.md.
        write(self.root / "a.md", "A\n")
        write(self.root / "b.md", "B\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "a.md", "to": "b.md", "reason": "chain hop"},
                {"from": "b.md", "to": "c.md", "reason": "chain end"},
            ],
        )
        out = validate_plan(load_plan(plan_path), self.root, allow_partial=True)
        self.assertEqual(
            [m["from"] for m in out["moves"]], ["a.md", "b.md"]
        )  # validated, not rejected
        result = apply_plan(plan_path, self.root, execute=True, allow_partial=True)
        self.assertEqual(result["mode"], "executed")
        # Hops kept, content intact: b.md's content ends at c.md, a.md's
        # content ends at b.md. Nothing at c.md is a.md's "A\n".
        self.assertEqual((self.root / "c.md").read_text(), "B\n")
        self.assertEqual((self.root / "b.md").read_text(), "A\n")
        self.assertFalse((self.root / "a.md").exists())
        # The journal records the ACTUAL executed pair per move — both
        # hops, not a rewrite to c.md.
        lines = [
            json.loads(line)
            for line in Path(result["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        moves = {e["src"]: e["dst"] for e in lines if e.get("op") == "move"}
        self.assertEqual(moves, {"a.md": "b.md", "b.md": "c.md"})
        # undo replays the hops in reverse execution order: exact restore.
        undone = undo_journal(Path(result["journal"]), self.root, execute=True)
        self.assertEqual({o["status"] for o in undone["operations"]}, {"restored"})
        self.assertEqual((self.root / "a.md").read_text(), "A\n")
        self.assertEqual((self.root / "b.md").read_text(), "B\n")
        self.assertFalse((self.root / "c.md").exists())

    def test_three_hop_chain_applies_and_undoes(self):
        # A longer chain: c→d, b→c, a→b in that order — three files,
        # three destinations, every hop journaled, undo exact.
        write(self.root / "a.md", "A\n")
        write(self.root / "b.md", "B\n")
        write(self.root / "c.md", "C\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "a.md", "to": "b.md", "reason": "hop 1"},
                {"from": "b.md", "to": "c.md", "reason": "hop 2"},
                {"from": "c.md", "to": "d.md", "reason": "chain end"},
            ],
        )
        result = apply_plan(plan_path, self.root, execute=True, allow_partial=True)
        self.assertEqual(result["mode"], "executed")
        self.assertEqual((self.root / "d.md").read_text(), "C\n")
        self.assertEqual((self.root / "c.md").read_text(), "B\n")
        self.assertEqual((self.root / "b.md").read_text(), "A\n")
        self.assertFalse((self.root / "a.md").exists())
        undone = undo_journal(Path(result["journal"]), self.root, execute=True)
        self.assertEqual({o["status"] for o in undone["operations"]}, {"restored"})
        self.assertEqual((self.root / "a.md").read_text(), "A\n")
        self.assertEqual((self.root / "b.md").read_text(), "B\n")
        self.assertEqual((self.root / "c.md").read_text(), "C\n")
        self.assertFalse((self.root / "d.md").exists())

    def test_chain_hop_colliding_with_existing_final_dst_refused(self):
        # Atomicity: the chain's FINAL destination exists as an
        # unrelated file — the pre-flight refuses (dst_exists) and the
        # disk is untouched, byte for byte.
        write(self.root / "a.md", "A\n")
        write(self.root / "b.md", "B\n")
        write(self.root / "c.md", "unrelated\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "a.md", "to": "b.md", "reason": "chain hop"},
                {"from": "b.md", "to": "c.md", "reason": "chain end"},
            ],
        )
        before = snapshot_tree(self.root)
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True, allow_partial=True),
            "dst_exists",
        )
        self.assertEqual(
            prune_slopkeeper(snapshot_tree(self.root)), prune_slopkeeper(before)
        )


class TestEdits(TempCase):
    def test_plan_scoped_partial_flag(self):
        # A delta plan declares "partial": true; no --allow-partial needed.
        plan_path = write_edit_plan(
            self.root,
            partial=True,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["moves"]), 1)


class TestPhase2EditTargetResolution(TempCase):
    """D9 hardening: validate resolves edit targets through the same
    post-move mapper execute uses, and refuses a target no move can
    create with execute's own read error."""

    def _dir_move_plan(self, edits, **overrides):
        write(self.root / "misc/nvidia/reply.md", "# reply\n")
        base = dict(
            moves=[
                {
                    "from": "misc/nvidia",
                    "to": "projects/nvidia",
                    "reason": "restructure",
                }
            ],
            edits=edits,
            keep=[
                "docs",
                "lead-gen",
                "spark-glm-5.3",
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

    def test_validate_dir_move_descendant_edit_passes(self):
        # The D9 shape must validate: the edit's file is the move-SOURCE
        # path of a file inside the moved directory.
        write(self.root / "misc/nvidia/reply.md", "# reply\n")
        plan_path = self._dir_move_plan(
            [
                {
                    "file": "misc/nvidia/reply.md",
                    "find": "reply",
                    "replace": "RESPONSE",
                    "reason": "edit a dir-move descendant",
                }
            ]
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["edits"]), 1)

    def test_validate_fails_edit_file_missing(self):
        # An edit on a file that exists nowhere and is not moved is
        # edit_stale (the pre-move contract check).
        plan_path = self._dir_move_plan(
            [
                {
                    "file": "nope/ghost.md",
                    "find": "x",
                    "replace": "y",
                    "reason": "typo",
                }
            ]
        )
        expect_code(
            self, lambda: validate_plan(load_plan(plan_path), self.root), "edit_stale"
        )

    def test_validate_error_matches_execute_error(self):
        # A moved edit file whose SOURCE is missing cannot be created by
        # the move: validate raises the same code execute's read would
        # raise (edit_unreadable), naming both paths — instead of
        # passing validate and failing execute (the 0.1.9 D9 class).
        plan_path = self._dir_move_plan(
            [
                {
                    "file": "misc/nvidia/ghost.md",
                    "find": "x",
                    "replace": "y",
                    "reason": "source vanished",
                }
            ]
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "edit_unreadable",
        )
        # The code is the same one execute's read path raises for a
        # missing file — but validate refuses first, so execute never
        # discovers it ("never discovered at execute time").
        from slopkeeper.apply import _read_edit_text

        expect_code(
            self,
            lambda: _read_edit_text(
                self.root / "projects" / "nvidia" / "ghost.md", "ghost"
            ),
            "edit_unreadable",
        )


class TestPhase6DeadGlobGate(TempCase):
    """D2's validate-time gate: a prefix_rewrites entry whose glob/brace
    citations cannot be rewritten coherently post-move is refused, with
    a conscious per-entry override."""

    def _glob_corpus(self):
        write(self.root / "reports/_org/2026-09-17_two-stage-pipeline-a.md", "# a\n")
        write(self.root / "reports/_org/2026-09-17_two-stage-pipeline-b.md", "# b\n")
        write(self.root / "reports/_org/other.md", "# other\n")
        write(self.root / "notes/team-sync.md", "cite reports/_org/x*.md here\n")

    def _plan(self, **overrides):
        base = dict(
            moves=[
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-a.md",
                    "to": "projects/vision-pipeline/2026-09-17_two-stage-pipeline-a.md",
                    "reason": "vision",
                },
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-b.md",
                    "to": "projects/vision-pipeline/2026-09-17_two-stage-pipeline-b.md",
                    "reason": "vision",
                },
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-glm-5.3",
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
        base.update(overrides)
        return write_plan(self.root, **base)

    def test_validate_rejects_dead_glob_prefix_rewrite(self):
        # A glob whose matches SPLIT across destinations cannot be
        # rewritten coherently — the swap would point at nothing (E2
        # narrowed the gate: zero-match patterns are convention strings
        # and pass).
        self._glob_corpus()
        write(
            self.root / "notes/team-sync.md",
            "cite reports/_org/2026-09-17_two-stage-pipeline*.md here\n",
        )
        plan_path = self._plan(
            moves=[
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-a.md",
                    "to": "projects/vision-pipeline/2026-09-17_two-stage-pipeline-a.md",
                    "reason": "vision",
                },
                {
                    "from": "reports/_org/2026-09-17_two-stage-pipeline-b.md",
                    "to": "other/place/2026-09-17_two-stage-pipeline-b.md",
                    "reason": "vision",
                },
            ],
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ],
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "prefix_rewrite_dead_glob",
        )

    def test_validate_allows_convention_glob(self):
        # E2: a pattern naming NO corpus file is a convention string
        # (the standup-digest output pattern) — the entry applies the
        # rename to it; refusing it regressed 0.1.9.
        self._glob_corpus()
        write(
            self.root / "skills/standup-digest/SKILL.md",
            "output: reports/_org/*_standup-rollup.md\n",
        )
        plan_path = self._plan(
            keep=[
                "docs",
                "lead-gen",
                "spark-glm-5.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
                "reports",
                "skills",
            ],
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ],
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["prefix_rewrites"]), 1)

    def test_validate_allows_placeholder_glob(self):
        # E2: <placeholder> groups are wildcards — the screens pattern
        # matches the one file that moves (to a single destination), so
        # the entry is coherent and the rewrite follows the files.
        self._glob_corpus()
        write(
            self.root / "skills/product-sync-prep/SKILL.md",
            "screens: `reports/_org/<date>_*-screens.md`\n",
        )
        plan_path = self._plan(
            keep=[
                "docs",
                "lead-gen",
                "spark-glm-5.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
                "reports",
                "skills",
            ],
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ],
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["prefix_rewrites"]), 1)

    def test_validate_allows_coherent_glob(self):
        # The glob reports/_org/2026-09-17_two-stage-pipeline*.md names
        # two files that both move into projects/vision-pipeline — the
        # entry is coherent and validates.
        self._glob_corpus()
        write(
            self.root / "notes/team-sync.md",
            "cite reports/_org/2026-09-17_two-stage-pipeline*.md here\n",
        )
        plan_path = self._plan(
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ]
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["prefix_rewrites"]), 1)

    def test_validate_acknowledge_broken_allows(self):
        self._glob_corpus()
        plan_path = self._plan(
            prefix_rewrites=[
                {
                    "from": "reports/_org/",
                    "to": "reports/org/",
                    "reason": "renamed",
                    "acknowledge_broken": True,
                }
            ]
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertTrue(plan["prefix_rewrites"][0]["acknowledge_broken"])
        md = render_markdown(plan)
        self.assertIn("acknowledged broken", md)

    def test_validate_ignores_external_only_citations(self):
        # The gate only scans tokens under the entry's own prefix: a
        # glob citing /tmp (out of corpus, the agent's problem) never
        # trips it, and a coherent in-corpus glob passes.
        self._glob_corpus()
        write(
            self.root / "notes/team-sync.md",
            "cite reports/_org/2026-09-17_two-stage-pipeline*.md here\n",
        )
        write(self.root / "elsewhere/scratch.md", "cite /tmp/other*.md here\n")
        plan_path = self._plan(
            prefix_rewrites=[
                {"from": "reports/_org/", "to": "reports/org/", "reason": "renamed"}
            ],
            keep=[
                "docs",
                "lead-gen",
                "spark-glm-5.3",
                "acme_dossier.md",
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "notes",
                "reports",
                "elsewhere",
            ],
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(len(plan["prefix_rewrites"]), 1)


class TestValidatorErrorCodes(TempCase):
    """TEST_PLAN plan.py matrix rows marked missing."""

    def test_wrong_root_rejected(self):
        plan = load_plan(write_plan(self.root, corpus_root=str(self.tmp / "elsewhere")))
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_wrong_root")

    def test_empty_moves_rejected(self):
        plan = load_plan(write_plan(self.root, moves=[]))
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_no_moves")

    def test_zero_move_partial_plan_valid(self):
        # T7 (parent note): a draft that filed every new file as
        # unresolved is a valid PARTIAL plan — empty moves, backlog
        # intact. Non-partial plans still need at least one move.
        plan_path = write_plan(self.root, moves=[], partial=True)
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(plan["moves"], [])
        self.assertTrue(plan["partial"])

    def test_duplicate_source_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["moves"][1]["from"] = plan["moves"][0]["from"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_dup_src")

    def test_noop_move_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["moves"][0]["to"] = plan["moves"][0]["from"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_noop_move")

    def test_dst_under_src_rejected(self):
        # A destination inside a moved source dir would nest a file
        # under a file after the rename.
        plan = load_plan(
            write_plan(
                self.root,
                moves=[
                    {"from": "lead-gen", "to": "archive/lead-gen", "reason": "dir"},
                    {
                        "from": "ledger_wb.html",
                        "to": "lead-gen/ledger_wb.html",
                        "reason": "lands inside the source",
                    },
                ],
            )
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_dst_under_src")

    def test_bad_keep_rejected(self):
        plan = load_plan(
            write_plan(self.root, keep=["WATCHLIST.md", {"reason": "no path key"}])
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_keep")

    def test_bad_edit_rejected(self):
        plan = load_plan(
            write_plan(
                self.root,
                edits=[{"file": "WATCHLIST.md", "find": "acme_dossier.md"}],
            )
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_edit")

    def test_bad_folder_rejected(self):
        plan = load_plan(write_plan(self.root, folders=[{"kind": "case"}]))
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_folder")

    def test_bad_kind_rejected(self):
        plan = load_plan(
            write_plan(
                self.root,
                folders=[{"path": "research/dossiers", "kind": "wizard"}],
            )
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_kind")


class TestDocsMagicSkip(TempCase):
    """The docs/ magic string is dead (DESIGN.md §6.1, settled).

    An uncovered top-level docs/ fails plan_incomplete like any other
    entry; what the plan deliberately ignored is a recorded "skip".
    """

    def test_uncovered_docs_fails_like_any_entry(self):
        plan = load_plan(write_plan(self.root))
        plan["keep"] = [k for k in plan["keep"] if k != "docs"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_incomplete")


class TestSkipField(TempCase):
    """The plan-level skip list (DESIGN.md §6.1): "we never looked at
    it" is a recorded decision, not a silent default."""

    def test_skip_satisfies_coverage(self):
        plan = load_plan(write_plan(self.root))
        plan["keep"] = [k for k in plan["keep"] if k != "docs"]
        plan["skip"] = ["docs"]
        out = validate_plan(plan, self.root)
        self.assertEqual(out["skip"], ["docs"])

    def test_skip_of_missing_entry_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["skip"] = ["nope"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_skip")

    def test_skip_nested_path_rejected(self):
        # Names only: a skip is a top-level decision, not a path list.
        plan = load_plan(write_plan(self.root))
        plan["skip"] = ["docs/site"]
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_skip")

    def test_skip_non_list_rejected(self):
        plan = load_plan(write_plan(self.root))
        plan["skip"] = "docs"
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_skip")

    def test_skip_of_a_loose_file_works(self):
        # skip names entries, not just folders: a loose file can be
        # deliberately unexamined too.
        plan = load_plan(write_plan(self.root))
        plan["keep"] = [k for k in plan["keep"] if k != "WATCHLIST.md"]
        plan["skip"] = ["WATCHLIST.md"]
        validate_plan(plan, self.root)  # should not raise


# --- malformed plan documents: the load/validate error surface ---


class TestMalformedPlans(TempCase):
    def test_plan_not_json(self):
        p = self.root.parent / "bad.json"
        p.write_text("{nope", encoding="utf-8")
        expect_code(self, lambda: load_plan(p), "plan_not_json")

    def test_plan_not_object(self):
        p = self.root.parent / "arr.json"
        p.write_text("[]", encoding="utf-8")
        expect_code(self, lambda: load_plan(p), "plan_not_object")

    def test_move_missing_keys_is_plan_bad_move(self):
        plan = json.loads(write_plan(self.root).read_text(encoding="utf-8"))
        plan["moves"] = [{"from": "a.md"}]  # no "to"
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_move")


class TestScanPaths(TempCase):
    """Phase 6: scan_paths — the declared override for the invisible layer."""

    def _ignored_target_plan(self, **over):
        write(self.root / ".cache" / "notes.md", "cite: docs/canonical.md\n")
        (self.root / ".slopkeeperignore").write_text(".cache/\n", encoding="utf-8")
        kw = {
            "moves": [
                {
                    "from": "docs/canonical.md",
                    "to": "canon/canonical.md",
                    "reason": "pile",
                }
            ],
            "edits": [
                {
                    "file": ".cache/notes.md",
                    "find": "docs/canonical.md",
                    "replace": "canon/canonical.md",
                    "reason": "fix citation in an ignored dir",
                }
            ],
            "partial": True,
        }
        kw.update(over)
        return write_plan(self.root, **kw)

    def test_edit_target_ignored_requires_declaration(self):
        # Without the declaration: edit_target_ignored stands.
        plan_path = self._ignored_target_plan(scan_paths=None)
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "edit_target_ignored",
        )

    def test_edit_target_allowed_when_declared_via_scan_paths(self):
        # Declared: the edit validates, kept relative (it is an
        # in-corpus path under a declared root).
        plan_path = self._ignored_target_plan(scan_paths=[str(self.root / ".cache")])
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["edits"][0]["file"], ".cache/notes.md")

    def test_edit_target_absolute_in_corpus_normalized_relative(self):
        # The pinned corpus-root rule wins over declarations: an
        # absolute path under the CORPUS root is an in-root edit, even
        # when scan_paths are declared.
        plan_path = self._ignored_target_plan(
            scan_paths=[str(self.root / ".cache")],
            edits=[
                {
                    "file": str(self.root / ".cache" / "notes.md"),
                    "find": "docs/canonical.md",
                    "replace": "canon/canonical.md",
                    "reason": "fix citation in an ignored dir",
                }
            ],
        )
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["edits"][0]["file"], ".cache/notes.md")

    def test_edit_target_under_external_scan_root_kept_absolute(self):
        # A scan root OUTSIDE the corpus behaves like an external root:
        # the absolute edit path is self-describing in journal and
        # render, and undo replays through the same resolution.
        ext = (self.tmp / "out-corpus").resolve()
        target = write(ext / "skill.md", "cite: docs/canonical.md\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "docs/canonical.md",
                    "to": "canon/canonical.md",
                    "reason": "pile",
                }
            ],
            edits=[
                {
                    "file": str(target),
                    "find": "docs/canonical.md",
                    "replace": "canon/canonical.md",
                    "reason": "fix citation",
                }
            ],
            scan_paths=[str(ext)],
            partial=True,
        )
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["edits"][0]["file"], str(target))

    def test_scan_path_must_be_absolute(self):
        plan_path = write_plan(self.root, scan_paths=[".cache"], partial=True)
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_scan_path",
        )

    def test_scan_path_must_exist(self):
        plan_path = write_plan(
            self.root, scan_paths=[str(self.root / "nope")], partial=True
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_scan_path",
        )

    def test_scan_path_must_be_a_directory(self):
        plain = write(self.root / "plain.txt", "not a dir\n")
        plan_path = write_plan(self.root, scan_paths=[str(plain)], partial=True)
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_scan_path",
        )

    def test_scan_path_skip_list_refused(self):
        # Tool layer is never read — naming it is a typo'd plan, refused.
        (self.root / ".git").mkdir()
        plan_path = write_plan(
            self.root, scan_paths=[str(self.root / ".git")], partial=True
        )
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "scan_path_skip_list",
        )

    def test_scan_paths_must_be_a_list(self):
        plan_path = write_plan(self.root, scan_paths=".cache", partial=True)
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_scan_path",
        )

    def test_scan_paths_normalize_to_resolved_posix(self):
        plan_path = self._ignored_target_plan(scan_paths=[str(self.root / ".cache")])
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["scan_paths"], [str(self.root / ".cache")])


class TestRuleMoves(TempCase):
    """Phase 8: rules expand into concrete preserve-name moves at validate."""

    def test_rule_expands_deterministically(self):
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/b.md": "b\n",
                "reports/_org/sub/c.md": "c\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root, rules=[{"match": "reports/_org/", "to": "reports/org"}]
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        # preserve-name flatten: nested files land as to/<basename>
        self.assertEqual(by_src["reports/_org/a.md"], "reports/org/a.md")
        self.assertEqual(by_src["reports/_org/sub/c.md"], "reports/org/c.md")
        expanded = sorted(
            m["from"] for m in out["moves"] if m["from"].startswith("reports/_org/")
        )
        self.assertEqual(
            expanded,
            ["reports/_org/a.md", "reports/_org/b.md", "reports/_org/sub/c.md"],
        )
        # every expanded reason names its rule — the gate shows provenance
        self.assertTrue(
            all(
                m["reason"].startswith("rule: reports/_org/")
                for m in out["moves"]
                if m["from"].startswith("reports/_org/")
            )
        )
        # the rule summary rides on the validated plan for the render row:
        # matched vs expanded (explicit/keep coverage) and flattened (B5)
        # are visible counts now
        self.assertEqual(
            out["rules"],
            [
                {
                    "match": "reports/_org/",
                    "to": "reports/org",
                    "kind": "prefix",
                    "matched": 3,
                    "expanded": 3,
                    "flattened": 1,
                }
            ],
        )

    def test_rule_conflict_refused(self):
        build_corpus(
            self.root,
            {"reports/_org/a.md": "a\n", "reports/_org/sub/b.md": "b\n"},
        )
        # B3 precedence: the FIRST matching rule wins and later rules
        # count as superseded — carve-outs are ordered before catch-alls
        plan = load_plan(
            write_plan(
                self.root,
                rules=[
                    {"match": "reports/_org/sub/", "to": "elsewhere"},
                    {"match": "reports/_org/", "to": "reports/org"},
                ],
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(by_src["reports/_org/sub/b.md"], "elsewhere/b.md")
        self.assertEqual(by_src["reports/_org/a.md"], "reports/org/a.md")
        rules = {r["match"]: r for r in out["rules"]}
        self.assertEqual(rules["reports/_org/sub/"]["expanded"], 1)
        self.assertNotIn("superseded", rules["reports/_org/sub/"])
        self.assertEqual(rules["reports/_org/"]["expanded"], 1)
        self.assertEqual(rules["reports/_org/"]["superseded"], 1)
        # Reversed order: the catch-all wins everything, the carve-out
        # is superseded — order is the contract, so the counts name it.
        plan = load_plan(
            write_plan(
                self.root,
                rules=[
                    {"match": "reports/_org/", "to": "reports/org"},
                    {"match": "reports/_org/sub/", "to": "elsewhere"},
                ],
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(by_src["reports/_org/sub/b.md"], "reports/org/b.md")
        self.assertEqual(
            {r["match"]: r.get("superseded", 0) for r in out["rules"]},
            {"reports/_org/": 0, "reports/_org/sub/": 1},
        )
        # an explicit move beats a rule disagreeing with it — no conflict,
        # one move entry, the rule just doesn't cover that file
        explicit = [
            {
                "from": "reports/_org/a.md",
                "to": "other/a.md",
                "reason": "explicit",
            },
            {
                "from": "acme_dossier.md",
                "to": "research/dossiers/acme_dossier.md",
                "reason": "r",
            },
            {
                "from": "globex_dossier.md",
                "to": "research/dossiers/globex_dossier.md",
                "reason": "r",
            },
            {
                "from": "handoff-tp2.md",
                "to": "spark-glm-5.3/handoff-tp2.md",
                "reason": "r",
            },
            {
                "from": "ledger_wb.html",
                "to": "archive/scraps/ledger_wb.html",
                "reason": "r",
            },
        ]
        plan = load_plan(
            write_plan(
                self.root,
                moves=explicit,
                rules=[{"match": "reports/_org/", "to": "reports/org"}],
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(by_src["reports/_org/a.md"], "other/a.md")
        self.assertEqual(
            {m["from"]: m["to"] for m in out["moves"]}.get("reports/_org/sub/b.md"),
            "reports/org/b.md",  # the rule still covers the rest
        )
        # same destination: the explicit move wins, no duplicate entry
        agreeing = [dict(explicit[0], to="reports/org/a.md")] + explicit[1:]
        plan = load_plan(
            write_plan(
                self.root,
                moves=agreeing,
                rules=[{"match": "reports/_org/", "to": "reports/org"}],
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(
            sum(1 for m in out["moves"] if m["from"] == "reports/_org/a.md"), 1
        )

    def test_rule_glob(self):
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/b.txt": "b\n",
                "reports/_org/sub/c.md": "c\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root, rules=[{"match": "reports/_org/*.md", "to": "reports/org"}]
            )
        )
        out = validate_plan(plan, self.root)
        srcs = {m["from"] for m in out["moves"]}
        self.assertIn("reports/_org/a.md", srcs)
        self.assertNotIn("reports/_org/b.txt", srcs)
        # `*` stays inside one path segment
        self.assertNotIn("reports/_org/sub/c.md", srcs)
        # a whole-segment `**` crosses directories (and requires one)
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "reports/_org/**/*.md", "to": "reports/org"}],
            )
        )
        out = validate_plan(plan, self.root)
        srcs = {m["from"] for m in out["moves"]}
        self.assertIn("reports/_org/sub/c.md", srcs)
        self.assertNotIn("reports/_org/a.md", srcs)

    def test_rule_dest_outside_root_refused(self):
        plan = load_plan(
            write_plan(self.root, rules=[{"match": "reports/_org/", "to": "../out"}])
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "path_escape")

    def test_rule_dest_existing_file_refused(self):
        plan = load_plan(
            write_plan(self.root, rules=[{"match": "docs/", "to": "WATCHLIST.md"}])
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_rules")

    def test_rule_matched_nothing_refused(self):
        plan = load_plan(write_plan(self.root, rules=[{"match": "nope/", "to": "x"}]))
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_rules")

    def test_bad_rules_shape_refused(self):
        for bad in (
            "rules",
            ["match-only"],
            [{"match": "a/"}],
            [{"match": "a/", "to": "b", "extra": 1}],
            [{"match": "", "to": "b"}],
            [{"match": "a/", "to": ""}],
        ):
            plan = load_plan(write_plan(self.root, rules=bad))
            expect_code(
                self,
                lambda p=plan: validate_plan(p, self.root),
                "plan_bad_rules",
            )

    def test_keep_and_unresolved_win_over_rules(self):
        build_corpus(
            self.root,
            {"reports/_org/a.md": "a\n", "reports/_org/b.md": "b\n"},
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "reports/_org/", "to": "reports/org"}],
                keep=[
                    "WATCHLIST.md",
                    "docs",
                    "lead-gen",
                    "spark-glm-5.3",
                    {"path": "reports/_org/a.md", "reason": "stays"},
                ],
                unresolved=[{"path": "reports/_org/b.md", "reason": "human decides"}],
            )
        )
        out = validate_plan(plan, self.root)
        srcs = {m["from"] for m in out["moves"]}
        self.assertNotIn("reports/_org/a.md", srcs)
        self.assertNotIn("reports/_org/b.md", srcs)

    def test_rule_files_inside_explicit_dir_move_are_skipped(self):
        # They travel with the directory; expanding them would be a
        # plan_nested_moves. The rule just doesn't double-cover them.
        build_corpus(self.root, {"big/x.md": "x\n", "big/sub/y.md": "y\n"})
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "big/", "to": "elsewhere"}],
                moves=[
                    {"from": "big", "to": "moved-big", "reason": "whole dir"},
                    {
                        "from": "acme_dossier.md",
                        "to": "research/dossiers/acme_dossier.md",
                        "reason": "r",
                    },
                    {
                        "from": "globex_dossier.md",
                        "to": "research/dossiers/globex_dossier.md",
                        "reason": "r",
                    },
                    {
                        "from": "handoff-tp2.md",
                        "to": "spark-glm-5.3/handoff-tp2.md",
                        "reason": "r",
                    },
                    {
                        "from": "ledger_wb.html",
                        "to": "archive/scraps/ledger_wb.html",
                        "reason": "r",
                    },
                ],
            )
        )
        out = validate_plan(plan, self.root)  # no plan_nested_moves
        self.assertEqual(
            sum(1 for m in out["moves"] if m["from"].startswith("big/")), 0
        )

    def test_pure_rule_plan_needs_no_moves(self):
        build_corpus(self.root, {"misc/a.md": "a\n"})
        plan = load_plan(
            write_plan(
                self.root,
                moves=[],
                partial=True,
                rules=[{"match": "misc/", "to": "sorted"}],
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(len(out["moves"]), 1)

    def test_keep_shadowing_warns_not_silences(self):
        # B4: a keep of a subtree must not silently disable rules under
        # it — the rule's matches are counted and the gate is warned.
        build_corpus(
            self.root,
            {"reports/_org/a.md": "a\n", "reports/_org/b.md": "b\n"},
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "reports/_org/", "to": "reports/org"}],
                keep=["reports", "WATCHLIST.md", "docs", "lead-gen", "spark-glm-5.3"],
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(out["rules"][0]["expanded"], 0)
        self.assertEqual(out["rules"][0]["shadowed"], 2)
        self.assertIn("keep/unresolved", " ".join(out["rule_warnings"]))

    def test_preserve_tree_keeps_subpaths(self):
        # B5: "preserve": "tree" keeps each file's path below the match
        # prefix instead of flattening it into <to>/<basename>.
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/sub/c.md": "c\n",
                "reports/_org/sub/deep/d.md": "d\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[
                    {
                        "match": "reports/_org/",
                        "to": "reports/org",
                        "preserve": "tree",
                    }
                ],
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(by_src["reports/_org/a.md"], "reports/org/a.md")
        self.assertEqual(by_src["reports/_org/sub/c.md"], "reports/org/sub/c.md")
        self.assertEqual(
            by_src["reports/_org/sub/deep/d.md"], "reports/org/sub/deep/d.md"
        )
        self.assertNotIn("flattened", out["rules"][0])

    def test_preserve_tree_glob_uses_static_prefix(self):
        build_corpus(
            self.root,
            {"reports/_org/sub/c.md": "c\n", "reports/_org/loose.md": "l\n"},
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[
                    {
                        "match": "reports/_org/**/*.md",
                        "to": "reports/org",
                        "preserve": "tree",
                    }
                ],
            )
        )
        out = validate_plan(plan, self.root)
        by_src = {m["from"]: m["to"] for m in out["moves"]}
        self.assertEqual(by_src["reports/_org/sub/c.md"], "reports/org/sub/c.md")
        self.assertNotIn("reports/_org/loose.md", by_src)  # `*` needs one segment

    def test_bad_preserve_value_refused(self):
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "docs/", "to": "d2", "preserve": "flat"}],
            )
        )
        expect_code(self, lambda: validate_plan(plan, self.root), "plan_bad_rules")

    def test_prefix_rewrites_validated(self):
        # B8: a declared rename of a dir emptied file by file.
        build_corpus(self.root, {"reports/_org/a.md": "a\n"})
        plan = load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[
                    {
                        "from": "reports/_org/a.md",
                        "to": "reports/org/a.md",
                        "reason": "one by one",
                    }
                ],
                prefix_rewrites=[{"from": "reports/_org/", "to": "reports/org/"}],
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(
            out["prefix_rewrites"],
            [
                {
                    "from": "reports/_org",
                    "to": "reports/org",
                    "reason": "",
                    "acknowledge_broken": False,
                }
            ],
        )
        # from must be an existing dir
        bad = load_plan(
            write_plan(
                self.root,
                partial=True,
                prefix_rewrites=[{"from": "nope/", "to": "x/"}],
            )
        )
        expect_code(
            self, lambda: validate_plan(bad, self.root), "plan_bad_prefix_rewrite"
        )
        # from == to refused
        bad = load_plan(
            write_plan(
                self.root,
                partial=True,
                prefix_rewrites=[{"from": "docs/", "to": "docs/"}],
            )
        )
        expect_code(
            self, lambda: validate_plan(bad, self.root), "plan_bad_prefix_rewrite"
        )

    def test_emptied_dir_suggestions(self):
        # B8: a directory emptied file by file gets a destination
        # breakdown, and a ready-to-paste suggestion on a clear majority.
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/b.md": "b\n",
                "reports/_org/sub/c.md": "c\n",
                "stray.md": "s\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[
                    {
                        "from": "reports/_org/a.md",
                        "to": "projects/x/a.md",
                        "reason": "r",
                    },
                    {
                        "from": "reports/_org/b.md",
                        "to": "projects/x/b.md",
                        "reason": "r",
                    },
                    {
                        "from": "reports/_org/sub/c.md",
                        "to": "projects/x/sub/c.md",
                        "reason": "r",
                    },
                    {"from": "stray.md", "to": "misc/stray.md", "reason": "r"},
                ],
            )
        )
        out = validate_plan(plan, self.root)
        sugs = {e["dir"]: e for e in out["emptied_dir_suggestions"]}
        self.assertIn("reports/_org", sugs)
        self.assertEqual(sugs["reports/_org"]["moved_files"], 3)
        self.assertEqual(
            sugs["reports/_org"]["suggestion"],
            {"from": "reports/_org/", "to": "projects/x/"},
        )

    def test_emptied_dir_not_suggested_without_majority(self):
        build_corpus(
            self.root,
            {"pile/a.md": "a\n", "pile/b.md": "b\n", "stray.md": "s\n"},
        )
        plan = load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[
                    {"from": "pile/a.md", "to": "one/a.md", "reason": "r"},
                    {"from": "pile/b.md", "to": "two/b.md", "reason": "r"},
                    {"from": "stray.md", "to": "misc/stray.md", "reason": "r"},
                ],
            )
        )
        out = validate_plan(plan, self.root)
        sugs = {e["dir"]: e for e in out["emptied_dir_suggestions"]}
        self.assertIn("pile", sugs)
        self.assertNotIn("suggestion", sugs["pile"])

    def test_rule_scales(self):
        # PRD checkpoint: a 10k-file tree expands in seconds from ONE
        # authored entry. The bound is generous (CI tmpfs variance);
        # the old O(n²) dst-under-src and dir-key checks took minutes.
        big = self.root / "big"
        for i in range(10000):
            write(big / f"f{i:05d}.md", "x\n")
        plan = load_plan(
            write_plan(self.root, partial=True, rules=[{"match": "big/", "to": "huge"}])
        )
        t0 = time.monotonic()
        out = validate_plan(plan, self.root)
        elapsed = time.monotonic() - t0
        self.assertEqual(len(out["moves"]), 4 + 10000)
        self.assertLess(elapsed, 10.0, f"expansion took {elapsed:.1f}s")


class TestStragglerReport(TempCase):
    """Phase 8: files left behind in folders the plan removes files from."""

    def test_stragglers_listed_with_nested_leftovers(self):
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/b.txt": "b\n",
                "reports/_org/sub/deep.md": "d\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "reports/_org/*.md", "to": "reports/org"}],
            )
        )
        out = validate_plan(plan, self.root)
        report = straggler_report(self.root, out)
        by_dir = {s["dir"]: s for s in report}
        self.assertIn("reports/_org", by_dir)
        remaining = by_dir["reports/_org"]["remaining"]
        self.assertIn("reports/_org/b.txt", remaining)
        self.assertIn("reports/_org/sub/deep.md", remaining)  # recursion
        self.assertNotIn("reports/_org/a.md", remaining)  # moved
        self.assertEqual(by_dir["reports/_org"]["count"], 2)

    def test_full_sweep_has_no_stragglers(self):
        build_corpus(
            self.root,
            {"reports/_org/a.md": "a\n", "reports/_org/sub/b.md": "b\n"},
        )
        plan = load_plan(
            write_plan(
                self.root, rules=[{"match": "reports/_org/", "to": "reports/org"}]
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(straggler_report(self.root, out), [])

    def test_root_scan_stays_top_level(self):
        # A root-file move scans "." non-recursively: its leftovers are
        # top-level entries (coverage's job), and recursing would list
        # the whole corpus behind any root-file move.
        plan = load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[
                    {
                        "from": "acme_dossier.md",
                        "to": "research/dossiers/acme_dossier.md",
                        "reason": "r",
                    }
                ],
            )
        )
        out = validate_plan(plan, self.root)
        report = straggler_report(self.root, out)
        by_dir = {s["dir"]: s for s in report}
        self.assertIn(".", by_dir)
        self.assertIn("globex_dossier.md", by_dir["."]["remaining"])
        self.assertNotIn("docs/canonical.md", by_dir["."]["remaining"])

    def test_unresolved_and_dir_moves_are_not_stragglers(self):
        build_corpus(self.root, {"misc/a.md": "a\n", "misc/sub/b.md": "b\n"})
        plan = load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[
                    {"from": "misc/sub", "to": "elsewhere/sub", "reason": "dir"},
                    {
                        "from": "docs/canonical.md",
                        "to": "docs2/canonical.md",
                        "reason": "r",
                    },
                ],
                unresolved=[{"path": "misc/a.md", "reason": "human decides"}],
            )
        )
        out = validate_plan(plan, self.root)
        self.assertEqual(straggler_report(self.root, out), [])


if __name__ == "__main__":
    unittest.main()
