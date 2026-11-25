# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P4 — plan scoping, case-collision gate, and the dedup action.

The scale roadmap's "complex corpora" trio: a bounded tend (plan.scope)
that covers/straggles/diffs within one subtree; a casefolded destination
collision check that prices the case-insensitive-mount clobber; and a
journaled dedup move (never a delete) that archives a byte-identical copy.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from helpers import (
    TempCase,
    build_corpus,
    expect_code,
    prune_slopkeeper,
    snapshot_tree,
    write,
    write_plan,
)
from slopkeeper import cli
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError, corpus_fingerprint
from slopkeeper.plan import load_plan, render_markdown, straggler_report, validate_plan


def _run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(list(argv))
    return code, buf.getvalue()


class TestPlanScope(TempCase):
    def _scope_corpus(self):
        # inbox/ is the subtree to tend; sibling/ must stay untouched.
        return {
            "inbox/a.md": "# a\n",
            "inbox/b.md": "# b\n",
            "sibling/keep.md": "# keep\n",
        }

    def _scoped_plan(self, **over):
        build_corpus(self.root, self._scope_corpus())
        base = dict(
            scope="inbox",
            moves=[
                {"from": "inbox/a.md", "to": "inbox/notes/a.md", "reason": "r"},
                {"from": "inbox/b.md", "to": "inbox/notes/b.md", "reason": "r"},
            ],
            keep=[],
            unresolved=[],
            folders=[],
            based_on={
                "corpus_fingerprint": corpus_fingerprint(self.root, scope="inbox")
            },
        )
        base.update(over)
        return write_plan(self.root, **base)

    def test_scoped_plan_validates_out_of_scope_uncovered(self):
        # The sibling/ subtree is NOT this tend's problem: coverage
        # operates within scope, so an uncovered sibling is not a complaint.
        plan = validate_plan(load_plan(self._scoped_plan()), self.root)
        self.assertEqual(plan["scope"], "inbox")
        self.assertEqual(len(plan["moves"]), 2)

    def test_unscoped_equivalent_would_demand_sibling_coverage(self):
        # Same moves, no scope → plan_incomplete on sibling/ (the contrast
        # that proves scoping actually narrows coverage).
        plan_path = self._scoped_plan()
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        del raw["scope"]
        plan_path.write_text(json.dumps(raw), encoding="utf-8")
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_incomplete",
        )

    def test_move_entering_scope_refused(self):
        # A move pulling an out-of-scope file INTO scope escapes the bound.
        plan_path = self._scoped_plan()
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        raw["moves"].append(
            {"from": "sibling/keep.md", "to": "inbox/notes/keep.md", "reason": "r"}
        )
        # scope fingerprint must still bind; the escape is the move check.
        plan_path.write_text(json.dumps(raw), encoding="utf-8")
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_scope_escape",
        )

    def test_move_leaving_scope_refused(self):
        plan_path = self._scoped_plan()
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        raw["moves"][0]["to"] = "elsewhere/a.md"  # leaves the scope
        plan_path.write_text(json.dumps(raw), encoding="utf-8")
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_scope_escape",
        )

    def test_scope_must_be_existing_dir(self):
        expect_code(
            self,
            lambda: validate_plan(
                load_plan(self._scoped_plan(scope="ghost")), self.root
            ),
            "plan_scope_escape",
        )

    def test_scoped_fingerprint_gate_ignores_sibling_change(self):
        # The whole point of a bounded tend: a change in a sibling scope is
        # NOT drift against this plan; a change inside scope is.
        plan_path = self._scoped_plan()
        # sibling change: dry-run apply still passes the staleness gate.
        (self.root / "sibling/keep.md").write_text("# keep changed\n")
        data = apply_plan(plan_path, self.root)  # dry-run
        self.assertEqual(data["mode"], "dry-run")
        # in-scope change (a source's CONTENT, so coverage still holds): the
        # scoped fingerprint moves → corpus_changed.
        (self.root / "inbox/a.md").write_text("# a, edited\n")
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root)
        self.assertEqual(cm.exception.code, "corpus_changed")

    def test_render_lists_only_scope_moves_and_names_scope(self):
        plan = validate_plan(load_plan(self._scoped_plan()), self.root)
        md = render_markdown(plan)
        self.assertIn("Scope: `inbox/`", md)
        self.assertIn("`inbox/a.md`", md)
        self.assertIn("`inbox/notes/a.md`", md)

    def test_straggler_and_emptied_within_scope(self):
        build_corpus(self.root, self._scope_corpus())
        # a.md moves; b.md stays → b.md is a straggler inside scope.
        plan_path = write_plan(
            self.root,
            scope="inbox",
            moves=[{"from": "inbox/a.md", "to": "inbox/notes/a.md", "reason": "r"}],
            keep=["b.md"],
            unresolved=[],
            folders=[],
            based_on={
                "corpus_fingerprint": corpus_fingerprint(self.root, scope="inbox")
            },
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        report = straggler_report(self.root, plan)
        # sibling/ is never a straggler source (no move touched it).
        self.assertTrue(all(not s["dir"].startswith("sibling") for s in report))


class TestCaseCollision(TempCase):
    def _case_plan(self):
        write(self.root / "A.md", "a\n")
        write(self.root / "B.md", "b\n")
        return write_plan(
            self.root,
            partial=True,
            moves=[
                {"from": "A.md", "to": "x/pile.md", "reason": "r"},
                {"from": "B.md", "to": "x/PILE.md", "reason": "r"},
            ],
        )

    def test_case_collision_is_warning_by_default(self):
        plan = validate_plan(load_plan(self._case_plan()), self.root)
        self.assertTrue(plan.get("case_collisions"))
        folds = {(c["dst"].casefold()) for c in plan["case_collisions"]}
        self.assertIn("x/pile.md", folds)

    def test_strict_case_refuses(self):
        expect_code(
            self,
            lambda: validate_plan(
                load_plan(self._case_plan()), self.root, strict_case=True
            ),
            "plan_dst_case_collision",
        )

    def test_distinct_case_destinations_pass_strict(self):
        write(self.root / "A.md", "a\n")
        write(self.root / "B.md", "b\n")
        plan_path = write_plan(
            self.root,
            partial=True,
            moves=[
                {"from": "A.md", "to": "x/alpha.md", "reason": "r"},
                {"from": "B.md", "to": "x/beta.md", "reason": "r"},
            ],
        )
        # No casefold collision → strict-case validates clean.
        validate_plan(load_plan(plan_path), self.root, strict_case=True)

    def test_collision_against_existing_file_detected(self):
        # A destination differing only by case from an existing file is a
        # clobber on a case-insensitive mount.
        write(self.root / "src.md", "s\n")
        write(self.root / "pile/REPORT.md", "existing\n")
        plan_path = write_plan(
            self.root,
            partial=True,
            moves=[{"from": "src.md", "to": "pile/report.md", "reason": "r"}],
        )
        plan = validate_plan(load_plan(plan_path), self.root)
        self.assertTrue(
            any(
                "report" in (c["dst"] + c["conflicts_with"]).lower()
                for c in plan.get("case_collisions", [])
            )
        )


class TestDedupAction(TempCase):
    def _dup_corpus(self):
        write(self.root / "report.md", "# report\n\nduplicate body\n")
        write(self.root / "report_copy.md", "# report\n\nduplicate body\n")

    def _dedup_plan(self, **over):
        base = dict(
            partial=True,
            moves=[],
            dedup=[
                {
                    "op": "dedup",
                    "keep": "report.md",
                    "drop": "report_copy.md",
                    "to": "archive/duplicates/report_copy.md",
                }
            ],
        )
        base.update(over)
        return write_plan(self.root, **base)

    def test_dedup_validates_and_expands_to_a_move(self):
        self._dup_corpus()
        plan = validate_plan(load_plan(self._dedup_plan()), self.root)
        # the dedup became a move (drop -> to) AND is recorded distinctly.
        self.assertEqual(plan["dedup"][0]["keep"], "report.md")
        by_src = {m["from"]: m["to"] for m in plan["moves"]}
        self.assertEqual(by_src["report_copy.md"], "archive/duplicates/report_copy.md")
        self.assertEqual(
            [m["op"] for m in plan["moves"] if m.get("op") == "dedup"], ["dedup"]
        )

    def test_dedup_rejects_non_identical(self):
        write(self.root / "report.md", "# report\n\nduplicate body\n")
        write(self.root / "report_copy.md", "# TOTALLY DIFFERENT\n")
        expect_code(
            self,
            lambda: validate_plan(load_plan(self._dedup_plan()), self.root),
            "plan_dedup_not_identical",
        )

    def test_dedup_rejects_malformed(self):
        expect_code(
            self,
            lambda: validate_plan(
                load_plan(self._dedup_plan(dedup=[{"op": "move", "keep": "a"}])),
                self.root,
            ),
            "plan_bad_dedup",
        )

    def test_dedup_rejects_keep_equals_drop(self):
        self._dup_corpus()
        expect_code(
            self,
            lambda: validate_plan(
                load_plan(
                    self._dedup_plan(
                        dedup=[
                            {
                                "op": "dedup",
                                "keep": "report.md",
                                "drop": "report.md",
                                "to": "archive/duplicates/report.md",
                            }
                        ]
                    )
                ),
                self.root,
            ),
            "plan_bad_dedup",
        )

    def test_dedup_apply_archives_never_deletes_and_undo_restores(self):
        self._dup_corpus()
        plan_path = self._dedup_plan()
        before = prune_slopkeeper(snapshot_tree(self.root))
        result = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(result["mode"], "executed")
        self.assertFalse(result["partial"])
        # the copy moved to the archive; the survivor stayed; nothing gone.
        self.assertFalse((self.root / "report_copy.md").exists())
        self.assertTrue((self.root / "archive/duplicates/report_copy.md").exists())
        self.assertTrue((self.root / "report.md").exists())
        # undo restores the copy BYTE-EXACT.
        from pathlib import Path as _P

        undone = undo_journal(_P(result["journal"]), self.root, execute=True)
        self.assertEqual(undone["mode"], "executed")
        self.assertTrue((self.root / "report_copy.md").exists())
        self.assertFalse((self.root / "archive/duplicates/report_copy.md").exists())
        self.assertEqual(
            (self.root / "report_copy.md").read_bytes(), b"# report\n\nduplicate body\n"
        )
        restored = prune_slopkeeper(snapshot_tree(self.root))
        self.assertEqual(restored, before)

    def test_render_shows_dedup_section_distinctly(self):
        self._dup_corpus()
        plan = validate_plan(load_plan(self._dedup_plan()), self.root)
        md = render_markdown(plan)
        self.assertIn("Duplicates to archive (dedup", md)
        self.assertIn("report.md", md)
        self.assertIn("archive/duplicates/report_copy.md", md)


class TestP4CLIFlags(TempCase):
    def test_inventory_scope_limits_and_binds(self):
        build_corpus(self.root, {"inbox/a.md": "# a\n", "sibling/s.md": "# s\n"})
        code, out = _run("inventory", str(self.root), "--scope", "inbox", "--summary")
        self.assertEqual(code, 0, out)
        data = json.loads(out)["data"]
        self.assertEqual(data["scope"], "inbox")
        self.assertEqual(sorted(data["files"]), ["inbox/a.md"])
        # the fingerprint is the SCOPE's, not the whole corpus's
        self.assertEqual(
            data["fingerprint"], corpus_fingerprint(self.root, scope="inbox")
        )

    def test_clusters_force_flag_runs_clean(self):
        code, out = _run("clusters", str(self.root), "--force")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["command"], "clusters")

    def test_plan_validate_strict_case_exits_2(self):
        write(self.root / "A.md", "a\n")
        write(self.root / "B.md", "b\n")
        plan_path = write_plan(
            self.root,
            partial=True,
            moves=[
                {"from": "A.md", "to": "x/pile.md", "reason": "r"},
                {"from": "B.md", "to": "x/PILE.md", "reason": "r"},
            ],
        )
        # default: warning, exits 0
        code, out = _run("plan", "validate", str(plan_path), str(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("case_collision", out)
        # --strict-case: refusal, exit 2 (plan_ prefix)
        code, out = _run(
            "plan", "validate", str(plan_path), str(self.root), "--strict-case"
        )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "plan_dst_case_collision")

    def test_render_out_is_full_detail_stdout_is_paginated(self):
        build_corpus(self.root, {f"pile/f{i:03d}.md": "x\n" for i in range(250)})
        plan_path = write_plan(
            self.root, partial=True, rules=[{"match": "pile/", "to": "out"}]
        )
        # stdout render (>200 moves) is summary-only and flags it
        code, out = _run("plan", "render", str(plan_path), str(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("per-move detail omitted", out)
        # --out writes the FULL detail (every move listed)
        gate = self.tmp / "gate.md"
        code, out = _run(
            "plan", "render", str(plan_path), str(self.root), "--out", str(gate)
        )
        self.assertEqual(code, 0, out)
        env = json.loads(out)
        self.assertEqual(env["data"]["written_to"], str(gate))
        self.assertEqual(env["next"]["command"].split()[1], "apply")
        self.assertIn(str(self.root), env["next"]["command"])
        md = gate.read_text()
        self.assertIn("## → `out/`", md)
        self.assertNotIn("per-move detail omitted", md)
        self.assertIn("`pile/f249.md`", md)


if __name__ == "__main__":
    unittest.main()
