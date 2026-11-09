# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""External references: refs --scan-path, plan external_roots, code-file edits.

The external-reference round's rows, per TEST_PLAN conventions (one
concern per class, corpora built via helpers, expect_code for every
SlopKeeperError code):

- refs <root> <basename> --scan-path <dir> finds citations OUTSIDE the
  corpus — skills, guides, cron scripts, scratch harnesses that apply's
  in-tree scan would never see.
- A plan may declare "external_roots" and edit files under them; without
  the declaration an outside edit is still path_escape (yesterday's
  contract, unchanged).
- External edits run after the in-root ones, are journaled with their
  absolute path, undo-able, and never churn the corpus fingerprint.
- The code-file edit row: a .py script citing a moved file by a hardcoded
  path string is rewritten in-plan and --strict-references passes (all
  prior golden fixtures were markdown).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path

from helpers import (
    TempCase,
    expect_code,
    prune_slopkeeper,
    snapshot_tree,
    write,
    write_plan,
)
from slopkeeper import cli
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import corpus_fingerprint
from slopkeeper.plan import load_plan, validate_plan
from slopkeeper.references import external_citation_lines


def run_main(argv: list[str]) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed stdout JSON)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class ExtCase(TempCase):
    """TempCase plus an existing external directory sibling to the corpus."""

    def setUp(self):
        super().setUp()
        self.ext = (self.tmp / "external").resolve()
        self.ext.mkdir()


def _ext_script(ext: Path) -> Path:
    """A scratch harness outside the corpus citing a corpus path."""
    return write(
        ext / "harness" / "run.sh", "#!/bin/sh\ncat tmp/docs/infra/config.yaml\n"
    )


class TestExternalCitationScan(ExtCase):
    """references.external_citation_lines: the scan itself."""

    def test_finds_citation_in_external_dir(self):
        _ext_script(self.ext)
        hits = external_citation_lines([self.ext], "config.yaml")
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit["file"], str(self.ext / "harness" / "run.sh"))
        self.assertEqual(hit["line"], 2)
        self.assertIn("tmp/docs/infra/config.yaml", hit["text"])
        self.assertTrue(hit["external"])

    def test_skips_dot_dirs_and_non_text_files(self):
        write(self.ext / ".git" / "hooks" / "probe.sh", "config.yaml\n")
        write(self.ext / "node_modules" / "dep" / "index.js", "config.yaml;\n")
        (self.ext / "photo.png").write_bytes(b"config.yaml as bytes")
        write(self.ext / "real.md", "mentions config.yaml here\n")
        hits = external_citation_lines([self.ext], "config.yaml")
        self.assertEqual([h["file"] for h in hits], [str(self.ext / "real.md")])
        self.assertTrue(hits[0]["external"])

    def test_self_citation_skipped(self):
        write(self.ext / "config.yaml", "name: config.yaml\n")
        self.assertEqual(external_citation_lines([self.ext], "config.yaml"), [])

    def test_per_file_cap_bounds_lines_per_file(self):
        write(
            self.ext / "many.md", "".join(f"line {i}: config.yaml\n" for i in range(6))
        )
        hits = external_citation_lines([self.ext], "config.yaml", per_file_cap=2)
        self.assertEqual(len(hits), 2)

    def test_missing_dir_is_empty(self):
        self.assertEqual(external_citation_lines([self.ext / "absent"], "x.md"), [])

    def test_multiple_dirs_merged_in_given_order(self):
        a = (self.tmp / "ext-a").resolve()
        b = (self.tmp / "ext-b").resolve()
        write(a / "one.md", "config.yaml\n")
        write(b / "two.md", "config.yaml\n")
        hits = external_citation_lines([a, b], "config.yaml")
        self.assertEqual(
            [h["file"] for h in hits], [str(a / "one.md"), str(b / "two.md")]
        )
        self.assertTrue(all(h["external"] for h in hits))


class TestRefsCliExternal(ExtCase):
    """refs --scan-path through cli.main (the agent-facing contract)."""

    def test_scan_path_merges_external_citations(self):
        _ext_script(self.ext)
        code, out = run_main(
            ["refs", str(self.root), "config.yaml", "--scan-path", str(self.ext)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "refs")
        self.assertEqual(out["data"]["citation_count"], 1)
        cit = out["data"]["citations"][0]
        self.assertTrue(cit["external"])
        self.assertEqual(cit["file"], str(self.ext / "harness" / "run.sh"))
        self.assertEqual(out["data"]["scan_paths"], [str(self.ext)])

    def test_without_scan_path_default_behavior_unchanged(self):
        # In-tree citations carry no external flag; scan_paths is empty.
        code, out = run_main(["refs", str(self.root), "acme_dossier.md"])
        self.assertEqual(code, 0)
        self.assertGreater(out["data"]["citation_count"], 0)
        self.assertEqual(out["data"]["scan_paths"], [])
        self.assertFalse(any(c.get("external") for c in out["data"]["citations"]))

    def test_in_tree_and_external_citations_distinguished(self):
        # The in-tree citation (WATCHLIST.md → acme_dossier.md) carries
        # no flag; the external one (the skill file) carries external:
        # true with an absolute path.
        write(self.ext / "skill.md", "Read acme_dossier.md first.\n")
        code, out = run_main(
            ["refs", str(self.root), "acme_dossier.md", "--scan-path", str(self.ext)]
        )
        self.assertEqual(code, 0)
        cits = out["data"]["citations"]
        in_tree = [c for c in cits if not c.get("external")]
        external = [c for c in cits if c.get("external")]
        self.assertEqual([c["file"] for c in in_tree], ["WATCHLIST.md"])
        self.assertEqual([c["file"] for c in external], [str(self.ext / "skill.md")])

    def test_missing_scan_dir_refused(self):
        # A typo'd dir must not read as "no external citations".
        code, out = run_main(
            [
                "refs",
                str(self.root),
                "config.yaml",
                "--scan-path",
                str(self.tmp / "nope"),
            ]
        )
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "scan_path_missing")

    def test_oversized_scan_dir_still_scans(self):
        # Phase 3: scan paths have no size ceiling — a dir past the old
        # 2000-file cap scans like any other; the only scan-path refusal
        # left is scan_path_missing. (Loud counts live in apply's
        # reference_scan, not here: refs reports citation sites.)
        for i in range(2001):
            write(self.ext / f"f{i:04}.txt", "config.yaml\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(
                [
                    "refs",
                    str(self.root),
                    "config.yaml",
                    "--scan-path",
                    str(self.ext),
                ]
            )
        out = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        external = [c for c in out["data"]["citations"] if c.get("external")]
        self.assertEqual(len(external), 2001)


class TestExternalRewritePass(ExtCase):
    """Phase 5: external roots join the rewrite pass; expect asserts counts."""

    def _corpus(self):
        root = self.tmp / "extcorpus"
        write(root / "docs/api.md", "# api\n")
        write(root / "docs/guide.md", "# guide\n")
        write(root / "w.md", "in-corpus citation docs/api.md\n")
        return root

    def _plan(self, root, **over):
        script = write(
            self.ext / "skills" / "s.md",
            "read tmp/docs/api.md and docs/guide.md\n",
        )
        kw = {
            "moves": [{"from": "docs", "to": "ref", "reason": "pile"}],
            "keep": ["w.md"],
            "folders": [],
            "external_roots": [{"path": str(self.ext), "expect": 2}],
        }
        kw.update(over)
        return write_plan(root, **kw), script

    def test_external_rewrite_journaled_and_undone(self):
        # The external skill's citations take the literal suffix swap
        # ("tmp/docs/api.md" → "tmp/ref/api.md"), journal under the
        # ABSOLUTE path, and undo restores byte-for-byte.
        root = self._corpus()
        plan_path, script = self._plan(root)
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")
        per = {r["file"]: r["count"] for r in out["rewrites"]["per_file"]}
        self.assertEqual(per, {"w.md": 1, str(script): 2})
        self.assertEqual(script.read_text(), "read tmp/ref/api.md and ref/guide.md\n")
        lines = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rw = [e for e in lines if e.get("op") == "rewrite"]
        self.assertEqual({e["file"] for e in rw}, {"w.md", str(script)})
        undo = undo_journal(Path(out["journal"]), root, execute=True)
        self.assertIn("reverted", {o["status"] for o in undo["operations"]})
        self.assertEqual(
            script.read_bytes(), b"read tmp/docs/api.md and docs/guide.md\n"
        )

    def test_expect_mismatch_aborts_before_write(self):
        # Wrong expect: the dry-count pass refuses before ANY write —
        # the external file AND the corpus are untouched (no moves).
        # (.slopkeeper is pruned: the tend lock legitimately appears.)
        root = self._corpus()
        plan_path, script = self._plan(
            root, external_roots=[{"path": str(self.ext), "expect": 5}]
        )

        def snapshot():
            return {
                p: os.stat(p).st_mtime_ns
                for p in list(self.ext.rglob("*")) + list(root.rglob("*"))
                if p.is_file() and ".slopkeeper" not in p.parts
            }

        before = snapshot()
        expect_code(
            self,
            lambda: apply_plan(plan_path, root, execute=True),
            "rewrite_expect_mismatch",
        )
        self.assertEqual(snapshot(), before)
        self.assertEqual(script.read_text(), "read tmp/docs/api.md and docs/guide.md\n")

    def test_expect_match_proceeds(self):
        # The declared count is exactly what the pass computes: apply
        # executes and the rewrites land.
        root = self._corpus()
        plan_path, script = self._plan(root)
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(script.read_text(), "read tmp/ref/api.md and ref/guide.md\n")


class TestExternalRootValidation(ExtCase):
    """plan.validate: external_roots gate the out-of-root edit escape hatch."""

    def _outside_edit(self, **over):
        edit = {
            "file": str(self.ext / "harness" / "run.sh"),
            "find": "tmp/docs/infra/config.yaml",
            "replace": "docs/infra/config.yaml",
            "reason": "external citation",
        }
        edit.update(over)
        return edit

    def test_edit_outside_root_rejected_without_external_roots(self):
        # Yesterday's contract, pinned: an outside edit is a path_escape.
        plan_path = write_plan(self.root, edits=[self._outside_edit()])
        expect_code(
            self, lambda: validate_plan(load_plan(plan_path), self.root), "path_escape"
        )

    def test_null_external_roots_behaves_as_absent(self):
        plan_path = write_plan(
            self.root, external_roots=None, edits=[self._outside_edit()]
        )
        expect_code(
            self, lambda: validate_plan(load_plan(plan_path), self.root), "path_escape"
        )

    def test_edit_outside_root_accepted_when_declared(self):
        _ext_script(self.ext)
        plan_path = write_plan(
            self.root, external_roots=[str(self.ext)], edits=[self._outside_edit()]
        )
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["edits"][0]["file"], str(self.ext / "harness" / "run.sh"))
        # Phase 5: roots normalize to {"path": ...} objects (the
        # optional "expect" count assertion rides on the same shape).
        self.assertEqual(out["external_roots"], [{"path": str(self.ext)}])

    def test_edit_outside_declared_roots_still_rejected(self):
        other = self.tmp / "elsewhere"
        other.mkdir()
        plan_path = write_plan(
            self.root,
            external_roots=[str(self.ext)],
            edits=[self._outside_edit(file=str(other / "run.sh"))],
        )
        expect_code(
            self, lambda: validate_plan(load_plan(plan_path), self.root), "path_escape"
        )

    def test_in_root_absolute_edit_still_normalized_relative(self):
        # An absolute path under the CORPUS root is an in-root edit, even
        # when external_roots are declared.
        plan_path = write_plan(
            self.root,
            external_roots=[str(self.ext)],
            edits=[
                {
                    "file": str(self.root / "WATCHLIST.md"),
                    "find": "acme_dossier.md",
                    "replace": "research/acme_dossier.md",
                    "reason": "in-root, given absolute",
                }
            ],
        )
        out = validate_plan(load_plan(plan_path), self.root)
        self.assertEqual(out["edits"][0]["file"], "WATCHLIST.md")

    def test_external_root_must_be_absolute(self):
        rel = str(self.ext.relative_to(self.tmp))
        plan_path = write_plan(self.root, external_roots=[rel])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_external_root",
        )

    def test_external_root_must_exist(self):
        plan_path = write_plan(self.root, external_roots=[str(self.tmp / "nope")])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_external_root",
        )

    def test_external_root_must_be_a_directory(self):
        plain = write(self.tmp / "plain.txt", "not a dir\n")
        plan_path = write_plan(self.root, external_roots=[str(plain)])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_external_root",
        )

    def test_external_root_must_be_a_string(self):
        plan_path = write_plan(self.root, external_roots=[42])
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_external_root",
        )

    def test_external_roots_must_be_a_list(self):
        plan_path = write_plan(self.root, external_roots=str(self.ext))
        expect_code(
            self,
            lambda: validate_plan(load_plan(plan_path), self.root),
            "plan_bad_external_root",
        )


class TestExternalEditsApply(ExtCase):
    """apply: external edits execute, journal, and undo like any edit."""

    def _plan(self, **over):
        _ext_script(self.ext)
        kw = dict(
            external_roots=[str(self.ext)],
            edits=[
                {
                    "file": str(self.ext / "harness" / "run.sh"),
                    "find": "tmp/docs/infra/config.yaml",
                    "replace": "docs/infra/config.yaml",
                    "reason": "external citation",
                }
            ],
        )
        kw.update(over)
        return write_plan(self.root, **kw)

    def _journal_ops(self, journal: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dry_run_lists_external_edit_at_absolute_path(self):
        dry = apply_plan(self._plan(), self.root)
        # Phase 3: the row is a superset of {file, reason} — it now also
        # carries the owned line and the before/after line text.
        self.assertEqual(len(dry["would_edit"]), 1)
        row = dry["would_edit"][0]
        self.assertEqual(row["file"], str(self.ext / "harness" / "run.sh"))
        self.assertEqual(row["reason"], "external citation")
        self.assertEqual(row["line"], 2)
        self.assertIn("cat docs/infra/config.yaml", row["after"])
        self.assertEqual(dry["mode"], "dry-run")

    def test_external_edit_executes_and_rewrites(self):
        out = apply_plan(self._plan(), self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        script = (self.ext / "harness" / "run.sh").read_text()
        self.assertIn("cat docs/infra/config.yaml", script)
        self.assertNotIn("tmp/docs/infra", script)

    def test_external_edit_is_journaled_with_absolute_path(self):
        out = apply_plan(self._plan(), self.root, execute=True)
        edit_ops = [
            e for e in self._journal_ops(Path(out["journal"])) if e.get("op") == "edit"
        ]
        self.assertEqual(len(edit_ops), 1)
        self.assertEqual(edit_ops[0]["file"], str(self.ext / "harness" / "run.sh"))
        self.assertEqual(edit_ops[0]["find"], "tmp/docs/infra/config.yaml")
        self.assertEqual(edit_ops[0]["replace"], "docs/infra/config.yaml")

    def test_external_edit_undo_reverts(self):
        out = apply_plan(self._plan(), self.root, execute=True)
        script = self.ext / "harness" / "run.sh"
        self.assertNotIn("tmp/docs/infra", script.read_text())
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        reverted = [op for op in undo["operations"] if "file" in op]
        # Phase 4: the WATCHLIST.md citation of the moved file is also
        # rewritten (and journaled), so undo reverts it alongside the
        # external edit.
        self.assertEqual(
            reverted,
            [
                {"file": str(script), "status": "reverted"},
                {"file": "WATCHLIST.md", "status": "reverted"},
            ],
        )
        self.assertIn("tmp/docs/infra/config.yaml", script.read_text())

    def test_stale_external_edit_touches_nothing_in_corpus(self):
        plan_path = self._plan()
        script = self.ext / "harness" / "run.sh"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        before = prune_slopkeeper(snapshot_tree(self.root))
        expect_code(
            self, lambda: apply_plan(plan_path, self.root, execute=True), "edit_stale"
        )
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        self.assertIn("exit 0", script.read_text())

    def test_external_edit_does_not_churn_corpus_fingerprint(self):
        # The gate is content-of-corpus only. Apply moves+external-edit,
        # undo, then apply the moves alone: the corpus fingerprint after
        # both runs is identical, so an outside rewrite never produces a
        # phantom corpus_changed on the next tend.
        write(self.root / "a.md", "A\n")
        plan_full = self._plan(
            moves=[{"from": "a.md", "to": "b.md", "reason": "pile"}], partial=True
        )
        out = apply_plan(plan_full, self.root, execute=True)
        fp_full = corpus_fingerprint(self.root)
        undo_journal(Path(out["journal"]), self.root, execute=True)
        plan_moves = write_plan(
            self.root,
            moves=[{"from": "a.md", "to": "b.md", "reason": "pile"}],
            partial=True,
        )
        apply_plan(plan_moves, self.root, execute=True)
        self.assertEqual(corpus_fingerprint(self.root), fp_full)

    def test_external_edits_run_after_in_root_edits(self):
        # Journal order is execution order: the in-root edit (which
        # resolves through the move chains) lands before the external
        # one, whatever order the plan listed them in.
        _ext_script(self.ext)
        plan_path = write_plan(
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
                    "file": str(self.ext / "harness" / "run.sh"),
                    "find": "tmp/docs/infra/config.yaml",
                    "replace": "docs/infra/config.yaml",
                    "reason": "external citation",
                },
                {
                    "file": "WATCHLIST.md",
                    "find": "acme_dossier.md",
                    "replace": "research/acme_dossier.md",
                    "reason": "fix citation",
                },
            ],
            external_roots=[str(self.ext)],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "spark-1.2.3",
                "docs",
                "lead-gen",
            ],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        edit_files = [
            e["file"]
            for e in self._journal_ops(Path(out["journal"]))
            if e.get("op") == "edit"
        ]
        self.assertEqual(
            edit_files, ["WATCHLIST.md", str(self.ext / "harness" / "run.sh")]
        )
        self.assertIn(
            "research/acme_dossier.md", (self.root / "WATCHLIST.md").read_text()
        )
        self.assertIn(
            "docs/infra/config.yaml", (self.ext / "harness" / "run.sh").read_text()
        )


class TestCodeFileEditRow(ExtCase):
    """The first code-file edit row: a .py script INSIDE the corpus cites
    a moving file via a hardcoded path string; the plan rewrites the
    string and --strict-references passes. All prior golden fixtures were
    markdown — this pins that code files are citable, editable citizens.
    """

    SCRIPT = "scripts/run_report.py"

    def _py_corpus(self):
        write(self.root / "notes" / "weekly.md", "# Weekly\n")
        write(
            self.root / self.SCRIPT,
            '#!/usr/bin/env python3\nREPORT = "notes/weekly.md"\n',
        )

    def _plan(self, **over):
        kw = dict(
            moves=[
                {
                    "from": "notes/weekly.md",
                    "to": "reports/weekly.md",
                    "reason": "reports pile",
                }
            ],
            edits=[
                {
                    "file": self.SCRIPT,
                    "find": "notes/weekly.md",
                    "replace": "reports/weekly.md",
                    "reason": "rewrite hardcoded path",
                }
            ],
            partial=True,
        )
        kw.update(over)
        return write_plan(self.root, **kw)

    def test_unresolved_citation_blocks_strict_apply(self):
        # Sanity: with no declared edit, the .py citation IS a reference
        # risk and --strict-references refuses (referenced_moves). The
        # plan opts out of the default rewrite pass — with it on, the
        # path-form citation would be rewritten and the risk satisfied.
        self._py_corpus()
        plan_path = self._plan(edits=[], rewrites={"mode": "off"})
        expect_code(
            self,
            lambda: apply_plan(
                plan_path, self.root, execute=True, strict_references=True
            ),
            "referenced_moves",
        )

    def test_edit_rewrites_script_and_strict_apply_passes(self):
        self._py_corpus()
        plan_path = self._plan()
        out = apply_plan(plan_path, self.root, execute=True, strict_references=True)
        self.assertEqual(out["mode"], "executed")
        # The script's hardcoded path string was rewritten...
        script = (self.root / self.SCRIPT).read_text()
        self.assertIn('REPORT = "reports/weekly.md"', script)
        self.assertNotIn("notes/weekly.md", script)
        # ...so the edit resolved the only citing file: no risks, no abort.
        self.assertEqual(out["reference_risks"], {})
        # And the moved file's content arrived intact.
        self.assertEqual(
            (self.root / "reports" / "weekly.md").read_text(), "# Weekly\n"
        )


if __name__ == "__main__":
    unittest.main()
