# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""scaffold.py — the AGENTS.md/STATE.md layer."""

from __future__ import annotations

import unittest

from helpers import TempCase, write_plan
from slopkeeper.apply import apply_plan
from slopkeeper.scaffold import scaffold_from_plan
from slopkeeper.plan import load_plan, validate_plan
from slopkeeper.status import load_manifest
from slopkeeper.status import status as corpus_status


class TestScaffold(TempCase):
    def test_execute_preserves_unresolved_and_last_plan(self):
        # T2: the documented loop is apply --execute → scaffold --execute,
        # and both record the manifest. Scaffold's re-record must carry the
        # unresolved backlog and last_plan forward — wiping them (the old
        # behavior) emptied still_unresolved on every tend, so the
        # CONTINUOUS loop never existed.
        plan_path = write_plan(
            self.root, unresolved=[{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        apply_plan(plan_path, self.root, execute=True)
        m = load_manifest(self.root)
        self.assertEqual(
            m["unresolved"], [{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        self.assertEqual(m["last_plan"], str(plan_path))
        scaffold_from_plan(plan_path, self.root, execute=True)
        m = load_manifest(self.root)
        self.assertEqual(
            m["unresolved"], [{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        self.assertEqual(m["last_plan"], str(plan_path))
        # `slopkeeper status` surfaces the surviving backlog.
        s = corpus_status(self.root)
        self.assertEqual([u["path"] for u in s["still_unresolved"]], ["WATCHLIST.md"])

    def test_dry_run_then_execute(self):
        plan_path = write_plan(self.root)
        out = scaffold_from_plan(plan_path, self.root)
        self.assertEqual(out["mode"], "dry-run")
        out = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertTrue((self.root / "AGENTS.md").exists())
        self.assertTrue((self.root / "research" / "dossiers" / "AGENTS.md").exists())
        roster = (self.root / "AGENTS.md").read_text()
        self.assertIn("research/dossiers", roster)

    def test_never_overwrites(self):
        plan_path = write_plan(self.root)
        scaffold_from_plan(plan_path, self.root, execute=True)
        (self.root / "AGENTS.md").write_text("# MY EDITS\n")
        out = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertIn("MY EDITS", (self.root / "AGENTS.md").read_text())
        self.assertTrue(out["suggested"], "should write .slopkeeper-suggested instead")


# --- roster contents + the state_md bot-kind default ---


class TestScaffoldRoster(TempCase):
    def test_roster_lists_folders_and_keeps(self):
        plan_path = write_plan(self.root)
        scaffold_from_plan(plan_path, self.root, execute=True)
        roster = (self.root / "AGENTS.md").read_text()
        # One row per folder entry, with kind and role columns.
        self.assertIn("| `research/dossiers/` | case |", roster)
        self.assertIn("Market research dossiers.", roster)
        # Keeps are listed too (folders + keeps), with their reasons.
        self.assertIn("| `WATCHLIST.md` | keep |", roster)
        for kept in ("docs", "lead-gen", "spark-1.2.3"):
            self.assertIn(f"| `{kept}` | keep |", roster)

    def test_roster_keep_rows_carry_reasons(self):
        # {path, reason} keep objects get their reason in the role column.
        plan_path = write_plan(
            self.root,
            keep=[
                {"path": "WATCHLIST.md", "reason": "living index"},
                "docs",
            ],
        )
        scaffold_from_plan(plan_path, self.root, execute=True)
        roster = (self.root / "AGENTS.md").read_text()
        self.assertIn("| `WATCHLIST.md` | keep | living index |", roster)

    def test_roster_role_cells_filled(self):
        # Phase 11 small fix: the role cell of a kept row is never
        # empty — the reason when given, an explicit marker when not
        # (an empty cell read as "why is this row here?").
        plan_path = write_plan(
            self.root,
            keep=[
                {"path": "WATCHLIST.md", "reason": "living index"},
                "docs",
                {"path": "lead-gen", "reason": ""},
                "spark-1.2.3",
            ],
        )
        scaffold_from_plan(plan_path, self.root, execute=True)
        roster = (self.root / "AGENTS.md").read_text()
        self.assertIn("| `WATCHLIST.md` | keep | living index |", roster)
        self.assertIn("| `docs` | keep | (kept in place) |", roster)
        self.assertIn("| `lead-gen` | keep | (kept in place) |", roster)

    def test_state_md_defaults_to_bot_kind_only(self):
        # The bot-only default lives in validate_plan (scaffold consumes the
        # normalized folders); explicit state_md always wins.
        plan_path = write_plan(
            self.root,
            folders=[
                {"path": "bots/alpha", "kind": "bot"},
                {"path": "topics/notes", "kind": "topic"},
                {"path": "bots/gamma", "kind": "bot", "state_md": False},
            ],
        )
        normalized = validate_plan(load_plan(plan_path), self.root)
        by_path = {f["path"]: f["state_md"] for f in normalized["folders"]}
        self.assertEqual(
            by_path, {"bots/alpha": True, "topics/notes": False, "bots/gamma": False}
        )


# --- AGENTS.md template role slots + roster keep rows ---


class TestTemplateVariants(unittest.TestCase):
    def test_role_fills_each_template_slot(self):
        from slopkeeper.scaffold import _render_agents_md

        slots = {
            "## Purpose": "<one paragraph: what this worker owns>",
            "## Scope": "<what belongs here>",
            "## What this is": "<one paragraph>",
        }
        for section, placeholder in slots.items():
            tpl = {"agents_md_skeleton": f"# <Name>\n{section}\n{placeholder}\n"}
            out = _render_agents_md(
                {"path": "x", "kind": "bot", "role": "Owns the thing."}, tpl
            )
            self.assertNotIn(placeholder, out)
            self.assertIn("Owns the thing.", out)
            # A filled slot consumes the role; nothing is appended.
            self.assertNotIn("## Role / scope", out)

    def test_role_appended_when_template_has_no_slot(self):
        from slopkeeper.scaffold import _render_agents_md

        out = _render_agents_md(
            {"path": "x", "kind": "bot", "role": "R."},
            {"agents_md_skeleton": "# <Name>\n"},
        )
        self.assertIn("## Role / scope\nR.", out)

    def test_empty_keep_rows_are_skipped(self):
        from slopkeeper.scaffold import _auto_roster

        out = _auto_roster(
            [{"path": "a", "kind": "bot", "role": "r"}],
            [{"path": "", "reason": "junk"}, "kept.md"],
        )
        self.assertIn("`a/`", out)
        self.assertIn("`kept.md`", out)
        self.assertNotIn("| `` |", out)

    def test_roster_table_has_no_blank_line_after_separator(self):
        # Field bug: the header constant's trailing newline + the "\n"
        # join left a blank line after |---|---|---|, so GFM rendered the
        # roster rows as plain paragraphs instead of a table.
        from slopkeeper.scaffold import _auto_roster

        out = _auto_roster([{"path": "a", "kind": "bot", "role": "r"}])
        self.assertIn("|---|---|---|\n| `a/` | bot | r |", out)
        self.assertNotIn("|---|---|---|\n\n", out)

    def test_kept_role_cell_never_empty(self):
        # _auto_roster is pure: the marker lands without a corpus.
        from slopkeeper.scaffold import _auto_roster

        out = _auto_roster([], ["WATCHLIST.md"])
        self.assertIn("| `WATCHLIST.md` | keep | (kept in place) |", out)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
