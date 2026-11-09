# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The end-to-end agent walk — the integration crown jewel (moved as-is)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.doctor import doctor
from slopkeeper.inventory import collect
from slopkeeper.plan import load_plan, render_markdown, validate_plan
from slopkeeper.scaffold import scaffold_from_plan


class TestStrictReferencesCoveredByRewrites(TempCase):
    def test_strict_references_satisfied_by_rewrites(self):
        # Phase 4, step 8: a risk is satisfied when the rewrite pass
        # covers it — same rule as an edit covering it today. The
        # default corpus's only citer is WATCHLIST.md (two exact-path
        # citations of moved root files); the pass rewrites both, so
        # --strict-references executes instead of refusing.
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True, strict_references=True)
        self.assertEqual(out["mode"], "executed")
        self.assertIn(
            "research/dossiers/acme_dossier.md",
            (self.root / "WATCHLIST.md").read_text(),
        )
        self.assertIn(
            "research/dossiers/globex_dossier.md",
            (self.root / "WATCHLIST.md").read_text(),
        )
        self.assertEqual(out["rewrites"]["count"], 2)


class TestEndToEndSkillRewrite(TempCase):
    def test_end_to_end_skill_rewrite(self):
        # Phase 6, closing the reference epic: a skill in a dot-dir —
        # content as of fp4, the files most likely to cite corpus
        # paths — is rewritten by ONE apply (no hand-authored edits),
        # the render shows the per-file summary to the human, and undo
        # restores the skill byte-for-byte.
        write(self.root / "docs/a.md", "# a\n")
        skill = write(
            self.root / ".claude" / "skills" / "s.md", "Read docs/a.md first.\n"
        )
        plan_path = write_plan(
            self.root,
            moves=[{"from": "docs/a.md", "to": "ref/a.md", "reason": "ref pile"}],
            partial=True,
        )
        # The gate: the golden render carries the rewrite summary with
        # the skill's per-file count, so the human approves real numbers.
        plan = validate_plan(load_plan(plan_path), self.root)
        from slopkeeper.apply import build_rewrite_summary

        md = render_markdown(plan, {}, build_rewrite_summary(plan, self.root))
        self.assertIn("Citation rewrites (automatic): 1 across 1 file(s)", md)
        self.assertIn("`.claude/skills/s.md`", md)
        # One apply rewrites the skill file; the journal says so.
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(skill.read_text(), "Read ref/a.md first.\n")
        journal_ops = [
            json.loads(line)
            for line in Path(out["journal"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            [e["file"] for e in journal_ops if e.get("op") == "rewrite"],
            [".claude/skills/s.md"],
        )
        # Undo restores it byte-for-byte.
        undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(skill.read_bytes(), b"Read docs/a.md first.\n")


class TestEndToEnd(TempCase):
    def test_full_agent_protocol(self):
        # 1. doctor
        d = doctor(self.root)
        self.assertTrue(d["next"]["command"].startswith("slopkeeper inventory"))
        # 2. inventory
        inv = collect(self.root)
        self.assertEqual(inv["stats"]["files"], 8)
        # 3-6. plan, validate, render, apply
        plan_path = write_plan(self.root)
        plan = validate_plan(load_plan(plan_path), self.root)
        md = render_markdown(plan)
        self.assertIn("slopkeeper apply", md)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 4)
        # 7. scaffold
        s = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertTrue((self.root / "AGENTS.md").exists())
        # 8. verify structure
        self.assertTrue(
            (self.root / "research" / "dossiers" / "globex_dossier.md").exists()
        )
        self.assertEqual(s["mode"], "executed")


if __name__ == "__main__":
    unittest.main()
