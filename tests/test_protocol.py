# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The end-to-end agent walk — the integration crown jewel (moved as-is)."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper import cli
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.doctor import doctor
from slopkeeper.inventory import collect
from slopkeeper.plan import load_plan, render_markdown, validate_plan
from slopkeeper.scaffold import scaffold_from_plan


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class TestEnvelopeShape(TempCase):
    """P5: the envelope is API, so its shape is asserted, not assumed.

    Every JSON command answers the same question the same way:
    `{tool, version, command, ok, data}` plus `next` on the success path,
    `error` on the failure path, and `warnings` on either when something
    soft happened. The audit's protocol promise is only worth what a test
    makes it worth — an agent parses these keys, so they are pinned here
    across the breadth of the command set rather than per command.
    """

    # (argv builder, command name) for the read-only surface.
    def _cases(self):
        root = str(self.root)
        plan = write_plan(self.root)
        return [
            (["doctor", root], "doctor"),
            (["inventory", root, "--fast"], "inventory"),
            (["clusters", root], "clusters"),
            (["status", root], "status"),
            (["refs", root, "acme_dossier.md"], "refs"),
            (["propose", root], "propose"),
            (["strategies"], "strategies.list"),
            (["strategies", "show", "hybrid"], "strategies.show"),
            (["strategies", "dirs"], "strategies.dirs"),
            (["roots", "list"], "roots.list"),
            (["plan", "validate", str(plan), root], "plan.validate"),
            (["apply", str(plan), root], "apply"),
            (["scaffold", str(plan), root], "scaffold"),
        ]

    def test_every_success_envelope_has_the_contract_keys(self):
        for argv, command in self._cases():
            with self.subTest(command=command):
                code, env = run_cli(argv)
                self.assertEqual(code, 0, env)
                self.assertEqual(env["tool"], "slopkeeper")
                self.assertTrue(env["version"])
                self.assertEqual(env["command"], command)
                self.assertTrue(env["ok"])
                self.assertIn("data", env)
                self.assertNotIn("error", env)
                # Every success path chains (the two exceptions are the
                # bare artifact commands: `plan schema` and `guide`).
                self.assertIn("next", env)
                for key in ("action", "command", "hint"):
                    self.assertIn(key, env["next"])

    def test_error_envelope_shape_and_exit_code(self):
        # Failure is the same envelope with ok false and an error object;
        # `plan_*` codes exit 2 (fixable by the caller), the rest 1.
        code, env = run_cli(["doctor", str(self.tmp / "not-a-corpus")])
        self.assertEqual(code, 1)
        self.assertFalse(env["ok"])
        self.assertNotIn("data", env)
        self.assertEqual(env["error"]["code"], "no_corpus")
        self.assertTrue(env["error"]["recovery"])
        self.assertNotIn("warnings", env)

    def test_plan_family_errors_exit_two(self):
        code, env = run_cli(
            ["plan", "validate", str(self.tmp / "x.json"), str(self.root)]
        )
        self.assertEqual(code, 2)
        self.assertEqual(env["error"]["code"], "plan_unreadable")

    def test_undo_chains_too(self):
        # `next_step`'s docstring promises every success path chains, and
        # `undo` was the last command that did not (P5). A dry-run chains
        # to the execute; an execute chains to the status that shows the
        # corpus it just put back.
        plan_path = write_plan(self.root)
        applied = apply_plan(plan_path, self.root, execute=True)
        code, dry = run_cli(["undo", applied["journal"], str(self.root)])
        self.assertEqual(code, 0, dry)
        self.assertEqual(dry["next"]["action"], "undo")
        self.assertIn("--execute", dry["next"]["command"])
        code, done = run_cli(["undo", applied["journal"], str(self.root), "--execute"])
        self.assertEqual(code, 0, done)
        self.assertEqual(done["next"]["action"], "status")
        self.assertIn(f"slopkeeper status {self.root}", done["next"]["command"])

    def test_warnings_are_a_list_of_tokens_when_present(self):
        # A casefold-colliding pair of destinations is the deterministic
        # soft condition: `plan validate` reports it as a warning (P4) and
        # P5 mirrors it onto the envelope. The token list must be strings,
        # deduped, and identical at both levels.
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "acme_dossier.md", "to": "docs/Note.md", "reason": "x"},
                {"from": "globex_dossier.md", "to": "docs/note.md", "reason": "y"},
            ],
            keep=[
                "WATCHLIST.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "docs",
                "lead-gen",
                "spark-1.2.3",
            ],
            partial=True,
        )
        code, env = run_cli(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 0, env)
        self.assertIsInstance(env["warnings"], list)
        self.assertTrue(all(isinstance(w, str) for w in env["warnings"]))
        self.assertEqual(len(env["warnings"]), len(set(env["warnings"])))
        self.assertIn("case_collision", env["warnings"])
        self.assertEqual(env["warnings"], env["data"]["warnings"])


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
