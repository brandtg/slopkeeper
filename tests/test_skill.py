# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`skill install` — the agent-setup surface.

HOME/XDG are pointed at a tmpdir so the real ~/.claude, ~/.config and
~/.hermes are never touched; the packaged SKILL.md is the expected
payload (the whole point is that installs match the binary).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slopkeeper import cli
from slopkeeper.skill import packaged_skill_text


def run_main(argv: list[str]) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed stdout JSON)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class SkillInstallCase(unittest.TestCase):
    """tmpdir HOME/XDG: agent-name targets resolve somewhere fake."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(self._rm_tmp)
        patcher = mock.patch.dict(
            os.environ,
            {"HOME": str(self.tmp / "home"), "XDG_CONFIG_HOME": str(self.tmp / "xdg")},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _rm_tmp(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)


class TestNamedAgents(SkillInstallCase):
    def test_opencode_resolves_through_xdg(self):
        code, out = run_main(["skill", "install", "opencode"])
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        dest = Path(out["data"]["installed"][0]["skill_dir"])
        self.assertEqual(dest, self.tmp / "xdg" / "opencode" / "skills" / "slopkeeper")
        self.assertEqual(
            (dest / "SKILL.md").read_text(encoding="utf-8"),
            packaged_skill_text(),
        )

    def test_claude_and_hermes_resolve_under_home(self):
        code, out = run_main(["skill", "install", "claude", "hermes"])
        self.assertEqual(code, 0)
        dirs = {r["target"]: r["skill_dir"] for r in out["data"]["installed"]}
        self.assertEqual(
            dirs["claude"],
            str(self.tmp / "home" / ".claude" / "skills" / "slopkeeper"),
        )
        self.assertEqual(
            dirs["hermes"],
            str(
                self.tmp
                / "home"
                / ".hermes"
                / "skills"
                / "software-development"
                / "slopkeeper"
            ),
        )
        for d in dirs.values():
            self.assertTrue((Path(d) / "SKILL.md").exists())

    def test_reinstall_skips_with_warning_and_writes_nothing(self):
        run_main(["skill", "install", "claude"])
        # A human-edited skill must survive an agent's re-run.
        dest = self.tmp / "home" / ".claude" / "skills" / "slopkeeper"
        (dest / "SKILL.md").write_text("# human-edited\n", encoding="utf-8")
        code, out = run_main(["skill", "install", "claude"])
        self.assertEqual(code, 0)
        record = out["data"]["installed"][0]
        self.assertEqual(record["action"], "skipped")
        self.assertIn("--force", record["warning"])
        self.assertEqual(
            out["data"]["warnings"],
            [record["warning"]],
        )
        self.assertEqual(
            (dest / "SKILL.md").read_text(encoding="utf-8"), "# human-edited\n"
        )

    def test_force_overwrites_and_says_so(self):
        run_main(["skill", "install", "claude"])
        dest = self.tmp / "home" / ".claude" / "skills" / "slopkeeper"
        (dest / "SKILL.md").write_text("# human-edited\n", encoding="utf-8")
        code, out = run_main(["skill", "install", "--force", "claude"])
        self.assertEqual(code, 0)
        record = out["data"]["installed"][0]
        self.assertEqual(record["action"], "overwritten")
        self.assertNotIn("warnings", out["data"])
        self.assertEqual(
            (dest / "SKILL.md").read_text(encoding="utf-8"),
            packaged_skill_text(),
        )

    def test_force_on_fresh_install_is_installed_not_overwritten(self):
        code, out = run_main(["skill", "install", "--force", "claude"])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["installed"][0]["action"], "installed")

    def test_unknown_agent_errors_loudly(self):
        code, out = run_main(["skill", "install", "zorp"])
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "skill_unknown_agent")
        self.assertIn("claude", out["error"]["recovery"])

    def test_agent_name_wins_over_local_dir(self):
        # `claude` in the cwd must never shadow the well-known agent.
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp)
        (self.tmp / "claude").mkdir()
        code, out = run_main(["skill", "install", "claude"])
        self.assertEqual(code, 0)
        self.assertIn(".claude", out["data"]["installed"][0]["skill_dir"])


class TestDirectoryTargets(SkillInstallCase):
    def test_installs_into_given_dir(self):
        skills = self.tmp / "my-agent" / "skills"
        code, out = run_main(["skill", "install", str(skills)])
        self.assertEqual(code, 0)
        record = out["data"]["installed"][0]
        self.assertIsNone(record["agent"])
        self.assertEqual(record["skill_dir"], str(skills / "slopkeeper"))
        self.assertEqual(record["action"], "installed")
        self.assertEqual(
            (skills / "slopkeeper" / "SKILL.md").read_text(encoding="utf-8"),
            packaged_skill_text(),
        )

    def test_relative_dir_resolves(self):
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp)
        code, out = run_main(["skill", "install", "./skills"])
        self.assertEqual(code, 0)
        self.assertEqual(
            out["data"]["installed"][0]["skill_dir"],
            str((self.tmp / "skills" / "slopkeeper").resolve()),
        )

    def test_existing_file_at_destination_errors(self):
        skills = self.tmp / "skills"
        skills.mkdir()
        (skills / "slopkeeper").write_text("occupant", encoding="utf-8")
        code, out = run_main(["skill", "install", str(skills)])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "skill_target_not_dir")

    def test_bare_word_that_is_neither_agent_nor_path_errors(self):
        code, out = run_main(["skill", "install", "cursor"])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "skill_unknown_agent")


if __name__ == "__main__":
    unittest.main()
