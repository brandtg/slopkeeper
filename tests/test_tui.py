# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""tests/test_tui.py — the bare-`slopkeeper` REPL.

The REPL drives the same functions as the CLI; these tests feed line
lists and capture output. No subprocess, no pty — the loop is a pure
function of (root, lines) → exit code + written text.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import os
import unittest

from helpers import TempCase, write, write_plan
from slopkeeper import cli
from slopkeeper.tui import _Session, _compute_matches, repl


def run_repl(lines: list[str], root=None, interactive: bool = False) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = repl(root, lines=iter(lines), out=buf, interactive=interactive)
    return code, buf.getvalue()


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


def seed_manifest(root) -> None:
    """Record a manifest via the ordinary CLI (the agent surface)."""
    code, out = run_cli(["status", str(root), "--record"])
    assert code == 0, out


class TestReplBasics(TempCase):
    def test_help_unknown_and_no_root_guard(self):
        code, out = run_repl(["/help", "/bogus", "/status", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("/apply", out)  # the command table is shown
        self.assertIn("Unknown command /bogus", out)
        self.assertIn("no corpus root", out)
        self.assertIn("/use", out)

    def test_bare_text_gets_a_hint(self):
        code, out = run_repl(["status tmp/docs", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("/help", out)

    def test_read_only_renders(self):
        code, out = run_repl(
            [
                f"/use {self.root}",
                "/doctor",
                "/inventory",
                "/clusters",
                "/strategies",
                "/cat WATCHLIST.md",
                "/quit",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("hybrid", out)
        self.assertIn("acme_dossier.md", out)  # /cat shows file content
        self.assertIn("cluster", out.lower())

    def test_use_of_missing_dir_errors_and_survives(self):
        code, out = run_repl([f"/use {self.tmp}/nope", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("not a directory", out)

    def test_error_surfaces_and_repl_survives(self):
        code, out = run_repl([f"/use {self.root}", "/draft", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("no_manifest", out)
        self.assertIn("first tend", out)  # the recovery hint

    def test_apply_before_any_plan_refuses(self):
        code, out = run_repl([f"/use {self.root}", "/apply", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("no compiled plan", out.lower())


class TestReplReviewLoop(TempCase):
    """The core loop: load a plan, decide per move, gate, execute, undo."""

    def setUp(self):
        super().setUp()
        seed_manifest(self.root)
        self.plan_path = write_plan(self.root)

    def test_full_review_loop(self):
        code, out = run_repl(
            [
                f"/use {self.root}",
                f"/load {self.plan_path}",
                "/moves",
                "/no 2",  # globex_dossier stays put
                "/defer 3 parking for later",  # handoff → backlog
                "/dest 4 archive/old/ledger_wb.html",
                "/plan",
                "/render",
                "/apply",
                "y",
                "/quit",
            ]
        )
        self.assertEqual(code, 0, out)
        # Decisions landed on disk:
        self.assertTrue((self.root / "globex_dossier.md").exists())  # /no left it
        self.assertTrue((self.root / "handoff-tp2.md").exists())  # /defer left it
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        self.assertTrue(
            (self.root / "archive/old/ledger_wb.html").exists()
        )  # /dest re-destination landed
        self.assertFalse((self.root / "ledger_wb.html").exists())
        # The rendered gate was shown before the confirm (heading + the
        # re-destined move).
        self.assertIn("Scaffolding plan", out)
        self.assertIn("ledger_wb.html", out)
        # Deferred move recorded as the human-decides backlog.
        code, out = run_cli(["status", str(self.root)])
        self.assertEqual(code, 0)
        unresolved = [u.get("path") for u in out["data"]["still_unresolved"]]
        self.assertIn("handoff-tp2.md", unresolved)

    def test_confirm_defaults_to_no(self):
        code, out = run_repl(
            [
                f"/use {self.root}",
                f"/load {self.plan_path}",
                "/plan",
                "/apply",
                "",  # empty → abort
                "/quit",
            ]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("aborted", out.lower())
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_dirty_decisions_require_replan(self):
        code, out = run_repl(
            [
                f"/use {self.root}",
                f"/load {self.plan_path}",
                "/no 1",
                "/apply",  # decisions not compiled yet
                "/quit",
            ]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("/plan", out)  # told to compile first
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_undo_list_dry_run_and_execute(self):
        code, out = run_cli(["apply", str(self.plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        code, out = run_repl(
            [
                f"/use {self.root}",
                "/undo",  # lists journals, dry-runs the latest
                "/undo 1",  # explicit dry-run
                "/undo 1 !",  # execute after confirm
                "y",
                "/quit",
            ]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("would_restore", out)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research/dossiers/acme_dossier.md").exists())

    def test_undo_bad_index_errors_without_crash(self):
        code, out = run_cli(["apply", str(self.plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, out)
        code, out = run_repl([f"/use {self.root}", "/undo 9", "/undo 0", "/quit"])
        self.assertEqual(code, 0, out)
        self.assertIn("no such journal", out.lower())

    def test_load_missing_and_invalid(self):
        code, out = run_repl(
            [f"/use {self.root}", f"/load {self.tmp}/no-such.json", "/quit"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("cannot read", out.lower())
        bad = self.tmp / "bad.json"
        bad.write_text("{not json")
        code, out = run_repl([f"/use {self.root}", f"/load {bad}", "/quit"])
        self.assertEqual(code, 0, out)
        self.assertIn("not valid json", out.lower())

    def test_draft_writes_inside_state_dir_and_lists(self):
        # A freshly drafted plan is already valid on disk (in .slopkeeper/,
        # which the fingerprint and the catalog both skip) — /apply can
        # run straight after a draft with no edits.
        write(self.root / "glm-notes.md", "# GLM runbook\n\ntensorfold lanes\n")
        code, out = run_repl([f"/use {self.root}", "/draft", "/moves", "/quit"])
        self.assertEqual(code, 0, out)
        draft_file = self.root / ".slopkeeper/draft-plan.json"
        self.assertTrue(draft_file.exists())
        plan = json.loads(draft_file.read_text())
        self.assertTrue(plan["partial"])

    def test_interactive_shows_prompt_and_confirm_default_no(self):
        # A real tty gets `> ` per line and `[y/N]` on confirms; piped
        # input (all other tests) stays prompt-free.
        code, out = run_repl(
            [
                f"/use {self.root}",
                f"/load {self.plan_path}",
                "/plan",
                "/apply",
                "",  # empty → default no
                "/quit",
            ],
            interactive=True,
        )
        self.assertEqual(code, 0, out)
        self.assertIn("> ", out)
        self.assertIn("[y/N]", out)
        self.assertIn("aborted", out.lower())
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_bare_jan_enters_repl(self):
        # No subcommand → the REPL on stdin/stdout; /quit exits 0.
        buf, sin = io.StringIO(), io.StringIO("/quit\n")
        real_stdin = sys.stdin
        sys.stdin = sin
        try:
            with contextlib.redirect_stdout(buf):
                code = cli.main([])
        finally:
            sys.stdin = real_stdin
        self.assertEqual(code, 0)
        self.assertIn("/help", buf.getvalue())
        # The agent surface is unchanged: subcommands still parse.
        code, out = run_cli(["plan", "schema"])
        self.assertEqual(code, 0)
        self.assertIn("moves", out)


class TestCompletion(TempCase):
    """Tab completion: pure matching logic, no readline involved.

    `text` mirrors readline's word semantics: everything after the last
    space/tab — so a word can contain slashes and `~`.
    """

    def setUp(self):
        super().setUp()
        self.plan_path = write_plan(self.root)

    def _s(self) -> _Session:
        s = _Session()
        s.root = self.root
        s.plan = json.loads(self.plan_path.read_text())
        return s

    # --- commands ---

    def test_command_prefixes(self):
        s = self._s()
        self.assertEqual(_compute_matches("/us", "/us", s), ["/use"])
        self.assertEqual(_compute_matches("/st", "/st", s), ["/status", "/strategies"])
        # Slash optional in the typed stem; returned with the slash.
        self.assertIn("/status", _compute_matches("st", "st", s))
        # Empty line: the whole table.
        self.assertIn("/quit", _compute_matches("", "", s))
        # After a space, /status takes no args — nothing.
        self.assertEqual(_compute_matches("/status ", "", s), [])

    def test_unknown_command_completes_nothing(self):
        s = self._s()
        self.assertEqual(_compute_matches("/bogus 1", "1", s), [])
        self.assertEqual(_compute_matches("/bogus ", "", s), [])

    # --- corpus-relative paths (cat / refs / dest) ---

    def test_corpus_relative_walks_components(self):
        s = self._s()
        # A directory completes with its trailing slash…
        self.assertEqual(_compute_matches("/cat do", "do", s), ["docs/"])
        # …so the next tab walks into it (word includes the dir part).
        self.assertEqual(
            _compute_matches("/cat docs/can", "docs/can", s),
            ["docs/canonical.md"],
        )
        # Trailing slash lists the dir's contents.
        self.assertEqual(
            _compute_matches("/cat docs/", "docs/", s), ["docs/canonical.md"]
        )

    def test_corpus_relative_case_insensitive_and_hidden(self):
        s = self._s()
        self.assertEqual(_compute_matches("/cat AC", "AC", s), ["acme_dossier.md"])
        # Hidden entries only when explicitly asked for.
        self.assertNotIn(".obsidian/", _compute_matches("/cat ", "", s))
        self.assertEqual(_compute_matches("/cat .ob", ".ob", s), [".obsidian/"])

    def test_corpus_relative_no_root_or_missing_dir(self):
        s = self._s()
        s.root = None
        self.assertEqual(_compute_matches("/cat do", "do", s), [])
        s.root = self.root
        self.assertEqual(_compute_matches("/cat no-such-dir/x", "no-such-dir/x", s), [])

    def test_refs_completes_like_cat(self):
        s = self._s()
        self.assertEqual(_compute_matches("/refs ac", "ac", s), ["acme_dossier.md"])

    def test_dest_arg0_is_move_numbers_arg1_is_corpus_path(self):
        s = self._s()
        self.assertEqual(_compute_matches("/dest ", "", s), ["1", "2", "3", "4"])
        self.assertEqual(_compute_matches("/dest 1 do", "do", s), ["docs/"])

    # --- move / journal numbers ---

    def test_move_numbers_follow_the_working_plan(self):
        s = self._s()
        self.assertEqual(_compute_matches("/no ", "", s), ["1", "2", "3", "4"])
        self.assertEqual(_compute_matches("/show 2", "2", s), ["2"])
        self.assertEqual(_compute_matches("/defer 1", "1", s), ["1"])
        # No plan loaded → no numbers.
        s.plan = None
        self.assertEqual(_compute_matches("/no ", "", s), [])

    def test_journal_numbers_come_from_disk(self):
        s = self._s()
        self.assertEqual(_compute_matches("/undo 1", "1", s), [])  # no journals yet
        code, out = run_cli(["apply", str(self.plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, out)
        self.assertEqual(_compute_matches("/undo ", "", s), ["1"])
        self.assertEqual(_compute_matches("/undo 9", "9", s), [])

    def test_no_second_arg_completions_where_there_are_none(self):
        s = self._s()
        # /defer's second arg is free text; /cat has no second arg.
        self.assertEqual(_compute_matches("/defer 3 parking", "parking", s), [])
        self.assertEqual(_compute_matches("/cat docs/can x", "x", s), [])

    # --- absolute / home paths (use / load) ---

    def test_absolute_paths_complete_from_fs(self):
        s = self._s()
        m = _compute_matches("/use " + str(self.tmp) + "/c", str(self.tmp) + "/c", s)
        self.assertIn(str(self.tmp) + "/corpus/", m)

    def test_home_dir_tilde_forms(self):
        s = self._s()
        fake_home = self.tmp / "fakehome"
        write(fake_home / "Documents" / "keep.txt", "x")
        write(fake_home / "notes.txt", "x")
        old = os.environ.get("HOME")
        os.environ["HOME"] = str(fake_home)
        try:
            # Bare "~": every home entry, tilde-prefixed, dirs with slash.
            self.assertEqual(
                sorted(_compute_matches("/use ~", "~", s)),
                ["~/Documents/", "~/notes.txt"],
            )
            # "~/Doc" walks into home by name.
            self.assertEqual(
                _compute_matches("/use ~/Doc", "~/Doc", s), ["~/Documents/"]
            )
            # And the next tab walks a component under it.
            self.assertEqual(
                _compute_matches("/use ~/Documents/keep", "~/Documents/keep", s),
                ["~/Documents/keep.txt"],
            )
        finally:
            if old is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old

    def test_load_completes_like_use(self):
        s = self._s()
        self.assertEqual(
            _compute_matches(
                "/load " + str(self.tmp) + "/pla", str(self.tmp) + "/pla", s
            ),
            [str(self.tmp) + "/plan.json"],
        )

    def test_tilde_mid_word_is_not_expanded(self):
        s = self._s()
        self.assertEqual(_compute_matches("/use x~y", "x~y", s), [])

    # --- strategies ---

    def test_strategy_ids(self):
        s = self._s()
        self.assertEqual(_compute_matches("/strategies hy", "hy", s), ["hybrid"])

    # --- no session state ---

    def test_everything_degrades_without_root(self):
        s = _Session()
        self.assertEqual(_compute_matches("/cat do", "do", s), [])
        self.assertEqual(_compute_matches("/dest 1 do", "do", s), [])
        self.assertEqual(_compute_matches("/undo ", "", s), [])
        self.assertEqual(_compute_matches("/no ", "", s), [])
        # Commands still complete — /use is how you fix it.
        self.assertEqual(_compute_matches("/us", "/us", s), ["/use"])


if __name__ == "__main__":
    unittest.main()
