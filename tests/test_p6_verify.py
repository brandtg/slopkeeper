# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P6 Slice C — the verify loop: the all-or-nothing promise, for semantics.

Everything before 0.3.0 guaranteed TOPOLOGY: moves were atomic, citations
were rewritten, undo could put the bytes back. None of it answered the only
question that matters when the corpus is a codebase — *does it still run?*
A plan that breaks an import or a Makefile path used to apply `ok: true`
and leave a beautifully journaled corpus that no longer builds.

The tests here are organized around the one promise that makes the feature
worth its risk (a plan file now carries shell commands): **a corpus whose
own checks do not pass is a corpus that was never touched.** The audit's
acceptance line — "apply leaves the tree byte-identical to pre-apply" — is
asserted with the repo's own `snapshot_tree` helper, and the mirror case
(`--no-auto-undo`) proves the operator can still choose to debug inside the
broken state without the tool pretending it succeeded.

Also pinned, because it is the part that gets forgotten: the commands are
PRINTED in the human gate. A plan file is untrusted input; the gate is the
trust boundary, and a check nobody saw is a check that surprises someone at
the first failure.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from helpers import TempCase, prune_slopkeeper, snapshot_tree, write_plan
from slopkeeper import cli
from slopkeeper import apply as apply_module
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError
from slopkeeper.plan import PLAN_TEMPLATE, render_markdown, validate_plan
from slopkeeper.verify import (
    DEFAULT_VERIFY_TIMEOUT,
    MAX_VERIFY_TIMEOUT,
    VERIFY_TAIL_CHARS,
    normalize_verify,
    run_verify,
)

HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


def tree_state(root: Path):
    """The corpus as the atomicity tests see it: tool residue pruned.

    The journal and tend.lock live in .slopkeeper/ and survive a rollback
    by design, so comparing raw snapshots would report a difference that
    has nothing to do with the corpus (helpers.prune_slopkeeper).
    """
    return prune_slopkeeper(snapshot_tree(root))


def keep_all(root: Path) -> list[str]:
    """Every top-level entry, so a one-file move passes coverage."""
    return sorted(
        p.name for p in root.iterdir() if p.name not in {".slopkeeper", "__pycache__"}
    )


TRUE = 'python -c "import sys; sys.exit(0)"'
FALSE = 'python -c "import sys; sys.exit(3)"'


class TestNormalization(TempCase):
    """One canonical shape, so validate, the gate and apply cannot disagree."""

    def test_string_shorthand(self):
        out = normalize_verify("pytest -q", self.root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["command"], "pytest -q")
        self.assertIsNone(out[0]["cwd"])
        self.assertEqual(out[0]["timeout"], DEFAULT_VERIFY_TIMEOUT)
        self.assertEqual(out[0]["name"], "pytest -q")

    def test_list_of_strings_and_objects_in_order(self):
        out = normalize_verify(
            ["ruff check .", {"command": "pytest -q", "name": "tests"}], self.root
        )
        self.assertEqual([e["command"] for e in out], ["ruff check .", "pytest -q"])
        self.assertEqual(out[1]["name"], "tests")

    def test_duplicate_command_and_cwd_collapses(self):
        # The same check declared twice would run twice and fail twice, and
        # an agent re-deriving a plan's fields does exactly this.
        out = normalize_verify(["pytest -q", "pytest -q"], self.root)
        self.assertEqual(len(out), 1)

    def test_same_command_in_different_cwd_is_not_a_duplicate(self):
        out = normalize_verify(
            [
                {"command": "npm test", "cwd": "web"},
                {"command": "npm test", "cwd": "docs"},
            ],
            self.root,
        )
        self.assertEqual(len(out), 2)

    def test_empty_forms_normalize_to_nothing(self):
        for raw in (None, [], {}, ""):
            self.assertEqual(normalize_verify(raw, self.root), [])

    def test_folder_entry_stamps_its_cwd_and_origin(self):
        (self.root / "reports").mkdir(exist_ok=True)
        out = normalize_verify("pytest -q", self.root, folder="reports")
        self.assertEqual(out[0]["cwd"], "reports")
        self.assertEqual(out[0]["folder"], "reports")

    def test_dot_cwd_means_the_corpus_root(self):
        # `{"cwd": "."}` is how an author says "run it at the top". Refusing
        # it as a malformed plan would only teach people to omit cwd.
        for raw in (".", "./", " . "):
            out = normalize_verify([{"command": "pytest -q", "cwd": raw}], self.root)
            self.assertIsNone(out[0]["cwd"], raw)

    def test_explicit_dot_overrides_the_folder_default(self):
        # A folder-level check that names "." deliberately runs at the corpus
        # root — the folder would otherwise inherit its own directory.
        out = normalize_verify(
            [{"command": "pytest -q", "cwd": "."}], self.root, folder="web"
        )
        self.assertIsNone(out[0]["cwd"])
        self.assertEqual(out[0]["folder"], "web")

    def test_normalized_output_renormalizes_unchanged(self):
        # `plan validate --out` writes the normalized plan; feeding that
        # artifact back must not trip the strict key check.
        once = normalize_verify(
            [{"command": "pytest -q", "cwd": None, "name": "t", "timeout": 30}],
            self.root,
        )
        self.assertEqual(normalize_verify(once, self.root), once)

    def _bad(self, raw):
        with self.assertRaises(SlopKeeperError) as cm:
            normalize_verify(raw, self.root, where="plan.verify")
        self.assertEqual(cm.exception.code, "plan_bad_verify")
        return cm.exception

    def test_refuses_non_string_non_object(self):
        self._bad([42])

    def test_refuses_empty_command(self):
        self._bad([{"command": "   "}])

    def test_refuses_unknown_key(self):
        # A typo'd key is a check that silently never runs — the one
        # outcome a verification feature cannot afford.
        self._bad([{"command": "pytest", "wrokspace": "web"}])

    def test_refuses_timeout_over_the_ceiling(self):
        err = self._bad([{"command": "pytest", "timeout": MAX_VERIFY_TIMEOUT + 1}])
        self.assertIn("ceiling", err.message)

    def test_refuses_non_positive_timeout(self):
        self._bad([{"command": "pytest", "timeout": 0}])
        self._bad([{"command": "pytest", "timeout": True}])

    def test_refuses_cwd_outside_the_corpus(self):
        # Reported as plan_bad_verify, not the generic path_escape: a bad
        # cwd is a malformed PLAN field, and every "your plan is wrong"
        # answer belongs in one exit class with one recovery.
        err = self._bad([{"command": "pytest", "cwd": "../elsewhere"}])
        self.assertIn("corpus", err.message.lower())

    def test_accepts_a_cwd_inside_the_corpus(self):
        (self.root / "web").mkdir()
        out = normalize_verify([{"command": "npm test", "cwd": "web"}], self.root)
        self.assertEqual(out[0]["cwd"], "web")


class TestPlanValidation(TempCase):
    """Where the field lives, and what an empty one does to payloads."""

    def test_plan_level_verify_normalizes_onto_the_plan(self):
        plan_path = write_plan(self.root, verify="pytest -q")
        validated = validate_plan(
            json.loads(plan_path.read_text()), self.root, allow_partial=True
        )
        self.assertEqual([v["command"] for v in validated["verify"]], ["pytest -q"])

    def test_absent_verify_stays_absent_from_the_plan(self):
        plan_path = write_plan(self.root)
        plan = json.loads(plan_path.read_text())
        self.assertNotIn("verify", plan)
        validated = validate_plan(plan, self.root, allow_partial=True)
        self.assertNotIn("verify", validated)

    def test_empty_verify_list_is_not_persisted_as_a_key(self):
        # Absent, not empty: a corpus with no checks must not grow
        # envelope/gate keys that read as "checked nothing, deliberately".
        plan_path = write_plan(self.root, verify=[])
        validated = validate_plan(
            json.loads(plan_path.read_text()), self.root, allow_partial=True
        )
        self.assertNotIn("verify", validated)

    def test_folder_verify_command_merges_after_plan_level(self):
        (self.root / "reports").mkdir(exist_ok=True)
        plan_path = write_plan(
            self.root,
            verify=["pytest -q"],
            folders=[
                {
                    "path": "reports",
                    "kind": "case",
                    "role": "r",
                    "verify_command": ["npm test", "tsc --noEmit"],
                }
            ],
        )
        validated = validate_plan(
            json.loads(plan_path.read_text()), self.root, allow_partial=True
        )
        self.assertEqual(
            [v["command"] for v in validated["verify"]],
            ["pytest -q", "npm test", "tsc --noEmit"],
        )
        # The folder entry keeps its own key (scaffold never runs it; apply
        # does) and the merged entry knows where it came from.
        self.assertEqual(
            validated["folders"][0]["verify_command"], ["npm test", "tsc --noEmit"]
        )
        self.assertEqual(validated["verify"][1]["cwd"], "reports")

    def test_folder_only_plan_still_carries_the_merged_run(self):
        (self.root / "reports").mkdir(exist_ok=True)
        plan_path = write_plan(
            self.root,
            folders=[
                {
                    "path": "reports",
                    "kind": "case",
                    "role": "r",
                    "verify_command": "pytest",
                }
            ],
        )
        validated = validate_plan(
            json.loads(plan_path.read_text()), self.root, allow_partial=True
        )
        self.assertEqual([v["command"] for v in validated["verify"]], ["pytest"])


class TestDryRunNeverExecutes(TempCase):
    """A dry run that ran the checks would be the worst surprise here."""

    def test_dry_run_reports_would_verify_and_touches_nothing(self):
        marker = self.root / "RAN"
        plan_path = write_plan(
            self.root,
            verify=[f"python -c \"open({str(marker)!r},'w').write('x')\""],
        )
        data = apply_plan(plan_path, self.root, execute=False)
        self.assertFalse(marker.exists(), "a dry run executed the verify command")
        self.assertEqual(len(data["would_verify"]), 1)
        self.assertEqual(data["would_verify"][0]["timeout"], DEFAULT_VERIFY_TIMEOUT)
        self.assertNotIn("verify", data)  # the run block is execute-only

    def test_dry_run_omits_the_key_when_nothing_is_declared(self):
        data = apply_plan(write_plan(self.root), self.root, execute=False)
        self.assertNotIn("would_verify", data)


class TestGateShowsTheCommands(TempCase):
    """The trust boundary: shell commands from a plan, printed for approval."""

    def _plan(self, **over):
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(self.root),
            "strategy": "hybrid",
            "based_on": {"corpus_fingerprint": "fp6:stub"},
            "moves": [{"from": "acme_dossier.md", "to": "research/acme.md"}],
            "keep": [],
            "unresolved": [],
            "folders": [],
        }
        plan.update(over)
        return plan

    def test_every_command_and_its_cwd_appears(self):
        md = render_markdown(self._plan(verify=[{"command": "npm test", "cwd": "web"}]))
        self.assertIn("Verification", md)
        self.assertIn("npm test", md)
        self.assertIn("`web`", md)
        self.assertIn("UNDOES the whole tend", md)
        self.assertIn("shell commands from the plan", md)

    def test_folder_origin_is_named(self):
        md = render_markdown(
            self._plan(
                folders=[{"path": "x", "kind": "case", "verify_command": "pytest"}]
            )
        )
        self.assertIn("pytest", md)
        self.assertIn("folder `x`", md)

    def test_no_section_without_commands(self):
        md = render_markdown(self._plan())
        self.assertNotIn("Verification", md)

    def test_schema_template_carries_an_empty_verify(self):
        # The template documents the field WITHOUT shipping a command:
        # unlike a move, a copy-pasted verify line is executable.
        self.assertIn("verify", PLAN_TEMPLATE)
        self.assertEqual(PLAN_TEMPLATE["verify"], [])


class TestRunVerify(TempCase):
    """The runner itself: statuses, tails, timeouts, missing cwd."""

    def _spec(self, command, **over):
        base = {
            "command": command,
            "name": command,
            "cwd": over.get("cwd"),
            "timeout": over.get("timeout", 30),
        }
        if over.get("folder"):
            base["folder"] = over["folder"]
        return base

    def test_pass_and_fail_statuses_with_returncode(self):
        rows, failure = run_verify([self._spec(TRUE)], self.root)
        self.assertEqual(rows[0]["status"], "passed")
        self.assertIsNone(failure)
        rows, failure = run_verify([self._spec(FALSE)], self.root)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["returncode"], 3)
        self.assertEqual(failure["command"], FALSE)

    def test_output_tail_is_captured_and_bounded(self):
        big = "python -c \"import sys; sys.stdout.write('x' * %d + 'END')\"" % (
            VERIFY_TAIL_CHARS * 2
        )
        rows, _ = run_verify([self._spec(big)], self.root)
        self.assertTrue(rows[0]["stdout"].endswith("END"))
        self.assertLessEqual(len(rows[0]["stdout"]), VERIFY_TAIL_CHARS)
        self.assertTrue(rows[0]["output_truncated"])

    def test_stops_at_the_first_failure(self):
        marker = self.root / "SECOND"
        rows, failure = run_verify(
            [
                self._spec(FALSE),
                self._spec(f"python -c \"open({str(marker)!r},'w')\""),
            ],
            self.root,
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(marker.exists())
        self.assertEqual(failure["status"], "failed")

    def test_timeout_is_a_failure_not_a_hang(self):
        rows, failure = run_verify(
            [self._spec('python -c "import time; time.sleep(30)"', timeout=1)],
            self.root,
        )
        self.assertEqual(rows[0]["status"], "timeout")
        self.assertIn("timed out", rows[0]["note"])
        self.assertEqual(failure, rows[0])

    def test_missing_cwd_skips_loudly_instead_of_failing(self):
        # Folders are scaffolded AFTER apply, so refusing the tend for a
        # check that points at a not-yet-created folder would make the
        # feature unusable on the first tend of a new layout.
        rows, failure = run_verify(
            [self._spec(TRUE, cwd="not-created", folder="not-created")], self.root
        )
        self.assertEqual(rows[0]["status"], "skipped_missing_cwd")
        self.assertIsNone(failure)


class TestApplyIntegration(TempCase):
    """The promise, end to end."""

    def test_passing_check_reports_and_the_tend_stands(self):
        plan_path = write_plan(self.root, verify=[{"command": TRUE, "name": "ok"}])
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(data["verify"]["status"], "passed")
        self.assertEqual(data["verify"]["ran"], 1)
        self.assertTrue(data["moved"])

    def test_no_verify_key_when_the_plan_has_no_checks(self):
        # The absence is a fact: an agent must be able to tell "this corpus
        # declares no checks" from "the checks ran".
        data = apply_plan(write_plan(self.root), self.root, execute=True)
        self.assertNotIn("verify", data)

    def test_failing_check_rolls_back_to_a_byte_identical_tree(self):
        # THE acceptance criterion of the slice.
        plan_path = write_plan(
            self.root,
            verify=[
                {
                    "command": "python -c \"import sys; print('boom', "
                    'file=sys.stderr); sys.exit(1)"',
                    "name": "gate",
                }
            ],
        )
        before = tree_state(self.root)
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "verify_failed")
        self.assertIs(cm.exception.data["verify"]["auto_undone"], True)
        self.assertEqual(cm.exception.data["verify"]["failed"]["name"], "gate")
        self.assertIn("boom", cm.exception.data["verify"]["failed"]["stderr"])
        self.assertEqual(tree_state(self.root), before, "the tend was not rolled back")

    def test_the_rollback_reverts_a_rewrite_on_a_line_that_already_said_it(self):
        # The live probe's failure mode: a stay-put file whose rewritten line
        # ALSO contained the destination string (a URL, a changelog line). The
        # reversal must be exact, not count-based, or the rolled-back tend
        # leaves a citation to a path that no longer exists — and the
        # byte-identical promise is false.
        (self.root / "WATCHLIST.md").write_text(
            "# Watchlist\n\nSee acme_dossier.md for flows; the published copy is at "
            "https://x.example/research/dossiers/acme_dossier.md.\n",
            encoding="utf-8",
        )
        plan_path = write_plan(self.root, verify=[FALSE])
        before = tree_state(self.root)
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "verify_failed")
        self.assertEqual(
            tree_state(self.root), before, "a rewrite survived the rollback"
        )

    def test_the_rollback_is_the_executed_cli_path_exit_2(self):
        plan_path = write_plan(self.root, verify=[FALSE])
        code, env = run_cli(["apply", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 2)
        self.assertIs(env["ok"], False)
        self.assertEqual(env["error"]["code"], "verify_failed")
        self.assertIs(env["data"]["verify"]["auto_undone"], True)
        self.assertIn("journal", env["data"])
        self.assertIn("plan render", env["error"]["suggested_command"])

    def test_the_undone_state_is_really_undone(self):
        plan_path = write_plan(self.root, verify=[FALSE])
        target = self.root / "research/dossiers/acme_dossier.md"
        with self.assertRaises(SlopKeeperError):
            apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(target.exists())
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_journaled_verify_lines_do_not_confuse_undo(self):
        # `verify` is a new op kind in a file undo replays; unknown ops are
        # skipped by design, and a tend that passed must still undo.
        plan_path = write_plan(self.root, verify=[TRUE])
        data = apply_plan(plan_path, self.root, execute=True)
        rows = [
            json.loads(line)
            for line in Path(data["journal"]).read_text().splitlines()
            if line.strip()
        ]
        self.assertIn("verify", [row["op"] for row in rows])
        undone = undo_journal(Path(data["journal"]), self.root, execute=True)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertNotIn("verify", {o.get("op") for o in undone["operations"]})

    def test_no_auto_undo_keeps_changes_and_is_loud(self):
        plan_path = write_plan(self.root, verify=[FALSE])
        before = tree_state(self.root)
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True, auto_undo=False)
        err = cm.exception
        self.assertEqual(err.code, "verify_failed")
        self.assertIs(err.data["verify"]["auto_undone"], False)
        self.assertTrue(err.suggested_command.startswith("slopkeeper undo"))
        # The changes are still there — that was the whole point of the flag.
        self.assertNotEqual(tree_state(self.root), before)
        # ...and the undo it suggests really does put everything back.
        undo_journal(Path(err.data["journal"]), self.root, execute=True, force=True)
        self.assertEqual(tree_state(self.root), before)

    def test_partial_apply_skips_the_checks(self):
        # A check that fails because HALF a plan applied would report a
        # broken corpus and blame the plan. The partial path is already
        # loud (T5) with `next` pointing at the undo.
        plan_path = write_plan(self.root, verify=[FALSE])
        real_move = apply_module._move_no_clobber
        boom = {"used": False}

        def flaky(src_abs, dst_abs):
            # One injected mid-run write failure, the same way the apply
            # suite injects them: a real OSError on the move itself.
            if str(src_abs).endswith("ledger_wb.html") and not boom["used"]:
                boom["used"] = True
                raise OSError("injected disk failure")
            return real_move(src_abs, dst_abs)

        with mock.patch("slopkeeper.apply._move_no_clobber", side_effect=flaky):
            data = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(data["partial"])
        self.assertEqual(data["verify"]["status"], "skipped")
        self.assertEqual(data["verify"]["note"], "partial-apply")
        self.assertIn("verify_skipped_partial", data["warnings"])

    def test_missing_folder_check_warns_instead_of_failing_the_tend(self):
        plan_path = write_plan(
            self.root,
            folders=[
                {
                    "path": "webhooks",
                    "kind": "case",
                    "role": "r",
                    "verify_command": TRUE,
                }
            ],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(data["verify"]["status"], "passed_with_skips")
        self.assertIn("verify_skipped_missing_cwd", data["warnings"])

    def test_folder_check_runs_inside_the_folder(self):
        # The move creates research/dossiers/ with the dossier in it, so a
        # folder check can assert its own contents — which is the point of
        # putting verify_command on a folder at all.
        plan_path = write_plan(
            self.root,
            folders=[
                {
                    "path": "research/dossiers",
                    "kind": "case",
                    "role": "r",
                    "verify_command": "test -f acme_dossier.md",
                }
            ],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(data["verify"]["status"], "passed")
        self.assertEqual(data["verify"]["commands"][0]["folder"], "research/dossiers")

    @unittest.skipUnless(HAS_PYTEST, "pytest is not installed")
    def test_a_real_pytest_suite_broken_by_a_move_leaves_no_trace(self):
        # The audit's acceptance scenario verbatim: a corpus with a trivial
        # pytest suite, a plan whose move breaks it, and apply proving the
        # tree is byte-identical to pre-apply afterwards.
        #
        # The dependency is an IMPORT, not a path string: a string literal
        # naming a moved path is exactly what the citation rewrite pass
        # FIXES (an earlier draft of this test asserted on one and the
        # check passed, because the pass had repaired the literal — the
        # feature working, the test asking the wrong question). An import
        # is the dependency nothing repairs, which is the case the verify
        # loop exists for.
        (self.root / "pkg").mkdir(exist_ok=True)
        (self.root / "pkg/__init__.py").write_text("")
        (self.root / "pkg/config.py").write_text("VALUE = 1\n")
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests/test_value.py").write_text(
            "from pkg.config import VALUE\n\n\ndef test_the_value_the_corpus_ships():\n"
            "    assert VALUE == 1\n"
        )
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "pkg/config.py", "to": "lib/config.py", "reason": "flatten"}
            ],
            keep=keep_all(self.root),
            folders=[],
            # -B and -p no:cacheprovider: the check itself must not litter,
            # or the byte-identical claim below would be measuring pytest's
            # cache policy rather than the rollback. What a check leaves
            # behind is its own business — see the caveat test below.
            verify=[f"{sys.executable} -B -m pytest -q -p no:cacheprovider tests"],
        )
        before = tree_state(self.root)
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "verify_failed")
        failed = cm.exception.data["verify"]["failed"]
        self.assertEqual(failed["cwd"], ".")
        # The check's own words, not just its exit code.
        self.assertIn("No module named 'pkg.config'", failed["stdout"])
        self.assertEqual(tree_state(self.root), before)

    def test_a_checks_own_side_effects_survive_the_rollback(self):
        # The limitation, stated as a test rather than left to be discovered:
        # the rollback reverses what the JOURNAL recorded — this tend's
        # moves, rewrites and edits. Anything the check itself wrote (a
        # cache dir, a build output, a report) was never part of the tend,
        # and undo does not guess about files it never journaled. Deleting
        # them would also break the tool's one absolute rule.
        (self.root / "pkg").mkdir(exist_ok=True)
        (self.root / "pkg/__init__.py").write_text("")
        (self.root / "pkg/config.py").write_text("VALUE = 1\n")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "pkg/config.py", "to": "lib/config.py", "reason": "flatten"}
            ],
            keep=keep_all(self.root),
            folders=[],
            verify=[
                "python -c \"open('REPORT.txt','w').write('from the check')\"; exit 1"
            ],
        )
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertIs(cm.exception.data["verify"]["auto_undone"], True)
        # The tend is gone...
        self.assertTrue((self.root / "pkg/config.py").exists())
        self.assertFalse((self.root / "lib/config.py").exists())
        # ...and the check's own artifact is exactly where it left it.
        self.assertEqual((self.root / "REPORT.txt").read_text(), "from the check")

    @unittest.skipUnless(HAS_PYTEST, "pytest is not installed")
    def test_a_check_the_pass_repairs_lets_the_tend_stand(self):
        # The other side of the same coin, pinned so the pair of behaviors
        # cannot silently swap: when the only thing the move broke was a
        # PATH STRING, the rewrite pass fixes it, the corpus's own check
        # passes, and the tend stands.
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "data/rows.csv").write_text("a,b\n")
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests/test_rows.py").write_text(
            "import os\n\n\ndef test_rows_file_is_where_the_code_says():\n"
            "    assert os.path.exists('data/rows.csv')\n"
        )
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "data/rows.csv", "to": "archive/rows.csv", "reason": "archive"}
            ],
            keep=keep_all(self.root),
            folders=[],
            verify=[f"{sys.executable} -m pytest -q tests"],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(data["verify"]["status"], "passed")
        # The literal the check asserts on was rewritten with the move.
        self.assertIn(
            "archive/rows.csv", (self.root / "tests/test_rows.py").read_text()
        )
