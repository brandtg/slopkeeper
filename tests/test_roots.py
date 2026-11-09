# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""roots.py + `status --all` — the multi-corpus registry.

SLOPKEEPER_HOME is pointed at a tmpdir so the real ~/.config is never
touched; the registry-path resolution itself is tested at the module
level with a patched environment.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from helpers import TempCase, build_corpus
from slopkeeper import cli
from slopkeeper.roots import ROOTS_SCHEMA, roots_path
from slopkeeper.status import MANIFEST_SCHEMA


def run_main(argv: list[str]) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed stdout JSON)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class RootsCase(TempCase):
    """Temp corpus + SLOPKEEPER_HOME pointed at a tmpdir (XDG never read)."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(
            os.environ,
            {
                "SLOPKEEPER_HOME": str(self.tmp / "config"),
                "XDG_CONFIG_HOME": str(self.tmp / "unused-xdg"),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class TestRootsPath(unittest.TestCase):
    def test_slopkeeper_home_wins_over_xdg(self):
        with mock.patch.dict(
            os.environ, {"SLOPKEEPER_HOME": "/a", "XDG_CONFIG_HOME": "/b"}
        ):
            self.assertEqual(roots_path(), Path("/a/roots.json"))

    def test_xdg_default(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/b"}):
            os.environ.pop("SLOPKEEPER_HOME", None)
            self.assertEqual(roots_path(), Path("/b/slopkeeper/roots.json"))

    def test_home_default(self):
        with mock.patch.dict(os.environ, {"HOME": "/h"}):
            os.environ.pop("SLOPKEEPER_HOME", None)
            os.environ.pop("XDG_CONFIG_HOME", None)
            self.assertEqual(roots_path(), Path("/h/.config/slopkeeper/roots.json"))


class TestRootsRegistry(RootsCase):
    def test_add_list_remove_round_trip(self):
        code, out = run_main(["roots", "add", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "roots.add")
        self.assertTrue(out["ok"])
        self.assertTrue(out["data"]["added"])
        self.assertEqual(out["data"]["roots"], [str(self.root)])

        code, out = run_main(["roots", "list"])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "roots.list")
        self.assertEqual(out["data"], [str(self.root)])

        code, out = run_main(["roots", "remove", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "roots.remove")
        self.assertEqual(out["data"]["roots"], [])

        code, out = run_main(["roots", "list"])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"], [])

    def test_readd_is_idempotent(self):
        run_main(["roots", "add", str(self.root)])
        code, out = run_main(["roots", "add", str(self.root)])
        self.assertEqual(code, 0)
        self.assertFalse(out["data"]["added"])
        self.assertEqual(out["data"]["roots"], [str(self.root)])

    def test_add_resolves_relative_to_absolute(self):
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp)
        code, out = run_main(["roots", "add", "corpus"])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["roots"], [str(self.root.resolve())])

    def test_add_stores_sorted_unique_under_slopkeeper_home(self):
        second = self.tmp / "zeta"  # sorts after self.root ("corpus")
        second.mkdir()
        run_main(["roots", "add", str(second)])
        run_main(["roots", "add", str(self.root)])
        run_main(["roots", "add", str(second)])  # dup dropped
        registry = json.loads(roots_path().read_text(encoding="utf-8"))
        self.assertEqual(registry["schema"], "slopkeeper/roots@1")
        self.assertEqual(registry["roots"], sorted([str(self.root), str(second)]))
        self.assertTrue(roots_path().is_relative_to(self.tmp / "config"))

    def test_add_nonexistent_or_file_errors(self):
        code, out = run_main(["roots", "add", str(self.tmp / "nowhere")])
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "root_not_dir")

        a_file = self.root / "WATCHLIST.md"
        code, out = run_main(["roots", "add", str(a_file)])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "root_not_dir")

    def test_remove_unregistered_errors(self):
        run_main(["roots", "add", str(self.root)])
        code, out = run_main(["roots", "remove", str(self.tmp / "elsewhere")])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "root_not_registered")
        # The registry is unchanged by the refusal.
        _, out = run_main(["roots", "list"])
        self.assertEqual(out["data"], [str(self.root)])

    def test_corrupt_registry_fails_cleanly(self):
        p = roots_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json", encoding="utf-8")
        code, out = run_main(["roots", "list"])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "roots_unreadable")

    def test_roots_field_must_be_a_list_of_strings(self):
        # T3(d): {"roots": "not-a-list"} used to iterate characters and
        # hand them back as registered roots; garbage refuses loudly.
        p = roots_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"schema": ROOTS_SCHEMA, "roots": "x"}), encoding="utf-8"
        )
        code, out = run_main(["roots", "list"])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "roots_bad_schema")
        # A non-list, non-string value fails the same way (today: a bare
        # TypeError → internal).
        p.write_text(json.dumps({"schema": ROOTS_SCHEMA, "roots": 5}), encoding="utf-8")
        code, out = run_main(["roots", "list"])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "roots_bad_schema")


class TestStatusAll(RootsCase):
    def _register(self, *paths: Path) -> None:
        for p in paths:
            code, out = run_main(["roots", "add", str(p)])
            self.assertEqual(code, 0, out)

    def test_two_corpora_one_tended_one_never(self):
        tended = self.root
        never = build_corpus(self.tmp / "corpus-b")
        self._register(never, tended)  # registry is sorted
        code, _ = run_main(["status", str(tended), "--record"])
        self.assertEqual(code, 0)
        (tended / "NEW-SLOP.md").write_text("# Globex memo\n\nmarket share.\n")

        code, out = run_main(["status", "--all"])
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["command"], "status")
        self.assertEqual(out["data"], sorted(out["data"], key=lambda e: e["root"]))
        self.assertEqual(
            [e["root"] for e in out["data"]],
            [str(tended.resolve()), str(never.resolve())],
        )

        b, a = out["data"]  # "corpus" < "corpus-b"
        self.assertFalse(b["never_tended"])
        self.assertEqual(b["new_count"], 1)
        self.assertEqual(b["modified_count"], 0)
        self.assertEqual(b["removed_count"], 0)
        self.assertEqual(b["stale_folders"], 0)
        self.assertEqual(b["unresolved_count"], 0)
        self.assertEqual(
            set(b),
            {
                "root",
                "ok",
                "never_tended",
                "new_count",
                "modified_count",
                "removed_count",
                "stale_folders",
                "unresolved_count",
            },
        )
        self.assertTrue(a["ok"])
        self.assertTrue(a["never_tended"])  # counts meaningless there
        self.assertEqual(a["new_count"], 0)

    def test_missing_root_reported_per_root_not_fatal(self):
        gone = self.tmp / "gone-corpus"
        gone.mkdir()
        self._register(gone, self.root)
        run_main(["status", str(self.root), "--record"])
        gone.rmdir()

        code, out = run_main(["status", "--all"])
        self.assertEqual(code, 0)  # not fatal
        self.assertTrue(out["ok"])
        live, missing = out["data"]  # "corpus" < "gone-corpus"
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["error"]["code"], "root_missing")
        self.assertTrue(live["ok"])
        self.assertFalse(live["never_tended"])

    def test_empty_registry_errors_cleanly(self):
        code, out = run_main(["status", "--all"])
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "no_roots")

    def test_status_without_root_or_all_errors_cleanly(self):
        code, out = run_main(["status"])
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "no_root")

    def test_per_root_status_still_works(self):
        # Regression pin: the positional path is untouched by nargs="?".
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "status")
        self.assertTrue(out["data"]["never_tended"])


class TestStatusAllPoisonedRoots(RootsCase):
    """T3(a)+(b): one poisoned state file must never kill the digest."""

    def _register(self, *paths: Path) -> None:
        for p in paths:
            code, out = run_main(["roots", "add", str(p)])
            self.assertEqual(code, 0, out)

    def _register_poisoned(self, poison: dict, *, disk_file: bool = False) -> Path:
        bad = self.tmp / "corpus-bad"
        bad.mkdir(parents=True)
        if disk_file:
            # Only files present on disk AND in the manifest reach the
            # change comparison — put one there so the garbage entry is
            # actually evaluated.
            (bad / "x.md").write_text("x\n", encoding="utf-8")
        (bad / ".slopkeeper").mkdir()
        (bad / ".slopkeeper" / "manifest.json").write_text(
            json.dumps(poison), encoding="utf-8"
        )
        self._register(bad, self.root)
        run_main(["status", str(self.root), "--record"])
        return bad

    def test_null_path_backlog_is_one_row_not_dead_digest(self):
        # The ticket's probe: `slopkeeper status --all` with one null-path root
        # returns ALL rows — today the TypeError zeroes the whole run.
        bad = self._register_poisoned(
            {
                "schema": MANIFEST_SCHEMA,
                "fingerprint": "x",
                "file_count": 0,
                "files": {},
                "unresolved": [{"path": None, "reason": "poison"}],
            }
        )
        code, out = run_main(["status", "--all"])
        self.assertEqual(code, 0, out)  # the digest survives
        self.assertTrue(out["ok"])
        rows = {r["root"]: r for r in out["data"]}
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[str(bad.resolve())]["ok"])
        # Per-root detail carries the skip count instead of a crash.
        code, out = run_main(["status", str(bad)])
        self.assertEqual(code, 0, out)
        self.assertEqual(out["data"]["skipped_bad_unresolved"], 1)
        self.assertEqual(out["data"]["still_unresolved"], [])

    def test_root_evaluation_failure_is_a_row_not_a_dead_run(self):
        # T3(b): status_all caught only SlopKeeperError, so any other crash
        # in one root's evaluation zeroed the whole digest. A failed root
        # is now a contained ok:false row; the others still report.
        bad = self._register_poisoned(
            {
                "schema": MANIFEST_SCHEMA,
                "files": {"x.md": 5},  # not a [size, mtime, sha] triple
                "unresolved": [],
            },
            disk_file=True,
        )
        code, out = run_main(["status", "--all"])
        self.assertEqual(code, 0, out)  # never a dead run
        rows = {r["root"]: r for r in out["data"]}
        self.assertTrue(rows[str(self.root.resolve())]["ok"])
        self.assertFalse(rows[str(bad.resolve())]["ok"])
        self.assertEqual(rows[str(bad.resolve())]["error"]["code"], "internal")


if __name__ == "__main__":
    unittest.main()
