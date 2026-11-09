# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""§5.5 tending lock: concurrent tends serialize, readers stay free.

The lock is simulated exactly the way a second agent would hold it:
raw fcntl on `.slopkeeper/tend.lock` from the test process. Everything
here is non-blocking — a contended acquire raises instead of waiting —
so no test can deadlock.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from helpers import TempCase, prune_slopkeeper, snapshot_tree, write_plan

from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError, TEND_LOCK_NAME, corpus_fingerprint
from slopkeeper.scaffold import scaffold_from_plan
from slopkeeper.status import status


def _one_move_plan(root: Path, src: str, dst: str, keeps: list[str]) -> Path:
    """Minimal complete single-move plan, fingerprinted NOW."""
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
        "moves": [{"from": src, "to": dst, "reason": "test"}],
        "keep": keeps,
        "unresolved": [],
        "folders": [],
    }
    p = root.parent / f"plan-{src}.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    return p


@contextmanager
def _held_lock(root: Path):
    """Hold the tend lock the way a rival agent would: fresh fd, flock NB."""
    lock_path = root / ".slopkeeper" / TEND_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock_path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _probe_lock_free(root: Path) -> None:
    """Assert the lock can be re-acquired non-blocking (raises if held)."""
    with _held_lock(root):
        pass


def _all_keeps(root: Path, *extra: str) -> list[str]:
    base = [
        "WATCHLIST.md",
        "docs",
        "lead-gen",
        "spark-glm-5.3",
        "globex_dossier.md",
        "handoff-tp2.md",
        "ledger_wb.html",
    ]
    return sorted(set(base) | set(extra))


class TestTendLock(TempCase):
    def test_second_concurrent_execute_raises_tending_in_progress(self):
        plan_path = write_plan(self.root)
        before = prune_slopkeeper(snapshot_tree(self.root))
        with _held_lock(self.root):
            with self.assertRaises(SlopKeeperError) as cm:
                apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "tending_in_progress")
        # The agent-facing contract: say what to do, including the
        # dead-agent case (a crashed process releases nothing it holds
        # — the OS does).
        self.assertIn("another tend is running", cm.exception.recovery)
        self.assertIn("dead agent", cm.exception.recovery)
        # Seatbelt: the refused apply moved nothing.
        self.assertEqual(prune_slopkeeper(snapshot_tree(self.root)), before)
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_sequential_applies_do_not_self_deadlock(self):
        # lock → release → lock again (fresh fd per acquisition): the
        # second apply must go through, not hang or refuse.
        p1 = _one_move_plan(
            self.root,
            "acme_dossier.md",
            "research/acme_dossier.md",
            _all_keeps(self.root),
        )
        r1 = apply_plan(p1, self.root, execute=True)
        p2 = _one_move_plan(
            self.root,
            "globex_dossier.md",
            "research/globex_dossier.md",
            _all_keeps(self.root, "research"),
        )
        r2 = apply_plan(p2, self.root, execute=True)
        self.assertEqual(r1["mode"], "executed")
        self.assertEqual(r2["mode"], "executed")
        self.assertTrue((self.root / "research" / "acme_dossier.md").exists())
        self.assertTrue((self.root / "research" / "globex_dossier.md").exists())

    def test_dry_run_takes_no_lock(self):
        plan_path = write_plan(self.root)
        with _held_lock(self.root):
            out1 = apply_plan(plan_path, self.root)
            out2 = apply_plan(plan_path, self.root)
        self.assertEqual(out1["mode"], "dry-run")
        self.assertEqual(out2["mode"], "dry-run")
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_lock_released_after_failed_preflight(self):
        plan_path = write_plan(self.root)
        (self.root / "WATCHLIST.md").write_text("mutated\n")
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "corpus_changed")
        _probe_lock_free(self.root)  # released despite the exception
        # And a subsequent apply goes through (the drift is ours).
        out = apply_plan(plan_path, self.root, execute=True, force=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["moved_count"], 4)

    def test_lock_released_when_execution_blows_up(self):
        plan_path = write_plan(self.root)
        boom = mock.patch(
            "slopkeeper.apply.shutil.move", side_effect=RuntimeError("disk gone")
        )
        with boom:
            with self.assertRaises(RuntimeError):
                apply_plan(plan_path, self.root, execute=True)
        _probe_lock_free(self.root)  # finally, not success-path-only
        # Nothing moved (the first move raised before moving), so the
        # same plan is still fresh and now succeeds.
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 4)

    def test_undo_execute_respects_lock_and_dry_run_does_not(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        journal = Path(out["journal"])
        with _held_lock(self.root):
            with self.assertRaises(SlopKeeperError) as cm:
                undo_journal(journal, self.root, execute=True)
            self.assertEqual(cm.exception.code, "tending_in_progress")
            # Still moved: the refused undo reversed nothing.
            self.assertFalse((self.root / "acme_dossier.md").exists())
            # Dry-run undo is a pure read — safe under contention.
            dry = undo_journal(journal, self.root, execute=False)
        self.assertEqual(dry["mode"], "dry-run")
        done = undo_journal(journal, self.root, execute=True)
        self.assertEqual(done["mode"], "executed")
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_status_record_respects_lock_and_plain_reads_take_none(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)  # manifest written
        with _held_lock(self.root):
            with self.assertRaises(SlopKeeperError) as cm:
                status(self.root, record=True)
            self.assertEqual(cm.exception.code, "tending_in_progress")
            # Plain status never locks — and the lock file is not slop.
            read = status(self.root, record=False)
        self.assertFalse(read["never_tended"])
        self.assertEqual(read["new_count"], 0)
        recorded = status(self.root, record=True)
        self.assertEqual(recorded["files_now"], 8)

    def test_scaffold_execute_respects_lock(self):
        plan_path = write_plan(self.root)
        with _held_lock(self.root):
            with self.assertRaises(SlopKeeperError) as cm:
                scaffold_from_plan(plan_path, self.root, execute=True)
            self.assertEqual(cm.exception.code, "tending_in_progress")
            self.assertFalse((self.root / "AGENTS.md").exists())
        out = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertTrue((self.root / "AGENTS.md").exists())

    def test_missing_fcntl_fails_soft_and_executes(self):
        # Windows path: no fcntl in stdlib → proceed WITHOUT the lock,
        # per §5.5 (zero-dep: no msvcrt fallback). sys.modules[None] is
        # the documented CPython way to make an import raise ImportError.
        plan_path = write_plan(self.root)
        sentry = sys.modules.get("fcntl")
        sys.modules["fcntl"] = None
        try:
            out = apply_plan(plan_path, self.root, execute=True)
        finally:
            if sentry is not None:
                sys.modules["fcntl"] = sentry
            else:
                del sys.modules["fcntl"]
        self.assertEqual(out["mode"], "executed")
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )

    def test_lock_file_remains_empty_after_apply(self):
        # The OS frees the lock on process death; the dotfile may stay.
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        lock_path = self.root / ".slopkeeper" / TEND_LOCK_NAME
        self.assertTrue(lock_path.exists())
        self.assertEqual(lock_path.read_bytes(), b"")

    def test_lock_file_does_not_move_the_fingerprint(self):
        fp1 = corpus_fingerprint(self.root)
        (self.root / ".slopkeeper").mkdir(exist_ok=True)
        (self.root / ".slopkeeper" / TEND_LOCK_NAME).touch()
        self.assertEqual(corpus_fingerprint(self.root), fp1)


if __name__ == "__main__":
    unittest.main()
