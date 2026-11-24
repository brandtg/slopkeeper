# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Journal integrity (P2, 0.2.7): one tend one journal, forgiving reads,
verified restores, and the write-time checks that make the journal worth
trusting.

The audit's framing: the journal is the LAST-RESORT recovery mechanism, so
"worked on the happy path" is not enough there. Every test drives a real
apply or a real undo against a temp corpus — A4's same-second collision,
B2's torn line, B1's mutated-then-undone file, B3's mid-run edit drift,
B4's destination race, B5's fingerprint forensics, B6's churn refusal,
B7's silent no-lock, and the two-process flock claim the single-process
lock tests could only approximate.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from helpers import TempCase, expect_code, write, write_edit_plan, write_plan

from slopkeeper import apply as apply_mod
from slopkeeper.apply import _default_journal, apply_plan, undo_journal
from slopkeeper.doctor import doctor
from slopkeeper.inventory import collect
from slopkeeper.models import corpus_fingerprint
from slopkeeper.scaffold import scaffold_from_plan
from slopkeeper.status import status

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

# The default corpus minus one file, as top-level keeps (the plans here
# move a single file; everything else is a recorded decision).
_DEFAULT_KEEPS = [
    "docs",
    "lead-gen",
    "spark-1.2.3",
    "WATCHLIST.md",
    "acme_dossier.md",
    "globex_dossier.md",
    "handoff-tp2.md",
    "ledger_wb.html",
]


def _journal_lines(journal: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _move_only_plan(
    root: Path, src: str, dst: str, name: str = "plan", keeps: list[str] | None = None
) -> Path:
    """A one-move plan with a fresh corpus binding and every other
    top-level entry kept (recomputed from disk, so a destination a
    previous apply created is covered too)."""
    tops = {p.name for p in root.iterdir() if not p.name.startswith(".")}
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
        "moves": [{"from": src, "to": dst, "reason": "p2 journal integrity"}],
        "keep": sorted((tops | set(keeps or []) | {dst.split("/")[0]}) - {src}),
        "unresolved": [],
        "folders": [],
    }
    p = root.parent / f"{name}.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    return p


class _FrozenSecond:
    """datetime stand-in whose now() never advances — A4's fixture: two
    tends genuinely inside one second, deterministically."""

    @staticmethod
    def now(tz=None):
        return datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)


# --- A4: one tend, one journal ---------------------------------------------


class TestJournalPerTend(TempCase):
    def test_two_applies_in_one_second_get_two_journals(self):
        p1 = _move_only_plan(self.root, "acme_dossier.md", "x/acme.md", "p1")
        with mock.patch.object(apply_mod, "datetime", _FrozenSecond):
            r1 = apply_plan(p1, self.root, execute=True)
            # p2's binding must describe the corpus as p1 left it.
            p2 = _move_only_plan(self.root, "globex_dossier.md", "y/globex.md", "p2")
            r2 = apply_plan(p2, self.root, execute=True)
        self.assertNotEqual(r1["journal"], r2["journal"])
        journals = sorted(Path(r1["journal"]).parent.glob("journal-*.jsonl"))
        self.assertEqual(
            [j.name for j in journals],
            ["journal-20261124T120000Z-2.jsonl", "journal-20261124T120000Z.jsonl"],
        )
        # The pre-fix bug: one file, and undoing it reverted BOTH tends.
        # Undo the first journal only — the second tend stays applied.
        undo_journal(Path(r1["journal"]), self.root, execute=True)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertTrue((self.root / "y" / "globex.md").exists())

    def test_same_second_names_take_the_next_counter(self):
        with mock.patch.object(apply_mod, "datetime", _FrozenSecond):
            first = _default_journal(self.root)
            first.parent.mkdir(parents=True, exist_ok=True)
            first.touch()
            second = _default_journal(self.root)
            second.touch()
            third = _default_journal(self.root)
        self.assertEqual(first.name, "journal-20261124T120000Z.jsonl")
        self.assertEqual(second.name, "journal-20261124T120000Z-2.jsonl")
        self.assertEqual(third.name, "journal-20261124T120000Z-3.jsonl")

    def test_tend_id_is_recorded_in_the_provenance_header(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        journal = Path(out["journal"])
        header = _journal_lines(journal)[0]
        self.assertEqual(header["op"], "plan")
        self.assertEqual(header["tend_id"], journal.stem)
        self.assertTrue(header["sha256"])
        # and undo names the tend it just reversed
        undone = undo_journal(journal, self.root, execute=False)
        self.assertEqual(undone["tend_id"], journal.stem)

    def test_explicit_journal_path_still_appends(self):
        # Naming a journal is an instruction to SHARE it (tests, scripted
        # batches) — the counter suffix is only for generated names.
        p1 = _move_only_plan(self.root, "acme_dossier.md", "x/acme.md", "p1")
        shared = self.root / ".slopkeeper" / "journal-batch.jsonl"
        r1 = apply_plan(p1, self.root, execute=True, journal_path=shared)
        p2 = _move_only_plan(self.root, "globex_dossier.md", "y/globex.md", "p2")
        r2 = apply_plan(p2, self.root, execute=True, journal_path=shared)
        self.assertEqual(r1["journal"], r2["journal"])
        heads = [e for e in _journal_lines(shared) if e.get("op") == "plan"]
        self.assertEqual(len(heads), 2)

    def test_journal_lines_are_whole_json_objects_after_a_clean_run(self):
        # fsync'd whole-line writes: every line parses, and the file ends
        # with a newline (no torn tail to tolerate on a clean run).
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        raw = Path(out["journal"]).read_text(encoding="utf-8")
        self.assertTrue(raw.endswith("\n"))
        for line in raw.splitlines():
            json.loads(line)  # raises on a partial line


# --- B2: a corrupt journal is still a journal ------------------------------


class TestJournalTornLines(TempCase):
    def _applied(self) -> Path:
        plan_path = write_plan(self.root)
        return Path(apply_plan(plan_path, self.root, execute=True)["journal"])

    def test_torn_last_line_still_undoes(self):
        journal = self._applied()
        lines = journal.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1][: len(lines[-1]) // 2]  # crash mid-write
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        undone = undo_journal(journal, self.root, execute=True)
        self.assertEqual(undone["mode"], "executed")
        self.assertEqual(undone["journal_corrupt_lines"], 1)
        self.assertNotIn("journal_corrupt_mid_lines", undone)
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_corrupt_middle_line_refuses_execute_without_force(self):
        journal = self._applied()
        lines = journal.read_text(encoding="utf-8").splitlines()
        lines.insert(2, '{"op": "move", "src": "acme_dossier.md", "dst"')
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        expect_code(
            self,
            lambda: undo_journal(journal, self.root, execute=True),
            "journal_corrupt",
        )
        # Refused atomically: nothing was restored.
        self.assertFalse((self.root / "acme_dossier.md").exists())
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )

    def test_corrupt_middle_line_undoes_what_it_can_with_force(self):
        journal = self._applied()
        lines = journal.read_text(encoding="utf-8").splitlines()
        lost = json.loads(lines[2])  # one journaled move stays applied
        lines[2] = "this is not json at all"
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        undone = undo_journal(journal, self.root, execute=True, force=True)
        self.assertEqual(undone["journal_corrupt_lines"], 1)
        self.assertEqual(undone["journal_corrupt_mid_lines"], 1)
        restored = {o["src"] for o in undone["operations"] if o["status"] == "restored"}
        self.assertNotIn(lost["src"], restored)
        self.assertTrue(restored)  # the rest of the tend still came back

    def test_dry_run_reports_corruption_and_says_execute_would_refuse(self):
        journal = self._applied()
        lines = journal.read_text(encoding="utf-8").splitlines()
        lines.insert(2, "{ broken")
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dry = undo_journal(journal, self.root, execute=False)
        self.assertEqual(dry["mode"], "dry-run")
        self.assertEqual(dry["journal_corrupt_lines"], 1)
        self.assertEqual(dry["journal_corrupt_mid_lines"], 1)
        self.assertIn("journal_corrupt_needs_force", dry["warnings"])
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )

    def test_undecodable_bytes_do_not_kill_the_undo(self):
        # read_text() strict used to make one stray non-UTF-8 byte anywhere
        # unreadable — worse than a malformed line, because nothing was
        # reported and nothing could be undone.
        journal = self._applied()
        with journal.open("ab") as jf:
            jf.write(b"\xff\xfe not utf8 at all\n")
        undone = undo_journal(journal, self.root, execute=True)
        self.assertEqual(undone["journal_corrupt_lines"], 1)
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_non_object_json_line_is_tolerated(self):
        journal = self._applied()
        with journal.open("a", encoding="utf-8") as jf:
            jf.write("[1, 2, 3]\n")
        undone = undo_journal(journal, self.root, execute=True)
        self.assertEqual(undone["journal_corrupt_lines"], 1)

    def test_everything_corrupt_is_journal_empty(self):
        journal = self._applied()
        journal.write_text("nope\nnope either\n", encoding="utf-8")
        expect_code(self, lambda: undo_journal(journal, self.root), "journal_empty")


# --- B1: undo verifies the journaled hash ----------------------------------


class TestUndoVerifiesHashes(TempCase):
    def test_mutated_file_restores_but_reports_content_changed(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        mutated = "edited by someone else after the tend\n"
        (self.root / "research" / "dossiers" / "acme_dossier.md").write_text(
            mutated, encoding="utf-8"
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        row = next(o for o in undone["operations"] if o.get("src") == "acme_dossier.md")
        self.assertEqual(row["status"], "restored")  # restored anyway
        self.assertTrue(row["content_changed"])
        self.assertEqual(undone["content_changed_count"], 1)
        self.assertIn("content_changed", undone["warnings"])
        # The human's file comes back — mutated content and all (deleting
        # it would be the worse lie).
        self.assertEqual((self.root / "acme_dossier.md").read_text(), mutated)
        journaled = next(
            e for e in _journal_lines(Path(out["journal"])) if e.get("op") == "move"
        )
        self.assertEqual(row["sha256_journaled"], journaled["sha256"])
        self.assertEqual(
            row["sha256_now"], hashlib.sha256(mutated.encode()).hexdigest()
        )

    def test_dry_run_undo_reports_would_change_before_anything_moves(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "research" / "dossiers" / "globex_dossier.md").write_text(
            "drift\n", encoding="utf-8"
        )
        dry = undo_journal(Path(out["journal"]), self.root, execute=False)
        row = next(o for o in dry["operations"] if o.get("src") == "globex_dossier.md")
        self.assertEqual(row["status"], "would_restore")
        self.assertTrue(row["content_changed"])
        self.assertFalse((self.root / "globex_dossier.md").exists())  # pure read

    def test_untouched_files_carry_no_flag(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        dry = undo_journal(Path(out["journal"]), self.root, execute=False)
        self.assertNotIn("content_changed_count", dry)
        self.assertNotIn("warnings", dry)
        self.assertTrue(all("content_changed" not in o for o in dry["operations"]))

    def test_directory_move_without_a_hash_is_never_flagged(self):
        write(self.root / "stuff/a.md", "# a\n")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "stuff", "to": "archive/stuff", "reason": "dir"}],
            keep=_DEFAULT_KEEPS + ["archive"],
            folders=[],
        )
        out = apply_plan(plan_path, self.root, execute=True)
        moves = [
            e for e in _journal_lines(Path(out["journal"])) if e.get("op") == "move"
        ]
        self.assertTrue(moves)
        self.assertTrue(all(e["sha256"] is None for e in moves))  # dir children
        dry = undo_journal(Path(out["journal"]), self.root, execute=False)
        self.assertTrue(all("content_changed" not in o for o in dry["operations"]))


# --- B3: an edit is re-verified at write time -------------------------------


class TestEditWriteTimeVerification(TempCase):
    def _writer_hiding_behind_the_moves(self, rel: str, new_text: str):
        """Wrap the move seam so an external writer hits `rel` mid-run."""
        real = apply_mod._move_no_clobber
        state = {"done": False}

        def moving_then_writing(src, dst):
            out = real(src, dst)
            if not state["done"]:
                state["done"] = True
                (self.root / rel).write_text(new_text, encoding="utf-8")
            return out

        return mock.patch.object(
            apply_mod, "_move_no_clobber", side_effect=moving_then_writing
        )

    def test_mid_run_drift_records_edit_stale_instead_of_wrong_replace(self):
        plan_path = write_edit_plan(self.root)
        with self._writer_hiding_behind_the_moves(
            "WATCHLIST.md", "nothing of the plan here anymore\n"
        ):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(out["partial"])
        stale = [
            e for e in out["errors"] if e.get("error", "").startswith("edit_stale")
        ]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["file"], "WATCHLIST.md")
        self.assertIn("expected exactly 1", stale[0]["error"])
        # Nothing was written: the drifted text stands as the writer left it.
        self.assertEqual(
            (self.root / "WATCHLIST.md").read_text(),
            "nothing of the plan here anymore\n",
        )
        self.assertEqual(out["moved_count"], 1)  # the move itself still happened
        # And the journal holds no edit op — undo has nothing to reverse.
        ops = [e.get("op") for e in _journal_lines(Path(out["journal"]))]
        self.assertNotIn("edit", ops)

    def test_ambiguous_find_at_write_time_is_refused_not_guessed(self):
        # find occurred exactly once at validate; by write time an
        # external writer had duplicated it. A blind replace(..., 1) would
        # edit the wrong line and journal the edit as applied.
        plan_path = write_edit_plan(self.root)
        drift = "acme_dossier.md and again acme_dossier.md\n"
        with self._writer_hiding_behind_the_moves("WATCHLIST.md", drift):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(
            any("occurred 2 time(s)" in e.get("error", "") for e in out["errors"])
        )
        self.assertEqual((self.root / "WATCHLIST.md").read_text(), drift)

    def test_clean_edit_still_applies_and_journals(self):
        plan_path = write_edit_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        self.assertIn(
            "research/acme_dossier.md", (self.root / "WATCHLIST.md").read_text()
        )
        edits = [
            e for e in _journal_lines(Path(out["journal"])) if e.get("op") == "edit"
        ]
        self.assertEqual(len(edits), 1)
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertIn("reverted", [o.get("status") for o in undone["operations"]])


# --- B4: the move is atomic and never replaces -----------------------------


class TestAtomicNoClobberMove(TempCase):
    def test_destination_created_mid_run_is_refused_not_clobbered(self):
        plan_path = write_plan(self.root)
        real = apply_mod._move_no_clobber
        state = {"raced": False}

        def racing_move(src, dst):
            # The audit's race: someone creates the destination after the
            # pre-flight and before this write.
            if not state["raced"]:
                state["raced"] = True
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("someone else's file\n", encoding="utf-8")
            return real(src, dst)

        with mock.patch.object(apply_mod, "_move_no_clobber", side_effect=racing_move):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(out["partial"])
        raced = [
            e for e in out["errors"] if e.get("error", "").startswith("dst_exists")
        ]
        self.assertEqual(len(raced), 1)
        # The owner's bytes survive — that is the whole point of B4.
        self.assertEqual(
            (self.root / raced[0]["dst"]).read_text(), "someone else's file\n"
        )
        self.assertTrue((self.root / raced[0]["src"]).exists())
        self.assertNotIn(raced[0]["dst"], {m["to"] for m in out["moved"]})
        # And the refused hop is not journaled, so undo cannot resurrect a
        # file that never left.
        journaled = {
            e["dst"]
            for e in _journal_lines(Path(out["journal"]))
            if e.get("op") == "move"
        }
        self.assertNotIn(raced[0]["dst"], journaled)

    def test_a_thread_creating_the_destination_mid_run_is_refused(self):
        # The audit's acceptance form for B4, with a real second thread:
        # after the pre-flight and before the write, another thread of
        # this process puts a file at the destination. Outcome is a
        # refusal, not a clobber.
        plan_path = write_plan(self.root)
        first_dst = self.root / "research" / "dossiers" / "acme_dossier.md"
        real_preflight = apply_mod.preflight_destinations

        def preflight_then_race(root, chains):
            out = real_preflight(root, chains)
            first_dst.parent.mkdir(parents=True, exist_ok=True)
            t = threading.Thread(
                target=lambda: first_dst.write_text("thread's file\n", encoding="utf-8")
            )
            t.start()
            t.join()  # deterministic: the race is won before the loop runs
            return out

        with mock.patch.object(
            apply_mod, "preflight_destinations", side_effect=preflight_then_race
        ):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(out["partial"])
        self.assertEqual(first_dst.read_text(), "thread's file\n")
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_undo_restore_refuses_to_clobber_a_reappeared_source(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        (self.root / "acme_dossier.md").write_text(
            "hand-made again\n", encoding="utf-8"
        )
        undone = undo_journal(Path(out["journal"]), self.root, execute=True)
        row = next(o for o in undone["operations"] if o.get("src") == "acme_dossier.md")
        self.assertEqual(row["status"], "blocked_dst_occupied")
        self.assertEqual(
            (self.root / "acme_dossier.md").read_text(), "hand-made again\n"
        )

    def test_move_preserves_bytes_and_leaves_no_source(self):
        plan_path = _move_only_plan(self.root, "acme_dossier.md", "x/acme.md")
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["moved_count"], 1)
        self.assertEqual(
            (self.root / "x" / "acme.md").read_bytes(), b"# Acme dossier\n\nflows.\n"
        )
        self.assertFalse((self.root / "acme_dossier.md").exists())
        # link+unlink keeps the content hash identical to the journaled one
        journaled = next(
            e
            for e in _journal_lines(Path(out["journal"]))
            if e.get("op") == "move" and e["src"] == "acme_dossier.md"
        )
        self.assertEqual(
            journaled["sha256"],
            hashlib.sha256(b"# Acme dossier\n\nflows.\n").hexdigest(),
        )

    def test_hardlink_unsupported_falls_back_to_shutil_move(self):
        # A filesystem that refuses hardlinks (EXDEV and friends) must
        # still move — with the last-moment existence re-check in force.
        plan_path = write_plan(self.root)
        err = OSError(errno.EXDEV, "Cross-device link")

        def cross_device_link(src, dst, follow_symlinks=True):
            raise err

        with mock.patch.object(apply_mod.os, "link", side_effect=cross_device_link):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(out["partial"])
        self.assertEqual(out["moved_count"], 4)
        self.assertTrue(
            (self.root / "research" / "dossiers" / "acme_dossier.md").exists()
        )

    def test_hardlink_unsupported_and_raced_destination_is_refused(self):
        # The fallback path keeps the no-clobber guarantee too: its guard
        # is the lexists() re-check, since rename() across devices cannot
        # be atomic-and-no-replace.
        plan_path = write_plan(self.root)
        err = OSError(errno.EPERM, "Operation not permitted")
        real = apply_mod._move_no_clobber
        state = {"raced": False}

        def link_fails(src, dst, follow_symlinks=True):
            raise err

        def racing_move(src, dst):
            if not state["raced"]:
                state["raced"] = True
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("mine\n", encoding="utf-8")
            return real(src, dst)

        with mock.patch.object(apply_mod.os, "link", side_effect=link_fails):
            with mock.patch.object(
                apply_mod, "_move_no_clobber", side_effect=racing_move
            ):
                out = apply_plan(plan_path, self.root, execute=True)
        raced = [
            e for e in out["errors"] if e.get("error", "").startswith("dst_exists")
        ]
        self.assertEqual(len(raced), 1)
        self.assertEqual((self.root / raced[0]["dst"]).read_text(), "mine\n")

    def test_directory_move_refuses_a_destination_that_appeared_mid_run(self):
        # A directory rename has no no-replace primitive; the belt is the
        # last-moment lexists() check, so a destination created after the
        # pre-flight is refused rather than replaced-or-nested. Injected
        # one line before the guard (sha256_file is the call that runs
        # immediately ahead of it).
        write(self.root / "stuff/a.md", "# a\n")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "stuff", "to": "archive/stuff", "reason": "dir"}],
            keep=_DEFAULT_KEEPS + ["archive"],
            folders=[],
        )
        real_hash = apply_mod.sha256_file
        state = {"raced": False}

        dst_dir = self.root / "archive" / "stuff"

        def racing_hash(p):
            # sha256_file(src) is the call that runs immediately before
            # the directory guard — the race lands exactly in that window.
            if not state["raced"] and Path(p) == self.root / "stuff":
                state["raced"] = True
                dst_dir.mkdir(parents=True, exist_ok=True)
                (dst_dir / "note.md").write_text("mine\n", encoding="utf-8")
            return real_hash(p)

        with mock.patch.object(apply_mod, "sha256_file", side_effect=racing_hash):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue(
            any(e.get("error", "").startswith("dst_exists") for e in out["errors"]),
            out["errors"],
        )
        self.assertTrue((self.root / "stuff" / "a.md").exists())
        self.assertEqual(
            (self.root / "archive" / "stuff" / "note.md").read_text(), "mine\n"
        )


# --- B5: the journal says which disk ran -----------------------------------


class TestFingerprintForensics(TempCase):
    def test_header_binds_plan_gate_and_execute_fingerprints(self):
        plan_path = write_plan(self.root)
        fp = corpus_fingerprint(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        header = _journal_lines(Path(out["journal"]))[0]
        self.assertEqual(header["plan_fingerprint"], fp)
        self.assertEqual(header["gate_fingerprint"], out["fingerprint"])
        self.assertEqual(header["execute_fingerprint"], out["fingerprint_execute"])
        self.assertEqual(out["fingerprint_execute"], out["fingerprint"])
        self.assertNotIn("warnings", out)  # a clean run says nothing extra

    def test_corpus_that_moves_after_the_gate_is_warned_and_recorded(self):
        plan_path = write_plan(self.root)
        gate = corpus_fingerprint(self.root)
        mutated = {"did": False}
        real_preflight = apply_mod._unwritable_targets

        def preflight_then_mutate(root, chains, rewrites, edits):
            # B5's window: after the gate's digest, before the first write,
            # something outside this process writes to the corpus. (A NEW
            # file, so the already-computed rewrite pass is untouched.)
            if not mutated["did"]:
                mutated["did"] = True
                (root / "late_note.md").write_text(
                    "appeared mid-run\n", encoding="utf-8"
                )
            return real_preflight(root, chains, rewrites, edits)

        with mock.patch.object(
            apply_mod, "_unwritable_targets", side_effect=preflight_then_mutate
        ):
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertIn("fingerprint_moved_during_apply", out["warnings"])
        header = _journal_lines(Path(out["journal"]))[0]
        self.assertEqual(header["plan_fingerprint"], gate)
        self.assertEqual(header["gate_fingerprint"], out["fingerprint"])
        self.assertNotEqual(header["gate_fingerprint"], header["execute_fingerprint"])
        # The execute digest is the corpus AS WRITTEN FROM — the new file
        # is in it, and it is the value the envelope reports.
        self.assertEqual(out["fingerprint_execute"], header["execute_fingerprint"])
        self.assertNotEqual(header["execute_fingerprint"], gate)


# --- B6: unstatable files are counted, not swallowed -----------------------


class TestUnstatableLoudness(TempCase):
    def _denied_stat(self, target: Path):
        """The churn race: the walk lists the file (is_file() True), then
        stat() fails — what a file vanishing mid-walk looks like."""
        real_is_file = Path.is_file
        real_stat = Path.stat

        def is_file(self, *args, **kwargs):
            if Path(self) == target:
                return True
            return real_is_file(self, *args, **kwargs)

        def stat(self, *args, **kwargs):
            if Path(self) == target:
                raise OSError(2, "No such file or directory")
            return real_stat(self, *args, **kwargs)

        return (
            mock.patch.object(Path, "is_file", is_file),
            mock.patch.object(Path, "stat", stat),
        )

    def _plan_around_churn(self):
        churn = self.root / "churny.md"
        write(churn, "here then gone\n")
        plan_path = _move_only_plan(
            self.root,
            "acme_dossier.md",
            "x/acme.md",
            name="churn",
            keeps=_DEFAULT_KEEPS + ["churny.md", "x"],
        )
        return churn, plan_path

    def test_apply_refuses_a_corpus_churning_under_the_walk(self):
        churn, plan_path = self._plan_around_churn()
        p1, p2 = self._denied_stat(churn)
        with p1, p2:
            expect_code(
                self,
                lambda: apply_plan(plan_path, self.root, execute=True),
                "corpus_unstable",
            )
        self.assertTrue((self.root / "acme_dossier.md").exists())  # refused pre-write

    def test_force_proceeds_and_carries_the_count(self):
        churn, plan_path = self._plan_around_churn()
        p1, p2 = self._denied_stat(churn)
        with p1, p2:
            out = apply_plan(plan_path, self.root, execute=True, force=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["skipped_unstatable"], 1)
        self.assertIn("skipped_unstatable", out["warnings"])
        self.assertEqual(out["moved_count"], 1)

    def test_inventory_names_the_count_and_the_paths(self):
        churn, _plan = self._plan_around_churn()
        p1, p2 = self._denied_stat(churn)
        with p1, p2:
            inv = collect(self.root, fast=True)
        self.assertEqual(inv["stats"]["skipped_unstatable"], 1)
        self.assertEqual(inv["stats"]["skipped_unstatable_paths"], ["churny.md"])

    def test_clean_walk_reports_zero_and_no_paths_key(self):
        inv = collect(self.root, fast=True)
        self.assertEqual(inv["stats"]["skipped_unstatable"], 0)
        self.assertNotIn("skipped_unstatable_paths", inv["stats"])

    def test_doctor_advises_on_the_churn(self):
        churn, _plan = self._plan_around_churn()
        p1, p2 = self._denied_stat(churn)
        with p1, p2:
            d = doctor(self.root)
        self.assertEqual(d["stats"]["skipped_unstatable"], 1)
        self.assertTrue(any("denied stat" in a for a in d["advice"]), d["advice"])
        self.assertTrue(any("churny.md" in a for a in d["advice"]))


# --- B7: no flock is not a silent no-op ------------------------------------


class TestLockingHonestyOnEnvelopes(TempCase):
    def _without_fcntl(self):
        return mock.patch.dict(sys.modules, {"fcntl": None})

    def test_apply_execute_warns_and_dry_run_does_not(self):
        plan_path = write_plan(self.root)
        with self._without_fcntl():
            dry = apply_plan(plan_path, self.root, execute=False)
            out = apply_plan(plan_path, self.root, execute=True)
        self.assertIn("no_locking_available", out["warnings"])
        self.assertNotIn("warnings", dry)

    def test_undo_scaffold_and_status_record_warn_too(self):
        plan_path = write_plan(self.root)
        journal = Path(apply_plan(plan_path, self.root, execute=True)["journal"])
        with self._without_fcntl():
            undone = undo_journal(journal, self.root, execute=True)
            scaf = scaffold_from_plan(plan_path, self.root, execute=True)
            rec = status(self.root, record=True)
            plain = status(self.root, record=False)
        self.assertIn("no_locking_available", undone["warnings"])
        self.assertIn("no_locking_available", scaf["warnings"])
        self.assertIn("no_locking_available", rec["warnings"])
        self.assertNotIn("warnings", plain)

    def test_warnings_are_deduped_and_order_stable(self):
        # Two soft conditions on one envelope: no lock AND a moved corpus.
        plan_path = write_plan(self.root)
        mutated = {"did": False}
        real_preflight = apply_mod._unwritable_targets

        def preflight_then_mutate(root, chains, rewrites, edits):
            if not mutated["did"]:
                mutated["did"] = True
                (root / "late_note.md").write_text("appeared\n", encoding="utf-8")
            return real_preflight(root, chains, rewrites, edits)

        with self._without_fcntl():
            with mock.patch.object(
                apply_mod, "_unwritable_targets", side_effect=preflight_then_mutate
            ):
                out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(
            out["warnings"], ["fingerprint_moved_during_apply", "no_locking_available"]
        )


# --- two-process race (the flock claim, tested across processes) -----------


class TestTwoProcessApplyRace(TempCase):
    """flock only means something across processes. The single-process
    lock tests cover the error path; these cover the serialization claim
    the guide makes."""

    def _run_apply(self, plan_path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "slopkeeper",
                "apply",
                str(plan_path),
                str(self.root),
                "--execute",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=CLI_ENV,
            timeout=60,
        )

    def test_second_process_refuses_while_the_first_holds_the_lock(self):
        plan_path = write_plan(self.root)
        lock_path = self.root / ".slopkeeper" / "tend.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            proc = self._run_apply(plan_path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        payload = json.loads(proc.stdout)
        # EXIT_ERROR: the exit-2 family is the plan_*/validation refusals;
        # a contended lock is a runtime condition, documented as exit 1.
        self.assertEqual(proc.returncode, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "tending_in_progress")
        self.assertTrue(payload["error"]["recovery"])
        self.assertTrue((self.root / "acme_dossier.md").exists())  # moved nothing

    def test_the_loser_undoes_nothing_and_the_winner_journal_is_untouched(self):
        plan_path = write_plan(self.root)
        winner = apply_plan(plan_path, self.root, execute=True)
        journals_before = sorted(
            p.name for p in (self.root / ".slopkeeper").glob("journal-*.jsonl")
        )
        lock_path = self.root / ".slopkeeper" / "tend.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            proc = self._run_apply(plan_path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        self.assertEqual(
            json.loads(proc.stdout)["error"]["code"], "tending_in_progress"
        )
        self.assertEqual(
            sorted(p.name for p in (self.root / ".slopkeeper").glob("journal-*.jsonl")),
            journals_before,  # no journal line was written by the refused run
        )
        self.assertTrue(Path(winner["journal"]).exists())

    def test_a_free_lock_lets_the_subprocess_tend_and_journal(self):
        # Sanity for the two above: the same command succeeds with no
        # holder, and lands its own journal.
        plan_path = _move_only_plan(self.root, "acme_dossier.md", "x/acme.md", "solo")
        proc = self._run_apply(plan_path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["data"]["mode"], "executed")
        self.assertTrue((self.root / "x" / "acme.md").exists())
        journals = list((self.root / ".slopkeeper").glob("journal-*.jsonl"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(_journal_lines(journals[0])[0]["tend_id"], journals[0].stem)


if __name__ == "__main__":
    unittest.main()
