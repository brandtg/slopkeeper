# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Scale smoke: the single-pass scanner has to STAY single-pass (P3/R6).

`uv run pytest -q tests/test_scan.py` proves the traversal count on a
ten-file fixture, which is the shape a refactor breaks silently: code
that re-walks for one more survey looks exactly like code that does not
until a real corpus makes the command crawl. This file runs the same
counting assertion at 5k files and holds the wall-clock floor the audit
set — sub-3s inventory, sub-5s doctor — with room for a CI box that is
slower than the one that measured it.

Skipped by default (generating the corpus costs tens of seconds) and
wired into a weekly cron instead of the per-commit gate:

    SLOPKEEPER_RUN_SLOW=1 uv run pytest -q tests/test_scale_smoke.py

Baselines and the rest of the numbers live in bench_reference_scan.py.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from bench_reference_scan import build_scale_tree

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("SLOPKEEPER_RUN_SLOW"),
        reason="scale smoke: set SLOPKEEPER_RUN_SLOW=1 (weekly CI)",
    ),
]

# Generous multiples of the 0.2.8 measurements on the dev box (0.10s
# inventory, 0.24s doctor at 5k), because the point is catching a
# regression back toward the seven-walk shape, not micro-tuning.
INVENTORY_BUDGET_S = 3.0
DOCTOR_BUDGET_S = 5.0
REFS_BUDGET_S = 3.0
WALKS_PER_COMMAND = 1


class TestScaleSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sk-scale-smoke-"))
        build_scale_tree(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _timed(self, fn):
        t0 = time.perf_counter()
        out = fn()
        return time.perf_counter() - t0, out

    def _counted(self, fn):
        """Run `fn` with os.walk counted (the traversal IS the contract)."""
        import os as _os

        real = _os.walk
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        _os.walk = counting
        try:
            elapsed, out = self._timed(fn)
        finally:
            _os.walk = real
        return elapsed, out, len(calls)

    def test_inventory_at_5k_is_fast_and_walks_once(self):
        from slopkeeper.inventory import collect

        elapsed, inv, walks = self._counted(lambda: collect(self.tmp, fast=True))
        self.assertEqual(inv["stats"]["files"], 5001)
        self.assertEqual(walks, WALKS_PER_COMMAND, "inventory re-walked the corpus")
        self.assertLess(elapsed, INVENTORY_BUDGET_S, f"inventory took {elapsed:.2f}s")

    def test_doctor_at_5k_is_fast_and_walks_once(self):
        from slopkeeper.doctor import doctor

        elapsed, data, walks = self._counted(lambda: doctor(self.tmp))
        self.assertEqual(data["stats"]["files"], 5001)
        self.assertEqual(walks, WALKS_PER_COMMAND, "doctor re-walked the corpus")
        self.assertLess(elapsed, DOCTOR_BUDGET_S, f"doctor took {elapsed:.2f}s")

    def test_refs_at_5k_with_distinct_names_is_fast_and_walks_once(self):
        # The prefilter's case: 5,001 candidate basenames, one
        # alternation that used to cost 2.2s per scan.
        from slopkeeper.references import scan_references_stream

        elapsed, out, walks = self._counted(lambda: scan_references_stream(self.tmp))
        self.assertEqual(out["stats"]["files_scanned"], 5001)
        self.assertGreater(out["stats"]["hits"], 5000)
        self.assertEqual(walks, WALKS_PER_COMMAND, "the refs scan re-walked the corpus")
        self.assertLess(elapsed, REFS_BUDGET_S, f"refs scan took {elapsed:.2f}s")

    def test_inventory_fingerprint_is_a_fold_not_a_second_walk(self):
        # The honest number (B5): the digest the document reports is the
        # digest of the entries it lists, so a corpus that moved under
        # the walk cannot be half-described.
        from slopkeeper.inventory import collect
        from slopkeeper.models import corpus_fingerprint_detail
        from slopkeeper.scan import prepare

        scan = prepare(self.tmp)
        inv = collect(self.tmp, fast=True, scan=scan)
        self.assertEqual(inv["fingerprint"], scan.fingerprint())
        late = self.tmp / "late_note.md"
        try:
            late.write_text("# late\n", encoding="utf-8")
            # The document still describes the walk it took, and the walk
            # it took did not contain this file.
            self.assertEqual(inv["fingerprint"], scan.fingerprint())
            self.assertNotEqual(
                corpus_fingerprint_detail(self.tmp)[0], inv["fingerprint"]
            )
        finally:
            late.unlink()


if __name__ == "__main__":
    unittest.main()
