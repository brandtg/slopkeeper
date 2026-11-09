# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""strategies.py — the strategy packs."""

from __future__ import annotations

import unittest

from slopkeeper.strategies import get_strategy, list_strategies


class TestStrategies(unittest.TestCase):
    def test_all_packages_load(self):
        ids = [s["id"] for s in list_strategies()]
        # by-kind (Phase 10): the mechanical baseline propose compares
        # against — a real rubric pack, not a propose-only special case.
        self.assertEqual(
            set(ids), {"role-bots", "topics", "projects", "hybrid", "by-kind"}
        )
        for sid in ids:
            pack = get_strategy(sid)
            self.assertIn("partitioning_rubric", pack)
            self.assertIn("agents_md_skeleton", pack["folder_template"])


if __name__ == "__main__":
    unittest.main()
