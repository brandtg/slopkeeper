# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""propose.py — candidate layouts with breakage costs (Phase 10).

The counts are the contract: move_count, citation_break_count,
code_anchor_count, unresolved_estimate — computed over one read-only
scan, with the citation-break rule identical to apply's
reference_risks so the fork data and the gate can never disagree.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper.apply import apply_plan
from slopkeeper.models import corpus_fingerprint
from slopkeeper.propose import build_candidates, candidate_moves


def _cand(out: dict, cand_id: str) -> dict:
    return next(c for c in out["candidates"] if c["id"] == cand_id)


class TestProposeCandidates(TempCase):
    def test_shape_three_candidates_all_counts(self):
        out = build_candidates(self.root)
        self.assertEqual(len(out["candidates"]), 3)
        self.assertEqual(
            {c["id"] for c in out["candidates"]},
            {"by-project", "by-kind", "hybrid"},
        )
        for c in out["candidates"]:
            for key in (
                "move_count",
                "citation_break_count",
                "code_anchor_count",
                "unresolved_estimate",
            ):
                self.assertIn(key, c)
            self.assertTrue(c["strategy"])
            self.assertTrue(c["strategy_name"])

    def test_by_project_keeps_project_subtrees_whole(self):
        # A README marks a project: its whole subtree stays put under
        # by-project (and hybrid) — never split. Everything without a
        # project signal is unresolved, not guessed.
        write(self.root / "proj/README.md", "# proj\n")
        write(self.root / "proj/src/a.py", "x = 1\n")
        write(self.root / "proj/notes.md", "n\n")
        out = build_candidates(self.root)
        by_project = _cand(out, "by-project")
        # spark-glm-5.3/ and lead-gen/ carry READMEs — projects, so
        # their READMEs stay; the default corpus's other 6 files have
        # no project signal and go unresolved
        self.assertEqual(by_project["move_count"], 0)
        self.assertEqual(by_project["unresolved_estimate"], 6)
        # by-kind flattens even the project: all 11 visible files move
        by_kind = _cand(out, "by-kind")
        self.assertEqual(by_kind["move_count"], 10)  # docs/canonical.md stays
        self.assertEqual(by_kind["unresolved_estimate"], 0)
        # hybrid: projects whole, leftovers kind-sorted
        hybrid = _cand(out, "hybrid")
        self.assertEqual(hybrid["move_count"], 5)
        self.assertEqual(hybrid["unresolved_estimate"], 0)

    def test_nested_project_dir_counts(self):
        # Project-ness is detected at any depth: a pyproject.toml deep
        # in a junk dir keeps ITS subtree, not the junk dir's.
        write(self.root / "misc/buried/pyproject.toml", "[tool]\n")
        write(self.root / "misc/buried/x.py", "x = 1\n")
        write(self.root / "misc/other.txt", "o\n")
        moves = candidate_moves(self.root, "by-project")
        self.assertEqual(moves, {})  # buried project stays; other.txt unresolved
        out = build_candidates(self.root)
        self.assertEqual(_cand(out, "by-project")["unresolved_estimate"], 7)

    def test_already_in_shape_moves_nothing(self):
        # A corpus already shaped like a candidate must read move_count
        # 0 under it — the count is honest, not aspirational.
        write(self.root / "docs/extra.md", "e\n")
        write(self.root / "data/x.json", "{}\n")
        write(self.root / "scripts/s.sh", "#!/bin/sh\n")
        out = build_candidates(self.root)
        # docs/, data/, scripts/ files keep their kind dir...
        moved = candidate_moves(self.root, "by-kind")
        self.assertNotIn("docs/extra.md", moved)
        self.assertNotIn("data/x.json", moved)
        self.assertNotIn("scripts/s.sh", moved)
        # ...while the default corpus's strays still move
        self.assertIn("acme_dossier.md", moved)
        by_kind = _cand(out, "by-kind")
        self.assertEqual(by_kind["move_count"], 7)  # + WATCHLIST, ledger, etc.


class TestProposeCounts(TempCase):
    def test_citation_break_counts_stay_put_citers(self):
        # B2 semantics: a stay-put citation the rewrite pass CAN fix is
        # churn (citation_rewrites), not a break. A break is a citation
        # the pass would NOT fix — here, a citing file that is not
        # valid UTF-8 (the pass never re-encodes a whole file).
        write(self.root / "proj/README.md", "# proj\n")
        write(self.root / "proj/NOTES.md", "see acme_dossier.md\n")
        (self.root / "proj/BROKE.md").write_bytes(b"see acme_dossier.md \xff\n")
        out = build_candidates(self.root)
        hybrid = _cand(out, "hybrid")
        # proj/NOTES.md's citation is rewritten by the pass — no break
        self.assertNotEqual(hybrid["citation_rewrites"], 0)
        # proj/BROKE.md's citation cannot be safely rewritten — a break
        self.assertEqual(hybrid["citation_break_count"], 1)
        # by-project moves nothing → nothing breaks
        self.assertEqual(_cand(out, "by-project")["citation_break_count"], 0)

    def test_code_anchor_counts_disturbed_dirs(self):
        write(self.root / "tools/run.py", 'D = os.path.join(ROOT, "slack", "data")\n')
        write(self.root / "slack/data/db.md", "x\n")
        write(self.root / "slack/keep.txt", "x\n")
        out = build_candidates(self.root)
        # by-project: slack never moves → no anchor cost
        self.assertEqual(_cand(out, "by-project")["code_anchor_count"], 0)
        # by-kind/hybrid: both slack files leave → both anchored dirs disturbed
        for cid in ("by-kind", "hybrid"):
            self.assertEqual(_cand(out, cid)["code_anchor_count"], 2)

    def test_counts_include_external_citers(self):
        # A skills store outside the corpus cites acme: without the
        # declaration the count cannot see it; with it, the break
        # appears. A count that ignores skills answers the fork wrong.
        ext = self.tmp / "ext"
        write(ext / "skill.md", "see acme_dossier.md\n")
        out = build_candidates(self.root)
        self.assertEqual(_cand(out, "hybrid")["citation_break_count"], 0)
        out = build_candidates(self.root, [ext])
        self.assertEqual(_cand(out, "hybrid")["citation_break_count"], 1)

    def test_counts_include_ignored_scan_dir(self):
        # An ignored subtree inside the corpus is invisible to the
        # walk but scannable when named — the plan-time override.
        write(self.root / ".slopkeeperignore", "cache/\n")
        write(self.root / "cache/notes.md", "acme_dossier.md lives here\n")
        out = build_candidates(self.root)
        self.assertEqual(_cand(out, "hybrid")["citation_break_count"], 0)
        out = build_candidates(self.root, [self.root / "cache"])
        self.assertEqual(_cand(out, "hybrid")["citation_break_count"], 1)

    def test_propose_is_read_only(self):
        import os

        def mtimes():
            return {
                str(p): os.stat(p).st_mtime for p in self.root.rglob("*") if p.is_file()
            }

        before_fp = corpus_fingerprint(self.root)
        before_mt = mtimes()
        build_candidates(self.root)
        self.assertEqual(corpus_fingerprint(self.root), before_fp)
        self.assertEqual(mtimes(), before_mt)
        self.assertFalse((self.root / ".slopkeeper").exists())


class TestProposeConsistency(TempCase):
    def test_break_count_equals_apply_risk_count(self):
        # The binding rule: for a fixed candidate, citation_break_count
        # equals the apply-time reference_risks count of the equivalent
        # plan — one uncovered-citation rule, two surfaces, never
        # disagreeing. A citation the pass fixes is churn on both
        # surfaces, never a risk.
        write(self.root / "proj/README.md", "# proj\n")
        write(self.root / "proj/NOTES.md", "see acme_dossier.md\n")
        out = build_candidates(self.root)
        hybrid = _cand(out, "hybrid")
        moved = candidate_moves(self.root, "hybrid")
        self.assertEqual(len(moved), hybrid["move_count"])
        plan_moves = [
            {"from": p, "to": f"{d}/{Path(p).name}", "reason": "t"}
            for p, d in sorted(moved.items())
        ]
        plan_path = write_plan(self.root, partial=True, moves=plan_moves)
        result = apply_plan(plan_path, self.root)  # dry-run
        self.assertEqual(len(result["reference_risks"]), hybrid["citation_break_count"])
        # the acme citation is rewritten by the pass — churn, not a risk
        self.assertEqual(result["reference_risks"], {})
        self.assertGreaterEqual(hybrid["citation_rewrites"], 1)
        self.assertIn("proj/NOTES.md", [r["file"] for r in result["would_rewrite"]])


if __name__ == "__main__":
    unittest.main()
