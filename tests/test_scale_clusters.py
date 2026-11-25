# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P4 (1): candidate blocking at scale — clusters/draft succeed at 5k.

The 3,000-file wall was the headline cliff: past it `clusters` and `draft`
refused outright (`similarity_cap` / `corpus_too_big`) and the only
documented answer was per-directory partitioning. Candidate blocking
(inverted rare-token→docs index, cosine on the pairs that share ≥K rare
tokens) turns the O(n²) matcher into O(Σ rare-token²), so the caps become
memory bounds overridable with --force rather than refusals.

The assertions are the three properties that matter, not a naive
"blocked == all-pairs" (which the per-directory matcher can never satisfy —
within a directory everything shares the template and over-merges):

1. a 5k corpus CLUSTERS and DRAFTS without a cap refusal (the headline);
2. blocking is PRECISE — it never fabricates a merge the all-pairs matcher
   would not also make (blocked pairs ⊆ all-pairs pairs), so clusters stay
   hints you can trust;
3. a genuine topic (files sharing rare content tokens) is recovered
   IDENTICALLY blocked vs all-pairs — blocking is lossless where it counts.

Skipped by default; weekly CI alongside the scanner smoke:

    SLOPKEEPER_RUN_SLOW=1 uv run pytest -q tests/test_scale_clusters.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from bench_reference_scan import build_scale_tree
from helpers import write
from slopkeeper.similar import (
    MAX_FILES,
    _neighbor_adjacency,
    build_vectors,
    candidate_pairs,
    collect_similarity,
    similarity_clusters,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("SLOPKEEPER_RUN_SLOW"),
        reason="scale clusters smoke: set SLOPKEEPER_RUN_SLOW=1 (weekly CI)",
    ),
]


def _pairs_of(clusters):
    out = set()
    for c in clusters:
        cs = sorted(c)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                out.add((cs[i], cs[j]))
    return out


class TestScaleClusters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sk-scale-clusters-"))
        build_scale_tree(cls.tmp)  # 5001 files

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _file_dicts(self, root):
        return [
            {"path": str(p.relative_to(root))} for p in root.rglob("*") if p.is_file()
        ]

    def test_clusters_at_5k_succeeds_without_force(self):
        # The headline: past the old 3000 cap, no similarity_cap refusal.
        files = self._file_dicts(self.tmp)
        self.assertGreater(len(files), MAX_FILES)
        data = collect_similarity(self.tmp, files, with_dupes=False)
        self.assertIn("clusters", data)
        self.assertIn("unclustered", data)

    def test_draft_at_5k_succeeds_without_force(self):
        # The re-tend loop's nearest-neighbour scan runs past the cap too.
        from slopkeeper.draft import draft
        from slopkeeper.status import write_manifest

        corpus = self.tmp / "draftcorpus"
        shutil.copytree(self.tmp, corpus)
        write_manifest(corpus)  # baseline = the whole 5k tree
        # A fresh file added AFTER the baseline, so the draft has new slop.
        (corpus / "brand_new_memo.md").write_text("# brand new memo\n\ncontext\n")
        out = draft(corpus)  # no force: 5k filed < MEMORY_MAX_FILES
        self.assertIsNotNone(out["plan"])

    def test_draft_past_memory_bound_refuses_then_force_succeeds(self):
        # The bound is a refusal, --force is the documented override.
        import slopkeeper.draft as draft_module
        from unittest import mock

        from slopkeeper.draft import draft
        from slopkeeper.models import SlopKeeperError
        from slopkeeper.status import write_manifest

        corpus = self.tmp / "draftbound"
        corpus.mkdir()
        for i in range(40):
            write(corpus / f"bulk/d{i % 4}/f{i:03d}.md", f"# file {i}\n\ncontent {i}\n")
        write_manifest(corpus)  # baseline before the new slop shot in
        (corpus / "slop_new.md").write_text("# slop new\n\ncontent\n")
        with mock.patch.object(draft_module, "MEMORY_MAX_FILES", 10):
            with self.assertRaises(SlopKeeperError) as cm:
                draft(corpus)
            self.assertEqual(cm.exception.code, "corpus_too_big")
            # --force runs the blocked scan anyway.
            out = draft(corpus, force=True)
            self.assertIsNotNone(out["plan"])


class TestBlockingPrecision(unittest.TestCase):
    """A corpus past MAX_FILES with ONE genuine topic cluster."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sk-block-precision-"))
        n_dirs = 65
        per_dir = 48  # 3120 noise files — past MAX_FILES so blocking engages
        for d in range(n_dirs):
            bulk = cls.tmp / f"noise/d{d:03d}"
            bulk.mkdir(parents=True)
            for i in range(per_dir):
                (bulk / f"note_{d:03d}_{i:03d}.md").write_text(
                    f"# note {d}-{i}\n\nuniqueword{d}_{i} filler{d}\n", encoding="utf-8"
                )
        # A real topic: 30 files about the same rare subject.
        q3 = cls.tmp / "q3"
        q3.mkdir()
        cls.q3_files = []
        for i in range(30):
            name = f"q3-revenue-line-{i:03d}.md"
            (q3 / name).write_text(
                f"# q3 revenue line {i}\n\nSee q3-revenue-overview.md and "
                f"q3-revenue-model.md for the q3 revenue rollup. drift {i}\n",
                encoding="utf-8",
            )
            cls.q3_files.append(f"q3/{name}")
        cls.vecs = build_vectors(
            [
                {"path": str(p.relative_to(cls.tmp))}
                for p in cls.tmp.rglob("*")
                if p.is_file()
            ]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_blocking_engages_past_max_files(self):
        self.assertGreater(len(self.vecs), MAX_FILES)

    def test_blocked_is_precise_subset_of_allpairs(self):
        allp = _pairs_of(similarity_clusters(self.vecs))
        blocked = _pairs_of(
            similarity_clusters(
                self.vecs, adjacency=_neighbor_adjacency(candidate_pairs(self.vecs))
            )
        )
        # Blocking never fabricates a merge: every blocked pair is one the
        # full matcher would also (possibly) make.
        self.assertTrue(
            blocked <= allp, "blocking introduced a merge all-pairs did not"
        )

    def test_genuine_topic_recovered_identically(self):
        allp = similarity_clusters(self.vecs)
        blocked = similarity_clusters(
            self.vecs, adjacency=_neighbor_adjacency(candidate_pairs(self.vecs))
        )

        def q3_members(clusters):
            for c in clusters:
                if any(f in self.q3_files for f in c):
                    return set(c)
            return set()

        self.assertEqual(q3_members(blocked), q3_members(allp))
        self.assertGreaterEqual(len(q3_members(allp)), 2)

    def test_blocked_deterministic(self):
        adj = _neighbor_adjacency(candidate_pairs(self.vecs))
        self.assertEqual(
            similarity_clusters(self.vecs, adjacency=adj),
            similarity_clusters(self.vecs, adjacency=adj),
        )

    def test_propose_by_topic_scales_past_cap(self):
        # propose clusters by-topic too — and that pass must use the
        # blocked matcher past MAX_FILES (P4 7), not the all-pairs wall.
        from slopkeeper import propose

        out = propose.build_candidates(self.tmp)
        self.assertEqual(
            [c["id"] for c in out["candidates"]],
            ["by-project", "by-role", "by-topic", "by-kind", "hybrid"],
        )
        bt = next(c for c in out["candidates"] if c["id"] == "by-topic")
        # The genuine q3 topic clusters; the noise never shares a folder.
        self.assertGreaterEqual(bt["move_count"], 2)
        moves = propose.candidate_moves(self.tmp, "by-topic")
        self.assertTrue(moves)
        self.assertTrue(all(d.startswith("topics/") for d in moves.values()))
        q3 = {p for p in moves if p.startswith("q3/")}
        self.assertGreaterEqual(len(q3), 2)


if __name__ == "__main__":
    unittest.main()
