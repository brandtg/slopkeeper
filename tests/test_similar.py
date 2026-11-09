# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""similar.py — clusters, hubs, stem/identical groups."""

from __future__ import annotations

import unittest

from helpers import TempCase
from collections import Counter
from slopkeeper.similar import (
    _cosine,
    build_vectors,
    cluster_label,
    collect_similarity,
    hubs,
    identical_groups,
    same_stem_groups,
    similarity_clusters,
)


class TestSimilarity(TempCase):
    def test_clusters_group_related_files(self):
        # Three related docs + two unrelated.
        (self.root / "market-analysis-audit.md").write_text(
            "# Market analysis audit\n\nchurn retention funnel onboarding\n" * 3
        )
        (self.root / "market-analysis-findings.md").write_text(
            "# Market analysis findings\n\nchurn retention funnel onboarding\n" * 3
        )
        (self.root / "market-analysis-history.md").write_text(
            "# Market analysis history\n\nchurn retention funnel onboarding\n" * 3
        )
        (self.root / "minutes.md").write_text("# Minutes\n\nstandup")
        (self.root / "itinerary.md").write_text("# Itinerary\n\nlayover")
        files = [
            {"path": p.name, "title": p.name, "head": p.read_text()[:400]}
            for p in sorted(self.root.glob("*.md"))
        ]
        data = collect_similarity(self.root, files)
        cov = {p for c in data["clusters"] for p in c["files"]}
        covid_cluster = next(
            (c for c in data["clusters"] if "market-analysis-audit.md" in c["files"]),
            None,
        )
        self.assertIsNotNone(covid_cluster)
        self.assertEqual(
            set(covid_cluster["files"]),
            {
                "market-analysis-audit.md",
                "market-analysis-findings.md",
                "market-analysis-history.md",
            },
        )
        self.assertNotIn("minutes.md", cov)
        self.assertIn("market analysis", covid_cluster["label"])

    def test_same_stem_and_identical_groups(self):
        (self.root / "scrape.html").write_text("<html>same</html>")
        (self.root / "scrape.txt").write_text("<html>same</html>")
        (self.root / "copy1.md").write_text("# same\n")
        (self.root / "copy2.md").write_text("# same\n")
        paths = ["scrape.html", "scrape.txt", "copy1.md", "copy2.md"]
        stems = same_stem_groups(paths)
        self.assertIn(["scrape.html", "scrape.txt"], stems)
        ident = identical_groups(self.root, paths)
        self.assertEqual(sorted(ident[0]), ["copy1.md", "copy2.md"])

    def test_hubs_rank_inbound_refs(self):
        refs = {"hub.md": ["a.md", "b.md", "c.md"], "leaf.md": ["a.md"]}
        out = hubs(refs, min_referrers=2)
        self.assertEqual(out[0]["basename"], "hub.md")
        self.assertEqual(out[0]["referrers"], 3)

    def test_stale_version_pair(self):
        # Same basename in two dirs: the older copy with DIFFERENT
        # content is the likely older version; identical copies are
        # not versions.
        import os

        (self.root / "a").mkdir()
        (self.root / "b").mkdir()
        (self.root / "a/report.md").write_text("new content\n")
        (self.root / "b/report.md").write_text("old content\n")
        os.utime(self.root / "a/report.md", (2000, 2000))
        os.utime(self.root / "b/report.md", (1000, 1000))
        (self.root / "c").mkdir()
        (self.root / "c/dup.md").write_text("same\n")
        (self.root / "d").mkdir()
        (self.root / "d/dup.md").write_text("same\n")
        paths = [
            "a/report.md",
            "b/report.md",
            "c/dup.md",
            "d/dup.md",
        ]
        out = collect_similarity(self.root, files=[{"path": p} for p in paths])
        stale = out["stale_versions"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["file"], "b/report.md")
        self.assertEqual(stale[0]["likely_newer"], "a/report.md")

    def test_identical_subtree_collapsed(self):
        # Four copies of one vendored subtree → ONE cluster entry
        # naming the representative and every copy; the copy files
        # leave the similarity input and the unclustered list.
        for d in ("vendor/one", "vendor/two", "vendor/three", "elsewhere"):
            (self.root / d).mkdir(parents=True)
            (self.root / d / "x.css").write_text("body { color: red }\n")
            (self.root / d / "y.js").write_text("alert(1);\n")
        (self.root / "loose.md").write_text("# loose\n")
        paths = [
            f"{d}/{n}"
            for d in ("vendor/one", "vendor/two", "vendor/three", "elsewhere")
            for n in ("x.css", "y.js")
        ] + ["loose.md"]
        out = collect_similarity(self.root, files=[{"path": p} for p in paths])
        subs = out["identical_subtrees"]
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["dir"], "elsewhere")
        copies = ["vendor/one", "vendor/three", "vendor/two"]
        self.assertEqual(subs[0]["copies"], copies)
        self.assertEqual(subs[0]["files"], 2)
        synth = [c for c in out["clusters"] if "identical subtree" in c["label"]]
        self.assertEqual(len(synth), 1)
        self.assertEqual(synth[0]["copies"], copies)
        # the copies are accounted for by the entry, not unclustered
        self.assertNotIn("vendor/two/x.css", out["unclustered"])
        self.assertNotIn("elsewhere/x.css", out["unclustered"])
        self.assertIn("loose.md", out["unclustered"])


# --- regression: precision over recall (TEST_PLAN similar.py rows) ---


# A token chain: each adjacent pair shares exactly one token and the
# endpoints share none. Greedy single-link without a mean-similarity floor
# would fold all five into one blob.
CHAIN_DOCS = [
    ("alpha-notes.md", "# alpha notes\n\nalpha notes alpha notes\n"),
    ("beta-notes.md", "# beta notes\n\nbeta notes beta notes\n"),
    ("beta-gamma.md", "# beta gamma\n\nbeta gamma beta gamma\n"),
    ("gamma-delta.md", "# gamma delta\n\ngamma delta gamma delta\n"),
    ("delta-epsilon.md", "# delta epsilon\n\ndelta epsilon delta epsilon\n"),
]


class TestPrecisionOverRecall(TempCase):
    def _docs(self) -> list[dict]:
        out = []
        for name, text in CHAIN_DOCS:
            (self.root / name).write_text(text, encoding="utf-8")
            out.append({"path": name, "title": name, "head": text[:400]})
        return out

    def test_token_chain_does_not_collapse_into_one_blob(self):
        data = collect_similarity(self.root, self._docs())
        clustered = {f for c in data["clusters"] for f in c["files"]}
        # The mean-similarity floor must stop the chain: delta-epsilon only
        # clears t1 against its one strong link (gamma-delta), so its MEAN
        # similarity to the 3-member blob falls below t1*ratio and it stays
        # out; alpha-notes never clears t1 against anything.
        self.assertNotIn("delta-epsilon.md", clustered)
        self.assertNotIn("alpha-notes.md", clustered)
        # ...while the strong middle of the chain stays together.
        self.assertEqual(len(data["clusters"]), 1)
        self.assertEqual(
            [sorted(c["files"]) for c in data["clusters"]],
            [["beta-gamma.md", "beta-notes.md", "gamma-delta.md"]],
        )

    def test_three_link_chain_merges_pinned(self):
        # PIN + FLAG: the anti-chaining floor only bites once a cluster has
        # 3+ members. A 3-token chain (a~b, b~c, a<->c = 0.0) still merges
        # into one blob — the docstring's precision-over-recall claim
        # overstates the current guarantee. Pinned so a real fix diffs here.
        docs = self._docs()
        vecs = build_vectors(docs)
        self.assertEqual(_cosine(vecs["beta-notes.md"], vecs["gamma-delta.md"]), 0.0)
        clusters = similarity_clusters(vecs)
        self.assertIn(
            {"beta-gamma.md", "beta-notes.md", "gamma-delta.md"},
            [set(c) for c in clusters],
        )

    def test_zero_vector_cosine_is_zero(self):
        self.assertEqual(_cosine(Counter(), Counter({"x": 2.0})), 0.0)
        self.assertEqual(_cosine(Counter({"x": 2.0}), Counter()), 0.0)
        self.assertEqual(_cosine(Counter(), Counter()), 0.0)

    def test_clusters_deterministic_across_runs(self):
        docs = self._docs()
        self.assertEqual(
            collect_similarity(self.root, docs), collect_similarity(self.root, docs)
        )
        self.assertEqual(
            similarity_clusters(build_vectors(docs)),
            similarity_clusters(build_vectors(docs)),
        )

    def test_cluster_label_earliest_positioned_anchor(self):
        # The label is ordered by token position in the anchor member — the
        # member where the shared tokens appear EARLIEST in the name, not
        # the first list entry (gamma-alpha here would give "gamma alpha").
        self.assertEqual(
            cluster_label(["gamma-alpha.md", "beta-alpha.md"]), "beta alpha gamma"
        )


# --- cluster_label fallbacks: shared tokens, common parent, unlabeled ---


class TestClusterLabel(unittest.TestCase):
    def test_unlabeled_when_no_tokens_and_split_parents(self):
        # Digit-only stems tokenize to nothing; parents disagree.
        self.assertEqual(cluster_label(["a/123.md", "b/456.md"]), "(unlabeled)")

    def test_common_parent_label_when_no_tokens(self):
        self.assertEqual(cluster_label(["x/123.md", "x/456.md"]), "x")


if __name__ == "__main__":
    unittest.main()
