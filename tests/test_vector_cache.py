# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""vectors.json cache (DESIGN.md §5.3) — the re-tend loop stops
re-vectorizing the whole catalog.

The cache is load-bearing: a corrupted-but-dict-shaped tf entry must be
USED (and visibly change draft output), while a missing/corrupt/wrong-
schema file is a silent full miss with identical results.
"""

from __future__ import annotations

import json
import os
import unittest
import unittest.mock as mock
from collections import Counter
from datetime import datetime

from helpers import TempCase, write_plan
from slopkeeper import draft as draft_module
from slopkeeper import similar as similar_module
from slopkeeper.apply import apply_plan
from slopkeeper.draft import draft
from slopkeeper.models import sha256_file
from slopkeeper.scaffold import scaffold_from_plan
from slopkeeper.similar import (
    VECTOR_CACHE_SCHEMA,
    build_vectors,
    build_vectors_cached,
    collect_similarity,
    vectors_from_tf,
)
from slopkeeper.status import write_manifest

SLOP = {"globex_new_memo.md", "minutes-log.md"}
# Filed (subdirectory) corpus after the standard first tend, minus
# TOOL_LAYER names, which _file_entries skips.
FILED = {
    "archive/scraps/ledger_wb.html",
    "docs/canonical.md",
    "lead-gen/README.md",
    "research/dossiers/acme_dossier.md",
    "research/dossiers/globex_dossier.md",
    "spark-glm-5.3/README.md",
    "spark-glm-5.3/handoff-tp2.md",
}
EDITED = "research/dossiers/globex_dossier.md"


def cache_path(root):
    return root / ".slopkeeper" / "vectors.json"


def read_cache(root):
    return json.loads(cache_path(root).read_text(encoding="utf-8"))


def write_cache(root, data):
    cache_path(root).write_text(json.dumps(data), encoding="utf-8")


def tf_spy(recorded):
    """Patch slopkeeper.similar._doc_tokens to record vectorized paths."""
    real = similar_module._doc_tokens

    def spy(path, title, head):
        recorded.append(path)
        return real(path, title, head)

    return mock.patch.object(similar_module, "_doc_tokens", side_effect=spy)


def counting_build_vectors():
    """Patch slopkeeper.draft.build_vectors to record each call's paths."""
    real = draft_module.build_vectors
    calls = []

    def counting(files):
        calls.append([f["path"] for f in files])
        return real(files)

    return mock.patch.object(draft_module, "build_vectors", side_effect=counting), calls


class TestDraftVectorCache(TempCase):
    def _tend_and_slop(self):
        """First tend, then two root-level slop files (one matchable)."""
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        scaffold_from_plan(plan_path, self.root, execute=True)
        (self.root / "globex_new_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        (self.root / "minutes-log.md").write_text(
            "# Minutes\n\nstandup agenda parking lot\n" * 3
        )

    def test_first_draft_writes_cache_second_reuses(self):
        self._tend_and_slop()
        out1 = draft(self.root)

        cache = read_cache(self.root)
        self.assertEqual(cache["schema"], VECTOR_CACHE_SCHEMA)
        datetime.fromisoformat(cache["built_at"])
        self.assertEqual(cache["params"], {"t1": 0.25})
        self.assertEqual(set(cache["docs"]), FILED)
        manifest = json.loads((self.root / ".slopkeeper" / "manifest.json").read_text())
        entry = cache["docs"][EDITED]
        self.assertEqual(entry["sha256"], manifest["files"][EDITED][2])
        # Raw tf counts, exactly what _doc_tokens would produce.
        self.assertEqual(entry["tf"]["globex"], 12)

        # Second draft on an unchanged corpus: build_vectors is called
        # for the new slop ONLY — no catalog-wide re-vectorization.
        patch, calls = counting_build_vectors()
        with patch:
            out2 = draft(self.root)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), SLOP)
        self.assertEqual(out2, out1)

        # Corrupt one cached tf to garbage: it must be USED (the cache
        # is load-bearing, not decorative) — the 0.40 via-dossier move
        # collapses, and the corrupt entry is not rewritten.
        cache = read_cache(self.root)
        cache["docs"][EDITED]["tf"] = {"zzz_garbage_token": 9}
        write_cache(self.root, cache)
        patch, calls = counting_build_vectors()
        with patch:
            out3 = draft(self.root)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), SLOP)
        self.assertEqual(out3["plan"]["moves"], [])
        reason = next(
            u["reason"]
            for u in out3["plan"]["unresolved"]
            if u["path"] == "globex_new_memo.md"
        )
        self.assertIn("acme_dossier.md", reason)
        self.assertEqual(
            read_cache(self.root)["docs"][EDITED]["tf"], {"zzz_garbage_token": 9}
        )

    def test_editing_one_filed_file_revectorizes_only_it(self):
        self._tend_and_slop()
        draft(self.root)  # warm the cache
        p = self.root / EDITED
        p.write_text(p.read_text() + "\nsetlist quartet reunion tour\n")
        write_manifest(self.root)  # re-tend: manifest carries the new sha
        new_sha = sha256_file(p)
        (self.root / "quartet-notes.md").write_text(
            "# Quartet notes\n\nsetlist quartet reunion tour\n" * 3
        )

        recorded = []
        with tf_spy(recorded):
            draft(self.root)
        # Exactly the edited file re-vectorized from the corpus (plus
        # the always-fresh slop); every other cached file was reused.
        self.assertEqual(set(recorded), {EDITED, "quartet-notes.md"})
        cache = read_cache(self.root)
        self.assertEqual(cache["docs"][EDITED]["sha256"], new_sha)
        self.assertEqual(cache["docs"][EDITED]["tf"].get("setlist"), 1)

    def test_params_change_invalidates_whole_cache(self):
        self._tend_and_slop()
        draft(self.root)
        self.assertEqual(read_cache(self.root)["params"], {"t1": 0.25})

        recorded = []
        with tf_spy(recorded), mock.patch.object(draft_module, "DEFAULT_T1", 0.4):
            draft(self.root)
        # A different t1 is a WHOLE-cache miss: every filed file is
        # re-vectorized (never a mixed-parameter match).
        self.assertEqual(set(recorded), FILED | SLOP)
        self.assertEqual(read_cache(self.root)["params"], {"t1": 0.4})

    def test_deleted_cache_is_a_full_miss_with_identical_results(self):
        self._tend_and_slop()
        out1 = draft(self.root)
        os.unlink(cache_path(self.root))

        recorded = []
        with tf_spy(recorded):
            out2 = draft(self.root)
        self.assertEqual(out2, out1)  # cache-miss path stays correct
        self.assertEqual(set(recorded), FILED | SLOP)
        self.assertEqual(set(read_cache(self.root)["docs"]), FILED)

    def test_corrupt_cache_silently_rebuilds(self):
        self._tend_and_slop()
        out1 = draft(self.root)
        for bad in [
            "{not json",
            json.dumps(
                {"schema": "slopkeeper/vectors@9", "params": {"t1": 0.25}, "docs": {}}
            ),
        ]:
            cache_path(self.root).write_text(bad)
            self.assertEqual(draft(self.root), out1)
            self.assertEqual(read_cache(self.root)["schema"], VECTOR_CACHE_SCHEMA)
            self.assertEqual(set(read_cache(self.root)["docs"]), FILED)

        # Dict-shaped garbage that crashes the idf math: full fallback,
        # never an error, cache rewritten from real content.
        cache = read_cache(self.root)
        for rel in FILED:
            cache["docs"][rel]["tf"] = {"boom": "not-a-number"}
        write_cache(self.root, cache)
        recorded = []
        with tf_spy(recorded):
            self.assertEqual(draft(self.root), out1)
        self.assertEqual(set(recorded), FILED | SLOP)
        self.assertEqual(read_cache(self.root)["docs"][EDITED]["tf"]["globex"], 12)


class TestClustersVectorCache(TempCase):
    def _docs(self):
        filler = "\n# churn retention funnel onboarding\n"
        texts = {
            "market-analysis-audit.md": "# Market analysis audit\n" + filler * 3,
            "market-analysis-findings.md": "# Market analysis findings\n" + filler * 3,
            "market-analysis-history.md": "# Market analysis history\n" + filler * 3,
            "minutes.md": "# Minutes\n\nstandup",
            "itinerary.md": "# Itinerary\n\nlayover",
        }
        for name, text in texts.items():
            (self.root / name).write_text(text)
        return [
            {"path": p.name, "title": p.name, "head": p.read_text()[:400]}
            for p in sorted(self.root.glob("*.md"))
        ]

    def test_clusters_identical_with_and_without_warm_cache(self):
        docs = self._docs()
        self.assertFalse(cache_path(self.root).exists())
        cold = collect_similarity(self.root, docs)  # writes the cache
        self.assertTrue(cache_path(self.root).exists())
        self.assertEqual(collect_similarity(self.root, docs), cold)
        os.unlink(cache_path(self.root))
        self.assertEqual(collect_similarity(self.root, docs), cold)
        # A corrupt cache file is a silent miss with the same output.
        cache_path(self.root).write_text("{oops")
        self.assertEqual(collect_similarity(self.root, docs), cold)


class TestBuildVectorsCached(unittest.TestCase):
    ENTRIES = [{"path": "a.md", "title": "A", "head": "alpha beta"}]

    def test_hit_uses_cached_tf_verbatim(self):
        cache = {"a.md": {"sha256": "cafe", "tf": {"setlist": 3}}}
        vecs, fresh = build_vectors_cached(self.ENTRIES, cache, lambda rel: "cafe")
        self.assertEqual(fresh, {})
        self.assertEqual(vecs, vectors_from_tf({"a.md": Counter({"setlist": 3})}))

    def test_hash_mismatch_or_null_never_reuses(self):
        cache = {"a.md": {"sha256": "cafe", "tf": {"setlist": 3}}}
        for sha in ("beef", None):
            vecs, fresh = build_vectors_cached(
                self.ENTRIES, cache, lambda rel, s=sha: s
            )
            self.assertEqual(vecs, build_vectors(self.ENTRIES))
            self.assertEqual(set(fresh), {"a.md"})

    def test_garbage_tf_falls_back_to_full_revectorization(self):
        cache = {"a.md": {"sha256": "cafe", "tf": {"setlist": "banana"}}}
        vecs, fresh = build_vectors_cached(self.ENTRIES, cache, lambda rel: "cafe")
        self.assertEqual(vecs, build_vectors(self.ENTRIES))
        self.assertEqual(fresh["a.md"]["tf"], {"alpha": 1, "beta": 1})


if __name__ == "__main__":
    unittest.main()
