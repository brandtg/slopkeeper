# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P5 normalize-on-match — the Mac corpus that cites itself in two alphabets.

The audit's coverage gap, verbatim: "no test puts a non-ASCII character in
a _filename_; and Unicode NFC/NFD normalization (the macOS decomposed-name
corpus) is untested and unhandled — `refs` basename matching does not
casefold accents, and the fingerprint treats `é` (NFC) and `e\\u0301` (NFD)
as unrelated files, which is byte-correct but agent-surprising."

The fix is one-sided on purpose, and the tests hold both halves of that:

* MATCHING folds (NFD + casefold) — a file stored decomposed is cited by
  text that composes it, and the scan, `refs`, the risk rows and the
  rewrite pass all find it;
* STORAGE does not — the fingerprint still treats the two spellings as the
  two different files they are on disk, and a path that exists byte-exact
  never resolves through a fold to a case-different neighbour.
"""

from __future__ import annotations

import json
import shutil
import unicodedata
import unittest
from pathlib import Path

from helpers import TempCase, write
from slopkeeper.models import corpus_fingerprint
from slopkeeper.references import (
    CitationIndex,
    citation_lines,
    refs_lookup,
    scan_references,
)

NFD = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731
NFC = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731


def _rel(root: Path) -> list[Path]:
    return sorted(
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and ".slopkeeper" not in p.parts
    )


class TestNormalizeOnMatch(TempCase):
    """The corpus file is NFD on disk; the prose that cites it is NFC."""

    def setUp(self):
        super().setUp()
        self.dir = self.tmp / "mac"
        shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "notes").mkdir(parents=True)
        # Stored DECOMPOSED (what the Finder writes).
        self.nfd_name = NFD("café dossier.md")
        write(self.dir / "notes" / self.nfd_name, "# contents\n")
        # Cited COMPOSED (what a keyboard, a terminal, and an agent write).
        self.nfc_name = NFC("café dossier.md")
        write(self.dir / "index.md", f"See notes/{self.nfc_name} for details.\n")
        self.rel = _rel(self.dir)

    def test_the_two_spellings_really_differ(self):
        # The fixture is only worth anything if it is the case in question.
        self.assertNotEqual(self.nfd_name, self.nfc_name)
        self.assertNotIn(self.nfc_name, {p.name for p in self.rel})

    def test_scan_finds_the_citation(self):
        refs = scan_references(self.dir, self.rel)
        self.assertEqual(refs.get(f"notes/{self.nfd_name}"), ["index.md"])

    def test_citation_lines_finds_the_site(self):
        hits = citation_lines(self.dir, self.nfc_name, self.rel)
        self.assertEqual([(h["file"], h["line"]) for h in hits], [("index.md", 1)])

    def test_refs_lookup_reports_it_as_existing(self):
        out = refs_lookup(self.dir, self.nfc_name, self.rel)
        self.assertTrue(out["exists"])
        self.assertFalse(out["nonexistent_path"])
        self.assertEqual(len(out["citations"]), 1)

    def test_bare_accented_basename_resolves(self):
        index = CitationIndex(self.rel)
        hits = citation_lines(self.dir, self.nfc_name, self.rel)
        self.assertEqual([h["file"] for h in hits], ["index.md"])
        self.assertTrue(index.basename_paths(self.nfc_name))

    def test_folded_matching_accepts_the_accented_citations(self):
        # Every spelling of a name answers to the file on disk once the
        # citation carries its path, and nothing collides that shouldn't.
        # Case and accents fold for MATCHING only. (A bare word with no
        # extension is not a citation candidate at all, so the decoration
        # words in the prose are not assertions either.)
        root = self.tmp / "corpus-folded"
        shutil.rmtree(root, ignore_errors=True)
        (root / "notes").mkdir(parents=True)
        write(root / "notes" / NFD("café.md"), "# c\n")
        write(root / "notes" / NFD("crème brûlée.md"), "# c\n")
        write(root / "notes" / NFC("crème.md"), "# c\n")
        write(root / "API.md", "# a\n")
        write(
            root / "index.md",
            "see notes/CAFÉ.MD and notes/café.md — the café à la crème\n"
            "recipe, the `API.MD` contract, and `crème.md` beside\n"
            "notes/crème brûlée.md in prose.\n",
        )
        refs = scan_references(root, _rel(root))
        cited = set(refs)
        # Exactly the four files, no coincidences:
        #   - the uppercase NFC path citation answers to the NFD file on
        #     disk (accents AND case fold for the match);
        #   - the spaced accented name matches across normal forms when
        #     the citation carries its path;
        #   - the ASCII case-variant answers to the ASCII file.
        self.assertEqual(
            cited,
            {
                "notes/" + NFD("café.md"),
                "notes/" + NFD("crème brûlée.md"),
                "notes/" + NFC("crème.md"),
                "API.md",
            },
        )
        # And the prose words alone (no extension) cited nothing.
        self.assertEqual(sorted(cited, key=len)[:1], ["API.md"])

    def test_apply_rewrites_the_citation(self):
        # The moved file's citation is FIXED, not silently left pointing at
        # a path that no longer exists — which is how the audit described
        # the pre-P5 behavior. Non-spaced name here: a spaced path cited
        # without backticks is the pre-existing single-token limitation
        # (the basename-after-a-slash C2 rule), the same for ASCII and for
        # accents, and test_the_gate_reports_it_too covers its loud side.
        from slopkeeper.apply import apply_plan

        root = self.tmp / "apply-nfd"
        shutil.rmtree(root, ignore_errors=True)
        (root / "notes").mkdir(parents=True)
        disk = NFD("café.md")
        text = NFC("café.md")
        write(root / "notes" / disk, "# contents\n")
        write(root / "index.md", f"See notes/{text} for details.\n")
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(root),
            "strategy": "hybrid",
            "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
            "moves": [
                {
                    "from": f"notes/{disk}",
                    "to": "archive/coffee.md",
                    "reason": "archive",
                }
            ],
            "keep": ["index.md", "notes"],
            "unresolved": [],
            "folders": [],
        }
        plan_path = self.tmp / "plan-nfd.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        out = apply_plan(plan_path, root, execute=True)
        self.assertEqual(out["mode"], "executed", out)
        self.assertEqual(out["rewrites"]["count"], 1)
        after = (root / "index.md").read_text(encoding="utf-8")
        self.assertIn("archive/coffee.md", after)
        self.assertNotIn("café", after)


class TestFoldingIsMatchOnly(TempCase):
    """The half that must NOT fold: storage, hashing, and case."""

    def test_fingerprint_still_sees_two_files(self):
        # NFC and NFD on disk are different bytes: the fingerprint says so,
        # because a plan bound to a renamed corpus must be drift, not a
        # match. Matching folds; identity does not.
        root = self.tmp / "both"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        write(root / f"{NFC('café')}.md", "# composed\n")
        write(root / f"{NFD('café')}.md", "# decomposed\n")
        before = corpus_fingerprint(root)
        (root / f"{NFD('café')}.md").write_text("# changed\n", encoding="utf-8")
        self.assertNotEqual(corpus_fingerprint(root), before)
        self.assertEqual(len(list(root.glob("*.md"))), 2)

    def test_case_is_not_folded_for_path_resolution(self):
        # `API.md` and `api.md` can coexist on a case-sensitive filesystem.
        # Folding CASE into path resolution would resolve one to the other
        # and the rewrite pass would then write the wrong path — so the
        # path leg folds COMPOSITION only. (Basenames keep their historical
        # case-insensitivity: that leg is a unique-name lookup already.)
        root = self.tmp / "case"
        shutil.rmtree(root, ignore_errors=True)
        (root / "docs").mkdir(parents=True)
        write(root / "docs/api.md", "# lower\n")
        index = CitationIndex([Path("docs/api.md")])
        self.assertIsNone(index.match_path("docs/API.md"))
        self.assertEqual(index.match_path("docs/api.md"), "docs/api.md")
        # ...and composition still resolves through the same call.
        self.assertEqual(
            CitationIndex([Path(f"docs/{NFD('café')}.md")]).match_path(
                f"docs/{NFC('café')}.md"
            ),
            f"docs/{NFD('café')}.md",
        )

    def test_ascii_corpus_answers_are_unchanged(self):
        # The fold paths are gated on non-ASCII: an ASCII corpus must not
        # gain or lose a single citation from this change.
        refs = scan_references(self.root, _rel(self.root))
        self.assertEqual(refs.get("acme_dossier.md"), ["WATCHLIST.md"])
        self.assertIn("globex_dossier.md", refs)

    def test_nfd_cited_by_nfd_and_nfc_cited_by_nfc_still_work(self):
        # The boring diagonal: same form on both sides still matches (the
        # byte-exact leg), for both forms.
        for cited, writer in ((NFD, NFC), (NFC, NFD)):
            with self.subTest(disk=cited("x"), text=writer("x")):
                root = self.tmp / f"same-{cited('x')}-{writer('x')}"
                shutil.rmtree(root, ignore_errors=True)
                root.mkdir(parents=True)
                name = cited("naïve notes.md")
                write(root / name, "# n\n")
                write(root / "citer.md", f"see {writer('naïve notes.md')} now\n")
                refs = scan_references(root, _rel(root))
                self.assertEqual(refs.get(name), ["citer.md"])


class TestFoldedSpanIntegrity(TempCase):
    """The rewrite pass writes by SPAN — folding must not move a span."""

    def test_hit_spans_address_the_original_line(self):
        from slopkeeper.references import BasenameMatcher

        line = f"before {NFC('café dossier.md')} after"
        matcher = BasenameMatcher([NFD("café dossier.md")])
        hits = list(matcher.finditer(line))
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        # The span indexes the ORIGINAL line, and the text is the original
        # bytes' spelling — not the folded form the match was computed on.
        self.assertEqual(line[hit.start(1) : hit.end(1)], hit.group(1))
        self.assertEqual(hit.group(1), NFC("café dossier.md"))

    def test_ascii_names_still_take_the_trie_path(self):
        # The fast path is untouched: an ASCII candidate set against an
        # ASCII line never builds a folded pattern.
        from slopkeeper.references import BasenameMatcher

        matcher = BasenameMatcher(["api.md"])
        hits = list(matcher.finditer("see api.md"))
        self.assertEqual([h.group(1) for h in hits], ["api.md"])
        self.assertFalse(matcher._slow_used())

    def test_mixed_corpus_keeps_exact_hits_first(self):
        # A name that appears in its own spelling is matched by the exact
        # leg; the folded leg only ever ADDS the other normal form.
        from slopkeeper.references import BasenameMatcher

        matcher = BasenameMatcher([NFD("café.md"), "plain.md"])
        hits = list(matcher.finditer(f"{NFD('café.md')} and plain.md"))
        self.assertEqual([h.group(1) for h in hits], [NFD("café.md"), "plain.md"])


if __name__ == "__main__":
    unittest.main()
