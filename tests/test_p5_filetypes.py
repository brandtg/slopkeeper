# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P5 filetype batch — what the corpus can now SEE.

Three acceptance lines from the audit, tested from the outside:

* every new format has an **inventory title** (a `.odt` whose title is
  `signal: none` is a bug, and a notebook whose title is `{` was one);
* every new format is **refs-citable** — and the by-name classics
  (`Makefile`, `LICENSE`, `README`) are citable at all, which the old
  "a basename must contain a dot" rule made impossible;
* every new format **tokenizes for clusters** — a notebook contributes its
  prose and definitions, not `cell_type` and `metadata`.

The containers are built with stdlib `zipfile` in the fixtures, so the test
proves the reader against the real format rather than a hand-written
string that happens to look like one.
"""

from __future__ import annotations

import json
import shutil
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from helpers import TempCase, write
from slopkeeper import inventory
from slopkeeper.inventory import NO_SIGNAL, collect
from slopkeeper.references import (
    CitationIndex,
    citation_lines,
    refs_lookup,
    scan_references,
)
from slopkeeper.similar import collect_similarity


def _zip(path: Path, members: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)
    return path


ODF_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
)


class TestContainerTitles(TempCase):
    """inventory-title: one real file per new kind."""

    def test_odt_title_is_the_heading_not_the_container(self):
        _zip(
            self.root / "plan.odt",
            {
                "mimetype": "application/vnd.oasis.opendocument.text",
                "content.xml": (
                    f"<office:text {ODF_NS}><office:automatic-styles/>"
                    '<text:h text:outline-level="1">Quarterly Plan</text:h>'
                    "<text:p>See ledger.csv for the numbers.</text:p>"
                    "</office:text>"
                ),
            },
        )
        inv = collect(self.root, scan=None)
        row = {f["path"]: f for f in inv["files"]}["plan.odt"]
        self.assertEqual(row["kind"], "binary")
        self.assertEqual(row["title"], "Quarterly Plan")
        self.assertIn("ledger.csv", row["head"])
        self.assertNotEqual(row["head"], NO_SIGNAL)

    def test_ods_title_comes_from_its_cells(self):
        _zip(
            self.root / "ledger.ods",
            {
                "mimetype": "application/vnd.oasis.opendocument.spreadsheet",
                "content.xml": (
                    f"<office:spreadsheet {ODF_NS}><table:table>"
                    "<table:table-cell><text:p>Revenue 2026</text:p>"
                    "</table:table-cell></table:table></office:spreadsheet>"
                ),
            },
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["ledger.ods"]
        self.assertEqual(row["title"], "Revenue 2026")

    def test_odp_title_comes_from_the_first_slide(self):
        _zip(
            self.root / "deck.odp",
            {
                "mimetype": "application/vnd.oasis.opendocument.presentation",
                "content.xml": (
                    f"<office:presentation {ODF_NS}>"
                    "<text:p>Acme go-to-market</text:p></office:presentation>"
                ),
            },
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["deck.odp"]
        self.assertEqual(row["title"], "Acme go-to-market")

    def test_epub_title_follows_container_to_the_opf(self):
        _zip(
            self.root / "book.epub",
            {
                "mimetype": "application/epub+zip",
                "META-INF/container.xml": (
                    "<container><rootfiles>"
                    '<rootfile full-path="OEBPS/content.opf"/>'
                    "</rootfiles></container>"
                ),
                "OEBPS/content.opf": (
                    "<package><metadata><dc:title>The Acme Handbook"
                    "</dc:title></metadata></package>"
                ),
            },
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["book.epub"]
        self.assertEqual(row["kind"], "binary")
        self.assertEqual(row["title"], "The Acme Handbook")

    def test_epub_without_a_title_is_loud_not_silent(self):
        # No <dc:title> is a real book, not an error: the head lands on the
        # NO_SIGNAL marker the pdf/docx path already used, so the agent can
        # tell "nothing extractable" from "not tried".
        _zip(
            self.root / "blank.epub",
            {
                "META-INF/container.xml": (
                    '<container><rootfiles><rootfile full-path="a.opf"/>'
                    "</rootfiles></container>"
                ),
                "a.opf": "<package><metadata></metadata></package>",
            },
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["blank.epub"]
        self.assertIsNone(row["title"])
        self.assertEqual(row["head"], NO_SIGNAL)

    def test_corrupt_container_is_loud_not_a_crash(self):
        (self.root / "broken.odt").write_bytes(b"not a zip at all")
        row = {f["path"]: f for f in collect(self.root)["files"]}["broken.odt"]
        self.assertIsNone(row["title"])
        self.assertEqual(row["head"], NO_SIGNAL)

    def test_legacy_office_uses_the_optional_external(self):
        # `.doc`/`.xls` are the only new formats that need a binary from
        # PATH; absent one the row is NO_SIGNAL (never silence, never a
        # crash), present one and the extracted text becomes the title.
        (self.root / "memo.doc").write_bytes(b"\xd0\xcf\x11\xe0fake ole")
        with mock.patch.object(inventory, "_which", return_value=None):
            row = {f["path"]: f for f in collect(self.root)["files"]}["memo.doc"]
        self.assertEqual(row["head"], NO_SIGNAL)

        with mock.patch.object(inventory, "_which", return_value="/usr/bin/antiword"):
            with mock.patch.object(
                inventory.subprocess,
                "run",
                return_value=mock.Mock(
                    returncode=0, stdout="# Memo title\nbody".encode(), stderr=b""
                ),
            ):
                row = {f["path"]: f for f in collect(self.root)["files"]}["memo.doc"]
        self.assertEqual(row["title"], "# Memo title")

    def test_svg_is_text_and_carries_its_own_title(self):
        write(
            self.root / "diagram.svg",
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<text>see ledger.csv</text></svg>\n",
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["diagram.svg"]
        self.assertEqual(row["kind"], "text")  # not the sniff: the table
        self.assertIn("ledger.csv", row["head"])


class TestNotebooks(TempCase):
    """.ipynb: a title that is not `{`, and tokens that are not JSON keys."""

    def _notebook(self, rel: str, cells: list[dict]) -> Path:
        doc = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        return write(self.root / rel, json.dumps(doc))

    def test_title_is_the_first_markdown_cell(self):
        self._notebook(
            "analysis.ipynb",
            [
                {"cell_type": "code", "source": ["import pandas as pd\n"]},
                {"cell_type": "markdown", "source": ["# Migration Notes\n", "body\n"]},
            ],
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["analysis.ipynb"]
        self.assertEqual(row["title"], "Migration Notes")
        self.assertNotEqual(row["title"], "{")

    def test_head_is_cells_not_json_boilerplate(self):
        self._notebook(
            "analysis.ipynb",
            [
                {
                    "cell_type": "markdown",
                    "source": ["# Funnel\n", "acme funnel notes\n"],
                },
                {
                    "cell_type": "code",
                    "source": ["def build_funnel(table):\n", "    return 1\n"],
                },
            ],
        )
        row = {f["path"]: f for f in collect(self.root)["files"]}["analysis.ipynb"]
        self.assertIn("Funnel", row["head"])
        self.assertIn("def build_funnel", row["head"])
        # The old head was 400 chars of JSON: keys and cell_type filled it.
        self.assertNotIn("cell_type", row["head"])
        self.assertNotIn("nbformat", row["head"])

    def test_malformed_notebook_falls_back_to_text(self):
        # A .ipynb that is not valid JSON is still a corpus file: the
        # graceful-degradation rule (inventory it, title null or raw),
        # never a crash and never a dropped row.
        write(self.root / "broken.ipynb", '{"cells": [')
        row = {f["path"]: f for f in collect(self.root)["files"]}["broken.ipynb"]
        self.assertEqual(row["kind"], "text")
        self.assertEqual(row["title"], '{"cells": [')

    def test_clusters_tokenize_the_cells(self):
        # Two notebooks about the same thing cluster; the JSON noise that
        # used to dominate every notebook vector no longer does.
        self._notebook(
            "funnel-a.ipynb",
            [{"cell_type": "markdown", "source": ["# Funnel conversion\n" * 4]}],
        )
        self._notebook(
            "funnel-b.ipynb",
            [{"cell_type": "markdown", "source": ["# Funnel conversion\n" * 4]}],
        )
        self._notebook(
            "unrelated.ipynb",
            [{"cell_type": "markdown", "source": ["# Sourdough hydration\n" * 4]}],
        )
        inv = collect(self.root)
        data = collect_similarity(self.root, inv["files"])
        clustered = {c["label"]: c["files"] for c in data["clusters"]}
        joined = [f for files in clustered.values() for f in files]
        self.assertIn("funnel-a.ipynb", joined)
        self.assertIn("funnel-b.ipynb", joined)
        self.assertNotIn("unrelated.ipynb", joined)

    def test_status_new_files_get_the_real_title(self):
        # `status` has its own head path; the audit's complaint was about
        # every surface that shows a title, so both must agree.
        self._notebook(
            "notes/new.ipynb",
            [{"cell_type": "markdown", "source": ["# Fresh Notebook\n"]}],
        )
        from slopkeeper.status import _head_for

        head = _head_for(self.root, "notes/new.ipynb")
        self.assertEqual(head["title"], "Fresh Notebook")


class TestAcceptanceMatrix(TempCase):
    """The audit's acceptance line for the whole batch, one table.

    "each new kind has inventory-title, refs-citable, and
    clusters-tokenize tests; every extension exists in exactly ONE table"
    — the last clause is `test_kinds.py`'s structural guarantee; this is
    the first three, run over every format P5 added, so a new kind cannot
    land with two of the three working.
    """

    # One phrase every fixture carries, so "did it tokenize" has a fixed
    # answer to look for.
    PHRASE = "Quantum Budgeting"

    def _fixtures(self) -> dict[str, tuple[str, bytes]]:
        """rel → (what a markdown file will cite it as, bytes on disk)."""
        p = self.PHRASE
        doc = f"<office:document-content {ODF_NS}>"
        odf = lambda body, mime: _zip_body(  # noqa: E731
            {
                "mimetype": mime,
                "content.xml": f"{doc}{body}</office:document-content>",
            }
        )
        para = f"<text:p>{p}</text:p>"
        cell = f"<table:table><table:table-cell>{para}</table:table-cell></table:table>"
        epub = _zip_body(
            {
                "mimetype": "application/epub+zip",
                "META-INF/container.xml": (
                    '<container><rootfiles><rootfile full-path="c.opf"/>'
                    "</rootfiles></container>"
                ),
                "c.opf": (
                    f"<package><metadata><dc:title>{p}</dc:title></metadata></package>"
                ),
            }
        )
        notebook = json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": [f"# {p}\n"]},
                    {
                        "cell_type": "code",
                        "source": ["def budget(x):\n", "    return x\n"],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
            }
        ).encode()
        svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{p}</text></svg>'
        return {
            "budget.odt": (
                "budget.odt",
                odf(para, "application/vnd.oasis.opendocument.text"),
            ),
            "budget.ods": (
                "budget.ods",
                odf(cell, "application/vnd.oasis.opendocument.spreadsheet"),
            ),
            "budget.odp": (
                "budget.odp",
                odf(para, "application/vnd.oasis.opendocument.presentation"),
            ),
            "budget.epub": ("budget.epub", epub),
            "budget.ipynb": ("budget.ipynb", notebook),
            "budget.svg": ("budget.svg", svg.encode()),
            "Makefile": ("Makefile", f"all: # {p} build\n".encode()),
            "Dockerfile": ("Dockerfile", f"FROM x # {p} image\n".encode()),
            "LICENSE": ("LICENSE", f"MIT License - {p}\n".encode()),
            "README": ("README", f"# {p}\n".encode()),
            ".env": (".env", b"PROJECT=quantum-budgeting\n"),
        }

    def _build(self, rel: str, cited: str, data: bytes) -> list[Path]:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(data)
        write(self.root / "citer.md", f"See {cited} for the Quantum Budgeting work.\n")
        return sorted(
            p.relative_to(self.root)
            for p in self.root.rglob("*")
            if p.is_file() and ".slopkeeper" not in p.parts
        )

    def test_every_new_kind_titles_cites_and_tokenizes(self):
        from slopkeeper.similar import _doc_tokens, tokenize

        for rel, (cited, data) in self._fixtures().items():
            with self.subTest(kind=rel):
                rel_paths = self._build(rel, cited, data)
                # 1. inventory-title: a real title, not null and not the
                # container's first byte.
                row = {f["path"]: f for f in collect(self.root)["files"]}[rel]
                self.assertTrue(row["title"], f"{rel}: no title")
                self.assertNotEqual(row["title"], "{")
                # 2. refs-citable: the citation the owner wrote is found.
                hits = citation_lines(self.root, cited, rel_paths)
                self.assertEqual(
                    [h["file"] for h in hits], ["citer.md"], f"{rel}: not citable"
                )
                # 3. clusters-tokenize: the title/head produce real tokens.
                tokens = _doc_tokens(rel, row["title"], row["head"])
                self.assertTrue(tokens, f"{rel}: nothing to cluster on")
                flat = " ".join(tokenize(f"{row['title']} {row['head'] or ''}"))
                self.assertIn(
                    "quantum", flat.replace("_", " "), f"{rel}: no content token"
                )
                for junk in ("cell_type", "nbformat", "<office:", "<?xml"):
                    self.assertNotIn(junk, str(tokens), f"{rel}: tokenizing {junk}")
                shutil.rmtree(self.root, ignore_errors=True)
                self.root.mkdir(parents=True)


def _zip_body(members: dict[str, str]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)
    return buf.getvalue()


class TestByNameIsCitable(TempCase):
    """refs-citable: the dotless classics enter the citation index."""

    def _corpus(self) -> list[Path]:
        write(self.root / "Makefile", "all:\n\techo see README and LICENSE\n")
        write(self.root / "README", "# The corpus readme\n\nSee Makefile.\n")
        write(self.root / "LICENSE", "MIT\n")
        write(self.root / ".env", "TOKEN=x # see README\n")
        write(self.root / "notes.md", "Build it with make -f Makefile today.\n")
        return sorted(
            p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()
        )

    def test_dotless_names_are_basenames(self):
        rel = self._corpus()
        index = CitationIndex(rel)
        for name in ("Makefile", "README", "LICENSE", ".env"):
            self.assertIn(name, index.by_basename, name)

    def test_prose_words_stay_out(self):
        # The rule the name table does NOT break: a bare word is still
        # never a basename, even when a file is named that way — "notes.md"
        # has a dot, and `today` (a prose word in the same line) is not a
        # citation of anything.
        rel = self._corpus()
        index = CitationIndex(rel)
        self.assertNotIn("today", index.by_basename)
        self.assertNotIn("build", index.by_basename)

    def test_citations_of_the_classics_are_found(self):
        rel = self._corpus()
        refs = {k: sorted(v) for k, v in scan_references(self.root, rel).items()}
        self.assertEqual(refs.get("Makefile"), ["README", "notes.md"])
        self.assertEqual(refs.get("README"), [".env", "Makefile"])
        self.assertEqual(refs.get("LICENSE"), ["Makefile"])

    def test_refs_lookup_is_honest_about_them(self):
        rel = self._corpus()
        out = refs_lookup(self.root, "Makefile", rel)
        self.assertTrue(out["exists"])
        self.assertFalse(out["ambiguous"])
        self.assertEqual({c["file"] for c in out["citations"]}, {"notes.md", "README"})

    def test_citation_lines_reports_the_line(self):
        rel = self._corpus()
        hits = citation_lines(self.root, "Makefile", rel)
        self.assertEqual({h["file"] for h in hits}, {"notes.md", "README"})
        self.assertTrue(all(h["kind"] == "basename" for h in hits))


if __name__ == "__main__":
    unittest.main()
