# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""inventory.py — corpus scan."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from helpers import TempCase, write
from slopkeeper.inventory import collect


class TestInventory(TempCase):
    def test_counts_and_skips_hidden(self):
        inv = collect(self.root)
        paths = {f["path"] for f in inv["files"]}
        self.assertEqual(len(paths), 8)
        self.assertNotIn(".obsidian/workspace.json", paths)
        self.assertIn("docs/canonical.md", paths)
        self.assertTrue(inv["stats"]["loose_top_level_files"] >= 5)

    def test_head_and_title(self):
        inv = collect(self.root)
        act = next(f for f in inv["files"] if f["path"] == "acme_dossier.md")
        self.assertEqual(act["title"], "Acme dossier")
        self.assertIn("flows", act["head"])


def _minimal_pdf(text: str) -> bytes:
    """A small valid single-page PDF with one text run — enough for
    pdftotext to extract `text`."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return out


class TestBinaryHeads(TempCase):
    """Phase 11: binary heads are loud — extracted text or the marker."""

    def test_binary_no_signal_marker(self):
        # Extractor forced absent: the literal marker, never a silent
        # None — for extractor-less exts too.
        (self.root / "doc.pdf").write_bytes(b"%PDF-1.4 not really")
        (self.root / "img.png").write_bytes(b"\x89PNG fake")
        with unittest.mock.patch("slopkeeper.inventory._which", return_value=None):
            inv = collect(self.root, fast=True)
        by_path = {f["path"]: f for f in inv["files"]}
        self.assertEqual(
            by_path["doc.pdf"]["head"], "signal: none (binary, no extractor)"
        )
        self.assertEqual(
            by_path["img.png"]["head"], "signal: none (binary, no extractor)"
        )
        self.assertIsNone(by_path["img.png"]["title"])

    def test_extractor_failure_still_loud(self):
        # Extractor present, content garbage: non-zero exit → marker.
        (self.root / "doc.pdf").write_bytes(b"not a pdf at all")
        with unittest.mock.patch(
            "slopkeeper.inventory._which", return_value="/usr/bin/pdftotext"
        ):
            inv = collect(self.root, fast=True)
        entry = next(f for f in inv["files"] if f["path"] == "doc.pdf")
        self.assertEqual(entry["head"], "signal: none (binary, no extractor)")

    @unittest.skipIf(shutil.which("pdftotext") is None, "pdftotext not installed")
    def test_pdf_head_with_extractor(self):
        (self.root / "doc.pdf").write_bytes(_minimal_pdf("Hello inventory extraction"))
        inv = collect(self.root, fast=True)
        entry = next(f for f in inv["files"] if f["path"] == "doc.pdf")
        self.assertIn("Hello inventory extraction", entry["head"])
        self.assertEqual(entry["title"], "Hello inventory extraction")

    def test_references_detected(self):
        inv = collect(self.root)
        refs = inv["references"]
        self.assertIn("WATCHLIST.md", refs["acme_dossier.md"])
        self.assertIn("WATCHLIST.md", refs["globex_dossier.md"])

    def test_references_are_path_keyed(self):
        # Phase 2: a path-form citation keys the cited file's full path,
        # not its basename.
        write(self.root / "docs/guide.md", "runbook: spark-glm-5.3/README.md\n")
        inv = collect(self.root)
        self.assertIn("docs/guide.md", inv["references"]["spark-glm-5.3/README.md"])

    def test_fast_skips_references(self):
        inv = collect(self.root, fast=True)
        self.assertNotIn("references", inv)


# --- _kind: extension table + NUL sniff for unknown extensions ---


class TestKind(unittest.TestCase):
    def test_kind_sniffs_nul_bytes_in_unknown_ext(self):
        from slopkeeper.inventory import _kind

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            blob = tmp / "blob.dat"
            blob.write_bytes(b"looks like text\x00but has a NUL")
            self.assertEqual(_kind(blob), "binary")
            plain = tmp / "plain.dat"
            plain.write_bytes(b"just text, no NUL")
            self.assertEqual(_kind(plain), "text")

    def test_kind_extension_table(self):
        from slopkeeper.inventory import _kind

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self.assertEqual(_kind(_w(tmp, "s.py")), "code")
            self.assertEqual(_kind(_w(tmp, "d.md")), "doc")
            self.assertEqual(_kind(_w(tmp, "p.png")), "binary")


def _w(tmp: Path, name: str) -> Path:
    p = tmp / name
    p.write_bytes(b"x=1\n")
    return p


if __name__ == "__main__":
    unittest.main()
