# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""kinds.py — one table, one answer per extension (P3/R7 acceptance).

The bug class this file exists to prevent is not a wrong branch, it is
five modules each keeping a private list of "what is binary" / "what is
code" and drifting on independent schedules: `similar.identical_groups`
hashing `.webp`/`.tar`/`.mp3`/`.db` copies inventory called binary,
`anchors` maintaining a byte-for-byte copy of inventory's code list,
`.sql` being "data" to `propose` and "code" to `inventory` in the same
run. Those copies are gone; what is left is the guarantee, tested from
the OUTSIDE — every consumer's public answer must be the table's answer.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from helpers import TempCase

from slopkeeper import anchors, inventory, kinds, propose, references, similar


def _ext_of(ext: str) -> str:
    return ext if ext.startswith(".") else "." + ext


ALL_EXTS = sorted(
    kinds.BINARY_EXTS
    | kinds.CODE_EXTS
    | kinds.DOC_EXTS
    | kinds.TEXT_EXTS
    | set().union(*kinds.KIND_EXTS.values())
)


class TestOneOwner(unittest.TestCase):
    """The names consumers import are aliases of the same objects."""

    def test_inventory_aliases(self):
        self.assertIs(inventory.BINARY_EXTS, kinds.BINARY_EXTS)
        self.assertIs(inventory.CODE_EXTS, kinds.CODE_EXTS)

    def test_anchors_code_list_is_the_one_code_list(self):
        # anchors.CODE_EXTS used to be a copy; a copy drifts waiting to
        # happen, which is the whole of R7.
        self.assertIs(anchors.CODE_EXTS, kinds.CODE_EXTS)

    def test_references_text_list_is_the_one_text_list(self):
        self.assertIs(references.TEXT_EXTS, kinds.TEXT_EXTS)

    def test_propose_buckets_are_the_one_buckets(self):
        self.assertIs(propose.KIND_EXTS, kinds.KIND_EXTS)

    def test_similar_carries_no_private_binary_list(self):
        # The inline nine-extension tuple is gone: the duplicate
        # detector asks the table.
        import inspect

        src = inspect.getsource(similar.identical_groups)
        self.assertNotIn('".png"', src)
        self.assertIn("kinds.is_binary", src)


class TestTableShape(unittest.TestCase):
    """Structural invariants the five old tables could not enforce."""

    def test_kind_axis_has_one_answer_per_extension(self):
        for ext in ALL_EXTS:
            answers = [
                ext in kinds.BINARY_EXTS,
                ext in kinds.CODE_EXTS,
                ext in kinds.DOC_EXTS,
            ]
            self.assertLessEqual(
                sum(answers),
                1,
                f"{ext} appears in more than one kind table",
            )

    def test_binary_is_never_scannable(self):
        overlap = kinds.BINARY_EXTS & kinds.TEXT_EXTS
        self.assertEqual(overlap, set())

    def test_code_and_doc_are_scannable(self):
        self.assertTrue(kinds.CODE_EXTS <= kinds.TEXT_EXTS)
        self.assertTrue(kinds.DOC_EXTS <= kinds.TEXT_EXTS)

    def test_propose_buckets_are_disjoint(self):
        names = list(kinds.KIND_EXTS)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                self.assertEqual(
                    kinds.KIND_EXTS[a] & kinds.KIND_EXTS[b],
                    set(),
                    f"{a} and {b} both claim an extension",
                )

    def test_propose_code_bucket_is_the_code_table(self):
        self.assertIs(kinds.KIND_EXTS["code"], kinds.CODE_EXTS)

    def test_sql_is_answered_once(self):
        # The audit's concrete drift: `.sql` was data to propose and
        # code to inventory in the same run.
        self.assertNotIn(".sql", kinds.KIND_EXTS["data"])
        self.assertIn(".sql", kinds.KIND_EXTS["code"])
        self.assertEqual(inventory._kind(Path("a.sql")), "code")

    # The routing buckets name a few formats the KIND table does not:
    # they are zip/XML containers the NUL sniff decides, and they are
    # listed here so a new one is a decision, not an oversight. (`.svg`
    # is the interesting case — XML that cites things, which is why P5
    # promotes it to a table entry rather than leaving it to the sniff.)
    ROUTING_ONLY_EXTS = {".docx", ".pptx", ".svg"}

    def test_only_documented_extensions_live_outside_the_kind_table(self):
        bucketed = set().union(*kinds.KIND_EXTS.values())
        kinded = kinds.BINARY_EXTS | kinds.CODE_EXTS | kinds.DOC_EXTS | kinds.TEXT_EXTS
        self.assertEqual(bucketed - kinded, self.ROUTING_ONLY_EXTS)

    def test_kind_table_members_are_all_routed(self):
        # A format the kind table knows but no candidate layout files is
        # allowed (nobody wants a `scripts/` for object files), but it
        # has to be a KNOWN gap: adding an extension to a kind table
        # without deciding its route fails here, which is the drift this
        # module exists to stop.
        bucketed = set().union(*kinds.KIND_EXTS.values())
        unrouted = (kinds.BINARY_EXTS | kinds.TEXT_EXTS) - bucketed - kinds.CODE_EXTS
        self.assertEqual(
            unrouted,
            {
                ".7z",
                ".bib",
                ".bz2",
                ".css",
                ".db",
                ".dylib",
                ".exe",
                ".ipynb",
                ".mov",
                ".pyc",
                ".scss",
                ".so",
                ".sqlite",
                ".ttf",
                ".woff",
                ".woff2",
                ".xz",
            },
            "the router and the kind table disagree about what exists",
        )


class TestClassify(TempCase):
    """`classify` is what the consumers actually ask."""

    def _file(self, rel: str, data: bytes):

        p = self.root / "probe" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_table_answers_without_touching_the_disk(self):
        # A path that does not exist still gets the table's answer:
        # classification is a lookup, not a read, whenever the extension
        # speaks.
        self.assertEqual(kinds.classify(self.root / "nope/x.png"), "binary")
        self.assertEqual(kinds.classify(self.root / "nope/x.py"), "code")
        self.assertEqual(kinds.classify(self.root / "nope/x.md"), "doc")
        self.assertEqual(kinds.classify(self.root / "nope/x.json"), "text")

    def test_unknown_extension_sniffs(self):
        text = self._file("script.dart", b"void main() {}\n")
        binary = self._file("blob.dat", b"\x00\x01\x02\x03")
        self.assertEqual(kinds.classify(text), "text")
        self.assertEqual(kinds.classify(binary), "binary")

    def test_unreadable_is_binary(self):
        # inventory's `_kind` made this call first, and it is the safe
        # direction: a name that cannot be read cannot cite anything.
        p = self._file("weird", b"plain text but missing later\n")
        p.unlink()
        self.assertEqual(kinds.classify(p), "binary")
        self.assertFalse(references.is_text_file(p))

    def test_sniff_false_means_table_only(self):
        self.assertEqual(kinds.classify(self.root / "x.qqq", sniff=False), "text")

    def test_pre_read_head_avoids_the_open(self):
        p = self._file("thing.qqq", b"\x00binary\n")
        self.assertEqual(kinds.classify(p, head=b"\x00binary\n"), "binary")
        p.unlink()  # the head is what decides, not the disk
        self.assertEqual(kinds.classify(p, head=b"hello\n"), "text")

    def test_every_table_member_classifies_as_expected(self):
        for ext in kinds.BINARY_EXTS:
            self.assertEqual(kinds.classify("a" + _ext_of(ext)), "binary")
        for ext in kinds.CODE_EXTS:
            self.assertEqual(kinds.classify("a" + _ext_of(ext)), "code")
        for ext in kinds.DOC_EXTS:
            self.assertEqual(kinds.classify("a" + _ext_of(ext)), "doc")
        tabled = kinds.BINARY_EXTS | kinds.CODE_EXTS | kinds.DOC_EXTS
        for ext in kinds.TEXT_EXTS - tabled:
            self.assertEqual(kinds.classify("a" + _ext_of(ext)), "text")


class TestConsumersAgree(TempCase):
    """The acceptance bar: one canonical answer per extension, per consumer."""

    def test_inventory_kind_is_classify(self):
        for ext in ALL_EXTS:
            p = self.root / f"probe{ext}"
            self.assertEqual(
                inventory._kind(p),
                kinds.classify(p),
                f"inventory disagrees with the table on {ext}",
            )

    def test_references_scannability_is_not_binary(self):
        for ext in ALL_EXTS:
            p = self.root / f"probe{ext}"
            self.assertEqual(
                references.is_text_file(p),
                kinds.classify(p) != "binary",
                f"the citation scan and inventory disagree on {ext}",
            )

    def test_anchors_reads_exactly_the_code_table(self):
        for ext in ALL_EXTS:
            self.assertEqual(
                kinds.is_code(f"a{ext}"),
                ext in kinds.CODE_EXTS,
            )

    def test_propose_routes_each_extension_once(self):
        for ext in ALL_EXTS:
            hits = [b for b, exts in kinds.KIND_EXTS.items() if ext in exts]
            self.assertLessEqual(len(hits), 1, f"{ext} routes two ways")

    def test_identical_groups_skips_what_inventory_calls_binary(self):
        # The audit's row verbatim: the inline list hashed .webp/.tar/
        # .mp3/.db copies anyway. One pair per previously-missed
        # extension, plus a text pair that must still be found.

        missed = [".webp", ".tar", ".mp3", ".db", ".sqlite", ".pyc"]
        paths = []
        for i, ext in enumerate(missed):
            for name in (f"dup{i}{ext}", f"dup{i}_copy{ext}"):
                fp = self.root / name
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(b"same bytes")
                paths.append(fp.relative_to(self.root).as_posix())
        (self.root / "twin.md").write_bytes(b"prose")
        (self.root / "twin_copy.md").write_bytes(b"prose")
        groups = similar.identical_groups(
            self.root, sorted(paths + ["twin.md", "twin_copy.md"])
        )
        self.assertEqual(groups, [["twin.md", "twin_copy.md"]])


if __name__ == "__main__":
    unittest.main()
