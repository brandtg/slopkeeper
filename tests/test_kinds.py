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
    # they are zip containers the NUL sniff decides, and they are listed
    # here so a new one is a decision, not an oversight. (`.svg` used to be
    # the interesting case here — XML that cites things — which is exactly
    # why P5's filetype batch promoted it into TEXT_EXTS instead of leaving
    # it to the sniff.)
    ROUTING_ONLY_EXTS = {".docx", ".pptx"}

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


class TestNameTable(unittest.TestCase):
    """P5 filetype batch: the files with no extension, answered once.

    `Makefile`, `Dockerfile`, `LICENSE`, `README`, `.env` — no extension
    means the extension tables are silent, and the OLD `is_text_file` was
    extension-gated alone, so the reference scan could not see them while
    inventory's NUL sniff called them text. One name table answers both
    questions, and the ordering rule (extension first, name only when the
    extension said nothing) is what keeps every existing answer intact.
    """

    CODE_NAMES = [
        "Makefile",
        "makefile",
        "Dockerfile",
        "Justfile",
        "Gemfile",
        "Rakefile",
        "Vagrantfile",
        "Jenkinsfile",
        "Procfile",
        "Brewfile",
    ]
    TEXT_NAMES = [
        ".gitignore",
        ".dockerignore",
        ".npmignore",
        ".gitattributes",
        ".editorconfig",
        ".shellcheckrc",
        ".env",
        ".env.local",
        ".env.production",
        "LICENSE",
        "LICENSE-MIT",
        "LICENCE",
    ]
    DOC_NAMES = [
        "README",
        "CHANGELOG",
        "CONTRIBUTING",
        "code_of_conduct.md".replace(".md", ""),
    ]

    def test_build_files_are_code(self):
        for name in self.CODE_NAMES:
            with self.subTest(name=name):
                self.assertEqual(kinds.kind_of(name), "code")
                self.assertTrue(kinds.is_code(name))

    def test_config_and_license_files_are_text(self):
        for name in self.TEXT_NAMES:
            with self.subTest(name=name):
                self.assertEqual(kinds.kind_of(name), "text")

    def test_prose_classics_are_doc(self):
        for name in self.DOC_NAMES:
            with self.subTest(name=name):
                self.assertEqual(kinds.kind_of(name), "doc")

    def test_all_name_table_entries_are_scannable_and_citable(self):
        names = self.CODE_NAMES + self.TEXT_NAMES + self.DOC_NAMES
        for name in names:
            with self.subTest(name=name):
                # The two answers the audit needed: readable as text (so it
                # can HOLD a citation) and vouched-for as a name (so it can
                # BE one) — no file on disk required for either.
                self.assertTrue(kinds.is_scannable(name))
                self.assertTrue(kinds.is_citable_name(name))

    def test_the_extension_still_wins_when_it_speaks(self):
        # The ordering rule, pinned: `README.txt` is text because .txt says
        # so, not because the name does; nothing that already had an answer
        # changed when the name table landed.
        self.assertEqual(kinds.kind_of("README.txt"), "text")
        self.assertEqual(kinds.kind_of("Makefile.md"), "doc")
        self.assertEqual(kinds.kind_of("LICENSE.png"), "binary")

    def test_prose_words_are_not_name_table_members(self):
        # `NOTICE` / `AUTHORS` / `NEWS` are deliberately OUT of the table:
        # basenames match case-INsensitively, so admitting them would make
        # the ordinary word in a sentence a citation of the file.
        for name in (
            "NOTICE",
            "AUTHORS",
            "NEWS",
            "readmelater",
            "licenses",
            "environment",
            "coding",
        ):
            with self.subTest(name=name):
                self.assertIsNone(kinds.name_kind(name))
                self.assertFalse(kinds.is_citable_name(name))

    def test_prefix_families_use_a_boundary(self):
        # The boundary is what keeps `.environment` from being a `.env`
        # while `.env.production` is one, and `LICENSEMIT` out while
        # `LICENSE-MIT` is in.
        self.assertEqual(kinds.kind_of(".env.production"), "text")
        self.assertIsNone(kinds.name_kind(".environment"))
        self.assertEqual(kinds.kind_of("LICENSE-MIT"), "text")
        self.assertIsNone(kinds.name_kind("LICENSEMIT"))

    def test_route_of_sends_the_classics_to_a_real_folder(self):
        # Before P5 every one of these fell through the extension buckets
        # and landed in `misc/` beside the unidentified pile.
        expected = {
            "Makefile": "code",
            "Dockerfile": "code",
            "Gemfile": "code",
            ".gitignore": "data",
            ".env": "data",
            "LICENSE": "docs",
            "README": "docs",
            "CHANGELOG": "docs",
        }
        for name, bucket in expected.items():
            with self.subTest(name=name):
                self.assertEqual(kinds.route_of(name), bucket)

    def test_route_of_agrees_with_the_buckets_for_real_extensions(self):
        # The router is one table whichever way it is asked: an extension in
        # a bucket routes to that bucket, and nothing routes twice.
        for bucket, exts in kinds.KIND_EXTS.items():
            for ext in sorted(exts):
                with self.subTest(bucket=bucket, ext=ext):
                    self.assertEqual(kinds.route_of(f"a{ext}"), bucket)
        self.assertIsNone(kinds.route_of("a.qqq"))

    def test_the_name_axis_has_one_owner_too(self):
        # R7's rule applied to the new axis: a NAME is not an extension, and
        # a name that WOULD be read as an extension means someone added an
        # extension to the wrong table.
        exts = kinds.BINARY_EXTS | kinds.CODE_EXTS | kinds.DOC_EXTS | kinds.TEXT_EXTS
        self.assertEqual(set(kinds.NAME_KINDS) & exts, set())
        self.assertEqual({p for p, _ in kinds.NAME_PREFIXES} & exts, set())

    def test_every_routed_name_is_a_known_name(self):
        # Routing and kinding share the name list: a routed name the kind
        # table does not vouch for would be routed but never scannable,
        # which is the same drift in a new costume.
        for name in kinds.NAME_ROUTING:
            with self.subTest(name=name):
                self.assertIsNotNone(
                    kinds.name_kind(name),
                    f"{name} routes but no kind table knows it",
                )

    def test_no_name_is_a_prose_word_the_matcher_would_confuse(self):
        # Defense against growing the table into prose: every entry must be
        # something nobody writes in a sentence mid-stream.
        for name in list(kinds.NAME_KINDS) + [p for p, _ in kinds.NAME_PREFIXES]:
            with self.subTest(name=name):
                self.assertGreaterEqual(len(name), 4)
                self.assertEqual(name, name.lower())


class TestOfficeContainers(unittest.TestCase):
    """P5 filetype batch: containers named in the table, not sniffed."""

    def test_office_and_epub_are_binary_by_table(self):
        for ext in (".odt", ".ods", ".odp", ".epub", ".doc", ".xls"):
            with self.subTest(ext=ext):
                self.assertEqual(kinds.kind_of(f"a{ext}"), "binary")
                # Table answer, no open: the path does not have to exist.
                self.assertEqual(
                    kinds.classify(Path("/nonexistent") / f"a{ext}"), "binary"
                )
                self.assertFalse(kinds.is_scannable(f"a{ext}"))

    def test_office_formats_route_to_docs(self):
        for ext in (".odt", ".ods", ".odp", ".epub", ".doc", ".xls"):
            with self.subTest(ext=ext):
                self.assertIn(ext, kinds.DOCS_EXTS)
                self.assertEqual(kinds.route_of(f"a{ext}"), "docs")

    def test_svg_is_text_by_table(self):
        # "SVG explicitly text (XML, cites things)": it used to be a
        # routing-only entry the NUL sniff decided, which made its
        # `xlink:href` paths invisible to the citation scan whenever the
        # sniff was skipped.
        self.assertIn(".svg", kinds.TEXT_EXTS)
        self.assertEqual(kinds.kind_of("diagram.svg"), "text")
        self.assertTrue(kinds.is_scannable("diagram.svg"))
        self.assertEqual(kinds.route_of("diagram.svg"), "media")


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
