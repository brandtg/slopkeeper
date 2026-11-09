# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""anchors.py — directories built in code (Phase 9).

join()/Path() literals and __file__ parent-chain math, resolved against
the Phase 2 dirs index: a literal that does not name an existing corpus
dir anchors nothing, which is what keeps the flag quiet.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from helpers import TempCase, write
from slopkeeper.anchors import anchored_move_risks, scan_code_anchors
from slopkeeper.ignore import corpus_walk
from slopkeeper.references import CitationIndex


def _rel_paths(root: Path) -> tuple[list[Path], CitationIndex]:
    rel_paths: list[Path] = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        d = Path(dirpath)
        for fn in filenames:
            rel_paths.append((d / fn).relative_to(root))
    return rel_paths, CitationIndex(rel_paths)


class TestCodeAnchors(TempCase):
    def _scan(self, files: dict[str, str]) -> dict[str, list[dict]]:
        for rel, text in files.items():
            write(self.root / rel, text)
        rel_paths, index = _rel_paths(self.root)
        return scan_code_anchors(self.root, rel_paths, index)

    def test_join_literals_flag_dirs(self):
        # PRD checkpoint: os.path.join(ROOT, "slack", "data") anchors
        # slack AND slack/data, with file:line evidence.
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "slack/data/db.md": "x\n",
                "tools/run.py": 'DATA = os.path.join(ROOT, "slack", "data")\n',
            }
        )
        self.assertEqual(set(anchors), {"slack", "slack/data"})
        ev = anchors["slack"][0]
        self.assertEqual(ev["file"], "tools/run.py")
        self.assertEqual(ev["line"], 1)
        self.assertIn("slack", ev["text"])

    def test_path_ctor_and_joinpath_flag_dirs(self):
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "tools/run.py": 'p = Path("slack")\nq = p.joinpath("keep.txt")\n',
            }
        )
        self.assertIn("slack", anchors)
        # joinpath("keep.txt") names a FILE, not a dir — no anchor.
        self.assertEqual([e["line"] for e in anchors["slack"]], [1])

    def test_anchors_evidence_uses_2000_cap(self):
        # C6 residual (Phase 15): anchors evidence truncated at 200
        # while citation_lines moved to 2000 — a long evidence line is
        # now usable for authoring an exact find.
        long_tail = "x" * 1500
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "tools/run.py": f'DATA = os.path.join(ROOT, "slack")  # {long_tail}\n',
            }
        )
        ev = anchors["slack"][0]
        self.assertGreater(len(ev["text"]), 200)
        self.assertLessEqual(len(ev["text"]), 2000)
        self.assertIn(long_tail[:100], ev["text"])

    def test_file_parent_math(self):
        # dirname(dirname(__file__)) anchors the parent dir of the
        # file's directory; Path(__file__).parent.parent likewise.
        anchors = self._scan(
            {
                "scripts/keep.txt": "x\n",
                "scripts/tools/calc.py": (
                    "from os.path import dirname\n"
                    "ROOT = dirname(dirname(__file__))\n"
                    "BASE = Path(__file__).parent.parent\n"
                ),
            }
        )
        self.assertIn("scripts", anchors)
        self.assertEqual({e["line"] for e in anchors["scripts"]}, {2, 3})

    def test_single_dirname_anchors_own_dir(self):
        anchors = self._scan({"scripts/calc.py": "HERE = dirname(__file__)\n"})
        self.assertEqual(set(anchors), {"scripts"})

    def test_non_path_literals_ignored(self):
        # PRD checkpoint: join("a", "b") where a is not a corpus dir
        # anchors nothing — the dirs index is the false-positive kill.
        anchors = self._scan({"tools/run.py": 'x = os.path.join("a", "b")\n'})
        self.assertEqual(anchors, {})

    def test_concat_only_anchors_existing_dirs(self):
        # join("slack", "nope") references slack/<something>: slack is
        # anchored (moving it breaks the call), slack/nope is not (it
        # does not exist).
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "tools/run.py": 'p = os.path.join("slack", "nope")\n',
            }
        )
        self.assertEqual(set(anchors), {"slack"})

    def test_non_literal_arg_breaks_the_run(self):
        # join("slack", flag, "data"): a non-literal arg between the
        # literals means they do not concatenate — slack/data is not
        # built by this call and is NOT anchored; slack alone is.
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "slack/data/db.md": "x\n",
                "tools/run.py": 'p = os.path.join("slack", flag, "data")\n',
            }
        )
        self.assertEqual(set(anchors), {"slack"})

    def test_non_code_files_ignored(self):
        # A markdown note describing a join call is prose, not code.
        anchors = self._scan(
            {
                "slack/keep.txt": "x\n",
                "notes.md": 'use os.path.join(ROOT, "slack") here\n',
            }
        )
        self.assertEqual(anchors, {})

    def test_multiline_call_body(self):
        anchors = self._scan(
            {
                "slack/data/db.md": "x\n",
                "tools/run.py": (
                    'DATA = os.path.join(\n    ROOT,\n    "slack",\n    "data",\n)\n'
                ),
            }
        )
        self.assertEqual(set(anchors), {"slack", "slack/data"})
        # evidence sits at the call's opening line, wherever the
        # literals themselves land
        self.assertEqual([e["line"] for e in anchors["slack"]], [1])
        self.assertEqual([e["line"] for e in anchors["slack/data"]], [1])

    def test_non_utf8_code_file_still_scans(self):
        # decode(errors="replace") mirrors the reference scan: mojibake
        # never crashes the pass, and the literal after it still counts.
        (self.root / "tools").mkdir(parents=True)
        (self.root / "tools/run.py").write_bytes(b'\xff\xfe = join("slack")\n')
        write(self.root / "slack/keep.txt", "x\n")
        rel_paths, index = _rel_paths(self.root)
        anchors = scan_code_anchors(self.root, rel_paths, index)
        self.assertEqual([e["file"] for e in anchors["slack"]], ["tools/run.py"])


class TestAnchoredMoveRisks(TempCase):
    def test_move_of_anchored_dir_flagged(self):
        write(self.root / "slack/data/db.md", "x\n")
        write(
            self.root / "tools/run.py",
            'DATA = os.path.join(ROOT, "slack", "data")\n',
        )
        rows = anchored_move_risks(
            self.root,
            {
                "moves": [
                    {"from": "slack", "to": "archive/slack", "reason": "r"},
                    {
                        "from": "docs/canonical.md",
                        "to": "docs2/canonical.md",
                        "reason": "r",
                    },
                ]
            },
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["from"], "slack")
        self.assertEqual(rows[0]["dirs"], ["slack", "slack/data"])
        self.assertEqual(rows[0]["sites"], 2)

    def test_move_of_anchored_subdir_flagged_alone(self):
        write(self.root / "slack/data/db.md", "x\n")
        write(
            self.root / "tools/run.py",
            'DATA = os.path.join(ROOT, "slack", "data")\n',
        )
        rows = anchored_move_risks(
            self.root, {"moves": [{"from": "slack/data", "to": "data", "reason": "r"}]}
        )
        self.assertEqual([r["dirs"] for r in rows], [["slack/data"]])

    def test_file_move_inside_anchored_dir_not_flagged(self):
        # The dir itself stays put; one file leaving it breaks nothing.
        write(self.root / "slack/data/db.md", "x\n")
        write(
            self.root / "tools/run.py",
            'DATA = os.path.join(ROOT, "slack", "data")\n',
        )
        rows = anchored_move_risks(
            self.root,
            {"moves": [{"from": "slack/data/db.md", "to": "x/db.md", "reason": "r"}]},
        )
        self.assertEqual(rows, [])

    def test_no_anchors_no_rows(self):
        rows = anchored_move_risks(
            self.root,
            {"moves": [{"from": "docs", "to": "docs2", "reason": "r"}]},
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
