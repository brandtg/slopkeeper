# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""doctor.py — triage report and strategy suggestion."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from helpers import TempCase, write
from slopkeeper.doctor import doctor
from slopkeeper.inventory import collect


class TestDoctor(TempCase):
    def test_doctor_reports(self):
        d = doctor(self.root)
        self.assertEqual(d["suggested_strategy"]["id"], "hybrid")
        self.assertIn("git", d["advice"][0])
        self.assertEqual(d["next"]["action"], "inventory")


# --- strategy heuristic + advice rows ---


class TestDoctorStrategyAdvice(TempCase):
    def _clean_corpus(self) -> Path:
        # Small, tidy, folderized: n < 15 and loose_ratio < 0.3.
        clean = self.tmp / "clean-corpus"
        write(clean / "notes" / "project-a.md", "# A\n")
        write(clean / "notes" / "project-b.md", "# B\n")
        write(clean / "admin" / "todo.md", "# C\n")
        return clean

    def test_small_clean_corpus_suggests_skip(self):
        d = doctor(self._clean_corpus())
        self.assertEqual(d["suggested_strategy"]["id"], "skip")
        self.assertIn("tidy", d["suggested_strategy"]["why"])

    def test_non_git_advice_appears_then_clears(self):
        clean = self._clean_corpus()
        d = doctor(clean)
        self.assertTrue(any("git init" in a for a in d["advice"]))
        (clean / ".git").mkdir()
        self.assertEqual(doctor(clean)["advice"], [])


# --- strategy heuristic branches ---


class TestNonRegularFiles(TempCase):
    """B6: devices/FIFOs are named by doctor, invisible to every walk."""

    def test_doctor_names_non_regular_files(self):
        import os as _os

        dev = self.root / ".claude" / "loop.md"
        dev.parent.mkdir(parents=True, exist_ok=True)
        _os.mkfifo(dev)
        data = doctor(self.root)
        named = [x["path"] for x in data["non_regular_files"]]
        self.assertIn(".claude/loop.md", named)
        self.assertTrue(
            any("non-regular" in a for a in data["advice"]),
            data["advice"],
        )


class TestDoctorStrategies(TempCase):
    def test_skip_strategy_small_and_tidy(self):
        root = self.tmp / "tidy"
        for name in ("one.md", "two.md", "three.md", "four.md"):
            write(root / "notes" / name, "# note\n")
        d = doctor(root)
        self.assertEqual(d["suggested_strategy"]["id"], "skip")

    def test_fallback_hybrid_mixed_corpus(self):
        # 36 files, 21 loose (ratio 0.58): no earlier branch fires, so
        # the default hybrid applies — and the loose-file advice shows.
        root = self.tmp / "mixed"
        for i in range(21):
            write(root / f"loose-{i}.md", "# loose\n")
        for d in ("alpha", "beta", "gamma"):
            for i in range(5):
                write(root / d / f"f{i}.md", "# f\n")
        d = doctor(root)
        self.assertEqual(d["suggested_strategy"]["id"], "hybrid")
        self.assertIn("loose files at top level", " ".join(d["advice"]))

    def test_topics_strategy_large_corpus(self):
        # >2000 files with a loose ratio under 0.6: the topics branch.
        root = self.tmp / "large"
        for i in range(900):
            write(root / f"loose-{i}.md", "# loose\n")
        for d in ("a", "b"):
            for i in range(600):
                write(root / d / f"f{i}.md", "# f\n")
        d = doctor(root)
        self.assertEqual(d["suggested_strategy"]["id"], "topics")
        self.assertTrue(any("800" in a for a in d["advice"]))


class TestDoctorNestedCorpora(TempCase):
    def test_nested_corpus_flagged(self):
        # Phase 7: a subtree with its own manifest is a separate corpus
        # — doctor names it in the data AND the advice, never silently.
        (self.root / "sub" / ".slopkeeper").mkdir(parents=True)
        (self.root / "sub" / ".slopkeeper" / "manifest.json").write_text("{}")
        d = doctor(self.root)
        self.assertEqual(d["nested_corpora"], ["sub"])
        self.assertIn("sub", " ".join(d["advice"]))
        self.assertIn(".slopkeeper/manifest.json", " ".join(d["advice"]))

    def test_no_nested_corpus_no_flag(self):
        d = doctor(self.root)
        self.assertEqual(d["nested_corpora"], [])


class TestDoctorCodeAnchors(TempCase):
    def test_code_anchors_reported(self):
        # Phase 9: dirs built in code are doctor output — top by
        # evidence count, with file:line rows, plus an advice line.
        write(self.root / "slack/data/db.md", "x\n")
        write(
            self.root / "tools/run.py",
            'DATA = os.path.join(ROOT, "slack", "data")\n',
        )
        d = doctor(self.root)
        dirs = [a["dir"] for a in d["code_anchors"]]
        self.assertIn("slack", dirs)
        self.assertIn("slack/data", dirs)
        slack_row = next(a for a in d["code_anchors"] if a["dir"] == "slack")
        self.assertEqual(slack_row["refs"], 1)
        self.assertEqual(slack_row["evidence"][0]["file"], "tools/run.py")
        self.assertTrue(any("built in code" in a for a in d["advice"]))

    def test_no_code_anchors_empty_field(self):
        d = doctor(self.root)
        self.assertEqual(d["code_anchors"], [])
        self.assertFalse(any("built in code" in a for a in d["advice"]))


class TestDoctorPhase11(TempCase):
    """Phase 11: the hand-rolled surveys become doctor output."""

    def test_truncation_culprit_named(self):
        # >8000 files with one heavy dir: the culprit is named from
        # per-top-dir counts taken past the cap, not guessed.
        for i in range(6000):
            write(self.root / f"big/f{i:05d}.md", "# x\n")
        for i in range(2010):
            write(self.root / f"small/f{i:05d}.md", "# x\n")
        d = doctor(self.root)
        self.assertTrue(d["stats"]["truncated"])
        self.assertGreaterEqual(d["top_dirs"].get("big", 0), 6000)
        self.assertTrue(any("big/" in a for a in d["advice"]))
        self.assertTrue(any("walk cap" in a for a in d["advice"]))

    def test_nested_repo_suggested_lines(self):
        # A nested repo's files ARE content since fp4 (only .git is
        # invisible) — detected, with a copy-paste ignore block that
        # silences the warning once copied.
        (self.root / "vendored/repo/.git").mkdir(parents=True)
        write(self.root / "vendored/repo/f.txt", "x\n")
        d = doctor(self.root)
        self.assertIn("vendored/repo", d["nested_repos"])
        self.assertIn("vendored/repo/", d["suggested_ignore_lines"])
        self.assertTrue(any("Nested git repo" in a for a in d["advice"]))
        write(self.root / ".slopkeeperignore", "vendored/repo/\n")
        d2 = doctor(self.root)
        self.assertNotIn("vendored/repo", d2["nested_repos"])
        self.assertEqual(d2["suggested_ignore_lines"], [])

    def test_ignore_stranding_warning(self):
        # The copied-.gitignore trap: ignored and visible files in one
        # folder — the tool sees only the visible half.
        write(self.root / ".slopkeeperignore", "*.log\n")
        write(self.root / "misc/keep.md", "visible\n")
        write(self.root / "misc/drop.log", "ignored\n")
        d = doctor(self.root)
        by_dir = {s["dir"]: s for s in d["ignore_stranding"]}
        self.assertIn("misc", by_dir)
        self.assertEqual(by_dir["misc"]["ignored"], 1)
        self.assertEqual(by_dir["misc"]["visible"], 1)
        self.assertTrue(any("not a" in a and "gitignore" in a for a in d["advice"]))

    def test_dotdir_cache_heuristic(self):
        # .jsonl-heavy dot dir → suggested ignore lines; a small one
        # stays below the heuristic.
        for i in range(12):
            write(self.root / f".cache/dump{i}.jsonl", "{}\n")
        for i in range(4):
            write(self.root / f".small/s{i}.jsonl", "{}\n")
        d = doctor(self.root)
        dirs = [c["dir"] for c in d["dotdir_cache_candidates"]]
        self.assertIn(".cache", dirs)
        self.assertNotIn(".small", dirs)
        self.assertTrue(any(".cache/" in a for a in d["advice"]))

    def test_stranding_counts_exclude_tool_layer_entries(self):
        # The visibility survey is the corpus walk now, so it excludes
        # what every other surface excludes: the ignore file itself
        # (counting it had doctor reporting "dangling citations" inside
        # .slopkeeperignore) and a symlink to nothing (no walk can read
        # it, so it is not a visible file). Both used to be counted by
        # the survey's private os.walk.
        write(self.root / ".slopkeeperignore", "*.log\n# node_modules/\n")
        write(self.root / "misc/keep.md", "visible\n")
        write(self.root / "misc/drop.log", "ignored\n")
        os.symlink("nowhere.md", self.root / "misc" / "dangling.md")
        d = doctor(self.root)
        by_dir = {s["dir"]: s for s in d["ignore_stranding"]}
        self.assertEqual(by_dir["misc"], {"dir": "misc", "ignored": 1, "visible": 1})
        tokens = [
            row["token"]
            for row in d["dangling_citations"]["sample"]
            if row["file"] == ".slopkeeperignore"
        ]
        self.assertEqual(tokens, [])

    def test_surveys_agree_with_the_inventory_stats(self):
        # One walk, one truth: what doctor's surveys name is what
        # inventory named, from the same records (they used to be two
        # walks that could disagree).
        write(self.root / "sub/.slopkeeper/manifest.json", "{}")
        write(self.root / "sub/inner.md", "# inner\n")
        os.mkfifo(self.root / "pipe")
        d = doctor(self.root)
        self.assertEqual(d["nested_corpora"], ["sub"])
        self.assertEqual([x["path"] for x in d["non_regular_files"]], ["pipe"])
        self.assertEqual(d["stats"]["nested_corpora"], ["sub"])
        self.assertNotIn(
            "sub/inner.md",
            [f["path"] for f in collect(self.root, fast=True)["files"]],
        )

    def test_dangling_citations_flagged(self):
        write(self.root / "guide.md", "see docs/gone.md and [[missing/link]]\n")
        d = doctor(self.root)
        self.assertGreaterEqual(d["dangling_citations"]["count"], 2)
        tokens = {s["token"] for s in d["dangling_citations"]["sample"]}
        self.assertIn("docs/gone.md", tokens)


if __name__ == "__main__":
    unittest.main()
