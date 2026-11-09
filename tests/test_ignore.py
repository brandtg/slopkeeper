# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`.slopkeeperignore` — the per-corpus invisible set (DESIGN.md §6.6).

One rule everywhere: ignored = invisible. The matcher, the walk, the
fingerprint, the validator and the tending loop are all pinned here.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from helpers import TempCase, expect_code, write, write_plan

from slopkeeper.apply import apply_plan
from slopkeeper.doctor import doctor
from slopkeeper.draft import TOOL_LAYER, _file_entries
from slopkeeper.ignore import (
    IGNORE_NAME,
    CorpusIgnore,
    corpus_walk,
    load_ignore,
    nested_corpora,
)
from slopkeeper.inventory import collect
from slopkeeper.models import SlopKeeperError, corpus_fingerprint
from slopkeeper.plan import validate_plan
from slopkeeper.status import status as corpus_status
from slopkeeper.status import write_manifest


def plan_dict(root, moves, **extra):
    """A minimal plan payload for validate_plan (no fingerprint gate here)."""
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "moves": moves,
    }
    plan.update(extra)
    return plan


class TestMatcher(unittest.TestCase):
    """The pattern language: gitignore's core rules, nothing more."""

    def test_basename_pattern_matches_at_any_depth(self):
        ig = CorpusIgnore(["*.log"])
        self.assertTrue(ig.is_ignored("debug.log"))
        self.assertTrue(ig.is_ignored("a/b/c.log"))
        self.assertFalse(ig.is_ignored("c.md"))

    def test_dir_only_pattern_swallows_subtree_at_any_depth(self):
        ig = CorpusIgnore(["build/"])
        self.assertTrue(ig.is_ignored("build", is_dir=True))
        self.assertTrue(ig.is_ignored("build/x.md"))
        self.assertTrue(ig.is_ignored("a/build/deep/y.md"))
        self.assertFalse(ig.is_ignored("builds/x.md"))
        self.assertFalse(ig.is_ignored("build", is_dir=False))

    def test_bare_dir_pattern_matches_dir_and_contents(self):
        # No trailing slash: gitignore matches files AND directories.
        ig = CorpusIgnore(["scratch"])
        self.assertTrue(ig.is_ignored("scratch", is_dir=True))
        self.assertTrue(ig.is_ignored("scratch/notes.md"))

    def test_leading_slash_anchors_to_root(self):
        ig = CorpusIgnore(["/docs/draft.md"])
        self.assertTrue(ig.is_ignored("docs/draft.md"))
        self.assertFalse(ig.is_ignored("other/docs/draft.md"))

    def test_interior_slash_anchors_too(self):
        ig = CorpusIgnore(["docs/sub"])
        self.assertTrue(ig.is_ignored("docs/sub", is_dir=True))
        self.assertTrue(ig.is_ignored("docs/sub/x.md"))
        self.assertFalse(ig.is_ignored("docs/other/x.md"))

    def test_doublestar_segment_spans_directories(self):
        ig = CorpusIgnore(["logs/**", "assets/**/*.png"])
        self.assertTrue(ig.is_ignored("logs/a/b.txt"))
        self.assertTrue(ig.is_ignored("logs/x.txt"))
        self.assertTrue(ig.is_ignored("assets/x/y.png"))
        self.assertFalse(ig.is_ignored("assets/x/y.jpg"))

    def test_single_star_does_not_cross_slash(self):
        ig = CorpusIgnore(["/docs/*.md"])
        self.assertTrue(ig.is_ignored("docs/a.md"))
        self.assertFalse(ig.is_ignored("docs/sub/a.md"))

    def test_comments_and_blank_lines_are_skipped(self):
        ig = CorpusIgnore(["# a comment", "", "   ", "*.tmp"])
        self.assertTrue(ig.is_ignored("x.tmp"))
        self.assertFalse(ig.is_ignored("# a comment"))

    def test_negation_is_not_supported_v1(self):
        # Documented gap (DESIGN §6.6): a leading '!' is a literal
        # fnmatch character until a real corpus asks for negation.
        ig = CorpusIgnore(["!keep.log"])
        self.assertFalse(ig.is_ignored("keep.log"))

    def test_wildcard_question_mark(self):
        ig = CorpusIgnore(["notes?.md"])
        self.assertTrue(ig.is_ignored("notes1.md"))
        self.assertFalse(ig.is_ignored("notes10.md"))

    def test_anchored_question_mark_stays_in_segment(self):
        ig = CorpusIgnore(["/docs/?.md"])
        self.assertTrue(ig.is_ignored("docs/a.md"))
        self.assertFalse(ig.is_ignored("docs/sub/a.md"))

    def test_anchored_dir_only_pattern(self):
        ig = CorpusIgnore(["/archive/"])
        self.assertTrue(ig.is_ignored("archive", is_dir=True))
        self.assertTrue(ig.is_ignored("archive/x.md"))
        self.assertFalse(ig.is_ignored("other/archive/x.md"))

    def test_bare_slash_line_is_skipped(self):
        ig = CorpusIgnore(["/"])
        self.assertFalse(ig.active())

    def test_missing_file_is_an_empty_ignore(self):
        ig = load_ignore("/nonexistent-root")
        self.assertFalse(ig.active())
        self.assertFalse(ig.is_ignored("anything"))


class TestCorpusWalk(TempCase):
    def test_ignored_subtree_is_never_descended(self):
        (self.root / "logs/deep").mkdir(parents=True)
        (self.root / "logs/x.log").write_text("log")
        (self.root / "logs/deep/a.md").write_text("deep")
        (self.root / "notes").mkdir()
        (self.root / "notes/b.md").write_text("b")
        (self.root / IGNORE_NAME).write_text("logs/\n")

        seen_files = set()
        for dirpath, _, filenames in corpus_walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            for fn in filenames:
                seen_files.add(f"{rel_dir}/{fn}" if rel_dir != "." else fn)
        self.assertNotIn("logs/x.log", seen_files)
        self.assertNotIn("logs/deep/a.md", seen_files)
        self.assertIn("notes/b.md", seen_files)

    def test_basename_pattern_prunes_files_everywhere(self):
        (self.root / "sub").mkdir()
        (self.root / "debug.log").write_text("a")
        (self.root / "sub/c.log").write_text("b")
        (self.root / IGNORE_NAME).write_text("*.log\n")
        seen_files = set()
        for dirpath, _, filenames in corpus_walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            for fn in filenames:
                seen_files.add(f"{rel_dir}/{fn}" if rel_dir != "." else fn)
        self.assertNotIn("debug.log", seen_files)
        self.assertNotIn("sub/c.log", seen_files)

    def test_hidden_skip_dirs_still_pruned(self):
        (self.root / ".git/config").parent.mkdir(parents=True)
        (self.root / ".git/config").write_text("git")
        seen_files = set()
        for dirpath, _, filenames in corpus_walk(self.root):
            seen_files.update(filenames)
        self.assertNotIn("config", seen_files)


class TestDotDirsAreContent(TempCase):
    """fp5 walk contract (reference-rewrites PRD Phase 1): dot-directories
    are content; the true tool layer and the ignore file stay invisible."""

    def test_corpus_walk_includes_dot_dirs(self):
        write(self.root / ".claude" / "skills" / "s.md", "# skill\n")

        def seen():
            out = []
            for dirpath, _, filenames in corpus_walk(self.root):
                rel = Path(dirpath).relative_to(self.root).as_posix()
                out.extend(f"{rel}/{fn}" if rel != "." else fn for fn in filenames)
            return out

        self.assertIn(".claude/skills/s.md", seen())
        # ...and the skip-list members stay invisible in the SAME walk.
        write(self.root / ".git" / "config", "git\n")
        write(self.root / "node_modules" / "x.js", "js\n")
        write(self.root / ".venv" / "pyvenv.cfg", "cfg\n")
        files = seen()
        self.assertNotIn(".git/config", files)
        self.assertNotIn("node_modules/x.js", files)
        self.assertNotIn(".venv/pyvenv.cfg", files)
        self.assertIn(".claude/skills/s.md", files)

    def test_slopkeeperignore_still_prunes_dot_dirs(self):
        # Per-corpus noise hiding in a dot-dir (.cache, transcript dumps)
        # is the ignore file's job, not a blanket rule.
        write(self.root / ".cache" / "dump.jsonl", "{}\n")
        (self.root / IGNORE_NAME).write_text(".cache/\n")
        seen = set()
        for dirpath, _, filenames in corpus_walk(self.root):
            seen.update(filenames)
        self.assertNotIn("dump.jsonl", seen)


class TestNestedCorpora(TempCase):
    """Phase 7: a subtree with its own manifest is a separate corpus."""

    def _nest(self, name: str = "docs") -> None:
        (self.root / name / ".slopkeeper").mkdir(parents=True, exist_ok=True)
        (self.root / name / ".slopkeeper" / "manifest.json").write_text("{}")

    def test_nested_corpus_pruned_and_named(self):
        write(self.root / "docs" / "x.md", "x\n")
        self._nest()
        nested = []
        seen = set()
        for dirpath, _, filenames in corpus_walk(self.root, nested_out=nested):
            seen.update(filenames)
        self.assertEqual(nested, ["docs"])
        # The subtree is invisible: neither the corpus file nor the
        # nested corpus's own state dir leaks into the parent's walk.
        self.assertNotIn("x.md", seen)

    def test_root_state_dir_is_not_a_nested_corpus(self):
        # The root's own .slopkeeper is skip-list, never "nested".
        (self.root / ".slopkeeper").mkdir(parents=True)
        (self.root / ".slopkeeper" / "manifest.json").write_text("{}")
        self.assertEqual(nested_corpora(self.root), [])

    def test_deep_nested_corpus_named_by_relpath(self):
        write(self.root / "a" / "b" / "f.md", "f\n")
        (self.root / "a" / "b" / ".slopkeeper").mkdir(parents=True)
        (self.root / "a" / "b" / ".slopkeeper" / "manifest.json").write_text("{}")
        self.assertEqual(nested_corpora(self.root), ["a/b"])

    def test_manifest_absent_means_not_nested(self):
        # Tool residue (.slopkeeper without a manifest) is just the skip
        # list; only a manifest makes a subtree a separate corpus.
        write(self.root / "sub" / "f.md", "f\n")
        (self.root / "sub" / ".slopkeeper" / "journal.jsonl").parent.mkdir(parents=True)
        (self.root / "sub" / ".slopkeeper" / "journal.jsonl").write_text("{}\n")
        self.assertEqual(nested_corpora(self.root), [])
        seen = set()
        for dirpath, _, filenames in corpus_walk(self.root):
            seen.update(filenames)
        self.assertIn("f.md", seen)  # ordinary content: still walked

    def test_nested_corpus_is_ordinary_content_again_without_manifest(self):
        write(self.root / "sub" / "f.md", "f\n")
        self._nest("sub")
        self.assertEqual(nested_corpora(self.root), ["sub"])
        (self.root / "sub" / ".slopkeeper" / "manifest.json").unlink()
        self.assertEqual(nested_corpora(self.root), [])


class TestFingerprint(TempCase):
    def test_prefix_is_fp5(self):
        self.assertTrue(corpus_fingerprint(self.root).startswith("fp5:"))

    def test_non_regular_files_are_not_content(self):
        # fp5 (0.1.6 field test B6): sandbox deny-list mounts show up as
        # char devices — they are not files, are not fingerprinted, and
        # a FIFO is never read (that would block forever).
        import os as _os

        dev = self.root / ".claude" / "loop.md"
        dev.parent.mkdir(parents=True, exist_ok=True)
        _os.mkfifo(dev)  # FIFO: not a regular file
        before = corpus_fingerprint(self.root)
        # A sandboxed/unsandboxed pair differs only by such entries:
        # creating it must not move the fingerprint vs the same corpus
        # with a REGULAR file there... assert the FIFO is invisible:
        inv = collect(self.root, fast=True)
        self.assertNotIn(".claude/loop.md", [f["path"] for f in inv["files"]])
        _os.unlink(dev)
        self.assertEqual(corpus_fingerprint(self.root), before)

    def test_ignoring_a_file_equals_removing_it(self):
        # (historical comment) fp5 hashes exactly the effective
        # (non-ignored) set: a corpus
        # with an ignored file fingerprints identically to one where
        # the file does not exist at all.
        (self.root / IGNORE_NAME).write_text("ledger_wb.html\n")
        after_ignore = corpus_fingerprint(self.root)
        (self.root / IGNORE_NAME).unlink()
        (self.root / "ledger_wb.html").unlink()
        self.assertEqual(corpus_fingerprint(self.root), after_ignore)

    def test_comment_only_ignore_edit_does_not_move_the_fingerprint(self):
        # Only the effective path set counts — the ignore file's own
        # bytes are never hashed (it is tool layer, not content).
        (self.root / IGNORE_NAME).write_text("ledger_wb.html\n")
        fp1 = corpus_fingerprint(self.root)
        (self.root / IGNORE_NAME).write_text("# a comment\nledger_wb.html\n")
        self.assertEqual(corpus_fingerprint(self.root), fp1)

    def test_unignoring_changes_the_fingerprint(self):
        (self.root / IGNORE_NAME).write_text("ledger_wb.html\n")
        before = corpus_fingerprint(self.root)
        (self.root / IGNORE_NAME).unlink()
        self.assertNotEqual(corpus_fingerprint(self.root), before)

    def test_old_prefix_plans_fail_cleanly(self):
        # fp2-stamped plans can never collide with fp3: refused, not applied.
        plan_path = write_plan(
            self.root,
            based_on={"corpus_fingerprint": "fp2:0123456789abcdef01234567:8"},
        )
        expect_code(
            self,
            lambda: apply_plan(plan_path, self.root, execute=True),
            "corpus_changed",
        )


class TestInventoryIgnores(TempCase):
    def test_ignored_paths_absent_from_inventory(self):
        (self.root / "archive").mkdir()
        (self.root / "archive/old.md").write_text("# old")
        (self.root / IGNORE_NAME).write_text("archive/\n")
        inv = collect(self.root, fast=True)
        self.assertFalse(any(f["path"].startswith("archive/") for f in inv["files"]))
        self.assertEqual(inv["stats"]["ignore_rules"], 1)
        # The fingerprint must agree with the file list.
        self.assertTrue(inv["fingerprint"].startswith("fp5:"))


class TestValidateIgnores(TempCase):
    def test_move_source_in_ignored_dir_is_refused(self):
        (self.root / "archive").mkdir()
        (self.root / "archive/old.md").write_text("# old")
        (self.root / IGNORE_NAME).write_text("archive/\n")
        plan = plan_dict(
            self.root,
            moves=[{"from": "archive/old.md", "to": "other/old.md", "reason": "r"}],
        )
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "move_source_ignored")

    def test_deep_source_ignored_via_ancestor_dir(self):
        (self.root / "archive/deep").mkdir(parents=True)
        (self.root / "archive/deep/x.md").write_text("x")
        (self.root / IGNORE_NAME).write_text("archive/\n")
        plan = plan_dict(
            self.root,
            moves=[{"from": "archive/deep/x.md", "to": "other/x.md", "reason": "r"}],
        )
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "move_source_ignored")

    def test_ignored_dir_is_a_legal_destination(self):
        # "Keep filing into archive/, never reorganize it" is the point.
        (self.root / "archive").mkdir()
        (self.root / IGNORE_NAME).write_text("archive/\n")
        plan = plan_dict(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "archive/acme_dossier.md",
                    "reason": "r",
                }
            ],
        )
        validated = validate_plan(plan, self.root, allow_partial=True)
        self.assertEqual(validated["moves"][0]["to"], "archive/acme_dossier.md")

    def test_edit_target_ignored_is_refused(self):
        (self.root / IGNORE_NAME).write_text("WATCHLIST.md\n")
        plan = plan_dict(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "r/acme_dossier.md",
                    "reason": "r",
                }
            ],
            edits=[
                {
                    "file": "WATCHLIST.md",
                    "find": "acme_dossier.md",
                    "replace": "r/acme_dossier.md",
                    "reason": "rewrite ref",
                }
            ],
        )
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(plan, self.root)
        self.assertEqual(cm.exception.code, "edit_target_ignored")

    def test_ignored_top_level_dir_needs_no_skip_entry(self):
        # Coverage treats ignored entries as non-content, like hidden ones.
        mini = self.tmp / "mini"
        mini.mkdir()
        (mini / "scratchpad").mkdir()
        (mini / "scratchpad/junk.bin").write_text("junk")
        (mini / "a.md").write_text("# a\n")
        (mini / IGNORE_NAME).write_text("scratchpad/\n")
        plan = plan_dict(
            mini,
            moves=[{"from": "a.md", "to": "topic/a.md", "reason": "r"}],
        )
        # Without the ignore file this is plan_incomplete; with it, valid.
        validated = validate_plan(plan, mini)
        self.assertEqual(validated["moves"][0]["to"], "topic/a.md")

    def test_refusal_is_at_apply_too(self):
        # apply re-runs validate: an ignore file added after authoring
        # is caught at apply time, never silently moved through.
        (self.root / "archive").mkdir()
        (self.root / "archive/old.md").write_text("# old")
        (self.root / IGNORE_NAME).write_text("archive/\n")
        plan = plan_dict(
            self.root,
            moves=[{"from": "archive/old.md", "to": "other/old.md", "reason": "r"}],
        )
        p = self.root.parent / "plan.json"
        p.write_text(json.dumps(plan), encoding="utf-8")
        expect_code(
            self,
            lambda: apply_plan(p, self.root, execute=True),
            "move_source_ignored",
        )
        self.assertTrue((self.root / "archive/old.md").exists())  # disk untouched


class TestStatusIgnores(TempCase):
    def test_ignored_new_files_never_surface(self):
        write_manifest(self.root)
        (self.root / "logs").mkdir()
        (self.root / "logs/x.log").write_text("log")
        (self.root / IGNORE_NAME).write_text("logs/\n")
        st = corpus_status(self.root)
        self.assertEqual(st["new_count"], 0)
        self.assertFalse(any("logs/" in f["path"] for f in st["new_files"]))

    def test_file_becoming_ignored_is_not_removed(self):
        # "The tool stopped looking" is not "the file vanished".
        (self.root / "scratch").mkdir()
        (self.root / "scratch/y.md").write_text("y")
        write_manifest(self.root)
        (self.root / IGNORE_NAME).write_text("scratch/\n")
        st = corpus_status(self.root)
        self.assertEqual(st["removed"], [])
        self.assertEqual(st["removed_now_ignored"], ["scratch/y.md"])


class TestToolLayerAndDoctor(TempCase):
    def test_ignore_file_is_tool_layer(self):
        self.assertIn(IGNORE_NAME, TOOL_LAYER)
        # _file_entries (draft's filed-file side) skips it too.
        (self.root / IGNORE_NAME).write_text("*.log\n")
        self.assertEqual(_file_entries(self.root, [IGNORE_NAME]), [])

    def test_ignore_file_never_walked_as_content(self):
        (self.root / IGNORE_NAME).write_text("# nothing\n")
        for _dirpath, _dirnames, filenames in corpus_walk(self.root):
            self.assertNotIn(IGNORE_NAME, filenames)

    def test_doctor_reports_the_ignore_file(self):
        (self.root / IGNORE_NAME).write_text("# comment\nlogs/\n*.log\n")
        d = doctor(self.root)
        self.assertEqual(d["ignore"], {"active": True, "rules": 2})
        self.assertTrue(any("slopkeeperignore" in a for a in d["advice"]))

    def test_doctor_without_ignore_file(self):
        d = doctor(self.root)
        self.assertEqual(d["ignore"], {"active": False, "rules": 0})


if __name__ == "__main__":
    unittest.main()
