# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P6 Slice D — git cooperation: optional, probe-based, fail-soft.

A corpus that is a git repository has its own idea of where its files
live, and SlopKeeper used to ignore it: a tend moved bytes on disk, git saw
a deletion plus an untracked file, and the operator's next `git add -A`
had to rediscover by content what the tool already knew. Worse, a tend
against a dirty tree could strand uncommitted work — the fingerprint gate
binds file CONTENTS and cannot see the index at all.

The three limits the audit sets are the ones these tests hold:

  * **Directories move on disk, never through git.** `git mv` moves what
    git TRACKS and would leave untracked or ignored content behind inside
    the old directory — a build output, a `.env`, a `node_modules`. A
    mover that silently leaves files is the exact failure this tool exists
    to prevent, so a dir move is one `rename()` and git detects the
    renames by content later (which is what rename detection is for).
  * **Git is never a dependency.** No git binary, a git that errors, a
    corpus that is not a repo: every case degrades to the pre-0.3.0
    filesystem path and says so on the payload. Nothing here may crash a
    tend.
  * **Undo stays git-agnostic.** The journal is the source of truth and
    `_undo_journal` does not shell out — but a git-cooperated tend STAGED
    moves in the index, and a rollback cannot unstage them, so undo says
    so loudly instead of letting someone find it at commit time.

The primary workflow is guarded too: untracked slop is what this tool
files, so `git status`'s `??` entries must NEVER trigger the dirty refusal.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from helpers import TempCase, prune_slopkeeper, snapshot_tree, write_plan
from slopkeeper import cli
from slopkeeper import gitops
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError

HAS_GIT = shutil.which("git") is not None


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """One git call against the corpus repo, with a deterministic identity."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=SlopKeeper Tests",
            "-c",
            "user.email=tests@slopkeeper.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def keep_all(root: Path, *moved: str) -> list[str]:
    """Every top-level entry the plan does not move itself.

    `plan validate` refuses an incomplete plan; these fixtures create extra
    files (untracked slop, a notes tree) beyond the shared one, so the
    keep list is derived from the disk each time rather than hardcoded.
    """
    top_moved = {m.split("/")[0] for m in moved}
    return sorted(
        p.name
        for p in root.iterdir()
        if p.name not in {".slopkeeper", "__pycache__"} and p.name not in top_moved
    )


def tree_state(root: Path):
    """The corpus with the tool's own residue removed.

    `.git/` is excluded too, and that exclusion IS a finding: a
    git-cooperated tend stages its renames in `.git/index`, and undo
    deliberately does not touch git — so the corpus is byte-identical
    after a rollback while the INDEX still differs. That asymmetry is what
    `git_index_staged_moves` exists to say out loud; comparing the index
    here would assert the opposite of the design.
    """
    files, dirs = prune_slopkeeper(snapshot_tree(root))

    def keep(name: str) -> bool:
        return name != ".git" and not name.startswith(".git/")

    return {k: v for k, v in files.items() if keep(k)}, {d for d in dirs if keep(d)}


@unittest.skipUnless(HAS_GIT, "git is not installed")
class TestProbe(TempCase):
    """The one-shot look that decides everything else in this slice."""

    def test_a_plain_directory_is_not_a_repo(self):
        state = gitops.probe(self.root)
        self.assertIs(state["repo"], False)
        self.assertIs(state["available"], False)
        self.assertEqual(state["reason"], "not_a_git_repo")

    def test_a_repo_probes_available_with_head_and_tracked_set(self):
        git(self.root, "init", "-q")
        git(self.root, "add", "acme_dossier.md")
        git(self.root, "commit", "-qm", "seed")
        state = gitops.probe(self.root)
        self.assertIs(state["repo"], True)
        self.assertIs(state["available"], True)
        self.assertRegex(state["head"] or "", r"^[0-9a-f]{40}$")
        self.assertIn("acme_dossier.md", state["tracked"])
        # Untracked slop is KNOWN but not a reason to refuse anything.
        self.assertNotIn("globex_dossier.md", state["tracked"])

    def test_an_empty_repo_has_no_head_and_still_probes_available(self):
        # `git init` with no commit is a real state a corpus can be in; the
        # tend must still cooperate (there is nothing staged to lose).
        git(self.root, "init", "-q")
        state = gitops.probe(self.root)
        self.assertIs(state["available"], True)
        self.assertIsNone(state["head"])

    def test_untracked_files_are_reported_and_not_dirty(self):
        git(self.root, "init", "-q")
        git(self.root, "add", "acme_dossier.md")
        git(self.root, "commit", "-qm", "seed")
        (self.root / "fresh-slop.md").write_text("new\n")
        state = gitops.probe(self.root)
        self.assertIn("fresh-slop.md", state["untracked"])
        self.assertEqual(gitops.dirty_sources(self.root, ["fresh-slop.md"]), [])

    def test_a_failing_git_fails_soft(self):
        # A git that cannot answer must not block tends: people --force
        # past a tool that breaks whenever git misbehaves.
        self.assertEqual(gitops.dirty_sources(self.root / "nope", ["a.md"]), [])
        broken = self.root / "nope"
        self.assertEqual(gitops.status_counts(broken), {"status_error": True})


@unittest.skipUnless(HAS_GIT, "git is not installed")
class TestGitMv(TempCase):
    """Tracked files move through git; everything else keeps the fs path."""

    def setUp(self):
        super().setUp()
        git(self.root, "init", "-q")
        git(self.root, "add", "acme_dossier.md", "globex_dossier.md", "docs")
        git(self.root, "commit", "-qm", "seed")
        (self.root / "untracked.md").write_text("slop\n")

    def _apply(self, plan_over=None, **apply_over):
        # `plan_over` goes into the plan file, `apply_over` into apply's
        # own flags (--force / --no-git live there, not in the JSON).
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
            **(plan_over or {}),
        )
        return apply_plan(plan_path, self.root, execute=True, **apply_over)

    def test_tracked_file_moves_via_git_and_stages_the_rename(self):
        data = self._apply()
        self.assertEqual(data["git"]["moved_via_git"], 1)
        self.assertEqual(data["git"]["moved_via_fs"], 0)
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        porcelain = git(self.root, "status", "--porcelain").stdout
        self.assertIn(
            "R  acme_dossier.md -> research/dossiers/acme_dossier.md", porcelain
        )

    def test_untracked_source_moves_on_disk_and_is_never_refused(self):
        # The primary workflow: filing the slop an agent dropped. Untracked
        # must not read as "dirty" or the tool refuses its own job.
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "untracked.md", "to": "docs/untracked.md", "reason": "docs"}
            ],
            keep=keep_all(self.root, "untracked.md"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"])
        self.assertEqual(data["git"]["moved_via_git"], 0)
        self.assertEqual(data["git"]["moved_via_fs"], 1)
        self.assertTrue((self.root / "docs/untracked.md").exists())

    def test_a_directory_move_never_goes_through_git(self):
        # git mv would move only what is tracked, silently leaving anything
        # else behind in the old directory — so dirs keep the fs path.
        (self.root / "notes/kept-untracked.txt").parent.mkdir(exist_ok=True)
        git(self.root, "add", "handoff-tp2.md")
        git(self.root, "commit", "-qm", "more")
        (self.root / "notes").mkdir(exist_ok=True)
        (self.root / "notes/a.md").write_text("a\n")
        (self.root / "notes/ignored.txt").write_text("scratch\n")
        git(self.root, "add", "notes/a.md")
        git(self.root, "commit", "-qm", "notes")
        plan_path = write_plan(
            self.root,
            moves=[{"from": "notes", "to": "docs/notes", "reason": "under docs"}],
            keep=keep_all(self.root, "notes"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(data["git"]["moved_via_git"], 0)
        self.assertEqual(data["git"]["moved_via_fs"], 1)
        # EVERYTHING came along, tracked or not — the reason for the rule.
        self.assertTrue((self.root / "docs/notes/a.md").exists())
        self.assertTrue((self.root / "docs/notes/ignored.txt").exists())
        self.assertFalse((self.root / "notes").exists())

    def test_journal_header_records_the_commit(self):
        data = self._apply()
        header = json.loads(Path(data["journal"]).read_text().splitlines()[0])
        self.assertEqual(header["git_head"], data["git"]["head"])
        self.assertIs(header["git_cooperated"], True)

    def test_git_block_counts_the_index_afterwards(self):
        data = self._apply()
        self.assertIs(data["git"]["repo"], True)
        self.assertIs(data["git"]["cooperated"], True)
        self.assertGreaterEqual(data["git"]["renamed"], 1)
        self.assertGreaterEqual(data["git"]["staged"], 1)

    def test_no_git_flag_skips_cooperation_entirely(self):
        data = self._apply(use_git=False)
        self.assertIs(data["git"]["cooperated"], False)
        self.assertEqual(data["git"]["moved_via_git"], 0)
        self.assertEqual(data["git"]["moved_via_fs"], 1)
        # The file still moved — cooperation is an enhancement, not a gate.
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        # ...and git saw nothing of it: no staged rename.
        porcelain = git(self.root, "status", "--porcelain").stdout
        self.assertNotIn("R  acme_dossier.md", porcelain)

    def test_head_is_still_recorded_when_cooperation_is_off(self):
        # `data.git` describes the corpus, not the flags: an operator who
        # passed --no-git still wants to know which commit they were at.
        data = self._apply(use_git=False)
        self.assertRegex(data["git"]["head"] or "", r"^[0-9a-f]{40}$")


@unittest.skipUnless(HAS_GIT, "git is not installed")
class TestDirtyRefusal(TempCase):
    """The gate the fingerprint cannot be: staged and worktree state."""

    def setUp(self):
        super().setUp()
        git(self.root, "init", -1 if False else "-q")
        git(self.root, "add", "acme_dossier.md", "globex_dossier.md", "docs")
        git(self.root, "commit", "-qm", "seed")

    def _plan(self, **over):
        return write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
            **over,
        )

    def test_worktree_modification_refuses_before_any_move(self):
        (self.root / "acme_dossier.md").write_text("# EDITED AND UNCOMMITTED\n")
        before = tree_state(self.root)
        plan_path = self._plan()  # authored AFTER the edit: fingerprint matches
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "git_dirty")
        self.assertIn("acme_dossier.md", cm.exception.message)
        # All-or-nothing: the refusal is a pre-flight, nothing moved.
        self.assertEqual(tree_state(self.root), before)
        self.assertTrue((self.root / "acme_dossier.md").exists())

    def test_force_moves_anyway_and_says_so(self):
        (self.root / "acme_dossier.md").write_text("# EDITED AND UNCOMMITTED\n")
        plan_path = self._plan()
        data = apply_plan(plan_path, self.root, execute=True, force=True)
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        self.assertEqual(data["git"]["dirty_sources"], 1)
        self.assertIn("git_dirty_forced", data["warnings"])

    def test_dirty_check_survives_as_a_cli_exit_1_not_2(self):
        # Not a plan-shape failure: the plan is fine, the DISK is the
        # problem, so it is the general error class (only plan_* and
        # verify_failed are the "fix the contract" class).
        (self.root / "acme_dossier.md").write_text("# EDITED\n")
        plan_path = self._plan()
        code, env = run_cli(["apply", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 1)
        self.assertEqual(env["error"]["code"], "git_dirty")
        self.assertIn("--force", env["error"]["suggested_command"])

    def test_no_git_flag_lifts_the_refusal(self):
        # `--no-git` means "I do not want git in this tend at all" — the
        # dirty refusal is part of that package, not a separate promise.
        (self.root / "acme_dossier.md").write_text("# EDITED\n")
        plan_path = self._plan()
        data = apply_plan(plan_path, self.root, execute=True, use_git=False)
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())
        self.assertEqual(data["git"]["dirty_sources"], 0)

    def test_a_dirty_file_the_plan_does_not_touch_is_not_refused(self):
        # The refusal is scoped to the plan's sources, not "any dirt in the
        # repo": a corpus with unrelated uncommitted work still tends.
        (self.root / "WATCHLIST.md").write_text("# edited elsewhere\n")
        (self.root / "WATCHLIST.md").chmod(0o644)
        (self.root / "staged elsewhere.md").write_text("x\n")
        git(self.root, "add", "WATCHLIST.md")
        git(self.root, "commit", "-qm", "watchlist committed")
        (self.root / "WATCHLIST.md").write_text("# edited again, uncommitted\n")
        plan_path = self._plan()
        # WATCHLIST.md is dirty but not a source; acme_dossier.md is clean.
        self.assertEqual(gitops.dirty_sources(self.root, ["acme_dossier.md"]), [])
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"])
        self.assertEqual(data["git"]["dirty_sources"], 0)
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())

    def test_staged_deletion_counts_as_dirty(self):
        git(self.root, "rm", "-q", "--cached", "globex_dossier.md")
        git(self.root, "add", "acme_dossier.md")
        (self.root / "acme_dossier.md").write_text("# changed and staged\n")
        git(self.root, "add", "acme_dossier.md")
        plan_path = self._plan()
        with self.assertRaises(SlopKeeperError) as cm:
            apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(cm.exception.code, "git_dirty")

    def test_added_but_uncommitted_file_is_movable(self):
        # `A` in the index is NEW slop someone already staged — the exact
        # thing this tool files. Refusing it would break the primary loop.
        (self.root / "staged-new.md").write_text("new\n")
        git(self.root, "add", "staged-new.md")
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "staged-new.md", "to": "docs/staged-new.md", "reason": "d"}
            ],
            keep=keep_all(self.root, "staged-new.md"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertTrue((self.root / "docs/staged-new.md").exists())
        self.assertEqual(data["git"]["dirty_sources"], 0)


@unittest.skipUnless(HAS_GIT, "git is not installed")
class TestUndoStaysGitAgnostic(TempCase):
    """The journal is the source of truth; the index is the operator's."""

    def setUp(self):
        super().setUp()
        git(self.root, "init", "-q")
        git(self.root, "add", "acme_dossier.md", "globex_dossier.md", "docs")
        git(self.root, "commit", "-qm", "seed")

    def test_undo_restores_files_and_warns_about_the_index(self):
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
        )
        before = tree_state(self.root)
        data = apply_plan(plan_path, self.root, execute=True)
        undone = undo_journal(Path(data["journal"]), self.root, execute=True)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertEqual(tree_state(self.root), before)
        self.assertIs(undone["git_cooperated"], True)
        self.assertIn("git_index_staged_moves", undone["warnings"])
        self.assertEqual(undone["git_head"], data["git"]["head"])

    def test_undo_never_shells_out_to_git(self):
        # The recovery path must not depend on a second recovery path: no
        # git subprocess is spawned by undo, so a missing binary cannot
        # strand a tend.
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        journal = Path(data["journal"])
        real_run = subprocess.run
        calls: list[list] = []

        def spy(cmd, *a, **kw):
            calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [cmd])
            return real_run(cmd, *a, **kw)

        with mock_patch(subprocess, "run", spy):
            undo_journal(journal, self.root, execute=True)
        self.assertFalse([c for c in calls if c and c[0] == "git"], calls)

    def test_a_pre_git_undoes_without_the_warning(self):
        # The warning is earned from the journal header, not from the
        # corpus happening to be a repo now: a tend that ran without
        # cooperation must not tell the operator to fix an index it never
        # touched.
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True, use_git=False)
        undone = undo_journal(Path(data["journal"]), self.root, execute=True)
        self.assertNotIn("git_index_staged_moves", undone.get("warnings", []))
        self.assertNotIn("git_cooperated", undone)

    def test_cli_undo_hint_names_the_escape_hatches(self):
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/dossiers/acme_dossier.md",
                    "reason": "dossier",
                }
            ],
            keep=keep_all(self.root, "acme_dossier.md"),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        code, env = run_cli(
            ["undo", str(Path(data["journal"])), str(self.root), "--execute"]
        )
        self.assertEqual(code, 0)
        self.assertIn("git reset", env["next"]["hint"])
        self.assertIn("git checkout -- .", env["next"]["hint"])


@unittest.skipUnless(HAS_GIT, "git is not installed")
class TestNonRepoUnchanged(TempCase):
    """The zero-regression guarantee for the other half of the corpora."""

    def test_no_git_key_at_all(self):
        plan_path = write_plan(self.root, folders=[])
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertNotIn("git", data)
        self.assertNotIn("git_unavailable", data.get("warnings", []))

    def test_journal_header_carries_no_git_fields(self):
        plan_path = write_plan(self.root, folders=[])
        data = apply_plan(plan_path, self.root, execute=True)
        header = json.loads(Path(data["journal"]).read_text().splitlines()[0])
        self.assertNotIn("git_head", header)
        self.assertNotIn("git_cooperated", header)

    def test_move_is_still_all_filesystem(self):
        plan_path = write_plan(self.root, folders=[])
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"])
        self.assertTrue((self.root / "research/dossiers/acme_dossier.md").exists())


@contextlib.contextmanager
def mock_patch(module, name, replacement):
    """Tiny patch helper (keeps the suite free of an import-time mock habit)."""
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)
