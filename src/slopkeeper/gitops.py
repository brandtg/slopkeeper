# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Git cooperation for apply (P6, Slice D): optional, probe-based, fail-soft.

A corpus that is a git repository has its own idea of where its files
live, and SlopKeeper used to ignore it: a tend moved files on disk, git
saw a deletion plus an untracked file, and the operator's next `git add
-A` had to rediscover by content what the tool already knew. Worse, a tend
against a dirty tree could throw away uncommitted work — the fingerprint
gate binds file CONTENTS to the plan, but staged state is invisible to it.

So when (and only when) the corpus is a repo and the `git` binary is
there, apply:

  * prefers ``git mv`` per FILE move, so the rename lands in the index and
    `git status` shows a rename instead of a loss;
  * refuses to move a source with uncommitted modifications unless
    ``--force`` (`git_dirty`) — the fingerprint catches content drift
    against the plan, never drift the operator has already staged;
  * records HEAD in the journal's provenance header, so a tend names the
    commit it ran against;
  * reports a `data.git` block on the execute envelope.

Three rules hold this together, and each is a deliberate limit:

  1. **Directories move on disk, never through git.** `git mv` on a
     directory moves what git tracks; untracked and ignored content inside
     (a build output, a `.env`, a `node_modules`) stays behind. A directory
     move that silently leaves files is the exact failure this tool exists
     to prevent, so the whole tree moves with one `rename()` and git
     discovers the renames by content at diff time — which is what rename
     detection IS for.
  2. **Every git call fails soft.** A git that errors, times out or is
     missing degrades to the pre-0.3.0 filesystem behavior and says so in
     the payload; nothing here can make a tend crash. Git is an
     enhancement, never a dependency: the wheel runs on a machine with no
     git and the corpus does not care.
  3. **Undo stays git-agnostic.** The journal is the source of truth and
     `_undo_journal` does not shell out. What undo DOES do is tell you
     when the tend it just rolled back had staged moves in the index
     (`git_index_staged_moves`), because those are real leftovers the
     journal cannot reach — `git reset` unstages them, and `git checkout
     -- .` remains the parallel escape hatch for a tracked file (AGENT_GUIDE).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# A git call is a boundary, not a loop: bounded, never hanging a tend that
# holds the corpus's tend lock. `git status` on a huge repo is the slow
# one; 60s is generous and `timeout` keeps a wedged filesystem from
# turning apply into a forever-command.
GIT_TIMEOUT = 60

# A porcelain entry whose worktree or index column holds one of these is
# an UNCOMMITTED CHANGE to tracked content — the thing `--force` is needed
# to move. `??` never appears (untracked files are listed with -uno for
# this check) and `A` is deliberately absent: a file staged as new is
# exactly the slop this tool exists to file, and refusing to move it
# would break the primary workflow.
_DIRTY_CODES = set("MDCTRU")


def _git(root: Path, *args: str) -> tuple[int, str, str]:
    """One git call: (returncode, stdout, stderr); (-1, '', err) on failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return -1, "", f"{type(e).__name__}: {e}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def probe(root: Path) -> dict:
    """The corpus's git state, decided ONCE per tend. Never raises.

    {available, repo, head, tracked, untracked, reason}: `available` is
    the whole go/no-go for git cooperation — a repo with no git binary, or
    a repo git itself cannot read, is simply tended the way it always was.
    `reason` names why when it is False, so the envelope can say so rather
    than leave the absence unexplained.
    """
    root = Path(root)
    state = {
        "available": False,
        "repo": False,
        "head": None,
        "tracked": frozenset(),
        "untracked": frozenset(),
        "reason": "not_a_git_repo",
    }
    if not (root / ".git").exists():  # a dir OR a worktree/submodule file
        return state
    state["repo"] = True
    if shutil.which("git") is None:
        state["reason"] = "git_missing"
        return state
    rc, out, _err = _git(root, "rev-parse", "--git-dir")
    if rc != 0:
        state["reason"] = "not_a_work_tree"
        return state
    rc, out, _ = _git(root, "rev-parse", "HEAD")
    # An empty repo has no HEAD; that is not a reason to skip cooperation.
    state["head"] = out.strip() if rc == 0 else None
    rc, out, _ = _git(root, "ls-files", "-z")
    if rc != 0:
        state["reason"] = "ls_files_failed"
        return state
    tracked = frozenset(p for p in out.split("\0") if p)
    rc, out, _ = _git(root, "status", "--porcelain", "--untracked-files=all", "-z")
    untracked = frozenset()
    if rc == 0:
        untracked = frozenset(
            entry[3:]
            for entry in out.split("\0")
            if len(entry) > 3 and entry[:2] == "??"
        )
    state.update(
        {"available": True, "tracked": tracked, "untracked": untracked, "reason": ""}
    )
    return state


def dirty_sources(root: Path, rel_paths: list[str]) -> list[str]:
    """Which of `rel_paths` git says have uncommitted changes.

    `git status --porcelain` filtered against the plan's sources — the
    check the audit prescribes, because the fingerprint gate binds CONTENT
    and says nothing about the index. A source that is modified in the
    worktree or staged-and-uncommitted is refused (`git_dirty`) unless
    --force: moving it would make uncommitted work hard to find again, and
    the journal can restore the file but not the intent.

    No pathspec on the command line: a rule-expanded plan can carry tens
    of thousands of sources, and passing them as argv would die with E2BIG
    — a crash in the pre-flight of the tool whose whole job is not to
    crash mid-tend. One unbounded status call plus a prefix match is both
    cheaper and safe. `A` (staged as new) is not dirty; `??` is not asked
    about (-uno): untracked slop is exactly what this tool files.

    Fail-soft: a git that cannot answer reports nothing dirty, because a
    tool that blocks tends whenever git misbehaves is a tool people
    --force past into uselessness.
    """
    if not rel_paths:
        return []
    rc, out, _ = _git(root, "status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        return []
    changed: list[str] = []
    for entry in out.splitlines():
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        if not any(c in _DIRTY_CODES for c in xy):
            continue
        if " -> " in path:  # a rename names `old -> new`; the NEW path is on disk
            path = path.rsplit(" -> ", 1)[1]
        changed.append(path.strip('"').rstrip("/"))
    if not changed:
        return []
    # Match by walking each CHANGED path's ancestors against the source set
    # (O(changes x depth)) rather than testing every source against every
    # change: a rule-expanded plan can carry tens of thousands of sources,
    # and the number of dirty files in a working corpus is small.
    sources = set(rel_paths)
    dirty: set[str] = set()
    for path in changed:
        parts = path.split("/")
        for i in range(len(parts), 0, -1):
            cand = "/".join(parts[:i])
            if cand in sources:
                dirty.add(cand)
                break
    return sorted(dirty)


def status_counts(root: Path) -> dict:
    """The post-tend shape of `git status`, as counts — the `data.git` block.

    `staged` counts entries with anything in the index column (which is
    what `git mv` produces), `renamed` the R entries, `modified`/`deleted`
    /`added` the rest, `untracked` the `??` entries — a tend that moved
    40 files and left 40 untracked ones is visible here, and that is the
    difference between "git cooperation ran" and "git cooperation is
    pretending".
    """
    rc, out, _ = _git(root, "status", "--porcelain", "--untracked-files=all")
    if rc != 0:
        return {"status_error": True}
    counts = {"staged": 0, "renamed": 0, "added": 0, "modified": 0, "deleted": 0}
    untracked = 0
    for entry in out.splitlines():
        if len(entry) < 4:
            continue
        xy = entry[:2]
        if xy == "??":
            untracked += 1
            continue
        index, worktree = xy[0], xy[1]
        if index != " ":
            counts["staged"] += 1
        if index == "R" or worktree == "R":
            counts["renamed"] += 1
        for code, key in (("A", "added"), ("M", "modified"), ("D", "deleted")):
            if code in xy:
                counts[key] += 1
    counts["untracked"] = untracked
    return counts


def move_file(root: Path, src_rel: str, dst_rel: str) -> bool:
    """`git mv` one tracked file; True when git did the move and staged it.

    False means "do it the filesystem way" and is returned for every
    failure, including the interesting one (the destination exists — the
    caller's `_move_no_clobber` then refuses it the same way it always
    has). Never raises: git is an enhancement, so a git that disagrees
    costs a rename in the index and nothing else.
    """
    rc, _out, _err = _git(root, "mv", "--", src_rel, dst_rel)
    return rc == 0
