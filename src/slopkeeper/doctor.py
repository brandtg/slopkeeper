# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`doctor`: preflight a corpus before any agent works on it.

Answers the questions the agent needs answered first: how big is this,
is it a git repo, how messy is the root, which strategy fits, and what
will the inventory cost in tokens.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .anchors import anchored_dirs
from .ignore import IGNORE_NAME, HIDDEN_SKIP_DIRS, load_ignore, nested_corpora
from .inventory import collect
from .models import STATE_DIR, SlopKeeperError
from .references import dangling_citations

# Rough token heuristic: ~4 chars per token on markdown/JSON-ish text.
CHARS_PER_TOKEN = 4

_JSONL_HEAVY_RATIO = 0.8
_JSONL_MIN_FILES = 10
_CACHE_DIR_FILES = 5000
_MAX_LISTED = 10


def _rel_walk_dirs(root: Path, ignore, nested: set[str]):
    """os.walk with the corpus's prune contract, but IGNORED FILES KEPT.

    The strandings detector needs ignored and visible files side by
    side (the copied-.gitignore trap), so dir-level ignores prune the
    descent (a fully-ignored subtree cannot strand) while per-file
    ignore status is computed inside. Skip-list members and nested
    corpora are pruned as everywhere else.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).resolve().relative_to(root)
        kept = []
        for d in dirnames:
            if d in HIDDEN_SKIP_DIRS or d == STATE_DIR:
                continue
            rel = (rel_dir / d).as_posix() if rel_dir.parts else d
            if ignore.is_ignored(rel, is_dir=True) or rel in nested:
                continue
            kept.append(d)
        dirnames[:] = sorted(kept)
        yield (
            str(rel_dir) if rel_dir.parts else ".",
            sorted(filenames),
        )


def _visibility_scan(
    root: Path, ignore, nested: set[str]
) -> tuple[list[dict], dict, list[Path], list[dict]]:
    """(stranded dirs, dot-dir stats, visible file list, non-regular files).

    stranded: dirs where ignored and non-ignored files coexist — a
    copied .gitignore strands half a folder and the tool sees only the
    visible half. dot-dir stats roll file counts and .jsonl counts up
    into each first-segment dot directory (the transcript-dump trap).
    The file list feeds the dangling-citation scan, so both answers
    cost one walk. Non-regular entries (devices, FIFOs, sockets —
    sandbox deny-list mounts) are named so a sandboxed run explains
    its own count drift instead of poisoning the fingerprint.
    """
    stranded: list[dict] = []
    dotdirs: dict[str, dict] = {}
    files: list[Path] = []
    non_regular: list[dict] = []
    for rel_dir, filenames in _rel_walk_dirs(root, ignore, nested):
        ignored_n = 0
        visible_n = 0
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir != "." else fn
            fp = root / rel
            try:
                mode = fp.stat().st_mode
            except OSError:
                mode = 0
            if mode and not stat.S_ISREG(mode):
                non_regular.append({"path": rel, "mode": stat.S_IFMT(mode)})
                continue
            if ignore.is_ignored(rel, is_dir=False):
                ignored_n += 1
            else:
                visible_n += 1
            files.append(Path(rel))
        if ignored_n and visible_n:
            stranded.append(
                {"dir": rel_dir, "ignored": ignored_n, "visible": visible_n}
            )
        first = rel_dir.split("/")[0]
        if first.startswith(".") and first != ".":
            st = dotdirs.setdefault(first, {"files": 0, "jsonl": 0})
            for fn in filenames:
                st["files"] += 1
                if fn.endswith(".jsonl"):
                    st["jsonl"] += 1
    return stranded, dotdirs, files, non_regular


def _nested_repos_and_modules(root: Path, ignore) -> tuple[list[str], list[str]]:
    """Nested git repos (dir/.git) and node_modules dirs, un-ignored only.

    A nested repo's files ARE corpus content since fp4 (only .git
    itself is invisible) — the trap is depending on it without
    realizing it is there, so it is detected and its dir suggested for
    the ignore file. Copying the suggested lines silences the warning
    (ignored dirs are filtered out here).
    """
    repos: list[str] = []
    modules: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        rel_dir = Path(dirpath).resolve().relative_to(root)
        hit_repo = ".git" in dirnames
        hit_node = "node_modules" in dirnames
        rel_str = str(rel_dir) if rel_dir.parts else "."
        if rel_dir.parts:  # the corpus's own .git is stats.git_repo's business
            if hit_repo and not ignore.is_ignored(rel_str, is_dir=True):
                repos.append(rel_str)
            if hit_node and not ignore.is_ignored(rel_str, is_dir=True):
                modules.append(rel_str)
        # never descend into the layers themselves
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in HIDDEN_SKIP_DIRS and d not in (".git", "node_modules")
        )
    return sorted(repos), sorted(modules)


def _dedupe_prefixes(dirs: list[str]) -> list[str]:
    """Drop dirs already covered by a suggested ancestor."""
    out = []
    for d in sorted(dirs):
        if not any(d == s or d.startswith(s.rstrip("/") + "/") for s in out):
            out.append(d)
    return out


def doctor(root: Path) -> dict:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise SlopKeeperError(
            "no_corpus",
            f"{root} is not a directory",
            "Pass the directory you want to organize.",
        )

    inv = collect(root, fast=True)
    s = inv["stats"]
    n = s["files"]

    # Messiness signals.
    loose = s["loose_top_level_files"]
    loose_ratio = loose / max(n, 1)
    readme_dirs = n - loose  # crude; refined below via dirs with README
    # All top-level dirs in the inventory count, dot-prefixed ones
    # included: dot-dirs are corpus content since fp4.
    readme_dirs = sum(
        1
        for d in inv["dirs"]
        if d["path"].count("/") == 0 and (root / d["path"] / "README.md").exists()
    )
    dir_count = s["dirs"]

    # Strategy heuristic.
    if n < 15 and loose_ratio < 0.3:
        strategy = (
            "skip",
            "Corpus is small and already tidy — an agent can "
            "organize it by hand faster than any pipeline.",
        )
    elif readme_dirs >= 4 and dir_count / max(dir_count + loose, 1) > 0.5:
        strategy = (
            "role-bots",
            "Many structured folders already exist; formalize "
            "them as role-bots with AGENTS.md/STATE.md.",
        )
    elif loose_ratio > 0.6:
        strategy = (
            "hybrid",
            "Root is a loose-file junk drawer; hybrid sweep: "
            "topics + archive + a few role folders.",
        )
    elif s["truncated"] or n > 2000:
        strategy = (
            "topics",
            "Large corpus; partition by subject with a "
            "depth-limited inventory (--fast, then per-dir).",
        )
    else:
        strategy = (
            "hybrid",
            "Mixed corpus; default to hybrid unless the "
            "owner names a single principle.",
        )

    # Token estimate for a full inventory read by the agent.
    est_chars = 0
    for f in inv["files"][:5000]:
        est_chars += 180 + len(str(f.get("head") or ""))
    est_tokens = est_chars // CHARS_PER_TOKEN

    advice = []
    if not s["git_repo"]:
        advice.append(
            "Not a git repo — run `git init && git add -A && git commit` "
            "first, or accept journal-only reversibility."
        )
    # §6.6: make the invisible set visible — never a silent empty walk.
    ig = load_ignore(root)
    if ig.active():
        advice.append(
            f"{IGNORE_NAME} is active ({ig.rule_count} rule(s)): ignored "
            "paths are invisible to inventory, draft, plans and the "
            "fingerprint — and citations from ignored files are invisible "
            "to --strict-references."
        )
    if loose > 20:
        advice.append(f"{loose} loose files at top level — the main cleanup target.")
    if n > 800:
        advice.append(
            "Corpus > 800 files: use `inventory --fast` plus per-dir "
            "inventories; reference scanning is skipped."
        )
    if s["truncated"]:
        # Phase 11: the culprit is named from per-top-dir counts taken
        # past the cap — "truncated" alone cost the field report a du.
        culprits = list(s["top_dir_file_counts"].items())[:3]
        named = ", ".join(f"`{d}/` ({c} files)" for d, c in culprits)
        advice.append(
            f"File count exceeded the walk cap ({n} inventoried); inventory "
            f"is partial. Largest top-level dirs: {named} — inventory "
            "those per-dir (`slopkeeper inventory <root>/<dir> --fast`)."
        )
    # Phase 7: nested corpora are named, never silently missing.
    nested = nested_corpora(root)
    if nested:
        advice.append(
            f"Nested corpus/corpora detected: {', '.join(nested)} — each "
            "carries its own .slopkeeper/manifest.json and is invisible to "
            "this corpus's inventory, status and fingerprint. Tend each "
            "separately (`slopkeeper status <path>`); a wholesale move of "
            "its directory is the nested corpus's own business."
        )
    # Phase 11 diagnostics: one shared walk answers the four questions
    # the reports answered by hand — nested repos, stranded folders,
    # cache-like dot dirs, and the visibility file list for dangling
    # citations.
    stranded, dotdirs, walk_files, non_regular = _visibility_scan(root, ig, set(nested))
    repos, modules = _nested_repos_and_modules(root, ig)
    suggested_lines: list[str] = []
    if repos or modules:
        suggested_lines = [f"{d}/" for d in _dedupe_prefixes(repos + modules)]
        advice.append(
            "Nested git repo(s) / node_modules detected: "
            + ", ".join(f"`{d}`" for d in repos + modules)
            + " — their files are corpus content (only .git itself is "
            "invisible), and their deps would bloat every scan. "
            "Suggested .slopkeeperignore block:\n    " + "\n    ".join(suggested_lines)
        )
    if stranded:
        worst = sorted(stranded, key=lambda x: -x["ignored"])[:3]
        advice.append(
            f"{len(stranded)} folder(s) mix ignored and visible files — a "
            "copied .gitignore strands half a folder and the tool sees "
            "only the visible half. `.slopkeeperignore` is not a "
            "gitignore. Worst: "
            + ", ".join(
                f"`{x['dir']}/` ({x['ignored']} ignored / {x['visible']} visible)"
                for x in worst
            )
        )
    cache_like = [
        d
        for d, st in sorted(dotdirs.items())
        if st["files"] > _CACHE_DIR_FILES
        or (
            st["files"] >= _JSONL_MIN_FILES
            and st["jsonl"] / st["files"] > _JSONL_HEAVY_RATIO
        )
    ]
    if cache_like:
        cache_lines = [f"{d}/" for d in cache_like]
        suggested_lines = _dedupe_prefixes(suggested_lines + cache_lines)
        advice.append(
            "Dot-dir(s) that look like caches or transcript dumps: "
            + ", ".join(f"`{d}`" for d in cache_like)
            + " — suggested .slopkeeperignore lines:\n    "
            + "\n    ".join(cache_lines)
        )
    if non_regular:
        named = ", ".join(f"`{x['path']}`" for x in non_regular[:_MAX_LISTED])
        more = len(non_regular) - _MAX_LISTED
        if more > 0:
            named += f" …and {more} more"
        advice.append(
            f"{len(non_regular)} non-regular file(s) (device/FIFO/socket — "
            "often a sandbox deny-list mount): " + named + ". They are "
            "pruned from every walk, the inventory and the fingerprint "
            "(fp5), so sandboxed and unsandboxed runs agree. If one is "
            "real content, fix the mount — the tool cannot read it."
        )
    dangling = dangling_citations(root, walk_files)
    if dangling:
        advice.append(
            f"{len(dangling)}+ dangling citation(s) — path-ish references "
            "that resolve to nothing (already-broken links, half-updated "
            "edits). Informational: fix by hand with edits; the rewrite "
            "pass never touches them. `refs --dangling` lists them."
        )
    # Phase 9: directories built in code — the cost of moving them that
    # no rewrite pass can fully fix (__file__ math is computed, not
    # written). Informational; evidence rows are in code_anchors.
    code_anchors = anchored_dirs(root)
    if code_anchors:
        top = sorted(code_anchors.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        names = ", ".join(f"`{d}`" for d, _ in top[:5])
        advice.append(
            f"Path(s) built in code: {names} — moving them breaks code "
            "references (join() literals are rewritten by apply; "
            "__file__ math is not). See code_anchors for file:line "
            "evidence before planning moves."
        )

    return {
        "root": str(root),
        "stats": s,
        "ignore": {"active": ig.active(), "rules": ig.rule_count},
        "nested_corpora": nested,
        # Phase 9: top dirs by evidence count; three rows of file:line
        # evidence each, full map via the plan-time surface.
        "code_anchors": [
            {"dir": d, "refs": len(ev), "evidence": ev[:3]}
            for d, ev in sorted(
                code_anchors.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )[:10]
        ],
        # Phase 11 diagnostics: the surveys the reports ran by hand.
        "top_dirs": s["top_dir_file_counts"],
        "nested_repos": repos,
        "node_modules": modules,
        "suggested_ignore_lines": suggested_lines,
        "ignore_stranding": stranded[:_MAX_LISTED],
        "non_regular_files": non_regular[:_MAX_LISTED],
        "dotdir_cache_candidates": [{"dir": d, **dotdirs[d]} for d in cache_like],
        "dangling_citations": {
            "count": len(dangling),
            "sample": dangling[:5],
        },
        "reference_scan_active": not (s["truncated"] or n > 800),
        "estimated_inventory_tokens": est_tokens,
        "suggested_strategy": {"id": strategy[0], "why": strategy[1]},
        "advice": advice,
        "next": {
            "action": "inventory",
            "command": f"slopkeeper inventory {root}" + (" --fast" if n > 800 else ""),
            "hint": "Then read a strategy pack (`slopkeeper strategies show <id>`), "
            "write the plan JSON, and validate it.",
        },
    }
