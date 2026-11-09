# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Shared test helpers: corpus factory, plan writers, temp-dir case base."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slopkeeper.models import SlopKeeperError, corpus_fingerprint

# A small corpus shaped like a real messy workspace (relpath -> content).
DEFAULT_CORPUS = {
    "docs/canonical.md": "# Canonical\n",
    "acme_dossier.md": "# Acme dossier\n\nflows.\n",
    "globex_dossier.md": "# Globex family\n\nmarket share.\n",
    "handoff-tp2.md": "# Handoff TP2\n\nlanes\n",
    "ledger_wb.html": "<html>junk</html>\n",
    "WATCHLIST.md": "# Watchlist\n\nSee acme_dossier.md for flows. "
    "Also globex_dossier.md.\n",
    "spark-1.2.3/README.md": "# runbook\n",
    "lead-gen/README.md": "# pipeline\n",
    ".obsidian/workspace.json": "{}",  # hidden, must be skipped
}


def write(path: Path, text: str = "") -> Path:
    """Write text to path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def touch(path: Path) -> Path:
    """Create path (empty) or refresh its mtime if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def build_corpus(root: Path, spec: dict | None = None) -> Path:
    """Materialize the default corpus at root; spec overrides/adds files."""
    files = dict(DEFAULT_CORPUS)
    if spec:
        files.update(spec)
    for rel, text in files.items():
        write(root / rel, text)
    return root


def snapshot_tree(root: Path):
    """Byte-exact tree snapshot over EVERY entry: ({relpath: bytes}, {dir relpaths}).

    Atomicity/undo tests compare before/after to prove the disk is
    untouched. Nothing is skipped: dot-dirs are corpus content (fp4),
    so hiding them would blind the comparison to changes inside e.g.
    .claude/. Tool residue (.slopkeeper/, tend.lock) is removed by the
    callers via prune_slopkeeper, not by a hidden filter here.
    """
    files: dict[str, bytes] = {}
    dirs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root).as_posix()
        dirs.add(rel)
        for fn in filenames:
            p = Path(dirpath) / fn
            files[(Path(rel) / fn).as_posix()] = p.read_bytes()
    return files, dirs


def prune_slopkeeper(snapshot):
    """Drop the tool's own .slopkeeper residue from a tree snapshot.

    §5.5 blesses the empty tend.lock dotfile remaining after ANY execute
    attempt (the lock is taken before the pre-flight, and the OS releases
    it on process death — the file itself stays). The atomicity
    comparisons assert the CORPUS is byte-identical, not the tool's
    dotdir, so .slopkeeper/ is pruned from both sides.
    """
    files, dirs = snapshot
    files = {
        k: v
        for k, v in files.items()
        if not (k == ".slopkeeper" or k.startswith(".slopkeeper/"))
    }
    dirs = {d for d in dirs if not (d == ".slopkeeper" or d.startswith(".slopkeeper/"))}
    return files, dirs


def expect_code(self, cm_body, code: str):
    """Assert a SlopKeeperError with exactly this machine code is raised."""
    with self.assertRaises(SlopKeeperError) as cm:
        cm_body()
    self.assertEqual(
        cm.exception.code,
        code,
        f"expected {code}, got {cm.exception.code}: {cm.exception.message}",
    )


def write_plan(root: Path, **overrides) -> Path:
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
        "moves": [
            {
                "from": "acme_dossier.md",
                "to": "research/dossiers/acme_dossier.md",
                "reason": "dossier research",
            },
            {
                "from": "globex_dossier.md",
                "to": "research/dossiers/globex_dossier.md",
                "reason": "dossier research",
            },
            {
                "from": "handoff-tp2.md",
                "to": "spark-1.2.3/handoff-tp2.md",
                "reason": "spark handoff",
            },
            {
                "from": "ledger_wb.html",
                "to": "archive/scraps/ledger_wb.html",
                "reason": "scrap",
            },
        ],
        "keep": ["WATCHLIST.md", "docs", "lead-gen", "spark-1.2.3"],
        "unresolved": [],
        "folders": [
            {
                "path": "research/dossiers",
                "kind": "case",
                "role": "Market research dossiers.",
                "state_md": False,
            }
        ],
    }
    plan.update(overrides)
    p = root.parent / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    return p


def write_edit_plan(root: Path, **overrides) -> Path:
    # WATCHLIST.md stays and cites acme_dossier.md; the plan moves
    # the dossier and rewrites the citation.
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "based_on": {"corpus_fingerprint": corpus_fingerprint(root)},
        "moves": [
            {
                "from": "acme_dossier.md",
                "to": "research/acme_dossier.md",
                "reason": "reference pile",
            }
        ],
        "edits": [
            {
                "file": "WATCHLIST.md",
                "find": "acme_dossier.md",
                "replace": "research/acme_dossier.md",
                "reason": "fix citation",
            }
        ],
        "keep": [
            "globex_dossier.md",
            "handoff-tp2.md",
            "ledger_wb.html",
            "spark-1.2.3",
            "docs",
            "lead-gen",
        ],
        "unresolved": [],
        "folders": [],
    }
    plan.update(overrides)
    p = root.parent / "plan-edit.json"
    p.write_text(json.dumps(plan))
    return p


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="slopkeeper-test-"))
        self.root = build_corpus(self.tmp / "corpus")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
