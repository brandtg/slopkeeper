# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Code-anchor detection (reference-rewrites PRD Phase 9).

A directory is ANCHORED when code builds its path at runtime: string
literals inside `join(...)`/`joinpath(...)`/`Path(...)` calls, or
`__file__`/`dirname` parent-chain math. Anchors are the cost of moving
a directory that the citation rewrite pass can only partially fix —
join literals are plain path tokens on text lines (the pass fixes
them), but `dirname(dirname(__file__))` is computed, not written, so
no rewrite can touch it. The flag exists so the human sees that cost
BEFORE approving a move; it never blocks anything.

The scan is deliberately conservative on three axes:

- Only code-suffix files are read (the code subset of TEXT_EXTS) — a
  markdown note mentioning `join("a", "b")` is prose, not code.
- Only literals that name a directory ALREADY IN the corpus (the Phase
  2 `dirs` index) are reported — `join("a", "b")` where `a` is not a
  corpus dir anchors nothing, which kills almost every false positive.
- Evidence is line-level {file, line, text} like `refs` output, so
  every flag is checkable by eye.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

from .ignore import corpus_walk
from .references import CitationIndex, _PER_FILE_CAP

# The code subset of TEXT_EXTS: languages where path segments are
# plausibly built with join()/Path()/dirname() math. Markdown, YAML,
# JSON and friends are citation territory (the reference scan), not
# anchor territory.
CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rb",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".kt",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
}

_MAX_EVIDENCE_PER_DIR = 64  # same posture as _MAX_MENTIONS_PER_TARGET
# Evidence text truncation — the same cap as citation_lines'
# _SNIPPET_CAP (references.py): long enough to author an exact `find`
# from without guessing (the 0.1.9 report's C6 residual: this was 200
# while citation_lines had moved to 2000).
_MAX_TEXT = 2000
_MAX_CALL_SCAN = 2000  # chars scanned into a join(...) call body

_JOIN_CALL_RE = re.compile(r"\b(?:os\.path\.)?join(?:path)?\s*\(")
_PATH_CTOR_RE = re.compile(r"\bPath\s*\(")
_STRING_LIT_RE = re.compile(r"""(["'])(?:\\.|(?!\1).)*\1""")
# dirname(dirname(__file__)) — the run of leading `dirname(` openers
# IS the level count; the closing parens are not needed for detection.
_DIRNAME_CHAIN_RE = re.compile(r"((?:\bdirname\s*\(\s*)+)__file__")
# Path(__file__).parent.parent — and __file__.parent, tolerating
# .resolve() calls anywhere in the chain.
_PARENT_CHAIN_RE = re.compile(
    r"(?:Path\s*\(\s*__file__\s*\)|\b__file__)"
    r"((?:\s*\.\s*(?:resolve\s*\(\s*\)|parent))+)"
)


def _anchored_from_levels(parts: list[str], levels: int) -> str | None:
    """The dir `levels` runtime steps up from a file in `parts`.

    dirname(__file__) (1 level) is the file's own directory;
    dirname(dirname(__file__)) (2) is its parent — the PRD's example.
    Anything reaching the corpus root or above anchors nothing: "" is
    not a dir in the index.
    """
    keep = len(parts) - (levels - 1)
    if keep <= 0:
        return None
    return "/".join(parts[:keep])


def _norm_dir(raw: str) -> str:
    return "/".join(p for p in raw.split("/") if p)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _call_anchors(
    text: str,
    regex: re.Pattern,
    index: CitationIndex,
    starts: list[int],
) -> list[tuple[str, int]]:
    """(dir, call-line) for every corpus dir a literal-bearing call anchors.

    Depth-1 string literals are collected in order; ADJACENT literals
    (only commas and whitespace between) concatenate into a path run —
    a non-literal argument between them breaks the run
    (`join("slack", flag, "data")` does not build slack/data). Every
    prefix-concatenation of a run that names a corpus dir anchors it:
    `join(ROOT, "slack", "data")` breaks if `slack` moves just as
    surely as if `slack/data` does. Evidence sits at the call's own
    line; the scan walks paren bodies across lines, budget-capped.
    """
    out: list[tuple[str, int]] = []
    for m in regex.finditer(text):
        open_paren = m.end() - 1
        line_no = bisect.bisect_right(starts, open_paren)
        items: list = []  # ("lit", text) at depth 1, or "sep"
        depth = 0
        i = open_paren
        end = min(len(text), open_paren + _MAX_CALL_SCAN)
        while i < end:
            ch = text[i]
            if ch in "\"'":
                lit = _STRING_LIT_RE.match(text, i)
                if lit is None:  # unterminated; stop scanning this call
                    break
                if depth == 1:
                    items.append(("lit", lit.group(0)[1:-1]))
                i = lit.end()
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and ch not in " \t\r\n,":
                items.append("sep")
            i += 1
        runs: list[list[str]] = []
        current: list[str] = []
        for item in items:
            if item == "sep":
                if current:
                    runs.append(current)
                    current = []
            else:
                current.append(item[1])
        if current:
            runs.append(current)
        for run in runs:
            parts: list[str] = []
            for raw_lit in run:
                piece = _norm_dir(raw_lit)
                if not piece:
                    continue
                parts.extend(piece.split("/"))
                candidate = "/".join(parts)
                if candidate in index.dirs:
                    out.append((candidate, line_no))
    return out


def _scan_line_for_file_math(line: str, file_dir_parts: list[str]) -> list[str]:
    """Dirs anchored by __file__ parent-chain math on one line."""
    anchored: list[str] = []
    for m in _DIRNAME_CHAIN_RE.finditer(line):
        levels = m.group(1).count("dirname")
        d = _anchored_from_levels(file_dir_parts, levels)
        if d:
            anchored.append(d)
    for m in _PARENT_CHAIN_RE.finditer(line):
        levels = m.group(1).count("parent")
        d = _anchored_from_levels(file_dir_parts, levels)
        if d:
            anchored.append(d)
    return anchored


def scan_code_anchors(
    root: Path, rel_paths: list[Path], index: CitationIndex
) -> dict[str, list[dict]]:
    """{dir: [{file, line, text}]} for every code-anchored corpus dir.

    `rel_paths` is the corpus file list (the same walk the index was
    built from); only its CODE_EXTS members are READ. Line-level
    evidence, deduped per dir, capped at _MAX_EVIDENCE_PER_DIR — the
    cap is a display bound, the flag is informational.
    """
    root = Path(root)
    anchors: dict[str, list[dict]] = {}
    for tp in rel_paths:
        if tp.suffix.lower() not in CODE_EXTS:
            continue
        rel_str = tp.as_posix()
        file_dir_parts = list(tp.parts[:-1])
        try:
            raw = (root / tp).read_bytes()[:_PER_FILE_CAP]
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        starts = _line_starts(text)
        lines = text.splitlines()
        found: dict[str, set[tuple[int, str]]] = {}
        for regex in (_JOIN_CALL_RE, _PATH_CTOR_RE):
            for d, line_no in _call_anchors(text, regex, index, starts):
                found.setdefault(d, set()).add(
                    (line_no, lines[line_no - 1].strip()[:_MAX_TEXT])
                )
        for lineno, line in enumerate(lines, 1):
            for d in _scan_line_for_file_math(line, file_dir_parts):
                found.setdefault(d, set()).add((lineno, line.strip()[:_MAX_TEXT]))
        for d, sites in found.items():
            bucket = anchors.setdefault(d, [])
            for line_no, line_text in sorted(sites):
                if len(bucket) >= _MAX_EVIDENCE_PER_DIR:
                    break
                bucket.append({"file": rel_str, "line": line_no, "text": line_text})
    return anchors


def _corpus_rel_paths(root: Path) -> list[Path]:
    """The shared walk as a rel-path list (anchors need the full set:
    the dirs index must know every corpus directory, code file or not)."""
    out: list[Path] = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        d = Path(dirpath)
        for fn in filenames:
            out.append((d / fn).relative_to(root))
    return out


def anchored_dirs(root: Path) -> dict[str, list[dict]]:
    """Whole-corpus anchor map (doctor, plan validate, render)."""
    root = Path(root)
    rel_paths = _corpus_rel_paths(root)
    return scan_code_anchors(root, rel_paths, CitationIndex(rel_paths))


def anchored_move_risks(root: Path, plan: dict) -> list[dict]:
    """Informational code_anchored rows for validate/render (Phase 9).

    A move is flagged when its source IS an anchored dir or CONTAINS
    one (moving `slack` moves the anchored `slack/data` with it).
    Moving one file out of an anchored dir is not flagged — the dir
    itself stays put. The rewrite pass still fixes literal citations;
    this is the cost the pass cannot fix (`__file__` math), visible
    before the human decides. Never blocks.
    """
    moves = plan.get("moves") or []
    if not moves:
        return []
    anchors = anchored_dirs(root)
    if not anchors:
        return []
    rows = []
    for m in moves:
        src = str(m.get("from", "")).rstrip("/")
        if not src:
            continue
        hit = sorted(a for a in anchors if a == src or a.startswith(src + "/"))
        if hit:
            rows.append(
                {
                    "from": str(m.get("from", "")),
                    "to": str(m.get("to", "")),
                    "dirs": hit,
                    "sites": sum(len(anchors[a]) for a in hit),
                }
            )
    return rows
