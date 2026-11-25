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

from . import kinds, scan
from .models import SlopKeeperError, resolve_chains
from .references import CitationIndex, _PER_FILE_CAP, _post_move_path

# The code subset of the scannable set: languages where path segments
# are plausibly built with join()/Path()/dirname() math. Markdown, YAML,
# JSON and friends are citation territory (the reference scan), not
# anchor territory. This was a private copy of inventory's list — the
# two drifted only because nothing stopped them; kinds.CODE_EXTS is the
# single table now (P3/R7) and this is the alias every importer knows.
CODE_EXTS = kinds.CODE_EXTS

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

    Thin projection of `_call_runs` + `_run_dirs`: the evidence surface
    needs to know WHICH dir a call builds, not which literals did it.
    """
    out: list[tuple[str, int]] = []
    for run in _call_runs(text, regex, starts):
        for matched, _count, _hole in _run_dirs(run, index):
            out.append((matched, run.line))
    return out


class _Run:
    """One run of ADJACENT string literals inside a call's argument list.

    `lits` holds (unquoted text, quote char, span start, span end) per
    literal, in source order. The span is what `suggest_anchor_edits`
    turns into an `edits` entry: the exact source text of the literals
    that compose a moving path, so the `find` it authors is byte-exact
    instead of a reconstruction that may not match.
    """

    __slots__ = ("lits", "start", "end", "line")

    def __init__(self, lits: list[tuple[str, str, int, int]], line: int):
        self.lits = lits
        self.line = line
        self.start = lits[0][2]
        self.end = lits[-1][3]


_WILD_RUN_RE = re.compile(r"\{[^{}]*\}")
_WILD_CACHE: dict[tuple[str, ...], re.Pattern | None] = {}
_WILD_CACHE_MAX = 4096


def _is_wild_segment(seg: str) -> bool:
    """A segment holding an f-string `{...}` hole: it stands for SOME text.

    The audit's second anchor fragility: `join(base, "releases", f"v{v}",
    "data")` used to be read as the literal path `releases/v{v}/data`,
    which names no directory, so the call anchored nothing and the gate
    stayed silent about code that builds exactly that path at runtime.
    """
    return bool(_WILD_RUN_RE.search(seg))


def _seg_pattern(seg: str) -> str:
    """One path segment as a regex body: literal text, `{...}` holes as `[^/]+`.

    The hole is a wildcard, but the text AROUND it is not: `f"v{v}"`
    matches `v1.2`, not `sub`. Reading the whole segment as "anything"
    would anchor every directory of the same depth, which is the difference
    between a useful flag and noise nobody reads.
    """
    out: list[str] = []
    pos = 0
    for m in _WILD_RUN_RE.finditer(seg):
        out.append(re.escape(seg[pos : m.start()]))
        out.append("[^/]+")
        pos = m.end()
    out.append(re.escape(seg[pos:]))
    return "".join(out)


def _wild_pattern(parts: tuple[str, ...]) -> re.Pattern | None:
    """A whole-path regex for a run containing holes, or None if unusable."""
    if parts in _WILD_CACHE:
        return _WILD_CACHE[parts]
    if all(not _WILD_RUN_RE.sub("", p).strip() for p in parts):
        # Every segment is nothing but a hole: the run constrains nothing,
        # and matching the whole corpus against it is noise, not evidence.
        pattern = None
    else:
        pattern = re.compile("/".join(_seg_pattern(p) for p in parts))
    if len(_WILD_CACHE) < _WILD_CACHE_MAX:
        _WILD_CACHE[parts] = pattern
    return pattern


def _match_dir(parts: list[str], index: CitationIndex) -> str | None:
    """The corpus directory a (possibly hole-bearing) segment run names."""
    if not any(_is_wild_segment(p) for p in parts):
        return index.match_dir("/".join(parts))
    if len(parts) < 2:
        # A lone hole anchors nothing: `join(base, f"x")` could be any
        # single segment, and calling every one of them a match would flag
        # the whole corpus.
        return None
    pattern = _wild_pattern(tuple(parts))
    if pattern is None:
        return None
    for d in index.dirs:
        if pattern.fullmatch(d):
            return d
    return None


def _run_dirs(run: _Run, index: CitationIndex) -> list[tuple[str, int, bool]]:
    """Every (corpus dir, literals-consumed, matched-through-a-hole) prefix.

    `join(ROOT, "slack", "data")` breaks if `slack` moves just as surely
    as if `slack/data` does, so every prefix counts, not just the longest.
    The literal count is what an `edits` entry replaces, and the
    through-a-hole flag is what keeps an f-string anchor out of the
    rewrite business: the directory it names is a runtime value, so no
    literal text can stand in for it.
    """
    out: list[tuple[str, int, bool]] = []
    parts: list[str] = []
    for i, (raw, _q, _s, _e) in enumerate(run.lits, 1):
        piece = _norm_dir(raw)
        if not piece:
            continue
        parts.extend(piece.split("/"))
        matched = _match_dir(parts, index)
        if matched is not None:
            out.append((matched, i, any(_is_wild_segment(p) for p in parts)))
    return out


def _call_runs(text: str, regex: re.Pattern, starts: list[int]) -> list[_Run]:
    """Every adjacent-literal run inside every `regex(...)` call body.

    Depth-1 string literals are collected in order; ADJACENT literals
    (only commas and whitespace between) concatenate into a path run — a
    non-literal argument between them breaks the run
    (`join("slack", flag, "data")` does not build slack/data).

    The scan is PER LINE, which is the audit's first anchor fragility: an
    unterminated quote used to `break` out of the whole call body, so one
    apostrophe in a comment — `join("sub", "data")  # don't move this` —
    silently un-anchored the directory. Now a quote that does not close on
    its own line is read as prose: the run breaks, the rest of the line is
    skipped (that is where a comment lives), and the call scan CONTINUES on
    the next line. The one case that still stops the scan is the line that
    carries the call's closing paren: past it the scan would be reading the
    enclosing scope's literals and billing them to this call.
    """
    out: list[_Run] = []
    for m in regex.finditer(text):
        open_paren = m.end() - 1
        line_no = bisect.bisect_right(starts, open_paren)
        items: list = []  # ("lit", text, quote, start, end) at depth 1, or "sep"
        depth = 0
        i = open_paren
        end = min(len(text), open_paren + _MAX_CALL_SCAN)
        while i < end:
            ch = text[i]
            if ch in "\"'":
                lit = _STRING_LIT_RE.match(text, i)
                if lit is None:
                    # Unterminated on this line: prose, not a literal.
                    items.append("sep")
                    nl = text.find("\n", i, end)
                    if nl == -1 or ")" in text[i:nl]:
                        break  # the call's tail (or its paren) is on that line
                    i = nl + 1
                    continue
                if depth == 1:
                    body, quote = lit.group(0)[1:-1], lit.group(0)[0]
                    items.append(("lit", body, quote, lit.start(), lit.end()))
                i = lit.end()
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and ch not in " \t\r\n,":
                # A string prefix (f/r/b/u) sitting directly on a quote is
                # part of THAT literal, not a separator between arguments:
                # without this, `join(base, f"sub", "data")` broke the run
                # at the `f` and the anchor vanished.
                j = i
                while j < end and text[j] in "frbuFRBU":
                    j += 1
                if j > i and j < end and text[j] in "\"'":
                    i = j
                    continue
                items.append("sep")
            i += 1
        runs: list[list] = []
        current: list = []
        for item in items:
            if item == "sep":
                if current:
                    runs.append(current)
                    current = []
            else:
                current.append(item[1:])
        if current:
            runs.append(current)
        for lits in runs:
            out.append(_Run(lits, line_no))
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
        if not kinds.is_code(tp):
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
    """The corpus file list (anchors need the full set: the dirs index
    must know every corpus directory, code file or not).

    The single-pass scanner's names-only view — anchors used to run its
    own `os.walk` to get what every other surface already had.
    """
    return scan.corpus_rel_paths(root)


def anchored_dirs(
    root: Path,
    *,
    rel_paths: list[Path] | None = None,
    index: CitationIndex | None = None,
) -> dict[str, list[dict]]:
    """Whole-corpus anchor map (doctor, plan validate, render).

    `rel_paths`/`index` let a caller that already holds the corpus (the
    one-pass `doctor`) hand the paths over instead of paying for another
    traversal; the default walks through the shared scanner.
    """
    root = Path(root)
    if rel_paths is None:
        rel_paths = _corpus_rel_paths(root)
    if index is None:
        index = CitationIndex(rel_paths)
    return scan_code_anchors(root, rel_paths, index)


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


# --- suggested edits: anchors the rewrite pass cannot reach (P6 Slice B) ------
#
# The pass already fixes a path literal written WHOLE: `join(base,
# "sub/data")` contains the token `sub/data`, the citation engine sees it,
# and the move rewrites it. What it cannot see — and never will — is the
# path that exists only as a RUN of literals: `join(base, "sub", "data")`
# holds no path token anywhere, so no citation matches it and the call
# keeps building the old path after the move. That is exactly the case the
# anchor scan exists to detect, and until now the detection had no
# follow-through: the gate told the human "this directory is built in code"
# and left the fix to be written by hand.
#
# So this turns the detection into a PROPOSAL. For a run whose literal
# prefix composes a directory the plan moves, it authors a plan `edits`
# entry replacing exactly those literals with one literal holding the new
# path — byte-exact `find` sliced from the source span, so the tool's
# existing exactly-once edit validation (`check_edits_after_rewrites`) does
# the safety work it already does for every other edit: a `find` that does
# not occur exactly once in the post-rewrite text refuses the plan instead
# of the tool guessing.
#
# Two rules keep it conservative, and both are counted rather than silent:
#
#   - a run containing an f-string hole (`f"v{v}"`) is never turned into an
#     edit. A wildcard is fine for DETECTING that code builds this path
#     (any segment matches); it is not fine for REWRITING one, because the
#     text to write depends on a runtime value. It stays an anchor row.
#   - at most one edit per run, for the LONGEST matching prefix: `join("a",
#     "b")` where both `a` and `a/b` move yields one edit, not two
#     overlapping finds the pass would apply on top of each other.
#
# Everything produced here is a suggestion on a plan the agent edits and a
# human approves — `draft --suggest-code-edits`, off by default.

_MAX_SUGGESTED_EDITS = 500


def _move_covering(target: str, srcs: set[str]) -> str | None:
    """The plan move that relocates `target` — itself or an ancestor dir.

    Deepest first, mirroring `references._move_covering`: when a plan
    moves both `sub/` and `sub/data/`, the entry that governs `sub/data`
    is the one the executor orders last and the one the literals must
    follow.
    """
    parts = target.split("/")
    for i in range(len(parts), 0, -1):
        cand = "/".join(parts[:i])
        if cand in srcs:
            return cand
    return None


def suggest_anchor_edits(
    root: Path,
    plan: dict,
    *,
    rel_paths: list[Path] | None = None,
    index: CitationIndex | None = None,
    limit: int = _MAX_SUGGESTED_EDITS,
) -> tuple[list[dict], dict]:
    """Plan `edits` entries that move the path literals with the files.

    Returns (edits, stats). Each edit is {file, find, replace, reason} in
    exactly the shape `validate_plan` normalizes, so a draft carrying them
    passes the same gates as a hand-authored one.

    `stats` is the honest accounting of what was NOT emitted:
    `runs` scanned, `suggested`, `skipped_wildcard` (an f-string hole: the
    directory is a runtime value, detected but never rewritable),
    `skipped_multiline` (a find that would span lines — the rewrite pass
    protects an edit's lines by matching `find` WITHIN a line, so a
    multi-line find could be applied twice), `skipped_ambiguous` (a find
    that does not occur exactly once in the file), `skipped_quote` (no
    usable quote style), `capped`.
    """
    root = Path(root)
    empty = {
        "runs": 0,
        "suggested": 0,
        "skipped_wildcard": 0,
        "skipped_multiline": 0,
        "skipped_ambiguous": 0,
        "skipped_quote": 0,
        "capped": False,
    }
    moves = [m for m in (plan.get("moves") or []) if isinstance(m, dict)]
    if not moves:
        return [], empty
    try:
        chains = resolve_chains(moves)
    except SlopKeeperError:
        return [], empty  # a malformed plan is validate's problem, not ours
    srcs = set(chains)
    if not srcs:
        return [], empty
    if rel_paths is None:
        rel_paths = scan.corpus_rel_paths(root)
    if index is None:
        index = CitationIndex(rel_paths)
    edits: list[dict] = []
    stats = dict(empty)
    seen: set[tuple[str, str]] = set()
    for tp in rel_paths:
        if not kinds.is_code(tp):
            continue
        rel_str = tp.as_posix()
        try:
            raw = (root / tp).read_bytes()[:_PER_FILE_CAP]
            text = raw.decode("utf-8")  # strict: an edit rewrites these bytes
        except (OSError, UnicodeDecodeError):
            continue
        starts = _line_starts(text)
        for regex in (_JOIN_CALL_RE, _PATH_CTOR_RE):
            for run in _call_runs(text, regex, starts):
                stats["runs"] += 1
                matches = _run_dirs(run, index)
                if not matches:
                    continue
                picked = _pick_run_edit(run, matches, chains, srcs, text)
                if picked is None:
                    if _run_has_wildcard(run):
                        stats["skipped_wildcard"] += 1
                    continue
                find, replace, old_dir, new_dir, why = picked
                if why:
                    stats[why] += 1
                    continue
                if find == replace or (rel_str, find) in seen:
                    continue
                if text.count(find) != 1:
                    stats["skipped_ambiguous"] += 1
                    continue
                if len(edits) >= limit:
                    stats["capped"] = True
                    return edits, stats
                seen.add((rel_str, find))
                edits.append(
                    {
                        "file": rel_str,
                        "find": find,
                        "replace": replace,
                        "reason": (
                            f"anchor: {old_dir} is built from literals in "
                            f"code and this plan moves it to {new_dir} "
                            "(--suggest-code-edits)"
                        ),
                    }
                )
                stats["suggested"] += 1
    return edits, stats


def _run_has_wildcard(run: _Run) -> bool:
    """True when any literal of the run carries an f-string brace hole."""
    return any(
        _is_wild_segment(p)
        for raw, _q, _s, _e in run.lits
        for p in _norm_dir(raw).split("/")
        if p
    )


def _pick_run_edit(
    run: _Run,
    matches: list[tuple[str, int, bool]],
    chains: dict[str, str],
    srcs: set[str],
    text: str,
) -> tuple[str, str, str, str, str] | None:
    """(find, replace, old_dir, new_dir, skip_reason) for one run.

    The LONGEST literal prefix that names a moving directory wins (see the
    module notes for why one, not several). The literals collapse into a
    single literal holding the post-move path: an extra argument to
    `join`/`joinpath`/`Path` is harmless, and one literal is the only form
    whose `find` is guaranteed to be exact source text.

    A prefix matched THROUGH an f-string hole is refused here even though
    the concrete directory it named (`releases/v1.2/data`, say) contains no
    braces at all: the value came from the disk, not from the source, and
    writing it back as a literal would freeze an interpolation — the worst
    kind of edit this tool could make, because it looks right and passes
    every text check.
    """
    for matched, count, through_hole in reversed(matches):
        if through_hole:
            continue  # try a shorter, hole-free prefix; else it is not rewritable
        src = _move_covering(matched, srcs)
        if src is None:
            continue
        new_dir = _post_move_path(matched, chains)
        if new_dir == matched:
            continue
        lits = run.lits[:count]
        find = text[lits[0][2] : lits[-1][3]]
        if "\n" in find:
            return (find, "", matched, new_dir, "skipped_multiline")
        quote = lits[0][1]
        if quote in new_dir:
            return (find, "", matched, new_dir, "skipped_quote")
        return find, f"{quote}{new_dir}{quote}", matched, new_dir, ""
    return None
