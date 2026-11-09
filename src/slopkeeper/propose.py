# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`propose`: candidate layouts with breakage costs (Phase 10).

The layout fork — by project or by kind? is moving the data sources
worth what it breaks? — becomes data: 2-3 candidate partitions, each
with a move count, a citation-break count, a code-anchor count and an
unresolved estimate. Output is ADVISORY JSON: it never authors a plan;
the agent picks a layout, writes the plan (rules or moves), and the
human still gates.

Each candidate implements one strategy's MECHANICAL core — the part of
the rubric that needs no judgment:

- by-project (projects): "detect project-ness by signals: README,
  package/pyproject files ... never split a project's internal
  structure." Dirs carrying a marker file keep their whole subtree in
  place; everything else has no project signal and is left unresolved.
- by-kind (by-kind): the mechanical floor — every file sorts into its
  kind dir (docs/data/scripts/media, misc for the unknown), structure
  flattened, nothing left unresolved.
- hybrid (hybrid): project subtrees stay whole (the projects rule
  hybrid inherits) and the leftovers kind-sort into hybrid's shared
  buckets (scripts/, data/, research/).

Counts, per candidate, over the same read-only scan:

- move_count: visible files whose dest differs from today.
- citation_break_count: moved files cited from a file that does NOT
  move under the candidate — reference_risk's rule over the corpus
  scan, so without declared citing dirs the count equals the
  equivalent plan's apply-time reference_risks count
  (test_propose_consistent_with_risk_scan). Declared dirs
  (--scan-path / --external-root) add their citations ON TOP: apply's
  risk report is in-corpus by design, but a fork count that ignores
  skills answers the question wrong.
- code_anchor_count: anchored dirs (Phase 9) containing at least one
  moved file — the candidate disturbs them; `__file__` math cannot be
  rewritten by any pass.
- unresolved_estimate: files the candidate leaves with NO dest.

`--scan-path` / `--external-root` add citing directories (an ignored
subtree, a skills store, a cron dir) to the citation scan — a break
count that ignores skills answers the fork question wrong.
"""

from __future__ import annotations

from pathlib import Path

from .anchors import CODE_EXTS, scan_code_anchors
from .ignore import corpus_walk
from .references import (
    CitationIndex,
    _PER_FILE_CAP,
    _basename_pattern,
    _line_targets,
    _text_files_under,
    enriched_risk,
    plan_rewrites,
    reference_risk,
    scan_references_stream,
)
from .strategies import get_strategy

# Kind buckets: the extension -> dest-dir mapping every candidate shares.
# Deliberately coarse — this is a fork-cost estimate, not a classifier.
KIND_EXTS: dict[str, set[str]] = {
    "docs": {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".org",
        ".pdf",
        ".docx",
        ".pptx",
        ".tex",
        ".html",
        ".htm",
    },
    "data": {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".jsonl",
        ".xml",
        ".sql",
        ".log",
    },
    "code": CODE_EXTS,
    "media": {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".mp3",
        ".mp4",
        ".wav",
        ".zip",
        ".gz",
        ".tar",
    },
}
KIND_DIR = {"docs": "docs", "data": "data", "code": "scripts", "media": "media"}

# projects.json's detection signals, as direct dir contents. A marker
# at the corpus ROOT does not make the root a project.
PROJECT_MARKERS = {
    "README.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "setup.py",
    "requirements.txt",
    "pom.xml",
}

CANDIDATES = [
    ("by-project", "projects"),
    ("by-kind", "by-kind"),
    ("hybrid", "hybrid"),
]


def _corpus_rel_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        d = Path(dirpath)
        for fn in filenames:
            out.append((d / fn).relative_to(root))
    return out


def _kind_of(rel: Path) -> str | None:
    ext = rel.suffix.lower()
    for kind, exts in KIND_EXTS.items():
        if ext in exts:
            return kind
    return None


def _project_roots(rel_paths: list[Path]) -> set[str]:
    """Rel dirs (any depth) directly holding a project marker file."""
    roots: set[str] = set()
    for rp in rel_paths:
        if rp.name in PROJECT_MARKERS and len(rp.parts) > 1:
            roots.add(str(rp.parent))
    return roots


def _in_project(rel: Path, roots: set[str]) -> bool:
    parts = rel.parts[:-1]
    return any("/".join(parts[:i]) in roots for i in range(1, len(parts) + 1))


def _dest_map(root: Path, cand_id: str) -> dict[str, str | None]:
    """{rel: dest dir} for every visible file; None = no assignment.

    "Stay" is spelled as the file's own dir, so a dest equal to the
    current dir is not a move — a corpus already in a candidate's shape
    moves nothing under it (the honest move count).
    """
    root = Path(root)
    rel_paths = _corpus_rel_paths(root)
    roots = _project_roots(rel_paths)
    dest: dict[str, str | None] = {}
    for rp in rel_paths:
        rel_str = rp.as_posix()
        kind = _kind_of(rp)
        in_proj = _in_project(rp, roots)
        if cand_id == "by-project":
            d: str | None = str(rp.parent) if in_proj else None
        elif cand_id == "by-kind":
            d = KIND_DIR[kind] if kind else "misc"
        elif cand_id == "hybrid":
            if in_proj:
                d = str(rp.parent)
            else:
                d = KIND_DIR.get(kind) if kind else None
        else:
            raise ValueError(f"unknown candidate {cand_id!r}")
        dest[rel_str] = d
    return dest


def _moved(dest: dict[str, str | None]) -> dict[str, str]:
    """The moving subset: a dest, and not the file's current dir."""
    return {p: d for p, d in dest.items() if d is not None and d != str(Path(p).parent)}


def _merged_refs(
    root: Path, rel_paths: list[Path], index: CitationIndex, extra_dirs: list[Path]
) -> dict[str, list[str]]:
    """The full citation map: corpus citers plus any declared dirs.

    Declared dirs are deduped against the corpus by resolved absolute
    path — a scan dir inside the corpus must not count its files twice.
    Citer strings are absolute for declared dirs (the external flag in
    refs output), corpus-relative for in-tree files.
    """
    refs = scan_references_stream(root)
    refs.pop("stats", None)
    if not extra_dirs:
        return refs
    seen = {(root / rp).resolve() for rp in rel_paths}
    base_pat = _basename_pattern(sorted(index.by_basename))
    for scan_dir in extra_dirs:
        for tp in _text_files_under(scan_dir):
            rp = tp.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                text = tp.read_bytes()[:_PER_FILE_CAP].decode("utf-8", errors="replace")
            except Exception:
                continue
            citing = str(tp)
            for line in text.splitlines():
                for target in _line_targets(line, index, base_pat):
                    bucket = refs.setdefault(target, [])
                    if citing not in bucket:
                        bucket.append(citing)
    return refs


def _scan_context(
    root: Path, extra_dirs: list[Path]
) -> tuple[list[Path], CitationIndex, dict[str, list[str]], dict[str, list[dict]]]:
    """One read-only scan shared by every candidate: paths, index,
    merged citation map, code anchors."""
    root = Path(root)
    rel_paths = _corpus_rel_paths(root)
    index = CitationIndex(rel_paths)
    refs = _merged_refs(root, rel_paths, index, extra_dirs)
    anchors = scan_code_anchors(root, rel_paths, index)
    return rel_paths, index, refs, anchors


def _counts_for(
    dest: dict[str, str | None],
    refs: dict[str, list[str]],
    anchors: dict[str, list[dict]],
    root: Path,
    rel_paths: list[Path],
    index: CitationIndex,
) -> dict[str, int]:
    moved = _moved(dest)
    # Apply's rule exactly: a moved path's destinations count as moved
    # too (chains), so citations from/to the same apply are not breakage.
    moved_set = set(moved) | {f"{d}/{Path(p).name}" for p, d in moved.items()}
    # B2: price the layout WITH the rewrite pass modeled — a citation
    # from a stay-put file is a break only when the pass would NOT fix
    # it (ambiguous line, not_utf8, ...). One linear pass per candidate;
    # the same engine apply's gate uses, so the two surfaces agree.
    chains = {p: (d + "/" + Path(p).name) for p, d in moved.items()}
    _rw, _sk, targets, sites, _unreadable, _skip_rw = plan_rewrites(
        root, index, chains, list(rel_paths)
    )
    breaks = 0
    rewrites_n = 0
    for p in moved:
        if reference_risk(p, refs, moved_set)["external_references"] > 0:
            if enriched_risk(p, refs, moved_set, sites)["uncovered_references"] > 0:
                breaks += 1
    rewrites_n = sum(targets.values())
    anchored_hit: set[str] = set()
    for p in moved:
        parts = Path(p).parts[:-1]
        for i in range(1, len(parts) + 1):
            d = "/".join(parts[:i])
            if d in anchors:
                anchored_hit.add(d)
    return {
        "move_count": len(moved),
        # Breaks the pass does NOT fix (ambiguous lines, not_utf8 citers,
        # ...) — the same uncovered rule apply's gate uses, so the two
        # surfaces agree. The churn the pass absorbs automatically is
        # priced separately as citation_rewrites.
        "citation_break_count": breaks,
        "citation_rewrites": rewrites_n,
        "code_anchor_count": len(anchored_hit),
        "unresolved_estimate": sum(1 for d in dest.values() if d is None),
    }


def build_candidates(root: Path, extra_scan_dirs: list[Path] | None = None) -> dict:
    """The fork as data: one candidate per strategy's mechanical core.

    Read-only — no fingerprint, no writes. `extra_scan_dirs` are
    absolute citing directories (--scan-path / --external-root).
    """
    root = Path(root).resolve()
    rel_paths, index, refs, anchors = _scan_context(
        root, [Path(d).resolve() for d in (extra_scan_dirs or [])]
    )
    candidates = []
    for cand_id, strategy_id in CANDIDATES:
        dest = _dest_map(root, cand_id)
        counts = _counts_for(dest, refs, anchors, root, rel_paths, index)
        pack = get_strategy(strategy_id)
        candidates.append(
            {
                "id": cand_id,
                "strategy": strategy_id,
                "strategy_name": pack["name"],
                "strategy_summary": pack["summary"],
                **counts,
            }
        )
    return {"candidates": candidates, "files": len(rel_paths)}


def candidate_moves(root: Path, cand_id: str) -> dict[str, str]:
    """The moving subset of one candidate: {rel: dest dir}.

    The equivalent-plan construction for tests (and any consumer that
    wants the concrete layout behind a count): moves are
    [{from: rel, to: dest/<basename>}], preserve-name like rule
    expansion.
    """
    return _moved(_dest_map(Path(root).resolve(), cand_id))
