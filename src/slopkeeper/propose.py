# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`propose`: candidate layouts with breakage costs (Phase 10, P4).

The layout fork — by project, by role, by topic, or by kind? is moving
the data sources worth what it breaks? — becomes data: five candidate
partitions, each with a move count, a citation-break count, a
code-anchor count, an unresolved estimate and an orphaned estimate.
Output is ADVISORY JSON: it never authors a plan; the agent picks a
layout, writes the plan (rules or moves), and the human still gates.

Each candidate implements one strategy's MECHANICAL core — the part of
the rubric that needs no judgment:

- by-project (projects): "detect project-ness by signals: README,
  package/pyproject files ... never split a project's internal
  structure." Dirs carrying a marker file keep their whole subtree in
  place; everything else has no project signal and is left unresolved.
- by-role (role-bots): "what IS this pile" as a bot workspace
  collection. A top-level dir directly holding its own AGENTS.md or
  README.md is a standing role (the rubric gives every role folder an
  AGENTS.md — its presence is the only judgment-free signal a tool can
  read); that whole subtree stays. Files under no role dir have no
  role signal → unresolved, never guessed: the unresolved count IS the
  answer about the pile.
- by-topic (topics): "partition by subject" as a deterministic pass —
  the TF-IDF content clusters (filename + title + head tokens) name
  `topics/<label>` folders; unclustered files stay unresolved. The
  topics rubric is judgment-heavy, so this is the comparison-cost
  estimate, never a suggested plan.
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
- orphaned_estimate: the same files counted as a cost — where judgment
  still has to land before this layout is real. by-kind and hybrid
  orphan nothing (0) and the number says so rather than pretending the
  layout is free; by-role and by-topic orphan exactly the files that
  carry no role signal or share no vocabulary.

`--scan-path` / `--external-root` add citing directories (an ignored
subtree, a skills store, a cron dir) to the citation scan — a break
count that ignores skills answers the fork question wrong.

An unknown candidate name is refused (propose_unknown_candidate)
rather than silently returning no moves. by-topic clusters with the
same candidate blocking `clusters` uses past MAX_FILES, so `propose`
scales with the corpus instead of hitting the all-pairs wall; below
the cap it is the exact matcher, unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import kinds
from .anchors import scan_code_anchors
from .models import SlopKeeperError
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
from .scan import corpus_rel_paths
from .strategies import get_strategy

# Kind buckets: the extension -> dest-dir mapping every candidate shares.
# Deliberately coarse — this is a fork-cost estimate, not a classifier.
# The sets are kinds.py's now (P3/R7): one owner per extension, and the
# buckets are disjoint, so an extension cannot be "data" here and "code"
# in inventory in the same run. That was real: `.sql` sat in both, and
# the dict order silently decided this module's answer. It is code
# (CODE_EXTS says so) and lands in scripts/ now.
KIND_EXTS: dict[str, frozenset[str]] = kinds.KIND_EXTS
KIND_DIR = {"docs": "docs", "data": "data", "code": "scripts", "media": "media"}

# Projects.json's detection signals, as direct dir contents. A marker
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

# role-bots.json's mechanical floor: a top-level dir that already carries
# its own AGENTS.md or README.md is a standing role (the rubric says each
# role folder gets an AGENTS.md; presence of one is the only judgment-free
# signal a tool can read). Everything else has no role signal → unresolved.
ROLE_MARKERS = {"README.md", "AGENTS.md"}

# A topic folder label, slugified to a single path segment.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _NON_SLUG_RE.sub("-", text.lower()).strip("-")[:40] or "topic"


CANDIDATES = [
    ("by-project", "projects"),
    ("by-role", "role-bots"),
    ("by-topic", "topics"),
    ("by-kind", "by-kind"),
    ("hybrid", "hybrid"),
]


def _role_dirs(rel_paths: list[Path]) -> set[str]:
    """Top-level dirs directly holding an AGENTS.md/README.md (role-bots)."""
    return {
        rp.parts[0]
        for rp in rel_paths
        if len(rp.parts) == 2 and rp.name in ROLE_MARKERS
    }


def _topic_dirs(rel_paths: list[Path], min_size: int = 2) -> dict[str, str]:
    """{rel: "topics/<label>"} from the similarity clusters ≥min_size.

    The topics rubric's mechanical floor — "partition by subject" — as a
    deterministic pass: TF-IDF content clusters (filename + title + head
    tokens) name the topics, one folder each. Judgment-heavy by nature, so
    this is the comparison-cost estimate, never a suggested plan. Files in
    no cluster are left unresolved by the caller.

    Past MAX_FILES the matcher runs candidate-blocked (same as
    `clusters`, P4): blocked pairs are a subset of all-pairs, so a topic
    folder is never a fabricated merge, and `propose` scales with the
    corpus instead of refusing with similarity_cap.
    """
    from .similar import (
        MAX_FILES,
        _neighbor_adjacency,
        build_vectors,
        candidate_pairs,
        cluster_label,
        similarity_clusters,
    )

    files = [{"path": rp.as_posix()} for rp in rel_paths]
    if not files:
        return {}
    vecs = build_vectors(files)
    adjacency = (
        _neighbor_adjacency(candidate_pairs(vecs)) if len(vecs) > MAX_FILES else None
    )
    clusters = similarity_clusters(vecs, adjacency=adjacency)
    out: dict[str, str] = {}
    for c in sorted(clusters, key=lambda c: (-len(c), c[0])):
        if len(c) < min_size:
            continue
        label = _slug(cluster_label(c) or "topic")
        for rel in c:
            out[rel] = f"topics/{label}"
    return out


def _corpus_rel_paths(root: Path) -> list[Path]:
    """The corpus file list, from the one traversal (scan.py)."""
    return corpus_rel_paths(root)


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

    Unknown candidate → propose_unknown_candidate (a caller typo is loud,
    not a silent empty candidate).
    """
    root = Path(root)
    rel_paths = _corpus_rel_paths(root)
    roots = _project_roots(rel_paths)
    role_dirs = _role_dirs(rel_paths) if cand_id == "by-role" else set()
    topics = _topic_dirs(rel_paths) if cand_id == "by-topic" else {}
    dest: dict[str, str | None] = {}
    for rp in rel_paths:
        rel_str = rp.as_posix()
        kind = _kind_of(rp)
        in_proj = _in_project(rp, roots)
        if cand_id == "by-project":
            d: str | None = str(rp.parent) if in_proj else None
        elif cand_id == "by-role":
            # A file under a top-level role dir (AGENTS.md/README.md) is
            # the role's — its whole subtree stays. No role signal → the
            # rubric's own rule: unresolved, never guessed.
            d = str(rp.parent) if rp.parts and rp.parts[0] in role_dirs else None
        elif cand_id == "by-topic":
            # Clustered files land in their topic folder; unclustered are
            # unresolved (the topics rubric is judgment-heavy — this is the
            # mechanical comparison core, not a suggested plan).
            d = topics.get(rel_str, None)
        elif cand_id == "by-kind":
            d = KIND_DIR[kind] if kind else "misc"
        elif cand_id == "hybrid":
            if in_proj:
                d = str(rp.parent)
            else:
                d = KIND_DIR.get(kind) if kind else None
        else:
            raise SlopKeeperError(
                "propose_unknown_candidate",
                f"unknown candidate {cand_id!r}",
                "Known candidates: "
                + ", ".join(cid for cid, _ in CANDIDATES)
                + " (see `slopkeeper propose`).",
            )
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
    unresolved_n = sum(1 for d in dest.values() if d is None)
    return {
        "move_count": len(moved),
        # Breaks the pass does NOT fix (ambiguous lines, not_utf8 citers,
        # ...) — the same uncovered rule apply's gate uses, so the two
        # surfaces agree. The churn the pass absorbs automatically is
        # priced separately as citation_rewrites.
        "citation_break_count": breaks,
        "citation_rewrites": rewrites_n,
        "code_anchor_count": len(anchored_hit),
        "unresolved_estimate": unresolved_n,
        # P4 (7): "files this layout would orphan unresolved" — the count
        # the agent needs to read the fork honestly. For the mechanical
        # cores (role-bots, topics) this is where judgment still has to
        # land; a by-kind/hybrid layout orphans nothing, and the number
        # says so rather than pretending the layout is free.
        "orphaned_estimate": unresolved_n,
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
    expansion. An unknown candidate is refused (propose_unknown_candidate)
    with the valid list.
    """
    return _moved(_dest_map(Path(root).resolve(), cand_id))
