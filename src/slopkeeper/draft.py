# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Draft plans: a deterministic first pass at the delta tend.

For the re-tend loop the agent's judgment is mostly "which existing
folder does this new file belong in". That has a mechanical core: the
new file's nearest already-filed neighbor (TF-IDF cosine, same vectors
as `clusters`) is usually decisive. `draft` proposes those moves with
the neighbor and score as the reason, and marks everything else
unresolved — honestly, with the reason.

The sweep is RECURSIVE by default: new files at any depth are draftable
slop (a file dumped into docs/ can belong in research/ just as much as
a root file does), with `depth` to bound it — N considers only files
within N directory levels of the root, the same convention as
inventory/clusters. Only the AGENTS.md/STATE.md/INDEX.md tool layer is
never touched.

The output is a PARTIAL plan (`"partial": true`) with zero folders and
zero edits: it only files new slop into folders that already exist.
The agent edits it; the human still gates it. Draft is a guess with a
citation, never a decision.

Zero dependencies; no LLM; no network.
"""

from __future__ import annotations

from pathlib import Path

from .models import SlopKeeperError, corpus_fingerprint
from .similar import (
    DEFAULT_T1,
    MAX_FILES,
    MEMORY_MAX_FILES,
    _cosine,
    _norm,
    build_vectors,
    build_vectors_cached,
    candidate_neighbors,
    identical_groups,
    load_vector_cache,
    rare_postings,
    tokenize,
    write_vector_cache,
)
from .status import _head_for, load_manifest, status

TOOL_LAYER = {"AGENTS.md", "STATE.md", "INDEX.md", ".slopkeeperignore"}
# A draft's duplicate losers archive here (P4, 4) — a journaled move, never
# a delete; undo restores the file byte-exact.
DUPLICATE_ARCHIVE = "archive/duplicates"


def _file_entries(root: Path, rels: list[str], cap: int = 4000) -> list[dict]:
    """Inventory-style entries (path/title/head) for already-filed files."""
    out = []
    for rel in rels[:cap]:
        if Path(rel).name in TOOL_LAYER:
            continue
        h = _head_for(root, rel)
        out.append({"path": rel, "title": h.get("title"), "head": h.get("head")})
    return out


def draft(
    root: Path,
    min_sim: float | None = None,
    depth: int | None = None,
    force: bool = False,
) -> dict:
    """Draft a partial delta plan for new slop, recursively by default.

    depth bounds the sweep: only new files within `depth` directory
    levels of the root are considered (1 = root + immediate
    subdirectories); None walks the whole corpus.

    force (P4, 1) is the memory-bound override: the nearest-neighbour
    scan is candidate-blocked past the old 3000 cap, so a re-tend on a
    large corpus no longer refuses outright — it refuses only past
    MEMORY_MAX_FILES unless --force.
    """
    root = root.resolve()
    if min_sim is None:
        min_sim = DEFAULT_T1
    manifest = load_manifest(root)
    if manifest is None:
        raise SlopKeeperError(
            "no_manifest",
            f"{root} has no .slopkeeper/manifest.json — nothing to draft "
            "against. Run a first tend first.",
            "Run `slopkeeper status <root>` to confirm, then do a full "
            "first tend (AGENT_GUIDE.md).",
        )

    st = status(root)
    # T7 honest accounting: status caps `new_files` at 2000 entries but
    # `new_count` is the TRUE total, so the draft can report exactly
    # what it never saw instead of letting the overflow vanish.
    new_files = st.get("new_files", [])
    total_new = st.get("new_count", len(new_files))

    def _undrafted(drafted: int, unresolved: int, deduped: int = 0) -> dict | None:
        """Count new files that got NO draft outcome, with the reason.

        Every considered file lands in moves, unresolved or dedup (a
        byte-identical duplicate archived instead of filed) — so a gap here
        means files the draft never had a chance at: beyond --depth, beyond
        status's new_files cap, or tool-layer.
        """
        lost = total_new - drafted - unresolved - deduped
        if lost <= 0:
            return None
        if depth is not None:
            reason = (
                f"beyond --depth {depth} — only new files within {depth} "
                "directory level(s) of the root were considered by this "
                "draft"
            )
        elif len(new_files) < total_new:
            reason = (
                f"beyond status's new_files cap (only {len(new_files)} of "
                f"{total_new} new files were listed) — they were never "
                "considered by this draft"
            )
        else:
            reason = (
                "not draftable slop — AGENTS.md/STATE.md/INDEX.md "
                "tool-layer files are skipped by design"
            )
        return {"count": lost, "reason": reason}

    # The sweep is recursive: every new file is draftable slop, at any
    # depth — a file dropped into docs/ can belong in research/ just as
    # much as a root file does. Only the tool layer is skipped (by
    # basename, at any depth), and --depth bounds how far the sweep
    # reaches (same convention as inventory: N levels below the root).
    new_slop = [
        f
        for f in new_files
        if (depth is None or len(Path(f["path"]).parts) <= depth + 1)
        and Path(f["path"]).name not in TOOL_LAYER
    ]

    # Duplicate slop (P4, 4): two brand-new files that are byte-identical
    # are not two things to file — keep one, archive the rest into the
    # duplicates folder as a journaled `dedup` move (never a delete; undo
    # restores). They are removed from the drafted set so the same file is
    # not both moved to a folder AND archived.
    dedups: list[dict] = []
    dedup_drops: set[str] = set()
    if len(new_slop) >= 2:
        for group in identical_groups(root, [f["path"] for f in new_slop]):
            keeper = group[0]
            for drop in group[1:]:
                dedup_drops.add(drop)
                dedups.append(
                    {
                        "op": "dedup",
                        "keep": keeper,
                        "drop": drop,
                        "to": f"{DUPLICATE_ARCHIVE}/{drop}",
                        "reason": f"draft: byte-identical duplicate of {keeper}",
                    }
                )
        if dedup_drops:
            new_slop = [f for f in new_slop if f["path"] not in dedup_drops]

    if not new_slop and not dedups:
        return {
            "plan": None,
            "drafted_count": 0,
            "unresolved_count": 0,
            "new_count": total_new,
            "undrafted": _undrafted(0, 0),
            "note": "Nothing new to draft since the last tend.",
        }

    filed = [k for k in manifest.get("files", {}) if "/" in k]
    if len(filed) > MEMORY_MAX_FILES and not force:
        raise SlopKeeperError(
            "corpus_too_big",
            f"{len(filed)} filed files exceeds the draft memory bound of "
            f"{MEMORY_MAX_FILES}",
            "Partition per-directory instead, or pass --force to run the "
            "blocked nearest-neighbour scan anyway (candidate pairs only, "
            "not the full O(n²) sweep).",
        )

    # Vector cache: reuse cached per-doc tf whose sha256 matches
    # the manifest@2 hash for that path. Keying on the MANIFEST (the
    # last-tended state), not the working disk, is the spec'd behavior —
    # a re-tend (apply/undo/--record) refreshes both. Slop is always
    # vectorized fresh: it is the O(new) cost the loop is allowed.
    # Params are content-only (see collect_similarity): min_sim tunes
    # matching, never tf, so it must not key — or thrash — the cache.
    params: dict = {}
    cache_docs = load_vector_cache(root, params)
    manifest_files = manifest.get("files", {})

    def _manifest_sha(rel: str) -> str | None:
        entry = manifest_files.get(rel)
        return entry[2] if isinstance(entry, list) and len(entry) > 2 else None

    corpus_vecs, fresh = build_vectors_cached(
        _file_entries(root, filed), cache_docs, _manifest_sha
    )
    slop_vecs = build_vectors(
        [
            {"path": f["path"], "title": f.get("title"), "head": f.get("head")}
            for f in new_slop
        ]
    )
    # The nearest-neighbour scan is the other O(n²): each new file used
    # to price a cosine against every filed file. Past MAX_FILES it
    # instead walks only the filed files sharing rare tokens with the
    # query (a true match always shares rare content), turning the re-tend
    # loop's cost into O(new × candidates) (P4, 1).
    block = len(corpus_vecs) > MAX_FILES
    corpus_norms = {rel: _norm(cv) for rel, cv in corpus_vecs.items()} if block else {}
    postings = rare_postings(corpus_vecs) if block else {}

    moves: list[dict] = []
    unresolved: list[dict] = []

    for path, sv in slop_vecs.items():
        best_rel, best_sim = None, 0.0
        if block:
            candidates = candidate_neighbors(sv, postings)
        else:
            candidates = list(corpus_vecs.keys())
        for rel in candidates:
            cv = corpus_vecs.get(rel)
            if cv is None:
                continue
            if block:
                s = _cosine(sv, cv, na=None, nb=corpus_norms.get(rel))
            else:
                s = _cosine(sv, cv)
            if s > best_sim:
                best_rel, best_sim = rel, s
        if best_rel is None or best_sim < min_sim:
            # Say WHY it's unresolved: no neighbor, or a weak one.
            weak = (
                (f"best neighbor {best_rel} only {best_sim:.2f} (min {min_sim})")
                if best_rel
                else "no neighbor found"
            )
            unresolved.append({"path": path, "reason": f"draft: {weak}"})
            continue
        folder = str(Path(best_rel).parent)
        dst = f"{folder}/{Path(path).name}"
        if dst == path:
            # Already in the folder of its nearest filed neighbor: not
            # slop. Record it honestly rather than emit a no-op move
            # (the validator refuses from == to).
            unresolved.append(
                {
                    "path": path,
                    "reason": (
                        f"draft: nearest filed neighbor {best_rel} "
                        f"({best_sim:.2f}) is in its own folder — "
                        "left in place"
                    ),
                }
            )
            continue
        # Tokens the two names share make the reason self-explanatory.
        shared = set(tokenize(Path(path).stem)) & set(tokenize(Path(best_rel).stem))
        link = f" via {Path(best_rel).name} ({shared or 'content'})"
        moves.append(
            {
                "from": path,
                "to": dst,
                "reason": f"draft: nearest filed neighbor {best_sim:.2f}{link}",
            }
        )

    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "partial": True,
        "draft": {
            "min_sim": min_sim,
            "depth": depth,
            "considered": len(new_slop) + len(dedup_drops),
            "proposed": len(moves),
            "unresolved": len(unresolved),
            "dedup": len(dedups),
        },
        "based_on": {
            "corpus_fingerprint": corpus_fingerprint(root),
            "inventory_at": st.get("last_tended"),
        },
        "moves": sorted(moves, key=lambda m: m["from"]),
        # Duplicate slop archived instead of filed (P4, 4): a journaled
        # move into the duplicates archive, validated for byte-identity and
        # rendered distinctly at the gate.
        "dedup": dedups,
        "edits": [],
        "keep": [],
        "unresolved": unresolved,
        "folders": [],
    }
    write_vector_cache(root, params, cache_docs, fresh)
    return {
        "plan": plan,
        "drafted_count": len(moves),
        "unresolved_count": len(unresolved),
        "dedup_count": len(dedups),
        "new_count": total_new,
        "undrafted": _undrafted(len(moves), len(unresolved), len(dedups)),
    }
