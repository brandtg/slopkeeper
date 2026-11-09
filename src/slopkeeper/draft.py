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
    _cosine,
    build_vectors,
    build_vectors_cached,
    load_vector_cache,
    tokenize,
    write_vector_cache,
)
from .status import _head_for, load_manifest, status

TOOL_LAYER = {"AGENTS.md", "STATE.md", "INDEX.md", ".slopkeeperignore"}


def _file_entries(root: Path, rels: list[str], cap: int = 4000) -> list[dict]:
    """Inventory-style entries (path/title/head) for already-filed files."""
    out = []
    for rel in rels[:cap]:
        if Path(rel).name in TOOL_LAYER:
            continue
        h = _head_for(root, rel)
        out.append({"path": rel, "title": h.get("title"), "head": h.get("head")})
    return out


def draft(root: Path, min_sim: float | None = None, depth: int | None = None) -> dict:
    """Draft a partial delta plan for new slop, recursively by default.

    depth bounds the sweep: only new files within `depth` directory
    levels of the root are considered (1 = root + immediate
    subdirectories); None walks the whole corpus.
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

    def _undrafted(drafted: int, unresolved: int) -> dict | None:
        """Count new files that got NO draft outcome, with the reason.

        Every considered file lands in moves or unresolved (below-
        threshold candidates are IN unresolved, each with its score) —
        so a gap here means files the draft never had a chance at:
        beyond --depth, beyond status's new_files cap, or tool-layer.
        """
        lost = total_new - drafted - unresolved
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
    if not new_slop:
        return {
            "plan": None,
            "drafted_count": 0,
            "unresolved_count": 0,
            "new_count": total_new,
            "undrafted": _undrafted(0, 0),
            "note": "Nothing new to draft since the last tend.",
        }

    filed = [k for k in manifest.get("files", {}) if "/" in k]
    if len(filed) > MAX_FILES:
        raise SlopKeeperError(
            "corpus_too_big",
            f"{len(filed)} filed files exceeds the draft cap of {MAX_FILES}",
            "Partition per-directory instead.",
        )

    # Vector cache (§5.3): reuse cached per-doc tf whose sha256 matches
    # the manifest@2 hash for that path. Keying on the MANIFEST (the
    # last-tended state), not the working disk, is the spec'd behavior —
    # a re-tend (apply/undo/--record) refreshes both. Slop is always
    # vectorized fresh: it is the O(new) cost the loop is allowed.
    params = {"t1": min_sim}
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

    moves: list[dict] = []
    unresolved: list[dict] = []

    for path, sv in slop_vecs.items():
        best_rel, best_sim = None, 0.0
        for rel, cv in corpus_vecs.items():
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
            "considered": len(new_slop),
            "proposed": len(moves),
            "unresolved": len(unresolved),
        },
        "based_on": {
            "corpus_fingerprint": corpus_fingerprint(root),
            "inventory_at": st.get("last_tended"),
        },
        "moves": sorted(moves, key=lambda m: m["from"]),
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
        "new_count": total_new,
        "undrafted": _undrafted(len(moves), len(unresolved)),
    }
