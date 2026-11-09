# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Similarity hints: pre-cluster the corpus so the agent names folders
instead of partitioning from scratch.

The agent's judgment is the hard part of corpus organization — but a large
share of it is mechanical pattern-finding that a deterministic pass can do
first: files that talk about the same thing (TF-IDF cosine over filename +
title + head tokens), scrape pairs (same stem, different ext), byte-identical
copies, and hub files that everything cites.

Policy: precision over recall. A false merge sends two unrelated files to
the same folder and the agent may not notice; a false split just leaves
two entries for the agent to reconcile. Clusters are HINTS, never a plan.

Zero dependencies: stdlib only, no LLM, no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .models import SlopKeeperError, sha256_file, state_dir

# Minimum similarity for a file to join a cluster (max-link over members)...
DEFAULT_T1 = 0.25
# ...and the average member similarity must stay within this ratio of T1,
# so a chain of loosely-related files cannot grow into one blob.
DEFAULT_RATIO = 0.5
MIN_NAME_TOKEN = 3
MAX_FILES = 3000  # O(n^2); beyond this, refuse loudly rather than crawl

STOPWORDS = frozenset(
    """
the a an and or of to in on for with without from by at as is are was were
be been being it its this that these those i you he she we they them his
her their our your my me not no yes but if then than so such very can could
should would will won do does did done doing have has had having about into
over under again further once here there when where why how all any both
each few more most other some only own same too just don now inc www com
http https html htm org pdf txt json md png jpg csv py sh log yaml yml
""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'+\-/.]{2,}")


def tokenize(text: str) -> list[str]:
    out = []
    for w in _TOKEN_RE.findall(text.lower()):
        w = w.strip("_'+-./")
        if len(w) < MIN_NAME_TOKEN or w in STOPWORDS or w.isdigit():
            continue
        out.append(w)
    return out


def _doc_tokens(path: str, title: str | None, head: str | None) -> Counter:
    """Filename tokens weigh most, then the H1 title, then the body head."""
    stem = Path(path).stem.lower().replace("-", " ").replace("_", " ").replace(".", " ")
    c: Counter = Counter()
    for t in tokenize(stem):
        c[t] += 8
    for t in tokenize(title or ""):
        c[t] += 3
    for t in tokenize(head or ""):
        c[t] += 1
    return c


def vectors_from_tf(docs: dict[str, Counter]) -> dict[str, Counter]:
    """Corpus-global TF-IDF step: document frequency over the WHOLE doc
    set, then sublinear-tf * idf weighting per doc.

    Vectors are not per-doc functions — idf depends on every other doc —
    so the vector cache (§5.3) stores raw per-doc tf only and this step
    is recomputed over the full corpus at match time. Given the same
    docs, the result is exactly what build_vectors produces.
    """
    df: Counter = Counter()
    for c in docs.values():
        for t in c:
            df[t] += 1
    n = len(docs)
    idf = {t: math.log(n / cnt) + 0.1 for t, cnt in df.items()}
    return {
        rel: Counter({t: (1 + math.log(w)) * idf.get(t, 2.0) for t, w in c.items()})
        for rel, c in docs.items()
    }


def build_vectors(files: list[dict]) -> dict[str, Counter]:
    """files: inventory-style dicts with path/title/head."""
    docs = {
        f["path"]: _doc_tokens(f["path"], f.get("title"), f.get("head")) for f in files
    }
    return vectors_from_tf(docs)


# --- vector cache (DESIGN.md §5.3) ------------------------------------------
#
# .slopkeeper/vectors.json maps relpath -> {sha256, tf} so a re-tend
# vectorizes only new/changed files. The cache is DERIVED state: a
# missing, corrupt, wrong-schema, or params-mismatched file is a full
# miss (the status-quo re-vectorization), never an error, and never a
# mixed-parameter match.

VECTOR_CACHE_SCHEMA = "slopkeeper/vectors@1"


def vector_cache_path(root: Path) -> Path:
    return state_dir(root) / "vectors.json"


def load_vector_cache(root: Path, params: dict) -> dict | None:
    """Cached {relpath: {sha256, tf}} entries for exactly `params`.

    Anything else — unparseable JSON, wrong schema string, a params
    mismatch — returns None: full re-vectorization, silently (the file
    is derived, so there is nothing to log).
    """
    try:
        raw = json.loads(vector_cache_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != VECTOR_CACHE_SCHEMA:
        return None
    if raw.get("params") != params or not isinstance(raw.get("docs"), dict):
        return None
    return raw["docs"]


def write_vector_cache(
    root: Path, params: dict, cache_docs: dict | None, fresh: dict
) -> None:
    """Merge freshly vectorized tf entries into the loaded cache and
    write it back. A no-op run (warm cache, nothing re-vectorized)
    skips the write; an unwritable cache is skipped silently — a
    derived file must never fail a command."""
    if cache_docs is not None and not fresh:
        return
    docs = dict(cache_docs) if cache_docs else {}
    docs.update(fresh)
    if not docs:
        return
    payload = {
        "schema": VECTOR_CACHE_SCHEMA,
        "built_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "docs": docs,
    }
    try:
        p = vector_cache_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError:
        pass


def build_vectors_cached(
    files: list[dict], cache_docs: dict | None, sha_of
) -> tuple[dict[str, Counter], dict]:
    """build_vectors with per-doc tf reuse keyed by content hash (§5.3).

    `sha_of(rel)` supplies the hash to key on — the manifest@2 hash on
    the draft path, a fresh sha256_file on the clusters path. A None
    hash never reuses: unhashable content cannot be validated. The
    cache holds RAW tf only; the corpus-global idf step is recomputed
    over the full doc set here, so the result is identical to
    build_vectors over the same files. Cached tf dicts whose garbage
    values crash the math fall back to a full re-vectorization —
    derived state is never an error. Returns (vecs, fresh entries).
    """
    docs: dict[str, Counter] = {}
    fresh: dict[str, dict] = {}
    shas: dict[str, str | None] = {}
    for f in files:
        rel = f["path"]
        sha = sha_of(rel)
        shas[rel] = sha
        cached = cache_docs.get(rel) if cache_docs else None
        if (
            sha is not None
            and isinstance(cached, dict)
            and cached.get("sha256") == sha
            and isinstance(cached.get("tf"), dict)
        ):
            docs[rel] = Counter(cached["tf"])
        else:
            docs[rel] = _doc_tokens(rel, f.get("title"), f.get("head"))
            fresh[rel] = {"sha256": sha, "tf": dict(docs[rel])}
    try:
        return vectors_from_tf(docs), fresh
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        for f in files:
            rel = f["path"]
            docs[rel] = _doc_tokens(rel, f.get("title"), f.get("head"))
            fresh[rel] = {"sha256": shas[rel], "tf": dict(docs[rel])}
        return vectors_from_tf(docs), fresh


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity_clusters(
    vecs: dict[str, Counter],
    t1: float = DEFAULT_T1,
    ratio: float = DEFAULT_RATIO,
) -> list[list[str]]:
    """Greedy single-link with a mean-similarity floor (anti-chaining)."""
    clusters: list[list[str]] = []
    for rel in sorted(vecs):
        v = vecs[rel]
        best_i, best_s = -1, 0.0
        for i, members in enumerate(clusters):
            sims = sorted((_cosine(v, vecs[m]) for m in members), reverse=True)
            if sims[0] >= t1 and sims[0] > best_s:
                if sum(sims) / len(sims) >= t1 * ratio:
                    best_s, best_i = sims[0], i
        if best_i >= 0:
            clusters[best_i].append(rel)
        else:
            clusters.append([rel])
    return [sorted(c) for c in clusters if len(c) >= 2]


def cluster_label(members: list[str], max_terms: int = 3) -> str:
    """Name-tokens shared by >= half the members, ordered by their position
    in the names; fallback to most common overall."""
    shared: Counter = Counter()
    for m in members:
        shared.update(
            set(tokenize(Path(m).stem.lower().replace("-", " ").replace("_", " ")))
        )
    if not shared:
        # Date-named diaries etc.: the common parent directory is the label.
        parents = {str(Path(m).parent) for m in members}
        if len(parents) == 1:
            return parents.pop()
        return "(unlabeled)"
    half = max(1, len(members) // 2)
    picked = {t for t, n in shared.most_common() if n >= half} or {
        t for t, _ in shared.most_common(max_terms)
    }

    def first_pos(m: str) -> tuple:
        stem = Path(m).stem.lower().replace("-", " ").replace("_", " ")
        return tuple(stem.find(t) if t in stem else 999 for t in sorted(picked))

    anchor = min(members, key=first_pos)
    stem = Path(anchor).stem.lower().replace("-", " ").replace("_", " ")
    ordered = [t for t in stem.split() if t in picked]
    ordered += [t for t in sorted(picked) if t not in ordered]
    return " ".join(ordered[:max_terms])


def same_stem_groups(paths: list[str]) -> list[list[str]]:
    """Files sharing a basename stem (scrape pairs, exports, images)."""
    groups: dict[str, list[str]] = {}
    for p in paths:
        groups.setdefault(Path(p).stem.lower(), []).append(p)
    return sorted(
        (sorted(v) for v in groups.values() if len(v) > 1),
        key=lambda g: g[0],
    )


def identical_groups(root: Path, paths: list[str]) -> list[list[str]]:
    """Byte-identical non-binary files (sha256 of first 1 MiB)."""
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        fp = root / p
        if fp.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".zip",
            ".gz",
            ".mp4",
            ".mov",
        ):
            continue
        try:
            with fp.open("rb") as f:
                h = hashlib.sha256(f.read(1 << 20)).hexdigest()
        except OSError:
            continue
        by_hash.setdefault(h, []).append(p)
    return sorted(
        (sorted(v) for v in by_hash.values() if len(v) > 1),
        key=lambda g: g[0],
    )


def _content_hash(root: Path, rel: str) -> str | None:
    """sha256 of the first 1 MiB (same budget as identical_groups)."""
    try:
        with (root / rel).open("rb") as f:
            return hashlib.sha256(f.read(1 << 20)).hexdigest()
    except OSError:
        return None


def stale_version_pairs(root: Path, paths: list[str]) -> list[dict]:
    """Same stem, different content, older mtime → likely older version.

    The field report's md5sum/mtime archaeology, as data: for each
    same-stem group whose hashes are NOT all equal, every file older
    than the newest (and differing from it) is flagged. Identical
    copies are identical_groups' business, not versions.
    """
    out: list[dict] = []
    for group in same_stem_groups(paths):
        entries = []
        for p in group:
            h = _content_hash(root, p)
            if h is None:
                continue
            try:
                mt = (root / p).stat().st_mtime
            except OSError:
                continue
            entries.append((p, h, mt))
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: (-e[2], e[0]))  # newest first
        newest_p, newest_h, _ = entries[0]
        for p, h, _mt in entries[1:]:
            if h != newest_h:
                out.append(
                    {
                        "file": p,
                        "likely_newer": newest_p,
                        "reason": "same stem, different content, older mtime",
                    }
                )
    return sorted(out, key=lambda e: e["file"])


def identical_subtrees(root: Path, paths: list[str], min_files: int = 2) -> list[dict]:
    """Directories whose file lists + content hashes are identical.

    The vendored-copy case: four copies of one design system dominate
    the clusters. Signature = sorted (basename, content-hash) pairs —
    flat, so a copy with rearranged subdirs still matches (v1;
    structure-aware signatures can come later). Dirs whose files are
    unreadable are skipped, never guessed.
    """
    by_dir: dict[str, list[str]] = {}
    for p in paths:
        by_dir.setdefault(str(Path(p).parent), []).append(p)
    sigs: dict[str, list[tuple[str, int]]] = {}
    for d, members in sorted(by_dir.items()):
        if len(members) < min_files:
            continue
        parts = []
        ok = True
        for p in sorted(members):
            h = _content_hash(root, p)
            if h is None:
                ok = False
                break
            parts.append(f"{Path(p).name}:{h}")
        if not ok:
            continue
        sigs.setdefault("|".join(parts), []).append((d, len(members)))
    out = []
    for dirs in sigs.values():
        if len(dirs) < 2:
            continue
        dirs.sort()
        out.append(
            {
                "dir": dirs[0][0],
                "copies": [d for d, _ in dirs[1:]],
                "files": dirs[0][1],
            }
        )
    return sorted(out, key=lambda g: g["dir"])


def hubs(refs: dict[str, list[str]], min_referrers: int = 3) -> list[dict]:
    """Rank inbound references: {basename, referrers, cited_by}.

    A hub moved alone breaks many stay-put files at once — move it with
    its citers, or rewrite the citers in the same plan.
    """
    ranked = sorted(
        ((name, [r for r in files if r != name]) for name, files in refs.items()),
        key=lambda kv: -len(kv[1]),
    )
    return [
        {"basename": name, "referrers": len(files), "cited_by": files[:20]}
        for name, files in ranked
        if len(files) >= min_referrers
    ]


def collect_similarity(root: Path, files: list[dict], with_dupes: bool = True) -> dict:
    """Full hint document for the clusters command.

    Vectors reuse .slopkeeper/vectors.json keyed by content hash (§5.3)
    and write the refreshed cache back; output is identical with or
    without a warm cache.
    """
    if len(files) > MAX_FILES:
        raise SlopKeeperError(
            "similarity_cap",
            f"{len(files)} files exceeds the similarity cap of {MAX_FILES}; "
            "partition per-directory instead.",
            "Cluster a subdirectory instead (`slopkeeper clusters <root>/<dir>`), "
            "or organize smaller scopes; the O(n^2) matcher needs this cap.",
        )
    paths = [f["path"] for f in files]
    # Identical subtrees (Phase 11) collapse to ONE cluster entry: their
    # files leave the vector input entirely and reappear as a single
    # labeled row naming every copy — four copies of a vendored design
    # system stop being four clusters of noise.
    subtrees = identical_subtrees(root, paths) if with_dupes else []
    collapsed = {
        p
        for g in subtrees
        for d in [g["dir"], *g["copies"]]
        for p in paths
        if str(Path(p).parent) == d
    }
    cluster_files = [f for f in files if f["path"] not in collapsed]
    params = {"t1": DEFAULT_T1}
    cache_docs = load_vector_cache(root, params)
    vecs, fresh = build_vectors_cached(
        cluster_files, cache_docs, lambda rel: sha256_file(root / rel)
    )
    write_vector_cache(root, params, cache_docs, fresh)
    clusters = similarity_clusters(vecs)
    labeled = [
        {
            "label": cluster_label(c) or "(unlabeled)",
            "size": len(c),
            "files": c,
        }
        for c in sorted(clusters, key=lambda c: (-len(c), c[0]))
    ]
    for g in subtrees:
        # appended after the similarity clusters: a different KIND of
        # row (exact copies, not similar text), named as such
        labeled.append(
            {
                "label": (
                    f"identical subtree: {g['dir']}/ "
                    f"(+{len(g['copies'])} identical copies)"
                ),
                "size": g["files"],
                "files": sorted(p for p in paths if str(Path(p).parent) == g["dir"]),
                "copies": g["copies"],
            }
        )
    covered = {p for c in clusters for p in c} | collapsed
    out: dict = {
        "clusters": labeled,
        "unclustered": sorted(p for p in paths if p not in covered),
    }
    if with_dupes:
        out["same_stem_groups"] = same_stem_groups(paths)
        out["identical_groups"] = identical_groups(root, paths)
        out["stale_versions"] = stale_version_pairs(root, paths)
        out["identical_subtrees"] = subtrees
    return out
