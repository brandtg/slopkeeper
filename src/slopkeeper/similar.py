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

from . import kinds
from .models import SlopKeeperError, sha256_file, state_dir

# Minimum similarity for a file to join a cluster (max-link over members)...
DEFAULT_T1 = 0.25
# ...and the average member similarity must stay within this ratio of T1,
# so a chain of loosely-related files cannot grow into one blob.
DEFAULT_RATIO = 0.5
MIN_NAME_TOKEN = 3
# Candidate blocking (P4, 0.2.9) turned the O(n²) matcher into a scan of
# pairs that share rare tokens, so the old second cliff is now a memory
# bound. MAX_FILES is the point past which the FULL all-pairs matcher is
# abandoned for the blocked one; MEMORY_MAX_FILES is the true ceiling,
# past which the command refuses unless --force (the corpus will not fit
# the inverted index + vectors in RAM at a sane size). The refusal text
# and the per-directory escape hatch are unchanged from before.
MAX_FILES = 3000
MEMORY_MAX_FILES = 50000
# A token is "rare" (a blocking key) when it lives in at most this
# fraction of documents; the floor keeps a handful of structural tokens
# (a name everyone shares) out of the candidate set where they would
# regenerate the O(n²) blowup the blocking exists to avoid.
BLOCK_DF_FRACTION = 0.2
BLOCK_DF_FLOOR = 32
# A pair must share at least this many rare tokens to be a candidate.
BLOCK_MIN_RARE_SHARED = 2


def _is_rare(df_count: int, n: int) -> bool:
    """A token is a blocking key when it appears in few enough documents."""
    return df_count <= max(BLOCK_DF_FLOOR, int(n * BLOCK_DF_FRACTION))


def candidate_pairs(vecs: dict[str, Counter], min_shared: int = BLOCK_MIN_RARE_SHARED):
    """Candidate (a, b) pairs that share ≥ `min_shared` rare tokens.

    The inverted token→docs index (a stdlib dict) replaces the all-pairs
    scan: a pair can only be a real match if it shares high-idf (rare)
    content, so the cosine is computed on the pairs that share ≥K rare
    tokens and every other pair is never priced. This is what lifts
    MAX_FILES from a correctness cliff to a memory bound (P4): the
    candidate set is proportional to Σ df(t)² over RARE t only, not n².
    Returns a set of ordered (a, b) tuples (a < b).
    """
    n = len(vecs)
    postings: dict[str, list[str]] = {}
    for rel in sorted(vecs):
        for t in vecs[rel]:
            postings.setdefault(t, []).append(rel)
    rare: dict[str, list[str]] = {}
    for t, docs in postings.items():
        if len(docs) >= 2 and _is_rare(len(docs), n):
            rare[t] = docs
    # shared-count over rare-token postings; only pairs seen ≥K times stay.
    counts: dict[tuple[str, str], int] = {}
    for docs in rare.values():
        for i in range(len(docs)):
            a = docs[i]
            for j in range(i + 1, len(docs)):
                b = docs[j]
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
    return {pair for pair, c in counts.items() if c >= min_shared}


def _neighbor_adjacency(pairs) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def rare_postings(vecs: dict[str, Counter]) -> dict[str, list[str]]:
    """Inverted rare-token → docs index over `vecs` (a blocking table).

    The per-query companion to `candidate_pairs`: build once over the
    filed corpus, then ask each new vector who its rare-token neighbours
    are. Common tokens (the name everyone shares) are left out of the
    table, so a query never walks a posting of length n.
    """
    n = len(vecs)
    postings: dict[str, list[str]] = {}
    for rel in sorted(vecs):
        for t in vecs[rel]:
            postings.setdefault(t, []).append(rel)
    return {t: docs for t, docs in postings.items() if _is_rare(len(docs), n)}


def candidate_neighbors(
    query: Counter,
    postings: dict[str, list[str]],
    min_shared: int = BLOCK_MIN_RARE_SHARED,
) -> list[str]:
    """Docs in `postings` sharing ≥`min_shared` rare tokens with `query`.

    The draft nearest-neighbour scan priced this way: an O(candidates)
    cosine instead of an O(filed) one, on the same argument that a pair
    reaching t1 must share rare content (P4, 1).
    """
    counts: dict[str, int] = {}
    for t in query:
        docs = postings.get(t)
        if not docs:
            continue
        for d in docs:
            counts[d] = counts.get(d, 0) + 1
    return [d for d, c in counts.items() if c >= min_shared]


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
    so the vector cache stores raw per-doc tf only and this step
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


# --- vector cache ------------------------------------------
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
    """build_vectors with per-doc tf reuse keyed by content hash.

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


def _norm(c: Counter) -> float:
    """‖c‖₂, for callers that compare one vector against many."""
    return math.sqrt(sum(w * w for w in c.values()))


def _cosine(a: Counter, b: Counter, na: float | None = None, nb: float | None = None):
    """cos(a, b) over the sparse TF-IDF vectors.

    `na`/`nb` are the callers' precomputed norms: the matcher compares
    one candidate against every cluster member, and re-deriving ‖b‖ for
    a vector that has not changed since the last comparison was the
    inner loop's warmest spot (P3). Left None, they are computed here —
    same answer either way.
    """
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
        na, nb = nb, na
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    if na is None:
        na = math.sqrt(sum(w * w for w in a.values()))
    if nb is None:
        nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity_clusters(
    vecs: dict[str, Counter],
    t1: float = DEFAULT_T1,
    ratio: float = DEFAULT_RATIO,
    adjacency: dict[str, set[str]] | None = None,
) -> list[list[str]]:
    """Greedy single-link with a mean-similarity floor (anti-chaining).

    `adjacency` (P4 candidate blocking) is the rel→candidate-neighbours
    map from `candidate_pairs`. When given, a vector only ever compares
    against clusters holding one of its candidates — a pair that shares
    no rare tokens can never clear t1, so pruning them is a speed win,
    not a recall change (the same clusters fall out, at a fraction of the
    cosine calls). None (the default, and every corpus under MAX_FILES)
    compares every cluster exactly as before.
    """
    clusters: list[list[str]] = []
    # Every norm, once: the greedy pass re-uses each vector as both side
    # of a comparison many times over, and a norm is a full pass over the
    # vector's terms.
    norms = {rel: _norm(v) for rel, v in vecs.items()}
    # cluster index -> members, plus a member -> which clusters to probe,
    # so a blocked run reads only the clusters its candidates live in.
    member_clusters: dict[str, set[int]] = {}
    for rel in sorted(vecs):
        v = vecs[rel]
        nv = norms[rel]
        if adjacency is None:
            probe = range(len(clusters))
        else:
            wanted: set[int] = set()
            for nb in adjacency.get(rel, ()):
                wanted |= member_clusters.get(nb, set())
            probe = sorted(wanted)
        best_i, best_s = -1, 0.0
        for i in probe:
            members = clusters[i]
            sims = sorted(
                (_cosine(v, vecs[m], nv, norms[m]) for m in members), reverse=True
            )
            if sims[0] >= t1 and sims[0] > best_s:
                if sum(sims) / len(sims) >= t1 * ratio:
                    best_s, best_i = sims[0], i
        if best_i >= 0:
            clusters[best_i].append(rel)
            member_clusters.setdefault(rel, set()).add(best_i)
        else:
            clusters.append([rel])
            member_clusters[rel] = {len(clusters) - 1}
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
    """Byte-identical non-binary files (sha256 of first 1 MiB).

    "Non-binary" is the ONE table's answer (kinds.is_binary, P3/R7).
    This used to carry its own inline list of nine extensions, which
    meant the duplicate detector hashed `.webp`, `.tar`, `.mp3` and
    `.db` copies that every other surface already knew were binary —
    noise rows, and a second definition of a word the tool has one of.
    """
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        fp = root / p
        if kinds.is_binary(fp):
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


# Version-y name signals. mtime alone is a coin flip after rsync, git
# checkout, or any sync tool that resets timestamps wholesale — an
# intentional version pair says so in the name (report_final.docx vs
# report_final.pdf), and that is the corroboration we require.
_VERSIONISH_RE = re.compile(
    r"(?:~|(?<![a-z])final(?![a-z])|(?<![a-z])copy(?![a-z])"
    r"|(?<![a-z])backup(?![a-z])|(?<![a-z])bak(?![a-z])|(?<![a-z])old(?![a-z])"
    r"|(?<![a-z])v\d+(?![a-z0-9]))",
    re.IGNORECASE,
)


def stale_version_pairs(root: Path, paths: list[str]) -> list[dict]:
    """Same stem, different content, older mtime, version-y name → pair.

    The field report's md5sum/mtime archaeology, as data: for each
    same-stem group whose hashes are NOT all equal, every file older
    than the newest (and differing from it) is flagged — but only when
    either basename carries a version signal (v\\d, final, copy,
    backup/bak, old, trailing ~). Without it, a sync tool's wholesale
    mtime reset manufactures "older versions" out of unrelated same-stem
    files (the touch-reset false positive); mtime ranks, the name
    agrees. Identical copies are identical_groups' business, not
    versions.
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
        if not (
            _VERSIONISH_RE.search(Path(newest_p).name)
            or any(_VERSIONISH_RE.search(Path(p).name) for p, h, _m in entries[1:])
        ):
            continue  # nothing in the group claims to be a version
        for p, h, _mt in entries[1:]:
            if h != newest_h:
                out.append(
                    {
                        "file": p,
                        "likely_newer": newest_p,
                        "reason": "same stem, different content, older mtime, "
                        "version-y name",
                    }
                )
    return sorted(out, key=lambda e: e["file"])


def _subtree_signature_files(root: Path, paths: list[str]) -> dict[str, list[str]]:
    """Every directory → the list of corpus-relative files under it.

    The recursive companion to the flat `by_parent` map (P4): a copied
    tree one level deep is invisible to a per-directory comparison but
    obvious once each directory owns its whole subtree's file list. Built
    by walking each path's ancestor chain once, so it costs O(files ×
    depth), not O(dirs × files).
    """
    under: dict[str, list[str]] = {}
    for p in paths:
        parts = Path(p).parts
        for i in range(1, len(parts)):  # "." (i=0) is never a candidate dir
            anc = "/".join(parts[:i])
            under.setdefault(anc, []).append(p)
    return under


def identical_subtrees(root: Path, paths: list[str], min_files: int = 2) -> list[dict]:
    """Directories whose WHOLE subtree (relative path + content) matches.

    The vendored-copy case: four copies of one design system dominate the
    clusters. Signature = the sorted `relative-path:content-hash` list of
    every file in the subtree (P4, 5) — so a copy is caught no matter how
    deep its files sit, and a rearranged subtree still matches (the
    relative path, not the on-disk order, is what is hashed). Dirs whose
    files are unreadable are skipped, never guessed.

    A directory qualifies at `min_files` files. To keep the report about
    COPIES rather than nesting, a directory whose subtree is identical to
    an already-reported directory that contains it is dropped: the four
    `vendor/<name>/` copies are reported once, not `vendor` plus each
    child. The shallowest identical directory of a group is kept.
    """
    under = _subtree_signature_files(root, paths)
    # Memoized content hashes: a shared file hashes once across all the
    # ancestor signatures that include it.
    hashes: dict[str, str | None] = {}

    def _hash(rel: str) -> str | None:
        if rel not in hashes:
            hashes[rel] = _content_hash(root, rel)
        return hashes[rel]

    sigs: dict[str, list[tuple[str, int]]] = {}
    for d, members in sorted(under.items()):
        if len(members) < min_files:
            continue
        parts = []
        ok = True
        prefix = d + "/"
        for p in sorted(members):
            h = _hash(p)
            if h is None:
                ok = False
                break
            parts.append(
                f"{p[len(prefix) :] if p.startswith(prefix) else Path(p).name}:{h}"
            )
        if not ok:
            continue
        sigs.setdefault("|".join(parts), []).append((d, len(members)))
    out: list[dict] = []
    for dirs in sigs.values():
        if len(dirs) < 2:
            continue
        # Keep only the shallowest directory of each identical group as the
        # representative; a deeper member is inside an already-reported
        # subtree (vendor/one under vendor) and adds no information.
        dirs.sort(key=lambda x: (x[0].count("/"), x[0]))
        kept: list[str] = []
        for d, _n in dirs:
            if any(d == k or d.startswith(k + "/") for k in kept):
                continue
            kept.append(d)
        if len(kept) < 2:
            continue
        kept.sort()
        out.append({"dir": kept[0], "copies": kept[1:], "files": len(under[kept[0]])})
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


def collect_similarity(
    root: Path,
    files: list[dict],
    with_dupes: bool = True,
    sha_of=None,
    force: bool = False,
    min_shared: int = BLOCK_MIN_RARE_SHARED,
) -> dict:
    """Full hint document for the clusters command.

    Vectors reuse .slopkeeper/vectors.json keyed by content hash
    and write the refreshed cache back; output is identical with or
    without a warm cache.

    `sha_of(rel)` replaces the per-file `sha256_file` the cache keys on.
    The clusters command passes the single-pass scan's hashes
    (`scan.CorpusScan.sha_of`): the inventory walk has already read and
    digested every one of those bytes, and re-hashing the corpus to look
    up a cache the same walk could have hit was the third full read of
    the tree in one command. None (the default) hashes on demand, for
    callers that arrive with a bare file list.

    Scale (P4, 1): under MAX_FILES every vector is compared to every
    cluster (the original matcher, exact). Past MAX_FILES the matcher
    switches to candidate blocking — only pairs sharing ≥`min_shared`
    rare tokens are ever priced — so a corpus of tens of thousands runs
    in memory rather than refusing. The hard ceiling is now MEMORY_MAX_FILES
    (the index + vectors will not fit below sane limits); past THAT the
    command refuses `similarity_cap` unless --force, and the recovery
    still names per-directory clustering as the cheap escape.
    """
    if len(files) > MEMORY_MAX_FILES and not force:
        raise SlopKeeperError(
            "similarity_cap",
            f"{len(files)} files exceeds the similarity memory bound of "
            f"{MEMORY_MAX_FILES}; the blocked matcher will not fit in RAM.",
            "Cluster a subdirectory instead (`slopkeeper clusters <root>/<dir>`), "
            "or organize smaller scopes; --force runs the blocked matcher "
            "anyway (candidate pairs only — it is O(Σ rare-token²), not "
            "O(n²), so past the old 3000 cap it is usually fine).",
        )
    paths = [f["path"] for f in files]
    # One subtree index for every "which files live under this dir"
    # question below (identical subtrees now own their whole subtree).
    under = _subtree_signature_files(root, paths)
    # Identical subtrees (Phase 11, recursive in P4) collapse to ONE
    # cluster entry: their files leave the vector input entirely and
    # reappear as a single labeled row naming every copy — four copies of
    # a vendored design system stop being four clusters of noise.
    subtrees = identical_subtrees(root, paths) if with_dupes else []
    collapsed = {
        p for g in subtrees for d in [g["dir"], *g["copies"]] for p in under.get(d, ())
    }
    cluster_files = [f for f in files if f["path"] not in collapsed]
    # The cached payload is per-doc tf — pure function of file CONTENT.
    # t1/ratio affect matching only, never vectorization, so they are
    # NOT part of the cache key (they were: `clusters` and
    # `draft --min-sim X` alternated and full-missed each other every
    # run). load_vector_cache still round-trips `params`, so any future
    # parameter that DOES change tf invalidates mechanically.
    params: dict = {}
    cache_docs = load_vector_cache(root, params)
    if sha_of is None:

        def sha_of(rel):  # noqa: E306 - the on-demand fallback
            return sha256_file(root / rel)

    vecs, fresh = build_vectors_cached(cluster_files, cache_docs, sha_of)
    write_vector_cache(root, params, cache_docs, fresh)
    # Blocking engages only past MAX_FILES: under it the all-pairs matcher
    # is both exact and cheap, so behavior there is bit-identical to 0.2.8.
    adjacency = None
    if len(vecs) > MAX_FILES:
        adjacency = _neighbor_adjacency(candidate_pairs(vecs, min_shared=min_shared))
    clusters = similarity_clusters(vecs, adjacency=adjacency)
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
                "files": sorted(under.get(g["dir"], ())),
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
