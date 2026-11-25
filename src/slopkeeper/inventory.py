# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Corpus inventory: the structured, token-conscious view an agent plans from.

Output is a single JSON document under the standard envelope. Designed so a
model can read it whole for a few hundred files, or work from `summary`
clusters for thousands.

Since 0.2.8 the document is folded from ONE pass (`scan.scan_corpus`):
the file list, the directory roll-ups, the symlinked-dir names, the
nested-corpus list and — the part that used to be a lie — the
fingerprint all describe the same walk. Before that, `collect` listed
one traversal and `corpus_fingerprint_detail` digested a SECOND one
moments later, so an agent could read a file list and a digest that
belonged to two different corpora (the audit's B5). Pass a prepared scan
in with `scan=` and the whole command costs one traversal, which is what
`doctor` and `clusters` now do.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from . import kinds
from .models import add_warning
from .references import scan_imports, scan_references

# HEAD_BYTES is re-exported: status.py imports the head budget from
# this module, and the number belongs to the pass that reads it.
from .scan import HEAD_BYTES, WalkReport, scan_corpus  # noqa: F401

# The extension tables live in kinds.py now (P3/R7: five consumers, five
# private copies of "what is binary" drifting apart). These names stay
# as the module-level aliases tests and status.py already import.
BINARY_EXTS = kinds.BINARY_EXTS
CODE_EXTS = kinds.CODE_EXTS

HEAD_CHARS = 400

DEFAULT_MAX_FILES = 8000

# Optional system extractors (Phase 11): probed from PATH per run, never
# required, never installed by us — the binary stays zero-dep. First
# available binary wins per extension; {file} is the only placeholder and
# output must land on stdout.
EXTRACTORS: dict[str, list[tuple[str, list[str]]]] = {
    ".pdf": [("pdftotext", ["pdftotext", "-l", "1", "{file}", "-"])],
    ".docx": [
        ("docx2txt", ["docx2txt", "{file}", "-"]),
        ("pandoc", ["pandoc", "-t", "plain", "{file}"]),
    ],
    ".pptx": [("unzip", ["unzip", "-p", "{file}", "ppt/slides/slide1.xml"])],
    # P5 filetype batch: the legacy binary MS formats are the ONLY new ones
    # that need an external binary at all (the ODF and EPUB containers are
    # zips, and stdlib `zipfile` reads those in-process below). Absent
    # tool → NO_SIGNAL, the same loud-not-silent rule as pdftotext.
    ".doc": [
        ("antiword", ["antiword", "-m", "UTF-8.txt", "{file}"]),
        ("catdoc", ["catdoc", "{file}"]),
    ],
    ".xls": [
        ("catdoc", ["catdoc", "{file}"]),
        ("xls2csv", ["xls2csv", "{file}"]),
    ],
}
NO_SIGNAL = "signal: none (binary, no extractor)"

_XML_TAG_RE = re.compile(r"<[^>]+>")
_XML_SPACE_RE = re.compile(r"\s+")

# --- containers stdlib reads itself (P5 filetype batch) --------------------
#
# The OOXML sibling trick: `.pptx` has always had its slide1 text pulled by
# shelling out to `unzip`. The OpenDocument formats are the same shape —
# a zip with one XML part holding the text — and `zipfile` is stdlib, so
# there is no binary to be absent and no PATH probe to miss. `.epub` is a
# zip too; its title lives in the OPF that `container.xml` points at.
_ZIP_TEXT_PARTS: dict[str, tuple[str, ...]] = {
    ".odt": ("content.xml",),
    ".ods": ("content.xml",),
    ".odp": ("content.xml",),
}
# ODF paragraphs and headings, in document order. Pulling the elements
# (rather than tag-stripping the whole part) keeps `<office:automatic-styles>`
# and spreadsheet formulas out of what becomes the file's TITLE.
_ODF_TEXT_RE = re.compile(r"<text:(h|p)\b[^>]*>(.*?)</text:\1>", re.S)
# How much of a container member or notebook is read: a title is a hint,
# and a 1 MiB budget is generous for one while bounding a pathological
# (or hostile) part to a bounded read.
_CONTAINER_CAP = 1024 * 1024


def _zip_text(fp: Path, members: tuple[str, ...]) -> str | None:
    """The text of the first present member of a zip container."""
    try:
        with zipfile.ZipFile(fp) as z:
            names = set(z.namelist())
            for member in members:
                if member not in names:
                    continue
                raw = z.read(member)[:_CONTAINER_CAP]
                text = _odf_paragraphs(raw.decode("utf-8", errors="replace"))
                if text:
                    return text
    except (OSError, zipfile.BadZipFile, KeyError, EOFError, ValueError):
        return None
    return None


def _odf_paragraphs(xml: str) -> str:
    """ODF text as lines: each `<text:h>`/`<text:p>` becomes one line."""
    lines: list[str] = []
    for _kind, inner in _ODF_TEXT_RE.findall(xml):
        flat = _XML_TAG_RE.sub(" ", inner)
        line = _XML_SPACE_RE.sub(" ", html.unescape(flat)).strip()
        if line:
            lines.append(line)
    if lines:
        return "\n".join(lines)
    # No paragraphs at all (a hand-written or unusual part): the pptx
    # fallback, tag-strip the whole thing, still beats no signal.
    stripped = _XML_SPACE_RE.sub(" ", html.unescape(_XML_TAG_RE.sub(" ", xml))).strip()
    return stripped or None


def _epub_title(fp: Path) -> str | None:
    """An EPUB's title: container.xml → the OPF it names → <dc:title>."""
    try:
        with zipfile.ZipFile(fp) as z:
            names = set(z.namelist())
            if "META-INF/container.xml" not in names:
                return None
            container = z.read("META-INF/container.xml").decode(
                "utf-8", errors="replace"
            )
            match = re.search(r'full-path="([^"]+\.opf)"', container, re.I)
            if not match:
                return None
            opf_name = html.unescape(match.group(1))
            if opf_name.lstrip("/") not in names:
                # The OPF path is container-relative; accept the plain and
                # leading-slash spellings, and fall back to the only .opf
                # in the archive when neither matches (some packagers write
                # a path the container does not).
                candidates = [n for n in names if n.lower().endswith(".opf")]
                if len(candidates) != 1:
                    return None
                opf_name = candidates[0]
            opf = z.read(opf_name)[:_CONTAINER_CAP].decode("utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile, KeyError, EOFError, ValueError):
        return None
    title = re.search(r"<dc:title\b[^>]*>(.*?)</dc:title>", opf, re.S | re.I)
    if not title:
        return None
    text = _XML_SPACE_RE.sub(" ", html.unescape(_XML_TAG_RE.sub(" ", title.group(1))))
    return text.strip() or None


def _notebook_text(fp: Path) -> str | None:
    """A Jupyter notebook as text: markdown cells, then code signatures.

    The audit's `.ipynb` row: the title used to be the first line of raw
    JSON, which is `{` for every notebook ever written. Cell-aware text
    fixes the title AND what the similarity layer tokenizes (inventory's
    `head` is what `similar._doc_tokens` reads), while the file stays ONE
    JSON document for hashing, dedup and `identical_groups` — nothing here
    rewrites or re-serializes it. None means "not a notebook we can read",
    and the caller falls back to the plain text path.
    """
    try:
        raw = fp.read_bytes()[:_CONTAINER_CAP]
    except OSError:
        return None
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    cells = doc.get("cells") if isinstance(doc, dict) else None
    if not isinstance(cells, list):
        return None
    lines: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source")
        if isinstance(source, list):
            source = "".join(str(s) for s in source)
        if not isinstance(source, str) or not source.strip():
            continue
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            lines.append(source.strip())
        else:
            # Code cells contribute their DEFINITION lines only: a whole
            # body of `import`s and boilerplate would otherwise dominate
            # the vector of every notebook in the corpus (the audit's
            # code-tokenization note, kept to the one shape that is
            # notebook-specific).
            for line in source.splitlines():
                s = line.strip()
                if s.startswith(("def ", "class ", "async def ", "#")):
                    lines.append(s)
    return "\n".join(lines) if lines else None


@lru_cache(maxsize=None)
def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _extract_head(fp: Path, ext: str) -> str | None:
    """First lines via a container reader or an optional system extractor;
    None = none/failure.

    Every failure mode (no extractor, non-zero exit, timeout, empty
    output) returns None so the caller emits the loud no-signal marker —
    a head is a hint, never a crash, and never silence.
    """
    # stdlib readers first: no binary to be absent, no PATH probe, no
    # subprocess (P5 filetype batch).
    if ext in _ZIP_TEXT_PARTS:
        text = _zip_text(fp, _ZIP_TEXT_PARTS[ext])
        if text:
            return text
    elif ext == ".epub":
        text = _epub_title(fp)
        if text:
            return text
    for binary, template in EXTRACTORS.get(ext, []):
        if _which(binary) is None:
            continue
        argv = [a.replace("{file}", str(fp)) for a in template]
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        text = proc.stdout.decode("utf-8", errors="replace")
        if ext == ".pptx":
            # slide1.xml: strip tags, unescape entities, collapse runs
            # of whitespace — the text is what survives.
            text = _XML_SPACE_RE.sub(" ", html.unescape(_XML_TAG_RE.sub(" ", text)))
        text = text.strip()
        if text:
            return text
    return None


def _kind(path: Path) -> str:
    """The kind of a file on disk (table, then the NUL sniff — kinds.py).

    Kept as this module's entry point because status.py imports it; the
    walk itself classifies without a second open via `FileEntry.kind`.
    """
    return kinds.classify(path)


def _notebook_head(fp: Path) -> dict | None:
    """{title, head} for a notebook, or None when it is not a notebook.

    The `.ipynb` row of the audit's filetype batch: the title used to be
    the first line of the JSON (`{`), and the head was 400 characters of
    cell metadata, so `clusters` tokenized boilerplate and every notebook
    in a corpus looked alike. The cell-aware text replaces BOTH — the
    title extraction and what the similarity layer reads — while the file
    itself stays one JSON document for hashing and dedup.
    """
    if fp.suffix.lower() != ".ipynb":
        return None
    text = _notebook_text(fp)
    if text is None:
        return None
    return {"title": _title_from_text(text, "doc"), "head": text[:HEAD_CHARS]}


def _head(entry) -> dict:
    """{title, head} for one inventoried file.

    The bytes are the walk's (`FileEntry.head`, the same open that
    produced the content hash); only a BINARY file still costs an
    external call, and that one is a subprocess, not a read.
    """
    kind = entry.kind
    if kind == "binary":
        # Loud by default: a binary head is either extracted text or the
        # explicit no-signal marker — never a silent None (the field
        # report's pdftotext/md5sum archaeology, now tool output).
        ext = entry.path.suffix.lower()
        extracted = _extract_head(entry.path, ext)
        if extracted is None:
            return {"title": None, "head": NO_SIGNAL}
        title = next(
            (s.strip()[:120] for s in extracted.splitlines() if s.strip()), None
        )
        return {"title": title, "head": extracted[:HEAD_CHARS]}
    if entry.head is None:
        # Unreadable (chmod 000, or a file that vanished mid-walk): the
        # graceful-degradation path the audit checked — title null, no
        # crash, the file still listed.
        return {"title": None, "head": None}
    notebook = _notebook_head(entry.path)
    if notebook is not None:
        return notebook
    text = entry.head.decode("utf-8", errors="replace")
    return {"title": _title_from_text(text, kind), "head": text[:HEAD_CHARS]}


def _title_from_text(text: str, kind: str) -> str | None:
    title = None
    if kind == "doc":
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                title = s.lstrip("#").strip()[:120]
                break
    if title is None:
        for line in text.splitlines():
            s = line.strip()
            if s:
                title = s[:120]
                break
    return title


def _prefix_clusters(names: list[str]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for n in names:
        token = n.split("_")[0].split("-")[0].split(" ")[0].lower()
        if len(token) >= 4:
            clusters.setdefault(token, []).append(n)
    return {
        k: v
        for k, v in sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        if len(v) >= 2
    }


def _scan_is_complete(scan) -> bool:
    """Does this scan carry everything the inventory document needs?

    `collect` folds hashes, kinds and head bytes out of the records it
    is handed, and a pass built without one of those facilities would
    silently produce a weaker document: a fingerprint digested from
    mtimes instead of content (a plan bound to the wrong thing), `kind:
    null` rows, null titles. Anything missing and collect walks for
    itself instead. An unreadable file is not a missing facility — it
    has the same shape in a complete scan.
    """
    from .models import SHA256_CAP_BYTES

    for e in scan.files:
        if e.kind is None:
            return False
        if e.kind != "binary" and e.head is None and not e.read_error:
            return False
        if not e.hashed and e.size <= SHA256_CAP_BYTES and not e.read_error:
            return False
    return True


def collect(
    root: Path,
    depth: int | None = None,
    fast: bool = False,
    max_files: int = DEFAULT_MAX_FILES,
    *,
    scan=None,
    scope: str | None = None,
) -> dict:
    """Walk root and build the inventory document (no envelope).

    `scan` is a prepared `scan.CorpusScan` (see scan.prepare): doctor
    and clusters hold one pass and hand it to every surface that wants
    the corpus, so the walk happens once per command. Omit it and
    collect runs its own. Either way the fingerprint in the document is
    the digest of THAT walk.

    `scope` (P4, 2, `inventory --scope inbox`) bounds the document to one
    corpus-relative directory subtree: only its files, dirs and the scope
    fingerprint are reported — the same view a scoped plan is authored
    against, so the plan and the inventory it came from agree on what is
    in scope.
    """
    root = root.resolve()
    if scope is not None:
        # Fold a scope-bound walk. A caller-supplied scan is ignored here
        # (it walked the whole corpus); scoping is its own single pass.
        scan = scan_corpus(
            root,
            want_sha=True,
            want_head=True,
            want_kind=True,
            report=WalkReport(),
            scope=scope,
        )
    if scan is None or not _scan_is_complete(scan):
        scan = scan_corpus(
            root,
            want_sha=True,
            want_head=True,
            want_kind=True,
            report=WalkReport(),
        )
    ignore_rules = scan.report.ignore_rules
    files: list[dict] = []
    dirs: dict[str, dict] = {}
    total_bytes = 0
    rel_paths: list[Path] = []
    truncated = False
    top_dir_counts: dict[str, int] = {}
    # fp6/A2: symlinked directories at any depth, named. The walk never
    # descends them (os.walk does not follow links) so they were
    # invisible — yet coverage validation counts them (iterdir) and the
    # fingerprint binds name+target. Inventory must not disagree with
    # what a plan is held to: they join `dirs` as zero-file ENTRIES and
    # are named in stats.
    symlinked_dirs: list[str] = []

    for visit in scan.visits:
        rel_dir = Path(visit.rel) if visit.rel != "." else Path()
        d_depth = len(rel_dir.parts)
        if depth is not None and d_depth > depth:
            # The depth cut stops DESCENT, so the old walk never produced
            # a visit below the cut at all: neither its files nor its own
            # child entries belong in this listing. (A visit AT the cut
            # still contributes its symlinked children — the old loop read
            # `dirnames` before clearing it, and coverage validation counts
            # those entries, so the quirk is the contract.)
            continue
        for rel_name, _target in visit.symlinked:
            symlinked_dirs.append(rel_name)
            dirs.setdefault(
                rel_name,
                {
                    "path": rel_name,
                    "file_count": 0,
                    "total_files": 0,
                    "last_modified": None,
                },
            )
        dir_key = visit.rel
        top_key = rel_dir.parts[0] if rel_dir.parts else "(root)"
        if truncated:
            # Past the cap the entries stop; the counting does not.
            # The truncation culprit is named from these totals, not
            # guessed after a silent walk-stop.
            for _e in visit.files:
                top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
            continue
        dirs.setdefault(
            dir_key,
            {"path": dir_key, "file_count": 0, "total_files": 0, "last_modified": None},
        )
        for entry in visit.files:
            if len(rel_paths) >= max_files:
                truncated = True
                top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
                continue
            rel = Path(entry.rel)
            total_bytes += entry.size
            top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
            kind = entry.kind
            e = {
                "path": entry.rel,
                "bytes": entry.size,
                "mtime": datetime.fromtimestamp(entry.mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                ),
                "ext": entry.path.suffix.lower(),
                "kind": kind,
            }
            # Binaries get a head too (Phase 11): extracted text via an
            # optional system extractor, or the loud no-signal marker.
            e.update(_head(entry))
            files.append(e)
            rel_paths.append(rel)
            # roll up dir stats
            parts = rel.parts[:-1]
            for i in range(len(parts) + 1):
                anc = "/".join(parts[:i]) or "."
                de = dirs.setdefault(
                    anc,
                    {
                        "path": anc,
                        "file_count": 0,
                        "total_files": 0,
                        "last_modified": None,
                    },
                )
                de["total_files"] += 1
                if i == len(parts):
                    de["file_count"] += 1
                m = de["last_modified"]
                mt = e["mtime"]
                if m is None or mt > m:
                    de["last_modified"] = mt

    top_dirs = {k: v for k, v in dirs.items() if k.count("/") == 0 and k != "."}
    top_files = [f for f in files if "/" not in f["path"]]

    # B6 + B5: ONE walk answers both — the digest of the corpus this
    # document lists, and what the walk could not see.
    # `skipped_unstatable` is the loud half: files that vanished or
    # denied stat mid-walk. Silence here used to poison the drift gate
    # (the digest of a corpus that is not the corpus); now inventory
    # names the count, doctor echoes it, and apply refuses to tend
    # against it.
    fingerprint, fp_health = scan.fingerprint_health()

    inventory = {
        "root": str(root),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "stats": {
            "files": len(files),
            "dirs": len([k for k in dirs if k != "."]),
            "total_bytes": total_bytes,
            "truncated": truncated,
            "git_repo": (root / ".git").exists(),
            "loose_top_level_files": len(top_files),
            "loose_top_level_dirs": len(
                [d for d in top_dirs if not (root / d / "README.md").exists()]
            ),
            "ignore_rules": ignore_rules,
            # B6: the walk's blind spot, counted (paths named in
            # `skipped_unstatable_paths` when non-zero). A non-zero
            # count means something is writing to the corpus under the
            # scan — `apply` refuses such a tend.
            "skipped_unstatable": fp_health["skipped_unstatable"],
            # Phase 11: per-top-dir totals, counted even past the walk
            # cap — the truncation culprit is named, never guessed.
            "top_dir_file_counts": dict(
                sorted(top_dir_counts.items(), key=lambda kv: -kv[1])
            ),
            # Phase 7: subtrees with their own .slopkeeper/manifest.json
            # are separate corpora — invisible above, named here so no
            # one has to `find` for why the counts dropped.
            "nested_corpora": sorted(set(scan.report.nested)),
            # fp6/A2: symlinked directories the walk does not descend —
            # named, so inventory agrees with coverage validation (the
            # audit's dirs: 0 surprise). Plans must cover these entries
            # (keep/unresolved/skip); moves through or onto them are
            # refused by default (symlink_escape; --allow-symlinks).
            "symlinked_dirs": sorted(set(symlinked_dirs)),
        },
        "dirs": [dirs[k] for k in sorted(dirs) if k != "."],
        "files": files,
        "clusters": {
            "by_prefix": {
                k: v[:40]
                for k, v in _prefix_clusters(
                    [f["path"].rsplit("/", 1)[-1] for f in files]
                ).items()
            },
            "by_ext": _count_by(files, "ext", 25),
            "by_month": _count_by(files, "mtime", 24, key_len=7),
        },
    }

    if fp_health["skipped_unstatable"]:
        # Named only when non-zero: the count is the always-on contract,
        # the paths are the diagnostic (B6).
        inventory["stats"]["skipped_unstatable_paths"] = fp_health[
            "skipped_unstatable_paths"
        ]

    if scope is not None:
        # The scoped view names its boundary, so the plan authored from it
        # and the inventory agree on what "in scope" meant.
        inventory["scope"] = scope

    if not fast and len(files) <= 800:
        scan_truncated: list[str] = []
        inventory["references"] = scan_references(root, rel_paths, scan_truncated)
        if scan_truncated:
            # P5: the embedded cross-reference map has a read cap per file,
            # and "nothing cites it" from a half-read file is not the same
            # claim as from a whole one. Counted, like every other cap here.
            inventory["stats"]["reference_scan_truncated_files"] = len(
                set(scan_truncated)
            )
            add_warning(inventory, "citation_scan_truncated")

        # P6 Slice A: the OTHER cross-reference index — which files IMPORT
        # which. Separate from `references` on purpose: an import is a
        # citation the rewrite pass never fixes, so merging the two maps
        # would let an agent read "cited by 3 files, the pass rewrites
        # them" for edges nothing repairs. Keyed the same way (a path or a
        # directory); each row is the importer, its line, the spec as
        # written, and the language, which is the whole answer to "what
        # breaks if this moves?".
        import_truncated: list[str] = []
        imports = scan_imports(root, rel_paths, import_truncated)
        import_stats = imports.pop("stats")
        inventory["imports"] = {
            target: [
                {
                    "file": row["file"],
                    "line": row["line"],
                    "spec": row["spec"],
                    "lang": row["lang"],
                }
                for row in rows
            ]
            for target, rows in imports.items()
        }
        inventory["stats"]["imports"] = import_stats
        if import_truncated:
            inventory["stats"]["import_scan_truncated_files"] = len(
                set(import_truncated)
            )
            add_warning(inventory, "citation_scan_truncated")

    return inventory


def _count_by(
    files: list[dict], key: str, limit: int, key_len: int | None = None
) -> dict:
    counts: dict[str, int] = {}
    for f in files:
        k = str(f.get(key) or "?")
        if key_len:
            k = k[:key_len]
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:limit])
