# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Corpus inventory: the structured, token-conscious view an agent plans from.

Output is a single JSON document under the standard envelope. Designed so a
model can read it whole for a few hundred files, or work from `summary`
clusters for thousands.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .ignore import corpus_walk, load_ignore, nested_corpora
from .models import corpus_fingerprint
from .references import is_text_file, scan_references

HEAD_BYTES = 4096
HEAD_CHARS = 400

BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".mp3",
    ".mp4",
    ".mov",
    ".wav",
    ".sqlite",
    ".db",
    ".pyc",
    ".so",
    ".dylib",
    ".exe",
    ".woff",
    ".woff2",
    ".ttf",
}
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
}
NO_SIGNAL = "signal: none (binary, no extractor)"

_XML_TAG_RE = re.compile(r"<[^>]+>")
_XML_SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=None)
def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _extract_head(fp: Path, ext: str) -> str | None:
    """First lines via an optional system extractor; None = none/failure.

    Every failure mode (no extractor, non-zero exit, timeout, empty
    output) returns None so the caller emits the loud no-signal marker —
    a head is a hint, never a crash, and never silence.
    """
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
    ext = path.suffix.lower()
    if ext in BINARY_EXTS:
        return "binary"
    if ext in CODE_EXTS:
        return "code"
    if ext == ".md":
        return "doc"
    if is_text_file(path):
        return "text"
    # Unknown extension: sniff for NUL bytes.
    try:
        with path.open("rb") as f:
            return "binary" if b"\0" in f.read(8192) else "text"
    except OSError:
        return "binary"


def _head(root: Path, rel: Path, kind: str) -> dict:
    if kind == "binary":
        # Loud by default: a binary head is either extracted text or the
        # explicit no-signal marker — never a silent None (the field
        # report's pdftotext/md5sum archaeology, now tool output).
        extracted = _extract_head(root / rel, rel.suffix.lower())
        if extracted is None:
            return {"title": None, "head": NO_SIGNAL}
        title = next(
            (s.strip()[:120] for s in extracted.splitlines() if s.strip()), None
        )
        return {"title": title, "head": extracted[:HEAD_CHARS]}
    try:
        with (root / rel).open("rb") as f:
            raw = f.read(HEAD_BYTES)
    except OSError:
        return {"title": None, "head": None}
    text = raw.decode("utf-8", errors="replace")
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


def collect(
    root: Path,
    depth: int | None = None,
    fast: bool = False,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict:
    """Walk root and build the inventory document (no envelope)."""
    root = root.resolve()
    ignore = load_ignore(root)
    files: list[dict] = []
    dirs: dict[str, dict] = {}
    total_bytes = 0
    rel_paths: list[Path] = []
    truncated = False
    top_dir_counts: dict[str, int] = {}

    for dirpath, dirnames, filenames in corpus_walk(root):
        rel_dir = Path(dirpath).resolve().relative_to(root)
        d_depth = len(rel_dir.parts)
        if depth is not None and d_depth >= depth:
            dirnames[:] = []
        dir_key = rel_dir.as_posix() if rel_dir.parts else "."
        top_key = rel_dir.parts[0] if rel_dir.parts else "(root)"
        if truncated:
            # Past the cap the entries stop; the counting does not.
            # The truncation culprit is named from these totals, not
            # guessed after a silent walk-stop.
            for _fn in filenames:
                top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
            continue
        dirs.setdefault(
            dir_key,
            {"path": dir_key, "file_count": 0, "total_files": 0, "last_modified": None},
        )
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            if len(rel_paths) >= max_files:
                truncated = True
                top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
                continue
            try:
                st = fp.stat()
            except OSError:
                continue
            rel = fp.relative_to(root)
            total_bytes += st.st_size
            top_dir_counts[top_key] = top_dir_counts.get(top_key, 0) + 1
            kind = _kind(fp)
            e = {
                "path": rel.as_posix(),
                "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                ),
                "ext": fp.suffix.lower(),
                "kind": kind,
            }
            # Binaries get a head too (Phase 11): extracted text via an
            # optional system extractor, or the loud no-signal marker.
            e.update(_head(root, rel, kind))
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

    inventory = {
        "root": str(root),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": corpus_fingerprint(root),
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
            "ignore_rules": ignore.rule_count,
            # Phase 11: per-top-dir totals, counted even past the walk
            # cap — the truncation culprit is named, never guessed.
            "top_dir_file_counts": dict(
                sorted(top_dir_counts.items(), key=lambda kv: -kv[1])
            ),
            # Phase 7: subtrees with their own .slopkeeper/manifest.json
            # are separate corpora — invisible above, named here so no
            # one has to `find` for why the counts dropped.
            "nested_corpora": nested_corpora(root),
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

    if not fast and len(files) <= 800:
        inventory["references"] = scan_references(root, rel_paths)

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
