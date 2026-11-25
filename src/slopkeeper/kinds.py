# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""What a file IS — the one table, the one answer (P3/R7).

Before 0.2.8 five surfaces each kept their own extension set:
`inventory.BINARY_EXTS`, `inventory.CODE_EXTS`, `anchors.CODE_EXTS` (a
byte-for-byte copy of the inventory one), the skip list
`similar.identical_groups` carried inline, `propose.KIND_EXTS` and
`references.TEXT_EXTS`. Every one of them drifted on its own schedule:
`identical_groups` hashed `.webp`/`.tar`/`.mp3`/`.db` files that
inventory already called binary, `anchors` maintained a duplicate of the
code list, and `.sql` was "code" to inventory and "data" to `propose` in
the same run. The audit's R7 calls that class of bug structural, not
accidental: two copies of a table drift waiting to happen.

So this module owns every extension set and answers the two questions
everyone asks:

- `kind_of(path)` — the TABLE answer, no I/O: "binary" | "code" | "doc"
  | "text", or None when the extension says nothing.
- `classify(path, sniff=True)` — the ANSWER: the table when it speaks,
  the NUL sniff when it does not. inventory's `_kind` used to be the
  only caller of the sniff; `references.is_text_file` now goes through
  here too, so "scannable for citations" and "kind: text" cannot
  disagree (an unlisted `.dart` was text to inventory and invisible to
  the reference scan).

The axes stay separate on purpose. `KIND_EXTS` (propose's destination
buckets: docs/data/code/media) is NOT the content kind — a `.pdf` is
binary content that belongs in `docs/`. What IS guaranteed is that each
axis has one owner and no member appears twice inside its own axis:
`tests/test_kinds.py` asserts the disjointness and the cross-consumer
agreement that the five old tables could not enforce.

Consumers import the constants FROM here (inventory/anchors/references/
similar/propose keep their module-level names as aliases so nothing
outside had to change).
"""

from __future__ import annotations

from pathlib import Path

# How many bytes the content sniff looks at. inventory's `_kind` used
# exactly this budget; now that `is_text_file` shares the decision, it
# is a contract number rather than one function's local habit.
SNIFF_BYTES = 8192

# Content that is not text. Also the `identical_groups` skip set: byte
# comparison of a JPEG is a waste of a hash, and the same rule now
# covers the extensions identical_groups used to miss (audit: the
# narrower inline list hashed .webp/.tar/.mp3/.db anyway).
#
# P5's filetype batch named the office containers explicitly rather than
# leaving them to the sniff: `.odt`/`.ods`/`.odp`/`.epub` are zips and
# `.doc`/`.xls` are OLE containers, so a table answer costs no open and no
# 8 KiB read per file. Their TITLES still come from extraction
# (inventory's `_extract_head`, where the ODF/EPUB parts are read with
# stdlib `zipfile` — no external binary to be absent).
BINARY_EXTS = frozenset(
    {
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
        ".odt",
        ".ods",
        ".odp",
        ".epub",
        ".doc",
        ".xls",
    }
)

# Code: languages where path segments get built with join()/Path()/
# dirname() math (anchors territory) and where a citation lives in a
# comment (the rewrite pass reads those too). inventory.CODE_EXTS and
# anchors.CODE_EXTS were the same 20 entries maintained twice; this is
# the one copy.
CODE_EXTS = frozenset(
    {
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
)

# The heading-aware document kind. Deliberately `.md` ONLY: the kind
# string is an output contract (`inventory.files[].kind`) and it drives
# title extraction (first `#` line), so widening it is a visible change
# to every inventory — the filetype batch that widens it is P5's, with
# its own tests.
DOC_EXTS = frozenset({".md"})

# The extensions the table vouches for as readable text — the fast path
# that keeps `is_text_file` from opening a file to decide. The NUL sniff
# still gets the last word on anything NOT listed here.
TEXT_EXTS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".org",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".html",
        ".htm",
        ".xml",
        ".tex",
        ".bib",
        ".css",
        ".scss",
        ".ipynb",
        ".log",
        ".jsonl",
        # SVG is XML that cites things (audit P5 filetype batch: "SVG
        # explicitly text"): it carries `xlink:href` and `<image
        # src="diagram-data.csv">` style paths, so it must be scannable
        # for citations and citable by name, not left to the NUL sniff.
        ".svg",
        # code, repeated so the set is readable standalone; frozenset
        # unions make the duplication harmless.
        *CODE_EXTS,
    }
)

# --- files known by NAME, not by extension (P5 filetype batch) --------------
#
# `Makefile`, `Dockerfile`, `LICENSE`, `.env` — the classics that carry no
# extension at all, and constantly cite filenames. The extension tables
# cannot answer for them, so the old `is_text_file` (extension-gated, the
# audit's line) made them invisible to the reference scan while inventory's
# sniff called them text. One name table now answers both questions — the
# kind, and whether the bare name is citable — and it is consulted only
# AFTER the extension tables have had their say, so no extension that
# already has an answer changes: `README.txt` is text (the extension
# speaks), `README` is doc (the name does).
#
# Exact lowercase names whose kind the table vouches for.
NAME_KINDS: dict[str, str] = {
    "makefile": "code",
    "dockerfile": "code",
    "justfile": "code",
    "gemfile": "code",
    "rakefile": "code",
    "vagrantfile": "code",
    "jenkinsfile": "code",
    "procfile": "code",
    "brewfile": "code",
    ".gitignore": "text",
    ".dockerignore": "text",
    ".npmignore": "text",
    ".gitattributes": "text",
    ".editorconfig": "text",
    ".shellcheckrc": "text",
}
# Name families: a documented prefix plus a separator (or nothing). The
# separator requirement is what keeps `readmelater` from being a README —
# `LICENSE-MIT` and `.env.production` are the shapes that matter.
#
# Deliberately NOT here: `NOTICE`, `AUTHORS`, `NEWS`. They are real
# classics, but they are also ordinary prose words, and a basename is
# matched case-INsensitively — filing `NEWS` would make the word "news"
# in a sentence a citation of it, and the rewrite pass would happily
# rewrite that word when the file moved. The name table is for files whose
# names no sentence uses.
NAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("readme", "doc"),
    ("changelog", "doc"),
    ("contributing", "doc"),
    ("code_of_conduct", "doc"),
    ("license", "text"),
    ("licence", "text"),
    (".env", "text"),
)
_NAME_BOUNDARY = ("", ".", "-", "_", " ")


def name_kind(name: str) -> str | None:
    """The table's answer for a NAME (no extension needed), or None.

    Lowercased, and only reached when no extension answered: see
    `NAME_KINDS` for why the ordering matters. This is also the one
    predicate `CitationIndex` asks when it decides whether a dotless name
    is a citable basename — `Makefile` has no dot, and the old rule
    ("a bare word is never a basename") made every file in this table
    uncitable.
    """
    low = name.lower()
    if low in NAME_KINDS:
        return NAME_KINDS[low]
    for prefix, kind in NAME_PREFIXES:
        if not low.startswith(prefix):
            continue
        rest = low[len(prefix) :]
        if rest == "" or rest[0] in _NAME_BOUNDARY:
            return kind
    return None


# --- propose's destination buckets (a different axis, one owner) -------
#
# `propose` files a corpus into docs/ data/ scripts/ media/ by extension.
# The buckets are a routing choice, not a claim about content: `.pdf` is
# binary content that belongs with the documents. Two rules keep the old
# drift out: the buckets are disjoint from each OTHER (`.sql` used to sit
# in both data and code — it is code, per CODE_EXTS, and says so once),
# and `code` IS `CODE_EXTS`, so a new language lands in one place.

DOCS_EXTS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".org",
        ".pdf",
        ".docx",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".epub",
        ".doc",
        ".xls",
        ".tex",
        ".html",
        ".htm",
    }
)
DATA_EXTS = frozenset(
    {
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
        ".log",
    }
)
MEDIA_EXTS = frozenset(
    {
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
    }
)

KIND_EXTS: dict[str, frozenset[str]] = {
    "docs": DOCS_EXTS,
    "data": DATA_EXTS,
    "code": CODE_EXTS,
    "media": MEDIA_EXTS,
}

# Where the by-NAME classics get filed (P5 filetype batch). The extension
# buckets cannot answer for a file with no extension, and `propose` used to
# drop every one of them into `misc/` — a corpus's Makefile, Dockerfile and
# LICENSE sorted with the unidentified pile. Same discipline as the
# buckets: one owner, and a prefix key matches on the same boundary rule
# `name_kind` uses.
NAME_ROUTING: dict[str, str] = {
    "makefile": "code",
    "dockerfile": "code",
    "justfile": "code",
    "gemfile": "code",
    "rakefile": "code",
    "vagrantfile": "code",
    "jenkinsfile": "code",
    "procfile": "code",
    "brewfile": "code",
    ".gitignore": "data",
    ".dockerignore": "data",
    ".npmignore": "data",
    ".gitattributes": "data",
    ".editorconfig": "data",
    ".shellcheckrc": "data",
    ".env": "data",
    "readme": "docs",
    "changelog": "docs",
    "contributing": "docs",
    "code_of_conduct": "docs",
    "license": "docs",
    "licence": "docs",
}


def _name_prefix_hit(name_low: str, table: dict[str, str]) -> str | None:
    """Exact name match, else a documented prefix + boundary in `table`."""
    if name_low in table:
        return table[name_low]
    for key, value in table.items():
        if not name_low.startswith(key):
            continue
        rest = name_low[len(key) :]
        if rest == "" or rest[0] in _NAME_BOUNDARY:
            return value
    return None


def route_of(path: Path | str) -> str | None:
    """`propose`'s destination bucket for `path`: docs|data|code|media|None.

    The extension buckets answer when the extension speaks; the name table
    answers when it does not (Makefile, LICENSE, .env). One owner for the
    routing answer, exactly as `kind_of` is the one owner for the content
    answer — `propose._kind_of` is this function.
    """
    p = Path(path)
    ext = p.suffix.lower()
    for bucket, exts in KIND_EXTS.items():
        if ext in exts:
            return bucket
    return _name_prefix_hit(p.name.lower(), NAME_ROUTING)


def kind_of(path: Path | str) -> str | None:
    """The table's answer for `path`: binary|code|doc|text, or None.

    None means the table is silent — the caller either sniffs (`classify`)
    or treats the file as unknown. No I/O.

    The extension speaks first; a name answers only when no extension did
    (so `README.txt` stays "text" by extension and `README` is "doc" by
    name — nothing that already had an answer changes).
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in BINARY_EXTS:
        return "binary"
    if ext in CODE_EXTS:
        return "code"
    if ext in DOC_EXTS:
        return "doc"
    if ext in TEXT_EXTS:
        return "text"
    return name_kind(p.name)


def sniff_binary(path: Path, budget: int = SNIFF_BYTES) -> bool:
    """True when the head holds a NUL, or the file cannot be read.

    Unreadable counts as binary: inventory's `_kind` made that call
    first, and it is the safe direction — an unreadable file is neither
    inventoried for its text nor rewritten, and a name that cannot be
    read cannot cite anything.
    """
    try:
        with Path(path).open("rb") as f:
            return b"\0" in f.read(budget)
    except OSError:
        return True


def classify(path: Path | str, sniff: bool = True, head: bytes | None = None) -> str:
    """The one kind answer: binary | code | doc | text.

    The table decides when the extension speaks. When it does not, the
    NUL sniff does — `head` is the caller's already-read first bytes
    (the single-pass scanner has them) so a classification never costs
    an extra open. With `sniff=False` an unknown extension falls back to
    "text", the pre-0.2.8 `is_text_file` shape for callers that have no
    file to open.
    """
    kind = kind_of(path)
    if kind is not None:
        return kind
    if not sniff:
        return "text"
    if head is None:
        return "binary" if sniff_binary(Path(path)) else "text"
    return "binary" if b"\0" in head else "text"


def is_binary(path: Path | str) -> bool:
    return classify(path) == "binary"


def is_code(path: Path | str) -> bool:
    """`path` is one of the table's code files (no I/O).

    The name table counts, not just the extension: a Makefile or Dockerfile
    builds paths out of literals exactly like a `.sh` does, and the anchor
    scan reads those (`anchors.scan_code_anchors` gates on this).
    """
    return kind_of(path) == "code"


def is_citable_name(name: str) -> bool:
    """Can a dotless `name` be a citation target?

    `CitationIndex.by_basename` requires a dot (the "a bare word is never a
    basename" rule that keeps prose out of the citation index) — which also
    kept every extensionless classic out of it. A name the table vouches
    for is not prose: `Makefile`, `Dockerfile`, `LICENSE`, `README` are
    file names people cite, so the index admits them.
    """
    return name_kind(name) is not None


def is_scannable(path: Path | str, head: bytes | None = None) -> bool:
    """`path` can be read as text and therefore can HOLD a citation.

    `references.is_text_file` is this function. It used to be a
    membership test on TEXT_EXTS alone, which made an unlisted `.dart`
    or `.tf` file invisible to the citation scan while inventory called
    it text — the two surfaces now ask the same question and get the
    same answer.
    """
    return classify(path, head=head) != "binary"
