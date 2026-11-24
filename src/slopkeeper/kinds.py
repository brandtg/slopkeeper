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
        # code, repeated so the set is readable standalone; frozenset
        # unions make the duplication harmless.
        *CODE_EXTS,
    }
)

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


def kind_of(path: Path | str) -> str | None:
    """The table's answer for `path`: binary|code|doc|text, or None.

    None means the extension is silent — the caller either sniffs
    (`classify`) or treats the file as unknown. No I/O.
    """
    ext = Path(path).suffix.lower()
    if ext in BINARY_EXTS:
        return "binary"
    if ext in CODE_EXTS:
        return "code"
    if ext in DOC_EXTS:
        return "doc"
    if ext in TEXT_EXTS:
        return "text"
    return None


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
    """`path`'s extension is a known code extension (table only, no I/O)."""
    return Path(path).suffix.lower() in CODE_EXTS


def is_scannable(path: Path | str, head: bytes | None = None) -> bool:
    """`path` can be read as text and therefore can HOLD a citation.

    `references.is_text_file` is this function. It used to be a
    membership test on TEXT_EXTS alone, which made an unlisted `.dart`
    or `.tf` file invisible to the citation scan while inventory called
    it text — the two surfaces now ask the same question and get the
    same answer.
    """
    return classify(path, head=head) != "binary"
