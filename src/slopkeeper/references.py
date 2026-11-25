# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Cross-reference scanning: which files cite which corpus paths.

Why it exists: moving a file (or renaming a directory) that is referenced
by other corpus files breaks those references. The apply step uses this
to stamp a risk level on every move; the agent uses it to fix references
in the plan (e.g. by including an edit step or flagging the move for
human attention).

Path-keyed (reference-rewrites PRD Phase 2): a citation is resolved to
the corpus path (or directory) it names — an exact path, a suffix after
`./`/`../`/a longer prefix, a directory prefix (including convention
strings like `reports/_org/<date>_<topic>.md`), a wikilink target, or a
basename. A bare word is never a citation; a basename with an extension
is, with word-ish boundaries — so is a name the kinds table vouches for
(`Makefile`, `LICENSE`, `README`), which has no extension and is not
prose either (P5 filetype batch). An ambiguous basename (several corpus
files share it) is NEVER attributed — the old first-seen hack attributed
a mention to whichever same-named file the scan saw first, which could
rewrite a citation onto the wrong file.

Matching is case- and composition-insensitive where it always was
(basenames) and now also for paths and directories (P5: an NFD-named Mac
file cited by NFC text is the same file); storage, hashing and the
fingerprint stay byte-exact, and no folded lookup ever overrides a
byte-exact one.

Bounded and quiet: text files only, per-file read cap. Un-capped since
Phase 3 (reference-rewrites PRD): no file-count ceiling anywhere — the
stream scan reads any corpus size linearly and reports loud counts.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
import unicodedata
from pathlib import Path

from . import kinds, scan
from .models import HIDDEN_SKIP_DIRS, SlopKeeperError, resolve_chains


# Extensions we are willing to read for reference scanning. The table
# lives in kinds.py now (P3/R7) so inventory, anchors, similar, propose
# and this module cannot disagree about what a file is; the name stays
# here as the alias test imports use. Membership is the FAST path —
# `is_text_file` falls back to the NUL sniff for anything unlisted, so
# a `.dart` or `.tf` file that inventory calls text is finally scannable
# for citations too (it used to be invisible to this module alone).
TEXT_EXTS = kinds.TEXT_EXTS

_PER_FILE_CAP = 512 * 1024  # don't index beyond 512KB of a single file
_MAX_MENTIONS_PER_TARGET = 64  # stop counting after this; it's "widely cited"
# Citation `text` snippets (sites, skipped rows, refs): long enough to
# author an exact `find` from without guessing (C6 — the 200-char cap
# truncated mid-token on long table rows).
_SNIPPET_CAP = 2000


def is_text_file(path: Path, head: bytes | None = None) -> bool:
    """Can `path` be read as text — and therefore hold a citation?

    kinds.classify's answer (extension table, then the NUL sniff), which
    is the same question inventory answers for `files[].kind`. Passing
    `head` (the first bytes, already read) keeps a scan that has the
    bytes from opening the file twice to decide whether to open it.
    """
    return kinds.is_scannable(path, head=head)


def _name_pattern(name: str) -> re.Pattern[str]:
    """Match `name` with word-ish boundaries on both sides.

    The negative lookarounds reject longer-name mentions (`oldapi.md.bak`
    must not count as a citation of `api.md`) while keeping path-ish
    mentions (`docs/api.md`), since `/` is neither a word char nor a dot.
    """
    return re.compile(
        # Leading: not preceded by a name-ish char (rejects xapi.md).
        # Trailing: not followed by a name-ish char, and not by
        # ".<alnum>" (rejects oldapi.md.bak / api.md.5) — but a bare
        # trailing period is ALLOWED, so the common sentence-final
        # citation "See dsa.md." is still detected.
        r"(?<![\w.-])" + re.escape(name) + r"(?![\w-]|\.[A-Za-z0-9])",
        re.IGNORECASE,
    )


def _name_searcher(name: str):
    """A `line -> bool` test for one queried name (P5 normalize-on-match).

    `refs <name>` and `refs --scan-path` take the name from the caller, and
    the caller's normal form need not be the filesystem's: an agent asking
    about `café.md` (NFC) means the file the Mac stored as `café.md`
    (NFD). An ASCII query — every ASCII corpus, and every query an agent
    types without accents — gets the historical pattern untouched; only a
    non-ASCII query adds the folded leg, so no existing answer changes.
    """
    raw = _name_pattern(name)
    if name.isascii():
        return raw.search
    folded = re.compile(
        r"(?<![\w.-])" + re.escape(_fold(name)) + r"(?![\w-]|\.[A-Za-z0-9])"
    )

    def search(line: str) -> bool:
        return bool(raw.search(line)) or bool(folded.search(_fold(line)))

    return search


# --- normalization (P5: normalize on MATCH, never on store) -----------------
#
# A corpus that has lived on a Mac holds both normal forms of the same
# name: the Finder writes `café.md` decomposed (NFD: `e` + U+0301) while
# the text that cites it — typed on Linux, copied out of a terminal,
# written by an agent — composes it (NFC: `é`). Byte-compared, those are
# two different names, so the citation was invisible: `refs` found no
# sites and the rewrite pass left it pointing at nothing. The audit's
# test-coverage gap names it (the fingerprint treats them as unrelated
# files, "byte-correct but agent-surprising").
#
# The fix is deliberately ONE-SIDED. Filenames, hashes, plans and the
# journal stay byte-exact — a rename that quietly normalized the corpus
# would be its own data bug, and `fingerprint` promises byte identity.
# Only MATCHING folds, at the moment a token is compared against the
# index. Two flavors, because the two jobs differ:
#
#   `_nfd`  composition-only, case-preserving — for path/dir/stem
#           resolution, where folding CASE too would start resolving
#           `API.md` to `api.md` on a case-sensitive filesystem (a
#           rewrite onto the wrong file, which is worse than the miss);
#   `_fold` composition + casefold — for basename matching, which has
#           always been case-INsensitive (`re.IGNORECASE`, and
#           `canonical_basename`'s casefold map), so folding further only
#           extends the rule that already exists.
#
# No cache on either: query strings come out of corpus text and are
# unbounded, and an lru_cache over them is a leak on a big corpus. Both
# short-circuit on `isascii()`, which is every ASCII corpus.


def _nfd(text: str) -> str:
    """Composition-insensitive key, case preserved."""
    return text if text.isascii() else unicodedata.normalize("NFD", text)


def _fold(text: str) -> str:
    """Case- and composition-insensitive key (the basename matcher's)."""
    if text.isascii():
        return text.casefold()
    return unicodedata.normalize("NFD", text).casefold()


def _fold_spans(text: str) -> tuple[str, list[int]]:
    """`(_fold(text), index_map)` with the map back to `text`'s positions.

    Folding is not length-preserving (NFD splits `é` into two characters,
    casefold expands `ß` into `ss`), so a match computed on the folded
    string cannot report where it landed in the ORIGINAL line — and the
    rewrite pass needs those spans to decide containment and to write.
    Each character of `text` contributes its own folded run to the output,
    so `index_map[j]` is the position in `text` of the character that
    produced folded position `j`, and a folded span `[a, b)` maps back to
    `[index_map[a], index_map[b])`. A character that folds away entirely
    (none in practice for path-ish text) keeps its own character so the
    map never misaligns.
    """
    if text.isascii():
        return text.casefold(), list(range(len(text) + 1))
    parts: list[str] = []
    positions: list[int] = []
    for i, ch in enumerate(text):
        folded = unicodedata.normalize("NFD", ch).casefold()
        if not folded:
            folded = ch
        parts.append(folded)
        positions.extend([i] * len(folded))
    positions.append(len(text))
    return "".join(parts), positions


# --- the citation index ------------------------------------------------------


class CitationIndex:
    """Every citable path in the corpus, plus its basenames and dirs.

    Built once per scan from the corpus rel paths. `by_basename` keeps
    the legacy candidate rules (>= 3 chars, contains a dot) — a bare
    word is never a basename — with ONE exception (P5 filetype batch): a
    name the kinds table vouches for (`Makefile`, `Dockerfile`, `LICENSE`,
    `README`...) is citable without a dot, because it is a file name and
    not prose. `ambiguous_basenames` is the honest replacement for
    first-seen attribution: several files share the basename, so a bare
    mention cannot be resolved.

    Every set/dict here holds the RAW corpus strings — the values that
    come out of the tool. The `norm_*` maps are lookup indexes over the
    same values (see the normalization block above); a fold collision
    keeps the first path the walk reached, which is deterministic because
    the walk is sorted.
    """

    def __init__(self, rel_paths: list[Path]):
        self.paths: set[str] = set()
        self.dirs: set[str] = set()
        self.by_basename: dict[str, list[str]] = {}
        self.by_stem: dict[str, list[str]] = {}
        self.norm_paths: dict[str, str] = {}
        self.norm_dirs: dict[str, str] = {}
        self.norm_stems: dict[str, str] = {}
        for p in rel_paths:
            rel = p.as_posix()
            self.paths.add(rel)
            self.norm_paths.setdefault(_nfd(rel), rel)
            parts = p.parts[:-1]
            for i in range(1, len(parts) + 1):
                d = "/".join(parts[:i])
                self.dirs.add(d)
                self.norm_dirs.setdefault(_nfd(d), d)
            base = p.name
            if (len(base) >= 3 and "." in base) or kinds.is_citable_name(base):
                self.by_basename.setdefault(base, []).append(rel)
            # Obsidian-style extensionless wikilink targets resolve
            # against stem paths (the rel path minus its extension).
            stem = p.with_suffix("").as_posix()
            self.by_stem.setdefault(stem, []).append(rel)
            self.norm_stems.setdefault(_nfd(stem), stem)
        self.ambiguous_basenames = {
            b for b, ps in self.by_basename.items() if len(ps) > 1
        }
        # The basename matcher is case- AND composition-insensitive, so a
        # hit's text ("Api.md", or the other normal form of the name) must
        # map back to the canonical corpus key.
        self._by_basename_fold: dict[str, str] = {}
        for b in self.by_basename:
            self._by_basename_fold.setdefault(_fold(b), b)

    def canonical_basename(self, matched: str) -> str | None:
        """The corpus basename a matcher hit names, folded both sides."""
        return self._by_basename_fold.get(_fold(matched))

    # --- normalize-on-match lookups -------------------------------------
    #
    # Exact first, always: the byte-exact answer is the truth and the
    # common case, and trying it first means an ASCII corpus never pays
    # for a fold. The folded leg is what finds the NFD/NFC twin.

    def match_path(self, cand: str) -> str | None:
        """The corpus path `cand` names (byte-exact or compositionally equal)."""
        if cand in self.paths:
            return cand
        return self.norm_paths.get(_nfd(cand))

    def match_dir(self, cand: str) -> str | None:
        """The corpus directory `cand` names, same rule."""
        if cand in self.dirs:
            return cand
        return self.norm_dirs.get(_nfd(cand))

    def stem_paths(self, cand: str) -> list[str] | None:
        """The files whose stem-path `cand` names (a wikilink target)."""
        paths = self.by_stem.get(cand)
        if paths is None:
            key = self.norm_stems.get(_nfd(cand))
            if key is not None:
                paths = self.by_stem.get(key)
        return paths

    def basename_paths(self, base: str) -> list[str] | None:
        """The files a bare basename `base` names, folded both sides."""
        paths = self.by_basename.get(base)
        if paths is None:
            canonical = self.canonical_basename(base)
            if canonical is not None:
                paths = self.by_basename.get(canonical)
        return paths


# --- tokenization ------------------------------------------------------------

_WIKILINK_RE = re.compile(
    r"\[\[(?P<target>[^\[\]|#]+)(?:#(?P<frag>[^\]|]*))?(?:\|(?P<alias>[^\]]*))?\]\]"
)
_MD_LINK_RE = re.compile(r'\]\((?P<target>[^)\s]+)(?:\s+"(?P<title>[^"]*)")?\)')
# D6: a backtick-quoted span is ONE token, spaces included — the
# spaced-filename case
# (`notes/Sample Transcript - Owner's Phone .md`)
# shattered at the spaces and was never recorded. Only interiors that
# name a path shape (a slash, or an extension tail) become tokens;
# plain `prose` stays out.
_BACKTICK_RE = re.compile(r"`(?P<target>[^`\n]+)`")
_BACKTICK_SHAPE_RE = re.compile(r"/|\.[A-Za-z0-9]+$")
# One atom of a path segment: a plain name char (dots, hyphens and glob
# stars included) or a brace group. Glob and brace tails survive into
# the rewrite decision (the 0.1.9 report's D2/D11: the old segment class
# stopped at `{` and `*`, so `reports/_org/x*.md` scanned as
# `reports/_org/` and the glob tail was mangled onto a dead pattern).
# `?` is deliberately NOT an atom — it is overwhelmingly a query marker
# in corpus text (`docs/api.md?x=1` in HTML/anchors), and gluing it
# would un-resolve real citations. `<`/`>` stay outside too, so
# `<placeholder>` convention strings still truncate at the angle bracket.
#
# P5: the class is `\w` (Unicode), not `[A-Za-z0-9_]`, plus the combining
# marks `re`'s `\w` leaves out (a mark is not alphanumeric, so `é` stored
# decomposed — `e` + U+0301, the form macOS writes — used to SHATTER the
# token into `caf` and the citation was invisible to every leg of the
# scan, not just the folded one). An ASCII corpus is unaffected: the
# classes agree there.
_PATH_MARKS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff"
_PATH_SEGMENT_ATOM = rf"(?:[\w.\-{_PATH_MARKS}]|\*|\{{[^{{}}\s/]*\}})"
_PATH_TOKEN_RE = re.compile(
    # Not preceded by a name-ish char (rejects xapi.md)...
    r"(?<![\w.\-])"
    # ...optional ./ or ../ prefixes...
    r"(?:\.{1,2}/)*"
    # ...first segment (starts with a name char), then one or more
    # /segments. A segment may be empty after the slash, so convention
    # strings truncate cleanly at the first non-path char:
    # `reports/_org/<date>_...` scans as `reports/_org/` — a directory
    # citation. Commas live only inside brace groups, so prose
    # ("docs/api.md, and …") still stops at the comma.
    #
    # The mark class is excluded from the FIRST character: a token that
    # begins with a combining mark is not a path, and refusing it keeps
    # the lookbehind's job of rejecting mid-word fragments.
    r"\w" + _PATH_SEGMENT_ATOM + r"*"
    r"(?:/" + _PATH_SEGMENT_ATOM + r"*)+"
)


_HOST_FIRST_SEG_RE = re.compile(r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")


def _resolves_in_corpus(
    rest: str, index: CitationIndex, citing_dir: str | None
) -> bool:
    """True when a host-CANDIDATE token actually names corpus content (A3).

    The pre-fix heuristic dropped any token whose first segment looked
    like a hostname, so `my.files/report.md` — a plain citation under a
    dotted directory (`data.backup/`, `v1.0/`, `my.files/` are common
    corpus layouts) — was silently never rewritten, and after --execute
    the citation pointed at nothing. If the corpus has the path, it IS
    a path, full stop. Containment is deliberately EXACT (the token, its
    directory prefixes, and the citing-dir read): a token whose tail
    alone matches some unrelated corpus file (`example.com/index.md`
    when the corpus has a root index.md) stays a host — the
    suffix/basename fallbacks belong to _resolve, not to this question.
    """
    if index.match_path(rest):
        return True
    stripped = rest.rstrip("/")
    if stripped and index.match_dir(stripped):
        return True
    cur = ""
    for seg in rest.split("/")[:-1]:
        cur = f"{cur}/{seg}" if cur else seg
        if index.match_dir(cur):
            return True
    if citing_dir is not None and _citing_dir_hit(
        _resolve_from(citing_dir, rest), index
    ):
        return True
    return False


def _urlish(
    line: str,
    start: int,
    token: str,
    index: CitationIndex | None = None,
    citing_dir: str | None = None,
) -> bool:
    """A token with a URL scheme, an email, or a bare host: never a
    corpus citation and never rewritten. `https://docs.acme.example/.../index.html`
    scans as the path token `docs.acme.example/...` (the regex cannot
    carry the scheme) and its tail used to resolve via the moved file's
    basename — mangling a web address into a bare filename. A `://`
    immediately before the token is a scheme; an `@` anywhere in the
    token is an email; a dotted first segment (after any ./ ../
    prefixes) matching a host pattern is a host — UNLESS the token
    resolves against the corpus (A3): with an `index` given, anything
    that names a corpus path, a corpus directory, a path under one, or
    a sibling of the citing file is a path, not a host. Version-y
    directory names (`spark-1.2.3`) are corpus paths, not hosts — the
    TLD test keeps them. Without an index (no corpus view), the
    historical conservative heuristic applies."""
    if line[:start].endswith("://"):
        return True
    if "@" in token:
        return True
    rest = token
    while rest.startswith("./") or rest.startswith("../"):
        rest = rest.partition("/")[2]
    first = rest.split("/", 1)[0]
    if "." not in first:
        return False
    if index is not None:
        if _resolves_in_corpus(rest, index, citing_dir):
            return False
        return bool(_HOST_FIRST_SEG_RE.match(first))
    tail = first.rsplit(".", 1)[-1]
    return len(tail) >= 2 and tail.isalpha()


def _path_tokens(
    line: str,
    index: CitationIndex | None = None,
    citing_dir: str | None = None,
) -> list[tuple[str, str]]:
    """Path-ish tokens on one line: (token, kind) with kind in
    wikilink|link|path. Bare basenames are not tokens — they are matched
    separately against the index's candidate basenames (a bare word is
    never a citation). URL-scheme and host tokens are not citations
    (C2) and are dropped here, for the scans and the rewrite pass
    alike; pass the CitationIndex (and the citing file's dir) so dotted
    corpus paths are never mistaken for hosts (A3)."""
    tokens: list[tuple[str, str]] = []
    for m in _WIKILINK_RE.finditer(line):
        tokens.append((m.group(1).strip(), "wikilink"))
    for m in _MD_LINK_RE.finditer(line):
        tokens.append((m.group(1).strip(), "link"))
    for m in _PATH_TOKEN_RE.finditer(line):
        if _urlish(line, m.start(), m.group(0), index=index, citing_dir=citing_dir):
            continue
        tokens.append((m.group(0), "path"))
    return tokens


def _token_spans(
    line: str,
    index: CitationIndex | None = None,
    citing_dir: str | None = None,
) -> list[tuple[str, str, int, int]]:
    """_path_tokens with replacement spans: (stripped_text, kind, start, end).

    The span is what a rewrite may substitute: the target inside the
    brackets for wikilinks and Markdown links (fragments, aliases and
    titles are never touched), the whole match for path tokens. A path
    token identical to a link span is dropped — the link site owns that
    span (one replacement, not two). index/citing_dir thread the A3
    corpus-resolution override to the host heuristic.
    """
    spans: list[tuple[str, str, int, int]] = []
    link_ranges: list[tuple[int, int]] = []
    backtick_ranges: list[tuple[int, int]] = []
    for m in _BACKTICK_RE.finditer(line):
        interior = m.group(1).strip()
        if not interior or not _BACKTICK_SHAPE_RE.search(interior):
            continue
        spans.append((interior, "backtick", m.start(1), m.end(1)))
        backtick_ranges.append((m.start(1), m.end(1)))
    for m in _WIKILINK_RE.finditer(line):
        spans.append((m.group(1).strip(), "wikilink", m.start(1), m.end(1)))
        link_ranges.append((m.start(1), m.end(1)))
    for m in _MD_LINK_RE.finditer(line):
        spans.append((m.group(1).strip(), "link", m.start(1), m.end(1)))
        link_ranges.append((m.start(1), m.end(1)))
    for m in _PATH_TOKEN_RE.finditer(line):
        # A path token inside a link target is part of that target —
        # the link site owns the region (one replacement, not two).
        if any(s <= m.start() and m.end() <= e for s, e in link_ranges):
            continue
        # Likewise inside a backtick span: the quoted interior IS the
        # token — a partial sub-token (`notes/Sample …`) must not split
        # it (D6).
        if any(s <= m.start() and m.end() <= e for s, e in backtick_ranges):
            continue
        if _urlish(line, m.start(), m.group(0), index=index, citing_dir=citing_dir):
            continue
        spans.append((m.group(0), "path", m.start(), m.end()))
    return spans


# --- resolution --------------------------------------------------------------


def _citing_dir_hit(cand: str, index: CitationIndex) -> tuple[str, str] | None:
    """The resolution when a citing-dir candidate names a corpus entry.

    The target is the CANONICAL corpus spelling, not the candidate's (P5):
    a citing file beside an NFD-named neighbor writes `café.md` composed,
    and the rewrite pass must move the citation onto the path that exists
    on disk, not the one the prose used.
    """
    hit = index.match_path(cand)
    if hit is not None:
        return ("rel_path", hit)
    stripped = cand.rstrip("/")
    hit = index.match_dir(stripped)
    if hit is not None:
        return ("rel_path", hit)
    return None


def _resolve(
    token: str,
    index: CitationIndex,
    is_wikilink: bool = False,
    citing_dir: str | None = None,
) -> tuple[str, str] | None:
    """Resolve a token to (kind, target), longest match first.

    Per suffix (longest first): an exact corpus path beats a directory
    prefix at the same length, and a longer suffix beats a shorter one —
    so `a/b/c` resolves to dir `b/c` over path `c`. Wikilink targets
    additionally match STEM paths (the rel path minus its extension —
    `[[old/dir/note]]` names `old/dir/note.md`); two files sharing a
    stem resolve to ambiguous. Falls back to the token's final segment
    as a basename (the "final segment of a matched token" rule). An
    ambiguous basename resolves to ("ambiguous", basename) and is never
    attributed to a file.

    D4: when `citing_dir` is given, the token is also read against the
    citing file's own directory. Explicitly-relative forms (`./`,
    `../`) are read there FIRST — that is what the form means. Root-ish
    forms get the citing-dir read only after the suffix match comes up
    empty: a truncated root path (`_org/x.md`) must not be reinterpreted
    as citing-dir-relative just because a same-named file happens to
    sit beside the citer (the 0.1.9 report's 1,034-row ambiguous flood
    came from exactly that misread). A hit returns
    ("rel_path", candidate) — the citation's form is the citing dir's,
    so the caller recomputes it rather than suffix-swapping.
    """
    token = token.strip()
    if not token or token in (".", ".."):
        return None
    parts = token.split("/")
    if citing_dir is not None and token.startswith(("./", "../")):
        hit = _citing_dir_hit(_resolve_from(citing_dir, token), index)
        if hit is not None:
            return hit
    for i in range(len(parts)):
        cand = "/".join(parts[i:]).rstrip("/")
        if not cand or cand in (".", ".."):
            continue
        hit = index.match_path(cand)
        if hit is not None:
            return ("path", hit)
        hit = index.match_dir(cand)
        if hit is not None:
            return ("dir", hit)
        if is_wikilink:
            paths = index.stem_paths(cand)
            if paths is not None:
                if len(paths) > 1:
                    return ("ambiguous", cand)
                return ("path", paths[0])
    if citing_dir is not None and "/" in token and not token.startswith("/"):
        hit = _citing_dir_hit(_resolve_from(citing_dir, token), index)
        if hit is not None:
            return hit
    base = parts[-1]
    paths = index.basename_paths(base)
    if paths is None:
        return None
    if len(paths) > 1:
        return ("ambiguous", index.canonical_basename(base) or base)
    return ("basename", paths[0])


class _Hit:
    """The slice of `re.Match` the basename call sites actually use.

    `group(1)` / `start(1)` / `end(1)` — the matched text as it appears
    in the line and its span. Keeping the shape means every consumer of
    the old compiled alternation keeps working untouched.
    """

    __slots__ = ("text", "lo", "hi")

    def __init__(self, text: str, lo: int, hi: int):
        self.text = text
        self.lo = lo
        self.hi = hi

    def group(self, n: int = 0) -> str:
        return self.text

    def start(self, n: int = 0) -> int:
        return self.lo

    def end(self, n: int = 0) -> int:
        return self.hi

    def span(self, n: int = 0) -> tuple[int, int]:
        return (self.lo, self.hi)


# A citation-shaped word: maximal run of the characters the boundary
# lookarounds argue about. A basename whose own characters all live in
# that class can only START at a run boundary (the lookbehind forbids
# `[\w.-]` before it) and can only END where the lookahead allows, so
# splitting the line here costs one C-speed pass and turns "try 5,000
# alternatives at every position" into a trie descent that dies after a
# hop or two on ordinary prose.
_RUN_RE = re.compile(r"[\w.-]+")
# The trailing half of `_name_pattern`'s boundary: what may NOT follow a
# match. A bare `.` is allowed ("See dsa.md." still cites dsa.md), `.bak`
# and `_2` are not.
_BOUND_RE = re.compile(r"[\w-]|\.[A-Za-z0-9]")
# The class the lookbehind forbids BEFORE a match — also what makes a
# run start a candidate start.
_PREV_RE = re.compile(r"[\w.-]")
_TERMINAL = "$"
_SIMPLE_NAME_RE = re.compile(r"[\w.-]+\Z")


class BasenameMatcher:
    """Every candidate basename, matched with word-ish boundaries.

    What it replaces (P3, R6's citation-scan prefilter): one regex whose
    alternation was every candidate basename in the corpus, longest
    first. `refs` on a 5k-file corpus with distinct names spent 2.2s in
    `base_pat.finditer` — `re` walks the whole alternation at every
    position of every line — and 0.24s more COMPILING it, per scan. The
    candidates now live in a character trie over their casefolded text,
    and the line is read as candidate starts plus one descent each.
    Same rule, same answer, no alternation.

    The semantics are `_name_pattern`'s, unchanged: a match starts where
    the previous character is not name-ish, ends where the lookahead
    permits, is case-insensitive, takes the longest candidate that
    qualifies at a start, and never overlaps the previous match.

    Two read modes, chosen once at build time. When every candidate is
    made of name-ish characters (the common corpus), a match must sit
    inside one `_RUN_RE` run, so the line is split once and each run
    start is the only position worth trying. When ANY candidate holds a
    character outside that class — `cpp+.md`, and the spaced filename
    that has always been the hard case (`Sample Transcript - Owner's
    Phone .md`, audit D6) — runs are not the unit any more and every
    position after a non-name character is a candidate, pre-filtered by
    the first-character set so ordinary prose still costs one lookup per
    word rather than a descent.

    Case-insensitivity is `re.IGNORECASE`'s, exactly — which for ASCII
    is `.lower()`, and NOT `.casefold()`: casefold folds `ß` to `ss` and
    would match a name the old pattern never matched. So the trie is
    only trusted where that equivalence is provable, an ASCII candidate
    set against an ASCII line; anything else goes to `_fallback()`'s
    compiled alternation. A corpus of German or Turkish filenames, or a
    line carrying an em dash, pays the old cost; the fuzzer
    (tests/test_references.py, TestBasenameMatcherParity) is what keeps
    that boundary honest rather than convenient.
    """

    __slots__ = (
        "_names",
        "_root",
        "_first",
        "_run_mode",
        "_ascii",
        "_slow",
        "_slow_folded",
    )

    def __init__(self, basenames: list[str]):
        self._names = list(basenames)
        self._root: dict = {}
        self._first: set[str] = set()
        for b in self._names:
            folded = b.lower()
            self._first.add(folded[0])
            node = self._root
            for ch in folded:
                node = node.setdefault(ch, {})
            node[_TERMINAL] = b
        self._run_mode = all(_SIMPLE_NAME_RE.match(b) for b in self._names)
        # The trie's `.lower()` keys are `re.IGNORECASE` only while both
        # sides are ASCII; see the class docstring.
        self._ascii = all(b.isascii() for b in self._names)
        self._slow: re.Pattern[str] | None = None
        self._slow_folded: re.Pattern[str] | None = None

    def _fallback(self) -> re.Pattern[str]:
        """The compiled alternation, for the rare line the trie cannot
        index safely: casefolding that changes length (`ß` → `ss`, `İ`,
        the ligatures) shifts every position after it, so a span
        computed on the folded text would lie. Building it costs the
        0.2s the trie exists to avoid, which is why it is built on
        first use and only ever for a corpus that actually contains
        such text."""
        if self._slow is None:
            alternation = "|".join(
                re.escape(b) for b in sorted(self._names, key=len, reverse=True)
            )
            self._slow = re.compile(
                r"(?<![\w.-])(" + alternation + r")(?![\w-]|\.[A-Za-z0-9])",
                re.IGNORECASE,
            )
        return self._slow

    def _folded(self) -> re.Pattern[str]:
        """The alternation over FOLDED names, for a corpus whose names are
        not pure ASCII (P5 normalize-on-match).

        `café.md` stored decomposed is cited as `café.md` composed: the raw
        alternation cannot see across that difference, this one can — both
        sides fold before they are compared. Built lazily like `_fallback`,
        and used only when a candidate name actually carries non-ASCII, so
        an ASCII corpus never pays for it and no ASCII answer changes.
        """
        if self._slow_folded is None:
            alternation = "|".join(
                re.escape(_fold(b)) for b in sorted(self._names, key=len, reverse=True)
            )
            self._slow_folded = re.compile(
                r"(?<![\w.-])(" + alternation + r")(?![\w-]|\.[A-Za-z0-9])"
            )
        return self._slow_folded

    def _starts(self, line: str):
        """Candidate match positions, left to right."""
        if self._run_mode:
            for m in _RUN_RE.finditer(line):
                yield m.start()
            return
        first = self._first
        prev = ""
        for i, ch in enumerate(line):
            if (not prev or not _PREV_RE.match(prev)) and ch.lower() in first:
                yield i
            prev = ch

    def _slow_used(self) -> bool:
        """Whether the compiled alternation was ever needed (tests)."""
        return self._slow is not None

    def _folded_hits(self, line: str) -> list[_Hit]:
        """Folded matches, mapped back into `line`'s own coordinates.

        The pattern runs on the folded text, so its spans are folded
        positions; `_fold_spans`' map turns them into positions in the
        ORIGINAL line (which is what the rewrite pass writes against) and
        the text is re-sliced from the original so a hit never carries
        bytes the file does not contain.
        """
        folded, positions = _fold_spans(line)
        out: list[_Hit] = []
        for m in self._folded().finditer(folded):
            lo, hi = m.start(1), m.end(1)
            if lo >= len(positions) or hi - 1 >= len(positions):
                continue
            start, end = positions[lo], positions[hi - 1] + 1
            out.append(_Hit(line[start:end], start, end))
        return out

    def finditer(self, line: str):
        if not (self._ascii and line.isascii()):
            raw = [
                _Hit(m.group(1), m.start(1), m.end(1))
                for m in self._fallback().finditer(line)
            ]
            if self._ascii or line.isascii():
                # Either side being ASCII puts composition out of reach:
                # an ASCII line cannot hold a non-ASCII name, and an
                # ASCII name needs no folding to match. Historical
                # behavior, byte for byte.
                return iter(raw)
            hits = list(raw)
            taken = [(h.lo, h.hi) for h in hits]
            for hit in self._folded_hits(line):
                # An exact/IGNORECASE hit already owns that span; the
                # folded leg adds the OTHER normal form, never a second
                # reading of the same one (finditer never overlaps).
                if any(lo < hit.hi and hit.lo < hi for lo, hi in taken):
                    continue
                hits.append(hit)
                taken.append((hit.lo, hit.hi))
            hits.sort(key=lambda h: h.lo)
            return iter(hits)
        hits: list[_Hit] = []
        end_of_last = 0
        root = self._root
        for start in self._starts(line):
            if start < end_of_last:
                continue  # matches never overlap (finditer's contract)
            node = root
            best = -1
            k = start
            n = len(line)
            while k < n:
                node = node.get(line[k].lower())
                if node is None:
                    break
                k += 1
                if node.get(_TERMINAL) is not None and _BOUND_RE.match(line, k) is None:
                    best = k  # keep descending: a longer candidate may win
            if best > 0:
                hits.append(_Hit(line[start:best], start, best))
                end_of_last = best
        return iter(hits)


def _basename_pattern(basenames: list[str]) -> BasenameMatcher | None:
    """One matcher for every candidate basename, longest first, with the
    same word-ish boundaries as _name_pattern.

    The name is the one every call site (and propose.py) already uses;
    what changed in 0.2.8 is the body — see BasenameMatcher.
    """
    if not basenames:
        return None
    return BasenameMatcher(basenames)


def _line_targets(
    line: str,
    index: CitationIndex,
    base_pat: BasenameMatcher | None,
    citing_dir: str | None = None,
) -> dict[str, str]:
    """Resolved citation targets on one line → the form that cited them
    (path|dir|wikilink|basename|rel_path), first form wins per target. Both
    extractors feed it: path-ish tokens (paths, dirs, wikilinks, link
    targets) and candidate basenames with word-ish boundaries (the
    "see api.md." prose form). citing_dir enables the D4 citing-dir
    read — the scan and the rewrite pass resolve identically."""
    out: dict[str, str] = {}
    for token, token_form in _path_tokens(line, index=index, citing_dir=citing_dir):
        resolved = _resolve(
            token,
            index,
            is_wikilink=token_form == "wikilink",
            citing_dir=citing_dir,
        )
        if resolved is not None and resolved[0] != "ambiguous":
            # A wikilink hit keeps its form; path/dir tokens report the
            # resolution kind (a link target IS a path citation).
            form = "wikilink" if token_form == "wikilink" else resolved[0]
            out.setdefault(resolved[1], form)
    if base_pat is not None:
        for m in base_pat.finditer(line):
            base = index.canonical_basename(m.group(1))
            if base is None:
                continue
            paths = index.by_basename[base]
            if len(paths) == 1:
                out.setdefault(paths[0], "basename")
    return out


# --- the scans ---------------------------------------------------------------


def scan_references(
    root: Path, rel_paths: list[Path], truncated: list[str] | None = None
) -> dict[str, list[str]]:
    """Return {corpus_path_or_dir: [relpaths of files citing it]}.

    Path-keyed: a citation of `docs/api.md` keys the file `docs/api.md`;
    any citation naming a path under a directory keys the DIRECTORY (a
    rename breaks them all); a bare unique basename keys its file.
    Ambiguous basenames are never attributed (Phase 2). Raw scan: every
    text file's mentions count, including files the plan moves.
    Exclusion of moved/rewritten files happens later in reference_risk,
    which knows the plan's full moved set.

    `truncated` collects the paths of files read only as far as the
    512 KiB per-file cap (P5: the one soft condition that used to be
    invisible — see `_scan_bytes`).
    """
    index = CitationIndex(rel_paths)
    if not index.paths:
        return {}
    base_pat = _basename_pattern(sorted(index.by_basename))
    refs: dict[str, list[str]] = {}
    for tp in rel_paths:
        rel_str = tp.as_posix()
        raw = _scan_bytes(root, tp, truncated)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        found: dict[str, None] = {}
        for line in text.splitlines():
            for target in _line_targets(
                line, index, base_pat, citing_dir=_parent(rel_str)
            ):
                if target != rel_str:  # a file doesn't cite itself
                    found.setdefault(target, None)
        for target in found:
            bucket = refs.setdefault(target, [])
            if len(bucket) < _MAX_MENTIONS_PER_TARGET:
                bucket.append(rel_str)
    return {k: v for k, v in refs.items() if v}


def _scan_bytes(
    root: Path, tp: Path, truncated: list[str] | None = None
) -> bytes | None:
    """The bytes to scan a file for citations, or None (not text, gone).

    One place for the read policy every scan shares: the extension table
    decides for the extensions it knows WITHOUT touching the disk, an
    unknown extension pays one bounded sniff (kinds.SNIFF_BYTES) before
    the file is opened for real, and a file that cannot be read is
    skipped the way every scan has always skipped it. `is_text_file` is
    the same decision without the bytes; callers that then read the file
    go through here so the decision costs no extra open for the
    extensions the table already speaks.

    `truncated` (P5) is the loudness channel for the one thing this policy
    used to lose in silence: a file bigger than `_PER_FILE_CAP` is read
    only as far as the cap, so a citation in its tail is invisible to the
    scan (while the rewrite pass, which reads whole files, could still
    change it — the two surfaces disagreeing with no signal anywhere).
    Callers that report pass a list; every over-cap file appends its path.
    """
    known = kinds.kind_of(tp)
    if known == "binary":
        return None
    if known is None:
        try:
            with (root / tp).open("rb") as f:
                head = f.read(kinds.SNIFF_BYTES)
        except OSError:
            return None
        if kinds.classify(tp, head=head) == "binary":
            return None
    try:
        raw = (root / tp).read_bytes()
    except OSError:
        return None
    if truncated is not None and len(raw) > _PER_FILE_CAP:
        truncated.append(tp.as_posix())
    return raw[:_PER_FILE_CAP]


def _walk_rel_paths(root: Path):
    """The corpus as corpus-relative paths, from the ONE traversal.

    The walk contract is scan.py's (skip-list + ignore file), and this
    is its names-only view: the scanner stats each file to decide
    regularity anyway, so no consumer pays for a second `os.walk` just
    to learn the names.
    """
    return scan.iter_rel_paths(root)


def scan_references_stream(root: Path, truncated: list[str] | None = None) -> dict:
    """Un-capped linear scan: {path_or_dir: [citing]} + a "stats" key.

    Phase 3 (reference-rewrites PRD): the 2000-file scan ceiling is
    gone. Resolution needs the complete path set before any token can
    be read (a citer may name a file the walk reaches later), so the
    names come off ONE traversal and the text files are streamed after,
    one read each under the per-file cap. Since 0.2.8 (R6) the names
    come from the single-pass scanner instead of a second `os.walk` per
    pass; nothing refuses at any file count and callers report the loud
    counts (files_scanned, hits) instead of ever skipping silently.
    Same resolution rules and shape as scan_references (which stays for
    callers that already hold rel paths, e.g. inventory) — the two
    agree exactly (test_stream_matches_eager).

    "hits" counts every resolved citation edge found, even past the
    per-target mentions cap (that cap is a display bound, not a scan
    bound). Callers pop "stats" to recover the plain refs mapping.
    `stats.truncated_files` counts the files read only to the 512 KiB cap
    (P5: the audit's "a citation beyond the cap in one file is dropped in
    total silence"), and `truncated` names them for the caller's envelope.
    """
    root = Path(root)
    rel_paths = list(scan.iter_rel_paths(root))
    index = CitationIndex(rel_paths)
    base_pat = _basename_pattern(sorted(index.by_basename))
    refs: dict[str, list[str]] = {}
    files_scanned = 0
    hits = 0
    over_cap: list[str] = [] if truncated is None else truncated
    for tp in rel_paths:
        rel_str = tp.as_posix()
        raw = _scan_bytes(root, tp, over_cap)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        files_scanned += 1
        found: dict[str, None] = {}
        for line in text.splitlines():
            for target in _line_targets(
                line, index, base_pat, citing_dir=_parent(rel_str)
            ):
                if target != rel_str:  # a file doesn't cite itself
                    found.setdefault(target, None)
        for target in found:
            hits += 1
            bucket = refs.setdefault(target, [])
            if len(bucket) < _MAX_MENTIONS_PER_TARGET:
                bucket.append(rel_str)
    out = {k: v for k, v in refs.items() if v}
    out["stats"] = {
        "files_scanned": files_scanned,
        "hits": hits,
        "truncated_files": len(set(over_cap)),
    }
    return out


# --- the citation rewrite pass (Phase 4) --------------------------------------


def _post_move_path(rel: str, chains: dict[str, str]) -> str:
    """A path's post-move location: the chain dst for a moved src, the
    dst-suffix for a descendant of a moved directory, else unchanged.

    chains maps each explicit src to its FINAL location (resolve_chains
    already rewrote file hops through moved ancestors), but a file under
    a moved DIRECTORY is not a chain key — the prefix rule covers it.
    """
    for src, dst in chains.items():
        if rel == src:
            return dst
        prefix = src.rstrip("/") + "/"
        if rel.startswith(prefix):
            return dst.rstrip("/") + "/" + rel[len(prefix) :]
    return rel


def _moved_target(
    chains: dict[str, str], target: str, prefixes: tuple[tuple[str, str], ...] = ()
) -> str | None:
    """The post-move path when `target` (a corpus path or directory)
    moves under the plan's chains or a declared prefix rewrite, else None.

    Each chain hop is a prefix rule — a directory move covers its
    descendants. Validate refuses nested sources (plan_nested_moves), so
    at most one rule can name a prefix of `target`. `prefixes` are the
    plan's explicit `prefix_rewrites` (B8: directories emptied file by
    file have no chain entry, so a convention-string citation of the old
    prefix would dangle) — the agent declares `{"from": "reports/_org/",
    "to": "reports/org/"}` and citations of anything under the old
    prefix follow the rename.
    """
    post = _post_move_path(target, chains)
    if post != target:
        return post
    return _prefix_target(target, prefixes)


def _prefix_target(target: str, prefixes: tuple[tuple[str, str], ...]) -> str | None:
    """The renamed destination when `target` lives under a declared
    prefix rewrite, else None. Chains win: this is only consulted when
    no chain covers the target."""
    for src, dst in prefixes:
        src_n = src.rstrip("/")
        if target == src_n or target.startswith(src_n + "/"):
            return dst.rstrip("/") + target[len(src_n) :]
    return None


def _prefix_rewrite_text(
    text: str, prefixes: tuple[tuple[str, str], ...]
) -> tuple[str, str] | None:
    """(new_text, old_prefix_target) when `text` itself starts with a
    declared prefix — for tokens that resolve to nothing (convention
    strings naming files that do not exist yet). None when no prefix
    applies or the swap would be a no-op."""
    for src, dst in prefixes:
        src_n = src.rstrip("/")
        if text == src_n or text.startswith(src_n + "/"):
            new = dst.rstrip("/") + text[len(src_n) :]
            if new != text:
                return new, src_n
    return None


def _expand_braces(text: str) -> list[str]:
    """Expand the FIRST brace group: `a{1,2}b` -> [`a1b`, `a2b`].

    Nested or unterminated braces are left alone (the text comes back
    unchanged); empty alternatives are dropped, and a group with only
    empty alternatives expands to the text with the group removed —
    `a{}b` names `ab`.
    """
    i = text.find("{")
    if i == -1:
        return [text]
    j = text.find("}", i + 1)
    if j == -1:
        return [text]
    head, body, tail = text[:i], text[i + 1 : j], text[j + 1 :]
    alts = [alt for alt in body.split(",") if alt != ""]
    if not alts:
        return [head + tail]
    return [head + alt + tail for alt in alts]


def _wild_pattern(variant: str) -> str:
    """The fnmatch pattern for one expanded variant: <placeholder> groups
    are wildcards (E2), so a convention string like
    `reports/_org/<date>_*-screens.md` matches the files it names."""
    return re.sub(r"<[^<>]*>", "*", variant)


def _expand_token(
    text: str,
    index: CitationIndex,
    chains: dict[str, str],
    prefixes: tuple[tuple[str, str], ...] = (),
) -> tuple[str, str] | None:
    """Corpus-checked expansion for glob/brace/placeholder tokens (D2/D11).

    The pattern is expanded against the corpus inventory: brace groups
    into alternatives, `<placeholder>` groups into wildcards (E2), then
    each variant fnmatched over the corpus paths. Every match must
    move, and all matches must land in ONE destination directory — then
    the token's directory part is rewritten to it and the glob/brace/
    placeholder basename part survives intact. Outcomes:

    - ("rewrite", new_text): every match moves into one destination.
    - ("skip", reason): the matches split across destinations
      (`glob_split`) or name real files that stay put (`glob_stays`) —
      a rewrite would be wrong.
    - ("convention", None): the pattern names NO corpus file — a
      convention string (E2). The caller applies the declared-prefix /
      moved-dir rename instead: `prefix_rewrites` exists exactly for
      these (the 0.1.9 report's standup-digest regression).
    """
    variants = _expand_braces(text) if "{" in text else [text]
    if "*" in text or "<" in text:
        matches: set[str] = set()
        for v in variants:
            matches.update(fnmatch.filter(index.paths, _wild_pattern(v)))
    else:
        # A brace expansion with no glob: folded lookup, so the variant
        # that names an NFD-named corpus file in the other normal form is
        # still "a corpus file" and not a convention string. The glob
        # branch above stays byte-exact on purpose — `fnmatch` runs over
        # the raw path set, and a folded variant would have to come back
        # through a fold-keyed index to be usable as a move target.
        matches = {index.match_path(v) or v for v in variants if index.match_path(v)}
    if not matches:
        return ("convention", None)
    dests = {}
    for p in matches:
        post = _moved_target(chains, p, prefixes)
        if post is not None and post != p:
            dests[p] = post
    if not dests:
        return ("skip", "glob_stays")
    dest_dirs = {_parent(dst) for dst in dests.values()}
    if len(dest_dirs) != 1 or "/" not in text:
        return ("skip", "glob_split")
    old_dir, base_part = text.rsplit("/", 1)
    if not base_part:
        return ("skip", "glob_split")
    return ("rewrite", next(iter(dest_dirs)) + "/" + base_part)


def _rewrite_token(text: str, kind: str, old: str, new: str) -> str | None:
    """The token's post-move text, or None when it needs no change.

    The token's FORM is preserved (C2: root path in, root path out): an
    exact path becomes the exact new path; a suffix citation
    (`../docs/api.md`) keeps its prefix and swaps the path tail; a
    directory-prefix citation (a convention string) keeps its tail and
    swaps the dir; a bare basename keeps the (presumably unique) name —
    a rename is the only case that changes it. A MULTI-SEGMENT token
    that resolved via its final segment (the token itself names no
    corpus path) is written as the full new path — never degraded to a
    bare basename, which names the file only from the target's own
    folder. (Callers skip the token entirely when it is neither the
    resolved path nor a suffix of it — a stale path or a URL tail is
    reported, never guessed.) Relative links whose CITING file also
    moves need recomputation, not replacement — `_rel_link` at the call
    site; the literal swap is for external files and root-path forms.

    Compared folded, sliced by character count (P5): the token may spell
    the target in the other Unicode normal form, and folding is
    per-character, so the last `len(old)` characters of `text` are
    exactly the characters that produced `old`'s folded tail. The
    ARITHMETIC stays byte-exact; only the EQUALITY tests fold.
    """
    fold_text = _fold(text)
    fold_old = _fold(old)
    if fold_text == fold_old:
        return new
    if fold_text.endswith("/" + fold_old):
        return text[: len(text) - len(old)] + new
    if kind == "dir" and fold_text.startswith(fold_old.rstrip("/") + "/"):
        return new + text[len(old) :]
    if kind == "basename":
        new_base = new.rstrip("/").rsplit("/", 1)[-1]
        if "/" in text:
            # Multi-segment token resolved via its final segment: the
            # same form out — the full new path, not a bare basename.
            return new if _fold(new_base) != fold_text else None
        if _fold(new_base) != fold_text:
            return new_base
        return None
    return None


def _parent(rel: str) -> str:
    """The posix parent dir of a rel path ('.' for a root-level file)."""
    return rel.rsplit("/", 1)[0] if "/" in rel else "."


def _rel_link(new_citing_dir: str, new_target: str, is_dir: bool) -> str:
    """The recomputed link target (Phase 5): posix relpath from the
    citing file's post-move directory to the target's post-move
    location. A file target names the basename relative to its new
    directory; a directory target IS the destination."""
    rel_dir = _parent(new_target) if not is_dir else new_target
    rel = os.path.relpath(rel_dir, new_citing_dir or ".")
    if is_dir:
        return rel
    base = new_target.rsplit("/", 1)[-1]
    return base if rel == "." else rel + "/" + base


def _resolve_from(dir_: str, rel_text: str) -> str:
    """`rel_text` read as relative to `dir_`, normalized (trailing
    slashes collapse; `..` resolves)."""
    return posixpath.normpath(posixpath.join(dir_ or ".", rel_text))


def _reads_as_relative(text: str, citing_pre: str, index: CitationIndex) -> bool:
    """True when `text` names its target relative to the citing file's
    own directory (C2's form test): an explicit `./`/`../` form, a bare
    name, or a path that resolves from there. A TRUNCATED root path
    (`_org/x.md` from `reports/sync/`) does not resolve from the
    citing dir — it takes the root-path swap instead — and a true
    relative citation inside a co-moved subtree keeps working untouched
    (the `assets/card-pepsi.webp` case), so form is decided by where
    the text actually resolves, not by its shape alone."""
    if text.startswith(("./", "../")):
        return True
    if "/" not in text:
        return True
    pre = _resolve_from(_parent(citing_pre), text)
    return (
        index.match_path(pre) is not None
        or index.match_dir(pre.rstrip("/")) is not None
    )


def _rewritable_link_target(text: str) -> bool:
    """Only relative corpus references are recomputed: absolute paths,
    URLs and scheme-y targets are never touched (a recomputation would
    mangle them, and they cannot name a corpus-relative path)."""
    return "://" not in text and not text.startswith("/")


def _link_path_part(text: str, kind: str) -> tuple[str, str]:
    """(path part, suffix) of a link target: markdown targets may carry
    a #fragment or ?query — the path part is what resolves and what a
    recompute replaces. Wikilink targets are already fragment-free (the
    regex splits them out) and keep any suffix as part of the target."""
    if kind == "link":
        cuts = [i for i in (text.find("#"), text.find("?")) if i != -1]
        cut = min(cuts) if cuts else len(text)
        return text[:cut], text[cut:]
    return text, ""


def _link_replacement(
    text: str,
    kind: str,
    start: int,
    index: CitationIndex,
    chains: dict[str, str],
    citing_pre: str,
    citing_post: str,
    prefixes: tuple[tuple[str, str], ...] = (),
) -> tuple[
    tuple[int, int, str, str, str] | str | None,
    tuple[int, int, str, str, str] | None,
]:
    """The recomputed replacement for one link site, plus its citation
    record — or a skip reason string, or None when the site needs no
    rewrite.

    Returns (replacement, cite): `replacement` is a
    (start, end, old, new, target) tuple, the string "ambiguous_basename"
    (the whole line must be skipped), or None; `cite` is a
    (start, end, token, kind, target) tuple recorded whenever the link
    names a MOVING target — even when the recomputed text happens to
    equal the old one (both ends move together: the gate must be able to
    say WHY nothing changed, B10), and None when the target does not
    move.

    Applies when the resolved target moved AND/OR the citing file
    moves; the new target is relpath(new_citing_dir, new_target_dir) +
    "/" + basename. Markdown suffixes (#frag, ?query) and wikilink
    fragments/aliases live outside the replaced span and survive. An
    ambiguous target skips the line (never guessed).
    """
    if kind == "link":
        # The regex target may carry a #fragment or ?query suffix — the
        # path part is what resolves and what gets replaced (the span
        # covers only the path part, so the suffix rides along intact).
        path_part, _suffix = _link_path_part(text, kind)
    else:
        path_part = text
    r = _resolve(
        path_part,
        index,
        is_wikilink=kind == "wikilink",
        citing_dir=_parent(citing_pre),
    )
    if r is None:
        return None, None
    if r[0] == "ambiguous":
        return "ambiguous_basename", None
    rkind, target = r
    new_target = _moved_target(chains, target, prefixes)
    cite = (
        (start, start + len(path_part), path_part, kind, target)
        if new_target is not None
        else None
    )
    citing_moved = citing_post != citing_pre
    if new_target is None and not citing_moved:
        return None, None
    if new_target is None:
        new_target = target
    # Form preservation (C2): the citation's own shape decides the
    # replacement. Relative forms (./ ../, bare names, wikilink stems —
    # anything that resolves against the citing file's directory) are
    # RECOMPUTED — and left untouched entirely when the old text still
    # names the post-move target from the post-move location (both ends
    # moved together: the 0.1.8 report's `./x.md` → `x.md` churn).
    # Root-path forms take the literal root-path swap.
    post_dir = _parent(citing_post)
    if (
        kind == "wikilink"
        or rkind == "rel_path"
        or (citing_moved and _reads_as_relative(path_part, citing_pre, index))
    ):
        # Relative semantics need a recomputation only when the citing
        # file itself moves (the relationship changes); a stay-put
        # citer takes the literal swap, which preserves `../` prefixes
        # exactly (C2: same form in, same form out). Wikilink stems
        # always recompute — the extensionless target is not a path.
        # Reached only when the target and/or the citing file moves.
        if _resolve_from(post_dir, path_part) == new_target:
            return None, cite
        new_text = _rel_link(post_dir, new_target, rkind == "dir")
        if path_part.endswith("/") and not new_text.endswith("/"):
            new_text += "/"
    else:
        new_text = _rewrite_token(path_part, rkind, target, new_target)
        if new_text is None:
            return None, cite
    # The replacement span covers only the path part — a markdown
    # #fragment/?query suffix rides along untouched, so it is NOT
    # appended here (Phase 4's lesson: the span is the contract).
    if new_text == path_part:
        return None, cite
    return (start, start + len(path_part), path_part, new_text, target), cite


def _line_rewrite(
    line: str,
    index: CitationIndex,
    base_pat: BasenameMatcher | None,
    chains: dict[str, str],
    citing_pre: str,
    citing_post: str,
    in_corpus: bool,
    prefixes: tuple[tuple[str, str], ...] = (),
) -> tuple[
    list[tuple[int, int, str, str, str]],
    list[tuple[str, str, int, int]],
    list[tuple[int, int, str, str, str]],
]:
    """Replacement candidates on one line, plus per-token skips and the
    line's citation sites of moving targets.

    Returns (replacements, skips, cites):

    - replacements: (start, end, old, new, target) tuples, ordered
      left-to-right.
    - skips: (reason, token, start, end) — SPANS left untouched, never
      guessed (C5: one ambiguous token no longer silences the rest of
      the line):
      - ambiguous_basename: a mention of a basename shared by several
        corpus files where at least one member moves — the mention
        cannot be attributed to the moving member (it may mean a
        stay-put one). A path-qualified mention inside a token that
        RESOLVES to a concrete path is NOT this: the container either
        moves (the token rewrite already covers the span) or it does
        not (nothing to do) — the 0.1.6 field report's 720 noisy rows
        were full paths to stay-put files read as ambiguous bare
        mentions (B9).
      - prefix_conflict: a basename mention sits inside a longer path
        token that resolves to nothing — the two readings of the same
        text cannot both be satisfied, so that span is never rewritten.
      - unresolved_path: a multi-segment token that resolved via its
        final segment but is neither the resolved path nor a suffix of
        it (C2) — a stale path, a URL tail, a foreign path. Reported,
        never degraded to a bare basename.
    - cites: (start, end, token, kind, target) for every site naming a
      MOVING target — recorded even when no rewrite results, so the
      gate can explain "cited but 0 rewrites" (B10) and the reference
      risks can carry {file, line, text, why_not_rewritten} (B2). An
      ambiguous mention records one cite per moving member.

    Link sites (Markdown links, wikilinks) in corpus files recompute
    their target instead of literal-swapping when the relationship
    changes (Phase 5, form-aware per C2): a relative citation whose
    old text still names the post-move target from the post-move
    location is left alone. External files and root-path tokens take
    the literal swap.
    """
    token_spans = _token_spans(line, index=index, citing_dir=citing_pre)
    resolved: list[tuple[int, int, str, str, str]] = []
    skips: list[tuple[str, str, int, int]] = []
    cites: list[tuple[int, int, str, str, str]] = []
    for text, kind, start, end in token_spans:
        if kind in ("wikilink", "link") and in_corpus:
            if not _rewritable_link_target(text):
                continue
            repl, cite = _link_replacement(
                text, kind, start, index, chains, citing_pre, citing_post, prefixes
            )
            if cite is not None:
                cites.append(cite)
            if repl is None:
                continue
            if isinstance(repl, str):
                skips.append((repl, text, start, end))
                continue
            resolved.append(repl)
            continue
        if _urlish(line, start, text, index=index, citing_dir=citing_pre):
            continue  # a scheme/host token is not a corpus citation (C2);
            # corpus-resolvable dotted paths are (A3) — _urlish checked
            # the index for exactly that.
        # Glob/brace/placeholder tokens get corpus-checked expansion
        # FIRST (D2/D11/E2): a swap is written only when every corpus
        # match moves into ONE destination directory; a split or
        # stay-put pattern is skipped with a row — never rewritten into
        # a pattern matching nothing. A pattern naming NO corpus file
        # is a convention string: the declared-prefix / moved-dir
        # salvage below applies the rename (prefix_rewrites exists for
        # exactly these — the 0.1.9 report's standup-digest regression).
        if ("*" in text or "{" in text or "<" in text) and "/" in text:
            outcome = _expand_token(text, index, chains, prefixes)
            if outcome is not None and outcome[0] != "convention":
                kind, payload = outcome
                salvage_target = _parent(text)
                if kind == "rewrite":
                    resolved.append((start, end, text, payload, salvage_target))
                elif payload == "glob_split" or (
                    _prefix_rewrite_text(text, prefixes) is not None
                    or _post_move_path(text, chains) != text
                ):
                    # glob_split always lands in the hand-edit list;
                    # glob_stays only when a rename was imminent and
                    # would have broken the citation.
                    skips.append((payload, text, start, end))
                    cites.append((start, end, text, "path", salvage_target))
                continue
        r = _resolve(text, index, citing_dir=_parent(citing_pre) if in_corpus else None)
        if r is None:
            # Unresolvable tokens can still carry a declared prefix
            # rename (a convention string naming a file that does not
            # exist yet) — the agent declared it, the render shows it.
            if prefixes:
                pr = _prefix_rewrite_text(text, prefixes)
                if pr is not None:
                    resolved.append((start, end, text, pr[0], pr[1]))
                    continue
            # D11: a token starting with a MOVED DIRECTORY's path
            # follows the rename — braces, shorthand slashes, whatever
            # follows (the 0.1.9 report's missed citations).
            post = _post_move_path(text, chains)
            if post != text:
                salvage_target = _parent(text) if "/" in text else text
                resolved.append((start, end, text, post, salvage_target))
                cites.append((start, end, text, "path", salvage_target))
            continue
        if r[0] == "ambiguous":
            # C5: skip THIS token, not the line. One cite per moving
            # member so the risk rows can say why (B2).
            skips.append(("ambiguous_basename", text, start, end))
            for p in index.basename_paths(text.rsplit("/", 1)[-1]) or []:
                if _moved_target(chains, p, prefixes):
                    cites.append((start, end, text, kind, p))
            continue
        rkind, target = r
        new_target = _moved_target(chains, target, prefixes)
        if new_target is None:
            continue
        cites.append((start, end, text, kind, target))
        if (
            in_corpus
            and (citing_post != citing_pre or rkind == "rel_path")
            and _reads_as_relative(text, citing_pre, index)
        ):
            # A relative citation from a MOVING citing file: if the old
            # text, read from the POST-move location, still names the
            # post-move target, nothing to write (no churn); otherwise
            # recompute the relative link. A stay-put citer takes the
            # literal swap below, which preserves `../` prefixes
            # exactly (C2: same form in, same form out).
            if _fold(_resolve_from(_parent(citing_post), text)) == _fold(new_target):
                continue
            new_text = _rel_link(_parent(citing_post), new_target, rkind == "dir")
            if text.endswith("/") and not new_text.endswith("/"):
                new_text += "/"
            if new_text == text:
                continue
            resolved.append((start, end, text, new_text, target))
            continue
        if (
            rkind == "basename"
            and "/" in text
            and not (
                _fold(text) == _fold(target)
                or _fold(target).endswith("/" + _fold(text))
            )
        ):
            # A multi-segment token resolved via its final segment, but
            # the token is neither the corpus path nor a suffix of it:
            # a stale path, a URL tail, a foreign path (C2). Skip and
            # report — a bare basename would name the file only from
            # the target's own folder.
            skips.append(("unresolved_path", text, start, end))
            continue
        new_text = _rewrite_token(text, rkind, target, new_target)
        if new_text is None or new_text == text:
            continue
        resolved.append((start, end, text, new_text, target))

    token_ranges = {(s, e) for _t, _k, s, e in token_spans}
    if base_pat is not None:
        for m in base_pat.finditer(line):
            base = index.canonical_basename(m.group(1))
            if base is None:
                continue
            paths = index.by_basename[base]
            start, end = m.start(1), m.end(1)
            # A basename directly after a "/" with no path token around
            # it is a URL/foreign-path tail, not a corpus citation (C2):
            # the host was dropped by _token_spans, and the tail must
            # not come back as a basename citation of the moved file.
            if (
                start > 0
                and line[start - 1] == "/"
                and not any(s <= start and end <= e for _t, _k, s, e in token_spans)
            ):
                continue

            # Resolve the CONTAINER first (B9): a mention inside a
            # longer token that resolves to a concrete path is not an
            # ambiguous bare mention — the container either moves (the
            # token rewrite owns the span) or it does not (nothing to
            # rewrite). Only an unresolvable/ambiguous container leaves
            # the basename reading live. Containment is inclusive on
            # both ends (the basename is usually the token's final
            # segment) but excludes exact equality — a span that IS a
            # token is the token candidate's own business.
            def _containing():
                return next(
                    (
                        (x, k)
                        for x, k, s, e in token_spans
                        if s <= start and end <= e and (s, e) != (start, end)
                    ),
                    None,
                )

            found = _containing()
            container, container_kind = found if found is not None else (None, None)
            if container is not None:
                container_path, _ = _link_path_part(container, container_kind)
                cr = _resolve(
                    container_path, index, is_wikilink=container_kind == "wikilink"
                )
                if cr is not None:
                    # Concrete: the token rewrite owns the span.
                    # Ambiguous: the container's own skip covers it.
                    continue
            if len(paths) > 1:
                # A mention of a shared basename cannot be attributed —
                # and when any member moves, staying silent would look
                # like "fixed" while the mention may name a stay-put
                # member. Skip the span loudly, and record a cite per
                # moving member so the risk rows can say why (B2).
                moving = [p for p in paths if _moved_target(chains, p, prefixes)]
                if moving:
                    cites.extend(
                        (start, end, m.group(1), "basename", p) for p in moving
                    )
                    skips.append(("ambiguous_basename", m.group(1), start, end))
                continue
            target = paths[0]
            new_target = _moved_target(chains, target, prefixes)
            if new_target is None:
                continue
            if (start, end) in token_ranges:
                continue  # the token candidate already covers this span
            if container is not None:
                # Inside a longer token that does NOT resolve: the text
                # has two readings (the token's path and the inner
                # basename) — never guessed (prefix_conflict).
                cites.append((start, end, m.group(1), "basename", target))
                skips.append(("prefix_conflict", m.group(1), start, end))
                continue
            cites.append((start, end, m.group(1), "basename", target))
            new_text = _rewrite_token(m.group(1), "basename", target, new_target)
            if new_text is None or new_text == m.group(1):
                continue
            resolved.append((start, end, m.group(1), new_text, target))

    # Nesting guard (the PRD's X / X/child case): when one replacement's
    # new text is contained in another's on the same line — a directory
    # cited alongside a path under it, both rewritten — the journal's
    # per-line count check could not replay the reversal unambiguously.
    # Skip the line loudly instead of writing an un-replayable journal.
    news = [new for _s, _e, _o, new, _t in resolved]
    for i, a in enumerate(news):
        for b in news[i + 1 :]:
            if a != b and (a in b or b in a):
                return [], [("prefix_conflict", "", 0, len(line))], cites
    # Span order (left to right): the writer consumes occurrences of
    # each old string leftmost-first, so a SHORTER old that prefixes a
    # longer one must never be applied while the longer span sits
    # unprocessed to its left ([[docs/api]] before docs/api.md).
    resolved.sort(key=lambda r: r[0])
    return resolved, skips, cites


_MAX_SITES_PER_TARGET = 200  # display bound per moved target, like the mentions cap


def plan_rewrites(
    root: Path,
    index: CitationIndex,
    chains: dict[str, str],
    scan_files: list[Path],
    prefixes: tuple[tuple[str, str], ...] = (),
    record_only: frozenset[str] | set[str] | None = None,
    skip_lines: dict[str, set[int]] | None = None,
    first_spans: dict[str, list[tuple[int, int, int]]] | None = None,
) -> tuple[
    dict[str, list[dict]],
    list[dict],
    dict[str, int],
    dict[str, list[dict]],
    set[str],
    list[dict],
]:
    """Compute the citation rewrites a plan's moves require.

    Phase 4 (reference-rewrites PRD): every chain hop is a prefix rule
    (a directory move covers its descendants). Each citing file and
    citation site gets a line-level token replacement; nothing else on
    the line is touched.

    index: CitationIndex over the PRE-move rel paths — citations name
    old paths. scan_files: paths to scan, at their CURRENT disk
    location (pre-move in a dry run; at execute the caller passes
    post-move paths — the content is identical either way). Relative
    paths are corpus files; ABSOLUTE paths are declared external-root
    files (Phase 5) — they never move, are reported under their
    absolute path, and their citations take the literal suffix swap
    (no link recomputation: a relpath from outside the corpus is
    meaningless). Each file's hits are reported under its POST-move
    path (_post_move_path). Edit files are scanned like any other
    (C3) — EXCEPT the lines their edits own (skip_lines, keyed by
    pre-move scan path): a plan edit runs after the pass and owns its
    line, so the pass never rewrites it (no double application) and
    the edit's find survives to do the work. record_only: pre-move
    paths inside "rewrite": false subtrees — scanned for citation
    SITES so the gate can show what a vendored subtree cites (C6),
    but never rewritten. prefixes: declared prefix_rewrites (B8) —
    see _moved_target.

    Only text files are scanned (the same eligibility the reference
    scans use — one engine, B2): a binary can never be rewritten
    without re-encoding, and listing its decode failure as a skip row
    was pure noise (the 0.1.6 report's 175 `not_utf8` rows for
    PDFs/WebPs). A TEXT file that fails to decode is still recorded,
    in `unreadable`.

    Returns (rewrites, skipped, target_counts, sites, unreadable,
    skip_rewrites):
      rewrites: {post_move_path: [{"line", "old", "new"}]} — one write
        per file; zero-hit files are absent (untouched).
      skipped: [{"file", "line", "reason", "token", "text"}] — spans
        left untouched (ambiguous_basename, prefix_conflict,
        unresolved_path) and text files that cannot be safely
        rewritten (not_utf8, line 0). Several rows can share a line —
        one per skipped token (C5).
      target_counts: {old_target_path: replacements} — which moved
        paths the pass actually fixes (strict-references coverage and
        the render's false-negative guard).
      sites: {old_target_path: [{file, line, text, token, kind,
        status}]} — every citation site of a moving target with what
        the pass will do about it (rewrite | no_change | edit_covered |
        ambiguous_basename | prefix_conflict | unresolved_path |
        rewrite_off). The one engine behind refs, reference_risks and
        the rewrites themselves (B2).
      unreadable: pre-move paths of text files that failed to decode.
      skip_rewrites: [{file, line, token, reason}] — the single bad
        lines (C2): spans that cite a MOVING path yet were skipped, so
        the agent can hand-edit exactly those lines.
    """
    root = Path(root)
    base_pat = _basename_pattern(sorted(index.by_basename))
    rewrites: dict[str, list[dict]] = {}
    skipped: list[dict] = []
    target_counts: dict[str, int] = {}
    sites: dict[str, list[dict]] = {}
    unreadable: set[str] = set()
    skip_rewrites: list[dict] = []
    skip_lines = skip_lines or {}
    for scan_rel in scan_files:
        # Resolved against the corpus: `is_text_file` sniffs the bytes for
        # the extensions the table is silent about (0.2.8), and a bare
        # relative name would be sniffed wherever the process happens to
        # be sitting — silently "binary" for a corpus file the CWD has no
        # copy of. External-root scan files are absolute already (join
        # leaves those alone).
        if not is_text_file(root / str(scan_rel)):
            continue
        report_rel = _post_move_path(str(scan_rel), chains)
        in_corpus = not os.path.isabs(report_rel)
        record = record_only is not None and str(scan_rel) in record_only
        owned = skip_lines.get(str(scan_rel)) or set()
        try:
            text = (root / scan_rel).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            # Not UTF-8: a rewrite would re-encode the whole file (the
            # edit_not_utf8 precedent) — recorded, never corrupted.
            unreadable.add(str(scan_rel))
            if not record:
                skipped.append(
                    {
                        "file": report_rel,
                        "line": 0,
                        "reason": "not_utf8",
                        "token": None,
                        "text": None,
                    }
                )
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            repls, line_skips, raw_cites = _line_rewrite(
                line,
                index,
                base_pat,
                chains,
                str(scan_rel),
                report_rel,
                in_corpus,
                prefixes,
            )
            # One logical citation can be seen twice (the path token and
            # the basename inside it) — keep the outermost span per
            # target, and judge the status by span containment in a
            # replacement (a suffix basename site is rewritten by the
            # token replacement that covers it).
            ordered_cites = sorted(raw_cites, key=lambda c: (c[0], -(c[1] - c[0])))
            line_cites: list[tuple[int, int, str, str, str]] = []
            for c in ordered_cites:
                s, e, _tok, _k, target = c
                if any(
                    t2 == target and s2 <= s and e <= e2 and (s2, e2) != (s, e)
                    for s2, e2, _t2, _k2, t2 in line_cites
                ):
                    continue
                line_cites.append(c)
            if lineno in owned:
                # A plan edit owns this line (C3): the edit runs after
                # the pass and does the work — never rewritten here.
                # D1 accountability: the edit replaces exactly ONE
                # occurrence (the find's first in the file), so a
                # citation on the line is edit_covered only when its
                # span overlaps that occurrence; everything else on the
                # line is edit_owned_uncovered — loud, in the skip list
                # and the risk rows — never silently dropped, never
                # labeled covered while fixed by nothing.
                spans = (first_spans or {}).get(str(scan_rel)) or []
                for start, end, token, kind, target in line_cites:
                    covered = any(
                        lineno == fl and start < fe and fs < end for fl, fs, fe in spans
                    )
                    bucket = sites.setdefault(target, [])
                    if len(bucket) < _MAX_SITES_PER_TARGET:
                        bucket.append(
                            {
                                "file": str(scan_rel),
                                "line": lineno,
                                "start": start,
                                "end": end,
                                "text": line.strip()[:_SNIPPET_CAP],
                                "token": token,
                                "kind": kind,
                                "status": (
                                    "edit_covered"
                                    if covered
                                    else "edit_owned_uncovered"
                                ),
                            }
                        )
                    if not covered:
                        skipped.append(
                            {
                                "file": report_rel,
                                "line": lineno,
                                "reason": "edit_owned_uncovered",
                                "token": token,
                                "text": line.strip()[:_SNIPPET_CAP],
                            }
                        )
                        skip_rewrites.append(
                            {
                                "file": report_rel,
                                "line": lineno,
                                "token": token,
                                "reason": "edit_owned_uncovered",
                            }
                        )
                continue
            repl_spans = [(s, e) for s, e, _o, _n, _t in repls]
            skip_by_span = {(s, e): reason for reason, _tok, s, e in line_skips}
            for start, end, token, kind, target in line_cites:
                bucket = sites.setdefault(target, [])
                if len(bucket) >= _MAX_SITES_PER_TARGET:
                    continue
                if record:
                    # A "rewrite": false subtree's own citation: shown
                    # with full detail, never rewritten (C6).
                    status = "rewrite_off"
                elif (start, end) in skip_by_span:
                    status = skip_by_span[(start, end)]
                elif any(rs <= start and end <= re for rs, re in repl_spans):
                    status = "rewrite"
                else:
                    # Names a moving path but needs no new text: the
                    # post-move form is identical (both ends moved, or
                    # the move kept the basename) — the "cited but 0
                    # rewrites" explanation (B10).
                    status = "no_change"
                bucket.append(
                    {
                        "file": str(scan_rel),
                        "line": lineno,
                        "start": start,
                        "end": end,
                        "text": line.strip()[:_SNIPPET_CAP],
                        "token": token,
                        "kind": kind,
                        "status": status,
                    }
                )
            if not record:
                for reason, token, start, end in line_skips:
                    skipped.append(
                        {
                            "file": report_rel,
                            "line": lineno,
                            "reason": reason,
                            "token": token,
                            "text": line.strip()[:_SNIPPET_CAP],
                        }
                    )
                    # A skipped span that cites a moving target is a
                    # bad line the agent must hand-edit (C2's
                    # skip_rewrites) — except a BARE ambiguous mention
                    # (D3): it names no corpus file, so there is no
                    # edit to write; the row stays in `skipped` only,
                    # where the audit trail keeps everything.
                    if (start, end) in {(c[0], c[1]) for c in line_cites} and (
                        "/" in (token or "") or reason != "ambiguous_basename"
                    ):
                        skip_rewrites.append(
                            {
                                "file": report_rel,
                                "line": lineno,
                                "token": token,
                                "reason": reason,
                            }
                        )
            if record:
                continue  # recorded above; the file is never rewritten
            for start, end, old, new, target in repls:
                rewrites.setdefault(report_rel, []).append(
                    {"line": lineno, "old": old, "new": new}
                )
                target_counts[target] = target_counts.get(target, 0) + 1
    return rewrites, skipped, target_counts, sites, unreadable, skip_rewrites


def citation_lines(
    root: Path,
    name: str,
    rel_paths: list[Path],
    per_file_cap: int = 5,
    truncated: list[str] | None = None,
) -> list[dict]:
    """Exact citation sites for `name`: {file, line, text, kind}.

    `name` resolves through the same index the scan uses — corpus path,
    directory, path suffix, or basename — so `refs <docs/api.md>` and
    `refs <api.md>` both work. An ambiguous basename returns [] (never
    attributed). A DIRECTORY name also matches citations of any path
    under it — the same aggregation reference_risk applies (a rename
    breaks every citation naming a child), so `refs` and the apply-time
    risk report name the same citing files (B2: the two used to
    disagree — refs missed exact-file citations inside the directory).
    Input for authoring plan `edits`: the agent sees the precise line
    and its surrounding text, writes find/replace against it, done.

    `truncated` names the files read only to the per-file cap (P5); a
    zero-hit answer from a corpus with one of those in the list is a
    "the cap may have hidden it" answer, and `refs` says so.
    """
    index = CitationIndex(rel_paths)
    resolved = _resolve(name, index)
    if resolved is None or resolved[0] == "ambiguous":
        return []
    target = resolved[1]
    is_dir = resolved[0] == "dir"
    child_prefix = target.rstrip("/") + "/"

    def _hit(targets: dict[str, str]) -> str | None:
        """The citation form when the line names the target — directly,
        or (for a directory) via any path under it."""
        if target in targets:
            return targets[target]
        if is_dir:
            for t, form in targets.items():
                if t.startswith(child_prefix):
                    return form
        return None

    base_pat = _basename_pattern(sorted(index.by_basename))
    out: list[dict] = []
    for tp in rel_paths:
        if tp.as_posix() == target:
            continue  # a file doesn't cite itself
        raw = _scan_bytes(root, tp, truncated)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        hits = 0
        for i, line in enumerate(text.splitlines(), 1):
            targets = _line_targets(
                line, index, base_pat, citing_dir=_parent(tp.as_posix())
            )
            kind = _hit(targets)
            if kind is not None:
                out.append(
                    {
                        "file": tp.as_posix(),
                        "line": i,
                        "text": line.strip()[:_SNIPPET_CAP],
                        "kind": kind,
                    }
                )
                hits += 1
                if hits >= per_file_cap:
                    break
    return out


def refs_lookup(
    root: Path,
    name: str,
    rel_paths: list[Path],
    per_file_cap: int = 5,
    truncated: list[str] | None = None,
) -> dict:
    """refs' resolution layer (D7), with the honest query state (A5).

    A name containing `/` that resolves only through the basename
    fallback names no corpus path — the fallback would return a
    DIFFERENT file's citations (the 0.1.9 report's
    `reports/_product/2026-09-15.md` → `reports/_sync/2026-09-15.md`
    mixup; the rewrite pass has had the equivalent unresolved_path
    guard since C2). Such a name is reported with
    `nonexistent_path: true` and matched only EXACTLY: lines where the
    string appears as a token. Suffix forms (`_org/x.md`) still
    resolve through the suffix loop and are unaffected, as are bare
    basenames (`refs <api.md>`).

    A5: "nobody cites it", "you typoed" and "ambiguous, never guessed"
    all used to return the same citation_count: 0 with ok: true — an
    agent could not tell "safe to move" from "the query named nothing".
    The CitationIndex already knows; the payload now says:
    `exists` (the name resolves to a corpus path/dir/unambiguous
    basename), `ambiguous` (several files share it — never attributed),
    and `ambiguous_paths` (the candidates). Still ok: true — these are
    valid query results, not errors.

    `truncated` (P5) collects the files the 512 KiB per-file cap cut
    short, so the caller can say "zero citations, and one file was only
    read halfway" instead of letting 0 read as proof.
    """
    index = CitationIndex(rel_paths)
    resolved = _resolve(name, index)
    nonexistent = "/" in name and resolved is not None and resolved[0] == "basename"
    exists = resolved is not None and not nonexistent
    ambiguous = resolved is not None and resolved[0] == "ambiguous"
    ambiguous_paths: list[str] = []
    if ambiguous:
        # by_basename candidates first; the stem form comes from the
        # wikilink-only resolution but is worth naming if present.
        ambiguous_paths = sorted(
            index.basename_paths(resolved[1]) or index.stem_paths(resolved[1]) or []
        )
    if nonexistent:
        # The exact-token comparison folds too: a name the caller spells in
        # the other normal form is the same query, and this branch is
        # already the "match the string, nothing else" one.
        wanted = _fold(name)
        exact: list[dict] = []
        for tp in rel_paths:
            if _fold(tp.as_posix()) == wanted:
                continue
            raw = _scan_bytes(root, tp, truncated)
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            hits = 0
            for i, line in enumerate(text.splitlines(), 1):
                if any(
                    _fold(t) == wanted
                    for t, _k, _s, _e in _token_spans(
                        line, index=index, citing_dir=_parent(tp.as_posix())
                    )
                ):
                    exact.append(
                        {
                            "file": tp.as_posix(),
                            "line": i,
                            "text": line.strip()[:_SNIPPET_CAP],
                            "kind": "path",
                        }
                    )
                    hits += 1
                    if hits >= per_file_cap:
                        break
        return {
            "citations": exact,
            "nonexistent_path": True,
            "exists": False,
            "ambiguous": False,
            "ambiguous_paths": [],
        }
    return {
        "citations": citation_lines(
            root, name, rel_paths, per_file_cap, truncated=truncated
        ),
        "nonexistent_path": False,
        "exists": exists,
        "ambiguous": ambiguous,
        "ambiguous_paths": ambiguous_paths,
    }


def dangling_citations(
    root: Path,
    rel_paths: list[Path],
    per_file_cap: int = 5,
    max_total: int = 0,
    prefix: str | None = None,
    truncated: list[str] | None = None,
) -> list[dict]:
    """Path-ish citation sites that resolve to nothing (Phase 11).

    Wikilinks, Markdown link targets and path tokens naming no corpus
    path, dir, stem or basename: already-broken links and half-updated
    references. Informational — never a rewrite candidate (the pass
    fixes citations of MOVED paths; these point at nothing) and never
    an error. Bare words are never counted: they are prose, not
    citations. URL-scheme tokens (https://…) are skipped — they are
    web addresses, not corpus paths. Same read policy as the
    reference scan: per-file 512 KiB cap, decode-replace. One site per
    citation (the link span owns its interior path tokens).

    D10: the default is UNBOUNDED (correctness over memory — the old
    200-row cap made a clean-move verification impossible); `max_total`
    caps explicitly when the caller wants it. `prefix` filters to
    tokens naming that directory (post-move cleanup checks:
    `--prefix reports/_org` lists every citation still naming the
    removed dir). Slash-words are not citations: a token with no file
    extension in its final segment, no trailing slash, and no corpus
    directory it could name ("engineering/product",
    "Easy/Medium/Hard") is prose and is skipped.
    """
    index = CitationIndex(rel_paths)
    prefix_norm = prefix.rstrip("/") + "/" if prefix else None
    dirs_sorted = sorted(index.dirs, key=len, reverse=True)

    def _is_slash_word(token: str) -> bool:
        if "/" not in token or token.endswith("/"):
            return False
        tail = token.rstrip("/").rsplit("/", 1)[-1]
        if "." in tail:
            return False
        return not any(token == d or token.startswith(d + "/") for d in dirs_sorted)

    out: list[dict] = []
    for tp in rel_paths:
        rel_str = tp.as_posix()
        raw = _scan_bytes(root, tp, truncated)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        hits = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            for token, kind, _s, _e in _token_spans(
                line, index=index, citing_dir=_parent(rel_str)
            ):
                if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*://", token):
                    continue
                if kind == "path" and _is_slash_word(token):
                    continue
                if prefix_norm is not None and not (
                    token == prefix_norm.rstrip("/") or token.startswith(prefix_norm)
                ):
                    continue
                r = _resolve(
                    token,
                    index,
                    is_wikilink=(kind == "wikilink"),
                    citing_dir=_parent(rel_str),
                )
                if r is None or (
                    # D7: a slashed token that only basename-matches a
                    # DIFFERENT path names a nonexistent path — the
                    # rewrite pass reports it unresolved_path; the
                    # dangling scan lists it too (the legit suffix form
                    # is a suffix of its target and stays resolved).
                    "/" in token
                    and r[0] == "basename"
                    and not r[1].endswith("/" + token)
                ):
                    out.append(
                        {
                            "file": rel_str,
                            "line": lineno,
                            "token": token,
                            "text": line.strip()[:_SNIPPET_CAP],
                            "kind": kind,
                        }
                    )
                    hits += 1
                    if hits >= per_file_cap:
                        break
            if hits >= per_file_cap:
                break
        if max_total and len(out) >= max_total:
            break
    return out[:max_total] if max_total else out


def _text_files_under(scan_dir: Path) -> list[Path]:
    """Text files anywhere under `scan_dir`, sorted for stable output.

    The skip-list (.git, .slopkeeper, .venv, node_modules, ...) is
    pruned; dot-directories are CONTENT as of fp4 and are walked — the
    same walk rules as the corpus-side scans — and the same extension
    sniff decides what is readable. Skip-list members are never read
    even when named (a loud per-dir status for that lives in the CLI,
    Phase 6).
    """
    out: list[Path] = []
    if not scan_dir.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(scan_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in HIDDEN_SKIP_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if is_text_file(p):
                out.append(p)
    return out


def external_citation_lines(
    scan_dirs: list[Path],
    name: str,
    per_file_cap: int = 5,
    truncated: list[str] | None = None,
) -> list[dict]:
    """Citation sites for `name` under directories OUTSIDE the corpus.

    Companion to citation_lines for `refs --scan-path <dir>`: skills,
    AGENT_GUIDE-style guides, cron scripts and scratch harnesses outside
    the corpus also cite corpus paths, and a move would break them
    silently — none of them show up in apply's in-tree reference scan.

    Same {file, line, text} shape as citation_lines, plus "external":
    True and an ABSOLUTE file path: the scan dir is arbitrary, so no
    corpus-relative path exists, and the flag lets agents filter the
    merged citation list. A file does not cite itself by its own
    basename. The per-file read cap is the same _PER_FILE_CAP as
    in-tree; there is no directory-size ceiling (Phase 3 — the old
    SCAN_FILE_LIMIT gate died with it). `truncated` names the files the
    cap cut short (P5), absolute paths included.
    """
    search = _name_searcher(name)
    kind = "path" if "/" in name else "basename"
    out: list[dict] = []
    wanted = _fold(name)
    for scan_dir in scan_dirs:
        for tp in _text_files_under(scan_dir):
            if _fold(tp.name) == wanted:
                continue  # a file doesn't cite itself by its own basename
            try:
                whole = tp.read_bytes()
                if truncated is not None and len(whole) > _PER_FILE_CAP:
                    truncated.append(str(tp))
                raw = whole[:_PER_FILE_CAP]
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            hits = 0
            for i, line in enumerate(text.splitlines(), 1):
                if search(line):
                    out.append(
                        {
                            "file": str(tp),
                            "line": i,
                            "text": line.strip()[:_SNIPPET_CAP],
                            "external": True,
                            "kind": kind,
                        }
                    )
                    hits += 1
                    if hits >= per_file_cap:
                        break
    return out


def reference_risk(
    path: str, refs: dict[str, list[str]], moved_relpaths: set[str]
) -> dict:
    """Summarize breakage risk for moving `path` (a file OR a directory).

    Path-keyed: direct citations of `path` plus citations of anything
    under it — renaming a directory breaks every citation that names a
    child. Refs that point FROM files that are themselves being moved
    are low risk (the agent can rewrite them in the same plan); refs
    from stay-put files are the dangerous kind.
    """
    target = path.rstrip("/")
    mentions = list(refs.get(target, []))
    prefix = target + "/"
    for key, citers in refs.items():
        if key.startswith(prefix):
            mentions.extend(citers)
    external = [m for m in mentions if m not in moved_relpaths]
    return {
        "path": target,
        "mentions": len(mentions),
        "external_references": len(external),
        "referenced_by": sorted(set(external))[:20],
    }


# --- the one risk engine (B2) -------------------------------------------------

# Per-site statuses the citation rewrite pass records, and why a site was
# NOT rewritten. `refs`, `reference_risks` and `propose`'s break estimate
# all read from the same engine, so the surfaces can never disagree.
SITE_WHYS = {
    "no_change": "post-move citation is identical — nothing to rewrite",
    "edit_covered": "file is named in plan edits (edits run after the moves)",
    "edit_owned_uncovered": (
        "the line is owned by a plan edit, but the edit's find does not "
        "cover this citation — it survives the edit; fix by hand"
    ),
    "rewrite_off": 'inside a "rewrite": false subtree — its references move with it',
    "not_utf8": "file is not valid UTF-8 — the pass never re-encodes; fix by hand",
    "not_recorded": "no rewrite site recorded — verify with refs, fix by hand",
}


def _site_covered(status: str) -> bool:
    return status in ("rewrite", "no_change", "edit_covered")


def _not_recorded_detail(
    root: Path, citer: str, target: str, index: CitationIndex
) -> dict:
    """Line/token detail for a not_recorded citer (D6).

    The citer names the target somewhere (the scan found it) but the
    pass recorded no site row — rewrites mode off, the per-target
    display cap, or a scan-artifact exclusion. The risk row used to
    lose the line entirely; one targeted read of the citing file
    recovers it.
    """
    base_pat = _basename_pattern(sorted(index.by_basename))
    citer_dir = _parent(citer)
    prefix = target.rstrip("/") + "/"
    try:
        raw = (root / citer).read_bytes()[:_PER_FILE_CAP]
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return {}
    for i, line in enumerate(text.splitlines(), 1):
        targets = _line_targets(line, index, base_pat, citing_dir=citer_dir)
        if target not in targets and not any(t.startswith(prefix) for t in targets):
            continue
        token = next(
            (
                t
                for t, _k, _s, _e in _token_spans(
                    line, index=index, citing_dir=citer_dir
                )
                if (r := _resolve(t, index, citing_dir=citer_dir)) is not None
                and (r[1] == target or r[1].startswith(prefix))
            ),
            None,
        )
        return {"line": i, "token": token}
    return {}


def enriched_risk(
    src: str,
    refs_full: dict[str, list[str]],
    moved: set[str],
    sites: dict[str, list[dict]],
    edit_skip: set[str] | frozenset[str] = frozenset(),
    no_rewrite: set[str] | frozenset[str] = frozenset(),
    unreadable: set[str] | frozenset[str] = frozenset(),
    root: Path | None = None,
    index: CitationIndex | None = None,
) -> dict:
    """reference_risk plus per-citation detail (B2).

    Every citing file gets its exact sites back — {file, line, text,
    token, status, why} — so the gate shows not just WHO cites a moved
    path but whether the pass fixes each citation, and why not when it
    does not. `uncovered_references` counts the citations the plan does
    NOT fix (the ones a human is actually approving risk for, and the
    only rows that make a move a risk);
    `external_references` stays the raw count (the guide's shape).

    Per-citation status comes from the recorded site rows first (C3:
    an edit-target file is scanned too, so its pass-rewritten lines
    read `rewrite`, and only the lines the pass could not take fall
    back to `edit_covered` — the hand edit runs after). A
    "rewrite": false citer is scanned record-only, so its rows carry
    full {line, text, token} detail under status `rewrite_off` (C6).
    """
    risk = reference_risk(src, refs_full, moved)
    target = src.rstrip("/")
    prefix = target + "/"
    site_rows: list[dict] = []
    for t, rows in sites.items():
        if t == target or t.startswith(prefix):
            site_rows.extend(rows)
    # D5: one cite per moving member yields K identical rows for the
    # same span — one row per (file, line, span, token). Disjoint
    # spans of the same token on one line stay distinct rows.
    seen_rows: set[tuple] = set()
    deduped_rows: list[dict] = []
    for row in site_rows:
        key = (
            row["file"],
            row["line"],
            row.get("start"),
            row.get("end"),
            row.get("token"),
        )
        if key in seen_rows:
            continue
        seen_rows.add(key)
        deduped_rows.append(row)
    site_rows = deduped_rows
    by_citer: dict[str, list[dict]] = {}
    for row in site_rows:
        by_citer.setdefault(row["file"], []).append(row)
    citations: list[dict] = []
    uncovered = 0
    for citer in risk["referenced_by"]:
        rows = by_citer.get(citer, [])
        if not rows:
            if citer in edit_skip:
                citations.append(
                    {
                        "file": citer,
                        "status": "edit_covered",
                        "why": SITE_WHYS["edit_covered"],
                    }
                )
                continue
            if citer in no_rewrite:
                citations.append(
                    {
                        "file": citer,
                        "status": "rewrite_off",
                        "why": SITE_WHYS["rewrite_off"],
                    }
                )
                uncovered += 1
                continue
            if citer in unreadable:
                citations.append(
                    {"file": citer, "status": "not_utf8", "why": SITE_WHYS["not_utf8"]}
                )
                uncovered += 1
                continue
            detail = (
                _not_recorded_detail(root, citer, target, index)
                if root is not None and index is not None
                else {}
            )
            citations.append(
                {
                    "file": citer,
                    "status": "not_recorded",
                    "why": SITE_WHYS["not_recorded"],
                    **detail,
                }
            )
            uncovered += 1
            continue
        for row in rows:
            status = row["status"]
            entry = {
                "file": citer,
                "line": row["line"],
                "text": row["text"],
                "token": row.get("token"),
                "status": status,
            }
            if _site_covered(status):
                citations.append(entry)
                continue
            if citer in no_rewrite:
                # Keep the recorded per-site detail under rewrite_off —
                # and count it: the pass never touches this subtree.
                entry["why"] = SITE_WHYS["rewrite_off"]
                uncovered += 1
            elif citer in edit_skip and status != "edit_owned_uncovered":
                # The pass left this span; the plan's edit takes it
                # after the rewrites (C3). D1's explicit
                # edit_owned_uncovered rows keep their status: the edit
                # does NOT take them.
                entry["status"] = "edit_covered"
                entry["why"] = SITE_WHYS["edit_covered"]
            else:
                entry["why"] = SITE_WHYS.get(
                    status,
                    "line left untouched (ambiguous) — fix by hand with edits",
                )
                uncovered += 1
            citations.append(entry)
    risk["citations"] = citations
    risk["uncovered_references"] = uncovered
    return risk


# --- edits and the rewrite pass in one file (C3/C4) ---------------------------


def _apply_line_replacements(text: str, reps: list[dict]) -> str | None:
    """`text` with the pass's per-line replacements applied — the same
    replay the apply writer performs. None when a site is missing (the
    caller falls back to the raw text; pre-write, that is a bug)."""
    lines = text.splitlines(keepends=True)
    per_line: dict[int, list[dict]] = {}
    for rep in reps:
        per_line.setdefault(rep["line"], []).append(rep)
    for lineno in sorted(per_line):
        idx = lineno - 1
        if idx >= len(lines):
            return None
        for rep in per_line[lineno]:
            if rep["old"] not in lines[idx]:
                return None
            lines[idx] = lines[idx].replace(rep["old"], rep["new"], 1)
    return "".join(lines)


def edit_skip_lines(
    root: Path, edits: list[dict], chains: dict[str, str]
) -> dict[str, set[int]]:
    """The line numbers each edit's `find` occupies, keyed by the edit
    file's PRE-move path (C3): those lines are the edit's — the rewrite
    pass never touches them, so the edit's find survives to do the work
    and the pass cannot double-apply. A find that occurs ONLY in
    post-rewrite text (the pass creates it, e.g. via prefix_rewrites)
    owns no line — the pass runs freely and the edit lands after."""
    out: dict[str, set[int]] = {}
    for e in edits:
        f = str(e["file"])
        pre_loc = Path(f) if Path(f).is_absolute() else Path(root) / f
        try:
            text = pre_loc.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # existence/encoding are the validate errors' job
        find = str(e["find"])
        lines = {i for i, line in enumerate(text.splitlines(), 1) if find in line}
        if lines:
            out.setdefault(f, set()).update(lines)
    return out


def edit_first_spans(
    root: Path, edits: list[dict]
) -> dict[str, list[tuple[int, int, int]]]:
    """The first occurrence of each edit's find in its file's raw text:
    {file: [(line, start_col, end_col)]} (D1).

    Execute replaces exactly ONCE per edit (`text.replace(find,
    replace, 1)`), so a citation on an edit-owned line is fixed by the
    edit only when its span overlaps the FIRST occurrence — a second
    mention of the find on the same line, or the find on a later line,
    survives the edit. Keyed by the edit file's PRE-move path (the
    same key edit_skip_lines uses). A find that occurs only in
    post-rewrite text (C4's second form) owns no line and has no span.
    """
    out: dict[str, list[tuple[int, int, int]]] = {}
    for e in edits:
        f = str(e["file"])
        pre_loc = Path(f) if Path(f).is_absolute() else Path(root) / f
        try:
            text = pre_loc.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        idx = text.find(str(e["find"]))
        if idx == -1:
            continue
        line_no = text.count("\n", 0, idx) + 1
        line_start = text.rfind("\n", 0, idx) + 1
        col = idx - line_start
        out.setdefault(f, []).append((line_no, col, col + len(str(e["find"]))))
    return out


def check_edits_after_rewrites(
    root: Path,
    moves: list[dict],
    chains: dict[str, str],
    edits: list[dict],
    prefix_rewrites: list[dict] | tuple = (),
    rel_paths: list[Path] | None = None,
) -> None:
    """Every edit's `find` must occur exactly once in the text the edit
    will run against (C3/C4's one rule). The rewrite pass runs first
    and never touches a line an edit owns (edit_skip_lines), so:

    - a find authored from the citation line as it stands TODAY is
      valid — the pass leaves that line to the edit;
    - a find authored from POST-rewrite text is valid when the pass
      itself produces it (a convention string renamed by
      prefix_rewrites) — the edit then lands on the rewritten line.

    Either way the count is taken on the post-rewrite view with the
    edit's own lines protected, and validate and apply check the SAME
    view (apply re-validates), so the two surfaces can never disagree
    about which text `find` matches. Raises the documented codes:

    - edit_stale: the target does not exist, or the find text does not
      occur in the post-rewrite view.
    - edit_unreadable: the target's post-move path cannot exist because
      the move that would create it has no source (D9 hardening — the
      same error execute's read raises, surfaced at validate).
    - edit_ambiguous: the find text occurs more than once.
    - edit_not_utf8: the target cannot be decoded strict-UTF-8 (the
      pass never re-encodes, and neither do we).

    A file inside a "rewrite": false subtree is checked against its
    raw text — the pass never rewrites it, so raw IS the post-rewrite
    view.
    """
    if not edits:
        return
    root = Path(root)
    if rel_paths is None:
        rel_paths = list(_walk_rel_paths(root))
    no_rewrite: set[str] = set()
    for m in moves or []:
        if isinstance(m, dict) and m.get("rewrite") is False:
            no_rewrite.add(str(m["from"]))
            pre = str(m["from"]).rstrip("/") + "/"
            no_rewrite.update(str(rp) for rp in rel_paths if str(rp).startswith(pre))
    raw_texts: dict[str, str] = {}
    scan: list[str] = []
    for e in edits:
        f = str(e["file"])
        # D9 hardening: resolve the edit's POST-move location through the
        # same mapper execute uses, and refuse — with execute's own read
        # error — when no move can create it. The pre-move path stays
        # the find-count view (moves do not change content).
        final = _post_move_path(f, chains)
        pre_loc = Path(f) if Path(f).is_absolute() else root / f
        if not pre_loc.exists():
            if final != f:
                raise SlopKeeperError(
                    "edit_unreadable",
                    f"cannot read {final}: the move {f} -> {final} has no source",
                    "The edit target's pre-move path does not exist, so the "
                    "plan cannot put the file where the edit applies; fix "
                    "the edit or the move.",
                )
            raise SlopKeeperError(
                "edit_stale",
                f"edit target does not exist: {f}",
                "The corpus changed since the plan was written; re-check the edit.",
            )
        try:
            raw = pre_loc.read_bytes()
        except OSError as err:
            raise SlopKeeperError("edit_unreadable", f"cannot read {f}: {err}")
        try:
            raw_texts[f] = raw.decode("utf-8")
        except UnicodeDecodeError as err:
            raise SlopKeeperError(
                "edit_not_utf8",
                f"{f} is not valid UTF-8: {err}",
                "Convert the file to UTF-8 first, or rewrite the citation by hand.",
            )
        if f not in no_rewrite and f not in scan:
            scan.append(f)  # several edits may share a file — scan once
    owned = edit_skip_lines(root, edits, chains)
    views = dict(raw_texts)
    if scan:
        index = CitationIndex(rel_paths)
        pr = tuple(
            (str(p.get("from", "")).strip(), str(p.get("to", "")).strip())
            for p in (prefix_rewrites or [])
            if isinstance(p, dict)
        )
        pr = tuple((a, b) for a, b in pr if a and b)
        rewrites, _sk, _t, _s, _u, _sr = plan_rewrites(root, index, chains, scan, pr)
        for f in scan:
            reps = rewrites.get(_post_move_path(f, chains))
            if not reps:
                continue
            own = owned.get(f) or set()
            if own:
                # The edit's own lines are its: the pass leaves them
                # untouched (plan_rewrites' skip_lines) — mirror that
                # here so the count is taken on the text the edit
                # actually applies to.
                reps = [r for r in reps if r["line"] not in own]
            if not reps:
                continue
            view = _apply_line_replacements(raw_texts[f], reps)
            if view is not None:
                views[f] = view
    for e in edits:
        f = str(e["file"])
        n = views[f].count(str(e["find"]))
        if n == 0:
            raise SlopKeeperError(
                "edit_stale",
                f"{f} no longer contains the find text",
                "Edits run after the rewrite pass, which leaves each "
                "edit's own line untouched — author `find` from the "
                "citation line as it stands (post-rewrite only when the "
                "pass itself produces that text).",
            )
        if n > 1:
            raise SlopKeeperError(
                "edit_ambiguous",
                f"find text occurs {n} times in {f}",
                "Make the find text unique, or split the edit.",
            )


# --- import citations (P6 Slice A): a citation kind of its own ----------------
#
# `import utils.helpers` names `utils/helpers.py` as surely as a markdown
# link names `docs/api.md`, and moving that file breaks it. It is still a
# DIFFERENT kind of citation for this tool, in two ways that decide the
# whole design:
#
#   * The rewrite pass NEVER rewrites an import. A dotted module path is
#     not a path token (`utils.helpers` has no slash and no extension, so
#     the token engine never sees it), and a language-safe edit of a module
#     path is AST work the core does not do. So the pass's promise — every
#     citation of a moved path gets fixed — does not reach here, and the
#     honest thing is to price the cost BEFORE the human approves rather
#     than discover it after.
#   * An unresolved import is the NORMAL case, never a defect: stdlib and
#     third-party packages name nothing in the corpus. So "resolved" is a
#     fact about the corpus and no surface here ever refuses a tend for it.
#
# Detection is a per-language statement grammar (regex, deliberately not a
# parser: this is a hint surface, and the audit keeps AST work out of the
# stdlib-only core) plus a module-path to tree-path resolver:
#
#   python   `import a.b.c`, `from a.b import c, d`, `from . import x`,
#            `importlib.import_module("a.b")` — a dotted name resolves
#            against the corpus root, the usual package roots (`src`,
#            `lib`), the citing file's own directory and its parent, and
#            finally any corpus directory that could serve as the package
#            root. Leading dots resolve where they mean to: up from the
#            citing file.
#   js/ts    `import x from "./y"`, `require("./y")`, `import("./y")`,
#            `export * from "./y"` — only RELATIVE specifiers are corpus
#            citations (a bare specifier names a package); resolved with
#            the usual extension and `index.*` fallbacks.
#   go       `import "example.com/m/pkg/util"` and the grouped form —
#            resolved to a package DIRECTORY, through the `go.mod` module
#            prefix when one is present.
#   rust     `mod foo;` (a sibling `foo.rs` / `foo/mod.rs`) and
#            `use crate::a::b;` / `use self::x;` / `use a::b;` (the first
#            segment being the crate name, dropped) under the crate root.
#
# Every rule is report-only. Nothing here is wired into `plan_rewrites`:
# the rewrite engine has zero knowledge of imports, and
# tests/test_p6_imports.py pins that an import line never appears in
# `would_rewrite`.


IMPORT_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".ts": "js",
    ".tsx": "js",
    ".go": "go",
    ".rs": "rust",
}

# A statement scan reads a code file under the same per-file cap as every
# other citation surface (`_scan_bytes`), and one file can carry at most
# this many import statements before the rest stop being evidence and start
# being noise (a generated stub is one file, not a dependency graph).
_MAX_IMPORTS_PER_FILE = 256
# Gate evidence rows per move, and importer names per move — display
# bounds, like every other cap in this module; the COUNTS stay exact.
_MAX_IMPORT_EVIDENCE = 12
_MAX_IMPORT_IMPORTERS = 20

_PY_COMMENT_RE = re.compile(r"\s+#.*$")
_PY_FROM_RE = re.compile(
    r"^[ \t]*from[ \t]+(\.+[A-Za-z_]\w*|\.+|[A-Za-z_]\w*(?:\.\w+)*)"
    r"[ \t]*import\b[ \t]*(.*)$"
)
_PY_IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+(.+)$")
_PY_DYN_RE = re.compile(r"""import_module\(\s*['"]([^'"]+)['"]""")
_PY_DOTTED_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_PY_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

_JS_QUOTED_RE = re.compile(
    r"""(?:\bfrom[ \t]+|(?<!\S)import[ \t]+|\brequire[ \t]*\([ \t]*"""
    r"""|import[ \t]*\([ \t]*)['"]([^'"\n]+)['"]"""
)
_LINE_COMMENT_RE = re.compile(r"(?:^|(?<=\s))//.*$")
_JS_EXTS = (
    "",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".vue",
    ".svelte",
    ".css",
)
_JS_INDEXES = (
    "index.ts",
    "index.tsx",
    "index.js",
    "index.jsx",
    "index.mjs",
    "index.cjs",
    "index.vue",
)

_GO_IMPORT_ONE_RE = re.compile(
    r"^[ \t]*(?:[\w.][\w.-]*[ \t]+)?import[ \t]+\"([^\"]+)\""
)
_GO_BLOCK_OPEN_RE = re.compile(r"^[ \t]*import[ \t]*\(")
_GO_BLOCK_ITEM_RE = re.compile(r"^[ \t]*(?:[\w.][\w.-]*[ \t]+)?\"([^\"]+)\"")
_GO_BLOCK_CLOSE_RE = re.compile(r"^[ \t]*\)")
_GO_MODULE_RE = re.compile(r"^module[ \t]+([^\s/][^\s]*)", re.M)

_RS_MOD_RE = re.compile(
    r"^[ \t]*(?:(?:pub|unsafe)(?:\([^)]*\))?[ \t]+)*mod[ \t]+(\w+)[ \t]*;"
)
_RS_USE_RE = re.compile(
    r"^[ \t]*(?:pub[ \t]+)?use[ \t]+(?:::)?([A-Za-z_]\w*(?:::[A-Za-z_0-9]\w*)*)"
)


def _tail_matches(dir_path: str, tail: str) -> bool:
    """True when `dir_path` ends with the multi-segment `tail`."""
    return dir_path == tail or dir_path.endswith("/" + tail)


def _py_name_list(rest: str) -> tuple[str, ...]:
    """The imported names of a from-import (parenthesized or not).

    `as` aliases and `*` are dropped: an alias renames the binding, not
    the module, and a star import names every module the package provides,
    which is no single file.
    """
    rest = rest.strip()
    if rest.startswith("("):
        rest = rest[1:]
    if rest.endswith(")"):
        rest = rest[:-1]
    out: list[str] = []
    for part in rest.split(","):
        name = part.strip().split(" as ")[0].strip()
        if _PY_IDENT_RE.match(name):
            out.append(name)
    return tuple(out)


def _python_sites(lines: list[str]) -> list[tuple[int, str, str, tuple[str, ...]]]:
    """(line, form, spec, names) for every python import statement.

    form is `py_module` (absolute dotted name), `py_relative` (leading
    dots: the spec keeps them and the resolver reads them as up-levels
    from the citing file) or `py_dynamic` (importlib.import_module).
    A from-import carries its names so the resolver can prefer the most
    SPECIFIC file that exists — `from pkg import mod` is `pkg/mod.py`, not
    `pkg/__init__.py`, when both do.
    """
    out: list[tuple[int, str, str, tuple[str, ...]]] = []
    i = 0
    n = len(lines)
    while i < n:
        no = i + 1
        code = _PY_COMMENT_RE.sub("", lines[i])
        m = _PY_FROM_RE.match(code)
        if m:
            mod, rest = m.group(1), m.group(2).strip()
            if rest.startswith("(") and ")" not in rest:
                # `from a import (` opens a continuation run: read on until
                # the closing paren, bounded so a torn file cannot swallow
                # the whole module.
                j = i
                while ")" not in rest and j + 1 < n and j - i < 40:
                    j += 1
                    rest += " " + _PY_COMMENT_RE.sub("", lines[j])
            form = "py_relative" if mod.startswith(".") else "py_module"
            out.append((no, form, mod, _py_name_list(rest)))
        else:
            m = _PY_IMPORT_RE.match(code)
            if m:
                for part in m.group(1).split(","):
                    name = part.strip().split(" as ")[0].strip()
                    if _PY_DOTTED_RE.match(name):
                        out.append((no, "py_module", name, ()))
            for dyn in _PY_DYN_RE.finditer(code):
                if _PY_DOTTED_RE.match(dyn.group(1)):
                    out.append((no, "py_dynamic", dyn.group(1), ()))
        i += 1
    return out


def _js_sites(lines: list[str]) -> list[tuple[int, str, str, tuple[str, ...]]]:
    """(line, form, spec, names) for relative js/ts module specifiers."""
    out: list[tuple[int, str, str, tuple[str, ...]]] = []
    seen: set[tuple[int, str]] = set()
    for i, line in enumerate(lines, 1):
        code = _LINE_COMMENT_RE.sub("", line)
        for m in _JS_QUOTED_RE.finditer(code):
            spec = m.group(1)
            if not spec.startswith(("./", "../", "/")):
                continue  # a bare specifier names a package, not the corpus
            if (i, spec) in seen:
                continue
            seen.add((i, spec))
            out.append((i, "js_relative", spec, ()))
    return out


def _go_sites(lines: list[str]) -> list[tuple[int, str, str, tuple[str, ...]]]:
    """(line, form, spec, names) for go import paths, both statement forms."""
    out: list[tuple[int, str, str, tuple[str, ...]]] = []
    in_block = False
    for i, line in enumerate(lines, 1):
        code = _LINE_COMMENT_RE.sub("", line)
        if in_block:
            if _GO_BLOCK_CLOSE_RE.match(code):
                in_block = False
                continue
            m = _GO_BLOCK_ITEM_RE.match(code)
            if m:
                out.append((i, "go_package", m.group(1), ()))
            continue
        if _GO_BLOCK_OPEN_RE.match(code):
            in_block = True
            continue
        m = _GO_IMPORT_ONE_RE.match(code)
        if m:
            out.append((i, "go_package", m.group(1), ()))
    return out


def _rust_sites(lines: list[str]) -> list[tuple[int, str, str, tuple[str, ...]]]:
    """(line, form, spec, names) for rust `mod` and `use` paths."""
    out: list[tuple[int, str, str, tuple[str, ...]]] = []
    for i, line in enumerate(lines, 1):
        code = _LINE_COMMENT_RE.sub("", line)
        m = _RS_MOD_RE.match(code)
        if m:
            out.append((i, "rs_mod", m.group(1), ()))
            continue
        m = _RS_USE_RE.match(code)
        if m and "::" in m.group(1):
            # A single-segment `use serde;` names a crate, not a module.
            out.append((i, "rs_use", m.group(1), ()))
    return out


_LANG_SCANNERS = {
    "python": _python_sites,
    "js": _js_sites,
    "go": _go_sites,
    "rust": _rust_sites,
}


class ImportResolver:
    """Module-path to tree-path resolution over a CitationIndex.

    Holds the derived tables a scan needs once (corpus directories by
    their last segment, the go module prefix, the rust crate root) and
    memoizes every resolution: a package imports the same module from a
    hundred files, and re-resolving it a hundred times is pure waste. The
    cache is per instance, which is exactly right — the post-move
    re-resolution `import_move_risks` does builds a SECOND resolver over
    the post-move index and must not read through the pre-move answers.
    """

    def __init__(
        self,
        index: CitationIndex,
        go_module: str | None = None,
        max_cache: int = 200_000,
    ):
        self.index = index
        self.go_module = go_module
        self.crate_root = "src" if index.match_dir("src") else ""
        self._by_last: dict[str, list[str]] = {}
        for d in index.dirs:
            self._by_last.setdefault(d.rsplit("/", 1)[-1], []).append(d)
        self._cache: dict[tuple, str | None] = {}
        self._max_cache = max_cache

    def _file(self, base: str, rel: str) -> str | None:
        """A corpus FILE at `base/rel`, or that path as a corpus directory."""
        joined = "/".join(p for p in (base, rel) if p)
        if not joined:
            return None
        hit = self.index.match_path(joined)
        if hit is not None:
            return hit
        return self.index.match_dir(joined)

    def _module(self, base: str, segs: list[str]) -> str | None:
        """The corpus file a dotted module names under `base`.

        The two shapes python actually stores a module in — `a/b.py` and
        the package `a/b/__init__.py` — then the bare directory (an
        import of a namespace package, or `from pkg import subpackage`).
        """
        rel = "/".join(p for p in (base, *segs) if p)
        if not rel:
            return None
        for cand in (f"{rel}.py", f"{rel}/__init__.py", rel):
            hit = self.index.match_path(cand)
            if hit is not None:
                return hit
            hit = self.index.match_dir(cand)
            if hit is not None:
                return hit
        return None

    def _nearest(self, hits: list[str], citing_dir: str) -> str | None:
        """The candidate sharing the longest dir prefix with the citer.

        The last-resort leg of python and go resolution finds every corpus
        directory that could serve as the package root; `a/b` under `x/`
        and under `y/` is genuinely ambiguous, and the one closest to the
        citing file is what a tool-chain would reach for. Ties break by
        path sort, so the answer is deterministic across runs.
        """
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        cur = citing_dir.split("/") if citing_dir else []

        def key(d: str) -> tuple:
            common = 0
            for a, b in zip(cur, d.split("/")):
                if a != b:
                    break
                common += 1
            return (-common, d)

        return sorted(hits, key=key)[0]

    def resolve_python(
        self, form: str, spec: str, names: tuple[str, ...], citing_dir: str
    ) -> str | None:
        """The most SPECIFIC corpus file a python import names.

        `from pkg import mod` prefers `pkg/mod.py` over `pkg/__init__.py`,
        because that is the file the statement actually loads; a name that
        turns out to be a function inside the module falls through to the
        module itself. That ordering is the whole reason the imported
        NAMES are captured at all.
        """
        dotted: list[str] = []
        base = ""
        if form == "py_relative":
            dots = len(spec) - len(spec.lstrip("."))
            tail = spec.lstrip(".")
            up = citing_dir.split("/") if citing_dir else []
            if dots > 1:
                up = up[: len(up) - (dots - 1)]
            base = "/".join(up)
            dotted.extend(f"{tail}.{n}" if tail else n for n in names)
            if tail:
                dotted.append(tail)
            legs: list[str] = [base]
        else:
            dotted.extend(f"{spec}.{n}" for n in names)
            dotted.append(spec)
            legs = ["", "src", "lib"]
            if citing_dir:
                legs.append(citing_dir)
                parent = _parent(citing_dir)
                if parent not in legs:
                    legs.append(parent)
        for cand in dotted:
            segs = cand.split(".")
            for leg in legs:
                hit = self._module(leg, segs)
                if hit is not None:
                    return hit
            if form == "py_relative":
                continue
            # Last resort: any corpus directory that could serve as the
            # package root — a repo whose package lives under myproj/ has
            # no sys.path-adjacent answer above.
            roots = self._by_last.get(segs[0], [])
            hits = sorted({h for h in (self._module(d, segs[1:]) for d in roots) if h})
            if hits:
                near = self._nearest(hits, citing_dir)
                if near is not None:
                    return near
        return None

    def resolve_js(self, spec: str, citing_dir: str) -> str | None:
        """A relative specifier against the citing file's own directory."""
        if spec.startswith("/"):
            rel = spec.lstrip("/")
        else:
            rel = posixpath.normpath("/".join([p for p in (citing_dir, spec) if p]))
        if rel.startswith("..") or rel in (".", ""):
            return None  # resolves outside the corpus root
        for ext in _JS_EXTS:
            hit = self.index.match_path(rel + ext)
            if hit is not None:
                return hit
        for idx in _JS_INDEXES:
            hit = self.index.match_path(f"{rel}/{idx}")
            if hit is not None:
                return hit
        return self.index.match_dir(rel)

    def resolve_go(self, spec: str, citing_dir: str) -> str | None:
        """A module-qualified import path as a package DIRECTORY.

        A go import names a directory, not a file (every `.go` file in the
        package is one import unit), so the target is the dir and moving
        any single file inside it is free while renaming the dir is not.
        """
        s = spec[len("vendor/") :] if spec.startswith("vendor/") else spec
        cands: list[str] = []
        if self.go_module and s.startswith(self.go_module + "/"):
            cands.append(s[len(self.go_module) + 1 :])
        cands.append(s)
        for c in cands:
            if not c:
                continue
            hit = self.index.match_dir(c)
            if hit is not None:
                return hit
        tail = "/".join(s.rsplit("/", 2)[-2:]) if s.count("/") >= 1 else s
        hits = sorted(
            {
                d
                for d in self._by_last.get(tail.rsplit("/", 1)[-1], [])
                if _tail_matches(d, tail)
            }
        )
        if hits:
            return self._nearest(hits, citing_dir)
        return None

    def resolve_rust(self, form: str, spec: str, citing_dir: str) -> str | None:
        if form == "rs_mod":
            for base in (citing_dir, self.crate_root, ""):
                for rel in (f"{spec}.rs", f"{spec}/mod.rs"):
                    hit = self._file(base, rel)
                    if hit is not None:
                        return hit
            return None
        segs = spec.split("::")
        if segs[0] == "crate":
            base, segs = self.crate_root, segs[1:]
        elif segs[0] == "self":
            base, segs = citing_dir, segs[1:]
        elif segs[0] == "super":
            base, segs = _parent(citing_dir), segs[1:]
        else:
            # The first segment is the crate name: a corpus path only makes
            # sense underneath the crate root.
            base, segs = self.crate_root, segs[1:]
        if not segs:
            return None
        rel = "/".join(segs)
        for cand in (f"{rel}.rs", f"{rel}/mod.rs"):
            hit = self._file(base, cand)
            if hit is not None:
                return hit
        return self.index.match_dir("/".join([p for p in (base, rel) if p]))

    def resolve(
        self,
        lang: str,
        form: str,
        spec: str,
        names: tuple[str, ...],
        citing_dir: str,
    ) -> str | None:
        key = (lang, form, spec, names, citing_dir)
        if key in self._cache:
            return self._cache[key]
        if lang == "python":
            hit = self.resolve_python(form, spec, names, citing_dir)
        elif lang == "js":
            hit = self.resolve_js(spec, citing_dir)
        elif lang == "go":
            hit = self.resolve_go(spec, citing_dir)
        elif lang == "rust":
            hit = self.resolve_rust(form, spec, citing_dir)
        else:
            hit = None
        if len(self._cache) < self._max_cache:
            self._cache[key] = hit
        return hit


def _go_module_prefix(root: Path, rel_paths: list[Path]) -> str | None:
    """The module path from the corpus root's go.mod, when there is one.

    One bounded read, and only when the corpus actually holds go files:
    without it an import path cannot become a corpus directory, because a
    go import is module-qualified (`example.com/m/pkg/util` for the
    corpus-relative `pkg/util`).
    """
    if not any(p.suffix.lower() == ".go" for p in rel_paths):
        return None
    try:
        text = (root / "go.mod").read_bytes()[:8192].decode("utf-8", errors="replace")
    except OSError:
        return None
    m = _GO_MODULE_RE.search(text)
    return m.group(1).strip().rstrip("/") if m else None


def import_sites(
    root: Path,
    rel_paths: list[Path],
    truncated: list[str] | None = None,
    index: CitationIndex | None = None,
) -> dict:
    """Every import statement in the corpus's code files, with its target.

    Returns {"sites": [{file, line, text, lang, spec, form, names,
    target}], "stats": {...}}. `target` is the corpus file or directory the
    statement names, or None: an import of `os`, `react` or `serde`
    resolves to nothing in the corpus, and that is normal — it is counted
    (`stats.unresolved`) and never an error. Self-imports are dropped (a
    file does not cite itself, and `from . import sibling` inside a
    package's own `__init__.py` is that case).

    Only the languages this module speaks a grammar for are read
    (`IMPORT_LANG_BY_EXT`), under the shared per-file read cap; the stats
    name how many files were scanned, so a zero is never read as a census.
    """
    root = Path(root)
    if index is None:
        index = CitationIndex(rel_paths)
    resolver = ImportResolver(index, go_module=_go_module_prefix(root, rel_paths))
    sites: list[dict] = []
    files_scanned = 0
    langs: dict[str, int] = {}
    resolved = 0
    over_cap: list[str] = [] if truncated is None else truncated
    for tp in rel_paths:
        lang = IMPORT_LANG_BY_EXT.get(tp.suffix.lower())
        if lang is None:
            continue
        raw = _scan_bytes(root, tp, over_cap)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        files_scanned += 1
        lines = text.splitlines()
        rel_str = tp.as_posix()
        citing_dir = _parent(rel_str)
        for line_no, form, spec, names in _LANG_SCANNERS[lang](lines)[
            :_MAX_IMPORTS_PER_FILE
        ]:
            target = resolver.resolve(lang, form, spec, names, citing_dir)
            if target == rel_str:
                target = None
            sites.append(
                {
                    "file": rel_str,
                    "line": line_no,
                    "text": lines[line_no - 1].strip()[:_SNIPPET_CAP],
                    "lang": lang,
                    "spec": spec,
                    "form": form,
                    "names": list(names),
                    "target": target,
                }
            )
            if target is not None:
                resolved += 1
            langs[lang] = langs.get(lang, 0) + 1
    return {
        "sites": sites,
        "stats": {
            "files_scanned": files_scanned,
            "statements": len(sites),
            "resolved": resolved,
            "unresolved": len(sites) - resolved,
            "langs": dict(sorted(langs.items())),
            "truncated_files": len(set(over_cap)),
        },
    }


def scan_imports(
    root: Path,
    rel_paths: list[Path],
    truncated: list[str] | None = None,
) -> dict:
    """{target: [import rows]} + a "stats" key — the import cross-reference.

    The import sibling of `scan_references`: path-keyed, so an import of
    anything under a moved DIRECTORY keys the directory (a rename breaks
    every import of it), in the same map shape an agent already reads for
    prose citations.
    """
    scanned = import_sites(root, rel_paths, truncated)
    edges: dict[str, list[dict]] = {}
    for row in scanned["sites"]:
        target = row["target"]
        if target is None:
            continue
        bucket = edges.setdefault(target, [])
        if len(bucket) < _MAX_MENTIONS_PER_TARGET:
            bucket.append(row)
    out = {k: v for k, v in edges.items() if v}
    out["stats"] = scanned["stats"]
    return out


def import_citation_lines(
    root: Path,
    name: str,
    rel_paths: list[Path],
    per_file_cap: int = 5,
    truncated: list[str] | None = None,
    sites: list[dict] | None = None,
) -> list[dict]:
    """Import sites naming `name`: {file, line, text, kind: "import"}.

    `name` resolves through the SAME `_resolve` the citation scan uses
    (corpus path, suffix, directory or unambiguous basename), and a
    DIRECTORY name matches imports of anything under it — the same
    aggregation `citation_lines` applies, so `refs` names the same files
    for an import edge as for a prose one. `sites` lets a caller that
    already ran the scan hand its rows over instead of reading the corpus
    a second time.
    """
    index = CitationIndex(rel_paths)
    resolved = _resolve(name, index)
    if resolved is None or resolved[0] == "ambiguous":
        return []
    target = resolved[1]
    is_dir = resolved[0] == "dir"
    child_prefix = target.rstrip("/") + "/"
    if sites is None:
        sites = import_sites(root, rel_paths, truncated)["sites"]
    out: list[dict] = []
    per_file: dict[str, int] = {}
    for row in sites:
        t = row["target"]
        if t is None or row["file"] == target:
            continue
        if not (t == target or (is_dir and t.startswith(child_prefix))):
            continue
        n = per_file.get(row["file"], 0)
        if n >= per_file_cap:
            continue
        per_file[row["file"]] = n + 1
        out.append(
            {
                "file": row["file"],
                "line": row["line"],
                "text": row["text"],
                "kind": "import",
                "import_lang": row["lang"],
                "spec": row["spec"],
            }
        )
    return out


def _move_covering(target: str, srcs: set[str]) -> str | None:
    """The plan move that relocates `target` — its own entry or an ancestor.

    Deepest first: a plan that moves both `pkg/` and `pkg/internal/`
    governs `pkg/internal/x.py` through the more specific entry, which is
    the one the executor orders last and the one the human should see.
    """
    parts = target.split("/")
    for i in range(len(parts), 0, -1):
        cand = "/".join(parts[:i])
        if cand in srcs:
            return cand
    return None


def import_move_risks(
    root: Path,
    plan: dict,
    rel_paths: list[Path] | None = None,
) -> list[dict]:
    """Informational `import_risk` rows for validate and the human gate.

    The question is not "does the path move" but "does the statement still
    RESOLVE after it does": every import whose target a move touches is
    re-resolved against the POST-move index, from the importer's post-move
    directory.

      - still names exactly the moved file — the move costs nothing here
        (a directory renamed around its importers keeps their relative and
        package-internal statements resolving), so the row is dropped;
      - `stale` — resolves to nothing: the code will not import it any
        more, and the rewrite pass will never fix it;
      - `retargeted` — resolves to a DIFFERENT file: the statement still
        runs and now means something else, the worse failure and the one
        nobody notices.

    Never blocks, never rewrites: these are rows for the gate to show and
    the human to weigh. A plan whose moves are malformed is validate's
    problem, not this function's — it reports no rows and moves on.
    """
    moves = [m for m in (plan.get("moves") or []) if isinstance(m, dict)]
    if not moves:
        return []
    try:
        chains = resolve_chains(moves)
    except SlopKeeperError:
        return []
    root = Path(root)
    if rel_paths is None:
        rel_paths = list(scan.iter_rel_paths(root))
    sites = import_sites(root, rel_paths)["sites"]
    if not sites:
        return []
    srcs = set(chains)
    touched = [
        (s, _move_covering(s["target"], srcs))
        for s in sites
        if s["target"] and _move_covering(s["target"], srcs)
    ]
    if not touched:
        return []
    post = CitationIndex(
        [Path(_post_move_path(p.as_posix(), chains)) for p in rel_paths]
    )
    post_resolver = ImportResolver(post, go_module=_go_module_prefix(root, rel_paths))
    rows: dict[str, dict] = {}
    for s, src in touched:
        want = _post_move_path(s["target"], chains)
        importer_post = _post_move_path(s["file"], chains)
        got = post_resolver.resolve(
            s["lang"], s["form"], s["spec"], tuple(s["names"]), _parent(importer_post)
        )
        if got == want:
            continue  # the statement still names exactly the moved file
        status = "stale" if got is None else "retargeted"
        row = rows.setdefault(
            src,
            {
                "from": src,
                "to": chains[src],
                "targets": set(),
                "importers": set(),
                "langs": set(),
                "stale": 0,
                "retargeted": 0,
                "evidence": [],
            },
        )
        row["targets"].add(s["target"])
        row["importers"].add(s["file"])
        row["langs"].add(s["lang"])
        row[status] += 1
        if len(row["evidence"]) < _MAX_IMPORT_EVIDENCE:
            row["evidence"].append(
                {
                    "file": s["file"],
                    "line": s["line"],
                    "text": s["text"],
                    "spec": s["spec"],
                    "lang": s["lang"],
                    "status": status,
                }
            )
    out: list[dict] = []
    for row in rows.values():
        importers = sorted(row["importers"])
        row["sites"] = row["stale"] + row["retargeted"]
        row["importer_count"] = len(importers)
        row["targets"] = sorted(row["targets"])
        row["langs"] = sorted(row["langs"])
        row["importers"] = importers[:_MAX_IMPORT_IMPORTERS]
        out.append(row)
    return sorted(out, key=lambda r: (-r["sites"], r["from"]))
