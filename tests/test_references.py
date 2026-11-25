# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""references.py — citation sites."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from helpers import write
from slopkeeper.references import citation_lines
from slopkeeper.references import (
    CitationIndex,
    _basename_pattern,
    _resolve,
    _token_spans,
    _urlish,
    _text_files_under,
    dangling_citations,
    is_text_file,
    reference_risk,
    refs_lookup,
    scan_references,
    scan_references_stream,
)


class TestTokenizerGlobsAndBraces(unittest.TestCase):
    """Phase 4: glob and brace tails survive into token spans (D2/D11)."""

    def test_token_spans_carry_glob_tail(self):
        spans = _token_spans("see reports/_org/x*.md and more\n")
        tokens = [t for t, _k, _s, _e in spans]
        self.assertIn("reports/_org/x*.md", tokens)

    def test_token_spans_carry_brace_group(self):
        spans = _token_spans("misc/a/convo/a-{1,2,3}.txt plus tail\n")
        tokens = [t for t, _k, _s, _e in spans]
        self.assertIn("misc/a/convo/a-{1,2,3}.txt", tokens)

    def test_tokenizer_regressions(self):
        # Commas stop tokens (prose lists), query strings are not glued,
        # convention strings still truncate at the angle bracket, URL
        # schemes are dropped, and relative prefixes survive.
        cases = {
            "docs/api.md, and prose": ["docs/api.md"],
            "href=docs/api.md?x=1 end": ["docs/api.md"],
            "reports/_org/<date>_<topic>.md conv": ["reports/_org/"],
            "./x.md ../y/z.md": ["../y/z.md"],
            "https://docs.acme.example/a/b?id=3": [],
        }
        for line, expected in cases.items():
            self.assertEqual(
                [t for t, _k, _s, _e in _token_spans(line)],
                expected,
                f"tokenization changed for {line!r}",
            )


class TestCitingDirResolution(unittest.TestCase):
    """D4 unit: _resolve reads a token against the citing dir — explicit
    relative forms first, root-ish forms only after the suffix match."""

    def _index(self):
        return CitationIndex(
            [
                Path("proj/tokens/colors.css"),
                Path("a/tokens/colors.css"),
                Path("b/tokens/colors.css"),
                Path("docs/api.md"),
            ]
        )

    def test_explicit_relative_resolves_from_citing_dir_first(self):
        r = _resolve("./tokens/colors.css", self._index(), citing_dir="proj")
        self.assertEqual(r, ("rel_path", "proj/tokens/colors.css"))

    def test_rootish_form_prefers_exact_root_path(self):
        # docs/api.md IS a root path: the suffix match wins over the
        # citing-dir read (a truncated root path is not reinterpreted).
        r = _resolve("docs/api.md", self._index(), citing_dir="proj")
        self.assertEqual(r, ("path", "docs/api.md"))

    def test_rootish_form_falls_back_to_citing_dir(self):
        # No root-level tokens/colors.css and the suffixes resolve to
        # nothing: the sibling beside the citer wins over the ambiguous
        # basename fallback.
        r = _resolve("tokens/colors.css", self._index(), citing_dir="proj")
        self.assertEqual(r, ("rel_path", "proj/tokens/colors.css"))

    def test_no_candidate_stays_ambiguous(self):
        r = _resolve("tokens/colors.css", self._index(), citing_dir="docs")
        self.assertEqual(r, ("ambiguous", "colors.css"))


class TestBacktickSpans(unittest.TestCase):
    """D6: a backtick-quoted span is ONE token, spaces included."""

    def test_backtick_span_with_spaces_is_one_token(self):
        line = "transcript: `notes/Sample Transcript - Owner's Phone .md`\n"
        spans = _token_spans(line)
        tokens = [(t, k) for t, k, _s, _e in spans]
        self.assertIn(
            ("notes/Sample Transcript - Owner's Phone .md", "backtick"), tokens
        )
        # No partial sub-token may split the quoted span.
        self.assertNotIn("notes/Sample", [t for t, _k in tokens])

    def test_plain_prose_backticks_stay_out(self):
        spans = _token_spans("he said `prose` and `api.md` stays\n")
        tokens = [t for t, _k, _s, _e in spans]
        self.assertNotIn("prose", tokens)
        self.assertIn("api.md", tokens)


class TestTextFilesUnder(unittest.TestCase):
    def test_dot_dirs_are_content_skip_list_is_not(self):
        # fp4 (reference-rewrites PRD Phase 1): an explicit scan dir sees
        # dot-directory content; skip-list members are never read even
        # when named.
        tmp = Path(tempfile.mkdtemp(prefix="sk-refs-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write(tmp / ".claude" / "skills" / "s.md", "# s\n")
        write(tmp / ".git" / "config", "git\n")
        found = [p.relative_to(tmp).as_posix() for p in _text_files_under(tmp)]
        self.assertIn(".claude/skills/s.md", found)
        self.assertNotIn(".git/config", found)


class TestRefs(unittest.TestCase):
    def test_citation_lines_finds_exact_sites(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "target.md").write_text("# Target\n")
            (root / "citer.md").write_text(
                "# Citer\n\nsee target.md for details\ntarget.md again here\n"
            )
            (root / "target.md")  # the file itself must be excluded
            hits = citation_lines(
                root, "target.md", [Path("target.md"), Path("citer.md")]
            )
            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0]["file"], "citer.md")
            self.assertEqual(hits[0]["line"], 3)
            self.assertIn("see target.md", hits[0]["text"])


# --- scan gates: text sniffing + moved-file semantics (pinned) ---


class TestReferenceScan(unittest.TestCase):
    def test_is_text_file_binary_extensions(self):
        self.assertFalse(is_text_file(Path("photo.png")))
        self.assertFalse(is_text_file(Path("bundle.zip")))
        self.assertFalse(is_text_file(Path("movie.mp4")))
        self.assertTrue(is_text_file(Path("notes.md")))
        self.assertTrue(is_text_file(Path("data.json")))

    def test_is_text_file_checks_extension_not_content(self):
        # PIN: the sniff is name-based only — a binary payload behind a
        # text extension is still "text". Pinned so a content-sniffing
        # change shows up here.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "blob.md"
            p.write_bytes(b"\x00\x01\x02not really text")
            self.assertTrue(is_text_file(p))

    def test_short_basename_citation_reported(self):
        # T10: basenames >= 3 chars are candidates; `a.md` (4) was invisible.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.md").write_text("# A\n")
            (root / "watcher.md").write_text("see a.md for details\n")
            paths = [Path("a.md"), Path("watcher.md")]
            refs = scan_references(root, paths)
            self.assertEqual(refs.get("a.md"), ["watcher.md"])
            risk = reference_risk("a.md", refs, set())
            self.assertEqual(risk["external_references"], 1)
            self.assertEqual(risk["referenced_by"], ["watcher.md"])

    def test_longer_name_mention_is_not_a_false_citation(self):
        # T10: `api.md` must NOT match inside `oldapi.md.bak` (no word
        # boundary), but a path-prefixed mention `docs/api.md` must still hit.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "api.md").write_text("# API\n")
            (root / "oldapi.md.bak").write_text("stale\n")
            (root / "notes.md").write_text("backup kept as oldapi.md.bak\n")
            paths = [Path("api.md"), Path("oldapi.md.bak"), Path("notes.md")]
            refs = scan_references(root, paths)
            self.assertNotIn("api.md", refs)

    def test_path_prefixed_citation_still_matches(self):
        # The lookarounds keep path-ish mentions: `docs` before `api.md`
        # is a path separator, not a word char or dot.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "api.md").write_text("# API\n")
            (root / "notes.md").write_text("see docs/api.md for details\n")
            paths = [Path("api.md"), Path("notes.md")]
            refs = scan_references(root, paths)
            self.assertEqual(refs.get("api.md"), ["notes.md"])

    def test_sentence_final_citation_detected(self):
        # T10 refinement: a bare trailing period is a sentence end, not
        # another extension — "see dsa.md." must be detected. Only
        # ".<alnum>" (oldapi.md.bak, api.md.5) closes the name.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "api.md").write_text("# API\n")
            (root / "notes.md").write_text("See api.md. It has details.\n")
            (root / "other.md").write_text("see oldapi.md.bak for the old one\n")
            paths = [Path("api.md"), Path("notes.md"), Path("other.md")]
            refs = scan_references(root, paths)
            self.assertEqual(refs.get("api.md"), ["notes.md"])

    def test_log_citation_found(self):
        # T10: .log/.jsonl are text for scanning purposes.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "notes.md").write_text("# Notes\n")
            (root / "activity.log").write_text("opened notes.md\n")
            (root / "events.jsonl").write_text('{"m": "notes.md"}\n')
            paths = [Path("notes.md"), Path("activity.log"), Path("events.jsonl")]
            refs = scan_references(root, paths)
            self.assertEqual(refs.get("notes.md"), ["activity.log", "events.jsonl"])

    def test_moved_file_exclusion_lives_in_reference_risk(self):
        # scan_references used to take a dead skip= param; it was REMOVED
        # (wiring it would drop moved files from the raw `mentions` count
        # that reference_risk reports — a behavior change). Raw scan output
        # keeps moved files; the exclusion lives in reference_risk's
        # moved_relpaths. Pinned so a change to either side diffs here.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "target.md").write_text("# Target\n")
            (root / "watcher.md").write_text("see target.md now\n")
            (root / "mover.md").write_text("target.md is great\n")
            paths = [Path("target.md"), Path("watcher.md"), Path("mover.md")]
            refs = scan_references(root, paths)
            self.assertEqual(refs.get("target.md"), ["watcher.md", "mover.md"])
            # Where the exclusion lives: external_references drops
            # the mover, leaving only the stay-put citer.
            risk = reference_risk("target.md", refs, {"mover.md"})
            self.assertEqual(risk["external_references"], 1)
            self.assertEqual(risk["referenced_by"], ["watcher.md"])


class TestPathKeyedScan(unittest.TestCase):
    """Phase 2 (reference-rewrites PRD): citations resolve to paths and
    directories, not bare basenames; ambiguous basenames are never
    attributed."""

    def _corpus(self):
        td = tempfile.mkdtemp(prefix="sk-pathrefs-")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        return Path(td)

    def test_dir_rename_cited(self):
        # A directory rename is a risk for citations of ANY path under it:
        # the longest-suffix token resolves to the child file, and
        # reference_risk aggregates descendant keys for a dir move.
        root = self._corpus()
        write(root / "reports/_sync/a.md", "# a\n")
        write(root / "reports/_sync/b.md", "# b\n")
        write(root / "notes.md", "synced via reports/_sync/a.md\n")
        paths = [
            Path("reports/_sync/a.md"),
            Path("reports/_sync/b.md"),
            Path("notes.md"),
        ]
        refs = scan_references(root, paths)
        self.assertEqual(refs.get("reports/_sync/a.md"), ["notes.md"])
        risk = reference_risk("reports/_sync", refs, set())
        self.assertEqual(risk["external_references"], 1)
        self.assertEqual(risk["referenced_by"], ["notes.md"])

    def test_bare_dir_citation(self):
        root = self._corpus()
        write(root / "misc/intake/x.md", "# x\n")
        write(root / "notes.md", "work happens in misc/intake\n")
        paths = [Path("misc/intake/x.md"), Path("notes.md")]
        refs = scan_references(root, paths)
        self.assertEqual(refs.get("misc/intake"), ["notes.md"])
        hits = citation_lines(root, "misc/intake", paths)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "dir")

    def test_convention_string_found_via_prefix(self):
        root = self._corpus()
        write(root / "reports/_org/2026-01-01_topic.md", "# r\n")
        write(root / "README.md", "file goes in reports/_org/<date>_<topic>.md\n")
        paths = [Path("reports/_org/2026-01-01_topic.md"), Path("README.md")]
        refs = scan_references(root, paths)
        self.assertEqual(refs.get("reports/_org"), ["README.md"])

    def test_dir_name_includes_citations_of_paths_under_it(self):
        # B2: `refs <dir>` must name the same citing files the apply-time
        # risk report does — a directory target includes citations of
        # any exact path under it (the two used to disagree, which made
        # the field report's gate untrustworthy).
        root = self._corpus()
        write(root / "reports/_sync/a.md", "# a\n")
        write(root / "jira/corrections.md", "see reports/_sync/a.md for the sync\n")
        write(root / "AGENTS.md", "logs go in reports/_sync/<date>.md\n")
        paths = [
            Path("reports/_sync/a.md"),
            Path("jira/corrections.md"),
            Path("AGENTS.md"),
        ]
        hits = citation_lines(root, "reports/_sync", paths)
        files = {h["file"] for h in hits}
        self.assertEqual(files, {"jira/corrections.md", "AGENTS.md"})

    def test_suffix_after_dotdot(self):
        root = self._corpus()
        write(root / "docs/api.md", "# api\n")
        write(root / "guides/intro.md", "see ../docs/api.md for details\n")
        paths = [Path("docs/api.md"), Path("guides/intro.md")]
        refs = scan_references(root, paths)
        self.assertEqual(refs.get("docs/api.md"), ["guides/intro.md"])

    def test_wikilink_variants(self):
        root = self._corpus()
        write(root / "docs/api.md", "# api\n")
        write(
            root / "notes.md",
            "[[docs/api.md]] and [[docs/api.md|alias]] and [[docs/api.md#sec]]\n",
        )
        paths = [Path("docs/api.md"), Path("notes.md")]
        refs = scan_references(root, paths)
        self.assertEqual(refs.get("docs/api.md"), ["notes.md"])
        hits = citation_lines(root, "docs/api.md", paths)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "wikilink")

    def test_no_bare_word_false_positive(self):
        root = self._corpus()
        write(root / "credentialed-policy-class.md", "# policy\n")
        write(root / "notes.md", "the credentialed approach\n")
        paths = [Path("credentialed-policy-class.md"), Path("notes.md")]
        self.assertEqual(scan_references(root, paths), {})

    def test_duplicate_basename_ambiguous(self):
        # Two notes.md: a bare mention resolves to NEITHER (the old scan
        # attributed it to first-seen — the wrong-file rewrite hazard).
        root = self._corpus()
        write(root / "dir1/notes.md", "# one\n")
        write(root / "dir2/notes.md", "# two\n")
        write(root / "watcher.md", "see notes.md for details\n")
        paths = [
            Path("dir1/notes.md"),
            Path("dir2/notes.md"),
            Path("watcher.md"),
        ]
        refs = scan_references(root, paths)
        self.assertNotIn("dir1/notes.md", refs)
        self.assertNotIn("dir2/notes.md", refs)
        # A path-qualified mention still resolves exactly.
        write(root / "other.md", "see dir2/notes.md for the real one\n")
        refs = scan_references(root, paths + [Path("other.md")])
        self.assertEqual(refs.get("dir2/notes.md"), ["other.md"])

    def test_ambiguous_citation_lines_return_empty(self):
        root = self._corpus()
        write(root / "dir1/notes.md", "# one\n")
        write(root / "dir2/notes.md", "# two\n")
        paths = [Path("dir1/notes.md"), Path("dir2/notes.md")]
        self.assertEqual(citation_lines(root, "notes.md", paths), [])


class TestStreamScan(unittest.TestCase):
    """Phase 3: the un-capped stream scan agrees with the eager one."""

    def _corpus(self, n_files: int) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="sk-stream-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write(tmp / "docs/api.md", "# api\n")
        write(tmp / "docs/guide.md", "See docs/api.md and ../_org/x.md.\n")
        write(tmp / "misc/intake/README.md", "# intake\n")
        write(tmp / "watcher.md", "docs/api.md, misc/intake, api.md, [[docs/api]]\n")
        # Bulk files cite a SIBLING (unique basenames, ≤2 citers per
        # target): with hundreds of files citing ONE target, the
        # 64-mention display cap makes bucket membership order-dependent
        # and the two variants could not agree set-wise. Equivalence is
        # this test's point; volume is TestRefsUncapped's.
        for i in range(n_files):
            sibling = f"f{(i + 1) % n_files:04}.txt"
            write(tmp / f"bulk/d{i % 10:02}/f{i:04}.txt", f"cite {sibling} here\n")
        return tmp

    def test_stream_matches_eager(self):
        # Both variants agree exactly on a 500-file tree; stats is
        # stream-only and counts every text file read and every
        # resolved citation edge (496 bulk + 1 guide + 2 watcher).
        root = self._corpus(496)  # +4 fixture files = 500
        rel_paths = [
            p.relative_to(root)
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".slopkeeper" not in p.parts
        ]
        eager = scan_references(root, rel_paths)
        full = scan_references_stream(root)
        stats = full.pop("stats")
        # Same targets, same citer sets — bucket order is incidental
        # (walk order vs sorted rel paths; reference_risk sorts anyway).
        self.assertEqual(
            {k: sorted(v) for k, v in full.items()},
            {k: sorted(v) for k, v in eager.items()},
        )
        self.assertEqual(
            stats, {"files_scanned": 500, "hits": 499, "truncated_files": 0}
        )

    def test_stream_ignores_skip_list_and_counts_zero(self):
        # The skip-list is invisible to the stream scan like every walk;
        # an empty corpus still returns loud zero stats, never a refusal.
        tmp = Path(tempfile.mkdtemp(prefix="sk-stream-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write(tmp / ".git/config", "[core]\n")
        write(tmp / "node_modules/x.js", "docs/api.md\n")
        full = scan_references_stream(tmp)
        self.assertEqual(set(full), {"stats"})
        self.assertEqual(
            full["stats"], {"files_scanned": 0, "hits": 0, "truncated_files": 0}
        )


class TestHostHeuristicPrefersCorpus(unittest.TestCase):
    """A3 (0.2.6): anything that resolves against the corpus is a PATH,
    whatever its first segment looks like. The pre-fix heuristic dropped
    ANY path token whose dotted first segment ended in an alpha tail — so
    the identical corpus renamed `myfiles` → `my.files` returned
    `would_rewrite: []` with no risk row either, and after --execute the
    citation pointed at nothing (the audit's controlled comparison).
    Dotted dirs (`my.files/`, `data.backup/`, `v1.0/`) are common corpus
    layouts; the basename fallback kept `refs` mostly working, so the
    breakage was invisible EXCEPT in the rewrite pass — the tool's own
    selling point."""

    def _idx(self, *paths):
        return CitationIndex([Path(p) for p in paths])

    def _toks(self, line, **kw):
        return [t for t, _k, _s, _e in _token_spans(line, **kw)]

    def test_dotted_corpus_dir_citation_survives(self):
        idx = self._idx("my.files/report.md", "notes.md")
        self.assertIn(
            "my.files/report.md",
            self._toks("see my.files/report.md now", index=idx),
        )

    def test_control_plain_dir_matches(self):
        idx = self._idx("myfiles/report.md", "notes.md")
        self.assertIn(
            "myfiles/report.md",
            self._toks("see myfiles/report.md now", index=idx),
        )

    def test_versioned_and_backup_dotted_dirs(self):
        idx = self._idx("v1.0/notes.md", "data.backup/x.md")
        self.assertIn("v1.0/notes.md", self._toks("read v1.0/notes.md ok", index=idx))
        self.assertIn(
            "data.backup/x.md", self._toks("read data.backup/x.md ok", index=idx)
        )

    def test_directory_citation_under_dotted_name(self):
        idx = self._idx("my.files/report.md")
        self.assertIn("my.files/", self._toks("browse my.files/ daily", index=idx))

    def test_token_under_a_dotted_dir_names_nothing_yet(self):
        # my.files/ IS a corpus dir: the token continues it, so it is a
        # (dangling) corpus citation, not a hostname — dangling mode
        # lists it rather than dropping it in silence.
        idx = self._idx("my.files/report.md")
        self.assertIn(
            "my.files/gone.md", self._toks("see my.files/gone.md ok", index=idx)
        )

    def test_host_token_still_dropped_even_when_tail_matches_a_file(self):
        # example.com/index.md with a root index.md in the corpus stays
        # a host: containment is EXACT (token, its dir prefixes, the
        # citing-dir read) — never a bare-tail match.
        idx = self._idx("index.md")
        self.assertEqual(
            self._toks("mirror example.com/index.md please", index=idx), []
        )

    def test_scheme_prefixed_token_dropped_even_if_resolvable(self):
        # The `://` rule outranks corpus resolution: a URL that happens
        # to name a real layout is still a URL.
        idx = self._idx("my.files/report.md")
        self.assertEqual(
            self._toks("https://my.files/report.md is not ours", index=idx), []
        )

    def test_email_marker_drops_the_token(self):
        idx = self._idx("my.files/report.md")
        self.assertTrue(_urlish("x", 0, "my.files/note@corp.md", index=idx))

    def test_relative_sibling_read_needs_citing_dir(self):
        idx = self._idx("a/dotted.dir/report.md", "a/x.md")
        self.assertEqual(self._toks("see dotted.dir/report.md ok", index=idx), [])
        self.assertIn(
            "dotted.dir/report.md",
            self._toks("see dotted.dir/report.md ok", index=idx, citing_dir="a"),
        )

    def test_no_index_keeps_the_legacy_conservatism(self):
        # Callers without a corpus view keep the historical drop: safer
        # to miss a rewrite than to mangle a URL.
        self.assertEqual(self._toks("see my.files/report.md now"), [])


class TestDanglingDottedDirs(unittest.TestCase):
    """A3's dangling-mode face: dotted-dir tokens surface as dangling
    instead of vanishing."""

    def test_dangling_reports_dotted_dir_token_not_the_host(self):
        tmp = Path(tempfile.mkdtemp(prefix="sk-dang-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write(tmp / "my.files/report.md", "# r\n")
        write(tmp / "notes.md", "see my.files/gone.md and example.com/x.md now\n")
        rel = [Path("my.files/report.md"), Path("notes.md")]
        toks = {s["token"] for s in dangling_citations(tmp, rel)}
        self.assertIn("my.files/gone.md", toks)
        self.assertNotIn("example.com/x.md", toks)


class TestRefsLookupHonestState(unittest.TestCase):
    """A5 (0.2.6): citation_count 0 was three facts wearing one coat —
    "nobody cites it", "you typoed", "ambiguous, never guessed". The
    CitationIndex already knew; refs_lookup now says so (still ok: true
    — every one of them is a valid query result)."""

    def _corpus(self):
        tmp = Path(tempfile.mkdtemp(prefix="sk-refs-h-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write(tmp / "reports/note.md", "# one\n")
        write(tmp / "tasks/note.md", "# two\n")
        write(tmp / "solo/report.md", "# solo\n")
        write(tmp / "index.md", "see solo/report.md\n")
        rel = [
            Path("reports/note.md"),
            Path("tasks/note.md"),
            Path("solo/report.md"),
            Path("index.md"),
        ]
        return tmp, rel

    def test_typo_reports_not_exists(self):
        root, rel = self._corpus()
        out = refs_lookup(root, "file_that_does_not_exist.md", rel)
        self.assertEqual(out["citations"], [])
        self.assertFalse(out["exists"])
        self.assertFalse(out["ambiguous"])
        self.assertEqual(out["ambiguous_paths"], [])
        self.assertFalse(out["nonexistent_path"])

    def test_ambiguous_basename_reports_candidates(self):
        root, rel = self._corpus()
        out = refs_lookup(root, "note.md", rel)
        self.assertEqual(out["citations"], [])  # still never attributed
        self.assertTrue(out["exists"])
        self.assertTrue(out["ambiguous"])
        self.assertEqual(out["ambiguous_paths"], ["reports/note.md", "tasks/note.md"])

    def test_resolvable_name_is_honest(self):
        root, rel = self._corpus()
        out = refs_lookup(root, "solo/report.md", rel)
        self.assertTrue(out["exists"])
        self.assertFalse(out["ambiguous"])
        self.assertEqual([c["file"] for c in out["citations"]], ["index.md"])

    def test_slashed_name_with_only_a_basename_match_not_exists(self):
        # D7's branch, A5's word: it names no corpus PATH.
        root, rel = self._corpus()
        out = refs_lookup(root, "nope/report.md", rel)
        self.assertTrue(out["nonexistent_path"])
        self.assertFalse(out["exists"])
        self.assertFalse(out["ambiguous"])


# --- the basename matcher (P3 prefilter) ------------------------------------
#
# `_basename_pattern` used to compile every candidate basename into one
# alternation. It is now a trie over casefolded candidates plus a
# run-splitting prefilter, which is a different machine — so the bar is
# not "the suite still passes", it is "the answer is identical". These
# tests hold the old pattern in the file itself and compare spans, on a
# table of the cases that used to be subtle and on a seeded fuzz over
# names that a corpus really has.

_OLD_PATTERN = r"(?<![\w.-])({alt})(?![\w-]|\.[A-Za-z0-9])"


def _old_hits(names: list[str], line: str):
    import re

    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pat = re.compile(_OLD_PATTERN.format(alt=alt), re.IGNORECASE)
    return [(m.group(1), m.start(1), m.end(1)) for m in pat.finditer(line)]


def _new_hits(names: list[str], line: str):
    matcher = _basename_pattern(sorted(names))
    assert matcher is not None
    return [(m.group(1), m.start(1), m.end(1)) for m in matcher.finditer(line)]


class TestBasenameMatcherParity(unittest.TestCase):
    def test_empty_candidate_set_is_no_matcher(self):
        self.assertIsNone(_basename_pattern([]))

    def test_hand_cases(self):
        cases = [
            # (candidates, line)
            (["api.md"], "see api.md"),
            (["api.md"], "See api.md."),  # bare trailing period is allowed
            (["api.md"], "oldapi.md.bak"),  # longer name and .bak: no match
            (["api.md"], "docs/api.md"),  # path mention: match at the tail
            (["api.md"], "api.md.bak"),
            (["api.md"], "api.md_5"),
            (["dsa.md", "dsa.md.md"], "dsa.md.md"),  # longest candidate wins
            (["a.md", "A.MD"], "x a.md A.MD"),  # case-insensitive both ways
            (["note one.md"], "read note one.md now"),  # spaced name (D6)
            (["cpp+.md"], "build cpp+.md"),  # punctuation outside the run class
            (["x.md"], "(x.md)"),
            (["x.md"], "x.md:"),
            (["1.2.3"], "v1.2.3"),  # a digit-led candidate mid-word: no
            (["log."], "the log. file"),
            (["api.md", "api.markdown"], "api.markdown"),
            (["a.md"], "ca.md"),  # lookbehind: preceded by a word char
            (["a.md"], "-a.md"),  # hyphen is name-ish: no match
            (["a.md"], "/a.md"),  # slash is not: match
            (["v1.0.md"], "my.files/v1.0.md"),
            (["x.md"], "x.md) and y.md"),  # two hits, spans preserved
        ]
        for names, line in cases:
            with self.subTest(names=names, line=line):
                self.assertEqual(_new_hits(names, line), _old_hits(names, line))

    def test_fuzz_against_the_old_pattern(self):
        # Seeded, so a failure is reproducible from the test alone.
        import random

        rng = random.Random(20261124)
        bits = [
            "api",
            "API",
            "dossier",
            "note",
            "report",
            "v1.0",
            "my.files",
            "cpp+",
            "a+b",
            "f000",
            "F000",
            "sp ace",
            "Sample Transcript - Owner's Phone .md",
            "old",
            "oldapi",
            "dsa",
            "a",
            "x",
            "md",
            "log",
            "trailing",
            "dots",
            "_org",
            "one[two]",
            "half#tag",
            "spark-1.2.3",
            "data.backup",
        ]
        exts = [".md", ".txt", ".MD", ".py", "", ".md.bak", ".json", ".tar.gz"]
        seps = [" ", "\t", "/", ", ", "(", ")", '"', "`", "[", "]", "|", "", "  ", "->"]
        prose = ["see", "the", "file", "api", "old", "note", "docs", "and", "x", "md"]
        comparisons = 0
        for _ in range(1200):
            names = set()
            for _n in range(rng.randint(1, 8)):
                names.add(rng.choice(bits) + rng.choice(exts))
            names = {n for n in names if len(n) >= 3 and "." in n}  # by_basename's rule
            if not names:
                continue
            line = "".join(
                rng.choice(bits + prose + [rng.choice(sorted(names))])
                + rng.choice(seps)
                for _ in range(rng.randint(1, 8))
            )
            comparisons += 1
            self.assertEqual(
                _new_hits(sorted(names), line),
                _old_hits(sorted(names), line),
                f"names={sorted(names)} line={line!r}",
            )
        self.assertGreater(comparisons, 800)

    def test_non_ascii_falls_back_to_the_compiled_alternation(self):
        # The trie keys are .lower(), which is `re.IGNORECASE` exactly
        # only while both sides are ASCII. Outside that the matcher must
        # hand the line to the real pattern rather than guess — and
        # `strasse.md` vs `straße.md` is the case that proves it:
        # casefold() would fold ß to ss and match, and the old pattern
        # never did.
        ascii_matcher = _basename_pattern(["api.md", "strasse.md"])
        self.assertEqual(
            [m.group(1) for m in ascii_matcher.finditer("see strasse.md")],
            ["strasse.md"],
        )
        self.assertFalse(ascii_matcher._slow_used())  # served by the trie
        unicode_matcher = _basename_pattern(["straße.md"])
        self.assertEqual(
            [m.group(1) for m in unicode_matcher.finditer("straße.md")], ["straße.md"]
        )
        self.assertEqual(list(unicode_matcher.finditer("strasse.md")), [])
        self.assertTrue(unicode_matcher._slow_used())  # the fallback answered


class TestRewriteCoversSniffedTypes(unittest.TestCase):
    """The rewrite pass must read the CORPUS, not the process CWD.

    `is_text_file` gained the NUL sniff in 0.2.8, which made every call
    site's path argument load-bearing: the pass used to ask about a bare
    relative name (harmless while the answer was a set lookup) and now
    opens that name wherever the process happens to sit — so a corpus
    `app/main.dart` sniffed as "binary" (i.e. skipped, its citation left
    broken) whenever the CWD had no such file, and a `Makefile` got
    classified from the REPO's Makefile. Same answer from any CWD is the
    contract this pins.
    """

    def setUp(self):
        import shutil
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="sk-rw-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "corpus"
        self.decoy = self.tmp / "elsewhere"
        self.decoy.mkdir()

    def _w(self, base: Path, rel: str, data):
        fp = base / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            fp.write_bytes(data)
        else:
            fp.write_text(data, encoding="utf-8")
        return fp

    def test_unlisted_citations_rewritten_identically_from_any_cwd(self):
        import os

        from slopkeeper.references import CitationIndex, plan_rewrites
        from slopkeeper.scan import corpus_rel_paths

        self._w(self.root, "docs/api.md", "# api\n")
        for rel in ("app/main.dart", "infra/main.tf", "noext", ".dotfile"):
            self._w(self.root, rel, "see docs/api.md for the contract\n")
        # Decoys: the SAME names in the CWD, full of NUL bytes. A CWD-side
        # sniff would call every corpus file binary and silently skip it.
        for rel in ("app/main.dart", "infra/main.tf", "noext", ".dotfile"):
            self._w(self.decoy, rel, b"\x00\x01binary imposter\n")

        rels = corpus_rel_paths(self.root)
        index = CitationIndex(rels)
        chains = {"docs": "reference/docs"}

        def run():
            return plan_rewrites(self.root, index, chains, rels)

        here = run()
        cwd = os.getcwd()
        try:
            os.chdir(self.decoy)
            there = run()
        finally:
            os.chdir(cwd)
        self.assertEqual(here, there)
        rewrites = here[0]
        for rel in ("app/main.dart", "infra/main.tf", "noext", ".dotfile"):
            self.assertIn(rel, rewrites, f"{rel} holds a citation and was skipped")
            sites = rewrites[rel]
            self.assertTrue(
                any("reference/docs/api.md" in str(row) for row in sites),
                f"{rel}: no rewrite toward the new path: {sites}",
            )


class TestSniffScannability(unittest.TestCase):
    """R7: `is_text_file` now agrees with inventory's `kind: text`."""

    def setUp(self):
        import shutil
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="sk-sniff-"))
        addCleanup = self.addCleanup
        addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, rel: str, text: str) -> Path:
        fp = self.tmp / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")
        return fp

    def test_unlisted_text_extension_is_scanned(self):
        # inventory called these `kind: text`; the citation scan refused
        # to open them, so a move left the citation broken in silence.
        self._write("docs/api.md", "# api\n")
        self._write("app/main.dart", "// see docs/api.md for the contract\n")
        self._write("infra/main.tf", "// module at docs/api.md\n")
        refs = scan_references_stream(self.tmp)
        self.assertIn("docs/api.md", refs)
        self.assertIn("app/main.dart", refs["docs/api.md"])
        self.assertIn("infra/main.tf", refs["docs/api.md"])

    def test_unlisted_binary_extension_is_still_skipped(self):
        self._write("docs/api.md", "# api\n")
        (self.tmp / "blob.dat").write_bytes(b"\x00\x01 see docs/api.md\n")
        refs = scan_references_stream(self.tmp)
        self.assertNotIn("blob.dat", refs.get("docs/api.md", []))
        self.assertFalse(is_text_file(self.tmp / "blob.dat"))

    def test_text_file_predicate_matches_inventory_kind(self):
        from slopkeeper.inventory import _kind

        fp = self._write("x/weird.qqq", "hello docs/api.md\n")
        self.assertEqual(_kind(fp), "text")
        self.assertTrue(is_text_file(fp))
        self.assertTrue(self._write("x/other.qqq", "abc\x00def\n") is not None)


if __name__ == "__main__":
    unittest.main()
