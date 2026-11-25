# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P6 Slice B — anchors: the two fragility fixes, and detection with follow-through.

The audit found the anchor scan silently giving up in the two most
ordinary ways real code is written:

  * an unterminated quote on a line — the apostrophe in a comment —
    `break` out of the whole call body, so
    `join(base, "sub", "data")  # don't move this`
    un-anchored the very directory the flag exists to protect; and
  * an f-string hole (`f"v{v}"`) was read as literal text, so a call
    building `releases/v1.2/data` matched no directory at all.

Both are fixed by reading the call body per line and treating a `{...}`
run as a wildcard segment that still honours the literal text around it
(`v{v}` matches `v1.2`, and does NOT match `sub`).

The second half of the slice is the point of the whole feature:
`--suggest-code-edits` turns a detected anchor into a proposed plan
`edits` entry for the one case the citation rewrite pass structurally
cannot reach — a path that exists only as a RUN of literals
(`join(base, "sub", "data")` holds no path token, so no citation ever
matches it). What is proposed is an ordinary edit, so the existing
exactly-once validation does the safety work, and the tests here pin both
the proposal and the refusals that keep it honest (an f-string run is
detected and NEVER rewritten, because freezing `f"v{1+1}"` to "v1.2"
would be the worst edit this tool could make).
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper import cli
from slopkeeper.anchors import (
    _call_runs,
    _JOIN_CALL_RE,
    _line_starts,
    _run_dirs,
    anchored_dirs,
    suggest_anchor_edits,
)
from slopkeeper.apply import apply_plan
from slopkeeper.models import HIDDEN_SKIP_DIRS, SlopKeeperError
from slopkeeper.plan import validate_plan
from slopkeeper.references import CitationIndex


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


def rel_of(root: Path) -> list[Path]:
    return sorted(
        (p.relative_to(root) for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.as_posix(),
    )


def keep_all(root: Path) -> list[str]:
    """Every top-level entry, so a one-file move passes coverage."""
    return sorted(p.name for p in root.iterdir() if p.name not in HIDDEN_SKIP_DIRS)


CODE = '''"""builds paths, badly commented."""
import os
from pathlib import Path

BASE = os.path.dirname(__file__)

ONE = os.path.join(BASE, "sub", "data")  # don't move this, it's delicate
TWO = os.path.join(
    BASE,  # it's fine here, don't worry
    "sub",
    "data",
)
THREE = os.path.join(BASE, "releases", f"v{1 + 1}", "data")
FOUR = os.path.join(BASE, "sub", VAR, "data")
FIVE = Path(BASE, "other", "thing")
SIX = os.path.join(BASE, "sub/data")
SEVEN = os.path.join(BASE, "nope", "nada")
'''

CORPUS = {
    "sub/data/keep.txt": "x\n",
    "sub/readme.md": "x\n",
    "releases/v2/data/ok.txt": "x\n",
    "other/thing.txt": "x\n",
    "code/build_paths.py": CODE,
}


class TestAnchorFragilities(TempCase):
    """The two audit fixes, detected on the exact shapes the audit names."""

    def setUp(self):
        super().setUp()
        for rel, text in CORPUS.items():
            write(self.root / rel, text)
        self.rels = rel_of(self.root)
        self.anchors = anchored_dirs(self.root, rel_paths=self.rels)

    def test_apostrophe_in_a_comment_no_longer_un_anchors_the_call(self):
        # Pre-fix, the `'` in "don't" ended the scan of that call body and
        # `sub/data` appeared anchored by nothing.
        evidence = {row["line"] for row in self.anchors.get("sub/data", [])}
        self.assertIn(7, evidence)  # ONE = ...  # don't move this, it's delicate

    def test_apostrophe_in_a_multiline_call_comment_is_survivable(self):
        # The harder case: the apostrophe is on a line INSIDE the call,
        # before the literals that matter. The scan now skips that line
        # and keeps reading, so the multi-line call anchors too.
        evidence = {row["line"] for row in self.anchors.get("sub/data", [])}
        self.assertIn(8, evidence)  # TWO = os.path.join(

    def test_fstring_hole_anchors_the_concrete_directory(self):
        # join(BASE, "releases", f"v{1 + 1}", "data") builds
        # releases/v2/data — which is a corpus directory. Before the fix
        # the segment `v{1 + 1}` matched nothing.
        self.assertIn("releases/v2/data", self.anchors)
        self.assertIn("THREE", self.anchors["releases/v2/data"][0]["text"])

    def test_a_hole_is_not_a_free_pass_for_any_directory(self):
        # The tightening that keeps the wildcard useful rather than noisy:
        # `v{...}` matches v1.2/v2, not an unrelated name. A run of
        # `f"v{1 + 1}", "data"` alone must not anchor `sub/data`.
        text = 'p = os.path.join(BASE, f"v{1 + 1}", "data")\n'
        index = CitationIndex(self.rels)
        runs = _call_runs(text, _JOIN_CALL_RE, _line_starts(text))
        matched = [d for run in runs for d, _n, _hole in _run_dirs(run, index)]
        self.assertNotIn("sub/data", matched)

    def test_a_run_of_only_holes_anchors_nothing(self):
        text = 'p = os.path.join(BASE, f"{a}", f"{b}")\n'
        index = CitationIndex(self.rels)
        runs = _call_runs(text, _JOIN_CALL_RE, _line_starts(text))
        matched = [d for run in runs for d, _n, _hole in _run_dirs(run, index)]
        self.assertEqual(matched, [])

    def test_a_variable_between_literals_still_breaks_the_run(self):
        # FOUR = join(BASE, "sub", VAR, "data") builds sub/<var>/data:
        # `sub` is anchored, `sub/data` is not (the run is broken).
        self.assertIn("sub", self.anchors)
        # FOUR is line 14; the `VAR` argument breaks the run, so the
        # literals either side never concatenate into sub/data.
        self.assertNotIn(14, {row["line"] for row in self.anchors.get("sub/data", [])})

    def test_a_non_path_call_anchors_nothing(self):
        self.assertNotIn("nope", str(self.anchors))

    def test_call_scan_does_not_read_past_its_own_paren(self):
        # The conservative half of the per-line fix: a comment apostrophe
        # no longer ends a scan, but each call still owns only its own
        # literals — nothing from the enclosing scope gets billed to it.
        text = (
            "p = os.path.join(\n"
            '    BASE, "sub", "data",  # it is fine\n'
            ")\n"
            'r = os.path.join(BASE, "other")\n'
        )
        index = CitationIndex(self.rels)
        runs = _call_runs(text, _JOIN_CALL_RE, _line_starts(text))
        self.assertEqual([r.line for r in runs], [1, 4])
        self.assertEqual([lit[0] for lit in runs[0].lits], ["sub", "data"])
        self.assertEqual([lit[0] for lit in runs[1].lits], ["other"])
        # And both anchor what they name.
        self.assertEqual(
            sorted(d for r in runs for d, _n, _h in _run_dirs(r, index)),
            ["other", "sub", "sub/data"],
        )


class TestSuggestedEdits(TempCase):
    """Detection with follow-through: a run of literals becomes an edit."""

    def setUp(self):
        super().setUp()
        for rel, text in CORPUS.items():
            write(self.root / rel, text)
        self.rels = rel_of(self.root)
        self.index = CitationIndex(self.rels)

    def _edits(self, moves):
        return suggest_anchor_edits(
            self.root, {"moves": moves}, rel_paths=self.rels, index=self.index
        )

    def test_a_literal_run_is_replaced_by_the_new_path(self):
        edits, _stats = self._edits([{"from": "sub", "to": "archive/sub"}])
        # The classic case the rewrite pass cannot reach: TWO literals
        # building one path, replaced by one literal holding the new one.
        found = [e for e in edits if e["find"] == '"sub", "data"']
        self.assertTrue(found, f"expected the run edit, got {edits}")
        self.assertEqual(found[0]["replace"], '"archive/sub/data"')
        self.assertEqual(found[0]["file"], "code/build_paths.py")
        self.assertIn("anchor:", found[0]["reason"])
        self.assertIn("--suggest-code-edits", found[0]["reason"])

    def test_a_whole_path_literal_is_replaced_too(self):
        # SIX = join(BASE, "sub/data") IS a path token and the citation
        # pass would fix it; suggesting the same edit here is harmless
        # (the pass owns its line, the edit runs after) and keeps the
        # rule simple: every literal run that names a moving dir.
        edits, _stats = self._edits([{"from": "sub", "to": "archive/sub"}])
        self.assertIn('"sub/data"', [e["find"] for e in edits])

    def test_an_f_string_run_is_never_rewritten(self):
        # THE safety test of this slice. `join(BASE, "releases",
        # f"v{1 + 1}", "data")` anchors releases/v2/data — a directory the
        # concrete disk has and the SOURCE does not name. Writing
        # "releases/v2/data" over those literals would freeze a runtime
        # value into the file: the worst edit this tool could make,
        # because it looks correct and passes every text check. So only
        # the hole-free prefix before it is ever rewritten.
        edits, _stats = self._edits([{"from": "releases", "to": "archive/releases"}])
        for e in edits:
            self.assertNotIn("{", e["find"], "an edit froze an interpolation")
            self.assertNotIn("v2", e["replace"])
        # The literal prefix IS rewritable, and is.
        self.assertIn('"releases"', [e["find"] for e in edits])

    def test_the_anchors_are_still_reported_when_edits_are_refused(self):
        # Refusing to EDIT must never cost the DETECTION: the gate row is
        # what the human acts on when the tool cannot fix it itself.
        anchors = anchored_dirs(self.root, rel_paths=self.rels)
        self.assertIn("releases/v2/data", anchors)

    def test_a_run_matched_only_through_a_hole_is_detected_never_written(self):
        # `join(BASE, f"{n}", "data")` names sub/data at runtime. It is
        # anchored (the flag is real) and it is NOT rewritable: there is no
        # literal text that stands for the segment, so the generator
        # refuses and COUNTS it.
        extra = CODE + 'EIGHT = os.path.join(BASE, f"{n}", "data")\n'
        (self.root / "code/build_paths.py").write_text(extra)
        edits, stats = self._edits([{"from": "sub", "to": "archive/sub"}])
        self.assertEqual(stats["skipped_wildcard"], 2)  # EIGHT + THREE
        for e in edits:
            self.assertNotIn("{", e["find"])

    def test_only_runs_naming_a_moving_directory_are_suggested(self):
        edits, _stats = self._edits([{"from": "other", "to": "misc/other"}])
        # FIVE = Path(BASE, "other", "thing") — only `other` is a corpus
        # directory, so only that literal is the find.
        self.assertEqual([e["find"] for e in edits], ['"other"'])
        self.assertEqual([e["replace"] for e in edits], ['"misc/other"'])

    def test_a_move_that_goes_nowhere_suggests_nothing(self):
        # `releases/v2/data` staying put is not a cost, and an edit whose
        # find equals its replace would be a no-op the gate still has to
        # read.
        edits, _stats = self._edits([{"from": "sub", "to": "sub"}])
        self.assertEqual(edits, [])

    def test_an_ambiguous_find_is_skipped_and_counted(self):
        # The exactly-once rule applies at authoring time too: a find the
        # file holds twice cannot be an edit, and the count says why.
        dup = CODE + 'COPY = os.path.join(BASE, "sub", "data")\n'
        (self.root / "code/build_paths.py").write_text(dup)
        edits, stats = self._edits([{"from": "sub", "to": "archive/sub"}])
        self.assertGreaterEqual(stats["skipped_ambiguous"], 1)
        self.assertNotIn('"sub", "data"', [e["find"] for e in edits])

    def test_stats_account_for_every_run(self):
        _edits, stats = self._edits([{"from": "sub", "to": "archive/sub"}])
        self.assertGreater(stats["runs"], 0)
        self.assertEqual(
            set(stats),
            {
                "runs",
                "suggested",
                "skipped_wildcard",
                "skipped_multiline",
                "skipped_ambiguous",
                "skipped_quote",
                "capped",
            },
        )

    def test_suggested_edits_survive_validation_and_apply(self):
        # The promise: what is generated is an ordinary plan edit, so the
        # existing exactly-once gate is the safety net and apply does the
        # rest. This is the round trip the audit asks for.
        moves = [{"from": "sub", "to": "archive/sub", "reason": "archive"}]
        edits, _stats = self._edits(moves)
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(self.root),
            "strategy": "hybrid",
            "moves": moves,
            "edits": edits,
            "keep": keep_all(self.root),
            "unresolved": [],
            "folders": [],
        }
        validated = validate_plan(
            json.loads(json.dumps(plan)), self.root, require_based_on=False
        )
        self.assertTrue(validated["edits"])
        plan_path = write_plan(
            self.root, moves=moves, edits=edits, keep=plan["keep"], folders=[]
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"], data["errors"])
        text = (self.root / "code/build_paths.py").read_text()
        self.assertIn('"archive/sub/data"', text)
        self.assertNotIn('"sub", "data"', text)
        # And the untouched f-string run is STILL interpolating.
        self.assertIn('f"v{1 + 1}"', text)

    def test_a_confusing_edit_is_refused_by_validation_not_applied(self):
        # The safety net in the other direction: hand an editor a find
        # that is not unique and validate refuses the plan.
        moves = [{"from": "sub", "to": "archive/sub", "reason": "archive"}]
        plan_path = write_plan(
            self.root,
            moves=moves,
            edits=[
                {
                    "file": "code/build_paths.py",
                    "find": '"sub"',
                    "replace": '"archive/sub"',
                    "reason": "ambiguous on purpose",
                }
            ],
            keep=keep_all(self.root),
            folders=[],
        )
        with self.assertRaises(SlopKeeperError) as cm:
            validate_plan(json.loads(plan_path.read_text()), self.root)
        self.assertEqual(cm.exception.code, "edit_ambiguous")


class TestDraftFlag(TempCase):
    """`draft --suggest-code-edits` wires the generator into the CLI."""

    def setUp(self):
        super().setUp()
        for rel, text in CORPUS.items():
            write(self.root / rel, text)
        # A manifest, so draft has something to tend against (it refuses
        # without one), and a new file whose folder is anchored in code.
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".slopkeeper").mkdir(exist_ok=True)
        (self.root / ".slopkeeper/manifest.json").write_text(
            json.dumps(
                {
                    "schema": "slopkeeper/manifest@2",
                    "root": str(self.root),
                    "fingerprint": "fp6:stub",
                    "files": {},
                    "dirs": {},
                    "unresolved": [],
                }
            )
        )
        write(self.root / "sub/data/new-drop.md", "# new\n")
        write(self.root / "sub/data/other-new.md", "# other new\n")

    def test_flag_off_leaves_edits_empty(self):
        out = self.tmp / "draft.json"
        code, env = run_cli(["draft", str(self.root), "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertNotIn("code_edits", env["data"])
        self.assertEqual(json.loads(out.read_text())["edits"], [])

    def test_flag_on_reports_the_count_and_the_refusals(self):
        out = self.tmp / "draft.json"
        code, env = run_cli(
            ["draft", str(self.root), "--out", str(out), "--suggest-code-edits"]
        )
        self.assertEqual(code, 0)
        self.assertIn("code_edits", env["data"])
        stats = env["data"]["code_edit_stats"]
        # The draft files new slop; the anchor runs above live in dirs the
        # draft does not move, so zero suggestions is the honest answer —
        # and the payload says so with its counts rather than silence.
        self.assertIsInstance(stats["runs"], int)
        self.assertEqual(env["data"]["code_edits"], stats["suggested"])
        if env["data"]["code_edits"]:
            self.assertIn("code path literal", env["next"]["hint"])
        else:
            # No suggestions: the hint must not claim any. The counts on
            # the payload are still the whole story.
            self.assertNotIn("code path literal", env["next"]["hint"])

    def test_draft_with_suggested_edits_still_validates(self):
        # Whatever draft proposes, the plan it writes must be validatable:
        # that is the contract between the two halves of the tool.
        out = self.tmp / "draft.json"
        run_cli(["draft", str(self.root), "--out", str(out), "--suggest-code-edits"])
        plan = json.loads(out.read_text())
        plan["based_on"] = {
            "corpus_fingerprint": run_cli(
                ["inventory", str(self.root), "--fast", "--summary"]
            )[1]["data"]["fingerprint"]
        }
        out.write_text(json.dumps(plan))
        code, env = run_cli(["plan", "validate", str(out), str(self.root)])
        self.assertEqual(code, 0, env)
