# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""plan.py — render_markdown: the human-gate document (golden files later)."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from helpers import TempCase, build_corpus, write, write_plan
from slopkeeper import cli
from slopkeeper.anchors import anchored_move_risks
from slopkeeper.plan import (
    load_plan,
    render_markdown,
    straggler_report,
    validate_plan,
)


class TestPlanRender(TempCase):
    def test_render_markdown_groups_and_flags(self):
        plan = load_plan(write_plan(self.root))
        out = validate_plan(plan, self.root)
        md = render_markdown(
            out,
            {
                "acme_dossier.md": {
                    "external_references": 1,
                    "referenced_by": ["WATCHLIST.md"],
                }
            },
        )
        self.assertIn("research/dossiers/", md)
        self.assertIn("⚠️ 1", md)
        self.assertIn("## Folders to scaffold", md)

    def test_edits_table_names_resolved_path(self):
        # D8: the render's edits table names the same post-move path the
        # dry run's would_edit rows name, with the authored source noted
        # when it differs.
        write(self.root / "misc/nvidia/reply.md", "# reply\n")
        plan = load_plan(
            write_plan(
                self.root,
                moves=[
                    {
                        "from": "misc/nvidia",
                        "to": "projects/nvidia",
                        "reason": "restructure",
                    }
                ],
                edits=[
                    {
                        "file": "misc/nvidia/reply.md",
                        "find": "reply",
                        "replace": "RESPONSE",
                        "reason": "dir descendant",
                    }
                ],
                keep=[
                    "docs",
                    "lead-gen",
                    "spark-glm-5.3",
                    "acme_dossier.md",
                    "globex_dossier.md",
                    "handoff-tp2.md",
                    "ledger_wb.html",
                    "WATCHLIST.md",
                ],
                unresolved=[],
                folders=[],
            )
        )
        out = validate_plan(plan, self.root)
        md = render_markdown(out)
        self.assertIn("`projects/nvidia/reply.md`", md)
        self.assertIn("(authored: `misc/nvidia/reply.md`)", md)

    def test_partial_scope_rendered(self):
        # Delta plans declare their scope in the render, so the human
        # gate is skimmable without opening the JSON.
        plan = load_plan(write_plan(self.root))
        plan["partial"] = True
        md = render_markdown(plan)
        self.assertIn("**Scope: PARTIAL**", md)
        plain = render_markdown({**plan, "partial": False})
        self.assertNotIn("**Scope: PARTIAL**", plain)

    def test_reference_risk_keyed_by_nested_source(self):
        # cmd_plan_render keys the reference report by move SOURCE PATH;
        # a basename-keyed lookup silently showed "—" for every nested
        # source (notes/x.md) — the gate hid the exact risk it exists to
        # surface. A src-keyed report must warn on a nested move.
        plan = load_plan(
            write_plan(
                self.root,
                moves=[
                    {
                        "from": "docs/canonical.md",
                        "to": "canon/canonical.md",
                        "reason": "t",
                    }
                ],
            )
        )
        out = validate_plan(plan, self.root, allow_partial=True)
        md = render_markdown(
            out,
            {
                "docs/canonical.md": {
                    "external_references": 1,
                    "referenced_by": ["WATCHLIST.md"],
                }
            },
        )
        self.assertIn("⚠️ 1", md)

    def test_footer_names_the_corpus_root(self):
        # T6: the render's closing line is a next.command — it must carry
        # the required <root> positional (apply exits 2 without it),
        # taken from the plan's own corpus_root.
        out = validate_plan(load_plan(write_plan(self.root)), self.root)
        md = render_markdown(out)
        self.assertIn(
            f"slopkeeper apply <plan.json> {out['corpus_root']} --execute", md
        )


class TestRenderRewriteSummary(TempCase):
    """Phase 4: the gate shows what the rewrite pass will do — real numbers."""

    def test_render_shows_rewrite_summary(self):
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "auto",
            "per_file": [{"file": "WATCHLIST.md", "count": 2}],
            "count": 2,
            "skipped": [{"file": "w.md", "line": 3, "reason": "prefix_conflict"}],
            "targets": {"acme_dossier.md": 1, "globex_dossier.md": 1},
        }
        md = render_markdown(plan, {}, summary)
        self.assertIn("Citation rewrites (automatic): 2 across 1 file(s)", md)
        self.assertIn("`WATCHLIST.md` | 2", md)
        self.assertIn("prefix_conflict", md)
        # Every cited move the pass covers gets no false-negative note.
        self.assertNotIn("Cited but 0 rewrites", md)

    def test_render_flags_cited_but_unrewritten_moves(self):
        # False-negative guard, human-visible: a move the report calls
        # cited but the pass fixes zero citations for is named.
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "auto",
            "per_file": [],
            "count": 0,
            "skipped": [],
            "targets": {},
        }
        refs = {
            "handoff-tp2.md": {
                "external_references": 2,
                "referenced_by": ["a.md", "b.md"],
            }
        }
        md = render_markdown(plan, refs, summary)
        self.assertIn("Cited but 0 rewrites", md)
        self.assertIn("`handoff-tp2.md`", md)

    def test_render_shows_rewrite_sites_line_by_line(self):
        # B1: the gate approves real edits — the rewrite section lists
        # each site as file:line with old → new.
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "auto",
            "per_file": [
                {
                    "file": "WATCHLIST.md",
                    "count": 1,
                    "lines": [
                        {
                            "line": 3,
                            "old": "acme_dossier.md",
                            "new": "research/dossiers/acme_dossier.md",
                        }
                    ],
                }
            ],
            "count": 1,
            "skipped": [],
            "targets": {"acme_dossier.md": 1},
        }
        md = render_markdown(plan, {}, summary)
        self.assertIn("| site | old | new |", md)
        self.assertIn("`WATCHLIST.md`:3", md)
        self.assertIn("`acme_dossier.md` | `research/dossiers/acme_dossier.md`", md)

    def test_render_skipped_rows_carry_token_and_text(self):
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "auto",
            "per_file": [],
            "count": 0,
            "skipped": [
                {
                    "file": "w.md",
                    "line": 4,
                    "reason": "ambiguous_basename",
                    "token": "notes.md",
                    "text": "see notes.md here",
                }
            ],
            "targets": {},
        }
        md = render_markdown(plan, {}, summary)
        self.assertIn("| file | line | reason | token | text |", md)
        self.assertIn("notes.md", md)
        self.assertIn("see notes.md here", md)

    def test_render_cited_but_unrewritten_explains_each_line(self):
        # B10: the false-negative guard names the citing line and why
        # the pass leaves it — no more unexplained silence.
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "auto",
            "per_file": [],
            "count": 0,
            "skipped": [],
            "targets": {},
        }
        refs = {
            "handoff-tp2.md": {
                "external_references": 2,
                "referenced_by": ["a.md", "b.md"],
                "uncovered_references": 2,
                "citations": [
                    {
                        "file": "a.md",
                        "line": 2,
                        "text": "see handoff-tp2.md",
                        "token": "handoff-tp2.md",
                        "status": "not_recorded",
                        "why": "no rewrite site recorded — verify with refs",
                    }
                ],
            }
        }
        md = render_markdown(plan, refs, summary)
        self.assertIn("Cited but 0 rewrites", md)
        self.assertIn("`a.md:2`", md)
        self.assertIn("no rewrite site recorded", md)

    def test_render_marks_rewrites_off(self):
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        summary = {
            "mode": "off",
            "per_file": [],
            "count": 0,
            "skipped": [],
            "targets": {},
        }
        md = render_markdown(plan, None, summary)
        self.assertIn("Citation rewrites: OFF", md)

    def test_render_marks_scan_paths_and_their_rewrite_rows(self):
        # Phase 6: declared scan paths are named at the gate (writes the
        # corpus walk never sees), and their rewrite rows carry the
        # external marker like external edits do.
        plan = load_plan(
            write_plan(
                self.root,
                scan_paths=[str(self.root / ".cache")],
                partial=True,
            )
        )
        summary = {
            "mode": "auto",
            "per_file": [{"file": str(self.root / ".cache" / "notes.md"), "count": 2}],
            "count": 2,
            "skipped": [],
            "targets": {},
        }
        md = render_markdown(plan, {}, summary)
        self.assertIn("Scan paths (ignored dirs the rewrite pass may touch)", md)
        self.assertIn(f"`{self.root / '.cache'}`", md)
        self.assertIn(f"`{self.root / '.cache' / 'notes.md'}` ⚠️ external", md)

    def test_render_omits_scan_paths_line_when_absent(self):
        plan = validate_plan(load_plan(write_plan(self.root)), self.root)
        md = render_markdown(plan, None, None)
        self.assertNotIn("Scan paths", md)


class TestRenderBasedOnFallback(TempCase):
    def test_fallback_writes_canonical_fingerprint_key(self):
        # T6 (parent note): the render-time based_on fallback for plans
        # without one must write the canonical `corpus_fingerprint` —
        # `inventory_sha256` is the pre-T1 alias, read-only for plans in
        # the wild; new output never uses that spelling (T1 wording).
        plan_path = write_plan(self.root)
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        del raw["based_on"]
        plan_path.write_text(json.dumps(raw), encoding="utf-8")

        captured: dict = {}

        def fake_render(
            plan,
            reference_report=None,
            rewrite_summary=None,
            stragglers=None,
            code_anchor_risks=None,
            emptied_dirs=None,
        ):
            captured["plan"] = plan
            return "# gate\n"

        out_path = self.tmp / "gate.md"
        with mock.patch("slopkeeper.cli.render_markdown", fake_render):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.main(
                    [
                        "plan",
                        "render",
                        str(plan_path),
                        str(self.root),
                        "--out",
                        str(out_path),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertIn("corpus_fingerprint", captured["plan"]["based_on"])
        self.assertNotIn("inventory_sha256", captured["plan"]["based_on"])


class TestRenderRulesAndStragglers(TempCase):
    """Phase 8: the gate lists the concrete expansion and the stragglers."""

    def test_render_lists_expansion_and_stragglers(self):
        build_corpus(
            self.root,
            {
                "reports/_org/a.md": "a\n",
                "reports/_org/b.txt": "b\n",
                "reports/_org/sub/c.md": "c\n",
            },
        )
        plan = load_plan(
            write_plan(
                self.root,
                rules=[{"match": "reports/_org/*.md", "to": "reports/org"}],
            )
        )
        out = validate_plan(plan, self.root)
        stragglers = straggler_report(self.root, out)
        md = render_markdown(out, {}, None, stragglers=stragglers)
        # the rule is summarized; the concrete moves are listed
        self.assertIn(
            "- Rule: `reports/_org/*.md` → `reports/org/` — expanded to 1 move(s)",
            md,
        )
        self.assertIn("`reports/_org/a.md`", md)
        before_stragglers = md.split("Stragglers")[0]
        self.assertNotIn("`reports/_org/b.txt`", before_stragglers)
        # the straggler table: folder, count, examples
        self.assertIn("Stragglers", md)
        self.assertIn("`reports/_org/` | 2", md)
        self.assertIn("`reports/_org/b.txt`", md)
        self.assertIn("`reports/_org/sub/c.md`", md)

    def test_render_caps_move_tables(self):
        # A 61k-move rule plan cannot gate on a 61k-row table: rows cap
        # loudly, the per-destination Summary stays complete.
        files = {f"pile/f{i:03d}.md": "x\n" for i in range(250)}
        build_corpus(self.root, files)
        plan = load_plan(
            write_plan(self.root, partial=True, rules=[{"match": "pile/", "to": "out"}])
        )
        out = validate_plan(plan, self.root)
        md = render_markdown(out, {}, None)
        # 254 = 250 expanded + 4 explicit (the write_plan defaults)
        self.assertIn("showing the first 200 of 254 moves", md)
        self.assertIn("| `out/` | 250 |", md)  # summary table: complete
        # folders starved by the cap never render as empty tables
        self.assertNotIn("## → `research/dossiers/`", md)


class TestRenderCodeAnchors(TempCase):
    """Phase 9: moves of code-anchored dirs surface at the gate."""

    def _plan_with_slack_move(self):
        return load_plan(
            write_plan(
                self.root,
                partial=True,
                moves=[{"from": "slack", "to": "archive/slack", "reason": "r"}],
            )
        )

    def test_move_of_anchored_dir_flagged(self):
        write(self.root / "slack/data/db.md", "x\n")
        write(
            self.root / "tools/run.py",
            'DATA = os.path.join(ROOT, "slack", "data")\n',
        )
        out = validate_plan(self._plan_with_slack_move(), self.root)
        risks = anchored_move_risks(self.root, out)
        self.assertEqual([r["from"] for r in risks], ["slack"])
        md = render_markdown(out, {}, None, code_anchor_risks=risks)
        self.assertIn("Code-anchored moves", md)
        self.assertIn("`slack` → `archive/slack`", md)
        self.assertIn("`slack/data`", md)
        self.assertIn("2 site(s)", md)

    def test_unanchored_move_has_no_section(self):
        write(self.root / "slack/data/db.md", "x\n")  # no citing code anywhere
        out = validate_plan(self._plan_with_slack_move(), self.root)
        risks = anchored_move_risks(self.root, out)
        self.assertEqual(risks, [])
        md = render_markdown(out, {}, None, code_anchor_risks=risks)
        self.assertNotIn("Code-anchored moves", md)


if __name__ == "__main__":
    unittest.main()
