# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Golden files: committed snapshots of the two human-facing surfaces.

render_markdown IS the human approval gate and the AGENTS.md/STATE.md pair
is the layer the owner lives in — their shape is API (TEST_PLAN.md,
"Golden files"). Any change to the rendered output fails here with a full
multiline diff in the assert message; update the snapshots deliberately
and review the diff:

    python3 -c "import sys; sys.path[:0] = ['tests', 'src']; \
import test_golden as g; g.regen()"

STATE.md is date-normalized (strategy templates may stamp today's date);
everything else is compared byte-exact.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slopkeeper.plan import render_markdown
from slopkeeper.scaffold import scaffold_from_plan

GOLDEN = Path(__file__).resolve().parent / "golden"

# A small fixed plan: 3 moves, 1 keep, 1 unresolved, 1 folder with
# boundaries, 1 edit. corpus_root is a FIXED fake path — render_markdown
# never touches disk, and a real tmp path would leak into the golden bytes.
GOLDEN_PLAN = {
    "schema": "slopkeeper/plan@1",
    "corpus_root": "/corpus",
    "strategy": "hybrid",
    "based_on": {
        "inventory_sha256": "0123456789abcdef0123456789abcdef",
        "inventory_at": "2026-11-02T00:00:00+00:00",
    },
    "moves": [
        {
            "from": "acme_dossier.md",
            "to": "research/dossiers/acme_dossier.md",
            "reason": "dossier research",
        },
        {
            "from": "globex_dossier.md",
            "to": "research/dossiers/globex_dossier.md",
            "reason": "dossier research",
        },
        {
            "from": "handoff-tp2.md",
            "to": "spark-glm-5.3/handoff-tp2.md",
            "reason": "spark handoff",
        },
    ],
    "keep": [{"path": "WATCHLIST.md", "reason": "living document, do not move"}],
    "unresolved": [{"path": "ledger_wb.html", "reason": "no signal — human decides"}],
    "folders": [
        {
            "path": "research/dossiers",
            "kind": "case",
            "role": "Market research dossiers.",
            "boundaries": [
                "finished dossiers move to archive/<year-month>",
                "no active research, only closed dossiers",
            ],
        }
    ],
    "edits": [
        {
            "file": "WATCHLIST.md",
            "find": "acme_dossier.md",
            "replace": "research/dossiers/acme_dossier.md",
            "reason": "rewrite citation to the moved dossier",
        },
        {
            "file": "/home/agent/harness/nightly.sh",
            "find": "tmp/docs/acme_dossier.md",
            "replace": "docs/acme_dossier.md",
            "reason": "external citation (cron harness)",
        },
    ],
    "external_roots": ["/home/agent/harness"],
}
GOLDEN_REFS = {
    "acme_dossier.md": {"external_references": 1, "referenced_by": ["WATCHLIST.md"]}
}

# state_md is explicit here: scaffold_from_plan reads RAW plan folders, and
# the bot-kind default lives in validate_plan (pinned in test_scaffold_md).
SCAFFOLD_FOLDERS = [
    {
        "path": "research/dossiers",
        "kind": "bot",
        "role": "Market research dossiers.",
        "boundaries": ["finished dossiers move to archive/<year-month>"],
        "state_md": True,
    }
]

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _render_golden_plan() -> str:
    # Phase 4: the gate now carries the rewrite summary. For this plan
    # the only citer (WATCHLIST.md) is named in edits, so the pass
    # rewrites nothing — the summary below is what apply would compute.
    summary = {
        "mode": "auto",
        "per_file": [],
        "count": 0,
        "skipped": [],
        "targets": {},
    }
    return render_markdown(dict(GOLDEN_PLAN), dict(GOLDEN_REFS), summary)


def _scaffold_pair(root: Path) -> tuple[str, str]:
    """Run scaffold_from_plan --execute on a fixed plan; return (AGENTS, STATE)."""
    plan = {
        "schema": "slopkeeper/plan@1",
        "corpus_root": str(root),
        "strategy": "hybrid",
        "folders": SCAFFOLD_FOLDERS,
    }
    plan_path = root.parent / "golden-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    scaffold_from_plan(plan_path, root, execute=True)
    folder = root / "research" / "dossiers"
    agents = (folder / "AGENTS.md").read_text(encoding="utf-8")
    state = _DATE_RE.sub("<date>", (folder / "STATE.md").read_text(encoding="utf-8"))
    return agents, state


def regen() -> None:
    """Rewrite the golden files from current output (review the diff!)."""
    GOLDEN.mkdir(exist_ok=True)
    GOLDEN.joinpath("render_plan.md").write_text(
        _render_golden_plan(), encoding="utf-8"
    )
    with tempfile.TemporaryDirectory() as td:
        agents, state = _scaffold_pair(Path(td) / "corpus")
    GOLDEN.joinpath("scaffold_AGENTS.md").write_text(agents, encoding="utf-8")
    GOLDEN.joinpath("scaffold_STATE.md").write_text(state, encoding="utf-8")


class TestGoldenRender(unittest.TestCase):
    maxDiff = None  # the full diff IS the failure message

    def test_render_markdown_matches_golden(self):
        golden = GOLDEN.joinpath("render_plan.md").read_text(encoding="utf-8")
        # assertEqual on multiline strings renders a full diff on failure.
        self.assertEqual(_render_golden_plan(), golden)


class TestGoldenGuideDrift(unittest.TestCase):
    """Phase 6: the guide's error-code table is contract, not prose.

    test_error_codes.py owns the two-way check (raised ⟷ documented);
    this is the same guard from the golden side so doc drift fails the
    golden suite too: every code raised in src must have a table row.
    """

    def test_error_codes_documented(self):
        from test_error_codes import documented_codes, raised_codes

        missing = raised_codes() - documented_codes()
        self.assertEqual(
            missing,
            set(),
            "codes raised in src/ but missing a row in AGENT_GUIDE.md's "
            "'Error codes you will see' table (DESIGN §4.1): "
            + ", ".join(sorted(missing)),
        )


class TestGoldenScaffold(unittest.TestCase):
    maxDiff = None  # the full diff IS the failure message

    def test_agents_state_md_pair_matches_golden(self):
        golden_agents = GOLDEN.joinpath("scaffold_AGENTS.md").read_text(
            encoding="utf-8"
        )
        golden_state = GOLDEN.joinpath("scaffold_STATE.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            agents, state = _scaffold_pair(Path(td) / "corpus")
        self.assertEqual(agents, golden_agents)
        self.assertEqual(state, golden_state)


if __name__ == "__main__":
    unittest.main()
