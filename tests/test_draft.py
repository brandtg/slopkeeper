# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""draft.py — routing new slop to known folders."""

from __future__ import annotations

import json
import unittest

from helpers import TempCase, expect_code, write_plan
from slopkeeper.apply import apply_plan
from slopkeeper.draft import draft
from slopkeeper.plan import load_plan, validate_plan
from slopkeeper.scaffold import scaffold_from_plan
import unittest.mock as mock
from slopkeeper import draft as draft_module


class TestDraft(TempCase):
    def _tend_first(self):
        """First tend so a manifest exists."""
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        scaffold_from_plan(plan_path, self.root, execute=True)

    def test_no_manifest_is_an_error(self):
        expect_code(self, lambda: draft(self.root), "no_manifest")

    def test_draft_routes_by_nearest_neighbor(self):
        self._tend_first()
        # Slop that looks like the filed globex dossier (name + content).
        (self.root / "globex_new_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        # Junk with no neighbor.
        (self.root / "minutes-log.md").write_text(
            "# Minutes\n\nstandup agenda parking lot\n" * 3
        )
        out = draft(self.root)
        self.assertEqual(out["new_count"], 2)
        self.assertEqual(out["drafted_count"], 1)
        self.assertEqual(out["unresolved_count"], 1)
        plan = out["plan"]
        self.assertTrue(plan["partial"])
        self.assertEqual(plan["moves"][0]["to"], "research/dossiers/globex_new_memo.md")
        self.assertIn("nearest filed neighbor", plan["moves"][0]["reason"])
        self.assertIn("minutes-log.md", [u["path"] for u in plan["unresolved"]])
        # The draft plan must survive the real validator.
        p = self.tmp / "draft.json"
        p.write_text(json.dumps(plan))
        validated = validate_plan(load_plan(p), self.root)
        self.assertEqual(len(validated["moves"]), 1)

    def test_nested_slop_is_drafted_and_tool_layer_skipped(self):
        # The sweep is recursive: a new file inside research/ routes
        # exactly like a root one would — it may even leave research/.
        # A NEW tool-layer file is never slop (an edited one is only
        # "modified" — scaffold records it at the first tend).
        self._tend_first()
        (self.root / "research" / "globex_new_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        (self.root / "research" / "STATE.md").write_text("# state\n")
        out = draft(self.root)
        self.assertEqual(out["new_count"], 2)
        self.assertEqual(out["drafted_count"], 1)
        plan = out["plan"]
        self.assertEqual(plan["moves"][0]["from"], "research/globex_new_memo.md")
        self.assertEqual(plan["moves"][0]["to"], "research/dossiers/globex_new_memo.md")
        self.assertEqual(plan["draft"]["considered"], 1)
        self.assertIsNone(plan["draft"]["depth"])
        # The tool-layer file is accounted in `undrafted`, never dropped.
        self.assertEqual(out["undrafted"]["count"], 1)
        self.assertIn("tool-layer", out["undrafted"]["reason"])

    def test_slop_already_in_its_neighbors_folder_is_honest_unresolved(self):
        # A new file whose nearest filed neighbor lives in its OWN
        # folder is not slop — the draft records it as unresolved with
        # the reason instead of emitting a no-op move (from == to is
        # plan_noop_move in the validator).
        self._tend_first()
        (self.root / "spark-glm-5.3" / "glm-notes.md").write_text(
            "# GLM runbook\n\ntensorfold lanes\n" * 3
        )
        out = draft(self.root)
        self.assertEqual(out["drafted_count"], 0)
        reasons = {u["path"]: u["reason"] for u in out["plan"]["unresolved"]}
        self.assertIn("spark-glm-5.3/glm-notes.md", reasons)
        self.assertIn("own folder", reasons["spark-glm-5.3/glm-notes.md"])

    def test_depth_bounds_the_sweep(self):
        # --depth N considers only files within N directory levels of
        # the root (same convention as inventory); anything deeper is
        # accounted in `undrafted` with the depth reason.
        self._tend_first()
        (self.root / "globex_new_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        deep = self.root / "research" / "deep"
        deep.mkdir(parents=True, exist_ok=True)
        (deep / "globex_older_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        out = draft(self.root, depth=1)
        self.assertEqual(out["new_count"], 2)
        self.assertEqual(out["drafted_count"], 1)  # only the root file
        self.assertEqual(out["plan"]["draft"]["depth"], 1)
        self.assertEqual(out["undrafted"]["count"], 1)
        self.assertIn("--depth 1", out["undrafted"]["reason"])


# --- gates: the corpus cap + honest unresolved reasons ---


class TestDraftGates(TempCase):
    def test_corpus_too_big_over_cap(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        (self.root / "fresh-slop.md").write_text("# fresh slop\n")
        # draft.py imports MAX_FILES into its own namespace, so the cap is
        # patchable here without materializing 3000 real files.
        with mock.patch.object(draft_module, "MAX_FILES", 2):
            expect_code(self, lambda: draft(self.root), "corpus_too_big")

    def test_low_confidence_unresolved_names_the_score(self):
        # An unresolved file must say WHY: which neighbor was found and how
        # weak the score was, so the agent can triage without re-running.
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        (self.root / "globex_followup_memo.md").write_text(
            "# Globex memo\n\nmarket share flows, globex_dossier follow-up\n" * 3
        )
        out = draft(self.root, min_sim=0.99)
        reasons = {u["path"]: u["reason"] for u in out["plan"]["unresolved"]}
        self.assertIn("globex_followup_memo.md", reasons)
        self.assertRegex(
            reasons["globex_followup_memo.md"],
            r"best neighbor \S+ only 0\.\d{2} \(min 0\.99\)",
        )


if __name__ == "__main__":
    unittest.main()
