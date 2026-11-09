# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""cli.py — in-process cli.main() coverage.

The subprocess tests in test_cli.py pin the exit-code contract but run
cli.py in child interpreters, so they raise no coverage. These call
main() directly with stdout captured and assert the returned int plus
the JSON envelope. Every cmd_* handler is exercised at least once;
mutating commands are run dry (or scoped to the temp corpus).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest

from helpers import TempCase, write
from slopkeeper import cli


def run_main(argv: list[str]) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed stdout JSON)."""
    code, text = run_main_raw(argv)
    return code, json.loads(text)


def run_main_raw(argv: list[str]) -> tuple[int, str]:
    """Run cli.main in-process; return (exit code, raw stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class TestCliMainReadOnly(TempCase):
    def test_plan_schema(self):
        # T6: the bare template artifact, not an envelope (DESIGN §2
        # exception, plan render's markdown precedent).
        code, out = run_main(["plan", "schema"])
        self.assertEqual(code, 0)
        self.assertEqual(out["schema"], "slopkeeper/plan@1")
        self.assertIn("moves", out)
        self.assertNotIn("data", out)

    def test_strategies_list(self):
        code, out = run_main(["strategies"])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "strategies.list")
        self.assertIn("hybrid", [s["id"] for s in out["data"]])

    def test_strategies_show_both_forms_same_envelope(self):
        envelopes = []
        for argv in (["strategies", "hybrid"], ["strategies", "show", "hybrid"]):
            code, out = run_main(argv)
            self.assertEqual(code, 0, argv)
            self.assertEqual(out["command"], "strategies.show")
            self.assertEqual(out["data"]["id"], "hybrid")
            envelopes.append(out)
        self.assertEqual(envelopes[0], envelopes[1])

    def test_unknown_strategy_exit_error_envelope(self):
        code, out = run_main(["strategies", "no-such-strategy"])
        self.assertEqual(code, 1)  # non-plan_* SlopKeeperError
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "unknown_strategy")

    def test_doctor(self):
        code, out = run_main(["doctor", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "doctor")
        self.assertIn("stats", out["data"])

    def test_inventory_full_and_summary(self):
        code, out = run_main(["inventory", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "inventory")
        self.assertIn("files", out["data"])

        out_path = self.tmp / "inv.json"
        code, out = run_main(
            ["inventory", str(self.root), "--summary", "--out", str(out_path)]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("heads", out["data"])
        self.assertTrue(out_path.exists())

    def test_clusters(self):
        code, out = run_main(["clusters", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "clusters")
        self.assertIn("clusters", out["data"])

    def test_inventory_out_reduces_stdout_to_counts(self):
        # T7: --out moves the full payload to the file — stdout carries
        # token-bounded accounting (stats + out_path), never the dump.
        out_path = self.tmp / "inv.json"
        code, text = run_main_raw(["inventory", str(self.root), "--out", str(out_path)])
        out = json.loads(text)
        self.assertEqual(code, 0)
        self.assertNotIn("files", out["data"])
        self.assertNotIn("dirs", out["data"])
        self.assertIn("stats", out["data"])
        self.assertEqual(out["data"]["out_path"], str(out_path))
        # A corpus filename must not leak onto stdout.
        self.assertNotIn("acme_dossier", text)
        self.assertTrue(out_path.exists())
        full = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertIn("files", full)
        self.assertIn("acme_dossier.md", [f["path"] for f in full["files"]])

    def test_clusters_out_writes_file_and_counts_envelope(self):
        # Same shape of output as inventory: full JSON in the --out file,
        # counts + out_path on stdout (the file lists are far too big for
        # the token budget).
        out_path = self.tmp / "clusters.json"
        code, text = run_main_raw(["clusters", str(self.root), "--out", str(out_path)])
        out = json.loads(text)
        self.assertEqual(code, 0)
        self.assertNotIn("clusters", out["data"])
        self.assertIn("cluster_count", out["data"])
        self.assertIn("unclustered_count", out["data"])
        self.assertEqual(out["data"]["out_path"], str(out_path))
        self.assertNotIn("acme_dossier", text)
        self.assertTrue(out_path.exists())
        full = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertIn("clusters", full)
        self.assertIn("unclustered", full)

    def test_refs(self):
        code, out = run_main(["refs", str(self.root), "acme_dossier.md"])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "refs")
        self.assertGreater(out["data"]["citation_count"], 0)

    def test_status_never_tended(self):
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "status")
        self.assertTrue(out["data"]["never_tended"])
        self.assertEqual(out["next"]["action"], "doctor")


class TestCliMainTendingLoop(TempCase):
    def test_status_record_then_diff(self):
        code, out = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(self.root / "stray_note.md", "# stray funding flows note\n")
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0)
        self.assertFalse(out["data"]["never_tended"])
        self.assertIn(
            "stray_note.md",
            out["data"]["modified"] + [f["path"] for f in out["data"]["new_files"]],
        )

    def test_draft_envelope_is_counts_only(self):
        # T7: the draft body lives ONLY in the --out file; the stdout
        # envelope carries token-bounded accounting, never the plan.
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(
            self.root / "globex_family_notes.md",
            "# Globex family\n\nmarket share.\n",
        )
        draft_path = self.tmp / "draft.json"
        code, out = run_main(["draft", str(self.root), "--out", str(draft_path)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "draft")
        self.assertNotIn("moves", json.dumps(out))
        self.assertNotIn("draft", out["data"])
        data = out["data"]
        self.assertEqual(data["drafted_count"] + data["unresolved_count"], 1)
        self.assertEqual(data["new_count"], 1)
        self.assertNotIn("undrafted", data)
        self.assertEqual(data["out_path"], str(draft_path))
        plan = json.loads(draft_path.read_text())
        self.assertTrue(plan["partial"])
        self.assertEqual(len(plan["moves"]) + len(plan["unresolved"]), 1)

    def test_draft_nothing_new(self):
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        draft_path = self.tmp / "d.json"
        code, out = run_main(["draft", str(self.root), "--out", str(draft_path)])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["drafted_count"], 0)
        self.assertEqual(out["data"]["new_count"], 0)
        self.assertIsNone(out["data"]["out_path"])
        self.assertFalse(draft_path.exists())
        self.assertEqual(out["next"]["action"], "status")

    def test_draft_default_out_inside_root_refused(self):
        # T7: the CWD-relative default must not land the draft in the
        # corpus (it would shift the fingerprint → corpus_changed at
        # apply). CWD == root and no --out → loud refusal, nothing written.
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(
            self.root / "globex_family_notes.md",
            "# Globex family\n\nmarket share.\n",
        )
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            code, out = run_main(["draft", str(self.root)])
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "draft_out_inside_root")
        self.assertFalse((self.root / "draft-plan.json").exists())

    def test_draft_out_explicitly_inside_root_refused(self):
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(
            self.root / "globex_family_notes.md",
            "# Globex family\n\nmarket share.\n",
        )
        code, out = run_main(
            ["draft", str(self.root), "--out", str(self.root / "inside.json")]
        )
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "draft_out_inside_root")
        self.assertFalse((self.root / "inside.json").exists())

    def test_zero_move_draft_is_valid_and_records_backlog(self):
        # T7 (parent note): a draft that filed EVERY new file as
        # unresolved (weak signals) is a valid partial plan with zero
        # moves — validate passes, and apply --execute records the
        # human-decides backlog instead of dying on plan_no_moves.
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(
            self.root / "minutes-log.md",
            "# Minutes\n\nstandup agenda parking lot\n" * 3,
        )
        draft_path = self.tmp / "draft.json"
        code, out = run_main(
            ["draft", str(self.root), "--min-sim", "0.99", "--out", str(draft_path)]
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(out["data"]["drafted_count"], 0)
        self.assertEqual(out["data"]["unresolved_count"], 1)
        plan = json.loads(draft_path.read_text())
        self.assertEqual(plan["moves"], [])
        self.assertEqual(len(plan["unresolved"]), 1)
        # Zero-move partial plans validate.
        code, out = run_main(["plan", "validate", str(draft_path), str(self.root)])
        self.assertEqual(code, 0, out)
        self.assertEqual(out["data"]["moves"], 0)
        # apply --execute records the manifest with the backlog intact.
        code, out = run_main(["apply", str(draft_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, out)
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(len(out["data"]["still_unresolved"]), 1)

    def test_draft_depth_bounds_the_sweep(self):
        # The sweep is recursive by default; --depth bounds it. A file
        # two levels deep is accounted in `undrafted` with the depth
        # reason, and the plan body never mentions it.
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        write(
            self.root / "globex_family_notes.md",
            "# Globex family\n\nmarket share.\n",
        )
        write(
            self.root / "docs" / "deep" / "globex_deeper_notes.md",
            "# Globex family\n\nmarket share.\n",
        )
        draft_path = self.tmp / "shallow-draft.json"
        code, out = run_main(
            [
                "draft",
                str(self.root),
                "--depth",
                "1",
                "--out",
                str(draft_path),
            ]
        )
        self.assertEqual(code, 0, out)
        data = out["data"]
        self.assertEqual(data["new_count"], 2)
        self.assertEqual(data["drafted_count"] + data["unresolved_count"], 1)
        self.assertEqual(data["undrafted"]["count"], 1)
        self.assertIn("--depth 1", data["undrafted"]["reason"])
        plan = json.loads(draft_path.read_text())
        self.assertEqual(plan["draft"]["depth"], 1)
        self.assertNotIn(
            "globex_deeper_notes.md",
            json.dumps(plan["moves"]) + json.dumps(plan["unresolved"]),
        )

    def test_draft_undrafted_beyond_status_cap(self):
        # T7: status caps new_files at 2000 — the overflow must not
        # vanish silently. The envelope accounts for it in `undrafted`.
        code, _ = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0)
        for i in range(2200):
            write(self.root / f"slop-{i:04d}.md", f"# slop {i}\n\nfiller {i}\n")
        draft_path = self.tmp / "big-draft.json"
        code, out = run_main(["draft", str(self.root), "--out", str(draft_path)])
        self.assertEqual(code, 0, out)
        data = out["data"]
        self.assertEqual(data["new_count"], 2200)
        self.assertEqual(data["drafted_count"] + data["unresolved_count"], 2000)
        self.assertEqual(data["undrafted"]["count"], 200)
        self.assertIn("cap", data["undrafted"]["reason"])
        self.assertIn("undrafted", out["next"]["hint"])
        # stdout carries no draft body; the file has every considered file.
        self.assertNotIn("moves", json.dumps(out))
        plan = json.loads(draft_path.read_text())
        self.assertEqual(len(plan["moves"]) + len(plan["unresolved"]), 2000)

    def test_plan_validate_render_apply_scaffold_undo(self):
        from helpers import write_plan

        plan_path = write_plan(self.root)

        code, out = run_main(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "plan.validate")
        self.assertTrue(out["data"]["valid"])

        # No --out: cmd_plan_render prints the raw markdown, not an envelope.
        code, text = run_main_raw(["plan", "render", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("acme_dossier.md", text)

        md_path = self.tmp / "gate.md"
        code, out = run_main(
            ["plan", "render", str(plan_path), str(self.root), "--out", str(md_path)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["written_to"], str(md_path))
        self.assertTrue(md_path.exists())

        code, out = run_main(["apply", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["mode"], "dry-run")
        self.assertEqual(out["next"]["action"], "apply")

        code, out = run_main(["apply", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["mode"], "executed")
        self.assertEqual(out["next"]["action"], "scaffold")

        code, out = run_main(["scaffold", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["mode"], "dry-run")

        code, out = run_main(["scaffold", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0)
        self.assertEqual(out["data"]["mode"], "executed")

        # Undo the journal we just wrote (dry-run: nothing moves).
        journals = sorted((self.root / ".slopkeeper").glob("journal-*.jsonl"))
        self.assertTrue(journals)
        code, out = run_main(["undo", str(journals[-1]), str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out["command"], "undo")
        self.assertEqual(out["data"]["mode"], "dry-run")

    def test_plan_validate_bad_plan_exit_validation(self):
        bad = self.tmp / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "schema": "slopkeeper/plan@0",
                    "corpus_root": str(self.root),
                    "moves": [],
                }
            )
        )
        code, out = run_main(["plan", "validate", str(bad), str(self.root)])
        self.assertEqual(code, 2)  # plan_* codes
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "plan_bad_schema")

    def test_undo_missing_journal_exit_error(self):
        code, out = run_main(
            ["undo", str(self.tmp / "no-journal.jsonl"), str(self.root)]
        )
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
