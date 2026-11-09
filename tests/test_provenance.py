# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""DESIGN.md §5.6 — approval provenance: bind the executed plan to its bytes.

apply --execute records plan_sha256 (manifest + journal header); render
--out writes .slopkeeper/last-render.json; apply compares and WARNS on
drift — never refuses, so the documented hand-edit-the-JSON flow (the
human edits the plan JSON after render) keeps working.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import unittest
from pathlib import Path

from helpers import TempCase, write_plan
from slopkeeper.apply import RENDER_WARNING, apply_plan, undo_journal
from slopkeeper.cli import main as cli_main
from slopkeeper.plan import (
    RENDER_SCHEMA,
    load_render_sidecar,
    render_sidecar_path,
    write_render_sidecar,
)
from slopkeeper.status import manifest_path


def plan_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestExecuteProvenance(TempCase):
    def test_execute_records_plan_sha_in_manifest_and_journal_header(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        expected = plan_sha(plan_path)
        # Manifest: the new field binds the snapshot to the plan bytes.
        manifest = json.loads(manifest_path(self.root).read_text("utf-8"))
        self.assertEqual(manifest["plan_sha256"], expected)
        self.assertEqual(manifest["last_plan"], str(plan_path))
        # Journal: the header is the FIRST line, before any move.
        lines = journal_lines(Path(out["journal"]))
        self.assertEqual(lines[0]["op"], "plan")
        self.assertEqual(lines[0]["sha256"], expected)
        self.assertIn("ts", lines[0])
        self.assertEqual(lines[1]["op"], "move")
        self.assertEqual(len([e for e in lines if e.get("op") == "move"]), 4)

    def test_undo_tolerates_the_plan_header_line(self):
        # The header is an unknown op to undo; reversal must replay the
        # moves exactly and not choke on (or reorder around) it.
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        undo = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(undo["mode"], "executed")
        restored = [op for op in undo["operations"] if op.get("status") == "restored"]
        self.assertEqual(len(restored), 4)
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

    def test_manifest_without_plan_has_null_plan_sha(self):
        # status --record / undo snapshots carry no plan: field stays,
        # value is None (additive field, no schema bump).
        from slopkeeper.status import write_manifest

        manifest = write_manifest(self.root)
        self.assertIsNone(manifest["plan_sha256"])


class TestRenderSidecar(TempCase):
    def test_render_out_writes_sidecar_with_plan_hash(self):
        plan_path = write_plan(self.root)
        out_md = self.tmp / "plan.md"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(
                ["plan", "render", str(plan_path), str(self.root), "--out", str(out_md)]
            )
        self.assertEqual(rc, 0)
        self.assertTrue(out_md.exists())
        # .slopkeeper/ did not exist before: the sidecar creates it.
        self.assertTrue(render_sidecar_path(self.root).exists())
        sidecar = load_render_sidecar(self.root)
        self.assertEqual(sidecar["schema"], RENDER_SCHEMA)
        self.assertEqual(sidecar["plan_sha256"], plan_sha(plan_path))
        self.assertIn("rendered_at", sidecar)

    def test_stdout_render_writes_no_sidecar(self):
        plan_path = write_plan(self.root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["plan", "render", str(plan_path), str(self.root)])
        self.assertEqual(rc, 0)
        self.assertIsNone(load_render_sidecar(self.root))
        self.assertFalse(render_sidecar_path(self.root).exists())


class TestRenderWarning(TempCase):
    def test_mismatched_sidecar_warns_on_dry_run_exit_zero(self):
        plan_path = write_plan(self.root)
        # A render of DIFFERENT bytes: the sidecar hashes another plan.
        other = self.tmp / "other.json"
        other.write_text('{"schema": "slopkeeper/plan@1"}', encoding="utf-8")
        write_render_sidecar(other, self.root)
        dry = apply_plan(plan_path, self.root)
        self.assertEqual(dry["mode"], "dry-run")
        self.assertEqual(dry["render_warning"], RENDER_WARNING)
        # Through the CLI: exit 0, success envelope, no error object —
        # the warning is data, never a refusal.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["apply", str(plan_path), str(self.root)])
        self.assertEqual(rc, 0)
        envelope = json.loads(buf.getvalue())
        self.assertTrue(envelope["ok"])
        self.assertNotIn("error", envelope)
        self.assertEqual(envelope["data"]["render_warning"], RENDER_WARNING)

    def test_matching_sidecar_has_no_warning_key(self):
        plan_path = write_plan(self.root)
        write_render_sidecar(plan_path, self.root)
        dry = apply_plan(plan_path, self.root)
        self.assertNotIn("render_warning", dry)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertNotIn("render_warning", out)

    def test_no_sidecar_no_warning_no_crash(self):
        plan_path = write_plan(self.root)
        dry = apply_plan(plan_path, self.root)
        self.assertNotIn("render_warning", dry)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertNotIn("render_warning", out)

    def test_hand_edited_plan_json_warns_and_still_executes(self):
        # The documented flow (AGENT_GUIDE step 6): render, then the
        # human edits the plan's JSON. Apply must WARN and still run.
        plan_path = write_plan(self.root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(
                [
                    "plan",
                    "render",
                    str(plan_path),
                    str(self.root),
                    "--out",
                    str(self.tmp / "plan.md"),
                ]
            )
        plan = json.loads(plan_path.read_text("utf-8"))
        plan["moves"][0]["reason"] = "human reworded this one"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        self.assertEqual(out["render_warning"], RENDER_WARNING)
        self.assertFalse((self.root / "acme_dossier.md").exists())

    def test_corrupt_sidecar_never_blocks_apply(self):
        plan_path = write_plan(self.root)
        sidecar = render_sidecar_path(self.root)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("{not json", encoding="utf-8")
        self.assertIsNone(load_render_sidecar(self.root))
        dry = apply_plan(plan_path, self.root)
        self.assertNotIn("render_warning", dry)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")


if __name__ == "__main__":
    unittest.main()
