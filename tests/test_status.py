# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""status.py — manifest and the tending loop."""

from __future__ import annotations

import contextlib
import io
import json
import os
import unittest

from helpers import TempCase, write_plan
from slopkeeper.apply import apply_plan
from slopkeeper.plan import load_plan
from slopkeeper.status import MANIFEST_SCHEMA, load_manifest, manifest_path
from slopkeeper.status import _walk_now
from slopkeeper.status import status as corpus_status
from slopkeeper.status import status_all
from helpers import expect_code, write
from slopkeeper.scaffold import scaffold_from_plan
from slopkeeper import cli


def run_main(argv: list[str]) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed stdout JSON)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class TestStatus(TempCase):
    def test_never_tended_then_record_then_diff(self):
        s = corpus_status(self.root)
        self.assertTrue(s["never_tended"])
        # Record a baseline.
        s = corpus_status(self.root, record=True)
        self.assertEqual(s["files_now"], 8)
        # Shoot some slop in.
        (self.root / "NEW-SLOP-globex-memo.md").write_text(
            "# Globex memo\n\nfresh slop about market share.\n"
        )
        (self.root / "acme_dossier.md").write_text("# CHANGED\n")
        s = corpus_status(self.root)
        self.assertEqual(s["new_count"], 1)
        self.assertIn("NEW-SLOP-globex-memo.md", [f["path"] for f in s["new_files"]])
        self.assertIn("acme_dossier.md", s["modified"])
        self.assertEqual(s["still_unresolved"], [])
        # Triage estimate is sane.
        self.assertGreater(s["estimated_triage_tokens"], 0)

    def test_stale_walk_contract_hint_cleared_by_record(self):
        # fp4 (reference-rewrites PRD Phase 1): a manifest recorded under
        # an older walk contract must not serve a misleading diff — it
        # names the problem and the one-command fix, and --record clears it.
        corpus_status(self.root, record=True)
        m = load_manifest(self.root)
        m["fingerprint"] = "fp3:0123456789abcdef0123456789abcdef:8"
        manifest_path(self.root).write_text(json.dumps(m), encoding="utf-8")
        s = corpus_status(self.root)
        self.assertTrue(s["walk_contract_stale"])
        self.assertIn("status --record", " ".join(s.get("advice", [])))
        s = corpus_status(self.root, record=True)
        self.assertFalse(s["walk_contract_stale"])

    def test_apply_executes_records_manifest(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        m = load_manifest(self.root)
        self.assertIsNotNone(m)
        self.assertEqual(m["file_count"], 8)  # moves don't change the count
        # ...and the tending loop continues from there.
        (self.root / "more-slop.md").write_text("# slop\n")
        s = corpus_status(self.root)
        self.assertEqual(s["new_count"], 1)
        self.assertEqual(s["last_tended"], m["recorded_at"])

    def test_record_preserves_unresolved_and_last_plan(self):
        # T2: --record re-baselines the file listing; it must not wipe the
        # unresolved backlog or last_plan that apply --execute recorded.
        plan_path = write_plan(
            self.root, unresolved=[{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        apply_plan(plan_path, self.root, execute=True)
        corpus_status(self.root, record=True)
        m = load_manifest(self.root)
        self.assertEqual(
            m["unresolved"], [{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        self.assertEqual(m["last_plan"], str(plan_path))
        self.assertEqual(len(corpus_status(self.root)["still_unresolved"]), 1)

    def test_manifest_survives_as_unresolved_tracker(self):
        plan_path = write_plan(
            self.root, unresolved=[{"path": "ledger_wb.html", "reason": "no signal"}]
        )
        plan = load_plan(plan_path)
        plan["moves"] = [m for m in plan["moves"]]  # keep
        apply_plan(plan_path, self.root, execute=True, allow_partial=True)
        m = load_manifest(self.root)
        self.assertEqual(m["unresolved"][0]["path"], "ledger_wb.html")


class TestRebaselineAndNested(TempCase):
    """Phase 7: heavy drift suggests --record; nested corpora are named."""

    def test_drift_hints_record(self):
        # The threshold floor is 2000 (max(20% of manifest, 2000)), so a
        # small manifest needs > 2000 changes to trip it — shoot in a
        # bulk-import-sized slop pile and the envelope must say
        # re-record, never "hand-build a zero-move plan".
        corpus_status(self.root, record=True)
        for i in range(2100):
            write(self.root / f"slop/file{i:04}.md", "# slop\n")
        s = corpus_status(self.root)
        self.assertEqual(s["heavy_drift"]["changed"], 2100)
        self.assertEqual(s["heavy_drift"]["threshold"], 2000)
        self.assertEqual(s["heavy_drift"]["new"], 2100)
        self.assertIn("status --record", " ".join(s.get("advice", [])))
        # The CLI envelope's next.command is the exact re-record command.
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(
            out["next"]["command"],
            f"slopkeeper status --record {self.root.resolve()}",
        )
        self.assertIn("Heavy drift", out["next"]["hint"])

    def test_light_drift_keeps_plan_next(self):
        # Below the threshold the response is a delta plan, not a
        # rebaseline — the hint must not fire for ordinary slop.
        corpus_status(self.root, record=True)
        write(self.root / "one-more-slop.md", "# slop\n")
        s = corpus_status(self.root)
        self.assertNotIn("heavy_drift", s)
        code, out = run_main(["status", str(self.root)])
        self.assertNotIn("--record", out["next"]["command"])

    def test_record_round_trip_clears_hint(self):
        # Re-recording IS the fix: the next status is quiet again.
        corpus_status(self.root, record=True)
        for i in range(2100):
            write(self.root / f"slop/file{i:04}.md", "# slop\n")
        self.assertIn("heavy_drift", corpus_status(self.root))
        s = corpus_status(self.root, record=True)
        self.assertNotIn("heavy_drift", s)
        # The backlog and last_plan survive the rebaseline (write_manifest
        # carries them forward); a plain status stays quiet.
        s = corpus_status(self.root)
        self.assertNotIn("heavy_drift", s)
        self.assertEqual(s["new_count"], 0)

    def test_nested_corpus_skipped_and_reported(self):
        # docs/ grows its own .slopkeeper/manifest.json: it is a separate
        # corpus now. The parent must NOT double-count it (no fake
        # removal of docs/canonical.md, no new_count for its files) and
        # must NAME it — never a silently shrunken diff.
        corpus_status(self.root, record=True)
        (self.root / "docs" / ".slopkeeper").mkdir(parents=True)
        (self.root / "docs" / ".slopkeeper" / "manifest.json").write_text(
            json.dumps({"schema": MANIFEST_SCHEMA}), encoding="utf-8"
        )
        write(
            self.root / "docs" / ".slopkeeper" / "notes.md",
            "nested corpus's own state\n",
        )
        s = corpus_status(self.root)
        self.assertEqual(s["nested_corpora"], ["docs"])
        self.assertNotIn("docs/canonical.md", s["removed"])
        self.assertEqual(s["removed_now_nested"], ["docs/canonical.md"])
        self.assertEqual(s["new_count"], 0)
        self.assertEqual(s["files_now"], 7)  # 8 minus the nested corpus's file
        # The walk itself agrees: the nested subtree yields nothing.
        self.assertNotIn("docs/canonical.md", _walk_now(self.root))
        self.assertNotIn("docs/.slopkeeper/notes.md", _walk_now(self.root))

    def test_nested_corpus_vanishes_back_to_content(self):
        # Removing the nested manifest returns the subtree to ordinary
        # content: the corpus is visible again (as new slop), and no
        # removed_now_nested residue remains.
        corpus_status(self.root, record=True)
        (self.root / "docs" / ".slopkeeper").mkdir(parents=True)
        (self.root / "docs" / ".slopkeeper" / "manifest.json").write_text("{}")
        self.assertEqual(corpus_status(self.root)["nested_corpora"], ["docs"])
        (self.root / "docs" / ".slopkeeper" / "manifest.json").unlink()
        s = corpus_status(self.root)
        self.assertEqual(s["nested_corpora"], [])
        self.assertEqual(s["removed_now_nested"], [])
        # The state dir is still skip-list, but docs/canonical.md is
        # back — it was never gone, so nothing is "new" (it is in the
        # manifest) and nothing is removed.
        self.assertEqual(s["new_count"], 0)
        self.assertEqual(s["removed"], [])


# --- gates: manifest schema, self-pollution pin, modified detection ---


class TestStatusGates(TempCase):
    def test_manifest_bad_schema_is_refused(self):
        mdir = self.root / ".slopkeeper"
        mdir.mkdir()
        (mdir / "manifest.json").write_text(
            json.dumps({"schema": "slopkeeper/manifest@0"})
        )
        expect_code(self, lambda: corpus_status(self.root), "manifest_bad_schema")
        # The --record path hits the same gate before re-recording.
        expect_code(
            self, lambda: corpus_status(self.root, record=True), "manifest_bad_schema"
        )

    def test_record_rebuilds_empty_on_corrupt_manifest(self):
        # --record is the documented recovery for a corrupt manifest —
        # it must rebuild (empty), not die on its own recovery path.
        # A wrong-schema manifest still refuses (pinned above); this is
        # the unreadable-JSON case.
        plan_path = write_plan(
            self.root, unresolved=[{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        apply_plan(plan_path, self.root, execute=True)
        (self.root / ".slopkeeper" / "manifest.json").write_text(
            "{not json", encoding="utf-8"
        )
        s = corpus_status(self.root, record=True)  # must not raise
        self.assertEqual(s["still_unresolved"], [])
        m = load_manifest(self.root)  # valid manifest again
        self.assertEqual(m["unresolved"], [])
        self.assertIsNone(m["last_plan"])

    def test_tool_layer_not_reported_as_new_slop(self):
        # Self-pollution pin: apply --execute records a manifest, then
        # scaffold --execute writes the AGENTS.md/STATE.md layer and
        # re-records, so the next `status` shows only real slop — never the
        # tool's own files. (status itself has no tool-layer filter; the
        # guarantee comes from scaffold's write_manifest re-record —
        # pinned mechanism.)
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        (self.root / "STATE.md").write_text("# state\n")
        (self.root / "INDEX.md").write_text("# index\n")
        scaffold_from_plan(plan_path, self.root, execute=True)
        s = corpus_status(self.root)
        self.assertEqual(s["new_count"], 0, s["new_files"])
        self.assertNotIn("AGENTS.md", [f["path"] for f in s["new_files"]])
        self.assertNotIn("STATE.md", [f["path"] for f in s["new_files"]])

    def test_modified_detection_same_size_edit(self):
        # A real content edit with identical byte size must still be
        # flagged: detection compares content hashes (§5.2), which
        # differ even when size and mtime were somehow identical.
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        p = self.root / "lead-gen" / "README.md"
        p.write_bytes(p.read_bytes().replace(b"pipeline", b"pipezinb"))
        s = corpus_status(self.root)
        self.assertIn("lead-gen/README.md", s["modified"])


# --- manifest@2: content hashes, touch immunity, v1 refusal (§5.2) ---


class TestManifestV2(TempCase):
    def test_manifest_records_content_hashes(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        m = load_manifest(self.root)
        self.assertEqual(m["schema"], MANIFEST_SCHEMA)
        self.assertEqual(MANIFEST_SCHEMA, "slopkeeper/manifest@2")
        for entry in m["files"].values():
            self.assertEqual(len(entry), 3)
            size, mtime_ns, sha = entry
            self.assertIsInstance(size, int)
            self.assertIsInstance(mtime_ns, int)
            self.assertTrue(sha is None or (isinstance(sha, str) and len(sha) == 64))

    def test_touch_not_reported_modified(self):
        # Flipped pin (was: touch → modified). Same content, same size,
        # new mtime: not modified. Detection is content-hash based.
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        p = self.root / "lead-gen" / "README.md"
        os.utime(p, (1_500_000_000, 1_500_000_000))
        s = corpus_status(self.root)
        self.assertEqual(s["modified"], [])

    def test_manifest_v1_on_disk_refused(self):
        # manifest@1 files keep the loud gate: the message says delete
        # and re-record. No silent reinterpretation of the old schema.
        mdir = self.root / ".slopkeeper"
        mdir.mkdir()
        (mdir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "slopkeeper/manifest@1",
                    "files": {"a.md": [5, 123]},  # v1: no hash slot
                }
            )
        )
        expect_code(self, lambda: corpus_status(self.root), "manifest_bad_schema")

    def test_null_hash_falls_back_to_size_mtime(self):
        # When either side lacks a hash (the sha256_file 64 MiB cap),
        # the pre-v2 rule applies: (size, mtime_ns) equality decides.
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        rel = "lead-gen/README.md"
        mp = self.root / ".slopkeeper" / "manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8"))

        # (a) null hash + mtime moved → modified (fallback catches it).
        p = self.root / rel
        m["files"][rel] = [p.stat().st_size, 1_000_000_000, None]
        mp.write_text(json.dumps(m), encoding="utf-8")
        os.utime(p, (1_500_000_000, 1_500_000_000))
        self.assertIn(rel, corpus_status(self.root)["modified"])

        # (b) null hash + size/mtime equal to disk → not modified.
        st = p.stat()
        m["files"][rel] = [st.st_size, st.st_mtime_ns, None]
        mp.write_text(json.dumps(m), encoding="utf-8")
        self.assertNotIn(rel, corpus_status(self.root)["modified"])


# --- folder health: STATE.md staleness + binary triage rows ---


class TestFolderHealth(TempCase):
    def test_stale_folders_detected(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        state = self.root / "spark-1.2.3" / "STATE.md"
        state.write_text("# STATE\n")
        # STATE.md is the newest thing in the folder: not stale.
        self.assertEqual(corpus_status(self.root)["stale_folders"], [])
        # Backdate STATE.md behind a newer file: stale.
        os.utime(state, (1_500_000_000, 1_500_000_000))
        (self.root / "spark-1.2.3" / "fresh-slop.md").write_text("# new\n")
        s = corpus_status(self.root)
        self.assertEqual([f["folder"] for f in s["stale_folders"]], ["spark-1.2.3"])

    def test_new_binary_file_reported_headless(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        (self.root / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        s = corpus_status(self.root)
        entry = next(f for f in s["new_files"] if f["path"] == "screenshot.png")
        self.assertEqual(entry["kind"], "binary")
        self.assertIsNone(entry["title"])
        self.assertIsNone(entry["head"])


class TestStateFileRobustness(TempCase):
    """T3: state-file content is data, never a crash.

    A null path in the backlog (older writer, hand edit) used to
    TypeError every later `status` walk; a list-shaped manifest
    surfaced as `internal`. Both are now loud-but-safe: the walk skips
    and counts, and foreign JSON is manifest_bad_schema.
    """

    def _poison_backlog(self) -> None:
        """Healthy manifest, then inject a null-path backlog entry."""
        plan_path = write_plan(
            self.root, unresolved=[{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        apply_plan(plan_path, self.root, execute=True)
        mp = self.root / ".slopkeeper" / "manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["unresolved"] = [
            {"path": None, "reason": "poison"},
            {"path": "WATCHLIST.md", "reason": "human decides"},
        ]
        mp.write_text(json.dumps(m), encoding="utf-8")

    def test_status_counts_and_skips_null_path_backlog(self):
        # The ticket's probe: manifest with `unresolved: [{"path": null,
        # "reason": "x"}]` → `slopkeeper status` exits 0 with the count.
        self._poison_backlog()
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 0, out)
        self.assertTrue(out["ok"])
        self.assertEqual(out["data"]["skipped_bad_unresolved"], 1)
        self.assertEqual(
            [u["path"] for u in out["data"]["still_unresolved"]],
            ["WATCHLIST.md"],
        )

    def test_record_drops_null_paths_and_counts_in_manifest(self):
        # The write side: a re-record drops the poison from the manifest
        # (apply/scaffold/undo all snapshot through write_manifest) and
        # records how many entries it skipped.
        self._poison_backlog()
        code, out = run_main(["status", str(self.root), "--record"])
        self.assertEqual(code, 0, out)
        m = load_manifest(self.root)
        self.assertEqual(
            m["unresolved"], [{"path": "WATCHLIST.md", "reason": "human decides"}]
        )
        self.assertEqual(m["skipped_bad_unresolved"], 1)
        # The next plain read sees a clean backlog: nothing left to skip.
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(out["data"]["skipped_bad_unresolved"], 0)
        self.assertEqual(len(out["data"]["still_unresolved"]), 1)

    def test_list_shaped_manifest_is_bad_schema_not_internal(self):
        # The ticket's probe: `echo '[]' > manifest.json` →
        # manifest_bad_schema, not internal.
        (self.root / ".slopkeeper").mkdir()
        (self.root / ".slopkeeper" / "manifest.json").write_text("[]", encoding="utf-8")
        code, out = run_main(["status", str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(out["error"]["code"], "manifest_bad_schema")


class TestStatusAll(TempCase):
    def test_per_root_error_isolation(self):
        # A corrupt manifest in ONE root must not poison the digest:
        # the row carries ok:false + the error, the rest stay ok.
        bad = self.tmp / "corpus2"
        (bad / ".slopkeeper").mkdir(parents=True)
        (bad / ".slopkeeper" / "manifest.json").write_text(
            json.dumps({"schema": "slopkeeper/manifest@0"})
        )
        rows = status_all([str(self.root), str(bad)])
        by_root = {r["root"]: r for r in rows}
        self.assertTrue(by_root[str(self.root)]["ok"])
        self.assertFalse(by_root[str(bad)]["ok"])
        self.assertEqual(by_root[str(bad)]["error"]["code"], "manifest_bad_schema")

    def test_digest_rows_are_counts(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        (self.root / "more-slop.md").write_text("# slop\n")
        rows = status_all([str(self.root)])
        row = rows[0]
        self.assertTrue(row["ok"])
        self.assertEqual(row["new_count"], 1)
        self.assertEqual(row["stale_folders"], 0)
        self.assertEqual(row["unresolved_count"], 0)
        self.assertNotIn("new_files", row)


if __name__ == "__main__":
    unittest.main()
