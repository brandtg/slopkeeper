# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P5 — agent ergonomics: envelope warnings, error.suggested_command,
path_map, and the citation scan's one silent cap.

The audit's agent-ergonomics roadmap named four gaps that all have the same
shape: the tool KNOWS something soft and the payload does not say it. A
scan that stopped reading at 512 KiB; a platform with no flock; a plan that
bound to nothing; a corpus whose paths the agent cached an apply ago. Each
of those is a field on an envelope here, and each test asserts the AGENT
view — what comes out of `cli.main` — not just the library dict, because
the envelope is the contract.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper import cli
from slopkeeper.apply import apply_plan, undo_journal
from slopkeeper.models import SlopKeeperError, add_warning, envelope
from slopkeeper.references import _PER_FILE_CAP
from slopkeeper.scaffold import scaffold_from_plan


def run_cli(argv) -> tuple[int, dict]:
    """Run cli.main in-process; return (exit code, parsed envelope)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class TestEnvelopeWarnings(TempCase):
    """P5 (1): `warnings` is an ENVELOPE field, mirrored from the payload."""

    def test_clean_run_has_no_warnings_key(self):
        # Absent-vs-present is the whole signal: an agent must be able to
        # test `out["warnings"]` and treat a missing key as "nothing soft
        # happened", exactly as data-level warnings have meant since P2.
        code, env = run_cli(["doctor", str(self.root)])
        self.assertEqual(code, 0)
        self.assertNotIn("warnings", env)
        self.assertNotIn("warnings", env["data"])

    def test_payload_warnings_mirror_to_the_envelope(self):
        # `plan validate` on a plan whose destinations casefold-collide
        # adds `case_collision` to the payload (P4). The envelope now
        # carries the same token, so one read of `out["warnings"]` covers
        # every command instead of knowing which payloads hide theirs.
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "acme_dossier.md", "to": "docs/Note.md", "reason": "x"},
                {"from": "globex_dossier.md", "to": "docs/note.md", "reason": "y"},
            ],
            keep=[
                "WATCHLIST.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "docs",
                "lead-gen",
                "spark-1.2.3",
            ],
            partial=True,
        )
        code, env = run_cli(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 0, env)
        self.assertIn("case_collision", env["warnings"])
        self.assertIn("case_collision", env["data"]["warnings"])

    def test_envelope_merges_explicit_and_payload_warnings(self):
        # The helper is the producer's path; the explicit argument is the
        # CLI's own. Union, ordered, deduped — no duplicates in JSON an
        # agent has to dedupe by hand.
        data = add_warning({}, "alpha")
        env = envelope("x", data, None, warnings=["beta", "alpha"])
        self.assertEqual(env["warnings"], ["alpha", "beta"])
        self.assertEqual(data["warnings"], ["alpha"])  # payload unchanged

    def test_non_dict_data_does_not_break_the_mirror(self):
        # `strategies list` passes a list as data; the mirror must not
        # assume a mapping.
        env = envelope("strategies.list", [{"id": "hybrid"}])
        self.assertNotIn("warnings", env)


class TestErrorSuggestedCommand(TempCase):
    """P5 (1): machine-readable recovery beside the prose `recovery`."""

    def test_absent_when_no_command_fixes_it(self):
        # The field is a promise, not a filler: no guess, no key.
        code, env = run_cli(
            ["plan", "validate", str(self.tmp / "nope.json"), str(self.root)]
        )
        self.assertEqual(code, 2)
        self.assertEqual(env["error"]["code"], "plan_unreadable")
        self.assertNotIn("suggested_command", env["error"])

    def test_unknown_strategy_names_the_listing_command(self):
        code, env = run_cli(["strategies", "no-such-rubric"])
        self.assertEqual(code, 1)
        self.assertEqual(env["error"]["code"], "unknown_strategy")
        self.assertTrue(env["error"]["recovery"])
        self.assertEqual(
            env["error"]["suggested_command"], "slopkeeper strategies list"
        )

    def test_corpus_changed_names_the_inventory_command(self):
        # The A1-era prose says "re-inventory"; an agent should not have to
        # read English to know what to run next.
        plan_path = write_plan(
            self.root, based_on={"corpus_fingerprint": "fp6:deadbeef:0"}
        )
        code, env = run_cli(["apply", str(plan_path), str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(env["error"]["code"], "corpus_changed")
        suggested = env["error"]["suggested_command"]
        self.assertTrue(suggested.startswith("slopkeeper inventory"))
        self.assertIn(str(self.root), suggested)

    def test_plan_no_based_on_names_the_inventory_command(self):
        plan_path = write_plan(self.root, based_on=None)
        code, env = run_cli(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 2)
        self.assertEqual(env["error"]["code"], "plan_no_based_on")
        self.assertIn("slopkeeper inventory", env["error"]["suggested_command"])

    def test_library_error_carries_the_field(self):
        err = SlopKeeperError("c", "m", "r", "slopkeeper doctor /tmp/x")
        self.assertEqual(err.as_dict()["suggested_command"], "slopkeeper doctor /tmp/x")
        self.assertNotIn("suggested_command", SlopKeeperError("c", "m", "r").as_dict())


class TestCitationScanTruncation(TempCase):
    """P5 (1): the 512 KiB per-file cap was the last silent skip."""

    def _huge(self, rel: str, cited: str) -> Path:
        # One file whose TAIL holds the only citation: read to the cap and
        # it is invisible, read whole and it is not. That is the case the
        # audit called "dropped in total silence".
        pad = "x" * (_PER_FILE_CAP + 2048)
        return write(self.root / rel, pad + f"\nsee {cited} for the rest\n")

    def test_refs_reports_the_cap_it_hit(self):
        self._huge("big.md", "acme_dossier.md")
        code, env = run_cli(["refs", str(self.root), "acme_dossier.md"])
        self.assertEqual(code, 0, env)
        self.assertEqual(env["data"]["truncated_files"], 1)
        self.assertIn("citation_scan_truncated", env["warnings"])

    def test_clean_corpus_says_nothing(self):
        code, env = run_cli(["refs", str(self.root), "acme_dossier.md"])
        self.assertEqual(code, 0, env)
        self.assertNotIn("truncated_files", env["data"])
        self.assertNotIn("warnings", env)

    def test_apply_dry_run_names_it_in_reference_scan(self):
        self._huge("big.md", "acme_dossier.md")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "big.md",
            ],
        )
        out = apply_plan(plan_path, self.root, execute=False)
        self.assertEqual(out["reference_scan"]["truncated_files"], 1)
        self.assertIn("citation_scan_truncated", out["warnings"])

    def test_dangling_survey_reports_it(self):
        # "No dangling citations" from a half-read file is a different claim.
        self._huge("big.md", "nowhere/at_all.md")
        code, env = run_cli(["refs", str(self.root), "--dangling"])
        self.assertEqual(code, 0, env)
        self.assertEqual(env["data"]["truncated_files"], 1)
        self.assertIn("citation_scan_truncated", env["warnings"])

    def test_doctor_reports_it(self):
        self._huge("big.md", "nowhere/at_all.md")
        code, env = run_cli(["doctor", str(self.root)])
        self.assertEqual(code, 0, env)
        self.assertIn("citation_scan_truncated", env["warnings"])
        self.assertEqual(env["data"]["dangling_citations"]["truncated_files"], 1)

    def test_the_gate_reports_it_too(self):
        # The human approving the render is the last line of defense; the
        # render has to admit its rewrite summary is a floor, not a census.
        self._huge("big.md", "acme_dossier.md")
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
                "big.md",
            ],
        )
        gate = self.tmp / "gate.md"
        code, env = run_cli(
            ["plan", "render", str(plan_path), str(self.root), "--out", str(gate)]
        )
        self.assertEqual(code, 0, env)
        self.assertEqual(env["data"]["truncated_files"], 1)
        self.assertIn("citation_scan_truncated", env["warnings"])


class TestPathMap(TempCase):
    """P5 (2): the canonical old→new map, on the three write envelopes."""

    def test_apply_executed_exposes_the_dict(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        self.assertEqual(
            out["path_map"],
            {
                "acme_dossier.md": "research/dossiers/acme_dossier.md",
                "globex_dossier.md": "research/dossiers/globex_dossier.md",
                "handoff-tp2.md": "spark-1.2.3/handoff-tp2.md",
                "ledger_wb.html": "archive/scraps/ledger_wb.html",
            },
        )
        # Same fact, two shapes: the ordered list is the journal's record,
        # the dict is the agent's lookup. They must agree.
        self.assertEqual(out["path_map"], {m["from"]: m["to"] for m in out["moved"]})

    def test_undo_exposes_the_inverse(self):
        plan_path = write_plan(self.root)
        out = apply_plan(plan_path, self.root, execute=True)
        back = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertEqual(
            back["path_map"], {new: old for old, new in out["path_map"].items()}
        )
        # Composing the two maps is the identity on every moved path —
        # which is the whole promise "undo puts it back".
        for old, new in out["path_map"].items():
            self.assertEqual(back["path_map"][new], old)

    def test_undo_blocked_move_is_not_claimed(self):
        # A restore that could not happen (something already sits at the
        # source path) is not a path change, so it must not appear in the
        # map an agent will trust.
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "acme_dossier.md",
                    "to": "research/acme_dossier.md",
                    "reason": "pile",
                }
            ],
            keep=[
                "globex_dossier.md",
                "handoff-tp2.md",
                "ledger_wb.html",
                "WATCHLIST.md",
                "docs",
                "lead-gen",
                "spark-1.2.3",
            ],
            partial=True,
        )
        out = apply_plan(plan_path, self.root, execute=True)
        write(self.root / "acme_dossier.md", "# occupier\n")
        back = undo_journal(Path(out["journal"]), self.root, execute=True)
        self.assertNotIn("research/acme_dossier.md", back["path_map"])
        self.assertEqual(
            [
                o["status"]
                for o in back["operations"]
                if o.get("src") == "acme_dossier.md"
            ],
            ["blocked_dst_occupied"],
        )

    def test_scaffold_maps_intended_to_actual(self):
        plan_path = write_plan(self.root)
        apply_plan(plan_path, self.root, execute=True)
        # An existing AGENTS.md the owner wrote: the never-clobber rule
        # sends the generated content to a sibling, and the agent needs to
        # know the file it asked for is NOT the file that holds its text.
        write(self.root / "research" / "dossiers" / "AGENTS.md", "# mine\n")
        out = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertEqual(
            out["path_map"],
            {
                "research/dossiers/AGENTS.md": (
                    "research/dossiers/AGENTS.md.slopkeeper-suggested"
                )
            },
        )
        # Writes that landed where asked are absent: a map of identical
        # pairs is noise beside the two entries that matter.
        self.assertNotIn("AGENTS.md", out["path_map"])

    def test_path_map_on_the_cli_envelope(self):
        plan_path = write_plan(self.root)
        code, env = run_cli(["apply", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, env)
        self.assertEqual(
            env["data"]["path_map"]["ledger_wb.html"], "archive/scraps/ledger_wb.html"
        )


class TestWarningsOnMutatingCommands(TempCase):
    """The envelope mirror must work for the writes, not only the reads."""

    def test_apply_execute_envelope_carries_payload_warnings(self):
        # --force is not needed to get here: the corpus moving between the
        # gate's digest and the first write is what B5 made loud, and P5
        # makes it loud at the ENVELOPE level too (an agent reading
        # out["warnings"] instead of digging into out["data"]).
        import unittest.mock as mock

        from slopkeeper import apply as apply_mod

        plan_path = write_plan(self.root)
        mutated = {"did": False}
        real_preflight = apply_mod._unwritable_targets

        def preflight_then_mutate(root, chains, rewrites, edits):
            if not mutated["did"]:
                mutated["did"] = True
                (root / "late_note.md").write_text(
                    "appeared mid-run\n", encoding="utf-8"
                )
            return real_preflight(root, chains, rewrites, edits)

        with mock.patch.object(
            apply_mod, "_unwritable_targets", side_effect=preflight_then_mutate
        ):
            code, env = run_cli(["apply", str(plan_path), str(self.root), "--execute"])
        self.assertEqual(code, 0, env)
        self.assertIn("fingerprint_moved_during_apply", env["warnings"])
        self.assertIn("fingerprint_moved_during_apply", env["data"]["warnings"])
