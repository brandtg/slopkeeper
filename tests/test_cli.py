# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""cli.py — exit codes, envelope shape, --out (subprocess).

Subprocess-level CLI cases per TEST_PLAN.md's cli section.
"""

# --- subprocess-level CLI contract (TEST_PLAN "cli.py (currently 0%)") ---
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from helpers import TempCase, build_corpus, write, write_plan

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "slopkeeper", *args],
        capture_output=True,
        text=True,
        env=CLI_ENV,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


class TestInventoryShapes(TempCase):
    """B11: --fields and --out/--summary shape the same payload."""

    def test_fields_projection_is_lean_and_titled(self):
        out = self.tmp / "lean.json"
        r = run_cli(
            "inventory",
            str(self.root),
            "--fields",
            "path,title,mtime",
            "--out",
            str(out),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        inv = json.loads(out.read_text())
        self.assertTrue(all(set(f) == {"path", "title", "mtime"} for f in inv["files"]))
        # titles are available without opening anything
        titled = {f["path"]: f["title"] for f in inv["files"]}
        self.assertIn("acme_dossier.md", titled)
        self.assertIn("Acme dossier", titled["acme_dossier.md"])
        # stdout carries counts + where the body is, never the dump
        stdout_payload = json.loads(r.stdout)["data"]
        self.assertEqual(stdout_payload["out_path"], str(out))
        self.assertNotIn("files", stdout_payload)

    def test_unknown_field_refused(self):
        r = run_cli("inventory", str(self.root), "--fields", "path,nope")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("inventory_bad_fields", r.stdout)

    def test_fields_projection_shapes_the_whole_document(self):
        # C7: dirs, clusters and references are dropped from the
        # projection (they doubled a 630-file corpus's output), and the
        # file is written compact — the lean view is actually lean.
        out = self.tmp / "lean.json"
        r = run_cli(
            "inventory",
            str(self.root),
            "--fast",
            "--fields",
            "path,title,mtime",
            "--out",
            str(out),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        inv = json.loads(out.read_text())
        self.assertNotIn("dirs", inv)
        self.assertNotIn("clusters", inv)
        self.assertNotIn("references", inv)
        # the fingerprint and stats survive (plan authoring needs them)
        self.assertIn("fingerprint", inv)
        self.assertIn("stats", inv)
        # compact JSON: no indentation newlines inside a file entry
        first_line = out.read_text().splitlines()[0]
        self.assertIn('"path"', first_line)
        self.assertNotIn('\n  "path"', out.read_text())

    def test_out_follows_summary(self):
        # --out used to ignore --summary and ship the full 440KB doc.
        out = self.tmp / "inv.json"
        r = run_cli("inventory", str(self.root), "--summary", "--out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        inv = json.loads(out.read_text())
        self.assertTrue(all(isinstance(f, str) for f in inv["files"]))
        self.assertNotIn(
            "title",
            inv["files"][0]
            if inv["files"] and isinstance(inv["files"][0], dict)
            else {},
        )


class TestRewritesOut(TempCase):
    """B1: the full line-level rewrite detail as JSON."""

    def test_apply_dry_run_rewrites_out(self):
        plan_path = write_plan(self.root)
        out = self.tmp / "rewrites.json"
        r = run_cli("apply", str(plan_path), str(self.root), "--rewrites-out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        detail = json.loads(out.read_text())
        self.assertIn("WATCHLIST.md", detail)
        self.assertEqual(
            detail["WATCHLIST.md"],
            [
                {
                    "line": 3,
                    "old": "acme_dossier.md",
                    "new": "research/dossiers/acme_dossier.md",
                },
                {
                    "line": 3,
                    "old": "globex_dossier.md",
                    "new": "research/dossiers/globex_dossier.md",
                },
            ],
        )

    def test_plan_render_rewrites_out(self):
        plan_path = write_plan(self.root)
        out = self.tmp / "rw.json"
        r = run_cli(
            "plan",
            "render",
            str(plan_path),
            str(self.root),
            "--out",
            str(self.tmp / "gate.md"),
            "--rewrites-out",
            str(out),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        detail = json.loads(out.read_text())
        self.assertIn("WATCHLIST.md", detail)
        self.assertIn("old", detail["WATCHLIST.md"][0])


class TestCLI(TempCase):
    def test_plan_schema_prints_bare_template(self):
        # T6: `slopkeeper plan schema > plan.json` must yield a plan skeleton —
        # the bare template JSON, NOT an envelope around it (the envelope
        # form failed plan_bad_schema; DESIGN §2 documents the exception
        # on plan render's markdown precedent).
        r = run_cli("plan", "schema")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["schema"], "slopkeeper/plan@1")
        self.assertIn("moves", out)
        self.assertNotIn("data", out)  # not an envelope

    def test_plan_schema_output_validates_as_plan(self):
        # The quickstart's step, end to end: schema → redirect to a file
        # → set corpus_root + the inventory fingerprint → one real move
        # → `slopkeeper plan validate` accepts it. Before T6 the schema command
        # printed an envelope, so plan.json failed plan_bad_schema.
        template = json.loads(run_cli("plan", "schema").stdout)
        inv = json.loads(run_cli("inventory", str(self.root), "--summary").stdout)
        template["corpus_root"] = str(self.root)
        template["based_on"] = {
            "corpus_fingerprint": inv["data"]["fingerprint"],
            "inventory_at": "2026-11-03T00:00:00Z",
        }
        template["moves"] = [
            {
                "from": "acme_dossier.md",
                "to": "research/dossiers/acme_dossier.md",
                "reason": "dossier research",
            }
        ]
        template["edits"] = []
        template["folders"] = []
        template["unresolved"] = []
        template["keep"] = [
            "WATCHLIST.md",
            "globex_dossier.md",
            "handoff-tp2.md",
            "ledger_wb.html",
            "spark-glm-5.3",
            "lead-gen",
            "docs",
        ]
        plan = self.tmp / "plan.json"
        plan.write_text(json.dumps(template), encoding="utf-8")
        r = run_cli("plan", "validate", str(plan), str(self.root))
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertTrue(json.loads(r.stdout)["data"]["valid"])

    def test_strategies_show_hybrid(self):
        # Both documented forms work: `strategies <id>` and
        # `strategies show <id>` (the doc syntax; parser accepts an
        # optional literal "show" before the id). Same envelope either way.
        r = run_cli("strategies", "hybrid")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["command"], "strategies.show")
        self.assertEqual(out["data"]["id"], "hybrid")
        r2 = run_cli("strategies", "show", "hybrid")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(json.loads(r2.stdout), out)

    def test_unknown_strategy_clean_error(self):
        r = run_cli("strategies", "no-such-strategy")
        self.assertEqual(r.returncode, 1)  # EXIT_ERROR for non-plan_ codes
        out = json.loads(r.stdout)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "unknown_strategy")

    def test_unknown_command_exits_nonzero(self):
        r = run_cli("definitely-not-a-command")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage", r.stderr)  # argparse refuses before any JSON

    def test_plan_validate_bad_plan_exits_nonzero_with_code(self):
        bad = self.tmp / "bad-plan.json"
        bad.write_text(
            json.dumps(
                {
                    "schema": "slopkeeper/plan@0",
                    "corpus_root": str(self.root),
                    "moves": [
                        {
                            "from": "acme_dossier.md",
                            "to": "research/dossiers/acme_dossier.md",
                        }
                    ],
                }
            )
        )
        r = run_cli("plan", "validate", str(bad), str(self.root))
        self.assertEqual(r.returncode, 2)  # EXIT_VALIDATION for plan_* codes
        self.assertIn("plan_bad_schema", r.stdout)
        out = json.loads(r.stdout)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "plan_bad_schema")

    def test_plan_validate_out_writes_full_payload(self):
        # Phase 8: counts on stdout, the rule-expanded move list and the
        # full straggler report in the --out JSON (the house envelope
        # convention — an expanded move list is unbounded).
        build_corpus(
            self.root,
            {"reports/_org/a.md": "a\n", "reports/_org/old.txt": "old\n"},
        )
        plan_path = write_plan(
            self.root, rules=[{"match": "reports/_org/*.md", "to": "reports/org"}]
        )
        out_path = self.tmp / "validated.json"
        r = run_cli(
            "plan", "validate", str(plan_path), str(self.root), "--out", str(out_path)
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        self.assertEqual(data["moves"], 5)  # 4 explicit + 1 expanded
        self.assertEqual(data["straggler_dirs"], 1)
        self.assertEqual(data["straggler_files"], 1)
        self.assertEqual(data["rules"][0]["expanded"], 1)
        full = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(len(full["moves"]), 5)
        self.assertEqual(
            full["stragglers"],
            [
                {
                    "dir": "reports/_org",
                    "count": 1,
                    "remaining": ["reports/_org/old.txt"],
                }
            ],
        )

    def test_refs_dangling_reports_broken_citation(self):
        # Phase 11: path-ish citations resolving to nothing, as data —
        # already-broken links the rewrite pass must never touch.
        write(self.root / "guide.md", "see docs/gone.md and [[missing/link]]\n")
        r = run_cli("refs", str(self.root), "--dangling")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        self.assertEqual(data["mode"], "dangling")
        self.assertEqual(data["count"], 2)
        tokens = {s["token"] for s in data["sites"]}
        self.assertIn("docs/gone.md", tokens)
        self.assertIn("missing/link", tokens)
        # a valid citation is not dangling
        self.assertNotIn("acme_dossier.md", tokens)
        # and refs without a basename (and without --dangling) refuses
        r2 = run_cli("refs", str(self.root))
        self.assertEqual(r2.returncode, 1)
        self.assertEqual(json.loads(r2.stdout)["error"]["code"], "no_basename")


class TestPhase12RefsNonexistentPath(TempCase):
    """D7: a slashed name that exists nowhere returns only exact
    citations of the string — never a same-basename file's."""

    def _corpus_with_sync(self):
        write(self.root / "reports/_sync/2026-09-15.md", "# sync\n")

    def test_refs_nonexistent_slashed_path_no_fallback(self):
        self._corpus_with_sync()
        r = run_cli("refs", str(self.root), "reports/_product/2026-09-15.md")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        self.assertTrue(data["nonexistent_path"])
        self.assertEqual(data["citation_count"], 0)
        # The basename fallback must not return the _sync file's cites.
        self.assertEqual(data["citations"], [])

    def test_refs_nonexistent_path_lists_exact_citations(self):
        self._corpus_with_sync()
        write(
            self.root / "w.md",
            "stale: reports/_product/2026-09-15.md (pre-move)\n",
        )
        r = run_cli("refs", str(self.root), "reports/_product/2026-09-15.md")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        self.assertTrue(data["nonexistent_path"])
        self.assertEqual(data["citation_count"], 1)
        self.assertEqual(data["citations"][0]["file"], "w.md")
        self.assertEqual(data["citations"][0]["line"], 1)

    def test_refs_dangling_lists_removed_path(self):
        # The dangling scan (informational) still lists citations of the
        # removed path — consistent with refs' exact result.
        self._corpus_with_sync()
        write(self.root / "w.md", "stale: reports/_product/2026-09-15.md\n")
        r = run_cli("refs", str(self.root), "--dangling")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        tokens = {s["token"] for s in data["sites"]}
        self.assertIn("reports/_product/2026-09-15.md", tokens)


class TestPhase13DanglingSurface(TempCase):
    """D10: --dangling unbounded by default, --limit/--prefix/--per-file
    honored, slash-words filtered."""

    def _dangling_corpus(self, n=250):
        # One file, n dangling tokens (extension-bearing, unique) —
        # per-file cap raised via --per-file so they all surface.
        lines = "".join(f"see gone/dir{i}/x.md here\n" for i in range(n))
        write(self.root / "w.md", lines)
        write(self.root / "keep.md", "fine\n")

    def test_dangling_default_unbounded(self):
        self._dangling_corpus(250)
        r = run_cli("refs", str(self.root), "--dangling", "--per-file", "300")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        self.assertEqual(data["count"], 250, "the old 200 cap is gone")

    def test_dangling_limit_caps(self):
        self._dangling_corpus(250)
        r = run_cli(
            "refs", str(self.root), "--dangling", "--per-file", "300", "--limit", "50"
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["data"]["count"], 50)

    def test_dangling_per_file_respected(self):
        self._dangling_corpus(10)
        r = run_cli("refs", str(self.root), "--dangling")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["data"]["count"], 5)

    def test_dangling_prefix_filters(self):
        write(self.root / "w.md", "a reports/_org/x.md b misc/y.md c\n")
        r = run_cli("refs", str(self.root), "--dangling", "--prefix", "reports/_org")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        tokens = [s["token"] for s in data["sites"]]
        self.assertIn("reports/_org/x.md", tokens)
        self.assertNotIn("misc/y.md", tokens)

    def test_dangling_ignores_slash_words(self):
        write(
            self.root / "w.md",
            "effort: engineering/product Easy/Medium/Hard; "
            "real: reports/_org (removed dir)\n",
        )
        write(self.root / "reports/keep.md", "# keep\n")
        r = run_cli("refs", str(self.root), "--dangling")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)["data"]
        tokens = [s["token"] for s in data["sites"]]
        self.assertNotIn("engineering/product", tokens)
        self.assertNotIn("Easy/Medium/Hard", tokens)
        self.assertIn("reports/_org", tokens)


class TestPropose(TempCase):
    """Phase 10: the layout fork as data (read-only, subprocess)."""

    def test_propose_candidates_and_counts(self):
        r = run_cli("propose", str(self.root))
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["command"], "propose")
        cands = out["data"]["candidates"]
        self.assertGreaterEqual(len(cands), 2)
        for c in cands:
            for key in (
                "move_count",
                "citation_break_count",
                "code_anchor_count",
                "unresolved_estimate",
            ):
                self.assertIn(key, c)
            self.assertTrue(c["strategy"])  # each candidate names its strategy

    def test_propose_read_only(self):
        # No .slopkeeper/ writes, no fingerprint effect, no mtimes moved.
        from slopkeeper.models import corpus_fingerprint

        def mtimes():
            return {
                str(p): os.stat(p).st_mtime for p in self.root.rglob("*") if p.is_file()
            }

        fp = corpus_fingerprint(self.root)
        mt = mtimes()
        r = run_cli("propose", str(self.root))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.root / ".slopkeeper").exists())
        self.assertEqual(corpus_fingerprint(self.root), fp)
        self.assertEqual(mtimes(), mt)

    def test_propose_consistent_with_risk_scan(self):
        # For a fixed candidate, citation_break_count equals the
        # apply-time reference_risks count of the equivalent plan —
        # one break rule, two surfaces.
        from slopkeeper.propose import candidate_moves

        write(self.root / "proj/README.md", "# proj\n")
        write(self.root / "proj/NOTES.md", "see acme_dossier.md\n")
        r = run_cli("propose", str(self.root))
        hybrid = next(
            c for c in json.loads(r.stdout)["data"]["candidates"] if c["id"] == "hybrid"
        )
        moved = candidate_moves(self.root, "hybrid")
        self.assertEqual(len(moved), hybrid["move_count"])
        plan_path = write_plan(
            self.root,
            partial=True,
            moves=[
                {"from": p, "to": f"{d}/{Path(p).name}", "reason": "t"}
                for p, d in sorted(moved.items())
            ],
        )
        r = run_cli("apply", str(plan_path), str(self.root))
        self.assertEqual(r.returncode, 0, r.stderr)
        risks = json.loads(r.stdout)["data"]["reference_risks"]
        self.assertEqual(len(risks), hybrid["citation_break_count"])
        self.assertNotIn("acme_dossier.md", risks)
        self.assertGreaterEqual(hybrid["citation_rewrites"], 1)

    def test_propose_counts_include_external(self):
        # A skills store outside the corpus cites acme: the count is
        # blind to it until the dir is declared, then it appears. The
        # citation itself is verifiable with refs --scan-path (the
        # guide's verification tool). apply's reference_risks stays
        # in-corpus by design (Phase 5: declared-dir citations are
        # rewrite-pass work, not risk rows) — propose counts MORE, on
        # purpose: a fork count that ignores skills answers wrong.
        from slopkeeper.propose import candidate_moves

        ext = self.tmp / "ext"
        write(ext / "skill.md", "see acme_dossier.md\n")
        r = run_cli("propose", str(self.root))
        hybrid = next(
            c for c in json.loads(r.stdout)["data"]["candidates"] if c["id"] == "hybrid"
        )
        r2 = run_cli("propose", str(self.root), "--external-root", str(ext))
        hybrid_ext = next(
            c
            for c in json.loads(r2.stdout)["data"]["candidates"]
            if c["id"] == "hybrid"
        )
        self.assertGreater(
            hybrid_ext["citation_break_count"], hybrid["citation_break_count"]
        )
        # the equivalent plan declares the root; the rewrite pass
        # covers the external citation (strict-references holds)
        moved = candidate_moves(self.root, "hybrid")
        plan_path = write_plan(
            self.root,
            partial=True,
            moves=[
                {"from": p, "to": f"{d}/{Path(p).name}", "reason": "t"}
                for p, d in sorted(moved.items())
            ],
            external_roots=[{"path": str(ext), "expect": 1}],
        )
        r3 = run_cli("apply", str(plan_path), str(self.root))
        self.assertEqual(r3.returncode, 0, r3.stdout)
        # and refs surfaces the exact citation the count saw
        r4 = run_cli("refs", str(self.root), "acme_dossier.md", "--scan-path", str(ext))
        sites = json.loads(r4.stdout)["data"]["citations"]
        self.assertTrue(any(s.get("external") for s in sites))

    def test_propose_bad_citing_dir_refused(self):
        r = run_cli("propose", str(self.root), "--scan-path", str(self.tmp / "nope"))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(r.stdout)["error"]["code"], "scan_path_missing")
        git_dir = self.root / ".git"
        git_dir.mkdir()
        r = run_cli("propose", str(self.root), "--scan-path", str(git_dir))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(r.stdout)["error"]["code"], "scan_path_skip_list")


class TestStalenessGate(TempCase):
    """T1: the staleness gate must fire for template-authored plans.

    Drives the real CLI end-to-end — `slopkeeper plan schema` supplies the plan
    body (never a hand-built dict), which is edited into a plan for the
    fixture corpus and applied. A stale fingerprint must raise
    `corpus_changed` under BOTH key spellings: the template's canonical
    `corpus_fingerprint` and the read-only `inventory_sha256` alias
    (plans authored from the pre-fix template carry it). Before the fix
    the gate read only `corpus_fingerprint`, so an `inventory_sha256`
    plan silently proceeded.
    """

    def _schema_template(self) -> dict:
        r = run_cli("plan", "schema")
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)  # T6: the bare template, no envelope

    def _apply_stale(self, based_on: dict) -> subprocess.CompletedProcess:
        template = self._schema_template()
        template["corpus_root"] = str(self.root)
        template["based_on"] = based_on
        template["moves"] = [
            {
                "from": "acme_dossier.md",
                "to": "research/dossiers/acme_dossier.md",
                "reason": "dossier research",
            }
        ]
        template["edits"] = []
        template["folders"] = []
        template["unresolved"] = []
        template["keep"] = [
            "WATCHLIST.md",
            "globex_dossier.md",
            "handoff-tp2.md",
            "ledger_wb.html",
            "spark-glm-5.3",
            "lead-gen",
            "docs",
        ]
        plan = self.tmp / "plan.json"
        plan.write_text(json.dumps(template), encoding="utf-8")
        return run_cli("apply", str(plan), str(self.root))

    def _assert_corpus_changed(self, r: subprocess.CompletedProcess):
        self.assertEqual(r.returncode, 1, r.stdout)  # EXIT_ERROR: corpus_changed
        env = json.loads(r.stdout)
        self.assertFalse(env["ok"])
        self.assertEqual(env["error"]["code"], "corpus_changed")

    def test_template_authored_plan_is_gated(self):
        # The template ships its fingerprint placeholder under whatever
        # key it currently documents; fill THAT key with a stale value
        # (as an agent following step 5 would) and apply. The plan is
        # otherwise valid, so corpus_changed is the only possible refusal.
        template = self._schema_template()
        for key in ("corpus_fingerprint", "inventory_sha256"):
            if key in template.get("based_on", {}):
                template["based_on"][key] = "stale"
        r = self._apply_stale(template["based_on"])
        self._assert_corpus_changed(r)

    def test_apply_gates_inventory_sha256_alias(self):
        # Plans in the wild carry the old key; apply must treat it as a
        # read-only alias of corpus_fingerprint and gate on it too.
        r = self._apply_stale(
            {"inventory_sha256": "stale", "inventory_at": "2026-01-01T00:00:00Z"}
        )
        self._assert_corpus_changed(r)


class TestNextCommandsAreRunnable(TempCase):
    """T6: every next.command must be paste-runnable.

    The envelope's promise is that the calling agent never has to guess
    what to run next — but the hints omitted the required <root>
    positional (following them exited 2), and several described the next
    action in prose without naming the corpus at all. Parametrized over
    every root-taking subcommand, each against a fresh fixture corpus:
    the emitted next.command must contain the exact root path the
    command was invoked with. Following every hint verbatim must error
    nowhere.
    """

    def fresh_root(self, name: str) -> Path:
        root = build_corpus(self.tmp / name / "corpus")
        return root

    def run_ok(self, *args: str) -> dict:
        r = run_cli(*args)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.loads(r.stdout)

    def assert_root_in_next(self, root: Path, env: dict):
        command = env["next"]["command"]
        self.assertIn(
            str(root.resolve()),
            command,
            f"next.command not runnable — missing <root>: {command}",
        )

    def test_next_command_contains_invoked_root(self):
        # Read-only passes: hint names the root it was invoked with.
        for cmd in ("doctor", "inventory", "clusters"):
            with self.subTest(command=cmd):
                root = self.fresh_root(cmd)
                env = self.run_ok(cmd, str(root))
                self.assert_root_in_next(root, env)

        with self.subTest(command="refs"):
            root = self.fresh_root("refs")
            env = self.run_ok("refs", str(root), "acme_dossier.md")
            self.assert_root_in_next(root, env)

        # status (healthy branch): needs a recorded manifest first. Also
        # pins the partial-wording fix: delta plans carry "partial": true
        # in the plan JSON — the old hint pointed at a --allow-partial
        # flag the plan never carries.
        with self.subTest(command="status"):
            root = self.fresh_root("status")
            self.run_ok("status", str(root), "--record")
            env = self.run_ok("status", str(root))
            self.assert_root_in_next(root, env)
            self.assertNotIn("--allow-partial", env["next"]["hint"])
            self.assertIn("partial", env["next"]["hint"])

        # draft with nothing new → slopkeeper status <root>
        with self.subTest(command="draft"):
            root = self.fresh_root("draft")
            self.run_ok("status", str(root), "--record")
            env = self.run_ok(
                "draft", str(root), "--out", str(self.tmp / "draft-none.json")
            )
            self.assert_root_in_next(root, env)

        # draft with new slop → slopkeeper plan validate <out> <root>
        with self.subTest(command="draft-with-slop"):
            root = self.fresh_root("draft-slop")
            self.run_ok("status", str(root), "--record")
            write(root / "globex_family_notes.md", "# Globex family\n\nmarket share.\n")
            env = self.run_ok(
                "draft", str(root), "--out", str(self.tmp / "draft-slop.json")
            )
            self.assertEqual(env["next"]["action"], "plan.validate")
            self.assert_root_in_next(root, env)

        # plan validate → slopkeeper plan render <plan> <root>
        with self.subTest(command="plan validate"):
            root = self.fresh_root("validate")
            plan = write_plan(root)
            env = self.run_ok("plan", "validate", str(plan), str(root))
            self.assert_root_in_next(root, env)

        # apply dry-run → slopkeeper apply <plan> <root> --execute
        with self.subTest(command="apply-dry"):
            root = self.fresh_root("apply-dry")
            plan = write_plan(root)
            env = self.run_ok("apply", str(plan), str(root))
            self.assert_root_in_next(root, env)

        # apply --execute → slopkeeper scaffold <plan> <root> --execute
        with self.subTest(command="apply-executed"):
            root = self.fresh_root("apply-exec")
            plan = write_plan(root)
            env = self.run_ok("apply", str(plan), str(root), "--execute")
            self.assert_root_in_next(root, env)

        # scaffold dry-run → slopkeeper scaffold <plan> <root> --execute
        with self.subTest(command="scaffold-dry"):
            root = self.fresh_root("scaffold-dry")
            plan = write_plan(root)
            env = self.run_ok("scaffold", str(plan), str(root))
            self.assert_root_in_next(root, env)


class TestEnvelopeCrossCutting(TempCase):
    def test_readonly_commands_return_command_and_data(self):
        # Cross-cutting: every JSON command answers {command, data} — the
        # agent-facing envelope is API. Two-plus read-only commands through
        # a real subprocess. (plan schema is exempt since T6: its output
        # is the bare template artifact, not an envelope.)
        cases = [
            (("strategies",), "strategies.list"),
            (("strategies", "hybrid"), "strategies.show"),
            (("doctor", str(self.root)), "doctor"),
        ]
        for args, command in cases:
            with self.subTest(args=args):
                r = run_cli(*args)
                self.assertEqual(r.returncode, 0, r.stderr)
                out = json.loads(r.stdout)
                self.assertIn("command", out)
                self.assertIn("data", out)
                self.assertEqual(out["command"], command)
                self.assertTrue(out["ok"])


class TestGuide(unittest.TestCase):
    # `slopkeeper guide` — the agent-facing distribution surface: the guide must
    # ride inside the package so pipx/venv installs carry it without a
    # checkout. Raw markdown like `plan schema` (the output IS the
    # artifact), never an envelope.

    def test_guide_prints_the_agent_guide(self):
        r = run_cli("guide")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("# AGENT_GUIDE — how to drive SlopKeeper", r.stdout)
        self.assertIn("Hard rules", r.stdout)

    def test_guide_is_raw_markdown_not_envelope(self):
        r = run_cli("guide")
        self.assertEqual(r.returncode, 0, r.stderr)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(r.stdout)

    def test_guide_matches_the_repo_canonical_copy(self):
        root_guide = REPO_ROOT / "AGENT_GUIDE.md"
        if not root_guide.exists():
            self.skipTest("not a repo checkout")
        r = run_cli("guide")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, root_guide.read_text())


class TestRefsScanPathOverrides(TempCase):
    """Phase 6: --scan-path overrides — ignored dirs scan, skip lists report."""

    def test_scan_path_ignored_dir_named(self):
        # Explicitly naming an ignored dir is the override: it IS
        # scanned, its hits carry "external": true (as today), and the
        # per-dir result marks it "ignored": true with a hit count —
        # never silently merged into "no citations".
        (self.root / ".slopkeeperignore").write_text(".cache/\n", encoding="utf-8")
        write(self.root / ".cache" / "skill.md", "Read acme_dossier.md first.\n")
        r = run_cli(
            "refs",
            str(self.root),
            "acme_dossier.md",
            "--scan-path",
            str(self.root / ".cache"),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertTrue(out["ok"])
        ext = [c for c in out["data"]["citations"] if c.get("external")]
        self.assertEqual(
            [c["file"] for c in ext], [str(self.root / ".cache" / "skill.md")]
        )
        self.assertEqual(
            out["data"]["scan_path_results"],
            [
                {
                    "path": str(self.root / ".cache"),
                    "status": "scanned",
                    "ignored": True,
                    "hits": 1,
                }
            ],
        )

    def test_scan_path_skip_list_reported(self):
        # A named skip-list dir is NEVER read — but the result carries a
        # per-dir row saying so, with the code, so an empty result is
        # never mistaken for "no citations".
        (self.root / ".git").mkdir()
        write(self.root / ".git" / "config.md", "acme_dossier.md\n")
        r = run_cli(
            "refs",
            str(self.root),
            "acme_dossier.md",
            "--scan-path",
            str(self.root / ".git"),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertTrue(out["ok"])
        self.assertEqual(
            out["data"]["scan_path_results"],
            [
                {
                    "path": str(self.root / ".git"),
                    "status": "skip_list",
                    "code": "scan_path_skip_list",
                }
            ],
        )
        # Nothing was read: no external citations, no scanned dirs.
        self.assertEqual(out["data"]["scan_paths"], [])
        self.assertFalse(any(c.get("external") for c in out["data"]["citations"]))
        self.assertGreater(out["data"]["citation_count"], 0)  # in-tree still counts


class TestMoveCommand(TempCase):
    """Phase 7: slopkeeper move — gate, journal, integrity count."""

    def test_move_gate_then_execute(self):
        # No --execute: the gate renders, nothing moves.
        r = run_cli(
            "move", "acme_dossier.md", "research/acme_dossier.md", str(self.root)
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("# Scaffolding plan", r.stdout)
        self.assertIn("`acme_dossier.md`", r.stdout)
        self.assertIn("slopkeeper move", r.stdout)  # footer names the real next
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research").exists())

        # --execute: exactly one journaled move, integrity match.
        r = run_cli(
            "move",
            "acme_dossier.md",
            "research/acme_dossier.md",
            str(self.root),
            "--execute",
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["command"], "move")
        self.assertEqual(out["data"]["moved_count"], 1)
        self.assertEqual(
            out["data"]["integrity"],
            {"src_files": 1, "dst_files": 1, "match": True},
        )
        self.assertIn(str(self.root.resolve()), out["next"]["command"])
        journal_ops = [
            json.loads(line)
            for line in Path(out["data"]["journal"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        moves = [e for e in journal_ops if e.get("op") == "move"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["src"], "acme_dossier.md")
        self.assertEqual(moves[0]["dst"], "research/acme_dossier.md")
        self.assertTrue((self.root / "research" / "acme_dossier.md").exists())
        # The generated plan is materialized for the audit trail.
        self.assertTrue(Path(out["data"]["plan"]).exists())

    def test_move_undo_restores(self):
        r = run_cli(
            "move",
            "acme_dossier.md",
            "research/acme_dossier.md",
            str(self.root),
            "--execute",
        )
        out = json.loads(r.stdout)
        r = run_cli("undo", out["data"]["journal"], str(self.root), "--execute")
        self.assertEqual(r.returncode, 0, r.stderr)
        undo = json.loads(r.stdout)
        self.assertIn("restored", [o["status"] for o in undo["data"]["operations"]])
        self.assertTrue((self.root / "acme_dossier.md").exists())
        self.assertFalse((self.root / "research" / "acme_dossier.md").exists())

    def test_move_count_check(self):
        # A directory move: src_files == dst_files, counted by a raw
        # subtree walk on both sides — the `find | wc -l` replacement.
        for i in range(3):
            write(self.root / "docs" / f"d{i}.md", "# d\n")
        r = run_cli("move", "docs", "ref", str(self.root), "--execute")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(
            out["data"]["integrity"],
            {"src_files": 4, "dst_files": 4, "match": True},
        )
        self.assertEqual(
            sorted(p.name for p in (self.root / "ref").iterdir() if p.is_file()),
            ["canonical.md", "d0.md", "d1.md", "d2.md"],
        )

    def test_move_refuses_ignored_source(self):
        (self.root / ".slopkeeperignore").write_text("secret/\n", encoding="utf-8")
        write(self.root / "secret" / "x.md", "x")
        r = run_cli("move", "secret", "public", str(self.root))
        self.assertEqual(r.returncode, 1, r.stdout)
        out = json.loads(r.stdout)
        self.assertEqual(out["error"]["code"], "move_source_ignored")
        self.assertTrue((self.root / "secret" / "x.md").exists())

    def test_move_refuses_existing_destination(self):
        write(self.root / "taken.md", "# taken\n")
        r = run_cli("move", "acme_dossier.md", "taken.md", str(self.root), "--execute")
        self.assertEqual(r.returncode, 1, r.stdout)
        out = json.loads(r.stdout)
        self.assertEqual(out["error"]["code"], "dst_exists")
        self.assertTrue((self.root / "acme_dossier.md").exists())


if __name__ == "__main__":
    unittest.main()
