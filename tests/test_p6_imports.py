# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""P6 Slice A — import citations: detected, reported, NEVER rewritten.

The audit's framing is the whole test plan for this file. An import
statement is a citation (`import utils.helpers` names `utils/helpers.py`)
that the rewrite pass has never in this tool's history fixed, because a
dotted module path is not a path token. Before 0.3.0 a plan could move a
module out from under forty importers, report `ok: true`, and leave the
corpus unimportable — the reference-integrity promise simply did not
reach code.

So these tests pin three things:

  1. the detection resolves the four languages the audit names against
     the real corpus tree (module-path to tree-path mapping);
  2. every surface an agent reads — `refs`, `inventory`, `plan validate`,
     the human gate — says the same thing about the same edge;
  3. the rewrite engine is UNTOUCHED. The audit's acceptance criterion is
     "zero changes to the rewrite engine", and a test that an import line
     never appears in `would_rewrite` is what makes that a fact rather
     than an intention — it is the assertion most likely to be broken by a
     future well-meant "just rewrite imports too" change.

The acceptance fixture is a few hundred .py/.ts files, because a resolver
that works on three files and dies on a package tree is the bug this
slice exists to fix.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from helpers import TempCase, write, write_plan
from slopkeeper import cli
from slopkeeper.apply import apply_plan
from slopkeeper.inventory import collect
from slopkeeper.models import HIDDEN_SKIP_DIRS
from slopkeeper.plan import render_markdown
from slopkeeper.references import (
    CitationIndex,
    import_citation_lines,
    import_move_risks,
    import_sites,
)


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
    """Every top-level entry, so a one-file move passes coverage.

    `plan validate` refuses an incomplete plan, and these tests each move
    one file out of a shared fixture — so every top-level entry has to be
    named (a kept directory covers its subtree, including the files a move
    takes out of it). Derived from the disk, never hardcoded: a fixture
    that gains a directory must not silently start failing coverage.
    """
    return sorted(p.name for p in root.iterdir() if p.name not in HIDDEN_SKIP_DIRS)


# A small package-shaped corpus: one per language, plus the traps.
SMALL_CORPUS = {
    "utils/__init__.py": "",
    "utils/helpers.py": "def h():\n    return 1\n",
    "utils/text/__init__.py": "",
    "utils/text/slugify.py": "def slugify(s):\n    return s\n",
    "app/__init__.py": "",
    "app/main.py": (
        "import utils.helpers\n"
        "from utils.text import slugify\n"
        "from . import sibling\n"
        "import os, sys\n"
    ),
    "app/sibling.py": "X = 1\n",
    "scripts/load.py": "import importlib\nm = importlib.import_module('utils.text')\n",
    "web/comp.ts": "import {h} from '../utils/helper'\n",
    "utils/helper.ts": "export const h = 1\n",
    "web/index.ts": "import './comp'\nimport {a} from 'react'\n",
    "web/util.js": "const t = require('./deep')\n",
    "web/deep/index.js": "module.exports = 1\n",
    "cmd/main.go": (
        'package main\n\nimport (\n\t"fmt"\n\t"mymod/pkg/util"\n)\n\n'
        "func main() { fmt.Println(util.X) }\n"
    ),
    "pkg/util/util.go": "package util\n\nvar X = 1\n",
    "go.mod": "module mymod\n\ngo 1.21\n",
    "src/lib.rs": (
        "mod widgets;\npub use crate::widgets::button;\nuse serde::Serialize;\n"
    ),
    "src/widgets.rs": "pub struct W;\n",
    "src/widgets/button.rs": "pub struct B;\n",
    "notes.md": "the module utils.helpers is documented here\n",
}


class TestImportDetection(TempCase):
    """The statement grammars and the module-path to tree-path mapping."""

    def setUp(self):
        super().setUp()
        for rel, text in SMALL_CORPUS.items():
            write(self.root / rel, text)
        self.rels = rel_of(self.root)
        self.sites = import_sites(self.root, self.rels)["sites"]

    def _site(self, file: str, spec: str):
        for s in self.sites:
            if s["file"] == file and s["spec"] == spec:
                return s
        raise AssertionError(f"no site for {file} {spec!r}: {self.sites}")

    def test_python_dotted_import_resolves_to_the_file(self):
        self.assertEqual(
            self._site("app/main.py", "utils.helpers")["target"], "utils/helpers.py"
        )

    def test_from_import_prefers_the_most_specific_file(self):
        # `from utils.text import slugify` loads utils/text/slugify.py, not
        # the package's __init__.py. Preferring the module would attribute
        # the edge to the wrong file and price the wrong move.
        self.assertEqual(
            self._site("app/main.py", "utils.text")["target"], "utils/text/slugify.py"
        )

    def test_explicit_relative_import_resolves_beside_the_citer(self):
        self.assertEqual(self._site("app/main.py", ".")["target"], "app/sibling.py")

    def test_dynamic_import_module_is_a_citation(self):
        self.assertEqual(
            self._site("scripts/load.py", "utils.text")["target"],
            "utils/text/__init__.py",
        )

    def test_stdlib_is_unresolved_and_that_is_not_an_error(self):
        # The stats distinction matters to an agent: `resolved`/`unresolved`
        # counts the scan's reach; nothing here ever refuses a tend for an
        # unresolved import, because `import os` names no corpus file.
        os_site = self._site("app/main.py", "os")
        self.assertIsNone(os_site["target"])
        stats = import_sites(self.root, self.rels)["stats"]
        self.assertGreater(stats["unresolved"], 0)
        self.assertGreater(stats["resolved"], 0)
        self.assertEqual(stats["resolved"] + stats["unresolved"], stats["statements"])

    def test_js_relative_specifier_resolves_with_extension_fallback(self):
        self.assertEqual(
            self._site("web/comp.ts", "../utils/helper")["target"], "utils/helper.ts"
        )
        # A bare directory specifier takes the package index, the same
        # fallback node itself applies.
        self.assertEqual(
            self._site("web/util.js", "./deep")["target"], "web/deep/index.js"
        )

    def test_js_bare_specifier_is_never_a_corpus_edge(self):
        # `import {a} from 'react'` is a package. Resolving it against the
        # tree by basename would invent an edge to any react.* in the
        # corpus — so it is not even recorded as a statement.
        self.assertNotIn(
            "react", {s["spec"] for s in self.sites if s["file"] == "web/index.ts"}
        )

    def test_go_import_resolves_to_the_package_directory(self):
        # A go import names a DIRECTORY (every .go file in it is one unit),
        # through the go.mod module prefix.
        self.assertEqual(
            self._site("cmd/main.go", "mymod/pkg/util")["target"], "pkg/util"
        )
        self.assertIsNone(self._site("cmd/main.go", "fmt")["target"])

    def test_rust_mod_and_use_resolve_under_the_crate_root(self):
        self.assertEqual(
            self._site("src/lib.rs", "widgets")["target"], "src/widgets.rs"
        )
        self.assertEqual(
            self._site("src/lib.rs", "crate::widgets::button")["target"],
            "src/widgets/button.rs",
        )
        self.assertIsNone(self._site("src/lib.rs", "serde::Serialize")["target"])

    def test_markdown_is_not_scanned_for_imports(self):
        # `the module utils.helpers is documented here` is prose. The
        # citation scan owns markdown; this scan reads only the languages
        # whose statements it can parse.
        self.assertNotIn("notes.md", {s["file"] for s in self.sites})

    def test_resolution_is_memoized_per_statement(self):
        # A package imports the same module from everywhere; the resolver
        # caches, so a 300-file corpus does not re-walk the index 300 times.
        resolver_index = CitationIndex(self.rels)
        first = import_sites(self.root, self.rels)
        second = import_sites(self.root, self.rels, index=resolver_index)
        self.assertEqual(first["sites"], second["sites"])


class TestPackageTreeScale(TempCase):
    """The audit's acceptance fixture: a few hundred .py/.ts files."""

    def setUp(self):
        super().setUp()
        # 6 packages × 40 modules + a web tree: 300 files, each importing
        # its neighbours and a shared util. Written once per test — the
        # point is that the resolver survives a real tree, so the count is
        # the test.
        for pkg in range(6):
            write(self.root / f"pkg{pkg}/__init__.py", "")
            for mod in range(40):
                body = "import utils.helpers\n"
                if mod % 3 == 0:
                    body += f"from pkg{pkg} import mod{mod - 1 if mod else 0}\n"
                if mod % 5 == 0:
                    body += "from ..utils import helpers\n"
                write(self.root / f"pkg{pkg}/mod{mod}.py", body)
        write(self.root / "utils/__init__.py", "")
        write(self.root / "utils/helpers.py", "def h():\n    return 1\n")
        for n in range(60):
            write(
                self.root / f"web/components/c{n}.ts",
                "import {h} from '../../utils/helper'\n",
            )
        write(self.root / "utils/helper.ts", "export const h = 1\n")
        self.rels = rel_of(self.root)

    def test_tree_is_the_size_the_audit_asks_for(self):
        code_files = [p for p in self.rels if p.suffix in {".py", ".ts"}]
        self.assertGreaterEqual(len(code_files), 300)

    def test_every_importer_of_the_shared_util_is_found(self):
        sites = import_sites(self.root, self.rels)["sites"]
        helpers = [s for s in sites if s["target"] == "utils/helpers.py"]
        # Every one of the 240 pkg modules names it — by `import
        # utils.helpers`, and the mod%5 files additionally by
        # `from ..utils import helpers`, so the row count is above the
        # file count and the FILE count is the contract.
        self.assertEqual(len({s["file"] for s in helpers}), 240)
        self.assertGreater(len(helpers), 240)
        ts = [s for s in sites if s["target"] == "utils/helper.ts"]
        self.assertEqual(len({s["file"] for s in ts}), 60)

    def test_moving_the_shared_util_prices_every_importer(self):
        # The audit's acceptance sentence, literally: moving
        # utils/helpers.py with stale importers surfaces import_risk rows.
        rows = import_move_risks(
            self.root,
            {"moves": [{"from": "utils/helpers.py", "to": "lib/helpers.py"}]},
            self.rels,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["from"], "utils/helpers.py")
        self.assertEqual(row["to"], "lib/helpers.py")
        self.assertGreaterEqual(row["importer_count"], 200)
        self.assertGreater(row["stale"], 0)
        self.assertEqual(row["langs"], ["python"])
        self.assertTrue(row["evidence"])

    def test_scan_is_linear_not_quadratic(self):
        # A guard against the obvious regression: one pass over code files,
        # memoized resolution. This tree must scan in well under a second.
        import time

        start = time.monotonic()
        import_sites(self.root, self.rels)
        self.assertLess(time.monotonic() - start, 3.0)


class TestRiskSemantics(TempCase):
    """Does the statement still RESOLVE after the move? That is the test."""

    def setUp(self):
        super().setUp()
        for rel, text in SMALL_CORPUS.items():
            write(self.root / rel, text)
        self.rels = rel_of(self.root)

    def _rows(self, moves):
        return import_move_risks(self.root, {"moves": moves}, self.rels)

    def test_move_that_breaks_an_import_reports_it_stale(self):
        rows = self._rows([{"from": "utils/helpers.py", "to": "lib/helpers.py"}])
        self.assertEqual([r["from"] for r in rows], ["utils/helpers.py"])
        self.assertGreater(rows[0]["stale"], 0)
        self.assertIn("app/main.py", rows[0]["importers"])

    def test_move_that_re_targets_an_import_says_so_loudly(self):
        # The worse failure and the one nobody notices: the statement still
        # runs, and now means a DIFFERENT file.
        write(self.root / "utils/other.py", "def h():\n    return 2\n")
        self.rels = rel_of(self.root)
        rows = self._rows([{"from": "utils/helpers.py", "to": "lib/helpers.py"}])
        self.assertTrue(rows, "expected a risk row")
        self.assertTrue(
            any(r["retargeted"] for r in rows) or any(r["stale"] for r in rows)
        )

    def test_importer_moving_with_its_target_costs_nothing(self):
        # A directory renamed AROUND its importers keeps their relative
        # statements resolving: web/index.ts imports './comp' and both
        # files move together, so no statement breaks.
        rows = self._rows([{"from": "web", "to": "frontend/web"}])
        self.assertEqual(rows, [])

    def test_package_rename_prices_its_absolute_imports(self):
        # `utils/` → `lib/utils/` keeps the directory NAME but changes the
        # module path, so `import utils.helpers` is stale. Renaming is not
        # the only expensive move.
        rows = self._rows([{"from": "utils", "to": "lib/utils"}])
        self.assertTrue(rows)
        self.assertGreater(sum(r["stale"] for r in rows), 0)

    def test_move_of_an_unimported_file_reports_nothing(self):
        rows = self._rows([{"from": "notes.md", "to": "docs/notes.md"}])
        self.assertEqual(rows, [])

    def test_malformed_plan_is_not_this_reports_problem(self):
        # validate refuses a cycle; the gate helper must not raise on one.
        rows = import_move_risks(
            self.root,
            {"moves": [{"from": "a.md", "to": "b.md"}, {"from": "b.md", "to": "a.md"}]},
            self.rels,
        )
        self.assertEqual(rows, [])


class TestSurfaces(TempCase):
    """refs, inventory, validate and the gate all say the same thing."""

    def setUp(self):
        super().setUp()
        for rel, text in SMALL_CORPUS.items():
            write(self.root / rel, text)
        self.rels = rel_of(self.root)

    def test_refs_lists_import_rows_beside_citations(self):
        code, env = run_cli(["refs", str(self.root), "utils/helpers.py"])
        self.assertEqual(code, 0)
        data = env["data"]
        self.assertGreater(data["import_count"], 0)
        kinds = {row["kind"] for row in data["imports"]}
        self.assertEqual(kinds, {"import"})
        row = data["imports"][0]
        for key in ("file", "line", "text", "kind", "import_lang", "spec"):
            self.assertIn(key, row)
        # citation_count stays the CITATION count: adding imports to it
        # would silently change a documented P1 field (A5's honest refs).
        self.assertEqual(data["citation_count"], len(data["citations"]))

    def test_refs_names_a_directory_by_its_children(self):
        # refs <dir> aggregates imports of anything under it, the same
        # rule citation_lines applies to prose citations.
        rows = import_citation_lines(self.root, "utils", self.rels)
        self.assertTrue(rows)
        specs = {r["spec"] for r in rows}
        self.assertIn("utils.helpers", specs)

    def test_inventory_carries_a_separate_import_index(self):
        inv = collect(self.root, fast=False)
        self.assertIn("imports", inv)
        self.assertEqual(inv["imports"]["utils/helpers.py"][0]["file"], "app/main.py")
        stats = inv["stats"]["imports"]
        self.assertGreater(stats["statements"], 0)
        # The prose map must not have absorbed the import edges: an agent
        # reads "the pass rewrites citations" from `references`, and an
        # import there would be a lie about coverage.
        for target, citers in inv["references"].items():
            if target == "utils/helpers.py":
                self.assertNotIn("app/main.py", citers)

    def test_fast_inventory_skips_the_import_scan(self):
        inv = collect(self.root, fast=True)
        self.assertNotIn("imports", inv)

    def test_plan_validate_counts_import_risk_moves(self):
        plan_path = write_plan(
            self.root,
            moves=[
                {"from": "utils/helpers.py", "to": "lib/helpers.py", "reason": "lib"}
            ],
            keep=keep_all(self.root),
            folders=[],
        )
        code, env = run_cli(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertGreaterEqual(env["data"]["import_risk_moves"], 1)
        self.assertGreaterEqual(env["data"]["import_stale"], 1)
        self.assertIn("import_risk", env.get("warnings", []))

    def test_clean_plan_carries_no_import_warning(self):
        plan_path = write_plan(
            self.root,
            moves=[{"from": "notes.md", "to": "docs/notes.md", "reason": "docs"}],
            keep=keep_all(self.root),
            folders=[],
        )
        code, env = run_cli(["plan", "validate", str(plan_path), str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(env["data"]["import_risk_moves"], 0)
        self.assertNotIn("import_risk", env.get("warnings", []))

    def test_gate_renders_the_import_section(self):
        from slopkeeper.plan import based_on_binding  # noqa: F401  (shape check)

        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(self.root),
            "strategy": "hybrid",
            "based_on": {"corpus_fingerprint": "fp6:x"},
            "moves": [{"from": "utils/helpers.py", "to": "lib/helpers.py"}],
            "keep": [],
            "unresolved": [],
            "folders": [],
        }
        md = render_markdown(
            plan,
            import_risks=import_move_risks(self.root, plan, self.rels),
        )
        self.assertIn("Import citations", md)
        self.assertIn("NEVER edits an import", md)
        self.assertIn("utils.helpers", md)

    def test_gate_omits_the_section_when_nothing_breaks(self):
        plan = {
            "schema": "slopkeeper/plan@1",
            "corpus_root": str(self.root),
            "moves": [{"from": "notes.md", "to": "docs/notes.md"}],
        }
        md = render_markdown(
            plan, import_risks=import_move_risks(self.root, plan, self.rels)
        )
        self.assertNotIn("Import citations", md)


class TestNeverRewritten(TempCase):
    """The audit's hard line: zero changes to the rewrite engine.

    These rows are the acceptance criterion for "NEVER rewritten". If a
    future change wires the import scan into `plan_rewrites` — a
    well-meant "just fix the imports too" — these fail, and they should:
    editing a module path safely is AST work, and the core is stdlib-only
    by design (AGENT_GUIDE: agent decides, tool guarantees).
    """

    def setUp(self):
        super().setUp()
        for rel, text in SMALL_CORPUS.items():
            write(self.root / rel, text)
        write(self.root / "docs/guide.md", "see utils/helpers.py for the helper\n")

    def test_import_lines_never_appear_in_would_rewrite(self):
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "utils/helpers.py",
                    "to": "lib/utils/helpers.py",
                    "reason": "layer",
                }
            ],
            keep=keep_all(self.root),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=False)
        rewritten_files = {row["file"] for row in data.get("would_rewrite", [])}
        # The prose citation in docs/guide.md IS rewritten...
        self.assertTrue(any("guide.md" in f for f in rewritten_files))
        # ...and no importer is.
        for f in rewritten_files:
            self.assertNotIn(
                f,
                {
                    "app/main.py",
                    "scripts/load.py",
                    "web/comp.ts",
                    "web/index.ts",
                    "web/util.js",
                    "cmd/main.go",
                    "src/lib.rs",
                },
            )
        # And the import statements are still verbatim on disk (a dry run
        # writes nothing, so this also proves the scan is read-only).
        self.assertIn("import utils.helpers", (self.root / "app/main.py").read_text())

    def test_execute_moves_the_file_and_leaves_the_import_stale(self):
        # The honest outcome: the tool tells you, in the gate, that this
        # will break. It does not pretend to fix it.
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "utils/helpers.py",
                    "to": "lib/utils/helpers.py",
                    "reason": "layer",
                }
            ],
            keep=keep_all(self.root),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"])
        self.assertIn("import utils.helpers", (self.root / "app/main.py").read_text())
        self.assertFalse((self.root / "utils/helpers.py").exists())

    def test_a_hand_authored_edit_is_how_you_fix_one(self):
        # The escape hatch the gate section points at: an ordinary edit,
        # exactly-once validated, is how an import gets updated.
        plan_path = write_plan(
            self.root,
            moves=[
                {
                    "from": "utils/helpers.py",
                    "to": "lib/utils/helpers.py",
                    "reason": "layer",
                }
            ],
            edits=[
                {
                    "file": "app/main.py",
                    "find": "import utils.helpers",
                    "replace": "import lib.utils.helpers",
                    "reason": "module path moved",
                }
            ],
            keep=keep_all(self.root),
            folders=[],
        )
        data = apply_plan(plan_path, self.root, execute=True)
        self.assertFalse(data["errors"])
        self.assertIn(
            "import lib.utils.helpers", (self.root / "app/main.py").read_text()
        )
