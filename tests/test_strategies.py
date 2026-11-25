# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""strategies.py — the strategy packs, and who is allowed to write one.

The first test is the shipped contract; the rest is P5's B10 work: four
search directories with override-by-id, the origin annotation that makes an
override legible, and load-time shape validation (the audit's
`folder_template` KeyError surfacing as `strategy_bad_pack` instead).

Every test that touches the environment isolates it: `$XDG_CONFIG_HOME` and
`$SLOPKEEPER_HOME` are pointed at temp dirs, so a developer's own rubrics
cannot make the shipped-pack assertion false and a test's pack cannot leak
into a neighbor.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from helpers import TempCase, write_plan
from slopkeeper.models import SlopKeeperError
from slopkeeper.strategies import (
    get_strategy,
    list_strategies,
    pack_dirs,
)

PACK = {
    "id": "acme",
    "name": "Acme house rules",
    "summary": "Everything under a year folder, code in scripts/.",
    "when_to_use": ["The owner thinks in years."],
    "partitioning_rubric": ["One folder per year.", "Code goes in scripts/."],
    "example": {"2026": "this year's active pile"},
    "folder_template": {
        "kind": "case",
        "state_md": False,
        "agents_md_skeleton": "# <Name>\n\n## Role / scope\n<one paragraph>",
    },
}


def _write_pack(directory: Path, doc: dict, name: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or doc['id']}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@contextmanager
def isolated_env(**overrides):
    """Point the two pack env vars at temp dirs (or unset them)."""
    saved = {k: os.environ.get(k) for k in ("XDG_CONFIG_HOME", "SLOPKEEPER_HOME")}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestStrategies(unittest.TestCase):
    def test_all_packages_load(self):
        with isolated_env(XDG_CONFIG_HOME=tempfile.mkdtemp(), SLOPKEEPER_HOME=None):
            ids = [s["id"] for s in list_strategies()]
            # by-kind (Phase 10): the mechanical baseline propose compares
            # against — a real rubric pack, not a propose-only special case.
            self.assertEqual(
                set(ids), {"role-bots", "topics", "projects", "hybrid", "by-kind"}
            )
            for sid in ids:
                pack = get_strategy(sid)
                self.assertIn("partitioning_rubric", pack)
                self.assertIn("agents_md_skeleton", pack["folder_template"])
                # P5: provenance rides on every pack, shipped ones too.
                self.assertEqual(pack["origin"], "packaged")
                self.assertTrue(Path(pack["source"]).is_file())


class TestPackDiscovery(TempCase):
    """B10: four directories, later wins, override by id."""

    def setUp(self):
        super().setUp()
        self.xdg = self.tmp / "config"
        self.home = self.tmp / "keeper-home"
        self.corpus_strategies = self.root / ".slopkeeper" / "strategies"

    def _env(self):
        return isolated_env(XDG_CONFIG_HOME=self.xdg, SLOPKEEPER_HOME=self.home)

    def test_load_order_is_packaged_then_xdg_then_home_then_corpus(self):
        with self._env():
            origins = [origin for origin, _ in pack_dirs(self.root)]
        self.assertEqual(origins, ["packaged", "user", "user", "corpus"])
        with self._env():
            paths = [str(d) for _, d in pack_dirs(self.root)]
        self.assertEqual(paths[1], str(self.xdg / "slopkeeper" / "strategies"))
        self.assertEqual(paths[2], str(self.home / "strategies"))
        self.assertEqual(paths[3], str(self.corpus_strategies))

    def test_a_new_user_pack_is_visible_everywhere(self):
        _write_pack(self.xdg / "slopkeeper" / "strategies", PACK)
        with self._env():
            rows = {s["id"]: s for s in list_strategies(self.root)}
        self.assertIn("acme", rows)
        self.assertEqual(rows["acme"]["origin"], "user")
        self.assertEqual(
            rows["acme"]["source"],
            str(self.xdg / "slopkeeper" / "strategies" / "acme.json"),
        )
        with self._env():
            pack = get_strategy("acme", self.root)
        self.assertEqual(pack["name"], "Acme house rules")
        self.assertEqual(pack["origin"], "user")

    def test_a_user_pack_overrides_the_shipped_one_by_id(self):
        # The point of the feature: edit the rubric under the shipped id and
        # every surface that reads the pack sees your text.
        mine = dict(PACK, id="hybrid", name="Hybrid (house flavour)")
        _write_pack(self.xdg / "slopkeeper" / "strategies", mine)
        with self._env():
            rows = {s["id"]: s for s in list_strategies(self.root)}
            pack = get_strategy("hybrid", self.root)
        self.assertEqual(rows["hybrid"]["origin"], "user")
        self.assertEqual(pack["name"], "Hybrid (house flavour)")
        # The other shipped packs are untouched.
        self.assertEqual(rows["by-kind"]["origin"], "packaged")

    def test_the_last_directory_wins(self):
        # Corpus beats $SLOPKEEPER_HOME beats $XDG_CONFIG_HOME beats shipped.
        _write_pack(self.xdg / "slopkeeper" / "strategies", dict(PACK, name="from xdg"))
        _write_pack(self.home / "strategies", dict(PACK, name="from home"))
        _write_pack(self.corpus_strategies, dict(PACK, name="from corpus"))
        with self._env():
            pack = get_strategy("acme", self.root)
        self.assertEqual(pack["name"], "from corpus")
        self.assertEqual(pack["origin"], "corpus")
        # Without a root, the corpus pack is out of the running.
        with self._env():
            pack = get_strategy("acme")
        self.assertEqual(pack["name"], "from home")

    def test_slopkeeper_home_is_optional(self):
        with isolated_env(XDG_CONFIG_HOME=self.xdg, SLOPKEEPER_HOME=None):
            dirs = [d for _, d in pack_dirs(self.root)]
        self.assertNotIn(self.home / "strategies", dirs)
        # ...and an unset XDG_CONFIG_HOME falls back to ~/.config, not a crash.
        with isolated_env(XDG_CONFIG_HOME=None, SLOPKEEPER_HOME=None):
            dirs = [str(d) for _, d in pack_dirs(None)]
        self.assertIn(str(Path.home() / ".config" / "slopkeeper" / "strategies"), dirs)

    def test_missing_directories_are_not_errors(self):
        with isolated_env(
            XDG_CONFIG_HOME=self.tmp / "nope", SLOPKEEPER_HOME=self.tmp / "also-nope"
        ):
            self.assertTrue(any(s["origin"] == "packaged" for s in list_strategies()))

    def test_unknown_strategy_names_the_available_ids(self):
        with isolated_env(XDG_CONFIG_HOME=self.xdg, SLOPKEEPER_HOME=None):
            with self.assertRaises(SlopKeeperError) as cm:
                get_strategy("nope")
        self.assertEqual(cm.exception.code, "unknown_strategy")
        self.assertIn("hybrid", cm.exception.recovery)
        self.assertEqual(cm.exception.suggested_command, "slopkeeper strategies list")

    def test_traversal_id_refuses(self):
        # The id can come from a plan's `strategy` key (an LLM wrote it) and
        # it is joined onto a directory: `../../etc/passwd` must never be
        # opened, and it must refuse as "no such strategy", not as a crash.
        for bad in ("../../etc/passwd", "../hybrid", "/abs/hybrid", "a/b", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(SlopKeeperError) as cm:
                    get_strategy(bad)
                self.assertEqual(cm.exception.code, "unknown_strategy")


class TestBadPacks(TempCase):
    """B10's second half: shape validated at LOAD, not at scaffold."""

    def setUp(self):
        super().setUp()
        self.xdg = self.tmp / "config" / "slopkeeper" / "strategies"

    def _expect_bad(self, doc: dict, fragment: str, name: str = "acme"):
        _write_pack(self.xdg, doc, name=name)
        with isolated_env(XDG_CONFIG_HOME=self.tmp / "config", SLOPKEEPER_HOME=None):
            with self.assertRaises(SlopKeeperError) as cm:
                get_strategy(name)
        self.assertEqual(cm.exception.code, "strategy_bad_pack")
        self.assertIn(fragment, cm.exception.message)
        return cm.exception

    def test_missing_folder_template_is_a_load_error(self):
        # The audit's exact case: this used to die in scaffold.py as a raw
        # KeyError → the `internal` envelope, no recovery, no file named.
        doc = {k: v for k, v in PACK.items() if k != "folder_template"}
        err = self._expect_bad(doc, "folder_template")
        self.assertIn("acme.json", err.message)
        self.assertTrue(err.recovery)

    def test_missing_template_keys_are_named(self):
        for key in ("kind", "agents_md_skeleton"):
            template = dict(PACK["folder_template"])
            template.pop(key)
            with self.subTest(key=key):
                self._expect_bad(dict(PACK, folder_template=template), key)

    def test_other_required_keys(self):
        for key in ("id", "name", "summary", "partitioning_rubric"):
            doc = {k: v for k, v in PACK.items() if k != key}
            with self.subTest(key=key):
                self._expect_bad(doc, key)

    def test_wrong_types_and_empty_values(self):
        cases = [
            (dict(PACK, partitioning_rubric="just a string"), "list"),
            (dict(PACK, partitioning_rubric=[]), "empty"),
            (dict(PACK, partitioning_rubric=["ok", "  "]), "non-empty strings"),
            (dict(PACK, summary="   "), "empty"),
            (dict(PACK, folder_template="not an object"), "str"),
            (
                dict(
                    PACK, folder_template=dict(PACK["folder_template"], state_md="yes")
                ),
                "boolean",
            ),
        ]
        for doc, fragment in cases:
            with self.subTest(fragment=fragment):
                self._expect_bad(doc, fragment)

    def test_unparseable_json_names_the_file(self):
        self.xdg.mkdir(parents=True, exist_ok=True)
        (self.xdg / "acme.json").write_text("{not json", encoding="utf-8")
        with isolated_env(XDG_CONFIG_HOME=self.tmp / "config", SLOPKEEPER_HOME=None):
            with self.assertRaises(SlopKeeperError) as cm:
                get_strategy("acme")
        self.assertEqual(cm.exception.code, "strategy_bad_pack")
        self.assertIn("acme.json", cm.exception.message)

    def test_id_must_match_the_filename(self):
        # An id that disagrees with its filename shadows the wrong pack.
        _write_pack(self.xdg, dict(PACK, id="hybrid"), name="acme")
        with isolated_env(XDG_CONFIG_HOME=self.tmp / "config", SLOPKEEPER_HOME=None):
            with self.assertRaises(SlopKeeperError) as cm:
                get_strategy("acme")
        self.assertEqual(cm.exception.code, "strategy_bad_pack")
        self.assertIn("does not match the filename", cm.exception.message)

    def test_one_bad_pack_does_not_hide_the_others_silently(self):
        # Loud is the point: a broken file in the search path fails the
        # command rather than vanishing from the list and leaving the user
        # wondering whether their edit took.
        _write_pack(self.xdg, PACK)
        (self.xdg / "broken.json").write_text("{", encoding="utf-8")
        with isolated_env(XDG_CONFIG_HOME=self.tmp / "config", SLOPKEEPER_HOME=None):
            with self.assertRaises(SlopKeeperError) as cm:
                list_strategies()
        self.assertEqual(cm.exception.code, "strategy_bad_pack")
        self.assertIn("broken.json", cm.exception.message)


class TestPacksReachTheConsumers(TempCase):
    """A rubric nobody renders is a rubric nobody wrote."""

    def test_scaffold_renders_from_the_corpus_pack(self):
        # The per-corpus dir is the one that travels with the corpus, so a
        # household convention lives IN the corpus and needs no release.
        mine = dict(
            PACK,
            id="acme",
            folder_template={
                "kind": "case",
                "state_md": False,
                "agents_md_skeleton": "# <Name>\n\n## Role / scope\n<one paragraph>",
            },
        )
        _write_pack(self.root / ".slopkeeper" / "strategies", mine)
        plan_path = write_plan(
            self.root,
            strategy="acme",
            folders=[
                {
                    "path": "research/dossiers",
                    "kind": "case",
                    "role": "Dossiers.",
                    "state_md": False,
                }
            ],
        )
        from slopkeeper.scaffold import scaffold_from_plan

        with isolated_env(XDG_CONFIG_HOME=self.tmp / "empty", SLOPKEEPER_HOME=None):
            out = scaffold_from_plan(plan_path, self.root, execute=True)
        self.assertEqual(out["mode"], "executed")
        written = (self.root / "research" / "dossiers" / "AGENTS.md").read_text()
        self.assertIn("# dossiers", written)
        self.assertIn("Dossiers.", written)

    def test_unknown_strategy_in_a_plan_refuses_scaffold(self):
        plan_path = write_plan(self.root, strategy="nope")
        from slopkeeper.scaffold import scaffold_from_plan

        with self.assertRaises(SlopKeeperError) as cm:
            scaffold_from_plan(plan_path, self.root, execute=False)
        self.assertEqual(cm.exception.code, "unknown_strategy")


class TestStrategiesCli(TempCase):
    def _run(self, *argv):
        from slopkeeper import cli
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, json.loads(buf.getvalue())

    def test_list_annotates_origin(self):
        code, env = self._run("strategies", "list")
        self.assertEqual(code, 0)
        self.assertEqual(env["command"], "strategies.list")
        self.assertTrue(all("origin" in row for row in env["data"]))
        self.assertTrue(all(row["origin"] == "packaged" for row in env["data"]))

    def test_dirs_names_the_search_path(self):
        code, env = self._run("strategies", "dirs", "--root", str(self.root))
        self.assertEqual(code, 0)
        self.assertEqual(env["command"], "strategies.dirs")
        self.assertEqual(
            [d["origin"] for d in env["data"]["dirs"]],
            ["packaged", "user", "corpus"]
            if not os.environ.get("SLOPKEEPER_HOME")
            else ["packaged", "user", "user", "corpus"],
        )
        self.assertEqual(env["data"]["root"], str(self.root))

    def test_show_with_root_carries_the_hint(self):
        code, env = self._run("strategies", "show", "hybrid", "--root", str(self.root))
        self.assertEqual(code, 0)
        self.assertEqual(env["command"], "strategies.show")
        self.assertEqual(env["data"]["origin"], "packaged")
        self.assertIn("plan validate", env["next"]["command"])
        self.assertIn(str(self.root), env["next"]["command"])

    def test_show_without_root_still_chains(self):
        # No corpus named: the honest next step is `strategies dirs`, not a
        # command with a placeholder root in it (every success path
        # chains — `next_step`'s promise, closed for `undo` in 0.2.10).
        code, env = self._run("strategies", "show", "hybrid")
        self.assertEqual(code, 0)
        self.assertEqual(env["next"]["action"], "strategies.dirs")
        self.assertNotIn("<root>", env["next"]["command"])


if __name__ == "__main__":
    unittest.main()
