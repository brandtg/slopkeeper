# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Partitioning strategies: rubric packs the calling agent applies.

A strategy is not code — it's a rubric the agent reads, plus folder templates
the scaffold step renders. Ships as JSON so agents can read it directly.

Pack discovery (P5, audit B10). Before 0.2.10 the packaged directory was the
only place a pack could live, so encoding a household's conventions meant
editing the installed package — a release cycle between an owner and their
own rubric. Four places are now searched, LATER WINS, and a pack overrides
another by its `id`:

1. packaged (`slopkeeper/strategies/*.json`) — what ships;
2. `$XDG_CONFIG_HOME/slopkeeper/strategies/*.json` (`~/.config/...` when the
   variable is unset) — the user's own rubrics;
3. `$SLOPKEEPER_HOME/strategies/*.json` — for people who keep slopkeeper
   state in one home directory rather than in XDG;
4. `<root>/.slopkeeper/strategies/*.json` — a per-corpus rubric that travels
   with the corpus it describes (and is never walked or fingerprinted:
   `.slopkeeper` is tool layer, not content).

An override is the point, not a side effect: a household that wants
`by-kind` to file `.ipynb` under `notebooks/` edits the rubric text and the
folder template under id `by-kind`, and every surface (propose's comparison,
scaffold's rendering, `strategies show`) picks it up with no release.

Shape is validated at load (B10's second half). A pack missing
`folder_template` used to surface three modules later as a raw `KeyError` in
`scaffold.py` — the `internal` envelope, no recovery string, and the agent
none the wiser about which file was wrong. `strategy_bad_pack` names the
file and the key instead, at the moment the pack is read.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .models import SlopKeeperError

# The packaged pack dir (what `uv build` ships as package-data).
_STRATEGY_DIR = Path(__file__).parent / "strategies"

# A strategy id is a single, boring filename stem. `get_strategy` joins the
# caller's string onto a directory, and that string can come from a plan's
# `strategy` key — i.e. from an LLM — so traversal (`../evil`) is refused
# here rather than by the filesystem. Dots are allowed (a pack may be
# `acme.by-kind.json`) but never as the leading character.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Keys a pack MUST carry, with the type each must have. `folder_template` is
# required because scaffold indexes into it (`template["kind"]`,
# `template["agents_md_skeleton"]`); requiring it here is what turns that
# late KeyError into a load-time error that names the file.
_REQUIRED: dict[str, type] = {
    "id": str,
    "name": str,
    "summary": str,
    "partitioning_rubric": list,
    "folder_template": dict,
}
# Keys the templates read, and the type they must have when present.
_TEMPLATE_REQUIRED: dict[str, type] = {"kind": str, "agents_md_skeleton": str}


def _bad_pack(path: Path, problem: str) -> SlopKeeperError:
    return SlopKeeperError(
        "strategy_bad_pack",
        f"strategy pack {path} is not usable: {problem}",
        "A pack is a JSON object with id, name, summary, partitioning_rubric"
        " (a non-empty list of strings) and folder_template (an object"
        " carrying kind and agents_md_skeleton). Fix the file, or delete it to"
        " fall back to the shipped rubric; `slopkeeper strategies list` shows"
        " every pack that did load, and `strategies dirs` names the search"
        " directories.",
        "slopkeeper strategies dirs",
    )


def _load_pack(path: Path) -> dict:
    """Read and validate one pack file (B10: loud at load, not at scaffold)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise _bad_pack(path, f"it cannot be read ({e})") from None
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise _bad_pack(path, f"it is not valid JSON ({e})") from None
    if not isinstance(doc, dict):
        raise _bad_pack(path, "the top level is not a JSON object")
    for key, typ in _REQUIRED.items():
        if key not in doc:
            raise _bad_pack(path, f"missing required key {key!r}")
        if not isinstance(doc[key], typ):
            raise _bad_pack(
                path, f"{key!r} is {type(doc[key]).__name__}, expected {typ.__name__}"
            )
        if isinstance(doc[key], str) and not doc[key].strip():
            raise _bad_pack(path, f"{key!r} is empty")
        if isinstance(doc[key], list):
            if not doc[key]:
                raise _bad_pack(path, f"{key!r} is empty")
            if not all(isinstance(x, str) and x.strip() for x in doc[key]):
                raise _bad_pack(path, f"{key!r} must be a list of non-empty strings")
    template = doc["folder_template"]
    for key, typ in _TEMPLATE_REQUIRED.items():
        if key not in template:
            raise _bad_pack(path, f"folder_template is missing {key!r}")
        if not isinstance(template[key], typ):
            raise _bad_pack(
                path,
                f"folder_template {key!r} is {type(template[key]).__name__},"
                f" expected {typ.__name__}",
            )
    if "state_md" in template and not isinstance(template["state_md"], bool):
        raise _bad_pack(path, "folder_template 'state_md' must be a boolean")
    sid = doc["id"]
    if not _ID_RE.match(sid):
        raise _bad_pack(path, f"id {sid!r} is not a single safe filename stem")
    if sid != path.stem:
        raise _bad_pack(path, f"id {sid!r} does not match the filename {path.stem!r}")
    return doc


def pack_dirs(root: Path | None = None) -> list[tuple[str, Path]]:
    """Every pack dir in load order: (origin, dir), lowest precedence first.

    A missing directory is simply not a search dir — an unset
    `$SLOPKEEPER_HOME` and a corpus with no `.slopkeeper/strategies` are the
    normal case, never an error. Exposed for `strategies dirs`, for tests,
    and for anyone debugging why their pack did not appear.
    """
    dirs: list[tuple[str, Path]] = [("packaged", _STRATEGY_DIR)]
    config_home = os.environ.get("XDG_CONFIG_HOME") or ""
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    dirs.append(("user", base / "slopkeeper" / "strategies"))
    keeper_home = os.environ.get("SLOPKEEPER_HOME")
    if keeper_home:
        dirs.append(("user", Path(keeper_home).expanduser() / "strategies"))
    if root is not None:
        dirs.append(("corpus", Path(root) / ".slopkeeper" / "strategies"))
    return dirs


def resolve_packs(root: Path | None = None) -> dict[str, dict]:
    """{id: pack} across every search dir, LATER DIR WINS (override by id).

    Each pack carries two provenance keys the shipped packs never had:
    `origin` ("packaged" | "user" | "corpus") and `source` (the file it was
    read from). They ride on `strategies show` too, so an agent can tell the
    owner's rubric from the shipped one without guessing.
    """
    merged: dict[str, dict] = {}
    for origin, directory in pack_dirs(root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            doc = _load_pack(path)
            doc["origin"] = origin
            doc["source"] = str(path)
            merged[doc["id"]] = doc
    return merged


def list_strategies(root: Path | None = None) -> list[dict]:
    """Every visible pack, id-sorted, with its origin (B10).

    The origin annotation is what makes an override legible: a household
    that shadowed `by-kind` sees `user` beside it instead of wondering
    whether their edit took effect.
    """
    return [
        {
            "id": d["id"],
            "name": d["name"],
            "summary": d["summary"],
            "origin": d["origin"],
            "source": d["source"],
        }
        for d in sorted(resolve_packs(root).values(), key=lambda d: d["id"])
    ]


def get_strategy(strategy_id: str, root: Path | None = None) -> dict:
    """One pack by id, user override first, `unknown_strategy` when absent.

    `root` (the corpus being tended) enables the per-corpus dir; omit it for
    the machine-wide view. The packaged dir is always the fallback, so
    deleting a bad user pack restores the shipped rubric.
    """
    if not isinstance(strategy_id, str) or not _ID_RE.match(strategy_id):
        # Not a filesystem name at all: the same refusal as an unknown id
        # (it genuinely names no pack), with the id rule in the recovery.
        available = [s["id"] for s in list_strategies(root)]
        raise SlopKeeperError(
            "unknown_strategy",
            f"no strategy {strategy_id!r}",
            f"A strategy id is one name of letters, digits, '.', '-' or '_'."
            f" Available: {available}.",
            "slopkeeper strategies list",
        )
    pack = resolve_packs(root).get(strategy_id)
    if pack is None:
        raise SlopKeeperError(
            "unknown_strategy",
            f"no strategy {strategy_id!r}",
            f"Available: {[s['id'] for s in list_strategies(root)]}. A pack is"
            " <id>.json in $XDG_CONFIG_HOME/slopkeeper/strategies,"
            " $SLOPKEEPER_HOME/strategies, or <root>/.slopkeeper/strategies"
            " (see `slopkeeper strategies dirs`).",
            "slopkeeper strategies list",
        )
    return pack
