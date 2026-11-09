# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The roots registry: which corpora you tend, and where to find them.

Tending is recurring and rarely a single corpus. The registry is the
list of corpus roots a scheduled job should walk, so the loop lives here
instead of an external shell script. It lives at
``$XDG_CONFIG_HOME/slopkeeper/roots.json`` (default ``~/.config/``);
``SLOPKEEPER_HOME`` overrides the whole base for tests and dotfile
managers. Content: schema tag plus sorted, unique, absolute paths.

Like the manifest, a corrupt or foreign-schema registry fails loudly
with a delete-and-rebuild hint — never silently reinterpreted, never
silently repaired. Zero dependencies; no LLM, no network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import SlopKeeperError

ROOTS_NAME = "roots.json"
ROOTS_SCHEMA = "slopkeeper/roots@1"


def roots_path() -> Path:
    base = os.environ.get("SLOPKEEPER_HOME")
    if base:
        return Path(base).expanduser() / ROOTS_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg).expanduser() / "slopkeeper" / ROOTS_NAME


def load_roots() -> list[str]:
    p = roots_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise SlopKeeperError(
            "roots_unreadable",
            f"roots registry at {p} is unreadable: {e}",
            "Delete it and re-add your corpora with `slopkeeper roots add <path>`.",
        )
    if not isinstance(data, dict) or data.get("schema") != ROOTS_SCHEMA:
        raise SlopKeeperError(
            "roots_bad_schema",
            f"roots registry schema {data.get('schema')!r} != {ROOTS_SCHEMA!r}",
            "Delete it and re-add your corpora with `slopkeeper roots add <path>`.",
        )
    # A non-list `roots` ({"roots": "not-a-list"}) would iterate
    # characters and hand them back as registered corpora — refuse the
    # same loud way as a foreign schema.
    registry = data.get("roots", [])
    if not isinstance(registry, list) or not all(isinstance(r, str) for r in registry):
        raise SlopKeeperError(
            "roots_bad_schema",
            "roots registry 'roots' field must be a list of strings, "
            f"got {type(registry).__name__}",
            "Delete it and re-add your corpora with `slopkeeper roots add <path>`.",
        )
    return list(registry)


def save_roots(roots: list[str]) -> Path:
    p = roots_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"schema": ROOTS_SCHEMA, "roots": sorted(set(roots))}, indent=1),
        encoding="utf-8",
    )
    return p


def add_root(path: str | Path) -> tuple[list[str], bool]:
    """Register a corpus root (idempotent). Returns (roots, was_added)."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise SlopKeeperError(
            "root_not_dir",
            f"{p} is not an existing directory",
            "Register only corpora that exist — SlopKeeper tends "
            "directories, it does not create them.",
        )
    roots = load_roots()
    if str(p) in roots:
        return roots, False
    roots = sorted(set(roots) | {str(p)})
    save_roots(roots)
    return roots, True


def remove_root(path: str | Path) -> list[str]:
    """Unregister a corpus root. The corpus itself is never touched."""
    target = str(Path(path).expanduser().resolve())
    roots = load_roots()
    if target not in roots:
        raise SlopKeeperError(
            "root_not_registered",
            f"{target} is not in the roots registry",
            "Check `slopkeeper roots list` for the registered paths.",
        )
    roots = [r for r in roots if r != target]
    save_roots(roots)
    return roots
