# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Agent skill installation (`slopkeeper skill install`).

Copies the packaged SKILL.md into an agent's skill directory — by
well-known agent name, or by explicit directory for agents this module
doesn't know. The packaged copy is canonical (src/slopkeeper/SKILL.md
symlinks the repo's skills/ dir, on AGENT_GUIDE.md's precedent), so an
install always matches the binary it ships with.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

from .models import SlopKeeperError

SKILL_NAME = "slopkeeper"


def _opencode_skills_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "opencode" / "skills"


# Well-known agents. Hermes nests skills under a category directory;
# everything else is a flat skills dir (each README example, §Agent setup).
AGENT_SKILLS_DIRS: dict[str, callable] = {
    "claude": lambda: Path.home() / ".claude" / "skills",
    "opencode": _opencode_skills_dir,
    "hermes": lambda: Path.home() / ".hermes" / "skills" / "software-development",
}


def packaged_skill_text() -> str:
    """The SKILL.md riding inside the installed package."""
    return (
        importlib.resources.files("slopkeeper")
        .joinpath("SKILL.md")
        .read_text(encoding="utf-8")
    )


def _resolve_skills_dir(raw: str) -> Path:
    """One target → the skills directory to install into.

    Known agent names win over same-named local directories; anything
    else must look like a path (has a separator, is dot-relative, or
    exists), so a typo'd agent name errors loudly instead of silently
    creating `./<typo>/slopkeeper/`.
    """
    if raw in AGENT_SKILLS_DIRS:
        return AGENT_SKILLS_DIRS[raw]()
    candidate = Path(raw).expanduser()
    looks_like_path = (
        os.sep in raw
        or (os.altsep is not None and os.altsep in raw)
        or raw.startswith(".")
        or raw == "~"
        or candidate.exists()
    )
    if not looks_like_path:
        raise SlopKeeperError(
            "skill_unknown_agent",
            f"unknown agent: {raw}",
            f"Known agents: {', '.join(sorted(AGENT_SKILLS_DIRS))}. "
            "Or pass a skills directory path to install into "
            "(the skill lands in <dir>/slopkeeper/).",
        )
    return candidate.resolve()


def install(targets: list[str], force: bool = False) -> list[dict]:
    """Copy the packaged skill into each target's skill directory.

    Idempotent: an already-present skill is left untouched with a
    warning — agents may re-run this on every session and must not
    clobber a skill the human has since edited. `--force` is the
    explicit refresh. Each result reports what happened.
    """
    text = packaged_skill_text()
    results = []
    for raw in targets:
        skills_dir = _resolve_skills_dir(raw)
        dest = skills_dir / SKILL_NAME
        if dest.exists() and not dest.is_dir():
            raise SlopKeeperError(
                "skill_target_not_dir",
                f"install destination is a file, not a directory: {dest}",
                "Remove it or pass a different skills directory.",
            )
        skill_file = dest / "SKILL.md"
        record: dict = {
            "target": raw,
            "agent": raw if raw in AGENT_SKILLS_DIRS else None,
            "skill_dir": str(dest),
        }
        existed = skill_file.exists()
        if existed and not force:
            record["action"] = "skipped"
            record["warning"] = (
                f"skill already present at {skill_file}; nothing written "
                "(pass --force to overwrite)"
            )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(text, encoding="utf-8")
            record["action"] = "overwritten" if existed else "installed"
        results.append(record)
    return results
