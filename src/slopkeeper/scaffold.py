# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Scaffold step: turn an approved plan into the folder-bot layer.

Writes, per folder entry: AGENTS.md (role/scope/boundaries) and optionally
STATE.md. Writes the root roster AGENTS.md. NEVER overwrites an existing
file — writes a `*.slopkeeper-suggested` sibling instead, so the tool can
run repeatedly on a living corpus without clobbering the owner's edits.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import add_warning, corpus_path, safe_relpath, tend_lock
from .plan import load_plan
from .status import write_manifest
from .strategies import get_strategy

ROOT_ROSTER_HEADER = """# Corpus roster

This folder is organized as folder-bots. Each directory below is a unit of
work with its own AGENTS.md (read it before working inside). Chats are
scratch; files are memory.

| folder | kind | role |
|---|---|---|
"""


def scaffold_from_plan(
    plan_path: Path,
    corpus_root: Path,
    execute: bool = False,
    strategy_id: str | None = None,
) -> dict:
    """Write the folder-bot layer (execute mode holds the tend lock).

    Dry-run is a pure read: no lock. Execute wraps the file writes and
    the manifest snapshot in .slopkeeper/tend.lock so a concurrent tend
    cannot interleave. A platform with no flock writes anyway and says
    so on the envelope — `warnings: ["no_locking_available"]` (B7):
    scaffold's idempotence promise ("run it twice, never clobber") is
    weaker without serialization, and a silent no-op lock made the
    payload identical to a protected run.
    """
    root = corpus_root.resolve()
    if not execute:
        return _scaffold_from_plan(
            plan_path, root, execute=False, strategy_id=strategy_id
        )
    with tend_lock(root) as lock:
        data = _scaffold_from_plan(
            plan_path, root, execute=True, strategy_id=strategy_id
        )
    if lock.unavailable:
        add_warning(data, "no_locking_available")
    return data


def _scaffold_from_plan(
    plan_path: Path,
    corpus_root: Path,
    execute: bool = False,
    strategy_id: str | None = None,
) -> dict:
    root = corpus_root.resolve()
    plan = load_plan(plan_path)
    strategy = get_strategy(strategy_id or plan.get("strategy", "hybrid"))
    template = strategy["folder_template"]

    writes: list[dict] = []  # {path, content, status}

    folders = plan.get("folders", [])
    for f in folders:
        rel = safe_relpath(f["path"], root)
        agents = corpus_path(root, rel / "AGENTS.md")
        content = f.get("agents_md") or _render_agents_md(f, template)
        writes.append({"path": agents, "content": content})
        # STATE.md gating: an explicit plan-entry value always wins; when
        # the entry omits the key, fall back to the strategy pack's
        # folder_template.state_md — but only for folders carrying the
        # pack's own kind (a role-bots pack's STATE default should not
        # stamp folders partitioned under some other kind). Before this,
        # scaffold read the raw plan (validate_plan's bot-kind default
        # never reaches it), so state_md:true in a pack did nothing.
        want_state = f.get("state_md")
        if want_state is None:
            kind = f.get("kind") or template.get("kind", "project")
            want_state = bool(template.get("state_md")) and kind == template.get("kind")
        if want_state:
            st = corpus_path(root, rel / "STATE.md")
            writes.append({"path": st, "content": _render_state_md(f, template)})

    roster_rel = Path("AGENTS.md")
    roster_content = plan.get("roster", {}).get("root_agents_md") or _auto_roster(
        folders, plan.get("keep", [])
    )
    writes.append(
        {
            "path": corpus_path(root, roster_rel),
            "content": roster_content,
            "is_roster": True,
        }
    )

    result = {"mode": "dry-run", "writes": [], "suggested": [], "skipped": []}
    for w in writes:
        target: Path = w["path"]
        status = "would_write"
        if target.exists():
            if target.read_text(encoding="utf-8", errors="replace") == w["content"]:
                status = "identical"
            else:
                sugg = target.with_suffix(target.suffix + ".slopkeeper-suggested")
                status = f"exists→suggested:{sugg.name}"
                result["suggested"].append(str(sugg))
                if execute:
                    sugg.write_text(w["content"], encoding="utf-8")
        elif execute:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(w["content"], encoding="utf-8")
        result["writes"].append({"path": str(target), "status": status})
    result["mode"] = "executed" if execute else "dry-run"
    if execute:
        # The tool layer itself is part of the tended state: record it so the
        # next `status` diff shows only real slop, not our own AGENTS.md files.
        # No unresolved passed: write_manifest carries the backlog and
        # last_plan forward, so the apply → scaffold loop keeps
        # still_unresolved alive.
        write_manifest(root)
    return result


def _render_agents_md(folder: dict, template: dict) -> str:
    body = template.get("agents_md_skeleton", "# <Name>\n")
    role = folder.get("role") or "(describe this folder's role here)"
    name = Path(folder["path"]).name
    body = body.replace("<Role name>", name)
    body = body.replace("<Name>", name)
    body = body.replace("<Topic>", name)
    body = body.replace("<Project>", name)
    # by-kind's skeleton opens "# <Kind>" — the category name is the
    # folder's own name (docs/, data/, scripts/...), same fill as <Name>.
    # The kind VALUE lives in the root roster table; no per-file Kind
    # line here (the trailing **Kind:** append duplicated whatever the
    # skeleton's own ## Kind section already said).
    body = body.replace("<Kind>", name)
    body = body.replace(
        "<bot|topic|case|project|archive>", folder.get("kind", "project")
    )
    if "## Role / scope" in body:
        body = body.replace(
            "## Role / scope\n<one paragraph>", f"## Role / scope\n{role}"
        )
    if "## Purpose" in body:
        body = body.replace("<one paragraph: what this worker owns>", role)
    if "## Scope" in body:
        body = body.replace("<what belongs here>", role)
    if "## What this is" in body:
        body = body.replace("<one paragraph>", role)
    # Append role if no slot consumed it.
    if role not in body:
        body += f"\n## Role / scope\n{role}\n"
    # Boundaries: filled from the folder entry, or the placeholder section
    # is dropped entirely (an unfilled template is noise to the reader).
    bounds = [str(b) for b in (folder.get("boundaries") or []) if str(b).strip()]
    if bounds:
        body = body.replace(
            "- <what belongs elsewhere, and where>", "\n".join(f"- {b}" for b in bounds)
        )
    else:
        body = re.sub(
            r"## Boundaries\n- <what belongs elsewhere, and where>\n?", "", body
        )
    # Join with an explicit blank line: a `---` hugging the last body
    # line renders as a setext heading in GFM, not a rule. (Skeletons
    # differ in trailing newlines — normalize here, once.)
    return f"{body.rstrip()}\n\n{_footer().lstrip()}"


def _render_state_md(folder: dict, template: dict) -> str:
    body = template.get("state_md_skeleton", "# STATE\n\n## Current status\n- \n")
    body = body.replace("<Role name>", Path(folder["path"]).name)
    body = body.replace("<date>", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
    return body


def _auto_roster(folders: list[dict], keeps: list | None = None) -> str:
    # rstrip: the header constant ends with "\n", and joining with "\n"
    # would leave a blank line between the |---| separator row and the
    # first data row — GFM then renders the rows as plain paragraphs,
    # not a table.
    lines = [ROOT_ROSTER_HEADER.rstrip("\n")]
    for f in folders:
        lines.append(f"| `{f['path']}/` | {f['kind']} | {f.get('role', '')} |")
    for k in keeps or []:
        # keep entries are plain strings or {path, reason} objects (plan schema).
        # The role cell is always filled (Phase 11 small fix): the
        # reason when given, an explicit marker when not — an empty
        # cell read as "why is this row here?".
        path, reason = (
            (k, "") if isinstance(k, str) else (k.get("path", ""), k.get("reason", ""))
        )
        if not path:
            continue
        lines.append(f"| `{path}` | keep | {reason or '(kept in place)'} |")
    lines.append("")
    lines.append(
        "_Generated by SlopKeeper. Edit freely; it will not be "
        "overwritten (changes land in *.slopkeeper-suggested)._"
    )
    return "\n".join(lines)


def _footer() -> str:
    return (
        "\n---\n_Generated by SlopKeeper — edit freely; regeneration "
        "writes *.slopkeeper-suggested instead of overwriting._\n"
    )
