# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The bare-`slopkeeper` REPL — a human front door over the same commands.

Run `slopkeeper` with no arguments and you get a `/command` prompt that drives
the SAME functions the JSON CLI exposes — no new write path, no new
error codes. What the REPL adds is a human rendering and an explicit
interactive confirm (default NO) in front of every `--execute`.

The intended loop is the re-tend review: `/draft` (or `/load` a plan
your agent wrote), walk the proposed moves with `/show`, decide each
one with `/no` `/defer` `/dest`, compile with `/plan`, read the rendered
gate with `/render`, then `/apply` and `/scaffold` — both of which stop
for a y/N confirm before touching anything. `/undo` is here too.

Scratch state lives in the corpus's own `.slopkeeper/` (drafts and the
compiled plan), which the fingerprint and the draft catalog both skip —
so the REPL can never poison a corpus the way a stray CWD artifact
would. First-tend authoring (naming a taxonomy from an inventory) is
agent work: point your agent at AGENT_GUIDE.md for that, or `/load` a
plan it wrote and review it here.

Zero dependencies; no LLM; no network.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

from . import VERSION
from .apply import apply_plan, build_ref_report, undo_journal
from .doctor import doctor
from .draft import draft as draft_plan
from .inventory import collect
from .models import SlopKeeperError, safe_relpath, state_dir
from .plan import load_plan, render_markdown, validate_plan
from .scaffold import scaffold_from_plan
from .similar import collect_similarity, hubs
from .status import _head_for, status as corpus_status
from .strategies import get_strategy, list_strategies

WIDTH = 78
CAT_LINES = 60


class _Quit(Exception):
    pass


class _Reader:
    """Line source with an optional inline prompt.

    Interactive stdin gets `> ` before every read and `[y/N]` on
    confirms; injected lines (tests, pipes) stay prompt-free so
    captured output is clean.
    """

    def __init__(self, it, out, interactive: bool):
        self._it = it
        self._out = out
        self._interactive = interactive

    def nextline(self, prompt: str = "") -> str | None:
        if self._interactive and prompt:
            self._out.write(prompt)
            self._out.flush()
        try:
            return next(self._it).strip()
        except StopIteration:
            return None


def _p(out, text: str = "") -> None:
    print(text, file=out)


def _trunc(text, n: int = WIDTH) -> str:
    s = str(text if text is not None else "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _stdin_lines():
    try:
        for line in sys.stdin:
            yield line
    except EOFError:
        return


class _Session:
    def __init__(self):
        self.root: Path | None = None
        self.plan: dict | None = None  # working plan (draft or /load'ed)
        self.decisions: dict[int, tuple] = {}  # move idx → ("drop"|"defer"|"dest", v)
        self.plan_path: Path | None = None  # compiled plan on disk (apply target)
        self.dirty = False  # decisions changed since the last /plan

    def require_root(self) -> Path:
        if self.root is None:
            raise SlopKeeperError(
                "no_root",
                "no corpus root — use `/use <path>` first",
                "Type /help for commands.",
            )
        return self.root

    def require_plan(self):
        if self.plan is None:
            raise SlopKeeperError(
                "no_plan",
                "no working plan — `/draft` one or `/load <plan.json>`",
                "/draft drafts a delta plan for new slop (any depth).",
            )
        return self.plan


# --- renderers ---------------------------------------------------------------


def _render_status(s: _Session, out) -> None:
    d = corpus_status(s.require_root())
    if d.get("never_tended"):
        _p(out, "Never tended. For a first tend: point your agent at")
        _p(out, "AGENT_GUIDE.md, or explore with /doctor and /inventory.")
        for a in d.get("advice", []):
            _p(out, f"  - {_trunc(a)}")
        return
    flags = []
    if d["new_count"]:
        flags.append(f"{d['new_count']} new")
    if d["modified"]:
        flags.append(f"{len(d['modified'])} modified")
    if d["removed"]:
        flags.append(f"{len(d['removed'])} removed")
    _p(out, f"Last tended {d['last_tended']}: {d['files_now']} files now")
    _p(out, "  " + ("; ".join(flags) if flags else "nothing changed — clean."))
    if d.get("folders_missing_agents_md"):
        _p(
            out,
            f"  folders missing AGENTS.md: {', '.join(d['folders_missing_agents_md'])}",
        )
    for sf in d.get("stale_folders", []):
        _p(
            out,
            "  stale STATE.md: "
            f"{sf['folder']}/ (newest file {sf['newest_file_mtime']})",
        )
    if d.get("still_unresolved"):
        _p(out, f"  unresolved backlog ({len(d['still_unresolved'])}):")
        for u in d["still_unresolved"][:10]:
            _p(out, f"    - {u['path']}: {_trunc(u.get('reason', ''), WIDTH - 22)}")


def _render_new(s: _Session, out) -> None:
    d = corpus_status(s.require_root())
    if d.get("never_tended"):
        _p(out, "Never tended — /status explains. No manifest to diff against.")
        return
    if not d["new_count"]:
        _p(out, "No new files since the last tend.")
        return
    _p(
        out,
        f"{d['new_count']} new since the last tend "
        f"(~{d['estimated_triage_tokens']} tokens to triage):",
    )
    for f in d["new_files"][:30]:
        _p(out, f"  {f['path']}")
        if f.get("title"):
            _p(out, f"      {f['title']}")
        if f.get("head"):
            first = f["head"].strip().splitlines()[0] if f["head"].strip() else ""
            _p(out, f"      {_trunc(first, WIDTH - 6)}")
    if d["new_count"] > len(d["new_files"]):
        _p(out, f"  … and {d['new_count'] - len(d['new_files'])} more (cap: 2000).")


def _render_doctor(s: _Session, out) -> None:
    d = doctor(s.require_root())
    st = d["stats"]
    _p(
        out,
        f"{st['files']} files, {st['dirs']} dirs, "
        f"{st['loose_top_level_files']} loose at root",
    )
    sug = d.get("suggested_strategy") or {}
    if sug.get("id"):
        _p(out, f"  suggested strategy: {sug['id']} — {sug.get('why', '')}")
    for a in d.get("advice", []):
        _p(out, f"  - {_trunc(a)}")


def _render_inventory(s: _Session, out) -> None:
    inv = collect(s.require_root(), fast=True)
    st = inv["stats"]
    _p(out, f"{st['files']} files, {st['dirs']} dirs, {st['total_bytes']} bytes")
    _p(out, f"  fingerprint {inv['fingerprint']}")
    cl = inv.get("clusters", {}).get("by_prefix", {})
    if cl:
        _p(
            out,
            "  prefixes: "
            + ", ".join(f"{k}×{len(v)}" for k, v in sorted(cl.items())[:8]),
        )


def _render_clusters(s: _Session, out) -> None:
    root = s.require_root()
    inv = collect(root)
    data = collect_similarity(root, inv["files"])
    if inv.get("references"):
        data["hubs"] = hubs(inv["references"])
    cl = data.get("clusters", [])
    _p(out, f"{len(cl)} clusters, {len(data.get('unclustered', []))} unclustered")
    for c in cl[:20]:
        preview = _trunc(", ".join(c["files"][:3]), WIDTH - 16)
        _p(out, f"  [{c['size']}] {c['label']}: {preview}")
    for kind in ("same_stem_groups", "identical_groups"):
        for g in data.get(kind, [])[:5]:
            _p(out, f"  {kind[:-7]}: {_trunc(' / '.join(g[:3]), WIDTH - 16)}")
    for h in data.get("hubs", [])[:5]:
        if isinstance(h, dict):
            _p(out, f"  hub: {h.get('basename')} (cited by {h.get('referrers')} files)")
        else:
            _p(out, f"  hub: {h}")


def _render_refs(s: _Session, arg: str, out) -> None:
    if not arg:
        raise SlopKeeperError(
            "no_basename", "usage: /refs <basename>", "e.g. /refs acme_dossier.md"
        )
    from .apply import _all_rel_paths
    from .references import citation_lines

    root = s.require_root()
    hits = citation_lines(root, arg, _all_rel_paths(root))
    if not hits:
        _p(out, f"No citations of {arg} found inside the corpus.")
        return
    _p(out, f"{len(hits)} citation(s) of {arg}:")
    for h in hits[:20]:
        _p(out, f"  {h['file']}:{h['line']}: {_trunc(h['text'].strip(), WIDTH - 20)}")


def _cat(s: _Session, arg: str, out) -> None:
    if not arg:
        raise SlopKeeperError(
            "no_path", "usage: /cat <file>", "A path relative to the corpus root."
        )
    root = s.require_root()
    fp = corpus_path_of(root, arg)
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        raise SlopKeeperError(
            "cat_unreadable", str(e), "Check the path (relative to the corpus root)."
        )
    _p(out, f"── {arg} ──")
    for line in lines[:CAT_LINES]:
        _p(out, "  " + _trunc(line))
    if len(lines) > CAT_LINES:
        _p(out, f"  … {len(lines) - CAT_LINES} more lines")


def corpus_path_of(root: Path, rel: str) -> Path:
    from .models import corpus_path

    return corpus_path(root, safe_relpath(rel, root))


# --- the review loop ---------------------------------------------------------


def _compile(s: _Session) -> dict:
    """Apply decisions to the working plan and return the compiled dict."""
    plan = s.require_plan()
    moves, unresolved = [], list(plan.get("unresolved", []))
    for i, m in enumerate(plan.get("moves", [])):
        d = s.decisions.get(i)
        if d is None:
            moves.append(m)
        elif d[0] == "drop":
            continue
        elif d[0] == "defer":
            unresolved.append({"path": m["from"], "reason": d[1]})
        elif d[0] == "dest":
            moves.append({**m, "to": d[1]})
    return {**plan, "moves": moves, "unresolved": unresolved}


def _save_plan(s: _Session, plan: dict) -> Path:
    assert s.root is not None
    path = state_dir(s.root) / "plan.json"
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    s.plan_path = path
    return path


def _render_moves(s: _Session, out) -> None:
    plan = s.require_plan()
    if not plan.get("moves"):
        _p(out, "The working plan has no moves.")
    for i, m in enumerate(plan.get("moves", [])):
        d = s.decisions.get(i)
        kind = d[0] if d else ""
        mark = {"drop": "✗", "defer": "…", "dest": "→"}.get(kind, " ")
        to = d[1] if kind == "dest" else m["to"]
        note = (
            " (dropped)"
            if kind == "drop"
            else (f" (deferred: {d[1]})" if kind == "defer" else "")
        )
        _p(out, f" {i + 1}{mark} {m['from']} → {to}{note}")
        _p(out, f"      {_trunc(m.get('reason', ''), WIDTH - 6)}")
    if plan.get("unresolved"):
        _p(out, f" unresolved from the draft ({len(plan['unresolved'])}):")
        for u in plan["unresolved"][:10]:
            _p(out, f"   ? {u['path']}: {_trunc(u.get('reason', ''), WIDTH - 22)}")


# --- commands ----------------------------------------------------------------


def _print_help(out) -> None:
    _p(out, "the same commands as `slopkeeper <subcommand>`, human-rendered:")
    for line in HELP_LINES:
        _p(out, "  " + line)


def _cmd_help(s, arg, lines, out) -> None:
    _print_help(out)


def _cmd_use(s, arg, lines, out) -> None:
    if not arg:
        raise SlopKeeperError(
            "no_root", "usage: /use <path>", "e.g. /use ~/Documents/notes"
        )
    root = Path(arg).expanduser().resolve()
    if not root.is_dir():
        raise SlopKeeperError(
            "no_corpus", f"not a directory: {root}", "Check the path."
        )
    s.root = root
    s.plan = None
    s.decisions = {}
    s.plan_path = None
    s.dirty = False
    _p(out, f"root: {root}")
    try:
        d = corpus_status(root)
        if d.get("never_tended"):
            _p(out, "  never tended — /doctor for a preflight.")
        else:
            _p(
                out,
                f"  last tended {d['last_tended']}; {d['new_count']} new, "
                f"{len(d['modified'])} modified since.",
            )
    except SlopKeeperError:
        pass


def _cmd_status(s, arg, lines, out) -> None:
    _render_status(s, out)


def _cmd_new(s, arg, lines, out) -> None:
    _render_new(s, out)


def _cmd_doctor(s, arg, lines, out) -> None:
    _render_doctor(s, out)


def _cmd_inventory(s, arg, lines, out) -> None:
    _render_inventory(s, out)


def _cmd_clusters(s, arg, lines, out) -> None:
    _render_clusters(s, out)


def _cmd_refs(s, arg, lines, out) -> None:
    _render_refs(s, arg, out)


def _cmd_cat(s, arg, lines, out) -> None:
    _cat(s, arg, out)


def _cmd_strategies(s, arg, lines, out) -> None:
    if arg:
        _p(out, json.dumps(get_strategy(arg), indent=2))
    else:
        for st in list_strategies():
            _p(out, f"  {st['id']}: {st.get('summary', st.get('name', ''))}")


def _cmd_draft(s, arg, lines, out) -> None:
    root = s.require_root()
    data = draft_plan(root)
    if data.get("plan") is None:
        _p(out, data.get("note", "Nothing to draft."))
        return
    path = state_dir(root) / "draft-plan.json"
    path.write_text(json.dumps(data["plan"], indent=1), encoding="utf-8")
    s.plan = data["plan"]
    s.decisions = {}
    s.plan_path = path
    s.dirty = False
    _p(
        out,
        f"Drafted {data['drafted_count']} move(s), "
        f"{data['unresolved_count']} unresolved of {data['new_count']} new. "
        f"Saved {path}",
    )
    if data.get("undrafted"):
        u = data["undrafted"]
        _p(out, f"  {u['count']} undrafted: {u['reason']}")
    _p(
        out,
        "  Walk them with /moves, /show <n>, then /no /defer /dest — /plan to compile.",
    )


def _cmd_load(s, arg, lines, out) -> None:
    s.require_root()
    if not arg:
        raise SlopKeeperError(
            "no_path", "usage: /load <plan.json>", "The plan your agent wrote."
        )
    p = Path(arg).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise SlopKeeperError(
            "plan_unreadable", f"cannot read {p}: {e}", "Check the path."
        )
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as e:
        raise SlopKeeperError(
            "plan_not_json",
            f"{p} is not valid JSON: {e}",
            "Ask your agent to re-save it.",
        )
    if not isinstance(plan, dict) or not isinstance(plan.get("moves"), list):
        raise SlopKeeperError(
            "plan_not_object", f"{p} is not a plan object", "It needs a `moves` list."
        )
    s.plan = plan
    s.decisions = {}
    s.plan_path = None
    s.dirty = True
    _p(
        out,
        f"Loaded {len(plan['moves'])} move(s), "
        f"{len(plan.get('unresolved', []))} unresolved "
        f"from {p}. /moves to walk them, /plan to compile.",
    )


def _cmd_moves(s, arg, lines, out) -> None:
    _render_moves(s, out)


def _move_index(s, arg: str) -> int:
    try:
        i = int(arg) - 1
    except ValueError:
        raise SlopKeeperError(
            "no_such_move", f"not a move number: {arg}", "/moves lists them."
        )
    plan = s.require_plan()
    if i < 0 or i >= len(plan.get("moves", [])):
        raise SlopKeeperError("no_such_move", f"no move #{arg}", "/moves lists them.")
    return i


def _cmd_show(s, arg, lines, out) -> None:
    i = _move_index(s, arg.split()[0] if arg else "")
    root = s.require_root()
    src = s.plan["moves"][i]["from"]
    h = _head_for(root, src)
    _p(out, f"── {src} ──")
    if h.get("title"):
        _p(out, f"  title: {h['title']}")
    if h.get("head"):
        for line in h["head"].strip().splitlines()[:10]:
            _p(out, "  " + _trunc(line))
    _p(out, f"  proposed: {s.plan['moves'][i]['to']}")


def _decide(s, arg: str, kind: str, value=None, out=None) -> None:
    parts = arg.split(maxsplit=1)
    i = _move_index(s, parts[0] if parts else "")
    if kind == "ok":
        s.decisions.pop(i, None)
        _p(out, f"  move {i + 1} back to keep.")
        return
    s.decisions[i] = (kind, value)
    s.dirty = True


def _cmd_ok(s, arg, lines, out) -> None:
    if arg.strip() == "all":
        s.decisions.clear()
        _p(out, "All decisions cleared.")
        return
    _decide(s, arg, "ok", out=out)


def _cmd_no(s, arg, lines, out) -> None:
    _decide(s, arg, "drop")
    _p(out, f"  move {arg}: dropped — the file stays where it is.")


def _cmd_defer(s, arg, lines, out) -> None:
    parts = arg.split(maxsplit=1)
    why = (
        parts[1].strip()
        if len(parts) > 1
        else "deferred in review — no destination chosen"
    )
    _decide(s, arg, "defer", why)
    _p(out, f"  move {parts[0]}: deferred to the backlog.")


def _cmd_dest(s, arg, lines, out) -> None:
    parts = arg.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        raise SlopKeeperError(
            "no_path", "usage: /dest <n> <new/destination/path>", "/moves lists them."
        )
    _decide(s, arg, "dest", parts[1].strip())
    _p(out, f"  move {parts[0]}: re-destined to {parts[1].strip()}")


def _cmd_plan(s, arg, lines, out) -> None:
    root = s.require_root()
    compiled = _compile(s)
    # A reviewed plan that deliberately leaves files in place (dropped
    # moves) no longer covers the corpus — that is the definition of
    # partial, and validate's coverage check must not fail on it.
    if any(k == "drop" for k, _ in s.decisions.values()):
        compiled["partial"] = True
    compiled = validate_plan(
        compiled, root, allow_partial=bool(compiled.get("partial"))
    )
    path = _save_plan(s, compiled)
    s.dirty = False
    _p(
        out,
        f"Valid: {len(compiled['moves'])} move(s), "
        f"{len(compiled.get('unresolved', []))} unresolved → {path}",
    )
    _p(out, "  /render to read the human gate, /apply to execute.")


def _cmd_render(s, arg, lines, out) -> None:
    if s.plan_path is None or s.dirty:
        _p(out, "Decisions changed — /plan first (it validates and saves).")
        return
    plan = load_plan(s.plan_path)
    _p(out, render_markdown(plan, build_ref_report(plan, s.require_root())))


def _cmd_apply(s, arg, lines, out) -> None:
    if s.plan_path is None or s.dirty:
        raise SlopKeeperError(
            "no_plan",
            "no compiled plan — /plan first (or /load + /plan)",
            "/moves and /no /defer /dest decide; /plan compiles.",
        )
    root = s.require_root()
    dry = apply_plan(s.plan_path, root, execute=False)
    would = len(dry.get("would_move", []))
    risks = len(dry.get("reference_risks", {}) or {})
    _p(
        out,
        f"Dry run: {would} move(s)"
        + (f", {risks} file(s) with reference risks" if risks else "")
        + ".",
    )
    for src, risk in (dry.get("reference_risks") or {}).items():
        _p(out, f"  ⚠ {src}: cited by {risk['external_references']} stay-put file(s)")
    if not _confirm(lines, out, f"Execute {would} move(s) with --execute?"):
        _p(out, "Aborted — nothing moved.")
        return
    data = apply_plan(s.plan_path, root, execute=True)
    _p(out, f"Executed: {data['moved_count']} moved, journal {data.get('journal')}")
    for err in data.get("errors", []):
        _p(out, f"  ✗ {err}")
    if data.get("partial"):
        _p(out, f"  PARTIAL — undo with: slopkeeper undo {data.get('journal')} {root}")
    _p(out, "  /scaffold writes the AGENTS.md layer.")


def _cmd_scaffold(s, arg, lines, out) -> None:
    if s.plan_path is None or s.dirty:
        raise SlopKeeperError("no_plan", "no compiled plan — /plan first", "")
    root = s.require_root()
    if not _confirm(lines, out, "Write the AGENTS.md/STATE.md layer?"):
        _p(out, "Aborted — nothing written.")
        return
    data = scaffold_from_plan(s.plan_path, root, execute=True)
    for w in data.get("writes", []):
        _p(out, f"  {w['status']}: {w['path']}")
    if data.get("suggested"):
        _p(out, "  Merge the *.slopkeeper-suggested files by hand.")


def _journals(root: Path) -> list[Path]:
    return sorted((root / ".slopkeeper").glob("journal-*.jsonl"))


def _cmd_undo(s, arg, lines, out) -> None:
    root = s.require_root()
    journals = _journals(root)
    if not journals:
        _p(out, "No journals — nothing to undo.")
        return
    for i, j in enumerate(journals, 1):
        _p(out, f"  {i}: {j.name}")
    parts = arg.split()
    idx = len(journals)  # default: the latest
    execute = False
    try:
        if parts and parts[0] != "!":
            idx = int(parts[0])
            execute = "!" in parts[1:]
        elif parts and parts[0] == "!":
            execute = True
    except ValueError:
        raise SlopKeeperError(
            "no_such_journal",
            f"not a journal number: {parts[0]}",
            "Pick from the list above.",
        )
    if idx < 1 or idx > len(journals):
        raise SlopKeeperError(
            "no_such_journal", f"no such journal #{idx}", "Pick from the list above."
        )
    journal = journals[idx - 1]
    data = undo_journal(journal, root, execute=False)
    for op in data["operations"]:
        _p(out, f"  {op.get('status')}: {op.get('src', op.get('file', ''))}")
    if not execute:
        _p(out, "  (dry run — add ! to execute: /undo %d !)" % idx)
        return
    if not _confirm(
        lines, out, f"Restore {len(data['operations'])} op(s) from {journal.name}?"
    ):
        _p(out, "Aborted — nothing restored.")
        return
    data = undo_journal(journal, root, execute=True)
    _p(out, f"Restored. manifest: {data.get('manifest', 'untouched')}")


def _confirm(reader, out, prompt: str) -> bool:
    ans = (reader.nextline(f"  {prompt} [y/N] ") or "").lower()
    return ans in ("y", "yes")


def _cmd_quit(s, arg, lines, out) -> None:
    raise _Quit()


COMMANDS = {
    "help": _cmd_help,
    "use": _cmd_use,
    "status": _cmd_status,
    "new": _cmd_new,
    "doctor": _cmd_doctor,
    "inventory": _cmd_inventory,
    "clusters": _cmd_clusters,
    "refs": _cmd_refs,
    "cat": _cmd_cat,
    "strategies": _cmd_strategies,
    "draft": _cmd_draft,
    "load": _cmd_load,
    "moves": _cmd_moves,
    "show": _cmd_show,
    "ok": _cmd_ok,
    "no": _cmd_no,
    "defer": _cmd_defer,
    "dest": _cmd_dest,
    "plan": _cmd_plan,
    "render": _cmd_render,
    "apply": _cmd_apply,
    "scaffold": _cmd_scaffold,
    "undo": _cmd_undo,
    "q": _cmd_quit,
    "quit": _cmd_quit,
    "exit": _cmd_quit,
}

HELP_LINES = [
    "/use <path>        set the corpus root",
    "/status            what changed since the last tend",
    "/new               the new slop, with titles and heads",
    "/doctor            preflight summary",
    "/inventory         corpus stats and fingerprint",
    "/clusters          similarity hints",
    "/refs <basename>   citation sites for a filename",
    "/cat <file>        show a corpus file",
    "/strategies [id]   list rubrics / show one",
    "/draft             auto-draft a delta plan for new slop (any depth)",
    "/load <plan.json>  load a plan (e.g. one your agent wrote) to review",
    "/moves             list the working plan's moves",
    "/show <n>          show the file a move would relocate",
    "/ok <n|all>        clear a decision on move n",
    "/no <n>            drop move n — the file stays where it is",
    "/defer <n> [why]   send move n to the unresolved backlog",
    "/dest <n> <to>     re-destination move n",
    "/plan              compile decisions, validate, save",
    "/render            show the rendered human-gate markdown",
    "/apply             dry-run, confirm, execute the plan",
    "/scaffold          write the AGENTS.md layer (confirm)",
    "/undo [n] [!]      list journals / dry-run / execute one",
    "/quit              leave (Ctrl-D also works)",
]


# --- completion ---------------------------------------------------------------
#
# Tab completes command names, filesystem paths one component at a time
# (corpus-relative where the argument is), and move/journal numbers.
# _compute_matches is pure and unit-tested; readline is just the transport.

_ACTIVE_SESSION: _Session | None = None
_MATCHES: list[str] = []


def _command_completions(text: str) -> list[str]:
    stem = text.lstrip("/")
    return ["/" + c for c in COMMANDS if c.startswith(stem)]


def _path_completions(text: str, base: Path | None) -> list[str]:
    """Complete one path component; `base` None = absolute/user paths."""
    dir_raw, _, name_raw = text.rpartition("/")
    if base is None:
        if dir_raw.startswith("~") or name_raw.startswith("~"):
            search = (
                Path(os.path.expanduser(dir_raw))
                if dir_raw
                else Path(os.path.expanduser("~"))
            )
            prefix = (dir_raw + "/") if dir_raw else "~/"
            if name_raw == "~":
                # "~" is the expansion marker, not a stem to match against.
                name_raw = ""
        else:
            search = Path(dir_raw) if dir_raw else Path(".")
            prefix = (dir_raw + "/") if dir_raw else ""
    else:
        search = base / dir_raw if dir_raw else base
        prefix = (dir_raw + "/") if dir_raw else ""
    try:
        entries = sorted(search.iterdir())
    except OSError:
        return []
    low = name_raw.lower()
    out = []
    for e in entries:
        if e.name.startswith(".") and not name_raw.startswith("."):
            continue
        if e.name.lower().startswith(low):
            out.append(prefix + e.name + ("/" if e.is_dir() else ""))
    return out


def _move_number_completions(s: _Session, text: str) -> list[str]:
    if s.plan is None:
        return []
    return [
        str(i + 1)
        for i in range(len(s.plan.get("moves", [])))
        if str(i + 1).startswith(text)
    ]


def _journal_completions(s: _Session, text: str) -> list[str]:
    if s.root is None:
        return []
    return [
        str(i) for i in range(1, len(_journals(s.root)) + 1) if str(i).startswith(text)
    ]


def _compute_matches(line: str, text: str, s: _Session) -> list[str]:
    """Matches for the trailing token `text` of `line` (tab at end of line).

    Token position: trim `text` off the line's end, then split — but if
    the cut lands mid-token, that last token is a PREFIX of the one
    being completed and must not count as a completed argument.
    """
    head = line[: len(line) - len(text)] if text else line
    toks = head.split()
    if head and not head[-1].isspace():
        toks = toks[:-1]
    if not toks:
        return _command_completions(text)
    cmd = toks[0].lstrip("/")
    if cmd not in COMMANDS:
        return []
    argidx = len(toks) - 1
    if argidx == 0:
        if cmd in ("use", "load"):
            return _path_completions(text, None)
        if cmd in ("cat", "refs"):
            return _path_completions(text, s.root) if s.root else []
        if cmd == "strategies":
            return [st["id"] for st in list_strategies() if st["id"].startswith(text)]
        if cmd in ("show", "no", "defer", "ok", "dest"):
            return _move_number_completions(s, text)
        if cmd == "undo":
            return _journal_completions(s, text)
        return []
    if cmd == "dest":
        return _path_completions(text, s.root) if s.root else []
    return []


def _readline_completer(text: str, state: int):
    global _MATCHES
    if state == 0:
        import readline

        _MATCHES = _compute_matches(
            readline.get_line_buffer(), text, _ACTIVE_SESSION or _Session()
        )
    return _MATCHES[state] if state < len(_MATCHES) else None


def _history_file() -> Path:
    """Readline history: ~/.local/state/slopkeeper/history (XDG_STATE_HOME)."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "slopkeeper" / "history"


def _setup_readline() -> None:
    """History + tab completion + emacs editing for interactive use.

    Stdlib readline only; absent (Windows) the REPL still works, just
    without history. The history file lives in the user's state dir.
    """
    try:
        import readline
    except ImportError:
        return
    readline.set_history_length(1000)
    # "/" stays IN the word being completed so paths tab component-wise.
    readline.set_completer_delims(" \t")
    readline.set_completer(_readline_completer)
    readline.parse_and_bind("tab: complete")
    hist = _history_file()
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(hist)
    except OSError:
        pass
    atexit.register(readline.write_history_file, hist)


def repl(root=None, *, lines=None, out=None, interactive=None) -> int:
    """Run the interactive loop. `lines`/`out` are injectable for tests.

    `interactive` defaults to true only for a real tty (piped input,
    including the tests, stays prompt-free so captured output is clean).
    """
    out = out if out is not None else sys.stdout
    if interactive is None:
        interactive = lines is None and sys.stdin.isatty()
    reader = _Reader(
        iter(lines) if lines is not None else _stdin_lines(),
        out,
        bool(interactive),
    )
    s = _Session()
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = s
    if interactive:
        _setup_readline()
    if root is not None:
        try:
            _cmd_use(s, str(root), reader, out)
        except SlopKeeperError as e:
            _p(out, f"✗ {e.code}: {e.message}")
    _p(out, f"slopkeeper {VERSION}. /help repeats this list; /quit to leave.")
    _p(out, "")
    _print_help(out)
    while True:
        raw = reader.nextline("> ")
        if raw is None:
            return 0
        line = raw
        if not line:
            continue
        if not line.startswith("/"):
            _p(out, "Type /help for commands — plain text is not a command.")
            continue
        name, _, arg = line[1:].partition(" ")
        try:
            handler = COMMANDS.get(name.lower())
            if handler is None:
                _p(out, f"Unknown command /{name} — /help lists commands.")
                continue
            handler(s, arg.strip(), reader, out)
        except _Quit:
            return 0
        except (KeyboardInterrupt, EOFError):
            return 0
        except SlopKeeperError as e:
            _p(out, f"✗ {e.code}: {e.message}")
            if e.recovery:
                _p(out, f"  {e.recovery}")
        except Exception as e:  # noqa: BLE001 — a REPL must survive its bugs
            _p(out, f"✗ internal: {type(e).__name__}: {e}")
            _p(out, "  Please report this; include the commands you ran.")
