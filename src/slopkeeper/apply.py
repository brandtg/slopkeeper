# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Plan execution: all-or-nothing moves with a journal and undo.

Safety invariants, in order:
1. Nothing happens without --execute (dry-run is the default).
2. The corpus must match the fingerprint the plan was authored against.
3. Every move is pre-flighted (collisions, chains, nesting) before ANY move.
4. Every executed move is journaled with a content hash; undo reverses exactly.
5. Nothing is ever deleted; a failed mid-run is recoverable via the journal.
6. A mid-run IO failure is journaled-partial, never silent: recorded in
   `errors`, the envelope carries `partial: true`, the process exits
   non-zero, and `next` points at the exact undo command.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .ignore import corpus_walk
from .models import (
    SlopKeeperError,
    corpus_fingerprint,
    corpus_path,
    resolve_chains,
    sha256_file,
    sha256_file_exact,
    state_dir,
    tend_lock,
)
from .plan import load_plan, load_render_sidecar, validate_plan
from .references import (
    CitationIndex,
    _apply_line_replacements,
    _post_move_path,
    _SNIPPET_CAP,
    _text_files_under,
    edit_first_spans,
    edit_skip_lines,
    enriched_risk,
    plan_rewrites,
    scan_references_stream,
)
from .status import load_manifest, write_manifest

# §5.6: a mismatch between the plan's bytes and the last render is a
# WARNING, never a refusal — the human may hand-edit the plan JSON after
# rendering (the documented AGENT_GUIDE flow). It only makes silent
# render→apply drift visible.
RENDER_WARNING = (
    "plan differs from last render — if the human approved "
    "the render, re-render or get explicit sign-off."
)


def _default_journal(root: Path) -> Path:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return state_dir(root) / f"journal-{ts}.jsonl"


def _all_rel_paths(root: Path) -> list[Path]:
    out = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        dirpath = Path(dirpath)
        for fn in filenames:
            out.append((dirpath / fn).relative_to(root))
    return out


def _moved_files_and_dsts(
    chains: dict[str, str], rel_paths: list[Path], root: Path
) -> set[str]:
    """Every file the plan moves — directly, or as a descendant of a moved
    directory — plus every destination. Citations FROM these files are
    fixed by the same apply, so they are not breakage.

    A src counts as a directory when any corpus file lives under it (the
    trailing-slash form and the bare-dir form both land here); a plain
    file src never has descendants, so it expands to itself.

    The directory test is one is_dir() stat per src, not an any() scan
    of every rel path: a rule-expanded plan (Phase 8) carries tens of
    thousands of file moves, and a scan per src is O(n²) there. The two
    tests answer identically — a src with no file under it contributes
    no descendants either way; the stat just gets there in O(1).
    """
    moved = set(chains.keys()) | set(chains.values())
    for src in chains:
        prefix = src.rstrip("/") + "/"
        if (root / src).is_dir():
            moved.update(str(rp) for rp in rel_paths if str(rp).startswith(prefix))
    return moved


def _shape_rewrite_summary(
    mode: str,
    rewrites: dict[str, list[dict]],
    skipped: list[dict],
    targets: dict[str, int],
    skip_rewrites: list[dict] | None = None,
) -> dict:
    """One shape for every rewrite summary (apply result, dry-run, render).

    per_file carries the full line detail ({line, old, new}) — B1: the
    gate must show WHAT the pass will change, not just how many.
    skip_rewrites (C2): the single bad lines — spans citing a moving
    path that the pass refused to guess at — for hand editing.
    """
    per_file = [
        {"file": f, "count": len(rs), "lines": rs} for f, rs in sorted(rewrites.items())
    ]
    return {
        "mode": mode,
        "per_file": per_file,
        "count": sum(r["count"] for r in per_file),
        "skipped": skipped,
        "skip_rewrites": skip_rewrites or [],
        "targets": targets,
    }


def _normalize_prefix_rewrites(plan: dict) -> tuple[tuple[str, str], ...]:
    """plan.prefix_rewrites as (from, to) tuples, in declared order."""
    raw = plan.get("prefix_rewrites") or []
    out = []
    for e in raw:
        if isinstance(e, dict):
            out.append((str(e.get("from", "")).strip(), str(e.get("to", "")).strip()))
    return tuple((a, b) for a, b in out if a and b)


def _run_rewrite_pass(
    root: Path,
    plan: dict,
    chains: dict[str, str],
    rel_paths: list[Path],
    exclude: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """The citation rewrite pass over a validated plan's moves.

    One engine for the render's numbers and the execution's writes (the
    gate and the run can never disagree). Returns the raw pass outputs
    plus the inputs the risk join needs (edit-skip and rewrite:false
    sets — both explain why a citing file got no rewrite). Edit files
    are scanned like any other (C3): edits run after the pass, so a
    hand edit and the pass's rewrites land in the same file; the
    "rewrite": false subtrees are scanned record-only (C6) so their
    citation rows carry full detail.
    """
    index = CitationIndex(rel_paths)
    scan, edit_skip, no_rewrite, owned_lines, first_spans = _rewrite_inputs(
        root, rel_paths, plan, chains, exclude
    )
    rewrites, skipped, targets, sites, unreadable, skip_rewrites = plan_rewrites(
        root,
        index,
        chains,
        scan,
        _normalize_prefix_rewrites(plan),
        record_only=frozenset(no_rewrite),
        skip_lines=owned_lines,
        first_spans=first_spans,
    )
    return {
        "rewrites": rewrites,
        "skipped": skipped,
        "skip_rewrites": skip_rewrites,
        "targets": targets,
        "sites": sites,
        "unreadable": unreadable,
        "edit_skip": edit_skip,
        "no_rewrite": no_rewrite,
    }


def _enriched_risk(
    src: str,
    refs_full: dict[str, list[str]],
    moved: set[str],
    sites: dict[str, list[dict]],
    unreadable: set[str],
    edit_skip: set[str],
    no_rewrite: set[str],
    root: Path | None = None,
    index: CitationIndex | None = None,
) -> dict:
    """reference_risk plus per-citation detail (B2) — a thin wrapper
    over references.enriched_risk (the one engine propose's break
    estimate and apply's gate share). root+index enable the D6
    not_recorded line lookup."""
    return enriched_risk(
        src,
        refs_full,
        moved,
        sites,
        edit_skip,
        no_rewrite,
        unreadable,
        root=root,
        index=index,
    )


def _run_artifacts(root: Path, *raw_paths: str | None) -> frozenset[str]:
    """This run's own artifact paths (plan file, gate render, rewrites
    JSON), corpus-relative when they sit inside the corpus — excluded
    from the rewrite pass so the tool never rewrites its own inputs
    mid-run."""
    out: set[str] = set()
    for raw in raw_paths:
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        try:
            out.add(p.relative_to(root).as_posix())
        except ValueError:
            continue
    return frozenset(out)


def build_rewrite_summary(
    plan: dict, root: Path, exclude: set[str] | frozenset[str] = frozenset()
) -> dict:
    """What the citation rewrite pass will do at apply (render side).

    The same engine on the pre-move disk, zero writes — the human gates
    on real numbers. mode "off" returns the off-summary so the render
    can say the pass will NOT run instead of implying it did nothing.
    """
    mode = (plan.get("rewrites") or {}).get("mode", "auto")
    if mode == "off":
        return _shape_rewrite_summary(mode, {}, [], {})
    chains = resolve_chains(plan["moves"])
    rel_paths = _all_rel_paths(root)
    result = _run_rewrite_pass(root, plan, chains, rel_paths, exclude)
    return _shape_rewrite_summary(
        mode,
        result["rewrites"],
        result["skipped"],
        result["targets"],
        result["skip_rewrites"],
    )


def build_gate_report(
    plan: dict,
    root: Path,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict, dict]:
    """(reference_risks, rewrite_summary) for the human gate — one pass.

    The rewrite pass runs FIRST and its per-site records feed the risk
    rows (B2): a citing file the pass will rewrite is marked as such,
    and only moves with UNCOVERED citations land in the report — a
    covered file is not a risk. Callers that only need the summary can
    ignore the first element.
    """
    mode = (plan.get("rewrites") or {}).get("mode", "auto")
    chains = resolve_chains(plan["moves"])
    rel_paths = _all_rel_paths(root)
    # D1: edit files are no longer wholesale-added to `moved` — their
    # citations are judged per-row now (the owned-line branch records
    # edit_covered vs edit_owned_uncovered precisely). Citations the
    # pass records no row for still read edit_covered via the
    # edit_skip join in enriched_risk.
    refs_full = scan_references_stream(root)
    refs_full.pop("stats", None)
    # A "rewrite": false subtree MOVES but its citations are not
    # rewritten — its outbound citations to moving paths are breakage,
    # so its files must not read as self-covered (B1's own trap).
    # D1: edit files are no longer wholesale-added here — their
    # citations are judged per-row (owned-line branch), and rows the
    # pass records none of still read edit_covered via the edit_skip
    # join in enriched_risk.
    moved = _moved_files_and_dsts(chains, rel_paths, root) - _no_rewrite_set(
        plan, rel_paths
    )
    if mode == "off":
        result = {
            "rewrites": {},
            "skipped": [],
            "targets": {},
            "sites": {},
            "unreadable": set(),
            "edit_skip": {e["file"] for e in plan.get("edits", [])},
            "no_rewrite": set(),
        }
    else:
        result = _run_rewrite_pass(root, plan, chains, rel_paths, exclude)
    ref_report: dict[str, dict] = {}
    gate_index = CitationIndex(rel_paths)
    for src in chains:
        risk = _enriched_risk(
            src,
            refs_full,
            moved,
            result["sites"],
            result["unreadable"],
            result["edit_skip"],
            result["no_rewrite"],
            root=root,
            index=gate_index,
        )
        if risk["uncovered_references"] > 0:
            ref_report[src] = risk
    summary = _shape_rewrite_summary(
        mode,
        result["rewrites"],
        result["skipped"],
        result["targets"],
        result["skip_rewrites"],
    )
    return ref_report, summary


def build_ref_report(plan: dict, root: Path) -> dict:
    """Per-source reference risks for the human gate (render + REPL).

    The same un-capped stream scan `apply` will run (Phase 3): a move
    whose path (or any path under it, for a directory move) is cited
    from a stay-put file — and not covered by a chain or an in-plan
    edit — lands in the report so the human sees it before approving.
    """
    ref_report, _summary = build_gate_report(plan, root)
    return ref_report


def _no_rewrite_set(plan: dict, rel_paths: list[Path]) -> set[str]:
    """Pre-move paths excluded from the rewrite scan by "rewrite": false
    moves (B1) — the move's own subtree plus, when it is a directory,
    every file under it."""
    no_rewrite: set[str] = set()
    for m in plan.get("moves") or []:
        if isinstance(m, dict) and m.get("rewrite") is False:
            no_rewrite.add(str(m["from"]))
            prefix = str(m["from"]).rstrip("/") + "/"
            no_rewrite.update(str(rp) for rp in rel_paths if str(rp).startswith(prefix))
    return no_rewrite


def _rewrite_inputs(
    root: Path,
    rel_paths: list[Path],
    plan: dict,
    chains: dict[str, str],
    exclude: set[str] | frozenset[str] = frozenset(),
) -> tuple[
    list[Path],
    set[str],
    set[str],
    dict[str, set[int]],
    dict[str, list[tuple[int, int, int]]],
]:
    """Scan files + edit-skip set for the citation rewrite pass.

    Shared by build_rewrite_summary (the render's numbers) and _apply_plan
    (the run's writes) so the gate and the execution can never disagree.

    scan: corpus rel paths (moved or not — content is pre-move either
    way), then the files under declared external_roots AND scan_paths
    (Phase 6: ignored subtrees the agent explicitly opened). Declared
    dirs are deduped against the corpus and each other by resolved
    absolute path — a file scanned twice would be rewritten twice, and
    the second write would find its old text gone ("rewrite site
    vanished mid-apply").

    A move may carry "rewrite": false (B1): a vendored subtree whose
    internal references move WITH it — rewriting its internal links
    would be wrong (the 0.1.6 report's prototype-v3 case). Its files
    are scanned RECORD-ONLY since C6 (the gate shows what a vendored
    subtree cites) and never rewritten; the third return value is that
    pre-move exclusion set, so the gate can label why a citing file
    got no rewrite.

    exclude: corpus-relative paths of THIS RUN'S OWN artifacts (the
    plan file, the gate render, the rewrites JSON) when they sit
    inside the corpus — rewriting the plan being applied would corrupt
    the audit trail mid-run.

    Returns (scan, edit_skip, no_rewrite, owned_lines, first_spans) —
    owned_lines keyed by pre-move scan path (C3: lines the pass leaves
    to the plan's edits); first_spans the D1 first-occurrence spans
    (the only citations on an owned line the edit actually fixes).

    edit_skip: post-move paths named in plan edits — returned for the
    risk join (a citation the pass could not take in an edit file
    reads edit_covered: the hand edit runs after the rewrites), NOT
    for excluding the files: the pass scans them like any other (C3)
    but skips the LINES their edits own, and validate_plan checks each
    find against the same protected view (C4).
    """
    scan: list[Path] = list(rel_paths)
    no_rewrite = _no_rewrite_set(plan, rel_paths)
    # C6: a "rewrite": false subtree stays IN the scan — record-only —
    # so the gate can show what it cites; the pass just never rewrites
    # it (plan_rewrites' record_only). Edit files stay in the scan too
    # (C3): edits run after the pass, in the same file.
    if exclude:
        scan = [rp for rp in scan if str(rp) not in exclude]
    declared: list[str] = []
    for entry in plan.get("external_roots") or []:
        # Normalized plans carry {"path": ...}; raw hand-built ones may
        # still use plain strings (the render tolerates both).
        declared.append(entry.get("path") if isinstance(entry, dict) else entry)
    declared.extend(plan.get("scan_paths") or [])
    if declared:
        # Built only when something is declared: resolving every corpus
        # path costs a stat walk, and with no declared dirs there is
        # nothing to dedup against.
        seen = {(root / rp).resolve() for rp in rel_paths}
        for raw_dir in declared:
            for tp in _text_files_under(Path(raw_dir)):
                rp = tp.resolve()
                if rp not in seen:
                    seen.add(rp)
                    scan.append(tp)
    edit_skip = {_post_move_path(e["file"], chains) for e in plan.get("edits", [])}
    edit_skip |= {str((root / f).resolve()) for f in edit_skip if not os.path.isabs(f)}
    # C3: the lines each edit owns are excluded from the pass (keyed by
    # pre-move scan path) — the edit runs after and does that work.
    owned_lines = edit_skip_lines(root, plan.get("edits", []) or [], chains)
    # D1: the first-occurrence spans — the only owned-line citations the
    # edit actually fixes.
    first_spans = edit_first_spans(root, plan.get("edits", []) or [])
    return scan, edit_skip, no_rewrite, owned_lines, first_spans


def _edit_final_path(plan_file: str, chains: dict[str, str]) -> str:
    """An edit target's post-move location — the ONE resolution path.

    plan.py:681-683's contract: edit.file is the move-SOURCE path when
    the file moves in this plan; the edit is applied post-move at the
    final location. `_post_move_path` covers both explicit file moves
    (chain keys) AND directory-move prefixes — a file under a moved
    DIRECTORY is not a chain key, so the naive chains.get() missed it
    and execute read the stale pre-move path (the 0.1.9 report's D9:
    `edit_unreadable` on 3 edits the dry run had passed). Validate, the
    dry run and execute all resolve edit targets through this wrapper,
    so the three surfaces can never disagree about where an edit lands.
    """
    return _post_move_path(plan_file, chains)


def _would_edit_rows(
    root: Path,
    chains: dict[str, str],
    rewrites: dict[str, list[dict]],
    ordered_edits: list[dict],
) -> list[dict]:
    """Dry-run would_edit rows with the final text of the edited line.

    D8: `{file, reason}` told the agent WHERE but not WHAT — the final
    text of an edited line was reviewable only by applying. Each row
    now carries the resolved post-move path (D9's mapper, so a
    dir-move descendant no longer names its stale source), the owned
    line number, and before/after line text: `before` is the line AS
    THE EDIT SEES IT — the post-rewrite, pre-edit view (the pass's own
    replacements for the file applied first, the edit's owned lines
    excluded from them by construction) — and `after` the same line
    with the edit's single replace applied. A find that lands on no
    single line (a validate-refused plan never gets here; a find
    spanning a newline has no single line) leaves them unset.
    """
    rows: list[dict] = []
    for e in ordered_edits:
        f = str(e["file"])
        final = _edit_final_path(f, chains)
        row: dict = {"file": final, "reason": e.get("reason", "")}
        pre_loc = Path(f) if Path(f).is_absolute() else root / f
        try:
            text = pre_loc.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            rows.append(row)
            continue
        reps = rewrites.get(final) or []
        view = text
        if reps:
            applied = _apply_line_replacements(text, reps)
            if applied is not None:
                view = applied
        idx = view.find(str(e["find"]))
        if idx != -1:
            line_no = view.count("\n", 0, idx) + 1
            view_lines = view.splitlines()
            after_lines = view.replace(
                str(e["find"]), str(e["replace"]), 1
            ).splitlines()
            if line_no <= len(view_lines) and line_no <= len(after_lines):
                row["line"] = line_no
                row["before"] = view_lines[line_no - 1][:_SNIPPET_CAP]
                row["after"] = after_lines[line_no - 1][:_SNIPPET_CAP]
        rows.append(row)
    return rows


def _edit_target(root: Path, file_str: str) -> Path:
    """Absolute path of an edit target.

    validate_plan normalizes in-root edit files to corpus-relative posix
    paths and external ones (the plan declares external_roots) to
    absolute paths — so the absolute/relative split IS the in/external
    split, and undo replays through the same resolution.
    """
    p = Path(file_str)
    return p if p.is_absolute() else corpus_path(root, p)


def _read_edit_text(path: Path, label: str) -> str:
    """Read an edit target strict-UTF-8, bytes → decode.

    The old `read_text(errors="replace")` silently turned every
    undecodable byte into U+FFFD and the write re-encoded the whole
    file — unrecoverable corruption on e.g. latin-1 corpora (undo
    restores by string-replace; the journaled sha256_before is never
    consulted). Refusing is the only safe move: an edit whose target
    cannot be decoded raises edit_not_utf8 instead.
    """
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise SlopKeeperError("edit_unreadable", f"cannot read {label}: {err}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise SlopKeeperError(
            "edit_not_utf8",
            f"{label} is not valid UTF-8: {err}",
            "Convert the file to UTF-8 first, or rewrite the citation by hand.",
        )


def _unwritable_targets(
    root: Path,
    chains: dict[str, str],
    rewrites: dict[str, list[dict]],
    edits: list[dict],
) -> set[str]:
    """Rewrite/edit targets that deny writes (B7 preflight).

    A target that does not exist yet (a post-move path before the moves)
    is judged by its nearest existing ancestor directory — the move will
    mkdir the rest. os.access is best effort (it lies for root) — the
    point is catching the sandbox deny-list and read-only-mount cases
    before a partial apply, not a guarantee.
    """
    bad: set[str] = set()

    def _probe_writable(p: Path) -> bool:
        probe = p if p.exists() else p.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return os.access(probe, os.W_OK)

    for report_rel in rewrites:
        if not _probe_writable(corpus_path(root, Path(report_rel))):
            bad.add(report_rel)
    for e in edits:
        final = _edit_final_path(e["file"], chains)
        if not _probe_writable(_edit_target(root, final)):
            bad.add(final)
    return bad


def _tree_has_files(p: Path) -> bool:
    """True when any FILE (visible or ignored) exists anywhere under p.

    D12's removal qualifier: empty directories — ignored or not — are
    removable; a file is data and blocks the removal loudly. The
    0.1.9 shape's leftover .claude/.cc-writes dirs are empty and
    remove cleanly; a stray .DS_Store refuses with the dir named.
    """
    for _dirpath, _dirnames, filenames in os.walk(p):
        if filenames:
            return True
    return False


def _empty_dir_shape(p: Path) -> list[str]:
    """Every subdirectory under p, relative and sorted (D12).

    A removal only ever runs on a tree with NO files at all, so the
    shape is pure empty directories — recorded in the rmdir journal op
    so undo recreates the exact structure (misc/.claude/.cc-writes and
    friends), not just the top-level dir.
    """
    out: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(p):
        d = Path(dirpath)
        for dn in dirnames:
            out.append((d / dn).relative_to(p).as_posix())
    return sorted(out)


def _merge_dir_move(
    src_abs: Path,
    dst_abs: Path,
    src: str,
    dst: str,
    root: Path,
    jf: "object",
    errors: list[dict],
    moved: list[dict],
) -> None:
    """E1: merge a directory move into an EXISTING destination folder.

    shutil.move would NEST the source dir under the destination
    (`projects/intake/intake/`) when the destination already exists —
    and earlier moves of the same plan create it (the 0.2.0 field
    report's charts/*.png → projects/intake/ before misc/intake →
    projects/intake). The plan's model — validate, the dry run and the
    rewrites — is the FLAT path, so the merge moves the children file
    by file, one journaled move op each (undo restores them exactly),
    refusing a name collision loudly. The vacated source dir is pruned
    along with any empty parents, the same way undo prunes.
    """
    dst_abs.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_abs.iterdir()):
        child_src = f"{src.rstrip('/')}/{child.name}"
        child_dst = f"{dst.rstrip('/')}/{child.name}"
        child_dst_abs = dst_abs / child.name
        if child_dst_abs.exists():
            errors.append(
                {
                    "src": child_src,
                    "error": f"merge collision: {child_dst} already exists",
                }
            )
            continue
        try:
            digest = sha256_file(child) if child.is_file() else None
            shutil.move(str(child), str(child_dst_abs))
        except OSError as err:
            errors.append({"src": child_src, "error": f"OSError: {err}"})
            continue
        try:
            jf.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "src": child_src,
                        "dst": child_dst,
                        "sha256": digest,
                        "op": "move",
                    }
                )
                + "\n"
            )
        except OSError as err:
            errors.append(
                {"src": child_src, "error": f"moved but journal write failed: {err}"}
            )
            continue
        moved.append({"from": child_src, "to": child_dst})
    # The source dir is empty now — remove it, matching the wholesale
    # move (which IS the rename: the source is gone). Empty PARENT
    # dirs are left alone: only remove_emptied removes those, and
    # undo recreates the source on the child restores either way.
    try:
        src_abs.rmdir()
    except OSError:
        pass


def _execution_order(chains: dict[str, str]) -> list[tuple[str, str]]:
    """Stable topological order over resolved move pairs.

    A move whose destination is another pending move's SOURCE runs after
    that source-side move: the src-side move frees the path, so the file
    chain a→b, b→c executes b→c then a→b and nothing is ever overwritten
    (the old sorted order renamed a.md onto b.md, destroying b.md's
    content silently). Ties keep sorted order. resolve_chains has
    already refused cycles, so every pass makes progress; the fallback
    keeps sorted order if one ever doesn't.
    """
    remaining = sorted(chains.items())
    order: list[tuple[str, str]] = []
    # Set probe instead of a scan of every remaining pair per item: a
    # rule-expanded plan can carry tens of thousands of moves, and the
    # O(n²) readiness test would dominate the run. Same rule — a pair is
    # ready iff its destination is no pending move's source.
    pending_srcs = {s for s, _ in remaining}
    while remaining:
        ready = [(s, d) for s, d in remaining if d not in pending_srcs]
        if not ready:  # unreachable post-resolve_chains: cycle safeguard
            order.extend(remaining)
            break
        order.extend(ready)
        done = {s for s, _ in ready}
        pending_srcs -= done
        remaining = [pair for pair in remaining if pair[0] not in done]
    return order


def apply_plan(
    plan_path: Path,
    corpus_root: Path,
    execute: bool = False,
    force: bool = False,
    allow_partial: bool = False,
    strict_references: bool = False,
    journal_path: Path | None = None,
) -> dict:
    """Dry-run or execute a plan (DESIGN.md §5.5 tending lock).

    Execute mode holds the advisory lock on .slopkeeper/tend.lock across
    the WHOLE pre-flight+execute span — acquisition happens before the
    fingerprint gate, and tend_lock's finally releases it, so a failed
    apply never wedges the corpus. Dry-runs take no lock: two agents may
    plan simultaneously; only journal append + moves/edits + manifest
    write serialize.
    """
    root = corpus_root.resolve()
    with tend_lock(root) if execute else nullcontext():
        return _apply_plan(
            plan_path,
            root,
            execute=execute,
            force=force,
            allow_partial=allow_partial,
            strict_references=strict_references,
            journal_path=journal_path,
        )


def _render_mismatch_warning(root: Path, plan_sha256: str) -> str | None:
    """§5.6: RENDER_WARNING iff the plan's bytes differ from the last render.

    No sidecar (never rendered, or corrupt/foreign-schema — see
    load_render_sidecar) means nothing to compare: no warning, no crash.
    """
    sidecar = load_render_sidecar(root)
    if not sidecar:
        return None
    rendered = sidecar.get("plan_sha256")
    if not rendered or rendered == plan_sha256:
        return None
    return RENDER_WARNING


def _apply_plan(
    plan_path: Path,
    corpus_root: Path,
    execute: bool = False,
    force: bool = False,
    allow_partial: bool = False,
    strict_references: bool = False,
    journal_path: Path | None = None,
) -> dict:
    root = corpus_root.resolve()
    plan = load_plan(plan_path)
    plan = validate_plan(plan, root, allow_partial=allow_partial)
    # §5.6 provenance: the hash of the exact plan bytes being applied —
    # recorded in the manifest, the journal header, and checked against
    # the last render. load_plan just read this file, so it is hashable.
    plan_sha256 = sha256_file_exact(plan_path)
    render_warning = _render_mismatch_warning(root, plan_sha256)

    # 1. Staleness gate. `corpus_fingerprint` is the canonical key;
    # `inventory_sha256` is a read-only alias — plans authored from the
    # pre-T1 template carry it, and both name the same inventory
    # fingerprint (inventory.py writes corpus_fingerprint(root) under
    # either name). Reading only one spelling let template-authored
    # plans apply silently against a drifted corpus.
    current = corpus_fingerprint(root)
    based_on = plan.get("based_on", {}) or {}
    recorded = based_on.get("corpus_fingerprint") or based_on.get("inventory_sha256")
    if recorded and recorded != current and not force:
        raise SlopKeeperError(
            "corpus_changed",
            f"corpus changed since the plan was authored "
            f"(plan: {recorded[:16]}, now: {current[:16]})",
            "Re-inventory and regenerate the plan, or pass --force if the "
            "changes are yours and expected.",
        )

    chains = resolve_chains(plan["moves"])

    # 1b. Edit pre-flight: validate_plan already ran the shared
    # check_edits_after_rewrites — each edit's find must occur exactly
    # once in the file's POST-rewrite text (C3/C4's one rule: edits run
    # after the pass, and apply re-validates, so the two surfaces can
    # never disagree). Nothing left to check here.
    edits = plan.get("edits", [])

    # 2. Pre-flight every destination on disk.
    collisions = []
    for src, dst in chains.items():
        if src == dst:
            continue
        dst_abs = corpus_path(root, Path(dst))
        if dst_abs.exists() and dst not in chains:
            collisions.append({"src": src, "dst": dst})
    if collisions:
        raise SlopKeeperError(
            "dst_exists",
            f"{len(collisions)} destination(s) already exist: "
            f"{[c['dst'] for c in collisions][:5]}",
            "Choose different destinations or resolve the collision in the plan.",
        )

    # 3. Reference risk scan + 3b. the citation rewrite pass — ONE
    # engine now (B2): the pass runs first and its per-site records
    # feed the risk rows, so every "cited" row can say whether the
    # pass fixes it and why not when it does not. Un-capped and loud
    # (Phase 3): the stream scan is linear and ALWAYS runs — counts
    # land in reference_scan, and there is no silent skip whose empty
    # risk set could read as "no risks". D1: a file the plan edits is
    # NOT wholesale-added to moved — its citations are judged per-row
    # (owned-line branch); rows the pass records none of still read
    # edit_covered via the edit_skip join in enriched_risk.
    scan_t0 = time.monotonic()
    refs_full = scan_references_stream(root)
    scan_stats = refs_full.pop("stats", {"files_scanned": 0, "hits": 0})
    rel_paths = _all_rel_paths(root)
    # Dir moves exclude their descendants: a citing file under the
    # moved dir is rewritten by this same apply, not breakage. A
    # "rewrite": false subtree moves but its citations are NOT
    # rewritten — its outbound citations to moving paths are breakage
    # (B1's own trap), so it does not read as self-covered.
    moved = _moved_files_and_dsts(chains, rel_paths, root) - _no_rewrite_set(
        plan, rel_paths
    )
    reference_scan = {
        "status": "ok",
        "files_scanned": scan_stats["files_scanned"],
        "elapsed_ms": int((time.monotonic() - scan_t0) * 1000),
    }

    # Citation rewrite pass (Phase 4): default-on; "off" is the
    # explicit opt-out (validated). Computed once against the PRE-move
    # disk (moves don't change content) and reported under POST-move
    # paths; execute writes them after the moves, before the edits.
    # Edit-covered files are skipped: edits win and run after (no
    # double application).
    rewrite_mode = (plan.get("rewrites") or {}).get("mode", "auto")
    rewrites: dict[str, list[dict]] = {}
    rewrite_skipped: list[dict] = []
    rewrite_targets: dict[str, int] = {}
    pass_result: dict = {
        "rewrites": {},
        "skipped": [],
        "skip_rewrites": [],
        "targets": {},
        "sites": {},
        "unreadable": set(),
        "edit_skip": set(),
        "no_rewrite": set(),
    }
    if rewrite_mode != "off":
        # Scan files + edit skips via the shared builder (Phase 6):
        # external_roots AND scan_paths join the pass, deduped — the
        # gate's numbers and this run's writes come from one code path.
        # The plan file is excluded when it lives inside the corpus
        # (rewriting the plan being applied would corrupt the audit
        # trail), as are the last render's own artifacts.
        exclude = _run_artifacts(root, str(plan_path))
        sidecar = load_render_sidecar(root)
        if sidecar:
            exclude = frozenset(exclude | set(sidecar.get("artifacts") or []))
        pass_result = _run_rewrite_pass(
            root,
            plan,
            chains,
            rel_paths,
            exclude=exclude,
        )
        rewrites = pass_result["rewrites"]
        rewrite_skipped = pass_result["skipped"]
        rewrite_targets = pass_result["targets"]
        # Count assertions (Phase 5): the dry-count pass precedes ALL
        # file writes — a mismatch refuses before a single move.
        for entry in plan.get("external_roots") or []:
            expect = (entry or {}).get("expect")
            if expect is None:
                continue
            root_str = str(entry["path"]).rstrip("/")
            actual = sum(
                len(v) for k, v in rewrites.items() if str(k).startswith(root_str + "/")
            )
            if actual != expect:
                raise SlopKeeperError(
                    "rewrite_expect_mismatch",
                    f"external root {entry['path']} declares expect={expect} "
                    f"but the rewrite pass computed {actual} citation rewrites",
                    'Verify the count with `refs --scan-path`, fix "expect", '
                    "or drop the assertion.",
                )

    # Enriched risk rows (B2): only moves with citations the plan does
    # NOT fix are risks — a covered file is not a risk. Each row lists
    # its citing sites with {file, line, text, status, why}.
    ref_report: dict[str, dict] = {}
    gate_index = CitationIndex(rel_paths)
    for src in chains:
        risk = _enriched_risk(
            src,
            refs_full,
            moved,
            pass_result["sites"],
            pass_result["unreadable"],
            pass_result["edit_skip"],
            pass_result["no_rewrite"],
            root=root,
            index=gate_index,
        )
        if risk["uncovered_references"] > 0:
            ref_report[src] = risk

    # --strict-references: a risk is satisfied when the rewrite pass
    # covers it — same rule as an edit covering it today. The enriched
    # rows make this per-CITATION: a move whose every citing line is
    # rewritten/identical/edit-covered passes; one ambiguous line in a
    # cited file refuses (never guessed).
    if ref_report and strict_references and execute:
        raise SlopKeeperError(
            "referenced_moves",
            f"{len(ref_report)} move(s) keep citations the plan does not fix: "
            f"{sorted(ref_report)[:5]}",
            "Rewrite the references (edits for the skipped lines), move the "
            "referencing files too, or accept the risk and re-run without "
            "--strict-references.",
        )

    def _rewrite_summary() -> dict:
        return _shape_rewrite_summary(
            rewrite_mode,
            rewrites,
            rewrite_skipped,
            rewrite_targets,
            pass_result["skip_rewrites"],
        )

    # 2b. External edits run AFTER the in-root ones: in-root edits resolve
    # through the move chains, external ones never do, and a fixed order
    # keeps the journal — and undo's reverse replay — deterministic
    # regardless of the plan's edit order.
    ordered_edits = [e for e in edits if not Path(e["file"]).is_absolute()] + [
        e for e in edits if Path(e["file"]).is_absolute()
    ]

    if execute:
        # Writability preflight (B7): a sandboxed environment (deny-listed
        # .claude/skills, a read-only mount) would otherwise fail PARTWAY
        # through the apply — moves done, rewrites refused. Every rewrite
        # and edit target is checked BEFORE the first move; anything
        # unwritable refuses the whole run with the paths named.
        unwritable = sorted(_unwritable_targets(root, chains, rewrites, ordered_edits))
        if unwritable:
            raise SlopKeeperError(
                "target_unwritable",
                f"{len(unwritable)} rewrite/edit target(s) are not writable: "
                f"{unwritable[:5]}",
                "The environment denies writes to these paths (sandbox "
                "deny-list, read-only mount). Run apply outside the sandbox "
                "or grant write access — nothing was moved.",
            )

    if not execute:
        dry = {
            "mode": "dry-run",
            "reference_scan": reference_scan,
            "would_move": [{"from": s, "to": d} for s, d in sorted(chains.items())],
            "move_count": len(chains),
            "would_edit": _would_edit_rows(root, chains, rewrites, ordered_edits),
            "reference_risks": ref_report,
            "fingerprint": current,
        }
        if rewrite_mode != "off":
            summary = _rewrite_summary()
            # Per-file would-be counts; zero writes happen in a dry run.
            dry["would_rewrite"] = summary["per_file"]
            dry["rewrite_skipped"] = summary["skipped"]
            # C2: the single bad lines — skipped spans citing a moving
            # path — for hand editing.
            dry["skip_rewrites"] = summary["skip_rewrites"]
        if render_warning:
            dry["render_warning"] = render_warning
        return dry

    # 4. Execute: mkdir, move, journal; then edits, journaled for undo.
    # _execution_order runs chain hops src-side first (b→c before a→b)
    # so a hop never lands on a path another pending move still owns.
    # The journal records the ACTUAL executed pair per move — both hops
    # of a file chain, not a rewrite — so undo replays it exactly.
    # First line is the §5.6 provenance header binding the journal to
    # the plan's exact bytes; undo skips non-move ops, so it tolerates
    # the header wherever it sits.
    journal = journal_path or _default_journal(root)
    journal.parent.mkdir(parents=True, exist_ok=True)
    moved, edits_applied, errors, rewrote = [], [], [], []
    with journal.open("a", encoding="utf-8") as jf:
        jf.write(
            json.dumps({"op": "plan", "sha256": plan_sha256, "ts": time.time()}) + "\n"
        )
        for src, dst in _execution_order(chains):
            src_abs = corpus_path(root, Path(src))
            dst_abs = corpus_path(root, Path(dst))
            if not src_abs.exists():
                errors.append({"src": src, "error": "vanished mid-apply"})
                continue
            # E1: the destination exists by the time this move runs —
            # an earlier move of this same plan created it. Merging
            # keeps the flat final paths the whole tool models;
            # shutil.move would nest the directory instead.
            if src_abs.is_dir() and dst_abs.exists():
                _merge_dir_move(src_abs, dst_abs, src, dst, root, jf, errors, moved)
                continue
            # Mid-run IO failures (EACCES, disk full) cannot be
            # pre-flighted: all-or-nothing covers pre-flight-checked
            # failures only. Match the vanished-mid-apply precedent —
            # record and continue. Earlier hops stay journaled and
            # undoable; the envelope reports partial and points at the
            # journal (T5).
            try:
                digest = sha256_file(src_abs)
                dst_abs.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_abs), str(dst_abs))
            except OSError as err:
                errors.append({"src": src, "error": f"OSError: {err}"})
                continue
            try:
                jf.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "src": src,
                            "dst": dst,
                            "sha256": digest,
                            "op": "move",
                        }
                    )
                    + "\n"
                )
            except OSError as err:
                # The move happened; record it, and say it is
                # unjournaled — undo cannot restore this hop.
                errors.append(
                    {"src": src, "error": f"moved but journal write failed: {err}"}
                )
            moved.append({"from": src, "to": dst})

        # 4b. Citation rewrites (Phase 4): after moves, before edits.
        # Each file is written ONCE with all its line replacements;
        # zero-hit files are absent from the plan (untouched). The
        # journal entry is the undo recipe: per-line old/new pairs.
        for report_rel in sorted(rewrites):
            reps = rewrites[report_rel]
            fp = corpus_path(root, Path(report_rel))
            try:
                text = fp.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as err:
                errors.append(
                    {"file": report_rel, "error": f"rewrite unreadable: {err}"}
                )
                continue
            lines = text.splitlines(keepends=True)
            per_line: dict[int, list[dict]] = {}
            for rep in reps:
                per_line.setdefault(rep["line"], []).append(rep)
            ok = True
            for lineno in sorted(per_line):
                idx = lineno - 1
                if idx >= len(lines):
                    ok = False
                    break
                for rep in per_line[lineno]:
                    if rep["old"] not in lines[idx]:
                        ok = False
                        break
                    lines[idx] = lines[idx].replace(rep["old"], rep["new"], 1)
                if not ok:
                    break
            if not ok:
                errors.append(
                    {"file": report_rel, "error": "rewrite site vanished mid-apply"}
                )
                continue
            try:
                # newline="" keeps every byte the replacement didn't
                # touch — endings ride along in the line strings.
                fp.write_text("".join(lines), encoding="utf-8", newline="")
            except OSError as err:
                errors.append({"file": report_rel, "error": f"OSError: {err}"})
                continue
            try:
                jf.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "op": "rewrite",
                            "file": report_rel,
                            "replacements": reps,
                            "count": len(reps),
                        }
                    )
                    + "\n"
                )
            except OSError as err:
                errors.append(
                    {
                        "file": report_rel,
                        "error": f"rewritten but journal write failed: {err}",
                    }
                )
            rewrote.append({"file": report_rel, "count": len(reps)})

        for e in ordered_edits:
            final = _edit_final_path(e["file"], chains)
            fp = _edit_target(root, final)
            try:
                before = sha256_file(fp)
                text = _read_edit_text(fp, final)
                fp.write_text(
                    text.replace(e["find"], e["replace"], 1), encoding="utf-8"
                )
            except OSError as err:
                errors.append({"file": final, "error": f"OSError: {err}"})
                continue
            except SlopKeeperError as err:
                # _read_edit_text wraps mid-run read failures
                # (edit_unreadable / edit_not_utf8) — same treatment:
                # recorded and continued, never a crash mid-apply.
                errors.append({"file": final, "error": f"{err.code}: {err.message}"})
                continue
            try:
                jf.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "op": "edit",
                            "file": final,
                            "find": e["find"],
                            "replace": e["replace"],
                            "sha256_before": before,
                        }
                    )
                    + "\n"
                )
            except OSError as err:
                errors.append(
                    {"file": final, "error": f"edited but journal write failed: {err}"}
                )
            edits_applied.append({"file": final})

        # 4c. remove_emptied (D12): directories the plan emptied —
        # journaled per-dir, undoable, and only when the run is
        # otherwise clean (a partial run removes nothing: loud failure
        # first, the T5 precedent).
        removed_dirs: list[str] = []
        removal_ran = False
        if plan.get("remove_emptied") and not errors:
            removal_ran = True
            for d in plan.get("remove_emptied_dirs") or []:
                dp = corpus_path(root, Path(d))
                if not dp.is_dir():
                    continue
                if _tree_has_files(dp):
                    errors.append(
                        {
                            "dir": d,
                            "error": "remove_emptied_refused: still holds files",
                        }
                    )
                    continue
                try:
                    empty_shape = _empty_dir_shape(dp)
                    shutil.rmtree(dp)
                except OSError as err:
                    errors.append({"dir": d, "error": f"OSError: {err}"})
                    continue
                try:
                    jf.write(
                        json.dumps(
                            {
                                "ts": time.time(),
                                "op": "rmdir",
                                "dir": d,
                                "empty_dirs": empty_shape,
                            }
                        )
                        + "\n"
                    )
                except OSError as err:
                    errors.append(
                        {"dir": d, "error": f"removed but journal write failed: {err}"}
                    )
                    continue
                removed_dirs.append(d)

    # 5. Record the tended state so `status` can diff future slop cheaply.
    unresolved = [
        {"path": u.get("path"), "reason": u.get("reason", "")}
        for u in (plan.get("unresolved") or [])
        if isinstance(u, dict)
    ]
    try:
        write_manifest(
            root,
            unresolved=unresolved,
            plan_path=str(plan_path),
            plan_sha256=plan_sha256,
        )
    except OSError as err:
        # The moves/edits above are done and journaled; a manifest
        # failure is reported, never a crash on top of them (the
        # manifest is a rebuildable snapshot — `status --record`).
        errors.append({"error": f"manifest update failed: {err}"})

    executed = {
        "mode": "executed",
        "reference_scan": reference_scan,
        "moved_count": len(moved),
        "moved": moved,
        "errors": errors,
        # Honest partial flag: true iff anything failed mid-run. The
        # CLI raises the exit code and points `next` at the undo when
        # this is set (T5: a non-empty errors list must never exit 0).
        "partial": bool(errors),
        "reference_risks": ref_report,
        "journal": str(journal),
        "fingerprint": current,
    }
    if removal_ran:
        executed["removed_dirs"] = removed_dirs
    if rewrite_mode != "off":
        summary = _rewrite_summary()
        # Per-file counts + total N across M files; the journal holds
        # the per-line recipe and undo replays it exactly.
        executed["rewrites"] = {
            "count": summary["count"],
            "files": len(summary["per_file"]),
            "per_file": summary["per_file"],
            "skipped": summary["skipped"],
            "skip_rewrites": summary["skip_rewrites"],
        }
    if render_warning:
        executed["render_warning"] = render_warning
    return executed


def undo_journal(journal_path: Path, corpus_root: Path, execute: bool = False) -> dict:
    """Reverse a journal (§5.5: execute mode runs under the tend lock).

    Dry-run undo is a pure read: no lock, safe beside a running tend.
    """
    root = corpus_root.resolve()
    with tend_lock(root) if execute else nullcontext():
        return _undo_journal(journal_path, root, execute=execute)


def _undo_journal(journal_path: Path, corpus_root: Path, execute: bool = False) -> dict:
    root = corpus_root.resolve()
    entries = []
    try:
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    except OSError as e:
        raise SlopKeeperError("journal_unreadable", str(e))
    if not entries:
        raise SlopKeeperError("journal_empty", "nothing to undo")

    ops = []
    for e in reversed(entries):
        if e.get("op") == "rewrite":
            # Reverse replay: per journaled line, `new` must be present
            # exactly the expected number of times (the rewrite's
            # fingerprint) — anything else means the content changed
            # since, and guessing could revert the wrong occurrence or
            # half of one. Refuse whole (already_changed), never a
            # partial revert. Unknown ops (future kinds) still fall
            # through to the skip below.
            fp = _edit_target(root, e["file"])
            if not fp.exists():
                ops.append({"file": e["file"], "status": "already_absent"})
                continue
            try:
                text = fp.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                ops.append({"file": e["file"], "status": "unreadable"})
                continue
            lines = text.splitlines(keepends=True)
            per_line: dict[int, list[dict]] = {}
            for rep in e.get("replacements", []):
                per_line.setdefault(rep["line"], []).append(rep)
            ok = True
            for lineno, reps in per_line.items():
                idx = lineno - 1
                if idx >= len(lines):
                    ok = False
                    break
                for rep in reps:
                    expected = sum(1 for r in reps if r["new"] == rep["new"])
                    if lines[idx].count(rep["new"]) != expected:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                ops.append({"file": e["file"], "status": "already_changed"})
                continue
            if execute:
                for lineno in sorted(per_line, reverse=True):
                    idx = lineno - 1
                    for rep in per_line[lineno]:
                        lines[idx] = lines[idx].replace(rep["new"], rep["old"], 1)
                try:
                    fp.write_text("".join(lines), encoding="utf-8", newline="")
                except OSError:
                    ops.append({"file": e["file"], "status": "unreadable"})
                    continue
            ops.append(
                {"file": e["file"], "status": "reverted" if execute else "would_revert"}
            )
            continue
        if e.get("op") == "edit":
            fp = _edit_target(root, e["file"])
            if not fp.exists():
                ops.append({"file": e["file"], "status": "already_absent"})
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                ops.append({"file": e["file"], "status": "unreadable"})
                continue
            if e["replace"] in text:
                if execute:
                    fp.write_text(
                        text.replace(e["replace"], e["find"], 1), encoding="utf-8"
                    )
                ops.append(
                    {
                        "file": e["file"],
                        "status": "reverted" if execute else "would_revert",
                    }
                )
            else:
                ops.append({"file": e["file"], "status": "already_changed"})
            continue
        if e.get("op") == "rmdir":
            # D12's removal, reversed: recreate the directory (empty,
            # best-effort) so the restored moves below land inside it.
            dp = corpus_path(root, Path(e["dir"]))
            if dp.exists():
                ops.append({"dir": e["dir"], "status": "already_present"})
                continue
            if execute:
                try:
                    dp.mkdir(parents=True, exist_ok=True)
                    # D12: the removed tree's empty subdirs (including
                    # ignored ones like .claude/.cc-writes) ride in the
                    # journal op — recreate the exact shape.
                    for sub in e.get("empty_dirs") or []:
                        (dp / sub).mkdir(parents=True, exist_ok=True)
                except OSError:
                    ops.append({"dir": e["dir"], "status": "unreadable"})
                    continue
            ops.append(
                {
                    "dir": e["dir"],
                    "status": "recreated" if execute else "would_recreate",
                }
            )
            continue
        if e.get("op") != "move":
            continue
        dst_abs = corpus_path(root, Path(e["dst"]))
        src_abs = corpus_path(root, Path(e["src"]))
        if not dst_abs.exists():
            ops.append({"src": e["src"], "status": "already_absent"})
            continue
        if src_abs.exists():
            ops.append({"src": e["src"], "status": "blocked_dst_occupied"})
            continue
        if execute:
            src_abs.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst_abs), str(src_abs))
            _prune_empty_dirs(dst_abs.parent, root)
        ops.append(
            {"src": e["src"], "status": "restored" if execute else "would_restore"}
        )
    result = {"mode": "executed" if execute else "dry-run", "operations": ops}
    if execute:
        # §5.4: the manifest describes the CURRENT tended state, and the
        # plan this journal came from did not happen — refresh the
        # snapshot so it reflects the post-undo disk.
        result["manifest"] = _refresh_manifest_after_undo(root)
    return result


def _refresh_manifest_after_undo(root: Path) -> str:
    """Rewrite the manifest post-undo; "updated" or "rebuilt".

    `unresolved` survives the reversal (human decisions are not reverted
    by tooling); `last_plan` is cleared (the plan did not happen);
    files/fingerprint/recorded_at are recomputed from disk by
    write_manifest — the same single snapshot code path apply uses.

    A broken manifest (absent or corrupt) never blocks an undo: the
    manifest is a snapshot, always rebuildable, so we simply write a
    fresh one and say so in the result.
    """
    try:
        manifest = load_manifest(root)
    except SlopKeeperError:
        manifest = None
    unresolved = (manifest or {}).get("unresolved") or []
    write_manifest(root, unresolved=unresolved)
    return "updated" if manifest is not None else "rebuilt"


def _prune_empty_dirs(start: Path, root: Path, max_up: int = 4) -> None:
    """Best-effort removal of empty dirs left behind by undo (never root)."""
    d = start
    for _ in range(max_up):
        if d == root or not d.is_dir() or d == d.parent:
            return
        try:
            if any(d.iterdir()):
                return
            d.rmdir()
        except OSError:
            return
        d = d.parent
