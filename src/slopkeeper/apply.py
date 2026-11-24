# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Plan execution: all-or-nothing moves with a journal and undo.

Safety invariants, in order:
1. Nothing happens without --execute (dry-run is the default).
2. A plan must carry a corpus binding (based_on.corpus_fingerprint, A1)
   and the corpus must match it; --force escapes the mismatch, never the
   missing binding (plan_no_based_on) and never a corpus that churned
   under the walk (corpus_unstable, B6).
3. Every move is pre-flighted (collisions, chains, nesting, symlink
   containment — A2) before ANY move, and destinations are re-checked
   immediately before each write. A destination that appears in that
   last window is refused (dst_exists), never clobbered — the move is
   an atomic no-replace link/unlink (B4).
4. Every executed move is journaled with a content hash, fsync'd line
   by line under a per-tend filename (A4), and undo VERIFIES that hash
   before restoring: a file edited after the move comes back with
   `content_changed` on its row rather than a silent lie (B1).
5. Nothing is ever deleted; a failed mid-run is recoverable via the journal.
6. A mid-run IO failure is journaled-partial, never silent: recorded in
   `errors`, the envelope carries `partial: true`, the process exits
   non-zero, and `next` points at the exact undo command. An edit whose
   find-text no longer occurs exactly once at write time is a recorded
   `edit_stale`, never a guessed replacement (B3).
7. Soft conditions that no check can refuse are LOUD on the envelope:
   `warnings` names a platform without locking (B7) and a corpus that
   moved between the gate and the first write (B5). The journal header
   carries the plan's fingerprint and the at-execute fingerprint
   alongside the tend id, so any later audit can tell which disk ran.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .ignore import corpus_walk
from .models import (
    SlopKeeperError,
    add_warning,
    corpus_fingerprint_detail,
    corpus_path,
    resolve_chains,
    safe_move_path,
    sha256_file,
    sha256_file_exact,
    state_dir,
    tend_lock,
)
from .plan import (
    based_on_binding,
    load_plan,
    load_render_sidecar,
    preflight_destinations,
    validate_plan,
)
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

# a mismatch between the plan's bytes and the last render is a
# WARNING, never a refusal — the human may hand-edit the plan JSON after
# rendering (the documented AGENT_GUIDE flow). It only makes silent
# render→apply drift visible.
RENDER_WARNING = (
    "plan differs from last render — if the human approved "
    "the render, re-render or get explicit sign-off."
)


def _default_journal(root: Path) -> Path:
    """A journal path per TEND — never a shared file (A4).

    The pre-0.2.7 name was second-granularity UTC plus append mode, so
    two `apply --execute` runs inside one second wrote into ONE journal
    and `undo` reverted both tends at once — the documented "undo one
    tend" model was wrong at second granularity (the 2026-11-23 audit's
    A4). Same-second tends now land in `-2`, `-3`… siblings; the tend id
    (the filename stem) rides in each journal's provenance header, so a
    journal always names the tend it records.

    Called only under the tend lock (execute mode), so the exists()
    probe cannot race another slopkeeper process; an explicitly passed
    `journal_path` (tests, scripted batch runs) keeps its append
    semantics — naming a file is an instruction to share it.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = state_dir(root)
    path = base / f"journal-{ts}.jsonl"
    n = 2
    while path.exists():
        path = base / f"journal-{ts}-{n}.jsonl"
        n += 1
    return path


def _journal_write(jf, doc: dict) -> None:
    """One whole journal line, durable before the next mutation (A4/B2).

    write + flush + os.fsync: the journal is the LAST-RESORT recovery
    mechanism, and a crash between two ops must never find it holding a
    half-written JSON object — the line either reached the disk whole or
    did not exist. Small N per tend, so the fsync tax is negligible
    against the moves it protects. OSError propagates to the caller's
    existing per-op handler (a line that cannot be journaled is recorded
    as `moved but journal write failed`, never swallowed).
    """
    jf.write(json.dumps(doc) + "\n")
    jf.flush()
    os.fsync(jf.fileno())


def _refuse_corpus_churn(health: dict, force: bool) -> int:
    """B6: a corpus that churned under the fingerprint walk is not tended.

    Returns the count of files the walk could not stat (0 = a clean
    walk); raises `corpus_unstable` when it is non-zero unless --force.
    Before 0.2.7 each such file hit a bare `continue`, so the digest
    quietly described LESS than the corpus and the drift gate compared
    two digests of two different disks — a plan authored against a
    stable corpus applied clean while something rewrote it under the
    scan. A churny corpus is not a plan problem, so the refusal names
    the suspect paths and the fix (stop the writer, re-run); --force is
    the operator's explicit "I know, proceed", and even then the count
    rides the envelope as `skipped_unstatable` — never swallowed.
    """
    n = int(health.get("skipped_unstatable") or 0)
    if not n or force:
        return n
    named = list(health.get("skipped_unstatable_paths") or [])
    detail = ""
    if named:
        detail = ": " + ", ".join(named) + (" …" if n > len(named) else "")
    raise SlopKeeperError(
        "corpus_unstable",
        f"{n} file(s) vanished or denied stat while the corpus was walked{detail}",
        "Something is writing to this corpus right now (an editor, a sync "
        "client, a build) — a fingerprint cannot bind a moving target, so "
        "nothing was moved. Stop the writer and re-run; --force proceeds "
        "anyway and records the count on the envelope.",
    )


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
            _journal_write(
                jf,
                {
                    "ts": time.time(),
                    "src": child_src,
                    "dst": child_dst,
                    "sha256": digest,
                    "op": "move",
                },
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


# errno values that mean "hardlinks are not a thing here" rather than
# "something is wrong with these files": EXDEV (cross-device) plus the
# EPERM/EMLINK/ENOTSUP/ENOSYS/EINVAL family filesystems and platforms
# report for an unsupported link(). Anything else is a real failure and
# propagates as an OSError for the caller to record.
_HARDLINK_UNSUPPORTED = frozenset(
    e
    for e in (
        errno.EXDEV,
        errno.EPERM,
        errno.EMLINK,
        getattr(errno, "ENOTSUP", None),
        errno.ENOSYS,
        errno.EINVAL,
        errno.EOPNOTSUPP,
    )
    if e is not None
)


def _move_no_clobber(src_abs: Path, dst_abs: Path) -> None:
    """Atomic no-replace FILE move (B4): the clobber window, closed.

    apply pre-flights every destination, then the move loop runs — and
    a file created by anyone else in that window was silently destroyed,
    because `shutil.move` is a bare rename and POSIX rename REPLACES.
    `os.link` is the stdlib's only atomic no-replace primitive: it fails
    EEXIST when the destination exists, so the race resolves in the
    owner's favor instead of the tool's. Then `os.unlink` removes the
    source, which completes the move (same inode, so content, mode and
    times ride along exactly as a rename would).

    Raises FileExistsError when the destination appeared mid-run — the
    caller records `dst_exists` and refuses THAT move (a partial run,
    never a clobber). Filesystems without hardlinks (and the rare
    cross-device link) fall back to a re-check + shutil.move: no
    stdlib primitive renames across devices, so that path keeps its
    last-moment `lexists` guard and is a last resort, not the default.

    follow_symlinks=False so a symlink SOURCE moves the LINK (its own
    bytes), never a copy of whatever it points at.
    """
    try:
        os.link(src_abs, dst_abs, follow_symlinks=False)
    except FileExistsError:
        raise
    except OSError as err:
        if err.errno not in _HARDLINK_UNSUPPORTED:
            raise
        if os.path.lexists(dst_abs):
            raise FileExistsError(
                errno.EEXIST, "Destination exists", str(dst_abs)
            ) from None
        shutil.move(str(src_abs), str(dst_abs))
        return
    os.unlink(src_abs)


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
    allow_symlinks: bool = False,
) -> dict:
    """Dry-run or execute a plan (tending lock).

    Execute mode holds the advisory lock on .slopkeeper/tend.lock across
    the WHOLE pre-flight+execute span — acquisition happens before the
    fingerprint gate, and tend_lock's finally releases it, so a failed
    apply never wedges the corpus. Dry-runs take no lock: two agents may
    plan simultaneously; only journal append + moves/edits + manifest
    write serialize.

    allow_symlinks (CLI --allow-symlinks) is the deliberate traversal
    flag for the A2 move-path containment policy; default False refuses
    moves through or onto symlinks (symlink_escape).

    A tend that ran WITHOUT the lock (a platform with no fcntl) says so
    on its own envelope — `warnings: ["no_locking_available"]` (B7).
    Failing soft was the old behavior; failing soft IN SILENCE was the
    bug: the agent read the same payload a serialized run produces.
    """
    root = corpus_root.resolve()
    if not execute:
        # Dry-runs take no lock, and no-lock means no locking warning to
        # report: nothing here needed serializing.
        return _apply_plan(
            plan_path,
            root,
            execute=False,
            force=force,
            allow_partial=allow_partial,
            strict_references=strict_references,
            journal_path=journal_path,
            allow_symlinks=allow_symlinks,
        )
    with tend_lock(root) as lock:
        data = _apply_plan(
            plan_path,
            root,
            execute=execute,
            force=force,
            allow_partial=allow_partial,
            strict_references=strict_references,
            journal_path=journal_path,
            allow_symlinks=allow_symlinks,
        )
    if lock.unavailable:
        add_warning(data, "no_locking_available")
    return data


def _render_mismatch_warning(root: Path, plan_sha256: str) -> str | None:
    """RENDER_WARNING iff the plan's bytes differ from the last render.

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
    allow_symlinks: bool = False,
) -> dict:
    root = corpus_root.resolve()
    plan = load_plan(plan_path)
    plan = validate_plan(
        plan,
        root,
        allow_partial=allow_partial,
        require_based_on=True,
        allow_symlinks=allow_symlinks,
    )
    # provenance: the hash of the exact plan bytes being applied —
    # recorded in the manifest, the journal header, and checked against
    # the last render. load_plan just read this file, so it is hashable.
    plan_sha256 = sha256_file_exact(plan_path)
    render_warning = _render_mismatch_warning(root, plan_sha256)

    # 1. Staleness gate — FIRES UNCONDITIONALLY (A1, 0.2.6). The pre-fix
    # form was `if recorded and ...`: a plan with no based_on skipped
    # the gate entirely (no --force needed) and a corpus mutated after
    # authoring applied clean (the 2026-11-23 audit's A1).
    # `corpus_fingerprint` is the canonical key; `inventory_sha256` is a
    # read-only alias — plans authored from the pre-T1 template carry it,
    # and both name the same inventory fingerprint (based_on_binding is
    # the one place that rule lives). validate_plan above already
    # refused a missing binding; this raise is the gate's own refusal —
    # in-process callers that opted out of validation never apply an
    # unbound plan either. --force escapes the MISMATCH only.
    #
    # B6 first: the digest is only evidence if the walk saw everything.
    # Files that vanished or denied stat mid-walk used to `continue`
    # silently, so the two digests being compared could describe two
    # different corpora and still "pass". The count refuses the tend
    # before the comparison is even meaningful.
    current, fp_health = corpus_fingerprint_detail(root)
    unstatable = _refuse_corpus_churn(fp_health, force)
    recorded = based_on_binding(plan)
    if not recorded:
        raise SlopKeeperError(
            "plan_no_based_on",
            "plan has no corpus binding: based_on.corpus_fingerprint is "
            "required (a non-empty string)",
            "author the plan from inventory output; the fingerprint is "
            "what makes the plan safe to apply. Run `slopkeeper inventory "
            "<root> --fast --summary` and record its fingerprint as "
            "based_on.corpus_fingerprint (see `slopkeeper plan schema`).",
        )
    if recorded != current and not force:
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

    # 2. Pre-flight every destination on disk. SHARED with validate_plan
    # (preflight_destinations) so the two surfaces cannot disagree; this
    # call re-observes the disk under the tend lock, after the
    # fingerprint gate — the last look before any write.
    preflight_destinations(root, chains)

    # 2a. Containment pre-flight (A2): validate_plan checked the plan's
    # DECLARED paths; this re-observes every chain-resolved FINAL
    # destination against the live disk under the tend lock — a symlink
    # created between authoring and this run is refused (symlink_escape)
    # before a single write, in dry-run and execute alike. The move loop
    # re-checks each hop immediately before its write as the TOCTOU belt.
    if not allow_symlinks:
        for src, dst in chains.items():
            if src == dst:
                continue
            safe_move_path(dst, root, destination=True)

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
        if unstatable:
            # B6: the dry run says it too — a churny corpus is worth
            # knowing about before the human approves the render.
            dry["skipped_unstatable"] = unstatable
            add_warning(dry, "skipped_unstatable")
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
    # First line is the provenance header binding the journal to
    # the plan's exact bytes; undo skips non-move ops, so it tolerates
    # the header wherever it sits.
    #
    # B5 (minimal forensics): a SECOND fingerprint, taken here under the
    # tend lock after every pre-flight and immediately before the first
    # write. The gate's digest and this one bound the window the run
    # actually wrote into: if they differ, something outside this process
    # moved the disk between the gate and the moves, and the envelope
    # says so (`warnings: ["fingerprint_moved_during_apply"]`). The
    # plan's own binding rides the header beside them, so an audit after
    # the fact can tell which corpus the plan believed, which the gate
    # saw, and which got moved (a --force apply is exactly the case where
    # plan ≠ gate in the header). P3's single-pass scanner makes the
    # extra walk free; until then the audit trail costs one more hashing
    # pass over the corpus.
    at_execute, exec_health = corpus_fingerprint_detail(root)
    unstatable = max(unstatable, _refuse_corpus_churn(exec_health, force))
    journal = journal_path or _default_journal(root)
    journal.parent.mkdir(parents=True, exist_ok=True)
    moved, edits_applied, errors, rewrote = [], [], [], []
    with journal.open("a", encoding="utf-8") as jf:
        _journal_write(
            jf,
            {
                "op": "plan",
                "tend_id": journal.stem,
                "sha256": plan_sha256,
                "ts": time.time(),
                # B5: plan binding / gate digest / at-execute digest —
                # the three instants one tend spans (a torn or edited
                # journal still says which corpus it belongs to).
                "plan_fingerprint": recorded,
                "gate_fingerprint": current,
                "execute_fingerprint": at_execute,
            },
        )
        for src, dst in _execution_order(chains):
            src_abs = corpus_path(root, Path(src))
            dst_abs = corpus_path(root, Path(dst))
            if not src_abs.exists():
                errors.append({"src": src, "error": "vanished mid-apply"})
                continue
            # A2 TOCTOU belt: resolve-and-contain the destination
            # IMMEDIATELY before the write (the audit's fix) — a symlink
            # created after the 2a pre-flight would otherwise be followed
            # by shutil.move, which does not refuse. Raises
            # symlink_escape: a containment failure stops the run rather
            # than writing where no walk has ever looked.
            safe_move_path(dst, root, destination=True, allow_symlinks=allow_symlinks)
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
            #
            # B4: the destination gets one last look INSIDE the write.
            # The pre-flight ran before the reference scan and the whole
            # rewrite pass — minutes of wall clock on a big corpus — and
            # a file created in the window used to be silently renamed
            # over. A FileExistsError here is that owner still holding
            # their file: record `dst_exists`, skip this hop, keep the
            # run undoable.
            try:
                digest = sha256_file(src_abs)
                dst_abs.parent.mkdir(parents=True, exist_ok=True)
                if src_abs.is_dir():
                    # A directory rename cannot be a no-replace link
                    # (no hardlinks to dirs): re-check and refuse if the
                    # path appeared since the pre-flight — shutil.move
                    # would replace an empty dir or bury the tree in a
                    # non-empty one.
                    if os.path.lexists(dst_abs):
                        raise FileExistsError(
                            errno.EEXIST, "Destination exists", str(dst_abs)
                        )
                    shutil.move(str(src_abs), str(dst_abs))
                else:
                    _move_no_clobber(src_abs, dst_abs)
            except FileExistsError as err:
                errors.append(
                    {
                        "src": src,
                        "dst": dst,
                        "error": f"dst_exists: {err.filename or dst} appeared "
                        "mid-apply — not overwritten",
                    }
                )
                continue
            except OSError as err:
                errors.append({"src": src, "error": f"OSError: {err}"})
                continue
            try:
                _journal_write(
                    jf,
                    {
                        "ts": time.time(),
                        "src": src,
                        "dst": dst,
                        "sha256": digest,
                        "op": "move",
                    },
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
                _journal_write(
                    jf,
                    {
                        "ts": time.time(),
                        "op": "rewrite",
                        "file": report_rel,
                        "replacements": reps,
                        "count": len(reps),
                    },
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
            except OSError as err:
                errors.append({"file": final, "error": f"OSError: {err}"})
                continue
            except SlopKeeperError as err:
                # _read_edit_text wraps mid-run read failures
                # (edit_unreadable / edit_not_utf8) — same treatment:
                # recorded and continued, never a crash mid-apply.
                errors.append({"file": final, "error": f"{err.code}: {err.message}"})
                continue
            # B3: re-verify the find text AT WRITE TIME, with the same
            # discipline the rewrite pass uses one loop above (it checks
            # its `old` is still on the line before touching it). The
            # exactly-once rule was validated against the POST-REWRITE
            # view of PRE-MOVE content; between then and now the moves
            # ran, the pass wrote, and anything outside this process may
            # have written too. A `replace(find, x, 1)` on drifted text
            # edits the wrong occurrence — or none at all, silently —
            # and journals the edit as applied either way. Absent or
            # ambiguous now records `edit_stale` and writes nothing:
            # loud, partial, undoable — never a guess.
            find = str(e["find"])
            hits = text.count(find)
            if hits != 1:
                errors.append(
                    {
                        "file": final,
                        "error": f"edit_stale: find text occurred {hits} time(s) "
                        "at write time, expected exactly 1 — the file changed "
                        "after validation; nothing was edited",
                    }
                )
                continue
            try:
                fp.write_text(
                    text.replace(find, str(e["replace"]), 1), encoding="utf-8"
                )
            except OSError as err:
                errors.append({"file": final, "error": f"OSError: {err}"})
                continue
            try:
                _journal_write(
                    jf,
                    {
                        "ts": time.time(),
                        "op": "edit",
                        "file": final,
                        "find": e["find"],
                        "replace": e["replace"],
                        "sha256_before": before,
                    },
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
                    _journal_write(
                        jf,
                        {
                            "ts": time.time(),
                            "op": "rmdir",
                            "dir": d,
                            "empty_dirs": empty_shape,
                        },
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
        # B5: the digest taken immediately before the first write. Equal
        # to `fingerprint` in a clean run; different means the corpus
        # moved between the gate and the moves (--force always looks
        # like this). Both are in the journal header too.
        "fingerprint_execute": at_execute,
    }
    if at_execute != current:
        # B5: drift the gate waved through (or --force waved through)
        # gets a machine-greppable flag on the envelope, not just a
        # number in the journal.
        add_warning(executed, "fingerprint_moved_during_apply")
    if unstatable:
        # B6 under --force: the count survives; silence was the bug.
        executed["skipped_unstatable"] = unstatable
        add_warning(executed, "skipped_unstatable")
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


def undo_journal(
    journal_path: Path,
    corpus_root: Path,
    execute: bool = False,
    force: bool = False,
) -> dict:
    """Reverse a journal (execute mode runs under the tend lock).

    Dry-run undo is a pure read: no lock, safe beside a running tend.

    force (CLI undo --force) is the escape hatch for a journal that is
    malformed in the MIDDLE (B2): the default refuses to replay a
    recovery record someone has edited, because the ops around the torn
    line may be missing. A torn TAIL — the crash-mid-write case — is
    always tolerated. See _undo_journal.
    """
    root = corpus_root.resolve()
    if not execute:
        return _undo_journal(journal_path, root, execute=False, force=force)
    with tend_lock(root) as lock:
        data = _undo_journal(journal_path, root, execute=execute, force=force)
    if lock.unavailable:
        add_warning(data, "no_locking_available")
    return data


def _undo_journal(
    journal_path: Path,
    corpus_root: Path,
    execute: bool = False,
    force: bool = False,
) -> dict:
    root = corpus_root.resolve()
    entries: list[dict] = []
    corrupt = 0
    mid_corrupt = 0
    tend_id = ""
    try:
        # B2: read the recovery record forgivingly. errors="replace" so
        # even an undecodable byte becomes a malformed LINE (counted and
        # reported below) instead of killing the whole undo with a
        # UnicodeDecodeError — the journal is the last resort, and the
        # old `read_text(...)` + a single try/OSError meant one torn line
        # (crash mid-write) or one hand edit made the tend
        # un-undoable outright.
        raw = journal_path.read_bytes().decode("utf-8", errors="replace")
    except OSError as e:
        raise SlopKeeperError("journal_unreadable", str(e))
    lines = raw.splitlines()
    content = [i for i, ln in enumerate(lines) if ln.strip()]
    last_content = content[-1] if content else -1
    for i in content:
        try:
            entry = json.loads(lines[i])
        except json.JSONDecodeError:
            corrupt += 1
            if i != last_content:
                mid_corrupt += 1
            continue
        if not isinstance(entry, dict):
            # A bare list/string/number is not an op record; same
            # tolerance as a malformed line (B2), never a crash on
            # `.get` three lines down.
            corrupt += 1
            if i != last_content:
                mid_corrupt += 1
            continue
        if entry.get("op") == "plan" and isinstance(entry.get("tend_id"), str):
            tend_id = entry["tend_id"]
        entries.append(entry)
    if not entries:
        raise SlopKeeperError("journal_empty", "nothing to undo")
    if mid_corrupt and execute and not force:
        raise SlopKeeperError(
            "journal_corrupt",
            f"{mid_corrupt} malformed line(s) before the end of {journal_path.name} "
            f"({corrupt} total) — undo refuses to replay a partial record",
            "A torn LAST line is a crash artifact and is tolerated; a "
            "malformed line in the MIDDLE means the journal was edited or "
            "truncated. Inspect the file, fix or drop the bad line, then "
            "undo again — or pass --force to undo the ops that did parse "
            "(the rest stay applied). Nothing was restored.",
        )

    ops = []
    content_changed_rows = 0
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
        # B1: the journaled content hash is a CHECKPOINT, not decoration
        # (the module docstring always said the move is "journaled with a
        # content hash"; nothing ever compared it). The file at `dst` may
        # have been edited after the tend — by an agent, an editor, a
        # sync client — and undo would restore that mutated content to
        # the original path and report a clean "restored". Hash it first:
        # a mismatch still restores (the human's file comes back, and
        # deleting it would be the worse lie) but says `content_changed`
        # on the row, with both digests for the diff. Moves reach parity
        # with the edit/rewrite reversal above, which already refuses on
        # drift (`already_changed`) instead of guessing.
        content_changed = False
        expected = e.get("sha256")
        actual: str | None = None
        if expected and os.path.isfile(dst_abs):
            actual = sha256_file(dst_abs)
            # None = too big / unreadable to hash: unverifiable, which is
            # not evidence of change — never flag on a missing hash.
            if actual is not None and actual != expected:
                content_changed = True
                content_changed_rows += 1
        if execute:
            src_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                if os.path.isfile(dst_abs):
                    # B4's discipline on the way back too: never rename
                    # over a file someone put at the source path.
                    _move_no_clobber(dst_abs, src_abs)
                else:
                    if os.path.lexists(src_abs):
                        raise FileExistsError(
                            errno.EEXIST, "Destination exists", str(src_abs)
                        )
                    shutil.move(str(dst_abs), str(src_abs))
            except FileExistsError:
                ops.append({"src": e["src"], "status": "blocked_dst_occupied"})
                continue
            except OSError as err:
                # The restore itself failed (EACCES, disk gone). Recorded,
                # never raised: the remaining ops still reverse, and the
                # row says which one did not and why.
                ops.append(
                    {"src": e["src"], "status": "restore_failed", "error": str(err)}
                )
                continue
            _prune_empty_dirs(dst_abs.parent, root)
        row: dict = {
            "src": e["src"],
            "status": "restored" if execute else "would_restore",
        }
        if content_changed:
            row["content_changed"] = True
            row["sha256_journaled"] = expected
            row["sha256_now"] = actual
        ops.append(row)
    result = {"mode": "executed" if execute else "dry-run", "operations": ops}
    if tend_id:
        # A4: one journal per tend, and the tend names itself — the
        # "undo exactly one tend" model the guide promises.
        result["tend_id"] = tend_id
    if corrupt:
        # B2: malformed lines are counted, never a crash. A torn TAIL is
        # tolerated outright; a middle-line corruption refused --execute
        # above (dry-run still reports so the human sees what execute
        # would do, and why it needs --force).
        result["journal_corrupt_lines"] = corrupt
        if mid_corrupt:
            result["journal_corrupt_mid_lines"] = mid_corrupt
            if not execute:
                add_warning(result, "journal_corrupt_needs_force")
    if content_changed_rows:
        # B1: the per-row flags are the detail; this is the count an
        # agent keys off without walking `operations`.
        result["content_changed_count"] = content_changed_rows
        add_warning(result, "content_changed")
    if execute:
        # the manifest describes the CURRENT tended state, and the
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
