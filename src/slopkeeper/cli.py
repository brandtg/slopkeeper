# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""slopkeeper CLI.

Every command speaks the same JSON envelope and sets `next` — the calling
agent should never have to guess what to run after a successful step.
Humans get the same output; it's valid JSON everywhere, plus markdown
renders for the approval gate.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import VERSION
from .apply import apply_plan, undo_journal
from .doctor import doctor
from .inventory import collect
from .models import (
    HIDDEN_SKIP_DIRS,
    PLAN_SCHEMA,
    SlopKeeperError,
    corpus_fingerprint,
    emit_json,
    envelope,
    next_step,
    safe_relpath,
    state_dir,
)
from .anchors import anchored_move_risks
from .draft import draft as draft_plan
from .plan import (
    PLAN_TEMPLATE,
    load_plan,
    render_markdown,
    straggler_report,
    validate_plan,
    write_render_sidecar,
)
from .propose import build_candidates
from .roots import add_root, load_roots, remove_root
from .scaffold import scaffold_from_plan
from .similar import collect_similarity, hubs
from .skill import install as install_skills
from .status import status as corpus_status
from .status import status_all
from .strategies import get_strategy, list_strategies

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_VALIDATION = 2


def _resolve_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _write_output(path: str | None, text: str) -> None:
    if path:
        Path(path).expanduser().write_text(text, encoding="utf-8")


# --- commands ---------------------------------------------------------------


def cmd_doctor(args) -> int:
    data = doctor(_resolve_dir(args.root))
    emit_json(
        envelope(
            "doctor",
            data,
            next_step(
                data["next"]["action"], data["next"]["command"], data["next"]["hint"]
            ),
        )
    )
    return EXIT_OK


INVENTORY_FIELD_KEYS = {"path", "bytes", "mtime", "ext", "kind", "title", "head"}


def _shape_inventory_files(inv: dict, fields: list[str]) -> dict:
    """A per-file field projection (B11): the lean view the gate work
    actually reads. C7: the projection shapes the WHOLE document — dirs,
    clusters and references are dropped (the tree lives in the file
    paths; the roll-ups are derivable), because for a 630-file corpus
    they alone doubled the output (149KB where the files were 85KB).
    Unknown fields refuse loudly (a typo would otherwise silently
    produce an empty projection)."""
    for f in fields:
        if f not in INVENTORY_FIELD_KEYS:
            raise SlopKeeperError(
                "inventory_bad_fields",
                f"unknown inventory field: {f}",
                f"fields are: {', '.join(sorted(INVENTORY_FIELD_KEYS))}.",
            )
    files = [{k: f[k] for k in fields if k in f} for f in inv["files"]]
    out = {k: v for k, v in inv.items() if k not in ("dirs", "clusters", "references")}
    out["files"] = files
    return out


def cmd_inventory(args) -> int:
    root = _resolve_dir(args.root)
    fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()]
    inv = collect(root, depth=args.depth, fast=args.fast, max_files=args.max_files)
    if fields:
        inv = _shape_inventory_files(inv, fields)
    payload = inv
    if args.summary:
        payload = {
            "root": inv["root"],
            "fingerprint": inv["fingerprint"],
            "stats": inv["stats"],
            "clusters": inv.get("clusters", {}),
            "dirs": inv.get("dirs", []),
            "files": [f["path"] for f in inv["files"]],
        }
    if args.out:
        # T7 + B11: with --out the shaped payload lives only in the
        # file — --summary and --fields shape the FILE too (they used
        # to be ignored, shipping the full ~440KB document anyway) —
        # and stdout carries token-bounded accounting (counts + where
        # the body is), never the dump itself. A --fields projection
        # is the agent-lean view: compact JSON, no pretty-print tax.
        indent = None if fields else 2
        _write_output(args.out, json.dumps(payload, indent=indent, default=str))
        stats_payload = {
            "root": inv["root"],
            "fingerprint": inv["fingerprint"],
            "stats": inv["stats"],
            "out_path": args.out,
        }
        if fields:
            stats_payload["fields"] = fields
        payload = stats_payload
    hint = (
        "Read `slopkeeper strategies show <id>` first; "
        "write the plan; then `slopkeeper plan validate`."
    )
    if args.out:
        hint = f"Full inventory JSON written to {args.out} — read it there. {hint}"
    emit_json(
        envelope(
            "inventory",
            payload,
            next_step(
                "plan",
                "(agent writes plan JSON from the strategy rubric, then: "
                f"slopkeeper plan validate <plan> {root})",
                hint,
            ),
        )
    )
    return EXIT_OK


def cmd_clusters(args) -> int:
    root = _resolve_dir(args.root)
    inv = collect(root, depth=args.depth)
    files = inv["files"]
    data = collect_similarity(root, files)
    if args.min_size > 1:
        data["clusters"] = [c for c in data["clusters"] if c["size"] >= args.min_size]
    if inv.get("references"):
        data["hubs"] = hubs(inv["references"], min_referrers=args.min_hub)
    payload = data
    if args.out:
        # T7: the full clusters payload (file lists per cluster) lives
        # only in the --out file — stdout carries counts + where the
        # body is, same shape of output as inventory.
        payload = {
            "cluster_count": len(data["clusters"]),
            "unclustered_count": len(data["unclustered"]),
            "same_stem_group_count": len(data.get("same_stem_groups", [])),
            "identical_group_count": len(data.get("identical_groups", [])),
            "out_path": args.out,
        }
        if data.get("hubs"):
            payload["hub_count"] = len(data["hubs"])
    hint = (
        "Clusters are hints, not a plan: name each cluster, merge or "
        "split as judgment dictates, then write the plan JSON."
    )
    if args.out:
        hint = f"Full clusters JSON written to {args.out} — read it there. {hint}"
    emit_json(
        envelope(
            "clusters",
            payload,
            next_step(
                "plan",
                "(agent names folders and assigns stragglers, then: "
                f"slopkeeper plan validate <plan> {root})",
                hint,
            ),
        )
    )
    if args.out:
        _write_output(args.out, json.dumps(data, indent=2, default=str))
    return EXIT_OK


def cmd_status(args) -> int:
    root = _resolve_dir(args.root) if args.root else None
    if args.all:
        roots = load_roots()
        if not roots:
            raise SlopKeeperError(
                "no_roots",
                "the roots registry is empty",
                "Add corpora with `slopkeeper roots add <path>`, or run "
                "`slopkeeper status <root>` directly.",
            )
        data = status_all(roots, record=args.record)
        pending = sum(
            1
            for e in data
            if not e["ok"]
            or e.get("never_tended")
            or e.get("new_count")
            or e.get("modified_count")
            or e.get("removed_count")
        )
        emit_json(
            envelope(
                "status",
                data,
                next_step(
                    "status",
                    "(per-root detail: slopkeeper status <root>)",
                    f"{len(data)} roots surveyed, {pending} need attention. "
                    "Digest is counts only — drill in per root.",
                ),
            )
        )
        return EXIT_OK
    if not args.root:
        raise SlopKeeperError(
            "no_root",
            "no corpus root given and --all not set",
            "`slopkeeper status <root>` for one corpus or "
            "`slopkeeper status --all` for the registry.",
        )
    data = corpus_status(root, record=args.record)
    if data.get("never_tended"):
        nxt = next_step(
            "doctor",
            f"slopkeeper doctor {root}",
            "Corpus was never tended; start with a doctor pass.",
        )
    elif data.get("walk_contract_stale"):
        # The diff is missing a walk set — planning against it authors
        # garbage. Point at the one-command fix the advice names.
        nxt = next_step(
            "status",
            f"slopkeeper status --record {root}",
            "Manifest predates the current walk contract; re-record "
            "before planning anything against this diff.",
        )
    elif data.get("heavy_drift"):
        # Phase 7 rebaseline: heavy drift means the manifest no longer
        # describes reality — re-record, never a hand-built zero-move
        # plan. `--record` carries unresolved and last_plan forward.
        hd = data["heavy_drift"]
        nxt = next_step(
            "status",
            f"slopkeeper status --record {root}",
            f"Heavy drift: {hd['changed']} files changed since the last "
            f"tend (threshold {hd['threshold']}) — the manifest no longer "
            "describes the corpus. Rebaseline instead of hand-building a "
            "zero-move plan.",
        )
    else:
        nxt = next_step(
            "plan",
            f"(triage new_files, then write a delta plan, then: "
            f"slopkeeper plan validate <plan> {root})",
            f"{data['new_count']} new, {len(data['modified'])} modified, "
            f"{len(data['removed'])} removed since last tend. Delta plans "
            'carry "partial": true at the top level — no flag needed '
            "(draft plans already do).",
        )
    emit_json(envelope("status", data, nxt))
    return EXIT_OK


def _draft_out_path(raw: str | None, root: Path) -> Path:
    """Resolve the draft's --out against CWD, refusing paths inside the corpus.

    A draft written inside the root would shift the corpus fingerprint —
    the very next apply would refuse with corpus_changed — and pollute
    the catalog the next draft reads. So the CWD-relative default
    (draft-plan.json) is refused when it resolves inside the root, as is
    any explicit path that does.
    """
    p = Path(raw or "draft-plan.json").expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if p.is_relative_to(root):
        raise SlopKeeperError(
            "draft_out_inside_root",
            f"refusing to write the draft inside the corpus: {p}",
            "Write the draft OUTSIDE the corpus — a draft inside the root "
            "shifts its fingerprint and the next apply refuses with "
            "corpus_changed. Run from another directory (the default "
            "draft-plan.json resolves against your CWD) or pass an "
            "explicit path elsewhere.",
        )
    return p


def cmd_draft(args) -> int:
    root = _resolve_dir(args.root)
    out_path = _draft_out_path(args.out, root)
    data = draft_plan(root, min_sim=args.min_sim, depth=args.depth)
    if data.get("plan") is None:
        # T7: the envelope is token-bounded accounting — the draft body
        # lives only in the --out file, never on stdout.
        payload = {
            "drafted_count": data["drafted_count"],
            "unresolved_count": data["unresolved_count"],
            "new_count": data["new_count"],
            "out_path": None,
        }
        if data.get("undrafted"):
            payload["undrafted"] = data["undrafted"]
        payload["note"] = data.get("note", "Nothing to draft.")
        nxt = next_step(
            "status",
            f"slopkeeper status {root}",
            payload["note"],
        )
    else:
        _write_output(str(out_path), json.dumps(data["plan"], indent=1))
        payload = {
            "drafted_count": data["drafted_count"],
            "unresolved_count": data["unresolved_count"],
            "new_count": data["new_count"],
            "out_path": str(out_path),
        }
        hint = (
            "DRAFT, NOT A DECISION: review every move and reason, "
            "reclassify as needed, then render for the human gate."
        )
        if data.get("undrafted"):
            payload["undrafted"] = data["undrafted"]
            und = data["undrafted"]
            hint += (
                f" {und['count']} new file(s) undrafted ({und['reason']}) — "
                "they need human triage."
            )
        nxt = next_step(
            "plan.validate",
            f"slopkeeper plan validate {out_path} {root}",
            hint,
        )
    emit_json(envelope("draft", payload, nxt))
    return EXIT_OK


def cmd_refs(args) -> int:
    from .apply import _all_rel_paths
    from .ignore import load_ignore
    from .references import (
        dangling_citations,
        external_citation_lines,
        refs_lookup,
    )

    root = _resolve_dir(args.root)
    ignore = load_ignore(root)
    if getattr(args, "dangling", False):
        # Phase 11: already-broken citations, as data. Informational —
        # never a rewrite candidate; the pass fixes citations of MOVED
        # paths and these point at nothing.
        sites = dangling_citations(
            root,
            _all_rel_paths(root),
            per_file_cap=args.per_file,
            max_total=getattr(args, "limit", 0) or 0,
            prefix=getattr(args, "prefix", None),
        )
        emit_json(
            envelope(
                "refs",
                {
                    "mode": "dangling",
                    "count": len(sites),
                    "sites": sites,
                },
                next_step(
                    "plan.edit",
                    "fix the listed file:line by hand with plan edits",
                    "Dangling citations name nothing that exists — half-updated "
                    "references or moved files nothing caught. Fix by hand; the "
                    "rewrite pass never touches them.",
                ),
            )
        )
        return EXIT_OK
    if not args.basename:
        raise SlopKeeperError(
            "no_basename",
            "refs needs a basename (or --dangling)",
            "Call `refs <root> <basename>` or `refs <root> --dangling`.",
        )
    scan_dirs = []
    scan_rows = []
    for raw in args.scan_path or []:
        d = _resolve_dir(raw)
        if not d.is_dir():
            # A typo'd dir must not read as "no external citations".
            raise SlopKeeperError(
                "scan_path_missing",
                f"--scan-path directory does not exist: {raw}",
                "Create it or fix the path.",
            )
        if any(part in HIDDEN_SKIP_DIRS for part in d.parts):
            # Tool layer is NEVER read — reported per-dir, never
            # silently empty (Phase 6): an empty result must not read
            # as "no citations". The row carries the code so agents
            # grepping output find the guide row.
            scan_rows.append(
                {"path": str(d), "status": "skip_list", "code": "scan_path_skip_list"}
            )
            continue
        scan_dirs.append(d)
        row = {"path": str(d), "status": "scanned"}
        if d.is_relative_to(root):
            rel = d.relative_to(root).as_posix()
            if rel and ignore.is_ignored(rel, is_dir=True):
                # Explicitly naming an ignored dir is the override: it
                # IS scanned (below), and the result says so.
                row["ignored"] = True
        scan_rows.append(row)
    ext_hits = external_citation_lines(
        scan_dirs, args.basename, per_file_cap=args.per_file
    )
    lookup = refs_lookup(
        root, args.basename, _all_rel_paths(root), per_file_cap=args.per_file
    )
    hits = lookup["citations"]
    if lookup["nonexistent_path"]:
        # D7: the name names no corpus path — say so instead of letting
        # the basename fallback return a different file's citations.
        hits = hits + ext_hits
        emit_json(
            envelope(
                "refs",
                {
                    "basename": args.basename,
                    "nonexistent_path": True,
                    "citation_count": len(hits),
                    "citations": hits,
                    "scan_paths": [str(d) for d in scan_dirs],
                    "scan_path_results": scan_rows,
                },
                next_step(
                    "plan.edit",
                    "the path does not exist in the corpus; only exact "
                    "citations of the string are listed — fix or drop them",
                    "A slashed name that exists nowhere must not fall back "
                    "to a same-basename file (the 0.1.9 report's D7).",
                ),
            )
        )
        return EXIT_OK
    # Per-dir hit counts: the agent sees which declared dir produced
    # what, and that a skip-list dir produced nothing because it was
    # never read.
    prefix_counts: dict[str, int] = {}
    for h in ext_hits:
        for d in scan_dirs:
            if h["file"] == str(d) or h["file"].startswith(str(d) + os.sep):
                prefix_counts[str(d)] = prefix_counts.get(str(d), 0) + 1
                break
    for row in scan_rows:
        if row["status"] == "scanned":
            row["hits"] = prefix_counts.get(row["path"], 0)
    hits += ext_hits
    emit_json(
        envelope(
            "refs",
            {
                "basename": args.basename,
                "citation_count": len(hits),
                "citations": hits,
                "scan_paths": [str(d) for d in scan_dirs],
                "scan_path_results": scan_rows,
            },
            next_step(
                "plan",
                "(declare edits: [{file, find, replace}] using these lines, "
                f"then: slopkeeper plan validate <plan> {root})",
                "find must match exactly once in the target file; replace with "
                "the file's new path after the move. Citations flagged "
                '"external": true sit outside the corpus (absolute file path) '
                "— declare external_roots (or scan_paths) to rewrite them in "
                "the plan. A --scan-path inside the tool-layer skip list "
                "(.git, .slopkeeper, .venv, node_modules, ...) is never read "
                "and reported as scan_path_skip_list.",
            ),
        )
    )
    return EXIT_OK


def _count_files(p: Path) -> int:
    """Raw file count under `p` — the `find | wc -l` replacement.

    Deliberately RAW: ignored files and nested-corpus subtrees count,
    because the integrity check is about bytes on disk, not the
    parent's visibility rules. A move that dropped a file must show in
    this number no matter what the walk contract thinks of it.
    """
    if not p.exists():
        return 0
    if p.is_file():
        return 1
    n = 0
    for _dirpath, _dirnames, filenames in os.walk(p):
        n += len(filenames)
    return n


def cmd_move(args) -> int:
    """`slopkeeper move <src> <dst> <root>` — one move, full audit trail.

    For a directory too big to plan (61k entries), the hand-rolled
    `mv` loses the journal and the `find | wc` loses the gate. This is
    both: a schema-complete one-move plan (fingerprint from a fresh
    walk), the standard human gate as the render on stdout (no
    --execute = nothing moves), and on --execute the standard apply
    path — journal, manifest re-record, rewrite pass — plus an
    integrity count across the move (src_files vs dst_files).
    """
    from .apply import build_gate_report

    root = _resolve_dir(args.root)
    src = safe_relpath(args.src, root).as_posix()
    dst = safe_relpath(args.dst, root).as_posix()
    plan = {
        "schema": PLAN_SCHEMA,
        "corpus_root": str(root),
        "strategy": "bulk-move",
        "based_on": {
            "corpus_fingerprint": corpus_fingerprint(root),
            "inventory_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
        "moves": [{"from": src, "to": dst, "reason": str(args.reason).strip()}],
        "edits": [],
        "rewrites": {"mode": "auto"},
        "external_roots": [],
        "scan_paths": [],
        "skip": [],
        "keep": [],
        "unresolved": [],
        "folders": [],
        # A bulk move is a delta plan by construction: it does not
        # cover (and must not demand coverage of) the rest of the
        # corpus. The corpus_changed gate still binds it to a fresh
        # fingerprint either way.
        "partial": True,
    }
    # Every refusal the plan path has applies here: path_escape,
    # plan_noop_move, move_source_ignored, plan_stale_src, dst_exists
    # (raised by apply's pre-flight on --execute). Nothing new to learn.
    plan = validate_plan(plan, root)

    if not args.execute:
        # The gate IS the render on stdout (plan render's §2 precedent):
        # the human approves real paths with real reference numbers.
        ref_report, rewrite_summary = build_gate_report(plan, root)
        print(
            render_markdown(
                plan,
                ref_report,
                rewrite_summary,
                footer=(
                    f"_Approve by re-running with --execute: "
                    f"`slopkeeper move {src} {dst} {root} --execute`_"
                ),
            )
        )
        return EXIT_OK

    # Materialize the plan for the audit trail (the journal header binds
    # its sha256; the manifest records last_plan) inside the state dir,
    # where no walk will ever see it.
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan_path = state_dir(root) / f"move-{ts}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")

    src_files = _count_files(root / src)
    data = apply_plan(plan_path, root, execute=True)
    integrity = {
        "src_files": src_files,
        "dst_files": _count_files(root / dst),
    }
    integrity["match"] = integrity["src_files"] == integrity["dst_files"]
    data = {**data, "integrity": integrity, "plan": str(plan_path)}
    if data.get("partial"):
        out = envelope(
            "move",
            data,
            next_step(
                "undo",
                f"slopkeeper undo {data['journal']} {root}",
                "PARTIAL MOVE: mid-run failures — read data.errors, then "
                "run this undo (dry-run is the default; add --execute to "
                "restore) before re-running the move.",
            ),
        )
        out["partial"] = True
        emit_json(out)
        return EXIT_ERROR
    emit_json(
        envelope(
            "move",
            data,
            next_step(
                "undo",
                f"slopkeeper undo {data['journal']} {root}",
                "One journaled move; integrity count is in the result "
                "(src_files == dst_files). Dry-run undo is the default; "
                "add --execute to restore.",
            ),
        )
    )
    return EXIT_OK


def cmd_propose(args) -> int:
    """Phase 10: the layout fork as data — advisory, never a plan.

    Read-only: no fingerprint access, no writes. Declared citing dirs
    are validated with the same codes `refs --scan-path` uses, so a
    typo'd dir or a skip-list member fails loudly, never as an empty
    count.
    """
    root = _resolve_dir(args.root)
    extra: list[Path] = []
    for raw in (args.scan_path or []) + (args.external_root or []):
        d = _resolve_dir(raw)
        if not d.is_dir():
            raise SlopKeeperError(
                "scan_path_missing",
                f"citing directory does not exist: {raw}",
                "Create it or fix the path.",
            )
        if any(part in HIDDEN_SKIP_DIRS for part in d.resolve().parts):
            raise SlopKeeperError(
                "scan_path_skip_list",
                f"citing directory {d} is tool-layer skip list",
                "Skip-list members are never read — name a directory outside them.",
            )
        extra.append(d)
    data = build_candidates(root, extra)
    emit_json(
        envelope(
            "propose",
            data,
            next_step(
                "plan.schema",
                "slopkeeper plan schema > plan.json",
                "Pick a layout above and author the plan — rules or moves; "
                "the counts are the costs. Then: slopkeeper plan validate "
                f"plan.json {root} (render for the human gate). propose "
                "never authors a plan.",
            ),
        )
    )
    return EXIT_OK


def cmd_strategies(args) -> int:
    tokens = list(args.tokens)
    if tokens and tokens[0] == "show":
        # `strategies show <id>` (documented form); a bare `<id>` still works.
        # With nothing after it, "show" is treated as the id (unknown_strategy).
        tokens = tokens[1:] or tokens
    if tokens:
        s = get_strategy(tokens[0])
        emit_json(envelope("strategies.show", s))
    else:
        emit_json(
            envelope(
                "strategies.list",
                list_strategies(),
                next_step(
                    "strategies.show",
                    "slopkeeper strategies show <id>",
                    "Read the full rubric before planning.",
                ),
            )
        )
    return EXIT_OK


def cmd_plan_validate(args) -> int:
    root = _resolve_dir(args.root)
    plan = load_plan(Path(args.plan))
    plan = validate_plan(plan, root, allow_partial=args.allow_partial)
    stragglers = straggler_report(root, plan)
    # Phase 9: informational cost rows for moves of code-anchored dirs.
    anchor_risks = anchored_move_risks(root, plan)
    # B8: emptied-by-file dirs ride on the validated plan itself.
    emptied = plan.get("emptied_dir_suggestions") or []
    data = {
        "plan_file": args.plan,
        "moves": len(plan["moves"]),
        "folders": len(plan.get("folders", [])),
        "unresolved": len(plan.get("unresolved", [])),
        "valid": True,
        # Counts on stdout, full payloads in --out (the house
        # convention): a rule-expanded move list and the straggler
        # report are both unbounded.
        "straggler_dirs": len(stragglers),
        "straggler_files": sum(s["count"] for s in stragglers),
        "code_anchored_moves": len(anchor_risks),
        "emptied_dirs": len(emptied),
    }
    if plan.get("rules"):
        data["rules"] = plan["rules"]
    if plan.get("rule_warnings"):
        data["rule_warnings"] = plan["rule_warnings"]
    if emptied:
        data["emptied_dir_suggestions"] = emptied
    if args.out:
        _write_output(
            args.out,
            json.dumps(
                {
                    **plan,
                    "stragglers": stragglers,
                    "code_anchor_risks": anchor_risks,
                },
                indent=1,
            ),
        )
        data["written_to"] = args.out
    emit_json(
        envelope(
            "plan.validate",
            data,
            next_step(
                "plan.render",
                f"slopkeeper plan render {args.plan} {root}",
                "Render the human-gate markdown and show it to the owner."
                + (
                    " Empty-by-file directories are listed — author"
                    " prefix_rewrites or hand edits for their convention"
                    " strings first."
                    if emptied
                    else ""
                ),
            ),
        )
    )
    return EXIT_OK


def cmd_plan_render(args) -> int:
    from .apply import _run_artifacts, build_gate_report
    from .models import corpus_fingerprint

    root = _resolve_dir(args.root)
    plan = load_plan(Path(args.plan))
    plan = validate_plan(plan, root, allow_partial=args.allow_partial)

    # Reference counts for the human gate: the same un-capped stream
    # scan apply will run (Phase 3 — no size gate anywhere). One pass
    # computes both the risks and the rewrite summary, and the risk
    # rows carry per-citation status (B2). This run's own artifacts
    # (plan file, gate, rewrites JSON) are excluded from the scan when
    # they sit inside the corpus.
    exclude = _run_artifacts(root, args.plan, args.out, args.rewrites_out)
    ref_report, rewrite_summary = build_gate_report(plan, root, exclude)
    # Phase 8: files left behind in folders the plan empties — visible
    # at the gate, capped in the render, full list in validate --out.
    stragglers = straggler_report(root, plan)
    # Phase 9: moves of code-anchored dirs — informational cost rows.
    anchor_risks = anchored_move_risks(root, plan)
    # B8: emptied-by-file dirs and their destination breakdowns ride on
    # the validated plan.
    emptied = plan.get("emptied_dir_suggestions") or []
    if "based_on" not in plan or not plan["based_on"]:
        # corpus_fingerprint is canonical (T1): the pre-T1
        # `inventory_sha256` spelling is a read-only alias for plans in
        # the wild — new fallbacks never write it.
        plan["based_on"] = {
            "corpus_fingerprint": corpus_fingerprint(root),
            "inventory_at": "unrecorded",
        }
    md = render_markdown(
        plan,
        ref_report,
        rewrite_summary,
        stragglers=stragglers,
        code_anchor_risks=anchor_risks,
        emptied_dirs=emptied,
    )
    if args.out:
        _write_output(args.out, md)
        # §5.6: bind the rendered gate to the plan's exact bytes so a
        # later apply can warn on silent render→apply drift. A stdout
        # render records nothing — and carries no artifact paths for
        # apply to exclude from the rewrite pass.
        write_render_sidecar(
            Path(args.plan),
            root,
            artifacts=[
                rel
                for rel in _run_artifacts(root, args.plan, args.out, args.rewrites_out)
            ],
        )
        payload = {"written_to": args.out, "bytes": len(md)}
    else:
        # §2 exception, on plan schema's precedent: the output IS the
        # artifact — a stdout render prints the gate markdown itself.
        print(md)
        payload = {"bytes": len(md)}
    if args.rewrites_out:
        # B1: the full line-level rewrite detail as JSON — the render
        # shows it capped, the file carries everything.
        detail = {
            row["file"]: row.get("lines", [])
            for row in rewrite_summary.get("per_file", [])
        }
        _write_output(args.rewrites_out, json.dumps(detail, indent=1))
        payload["rewrites_out"] = args.rewrites_out
    if args.out:
        emit_json(envelope("plan.render", payload))
    return EXIT_OK


def cmd_plan_schema(_args) -> int:
    # §2 exception, on plan render's markdown precedent: the output IS
    # the artifact. `slopkeeper plan schema > plan.json` must yield a plan
    # skeleton — the envelope form failed plan_bad_schema.
    print(json.dumps(PLAN_TEMPLATE, indent=2))
    return EXIT_OK


def cmd_guide(_args) -> int:
    # §2 exception, on plan schema's precedent: the output IS the artifact
    # the agent reads. Packaged so pipx/venv installs carry the guide even
    # with no repo checkout (src/slopkeeper/AGENT_GUIDE.md symlinks the root
    # copy — one canonical file).
    guide = importlib.resources.files("slopkeeper").joinpath("AGENT_GUIDE.md")
    print(guide.read_text(encoding="utf-8"), end="")
    return EXIT_OK


def cmd_apply(args) -> int:
    root = _resolve_dir(args.root)
    data = apply_plan(
        Path(args.plan),
        root,
        execute=args.execute,
        force=args.force,
        allow_partial=args.allow_partial,
        strict_references=args.strict_references,
    )
    if args.rewrites_out:
        # B1: the line-level rewrite detail as JSON, matching
        # plan render --rewrites-out (dry-run or executed alike).
        rows = (
            data.get("would_rewrite")
            or (data.get("rewrites") or {}).get("per_file")
            or []
        )
        if rows:
            detail = {row["file"]: row.get("lines", []) for row in rows}
            _write_output(args.rewrites_out, json.dumps(detail, indent=1))
            data = {**data, "rewrites_out": args.rewrites_out}
    if data["mode"] == "executed" and data.get("partial"):
        # T5: mid-run failures are loud. The apply DID run (ok stays
        # true), but the envelope says partial, `next` is the exact
        # undo for this journal, and the process exits non-zero — an
        # agent keying off the exit code can never miss it.
        out = envelope(
            "apply",
            data,
            next_step(
                "undo",
                f"slopkeeper undo {data['journal']} {root}",
                "PARTIAL APPLY: mid-run failures — read data.errors, fix "
                "the cause, then run this undo (dry-run is the default; "
                "add --execute to restore) before re-applying.",
            ),
        )
        out["partial"] = True
        emit_json(out)
        return EXIT_ERROR
    if data["mode"] == "executed":
        nxt = next_step(
            "scaffold",
            f"slopkeeper scaffold {args.plan} {root} --execute",
            "Moves are journaled; write the AGENTS.md/STATE.md layer next.",
        )
    else:
        nxt = next_step(
            "apply",
            f"slopkeeper apply {args.plan} {root} --execute",
            "Dry-run looked clean; re-run with --execute to move files.",
        )
    emit_json(envelope("apply", data, nxt))
    return EXIT_OK


def cmd_undo(args) -> int:
    data = undo_journal(
        Path(args.journal), _resolve_dir(args.root), execute=args.execute
    )
    emit_json(envelope("undo", data))
    return EXIT_OK


def cmd_scaffold(args) -> int:
    root = _resolve_dir(args.root)
    data = scaffold_from_plan(
        Path(args.plan),
        root,
        execute=args.execute,
        strategy_id=args.strategy,
    )
    if data["mode"] == "executed":
        nxt = next_step(
            "report",
            f"(present the {root} roster + suggested files to the owner)",
            "Done. Anything 'exists→suggested' needs a human merge.",
        )
    else:
        nxt = next_step(
            "scaffold",
            f"slopkeeper scaffold {args.plan} {root} --execute",
            "Dry-run looked clean; re-run with --execute to write.",
        )
    emit_json(envelope("scaffold", data, nxt))
    return EXIT_OK


def cmd_roots_add(args) -> int:
    root = _resolve_dir(args.path)
    roots, added = add_root(root)
    emit_json(
        envelope(
            "roots.add",
            {"root": str(root), "added": added, "roots": roots},
            next_step(
                "roots.list",
                "slopkeeper roots list",
                "Registered. The scheduled pass is one call: "
                "`slopkeeper status --all`.",
            ),
        )
    )
    return EXIT_OK


def cmd_roots_remove(args) -> int:
    root = _resolve_dir(args.path)
    roots = remove_root(root)
    emit_json(
        envelope(
            "roots.remove",
            {"root": str(root), "roots": roots},
            next_step(
                "roots.list",
                "slopkeeper roots list",
                "Unregistered; the corpus itself is untouched.",
            ),
        )
    )
    return EXIT_OK


def cmd_skill_install(args) -> int:
    installed = install_skills(args.targets, force=args.force)
    warnings = [r["warning"] for r in installed if "warning" in r]
    data: dict = {"skill": "slopkeeper", "installed": installed}
    if warnings:
        data["warnings"] = warnings
    emit_json(
        envelope(
            "skill.install",
            data,
            next_step(
                "guide",
                "slopkeeper guide",
                "Skill installed. The agent reads `slopkeeper guide` for the "
                "full protocol before tending anything.",
            ),
        )
    )
    return EXIT_OK


def cmd_roots_list(_args) -> int:
    roots = load_roots()
    emit_json(
        envelope(
            "roots.list",
            roots,
            next_step(
                "status",
                "slopkeeper status --all",
                f"{len(roots)} roots registered."
                if roots
                else "Nothing registered — add a corpus with "
                "`slopkeeper roots add <path>`.",
            ),
        )
    )
    return EXIT_OK


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slopkeeper",
        description="Inventory, partition and scaffold messy file corpora. "
        "Agent-first: JSON envelope + next-step hints; nothing "
        "moves without an approved plan and --execute. "
        "Run with no subcommand for the interactive REPL.",
    )
    p.add_argument("--version", action="version", version=f"slopkeeper {VERSION}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser(
        "repl", help="the interactive REPL (what bare `slopkeeper` starts)"
    )
    sp.add_argument("root", nargs="?", help="corpus root to start in")
    sp.set_defaults(func=None)

    sp = sub.add_parser("doctor", help="preflight a corpus; pick a strategy")
    sp.add_argument("root")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("inventory", help="structured JSON inventory")
    sp.add_argument("root")
    sp.add_argument("--depth", type=int, default=None)
    sp.add_argument(
        "--fast",
        action="store_true",
        help="skip reference scanning and heads (large corpora)",
    )
    sp.add_argument(
        "--summary", action="store_true", help="clusters + paths only (token-lean)"
    )
    sp.add_argument("--max-files", type=int, default=8000)
    sp.add_argument(
        "--fields",
        default=None,
        help="comma-separated per-file field projection (B11), e.g. "
        "path,title,mtime (~15KB where the full inventory is ~440KB); "
        "shapes --out too",
    )
    sp.add_argument(
        "--out",
        help="write the (summary/fields-shaped) inventory JSON here "
        "(stdout carries counts only)",
    )
    sp.set_defaults(func=cmd_inventory)

    sp = sub.add_parser(
        "clusters",
        help="similarity hints: content clusters, scrape pairs, dupes, hubs",
    )
    sp.add_argument("root")
    sp.add_argument("--depth", type=int, default=None)
    sp.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="drop clusters smaller than this (default 2)",
    )
    sp.add_argument(
        "--min-hub",
        type=int,
        default=3,
        help="min referrers for a hub file (default 3)",
    )
    sp.add_argument(
        "--out",
        help="write the full clusters JSON here (stdout carries counts only)",
    )
    sp.set_defaults(func=cmd_clusters)

    sp = sub.add_parser(
        "status",
        help="diff the corpus against the last tend (manifest)",
    )
    sp.add_argument("root", nargs="?", help="corpus root (omit with --all)")
    sp.add_argument(
        "--record",
        action="store_true",
        help="write/refresh the manifest, carrying forward unresolved and "
        "last_plan (also done by apply --execute)",
    )
    sp.add_argument(
        "--all",
        action="store_true",
        help="one digest over every registered root (see: roots)",
    )
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser(
        "draft",
        help="auto-draft a PARTIAL delta plan for new slop (recursive by default)",
    )
    sp.add_argument("root")
    sp.add_argument(
        "--min-sim",
        type=float,
        default=None,
        help="min cosine to the nearest filed file (default: cluster T1)",
    )
    sp.add_argument(
        "--depth",
        type=int,
        default=None,
        help="only consider new files within N directory levels of the "
        "root (1 = root + immediate subdirs; default: the whole corpus)",
    )
    sp.add_argument(
        "--out",
        default="draft-plan.json",
        help="where to write the draft plan JSON (resolved against your "
        "CWD; refused if inside the corpus root)",
    )
    sp.set_defaults(func=cmd_draft)

    sp = sub.add_parser(
        "refs",
        help="exact citation lines for a basename (input for plan edits)",
    )
    sp.add_argument("root")
    sp.add_argument(
        "basename",
        nargs="?",
        default=None,
        help="corpus path, directory, path suffix, or unambiguous basename",
    )
    sp.add_argument(
        "--dangling",
        action="store_true",
        help="report path-ish citations that resolve to nothing (informational)",
    )
    sp.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max dangling rows (default: unbounded — correctness over memory)",
    )
    sp.add_argument(
        "--prefix",
        default=None,
        help="dangling mode: only tokens naming this corpus directory",
    )
    sp.add_argument(
        "--per-file",
        type=int,
        default=5,
        help="max citation lines per file (default 5)",
    )
    sp.add_argument(
        "--scan-path",
        action="append",
        default=None,
        metavar="DIR",
        help="also scan this directory OUTSIDE the corpus for citations (repeatable)",
    )
    sp.set_defaults(func=cmd_refs)

    sp = sub.add_parser(
        "move",
        help="move one file or directory with a full audit trail (gate + journal)",
    )
    sp.add_argument("src", help="path inside the corpus (file or directory)")
    sp.add_argument("dst", help="destination path inside the corpus")
    sp.add_argument("root", help="corpus root")
    sp.add_argument(
        "--execute",
        action="store_true",
        help="run the move (default: render the human gate only)",
    )
    sp.add_argument(
        "--reason", default="bulk move", help="one-line reason shown at the gate"
    )
    sp.set_defaults(func=cmd_move)

    sp = sub.add_parser(
        "propose", help="candidate layouts with breakage costs (read-only)"
    )
    sp.add_argument("root")
    sp.add_argument(
        "--scan-path",
        action="append",
        default=[],
        help="absolute dir with citations to count (repeatable; e.g. a skills store)",
    )
    sp.add_argument(
        "--external-root",
        action="append",
        default=[],
        help="absolute dir OUTSIDE the corpus with citations to count (repeatable)",
    )
    sp.set_defaults(func=cmd_propose)

    sp = sub.add_parser("strategies", help="list or show partitioning rubrics")
    sp.add_argument(
        "tokens",
        nargs="*",
        metavar="[show] <id>",
        help="strategy id, optionally preceded by 'show'",
    )
    sp.set_defaults(func=cmd_strategies)

    sp = sub.add_parser("plan", help="plan subcommands")
    psub = sp.add_subparsers(dest="plan_cmd", required=True)

    v = psub.add_parser("validate", help="validate a plan JSON against the corpus")
    v.add_argument("plan")
    v.add_argument("root")
    v.add_argument("--allow-partial", action="store_true")
    v.add_argument(
        "--out",
        default=None,
        help="write the validated plan (expanded moves + stragglers) as JSON here",
    )
    v.set_defaults(func=cmd_plan_validate)

    r = psub.add_parser("render", help="render the human-gate markdown")
    r.add_argument("plan")
    r.add_argument("root")
    r.add_argument("--allow-partial", action="store_true")
    r.add_argument("--out", default=None)
    r.add_argument(
        "--rewrites-out",
        default=None,
        help="write the full line-level rewrite detail ({file: [{line, old, "
        "new}]}) as JSON here (B1)",
    )
    r.set_defaults(func=cmd_plan_render)

    s = psub.add_parser("schema", help="print the plan JSON template")
    s.set_defaults(func=cmd_plan_schema)

    sp = sub.add_parser("guide", help="print the agent guide (AGENT_GUIDE.md)")
    sp.set_defaults(func=cmd_guide)

    sp = sub.add_parser("apply", help="execute an approved plan (dry-run default)")
    sp.add_argument("plan")
    sp.add_argument("root")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument(
        "--force", action="store_true", help="skip the corpus-changed fingerprint gate"
    )
    sp.add_argument("--allow-partial", action="store_true")
    sp.add_argument(
        "--strict-references",
        action="store_true",
        help="abort instead of warning on referenced moves",
    )
    sp.add_argument(
        "--rewrites-out",
        default=None,
        help="write the line-level rewrite detail ({file: [{line, old, new}]}) "
        "as JSON here (B1)",
    )
    sp.set_defaults(func=cmd_apply)

    sp = sub.add_parser("undo", help="reverse a journal's moves (dry-run default)")
    sp.add_argument("journal")
    sp.add_argument("root")
    sp.add_argument("--execute", action="store_true")
    sp.set_defaults(func=cmd_undo)

    sp = sub.add_parser(
        "scaffold", help="write AGENTS.md/STATE.md layer (dry-run default)"
    )
    sp.add_argument("plan")
    sp.add_argument("root")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--strategy", default=None)
    sp.set_defaults(func=cmd_scaffold)

    sp = sub.add_parser(
        "roots", help="registry of tended corpus roots (for status --all)"
    )
    rsub = sp.add_subparsers(dest="roots_cmd", required=True)

    ra = rsub.add_parser("add", help="register a corpus root (idempotent)")
    ra.add_argument("path")
    ra.set_defaults(func=cmd_roots_add)

    rr = rsub.add_parser("remove", help="unregister a corpus root")
    rr.add_argument("path")
    rr.set_defaults(func=cmd_roots_remove)

    rl = rsub.add_parser("list", help="list registered roots")
    rl.set_defaults(func=cmd_roots_list)

    sp = sub.add_parser(
        "skill", help="install the shipped agent skill (SKILL.md format)"
    )
    ssub = sp.add_subparsers(dest="skill_cmd", required=True)

    si = ssub.add_parser(
        "install", help="copy the packaged skill into agents' skill directories"
    )
    si.add_argument(
        "targets",
        nargs="+",
        metavar="agent|dir",
        help="agent name (claude, opencode, hermes) or a skills directory path",
    )
    si.add_argument(
        "--force",
        action="store_true",
        help="overwrite an already-installed skill (default: skip with a warning)",
    )
    si.set_defaults(func=cmd_skill_install)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) is None or getattr(args, "func", None) is None:
        # Bare `slopkeeper` (or `slopkeeper repl`) is the human REPL; subcommands keep
        # the JSON envelope for agents (DESIGN §2.7).
        from .tui import repl

        return repl(getattr(args, "root", None))
    try:
        return args.func(args)
    except SlopKeeperError as e:
        emit_json(
            {
                "tool": "slopkeeper",
                "version": VERSION,
                "command": getattr(args, "command", ""),
                "ok": False,
                "error": e.as_dict(),
            }
        )
        return EXIT_VALIDATION if e.code.startswith("plan_") else EXIT_ERROR
    except Exception as e:  # noqa: BLE001 — the agent needs the envelope even on bugs
        emit_json(
            {
                "tool": "slopkeeper",
                "version": VERSION,
                "command": getattr(args, "command", ""),
                "ok": False,
                "error": {
                    "code": "internal",
                    "message": f"{type(e).__name__}: {e}",
                    "recovery": "Report this; include the command.",
                },
            }
        )
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
