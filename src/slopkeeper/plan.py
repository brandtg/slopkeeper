# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Plan contract: validate agent-authored mapping plans, render them for humans.

A plan is the ONLY way anything moves. It is a JSON document the agent writes
after reading the inventory, the human approves (possibly after editing),
and `slopkeeper apply` executes all-or-nothing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    HIDDEN_SKIP_DIRS,
    MANIFEST_NAME,
    PLAN_SCHEMA,
    STATE_DIR,
    SlopKeeperError,
    corpus_path,
    resolve_chains,
    safe_relpath,
    sha256_file_exact,
    state_dir,
)
from .ignore import IGNORE_NAME, corpus_walk, glob_to_regex, load_ignore
from .references import (
    CitationIndex,
    _PER_FILE_CAP,
    _expand_token,
    _post_move_path,
    _token_spans,
    _walk_rel_paths,
    check_edits_after_rewrites,
    is_text_file,
)

VALID_FOLDER_KINDS = {"bot", "case", "topic", "project", "archive", "scratch"}


def load_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SlopKeeperError(
            "plan_not_json",
            f"plan is not valid JSON: {e}",
            "Fix the JSON syntax and re-validate.",
        )
    except OSError as e:
        raise SlopKeeperError("plan_unreadable", str(e))
    if not isinstance(plan, dict):
        raise SlopKeeperError("plan_not_object", "plan must be a JSON object")
    return plan


def _req(plan: dict, key: str):
    if key not in plan:
        raise SlopKeeperError(
            "plan_missing_field",
            f"plan is missing required field '{key}'",
            f"Add '{key}'. See: slopkeeper plan schema",
        )
    return plan[key]


def _external_target(raw: str, ext_roots: list[Path]) -> Path | None:
    """The resolved path when `raw` is an absolute path under one of the
    declared external roots, else None (handle it as an in-root path).

    External edit paths are ABSOLUTE by contract: with more than one
    declared root a root-relative path would be ambiguous, and an
    absolute path is self-describing in the plan, the journal and the
    render. resolve() makes the containment test symlink-safe the same
    way safe_relpath's absolute-form check is.
    """
    p = Path(raw)
    if not p.is_absolute():
        return None
    rp = p.resolve()
    return rp if any(rp.is_relative_to(er) for er in ext_roots) else None


def _keep_path(k) -> str | None:
    """keep entries may be plain strings or {path, reason} objects."""
    if isinstance(k, dict):
        return str(k.get("path", "")) or None
    if isinstance(k, str):
        return k
    return None


def _reject_dead_glob_prefix_rewrites(
    root: Path, entries: list[dict], chains: dict[str, str]
) -> None:
    """Refuse a prefix_rewrites entry whose glob/brace citations die (D2).

    Every glob/brace token cited under the entry's source prefix is
    expanded against the corpus inventory (the same engine the rewrite
    pass uses, so validate and apply can never disagree). A token is
    coherent when every corpus file it names moves into ONE destination
    directory — the pass rewrites it there. Anything else (matches stay
    put, split across destinations, or the pattern names no corpus file
    at all) would leave the rewritten pattern pointing at nothing, so
    validate refuses with prefix_rewrite_dead_glob. The per-entry
    `acknowledge_broken: true` flag consciously accepts that — the
    agent's own out-of-corpus references (`/tmp`, `~/Downloads`) are the
    agent's problem, accepted explicitly, never silently.
    """
    if not entries:
        return
    active = [e for e in entries if not e.get("acknowledge_broken")]
    if not active:
        return
    all_prefixes = tuple(
        (str(e["from"]).strip().rstrip("/"), str(e["to"]).strip().rstrip("/"))
        for e in entries
    )
    index = CitationIndex(_walk_rel_paths(root))
    checked: set[str] = set()
    for tp in _walk_rel_paths(root):
        if not is_text_file(tp):
            continue
        try:
            raw = (root / tp).read_bytes()[:_PER_FILE_CAP]
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            for token, _kind, _s, _e in _token_spans(line):
                if (
                    not ("*" in token or "{" in token or "<" in token)
                    or token in checked
                ):
                    continue
                for e in active:
                    src_n = str(e["from"]).strip().rstrip("/")
                    if token != src_n and not token.startswith(src_n + "/"):
                        continue
                    checked.add(token)
                    outcome = _expand_token(token, index, chains, all_prefixes)
                    # E2: a pattern naming no corpus file is a
                    # convention string — the entry applies the rename
                    # to it (never refused). Only a SPLIT or stay-put
                    # pattern is dead: the swap would point at nothing.
                    if outcome is not None and outcome[0] == "skip":
                        raise SlopKeeperError(
                            "prefix_rewrite_dead_glob",
                            f'prefix_rewrites entry "{e["from"]}" -> "{e["to"]}" '
                            f"cannot rewrite the glob/brace citation {token!r} "
                            "coherently post-move",
                            "Every corpus file the pattern names must move into "
                            "one destination directory. Fix the plan's moves, "
                            'drop the entry, or set "acknowledge_broken": true '
                            "on the entry to accept the broken citation "
                            "consciously.",
                        )


def _reject_merge_collisions(root: Path, moves: list[dict]) -> None:
    """E1: refuse a statically knowable merge collision, naming both moves.

    A directory move whose destination is created by another move of
    the same plan MERGES file by file at execute (the flat final paths
    the whole tool models). A merge is only impossible when a child
    name collides: the target exists on disk, or another move's
    destination IS the target or lives under it. Two directory moves
    onto the same created folder collide per child at execute (loud,
    partial) — their child sets are not compared here.
    """
    if not moves:
        return
    for m in moves:
        src = str(m["from"])
        dst = str(m["to"]).rstrip("/")
        src_abs = root / src
        if not src_abs.is_dir() or (root / dst).exists():
            # An existing destination is the dst_exists pre-flight's
            # business (a collision, never a merge).
            continue
        created = any(
            t == dst or t.startswith(dst + "/") or dst.startswith(t.rstrip("/") + "/")
            for other in moves
            if other is not m
            for t in [str(other["to"]).rstrip("/")]
        )
        if not created:
            continue
        for child in src_abs.iterdir():
            target = dst + "/" + child.name
            blocked = (root / target).exists() or any(
                t == target or t.startswith(target + "/")
                for other in moves
                if other is not m
                for t in [str(other["to"]).rstrip("/")]
            )
            if blocked:
                raise SlopKeeperError(
                    "merge_collision",
                    f'directory move "{src}" -> "{dst}" collides: "{target}" '
                    "exists or is created by another move",
                    "The destination folder is created by another move of "
                    "this plan; the merge refuses a name collision. Rename "
                    "one destination, or drop one of the two moves.",
                )


def validate_plan(plan: dict, root: Path, allow_partial: bool = False) -> dict:
    """Validate a plan against the corpus on disk. Returns the normalized plan.

    Raises SlopKeeperError with a code the agent can act on. A plan may
    declare its own scope with "partial": true (delta plans), which is
    equivalent to passing --allow-partial.
    """
    if plan.get("partial") is True:
        allow_partial = True
    if plan.get("schema") != PLAN_SCHEMA:
        raise SlopKeeperError(
            "plan_bad_schema",
            f"plan.schema must be {PLAN_SCHEMA!r}, got {plan.get('schema')!r}",
            "Regenerate the plan from the current schema: slopkeeper plan schema",
        )

    # Phase 4: the citation rewrite pass is default-on ("auto" when the
    # field is absent — alpha plans intentionally gain it); "off" is the
    # explicit opt-out. Anything else is refused loudly.
    rewrites = plan.get("rewrites")
    if rewrites is not None:
        if (
            not isinstance(rewrites, dict)
            or set(rewrites) - {"mode"}
            or ("mode" in rewrites and rewrites["mode"] not in ("auto", "off"))
        ):
            raise SlopKeeperError(
                "plan_bad_rewrites",
                f'plan.rewrites must be {{"mode": "auto"}} or {{"mode": "off"}}, '
                f"got {rewrites!r}",
                'The rewrite pass is on by default; set "rewrites": {"mode": '
                '"off"} to opt out. "mode" is the only key.',
            )

    root = root.resolve()
    # `.slopkeeperignore` (DESIGN §6.6): ignored paths are invisible to
    # the plan pipeline. Checked at validate AND apply (apply re-runs
    # this function), so an ignore file edited after authoring is
    # caught, never silently moved through.
    ignore = load_ignore(root)
    # D12: remove_emptied opts into journaled removal of directories
    # the plan empties. Bool or absent; anything else refused loudly.
    if "remove_emptied" in plan and not isinstance(plan["remove_emptied"], bool):
        raise SlopKeeperError(
            "plan_bad_remove_emptied",
            f"plan.remove_emptied must be a boolean, got {plan['remove_emptied']!r}",
            'Set "remove_emptied": true to have apply remove directories '
            "the plan emptied (journaled, undoable), or drop the field.",
        )
    if Path(_req(plan, "corpus_root")).resolve() != root:
        raise SlopKeeperError(
            "plan_wrong_root",
            f"plan.corpus_root {plan['corpus_root']} != {root}",
            "Regenerate or re-target the plan for this corpus.",
        )

    moves = _req(plan, "moves")
    if not isinstance(moves, list):
        raise SlopKeeperError("plan_no_moves", "plan.moves must be a list")

    seen_src, seen_dst = set(), {}
    normalized: list[dict] = []

    def _normalize_move(m, label: str) -> dict:
        """One move through the shared contract: path safety, no-op check,
        duplicate source/destination. Explicit moves and Phase 8 rule
        expansions go through this same body, so an expanded plan journals
        exactly like its hand-written equivalent."""
        if not isinstance(m, dict) or "from" not in m or "to" not in m:
            raise SlopKeeperError("plan_bad_move", f"{label} must have 'from' and 'to'")
        src = safe_relpath(m["from"], root)
        dst = safe_relpath(m["to"], root)
        if src == dst:
            raise SlopKeeperError("plan_noop_move", f"{label}: from == to ({src})")
        if src in seen_src:
            raise SlopKeeperError("plan_dup_src", f"two moves share source {src}")
        if dst in seen_dst:
            recovery = (
                "Each move needs a distinct destination file path — two "
                "sources cannot land on one file (chains a->b, b->c are "
                "resolved automatically and are not this error)."
            )
            if str(m["to"]).endswith("/") or (root / dst).is_dir():
                # The classic mistake: 'to' names a directory (usually a
                # filename omitted from the plan), so say exactly that —
                # "resolve the collision" alone costs the agent a
                # round-trip re-reading the schema.
                recovery = (
                    "Destination must be the full file path, not a directory — "
                    "the filename is missing from 'to' "
                    f"(likely {dst}/{src.name})."
                )
            raise SlopKeeperError(
                "plan_dup_dst",
                f"two moves share destination {dst} ({seen_dst[dst]} and {src})",
                recovery,
            )
        seen_src.add(src)
        seen_dst[dst] = src
        out = {
            "from": src.as_posix(),
            "to": dst.as_posix(),
            "reason": str(m.get("reason", "")).strip(),
        }
        # B1: a move may opt its subtree out of the citation rewrite
        # pass ("rewrite": false) — a vendored prototype whose internal
        # references travel with it. Absent/true = the default pass.
        if "rewrite" in m:
            if not isinstance(m["rewrite"], bool):
                raise SlopKeeperError(
                    "plan_bad_move",
                    f"{label}: 'rewrite' must be a boolean (got {m['rewrite']!r})",
                    'Set "rewrite": false to skip the rewrite pass for this '
                    "subtree, or drop the key.",
                )
            if m["rewrite"] is False:
                out["rewrite"] = False
        return out

    for i, m in enumerate(moves):
        normalized.append(_normalize_move(m, f"moves[{i}]"))

    # Rule-based moves (Phase 8): `rules` entries expand into concrete
    # moves at validate — BEFORE the gate, so the render lists real
    # paths and apply/undo need nothing new. `match` is a path prefix
    # (`reports/_org/`) or a `/`-aware glob (`reports/_org/*.md`;
    # `**` crosses directories, `*`/`?` do not).
    #
    # Precedence (B3): explicit moves beat rules — a rule never
    # overrides a recorded explicit move, whatever the destination —
    # and among rules the FIRST match wins, so carve-outs are ordered
    # before their catch-all. A later rule whose match was already
    # taken counts as superseded (reported per rule, never silent).
    # "preserve": "tree" (B5) keeps each matched file's path BELOW the
    # match prefix instead of flattening it into <to>/<basename>.
    raw_rules = plan.get("rules") or []
    if not isinstance(raw_rules, list):
        raise SlopKeeperError(
            "plan_bad_rules",
            'rules must be a list of {"match", "to"} objects',
            'Declare "rules": [{"match": "reports/_org/", "to": "reports/org/"}] '
            "— or omit it for explicit moves only.",
        )
    rules: list[dict] = []
    for i, r in enumerate(raw_rules):
        if (
            not isinstance(r, dict)
            or set(r) - {"match", "to", "preserve", "rewrite"}
            or not isinstance(r.get("match"), str)
            or not r["match"].strip()
            or not isinstance(r.get("to"), str)
            or not r["to"].strip()
        ):
            raise SlopKeeperError(
                "plan_bad_rules",
                f'rules[{i}] must be {{"match": ..., "to": ...}} with '
                f"non-empty string values, got {r!r}",
                '"match" is a corpus-relative path prefix or glob, "to" the '
                "destination directory the matched files land in. Optional "
                'keys: "preserve": "tree" and "rewrite": false.',
            )
        if "preserve" in r and r["preserve"] != "tree":
            raise SlopKeeperError(
                "plan_bad_rules",
                f'rules[{i}]: "preserve" must be "tree" (got {r["preserve"]!r})',
                'Use "preserve": "tree" to keep each file\'s subdirectory '
                "path below the match prefix, or drop the key to flatten.",
            )
        if "rewrite" in r and not isinstance(r["rewrite"], bool):
            raise SlopKeeperError(
                "plan_bad_rules",
                f'rules[{i}]: "rewrite" must be a boolean (got {r["rewrite"]!r})',
                'Set "rewrite": false to skip the citation rewrite pass for '
                "the rule's matches, or drop the key.",
            )
        match_raw = r["match"].strip()
        to_dir = safe_relpath(r["to"], root)  # path_escape on traversal
        to_abs = root / to_dir
        if to_abs.is_file():
            raise SlopKeeperError(
                "plan_bad_rules",
                f"rules[{i}]: destination {to_dir} is an existing file",
                "Rule destinations name directories — matched files land "
                "inside as <to>/<basename>.",
            )
        preserve_tree = r.get("preserve") == "tree"
        if any(ch in match_raw for ch in "*?["):
            rx = glob_to_regex(match_raw)
            # The static leading directory prefix of the glob: the
            # segments before the first wildcard. `preserve: tree`
            # keeps the matched path below it (a pattern with no static
            # prefix preserves from the corpus root).
            static_parts = []
            for seg in match_raw.split("/"):
                if any(ch in seg for ch in "*?["):
                    break
                static_parts.append(seg)
            static_prefix = "/".join(static_parts)

            def matcher(rel: str, _rx=rx) -> bool:
                return _rx.match(rel) is not None

            kind = "glob"
        else:
            prefix = match_raw.strip("/")
            if not prefix:
                raise SlopKeeperError(
                    "plan_bad_rules",
                    f"rules[{i}]: empty path prefix",
                    "Name a directory inside the corpus, e.g. "
                    '"match": "reports/_org/".',
                )
            static_prefix = prefix

            def matcher(rel: str, _p=prefix) -> bool:
                return rel == _p or rel.startswith(_p + "/")

            kind = "prefix"

        def _dest(rel: str, _to=to_dir, _sp=static_prefix, _pt=preserve_tree) -> str:
            # preserve: tree keeps the path below the match's static
            # prefix; the plain rule preserves the name (flatten). A
            # file that IS the prefix (rel == static prefix) has no
            # suffix to preserve — it falls back to the basename.
            if _pt:
                suffix = (
                    rel[len(_sp) :].lstrip("/") if rel.startswith(_sp + "/") else ""
                )
                if suffix:
                    return (_to / Path(suffix)).as_posix()
            return (_to / Path(rel).name).as_posix()

        rules.append(
            {
                "match": match_raw,
                "to": to_dir.as_posix(),
                "kind": kind,
                "matcher": matcher,
                "dest": _dest,
                "static_prefix": static_prefix,
                "preserve": preserve_tree,
                # True iff the rule opts its matches out of the rewrite
                # pass (the expansion stamps it on each move — C1).
                "rewrite": r.get("rewrite") is False,
            }
        )

    if rules:
        # One corpus walk feeds every rule (only rule plans pay for it).
        visible: list[str] = []
        for dirpath, _dirnames, filenames in corpus_walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            for fn in filenames:
                visible.append((rel_dir / fn).as_posix() if rel_dir.parts else fn)
        # First match wins (B3): a file matched by an earlier rule never
        # reaches a later one. `match_count` still counts every rule's
        # raw matches (the stale-rule guard), `superseded` counts what a
        # later rule lost to an earlier one.
        rule_dst: dict[str, str] = {}
        rule_idx: dict[str, int] = {}
        match_count = [0] * len(rules)
        superseded_count = [0] * len(rules)
        for rel in visible:
            winner = None
            for ri, rule in enumerate(rules):
                if rule["matcher"](rel):
                    match_count[ri] += 1
                    if winner is None:
                        winner = ri
                    else:
                        superseded_count[ri] += 1
            if winner is not None:
                rule_dst[rel] = rules[winner]["dest"](rel)
                rule_idx[rel] = winner
        for ri, rule in enumerate(rules):
            if match_count[ri] == 0:
                raise SlopKeeperError(
                    "plan_bad_rules",
                    f"rule {rule['match']!r} matched no visible files",
                    "Check the path (or the corpus already moved) — a rule "
                    "that matches nothing is a stale or typo'd plan.",
                )
        # Recorded decisions win over rules: explicit moves (no matter
        # the destination — the human's recorded choice is the more
        # specific decision), keep and unresolved entries, and anything
        # inside an explicitly moved directory (it travels with the dir
        # — expanding it would be a plan_nested_moves).
        explicit_dst = {m["from"]: m["to"] for m in normalized}
        keep_exempt = {kp for kp in (_keep_path(k) for k in plan.get("keep", [])) if kp}
        unresolved_exempt = {
            u.get("path")
            for u in plan.get("unresolved", [])
            if isinstance(u, dict) and isinstance(u.get("path"), str)
        }
        exempt_prefixes = tuple(
            sorted(
                e.rstrip("/") + "/"
                for e in list(keep_exempt) + list(unresolved_exempt)
                if (root / e).is_dir()
            )
        )
        dir_src_prefixes = tuple(
            sorted(
                m["from"].rstrip("/") + "/"
                for m in normalized
                if (root / m["from"]).is_dir()
            )
        )
        expanded_count = [0] * len(rules)
        shadowed_count = [0] * len(rules)
        flattened_count = [0] * len(rules)
        for rel in sorted(rule_dst):
            dst = rule_dst[rel]
            ri = rule_idx[rel]
            if rel in explicit_dst:
                continue  # explicit moves beat rules (B3) — no conflict
            if rel.startswith(dir_src_prefixes):
                continue  # moves with its directory; expanding would nest
            if (
                rel in keep_exempt
                or rel in unresolved_exempt
                or rel.startswith(exempt_prefixes)
            ):
                # Recorded decisions shadow the rule (B4): counted and
                # reported, never silent.
                shadowed_count[ri] += 1
                continue
            normalized.append(
                _normalize_move(
                    {
                        "from": rel,
                        "to": dst,
                        "reason": f"rule: {rules[ri]['match']} -> {rules[ri]['to']}",
                        # C1: an expanded move takes the rule's
                        # rewrite opt-out — and ONLY that. rules[i]
                        # ["rewrite"] is True iff the rule declared
                        # "rewrite": false; a rule without the key
                        # (or with true) expands to the default pass.
                        **({"rewrite": False} if rules[ri]["rewrite"] else {}),
                    },
                    f"rule {rules[ri]['match']!r}",
                )
            )
            expanded_count[ri] += 1
            # B5 visibility: a flatten (no preserve:tree) that pulls a
            # file from BELOW the match prefix loses its subdirectory.
            sub = rel[len(rules[ri]["static_prefix"]) :].strip("/")
            if not rules[ri]["preserve"] and "/" in sub:
                flattened_count[ri] += 1
        rule_warnings: list[str] = []
        for ri, rule in enumerate(rules):
            if expanded_count[ri] == 0 and shadowed_count[ri] > 0:
                rule_warnings.append(
                    f"rule {rule['match']!r} matched {match_count[ri]} file(s) "
                    f"but all {shadowed_count[ri]} are inside keep/unresolved "
                    "entries — it expanded to nothing (keep shadows rules)"
                )
        plan["rule_warnings"] = rule_warnings
        plan["rules"] = [
            {
                "match": r["match"],
                "to": r["to"],
                "kind": r["kind"],
                **({"preserve": "tree"} if r["preserve"] else {}),
                **({"rewrite": False} if r["rewrite"] else {}),
                "matched": match_count[i],
                "expanded": expanded_count[i],
                **({"superseded": superseded_count[i]} if superseded_count[i] else {}),
                **({"shadowed": shadowed_count[i]} if shadowed_count[i] else {}),
                **({"flattened": flattened_count[i]} if flattened_count[i] else {}),
            }
            for i, r in enumerate(rules)
        ]

    # Prefix rewrites (optional, B8): declared renames of directories the
    # plan empties FILE BY FILE. Such a directory has no chain entry (no
    # single move covers it), so convention-string citations of the old
    # prefix would dangle; the agent declares the rename and the rewrite
    # pass swaps the prefix on every citation that names it. Explicit
    # and shown at the gate — a mechanical rewrite can be wrong (the
    # 0.1.6 report's semantic-retarget case), so nothing is inferred.
    pr_raw = plan.get("prefix_rewrites") or []
    if not isinstance(pr_raw, list):
        raise SlopKeeperError(
            "plan_bad_prefix_rewrite",
            "prefix_rewrites must be a list of {from, to} objects",
            'Declare "prefix_rewrites": [{"from": "reports/_org/", '
            '"to": "reports/org/"}] — or omit it.',
        )
    normalized_pr: list[dict] = []
    for i, e in enumerate(pr_raw):
        if (
            not isinstance(e, dict)
            or set(e) - {"from", "to", "reason", "acknowledge_broken"}
            or not isinstance(e.get("from"), str)
            or not e["from"].strip()
            or not isinstance(e.get("to"), str)
            or not e["to"].strip()
            or not isinstance(e.get("acknowledge_broken", False), bool)
        ):
            raise SlopKeeperError(
                "plan_bad_prefix_rewrite",
                f'prefix_rewrites[{i}] must be {{"from", "to"}} strings, got {e!r}',
                '"from" is the old directory prefix (corpus-relative), "to" '
                'the new one. An optional "reason" is shown at the gate; '
                '"acknowledge_broken": true consciously accepts a citation '
                "the entry cannot fix (D2's dead-glob gate).",
            )
        src = safe_relpath(e["from"], root)
        dst = safe_relpath(e["to"], root)
        src_posix = src.as_posix()
        dst_posix = dst.as_posix()
        if src_posix == dst_posix:
            raise SlopKeeperError(
                "plan_bad_prefix_rewrite",
                f"prefix_rewrites[{i}]: from == to ({src_posix})",
                "A no-op prefix rewrite would rewrite nothing — drop it.",
            )
        if not (root / src_posix).is_dir():
            raise SlopKeeperError(
                "plan_bad_prefix_rewrite",
                f"prefix_rewrites[{i}]: 'from' is not an existing directory: "
                f"{src_posix}",
                "Prefix rewrites rename a directory that exists at plan "
                "time (usually one the plan empties file by file).",
            )
        normalized_pr.append(
            {
                "from": src_posix,
                "to": dst_posix,
                "reason": str(e.get("reason", "")).strip(),
                "acknowledge_broken": bool(e.get("acknowledge_broken", False)),
            }
        )
    plan["prefix_rewrites"] = normalized_pr

    if not normalized and not allow_partial:
        # T7: a zero-move PARTIAL plan is valid — the draft may file
        # every new file as unresolved (weak signals); validate, then
        # apply --execute records the human-decides backlog. Non-partial
        # plans still need at least one move — a rule that expands to
        # nothing does not count (Phase 8).
        raise SlopKeeperError(
            "plan_no_moves",
            "plan.moves is empty and no rule expanded to a move",
            'A zero-move plan is valid only with "partial": true (draft '
            "plans carry it) — add that key, propose at least one move, or "
            "add a rule that matches visible files.",
        )

    # A destination must not be inside a source (would nest a file under a
    # file). A destination EQUAL to another move's source is a chain
    # (a->b, b->c) — legal: resolve_chains keeps the hops and the executor
    # orders them so nothing is overwritten — so only the path-prefix
    # nesting case is rejected here. Ancestor set-membership (not a
    # pairwise scan): a rule-expanded plan can carry tens of thousands of
    # moves, and the destinations' ancestor chains are shallow.
    src_strs = {m["from"] for m in normalized}
    for m in normalized:
        parts = m["to"].split("/")
        for i in range(1, len(parts)):
            ancestor = "/".join(parts[:i])
            if ancestor in src_strs and ancestor != m["from"]:
                raise SlopKeeperError(
                    "plan_dst_under_src",
                    f"move destination {m['to']} is inside moved path {ancestor}",
                )

    # Existence + nested-move checks on disk.
    for m in normalized:
        src = corpus_path(root, Path(m["from"]))
        if not src.exists():
            raise SlopKeeperError(
                "plan_stale_src",
                f"source does not exist: {m['from']}",
                "The corpus changed since the plan was written. "
                "Re-inventory and regenerate the plan.",
            )
        if src.is_dir():
            prefix = m["from"].rstrip("/") + "/"
            nested = [o["from"] for o in normalized if o["from"].startswith(prefix)]
            if nested:
                raise SlopKeeperError(
                    "plan_nested_moves",
                    f"{m['from']} is a directory but its contents are moved "
                    f"separately: {nested[:3]}",
                    "Move the directory whole, or move only its files.",
                )
        if ignore.is_ignored(m["from"], src.is_dir()):
            raise SlopKeeperError(
                "move_source_ignored",
                f"move source {m['from']} is ignored by .slopkeeperignore",
                "The tool never touches ignored paths — drop the move, or "
                "remove the pattern from .slopkeeperignore.",
            )

    chains = resolve_chains(normalized)  # raises on cycles

    # E1: a directory move whose destination is created by another move
    # of this plan MERGES file by file at execute — refuse a statically
    # knowable name collision now, naming both moves.
    _reject_merge_collisions(root, normalized)

    # D2's validate-time gate: a prefix_rewrites entry whose glob/brace
    # citations cannot be coherently rewritten post-move is refused —
    # the swap would name nothing. Conscious override: acknowledge_broken.
    _reject_dead_glob_prefix_rewrites(root, plan["prefix_rewrites"], chains)

    # External roots (optional, additive in plan@1): directories OUTSIDE
    # the corpus that this plan's edits AND citation rewrites may touch
    # (Phase 5) — skills, guides, cron scripts and scratch harnesses
    # cite corpus paths too, and a move would break them silently.
    # Entries are absolute dir paths, or {"path": ..., "expect": N}
    # where expect is a would-be-rewrite count assertion: the apply
    # computes the count first and refuses (rewrite_expect_mismatch)
    # before ANY write on mismatch. Absent or empty → exactly the
    # historical contract: every edit target must live inside the root
    # (an outside path is a path_escape).
    ext_raw = plan.get("external_roots") or []
    if not isinstance(ext_raw, list):
        raise SlopKeeperError(
            "plan_bad_external_root",
            "external_roots must be a list of absolute directory paths "
            '(or {"path": ..., "expect": N} objects)',
            'Declare "external_roots": ["/abs/dir", ...] — or omit it to '
            "edit inside the corpus only.",
        )
    ext_roots: list[Path] = []
    normalized_ext: list[dict] = []
    for x in ext_raw:
        if isinstance(x, str):
            path_raw, expect = x, None
        elif isinstance(x, dict):
            if not isinstance(x.get("path"), str) or set(x) - {"path", "expect"}:
                raise SlopKeeperError(
                    "plan_bad_external_root",
                    "external root objects must be "
                    f'{{"path": ..., "expect": N}}, got {x!r}',
                    'Declare {"path": "/abs/dir", "expect": 3} — "path" is '
                    'required, "expect" optionally asserts a rewrite count.',
                )
            path_raw = x["path"]
            expect = x.get("expect")
            if expect is not None and (
                isinstance(expect, bool) or not isinstance(expect, int) or expect < 0
            ):
                raise SlopKeeperError(
                    "plan_bad_external_root",
                    "external root expect must be a non-negative integer, "
                    f"got {expect!r}",
                    'Set "expect" to the number of citations the pass should '
                    "rewrite in this root (refs counts them), or drop it.",
                )
        else:
            raise SlopKeeperError(
                "plan_bad_external_root",
                f"external_roots entries must be strings or objects, got {x!r}",
                'Use "/abs/dir" or {"path": "/abs/dir", "expect": 3}.',
            )
        p = Path(path_raw)
        if not p.is_absolute() or not p.is_dir():
            raise SlopKeeperError(
                "plan_bad_external_root",
                f"external root must be an absolute, existing directory: {path_raw!r}",
                "Create the directory, fix the path, or drop the entry.",
            )
        p = p.resolve()
        ext_roots.append(p)
        entry: dict = {"path": p.as_posix()}
        if expect is not None:
            entry["expect"] = expect
        normalized_ext.append(entry)
    plan["external_roots"] = normalized_ext

    # Scan paths (optional, additive in plan@1, Phase 6): absolute dirs
    # the corpus walk never sees — ignored subtrees, per-corpus noise —
    # that this plan's citation rewrites AND edits may touch. Like
    # external_roots but for the corpus's own invisible layer: files
    # under them join the rewrite pass (reported/journaled under their
    # absolute path) and an edit targeting them is allowed despite
    # edit_target_ignored. Naming a skip-list member (.git, ...) is
    # refused — tool layer is never read, by anyone.
    scan_raw = plan.get("scan_paths") or []
    if not isinstance(scan_raw, list):
        raise SlopKeeperError(
            "plan_bad_scan_path",
            "scan_paths must be a list of absolute directory paths",
            'Declare "scan_paths": ["/abs/dir", ...] — or omit it.',
        )
    scan_roots: list[Path] = []
    for x in scan_raw:
        if not isinstance(x, str) or not x:
            raise SlopKeeperError(
                "plan_bad_scan_path",
                f"scan_paths entries must be absolute directory paths, got {x!r}",
                'Declare "scan_paths": ["/abs/dir", ...] — or omit it.',
            )
        p = Path(x)
        if not p.is_absolute() or not p.is_dir():
            raise SlopKeeperError(
                "plan_bad_scan_path",
                f"scan path must be an absolute, existing directory: {x!r}",
                "Create the directory, fix the path, or drop the entry.",
            )
        p = p.resolve()
        if any(part in HIDDEN_SKIP_DIRS for part in p.parts):
            raise SlopKeeperError(
                "scan_path_skip_list",
                f"scan path {p} is a tool-layer skip-list directory",
                "Skip-list members (.git, .slopkeeper, .venv, node_modules, "
                "...) are never read — name a directory outside them.",
            )
        scan_roots.append(p)
    plan["scan_paths"] = [p.as_posix() for p in scan_roots]

    # Edits: declarative reference rewrites applied after the moves.
    # edit.file is the move-SOURCE path if the file is moving in this plan,
    # else its current path. Applied post-move at the final location.
    # An edit under a declared external root (or, Phase 6, a declared
    # scan path) is kept ABSOLUTE (see _external_target); everything
    # else must be inside the corpus.
    edits = []
    for i, e in enumerate(plan.get("edits", [])):
        if not isinstance(e, dict) or not {"file", "find", "replace"} <= set(e):
            raise SlopKeeperError(
                "plan_bad_edit",
                f"edits[{i}] must have 'file', 'find' and 'replace'",
                "See: slopkeeper plan schema",
            )
        raw = e["file"]
        ext = None
        if isinstance(raw, str) and Path(raw).is_absolute():
            # An absolute path under the CORPUS root is an in-root edit
            # (normalized relative below), even when external_roots or
            # scan_paths are declared; anything else absolute is an
            # external/declared target only when under a declared root.
            if not Path(raw).resolve().is_relative_to(root):
                ext = _external_target(raw, ext_roots + scan_roots)
        edits.append(
            {
                "file": (
                    ext.as_posix()
                    if ext is not None
                    else safe_relpath(raw, root).as_posix()
                ),
                "find": str(e["find"]),
                "replace": str(e["replace"]),
                "reason": str(e.get("reason", "")).strip(),
            }
        )
        if ext is None and ignore.is_ignored(edits[-1]["file"]):
            # Phase 6: an ignored edit target is writable only when its
            # directory is declared (external_roots above, or a
            # scan_path whose root contains this relative path).
            resolved = (root / edits[-1]["file"]).resolve()
            if not any(resolved.is_relative_to(sr) for sr in scan_roots):
                raise SlopKeeperError(
                    "edit_target_ignored",
                    f"edit target {edits[-1]['file']} is ignored by .slopkeeperignore",
                    "Declare its directory via scan_paths (or external_roots) "
                    "to allow the write, drop the edit, or remove the pattern "
                    "from .slopkeeperignore.",
                )
    plan["edits"] = edits

    # C3/C4's one rule, checked where the plan contract lives: edits run
    # AFTER the citation rewrite pass, so each edit's `find` must occur
    # exactly once in the file's POST-rewrite text. apply re-validates,
    # so validate and apply can never disagree about which text `find`
    # matches (the 0.1.8 report's validate-passes/apply-fails split).
    if edits:
        check_edits_after_rewrites(
            root,
            normalized,
            chains,
            edits,
            plan.get("prefix_rewrites") or [],
        )

    # Unresolved backlog entries: each needs a non-empty string `path`.
    # apply stores u.get("path") into the manifest — a null path would
    # TypeErrors every later `status` walk (root / None), so the gate
    # refuses instead of poisoning the state file. Validated
    # unconditionally (also under --allow-partial): the backlog is
    # written to the manifest on every execute.
    raw_unresolved = plan.get("unresolved", [])
    if not isinstance(raw_unresolved, list):
        raise SlopKeeperError(
            "plan_bad_unresolved",
            "unresolved must be a list of {path, reason} objects, "
            f"got {type(raw_unresolved).__name__}",
            'Use "unresolved": [{"path": "<file>", "reason": "<why it waits>"}] '
            "— or drop the entry.",
        )
    for i, u in enumerate(raw_unresolved):
        if (
            not isinstance(u, dict)
            or not isinstance(u.get("path"), str)
            or not u["path"].strip()
        ):
            raise SlopKeeperError(
                "plan_bad_unresolved",
                f"unresolved[{i}] needs a non-empty string 'path' (got {u!r})",
                'Every unresolved entry is {"path": "<file>", "reason": "<why>"} '
                "— fix or drop it.",
            )

    # Deliberate skips (DESIGN §6.1): top-level entries the plan
    # ignores, named explicitly so "we never looked at it" is a recorded
    # decision instead of a magic default. Names only — one path
    # component — and each must exist at the top level (a skip of
    # something that isn't there is a typo'd plan, not a decision).
    skip_raw = plan.get("skip", [])
    if not isinstance(skip_raw, list) or not all(
        isinstance(s, str) and s.strip() and "/" not in s for s in skip_raw
    ):
        raise SlopKeeperError(
            "plan_bad_skip",
            "skip must be a list of top-level entry names",
            'Use "skip": ["docs"] — plain names (no slashes) of entries '
            "the plan deliberately did not examine.",
        )
    skip = set(skip_raw)
    for s in sorted(skip):
        if not (root / s).exists():
            raise SlopKeeperError(
                "plan_bad_skip",
                f"skip names a nonexistent entry: {s}",
                "Skip only entries that exist at the top level of the corpus.",
            )
    plan["skip"] = sorted(skip)

    # Coverage: every top-level entry must be moved, kept, edited, or
    # unresolved (or named in skip). Nested entries are NOT individually
    # required — moving a directory covers its contents — but nothing
    # stops nested sources: a recursive sweep moves docs/x.md directly.
    if not allow_partial:
        covered = {"."}
        for m in normalized:
            covered.add(Path(m["from"]).parts[0])
            covered.add(Path(m["to"]).parts[0])  # destinations are covered too
        for e in plan["edits"]:
            if Path(e["file"]).is_absolute():
                continue  # external edit: outside the corpus, covers nothing
            covered.add(Path(e["file"]).parts[0])  # edited in place: covered

        keeps = set()
        for k in plan.get("keep", []):
            kp = _keep_path(k)
            if kp is None:
                raise SlopKeeperError(
                    "plan_bad_keep",
                    f"keep entry is neither a string nor a {{path}} object: {k!r}",
                    'Use "keep": ["name", {"path": "name", "reason": "..."}].',
                )
            keeps.add(Path(kp).parts[0])
        # The never-content set (tool layer + legacy state) is excluded from
        # coverage the way the old blanket dot-rule was — it is the walk
        # contract, not a plan decision. Everything else visible at the top
        # level must be covered.
        hidden = HIDDEN_SKIP_DIRS | {STATE_DIR}
        for entry in list(root.iterdir()):
            if entry.name in skip or entry.name in hidden:
                continue  # declared skips and the never-content set are not content
            if entry.name.startswith(".") and not entry.is_dir():
                # Root-level dot-FILES (.env, .gitignore, ...): config and
                # secrets, not sweepable content. Root-level dot-DIRS are
                # ordinary content as of fp4 and MUST be covered — a plan
                # that ignores .claude/ silently would strand it.
                continue
            if ignore.is_ignored(entry.name, entry.is_dir()):
                continue  # .slopkeeperignore: invisible is not uncovered
            if entry.is_dir() and (entry / STATE_DIR / MANIFEST_NAME).exists():
                # Phase 7: a nested corpus is a separate tended tree —
                # invisible to the parent's walk, so not the plan's to
                # cover (the same rule the walk applies everywhere).
                continue
            if entry.name not in covered:
                unresolved = {
                    u.get("path")
                    for u in plan.get("unresolved", [])
                    if isinstance(u, dict)
                }
                if entry.name not in unresolved and entry.name not in keeps:
                    raise SlopKeeperError(
                        "plan_incomplete",
                        f"top-level entry '{entry.name}' is not covered by moves, "
                        f"keep, or unresolved",
                        "Add it to moves, 'keep', or 'unresolved' — or re-validate "
                        "with --allow-partial for a partial sweep. Root-level "
                        "dot-files (.env, ...) are exempt; content dot-directories "
                        "(.claude, ...) are not.",
                    )

    # Folders.
    folders = []
    for f in plan.get("folders", []):
        if "path" not in f:
            raise SlopKeeperError("plan_bad_folder", f"folder entry missing path: {f}")
        kind = f.get("kind", "project")
        if kind not in VALID_FOLDER_KINDS:
            raise SlopKeeperError(
                "plan_bad_kind",
                f"folder {f['path']}: unknown kind {kind!r}",
                f"kind must be one of {sorted(VALID_FOLDER_KINDS)}",
            )
        folders.append(
            {
                "path": safe_relpath(f["path"], root).as_posix(),
                "kind": kind,
                "role": str(f.get("role", "")).strip(),
                "boundaries": [
                    str(b) for b in f.get("boundaries", []) if str(b).strip()
                ],
                "state_md": bool(f.get("state_md", kind == "bot")),
                "agents_md": f.get("agents_md"),
            }
        )
    plan["folders"] = folders
    plan["moves"] = normalized
    # B8 advisory: directories the plan empties file by file, with
    # destination breakdowns (and a majority suggestion) for authoring
    # prefix_rewrites. Computed here so validate --out, the render and
    # apply all read the same field.
    plan["emptied_dir_suggestions"] = emptied_dir_suggestions(root, plan)
    # D12: when the plan opts into removal, the exact list is computed
    # here so validate --out, the render and apply all read the same
    # field — the gate shows what will be removed before approval.
    if plan.get("remove_emptied"):
        plan["remove_emptied_dirs"] = dirs_emptied_by(root, plan)
    return plan


def straggler_report(root: Path, plan: dict) -> list[dict]:
    """Files left behind in every folder this plan removes files from (Phase 8).

    Advisory — visible at the gate, never fatal: a plan that sweeps half
    of misc/ and strands the rest must show the strands instead of
    passing silently (the field report's agent scripted this check by
    hand). Called with a VALIDATED plan: `moves` is the normalized,
    rule-expanded list. Returns [{"dir": ..., "count": N, "remaining":
    [paths]}] sorted by folder; `render_markdown` caps the display, the
    validate --out JSON carries the full list.

    A folder is scanned only when a FILE-source move removes files from
    it — a wholesale directory move leaves nothing behind inside itself,
    and neither does any folder inside one. The corpus root is scanned
    non-recursively: its leftovers are top-level entries, which the
    coverage check governs, and recursing there would list the whole
    corpus behind any root-file move. Move sources, keep entries and
    unresolved entries are recorded decisions, not stragglers.
    """
    root = root.resolve()
    moves = plan.get("moves", [])
    if not moves:
        return []
    ignore = load_ignore(root)
    srcs = {m["from"] for m in moves}
    dir_srcs = {s for s in srcs if (root / s).is_dir()}
    dir_src_prefixes = tuple(sorted(s.rstrip("/") + "/" for s in dir_srcs))
    parents = sorted(
        {
            str(Path(s).parent)
            for s in srcs
            if s not in dir_srcs and not s.startswith(dir_src_prefixes)
        }
    )
    exempt = set(srcs)
    for entry in plan.get("keep", []):
        kp = _keep_path(entry)
        if kp:
            exempt.add(kp)
    for u in plan.get("unresolved", []):
        if isinstance(u, dict) and isinstance(u.get("path"), str):
            exempt.add(u["path"])
    prefix_exempt = tuple(
        sorted(e.rstrip("/") + "/" for e in exempt if (root / e).is_dir())
    )
    report = []
    for d in parents:
        base = root if d == "." else root / d
        remaining = []
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(root)
            kept = []
            for dn in dirnames:
                if dn in HIDDEN_SKIP_DIRS or dn == STATE_DIR:
                    continue
                rel_dn = (rel_dir / dn).as_posix() if rel_dir.parts else dn
                if ignore.is_ignored(rel_dn, is_dir=True):
                    continue
                if (Path(dirpath) / dn / STATE_DIR / MANIFEST_NAME).exists():
                    continue  # nested corpus: a separate tended tree
                kept.append(dn)
            dirnames[:] = sorted(kept)
            if d == "." and not rel_dir.parts:
                dirnames[:] = []  # the root scan stays at the top level
            for fn in filenames:
                rel = (rel_dir / fn).as_posix() if rel_dir.parts else fn
                if fn == IGNORE_NAME and not rel_dir.parts:
                    continue
                if ignore.is_ignored(rel, is_dir=False):
                    continue
                if rel in exempt or rel.startswith(prefix_exempt):
                    continue
                remaining.append(rel)
        if remaining:
            report.append(
                {"dir": d, "count": len(remaining), "remaining": sorted(remaining)}
            )
    return report


def dirs_emptied_by(root: Path, plan: dict) -> list[str]:
    """Directories the plan empties file by file (D12): every visible
    file under the dir moves (by its own move or wholesale inside a
    moved directory). The removal list behind `remove_emptied: true` —
    the same evidence emptied_dir_suggestions renders, plus ancestor
    dirs the per-file moves empty indirectly, minus the destination
    breakdown. "." is never a candidate."""
    root = root.resolve()
    moves = plan.get("moves", [])
    if not moves:
        return []
    dir_src_prefixes = tuple(
        sorted(
            m["from"].rstrip("/") + "/" for m in moves if (root / m["from"]).is_dir()
        )
    )
    file_srcs = {m["from"] for m in moves if (root / m["from"]).is_file()}
    candidates: set[str] = {str(Path(s).parent) for s in file_srcs}
    candidates.discard(".")
    candidates = {
        c for c in candidates if not any(c.startswith(p) for p in dir_src_prefixes)
    }
    # Ancestors of the candidates can be emptied indirectly (misc/a/x.md
    # and misc/b/y.md moving empties misc/a, misc/b AND misc).
    for cand in list(candidates):
        parts = Path(cand).parts
        for i in range(1, len(parts)):
            ancestor = "/".join(parts[:i])
            if not any(ancestor.startswith(p) for p in dir_src_prefixes):
                candidates.add(ancestor)
    if not candidates:
        return []

    def _moved_prefix_ok(rel: str) -> bool:
        return rel in file_srcs or rel.startswith(dir_src_prefixes)

    all_files: list[str] = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        for fn in filenames:
            all_files.append((rel_dir / fn).as_posix() if rel_dir.parts else fn)

    out: list[str] = []
    for cand in sorted(candidates, key=lambda c: (-c.count("/"), c)):
        prefix = cand + "/"
        inside = [f for f in all_files if f.startswith(prefix)]
        if not inside:
            continue
        if any(not _moved_prefix_ok(f) for f in inside):
            continue
        out.append(cand)
    return out


def emptied_dir_suggestions(root: Path, plan: dict) -> list[dict]:
    """Directories the plan empties FILE BY FILE (Phase 8 rules aside).

    B8: a directory whose files all move individually has no chain
    entry, so convention-string citations of the old prefix
    (`reports/_org/<date>_<topic>.md`) are NOT rewritten by the pass —
    the fix is an explicit `prefix_rewrites` entry, which the agent
    should author deliberately (a mechanical rewrite can be wrong).
    This report gives the agent the candidates and the destination
    breakdown; when one destination folder takes a clear majority, a
    ready-to-paste suggestion rides along. Advisory — never applied
    automatically.

    A directory counts as emptied when every visible file under it is
    moved (by its own move, or wholesale inside a moved directory).
    The candidates are the parent directories of the file sources —
    the dirs the slop was dumped into, and the prefixes convention
    strings name. "." is never a candidate (root-level coverage is
    the plan's own business).
    """
    root = root.resolve()
    moves = plan.get("moves", [])
    if not moves:
        return []
    dir_src_prefixes = tuple(
        sorted(
            m["from"].rstrip("/") + "/" for m in moves if (root / m["from"]).is_dir()
        )
    )
    file_srcs = {m["from"] for m in moves if (root / m["from"]).is_file()}
    dst_by_src = {m["from"]: m["to"] for m in moves}

    # One walk; the candidates are the PARENT dirs of the file sources
    # themselves — the dirs the slop was dumped into, and the prefixes
    # convention strings name. Ancestor chains above them are covered
    # by these (and are not where citations point).
    candidates: set[str] = {str(Path(s).parent) for s in file_srcs}
    candidates.discard(".")
    candidates = {
        c for c in candidates if not any(c.startswith(p) for p in dir_src_prefixes)
    }
    if not candidates:
        return []

    def _moved_prefix_ok(rel: str) -> bool:
        return rel in file_srcs or rel.startswith(dir_src_prefixes)

    all_files: list[str] = []
    for dirpath, _dirnames, filenames in corpus_walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        for fn in filenames:
            all_files.append((rel_dir / fn).as_posix() if rel_dir.parts else fn)

    out: list[dict] = []
    for cand in sorted(candidates):
        prefix = cand + "/"
        inside = [f for f in all_files if f.startswith(prefix)]
        if not inside:
            continue
        unmoved = [f for f in inside if not _moved_prefix_ok(f)]
        if unmoved:
            continue
        dests: dict[str, int] = {}
        for f in inside:
            dst = dst_by_src.get(f)
            if dst is None:
                continue
            d = str(Path(dst).parent)
            dests[d] = dests.get(d, 0) + 1
        if not dests:
            continue
        ranked = sorted(dests.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(dests.values())
        top, top_n = ranked[0]
        entry: dict = {
            "dir": cand,
            "moved_files": total,
            "destinations": ranked[:3],
        }
        if total >= 2 and top_n / total >= 2 / 3:
            entry["suggestion"] = {"from": cand + "/", "to": top + "/"}
        out.append(entry)
    return out


# Per-site statuses the citation rewrite pass reports (B2). A covered
# site is one the plan fixes — it is not gate-worthy risk detail.
_REWRITE_COVERED_STATUSES = ("rewrite", "no_change", "edit_covered")


def render_markdown(
    plan: dict,
    reference_report: dict | None = None,
    rewrite_summary: dict | None = None,
    footer: str | None = None,
    stragglers: list[dict] | None = None,
    code_anchor_risks: list[dict] | None = None,
    emptied_dirs: list[dict] | None = None,
) -> str:
    """Human-gate document: grouped, risk-stamped, diff-friendly.

    rewrite_summary (Phase 4), when given, is
    {"mode": "auto"|"off", "per_file": [{"file", "count"}], "count": N,
    "skipped": [...], "targets": {old_path: n}} — what the citation
    rewrite pass will do at apply, so the human gates on real numbers.

    footer (Phase 7) replaces the closing approve-by line — `move`
    renders the same gate but its approve action is re-running the move
    command with --execute, not editing a plan file.

    stragglers (Phase 8), when given, is straggler_report(root, plan)'s
    output: files left behind in folders the plan removes files from —
    advisory, capped here, full list in the validate --out JSON.

    code_anchor_risks (Phase 9), when given, is anchored_move_risks()'
    output: moves whose source is (or contains) a directory built in
    code. Informational — the gate shows the cost, the human decides.

    emptied_dirs (B8), when given, is emptied_dir_suggestions()' output:
    directories the plan empties file by file, with destination
    breakdowns and a ready-to-paste prefix_rewrites suggestion when one
    destination takes a clear majority. Advisory — the agent authors
    the prefix_rewrites entry, the render just shows the evidence.

    Rule plans (Phase 8) may carry tens of thousands of expanded moves;
    the per-folder tables show the first _MOVE_TABLE_CAP rows and say
    so, while the Summary table (one row per destination) stays
    complete — the gate never silently shrinks what it shows.
    """
    _MOVE_TABLE_CAP = 200
    _STRAGGLER_DIRS = 10
    _STRAGGLER_EXAMPLES = 5
    _REWRITE_ROW_CAP = 500

    def _cell(text: object) -> str:
        """Table-cell-safe inline code: backticks and pipes out."""
        return (str(text).replace("`", "'").replace("|", "\\|").replace("\n", " "))[
            :160
        ]

    moves = plan.get("moves", [])
    chains = resolve_chains(moves)
    by_folder: dict[str, list[dict]] = {}
    for m in moves:
        dst = chains[m["from"]]
        folder = str(Path(dst).parent)
        by_folder.setdefault(folder if folder != "." else "(root)", []).append(
            {**m, "final": dst}
        )

    lines = ["# Scaffolding plan", ""]
    meta = plan.get("based_on", {})
    lines.append(f"- Strategy: **{plan.get('strategy', '?')}**")
    lines.append(f"- Corpus: `{plan.get('corpus_root')}`")
    ext_roots = plan.get("external_roots") or []

    def _ext_label(r: object) -> str:
        # Raw plans carry plain path strings; validated plans carry
        # {"path": ..., "expect": N?} objects (Phase 5).
        if isinstance(r, dict):
            label = f"`{r.get('path')}`"
            if r.get("expect") is not None:
                label += f" (expect {r['expect']} rewrites)"
            return label
        return f"`{r}`"

    if ext_roots:
        # The human gate must show that this plan writes outside the
        # corpus — and any expect count assertion riding on a root.
        lines.append(
            "- External edit roots: " + ", ".join(_ext_label(r) for r in ext_roots)
        )
    scan_paths = plan.get("scan_paths") or []
    if scan_paths:
        # Phase 6: declared scan paths are writes the corpus walk never
        # sees (ignored subtrees) — the gate names them like external
        # roots so "how did .cache/ get rewritten" has an answer.
        lines.append(
            "- Scan paths (ignored dirs the rewrite pass may touch): "
            + ", ".join(f"`{p}`" for p in scan_paths)
        )
    if meta:
        # corpus_fingerprint is canonical; inventory_sha256 is the
        # pre-T1 spelling still present in plans in the wild.
        fp = meta.get("corpus_fingerprint") or meta.get("inventory_sha256") or "?"
        lines.append(
            f"- Based on inventory: `{fp[:16]}` ({meta.get('inventory_at', '?')})"
        )
    lines.append(
        f"- Moves: {len(moves)} · Folders: {len(plan.get('folders', []))} "
        f"· Unresolved: {len(plan.get('unresolved', []))}"
    )
    for r in plan.get("rules") or []:
        # Phase 8: the rule itself is summarized; the concrete moves it
        # expanded into are listed in the tables below — the human
        # approves real paths, never a pattern. B3/B4/B5: matched vs
        # expanded (explicit/keep coverage), superseded (lost to an
        # earlier rule), shadowed (keep wins) and flattened
        # (subdirectory paths lost) are all visible.
        expanded = r.get("expanded")
        parts = []
        if isinstance(r.get("matched"), int) and r["matched"] != expanded:
            parts.append(f"matched {r['matched']}")
        if r.get("superseded"):
            parts.append(f"{r['superseded']} superseded by an earlier rule")
        if r.get("shadowed"):
            parts.append(f"{r['shadowed']} inside keep/unresolved")
        count = ""
        if isinstance(expanded, int):
            count = f" — expanded to {expanded} move(s)"
        if parts:
            count += " (" + ", ".join(parts) + ")"
        if r.get("preserve") == "tree":
            count += ", preserve: tree"
        if r.get("rewrite") is False:
            count += ", rewrite: off"
        lines.append(f"- Rule: `{r.get('match')}` → `{r.get('to')}/`{count}")
        if r.get("flattened"):
            lines.append(
                f"  - ⚠️ flattens {r['flattened']} file(s) from subdirectories "
                "of the match — their subdir paths are lost in the destination "
                '(use "preserve": "tree" on the rule to keep them)'
            )
    lines.append("")
    if plan.get("partial"):
        lines.append("- **Scope: PARTIAL** — delta plan; only the listed files move.")
        lines.append("")
    skips = plan.get("skip", [])
    if skips:
        lines.append(
            "- **Skipped (never examined)**: " + ", ".join(f"`{s}`" for s in skips)
        )
        lines.append("")
    for w in plan.get("rule_warnings") or []:
        lines.append(f"- ⚠️ {w}")
    if plan.get("rule_warnings"):
        lines.append("")
    prefix_rw = plan.get("prefix_rewrites") or []
    if prefix_rw:
        lines.append(
            "- Prefix rewrites (declared renames applied by the citation "
            "rewrite pass): "
            + ", ".join(
                f"`{e['from']}` → `{e['to']}/`"
                + (" ⚠️ acknowledged broken" if e.get("acknowledge_broken") else "")
                for e in prefix_rw
            )
        )
        lines.append("")

    # Skimmable summary: where things go, one line per folder.
    if by_folder:
        lines.append("## Summary")
        lines.append("")
        lines.append("| destination | files |")
        lines.append("|---|---|")
        for folder in sorted(by_folder):
            lines.append(f"| `{folder}/` | {len(by_folder[folder])} |")
        lines.append("")

    refs = reference_report or {}
    rows_shown = 0
    truncated = False
    for folder in sorted(by_folder):
        if rows_shown >= _MOVE_TABLE_CAP:
            truncated = True
            break  # never an empty table for a starved folder
        lines.append(f"## → `{folder}/`")
        lines.append("")
        lines.append("| from | final | refs | reason |")
        lines.append("|---|---|---|---|")
        for m in sorted(by_folder[folder], key=lambda x: x["from"]):
            if rows_shown >= _MOVE_TABLE_CAP:
                truncated = True
                break
            rows_shown += 1
            # ref_report is keyed by move SOURCE PATH (cli.py cmd_plan_render);
            # fall back to basename for hand-built basename-keyed reports.
            # Looking up by basename alone hid the risk for every nested
            # source (notes/x.md) — the gate showed "—" exactly when the
            # move most needed a warning.
            base = Path(m["from"]).name
            r = refs.get(m["from"]) or refs.get(base, {})
            # Enriched rows (B2) carry uncovered_references — citations
            # the plan does not fix. A covered file is not a risk, so
            # the cell counts only those when the field is present.
            if "uncovered_references" in r:
                n = r["uncovered_references"]
            else:
                n = r.get("external_references", 0)
            ref_cell = f"⚠️ {n}" if n else "—"
            reason = (m.get("reason") or "").replace("|", "\\|")
            if m.get("rewrite") is False:
                reason = (reason + " [rewrite: off]").strip()
            lines.append(f"| `{m['from']}` | `{m['final']}` | {ref_cell} | {reason} |")
        lines.append("")
    if truncated:
        lines.append(
            f"(showing the first {rows_shown} of {len(moves)} moves — the "
            "Summary table above is complete; full list in the validated "
            "plan JSON: `plan validate --out`)"
        )
        lines.append("")

    if stragglers:
        lines.append("## ⚠️ Stragglers — files left behind in folders this plan empties")
        lines.append("")
        lines.append("| folder | left | examples |")
        lines.append("|---|---|---|")
        for i, s in enumerate(stragglers):
            if i >= _STRAGGLER_DIRS:
                break
            examples = s["remaining"][:_STRAGGLER_EXAMPLES]
            cell = ", ".join(f"`{x}`" for x in examples)
            more = s["count"] - len(examples)
            if more > 0:
                cell += f" …and {more} more"
            lines.append(f"| `{s['dir']}/` | {s['count']} | {cell} |")
        hidden = len(stragglers) - _STRAGGLER_DIRS
        if hidden > 0:
            lines.append(f"| …and {hidden} more folder(s) with leftovers | | |")
        lines.append("")
        lines.append(
            "Advisory: move them, keep them, or list them as unresolved — "
            "full list in `plan validate --out`."
        )
        lines.append("")

    if emptied_dirs:
        # B8: directories emptied file by file. Convention strings
        # naming them are not rewritten unless the plan declares a
        # prefix_rewrites entry — the evidence for that decision is
        # shown here, with a ready-to-paste suggestion when one
        # destination takes a clear majority.
        lines.append("## 📁 Directories emptied file by file")
        lines.append("")
        lines.append(
            "| directory | files moved | destinations | suggested prefix rewrite |"
        )
        lines.append("|---|---|---|---|")
        for e in emptied_dirs:
            dests = ", ".join(f"`{d}/` ({n})" for d, n in e.get("destinations", []))
            sug = e.get("suggestion") or {}
            if sug:
                sug_cell = f"`from: {sug['from']}`, `to: {sug['to']}`"
            else:
                sug_cell = "— (no majority — decide by hand)"
            lines.append(
                f"| `{e['dir']}/` | {e.get('moved_files', 0)} | {dests} | {sug_cell} |"
            )
        lines.append("")
        lines.append(
            "Convention strings naming these directories are NOT rewritten "
            "unless the plan declares `prefix_rewrites` — add it (or hand "
            "`edits`) before approving, or citations of the old prefix will "
            "dangle."
        )
        lines.append("")

    # D12: when the plan opts into removal, the gate lists exactly what
    # will be removed — before approval, never after the fact.
    remove_dirs = plan.get("remove_emptied_dirs") or []
    if plan.get("remove_emptied"):
        lines.append("## 🗑️ Directories this plan will remove (remove_emptied)")
        lines.append("")
        if remove_dirs:
            for d in remove_dirs:
                lines.append(f"- `{d}/` — emptied by the moves above")
        else:
            lines.append("- (none qualify — nothing will be removed)")
        lines.append("")
        lines.append(
            "Removal is journaled (`rmdir` ops) and undoable; a directory "
            "that still holds any file is never removed. A partial run "
            "removes nothing."
        )
        lines.append("")

    if code_anchor_risks:
        lines.append("## 🧯 Code-anchored moves — directories built in code")
        lines.append("")
        lines.append("| move | anchored dirs | evidence |")
        lines.append("|---|---|---|")
        for r in code_anchor_risks:
            dirs_cell = ", ".join(f"`{d}`" for d in r["dirs"])
            lines.append(
                f"| `{r['from']}` → `{r['to']}` | {dirs_cell} | {r['sites']} site(s) |"
            )
        lines.append("")
        lines.append(
            "Informational: join() literals are fixed by the rewrite pass, "
            "but `__file__` math is computed, not written — moving these "
            "dirs costs code changes. Evidence: `refs` or doctor "
            "code_anchors."
        )
        lines.append("")

    if rewrite_summary is not None:
        if rewrite_summary.get("mode") == "off":
            lines.append(
                "## 🔁 Citation rewrites: OFF — citations of moved paths "
                "are NOT fixed by this apply (plan sets rewrites mode off)."
            )
            lines.append("")
        else:
            per_file = rewrite_summary.get("per_file") or []
            total = rewrite_summary.get("count", 0)
            files = len(per_file)
            lines.append(
                f"## 🔁 Citation rewrites (automatic): {total} across {files} file(s)"
            )
            lines.append("")
            if per_file:
                lines.append("| file | rewrites |")
                lines.append("|---|---|")
                for row in per_file:
                    file_cell = f"`{row['file']}`"
                    if Path(str(row["file"])).is_absolute():
                        # External rewrite rows mark the write like
                        # external edits do: outside the corpus.
                        file_cell += " ⚠️ external"
                    lines.append(f"| {file_cell} | {row['count']} |")
                lines.append("")
            # B1: the actual edits, site by site — the gate approves
            # real changes, not a count.
            detail_rows = []
            for row in per_file:
                for rep in row.get("lines") or []:
                    detail_rows.append(
                        (f"`{row['file']}`:{rep['line']}", rep["old"], rep["new"])
                    )
            if detail_rows:
                lines.append("Every rewrite, site by site:")
                lines.append("")
                lines.append("| site | old | new |")
                lines.append("|---|---|---|")
                for i, (site, old, new) in enumerate(detail_rows):
                    if i >= _REWRITE_ROW_CAP:
                        break
                    lines.append(f"| {site} | `{_cell(old)}` | `{_cell(new)}` |")
                if len(detail_rows) > _REWRITE_ROW_CAP:
                    lines.append(
                        f"…and {len(detail_rows) - _REWRITE_ROW_CAP} more "
                        "(full list: `apply --rewrites-out rewrites.json`)"
                    )
                lines.append("")
            skipped = rewrite_summary.get("skipped") or []
            if skipped:
                lines.append("Spans left untouched (ambiguous — verify by hand):")
                lines.append("")
                lines.append("| file | line | reason | token | text |")
                lines.append("|---|---|---|---|---|")
                for s in skipped:
                    lines.append(
                        f"| `{s.get('file', '')}` | {s.get('line', '')} "
                        f"| {s.get('reason', '')} | `{_cell(s.get('token') or '')}` "
                        f"| `{_cell(s.get('text') or '')}` |"
                    )
                lines.append("")
            # C2: the single bad lines — a skipped span that cites a
            # MOVING path. The agent hand-edits exactly these.
            skip_rewrites = rewrite_summary.get("skip_rewrites") or []
            if skip_rewrites:
                lines.append(
                    "⚠️ **Skipped citations of moving paths** — the pass "
                    "refused to guess; hand-edit these lines:"
                )
                lines.append("")
                lines.append("| file | line | token | reason |")
                lines.append("|---|---|---|---|")
                for s in skip_rewrites:
                    lines.append(
                        f"| `{s.get('file', '')}` | {s.get('line', '')} "
                        f"| `{_cell(s.get('token') or '')}` "
                        f"| {s.get('reason', '')} |"
                    )
                lines.append("")
            # False-negative guard: a move the report calls cited but the
            # pass fixes zero citations for is human-visible, WITH the
            # citing lines and the reason each was not rewritten (B10 —
            # no more unexplained "cited but 0 rewrites").
            targets = rewrite_summary.get("targets") or {}
            edit_files = {
                str(e.get("file", ""))
                for e in plan.get("edits", [])
                if not Path(str(e.get("file", ""))).is_absolute()
            }
            refs = reference_report or {}
            unrewritten = [
                src
                for src, risk in refs.items()
                if risk.get("external_references", 0) > 0
                and not any(
                    t == src or t.startswith(src.rstrip("/") + "/") for t in targets
                )
                and any(c not in edit_files for c in risk.get("referenced_by", []))
            ]
            if unrewritten:
                lines.append(
                    "⚠️ **Cited but 0 rewrites** — the citing lines and why "
                    "the pass leaves each one (fix by hand with edits):"
                )
                lines.append("")
                # D5: the same citation can be a risk of several moves
                # (one cite per moving member) — the hand-edit list
                # shows each (file, line, token) once, under its first
                # src.
                shown_citations: set[tuple] = set()
                for src in sorted(unrewritten):
                    risk = refs.get(src) or {}
                    printable = []
                    had_uncovered = False
                    for c in risk.get("citations") or []:
                        if c.get("status") in _REWRITE_COVERED_STATUSES:
                            continue
                        had_uncovered = True
                        key = (c.get("file"), c.get("line"), c.get("token"))
                        if key in shown_citations:
                            continue
                        shown_citations.add(key)
                        printable.append(c)
                    if not printable:
                        if not had_uncovered:
                            for citer in risk.get("referenced_by", [])[:5]:
                                lines.append(f"- `{src}`")
                                lines.append(f"  - `{citer}` — no rewrite (see refs)")
                        # all-uncovered-but-duplicate: already shown
                        # under its first src
                        continue
                    lines.append(f"- `{src}`")
                    for c in printable[:5]:
                        why = c.get("why") or c.get("status") or "not rewritten"
                        loc = (
                            f"{c.get('file')}:{c.get('line')}"
                            if c.get("line")
                            else str(c.get("file", "?"))
                        )
                        lines.append(f"  - `{loc}` — {why}")
                lines.append("")

    unresolved = plan.get("unresolved", [])
    if unresolved:
        lines.append("## ⚠️ Unresolved (left in place)")
        lines.append("")
        for u in unresolved:
            lines.append(f"- `{u.get('path')}` — {u.get('reason', '')}")
        lines.append("")

    edits = plan.get("edits", [])
    if edits:
        lines.append("## ✏️ Reference rewrites (applied after moves)")
        lines.append("")
        lines.append("| file | find | replace | reason |")
        lines.append("|---|---|---|---|")
        # Name the same path the dry run's would_edit rows name (D8):
        # the post-move location resolved through the move chains, with
        # the authored source path noted when it differs.
        chains = resolve_chains(plan.get("moves") or [])
        for e in edits:
            authored = str(e.get("file", ""))
            final = _post_move_path(authored, chains)
            find = str(e.get("find", "")).replace("|", "\\|")
            repl = str(e.get("replace", "")).replace("|", "\\|")
            reason = str(e.get("reason", "")).replace("|", "\\|")
            file_cell = f"`{final}`"
            if Path(authored).is_absolute():
                file_cell += " ⚠️ external"
            elif final != authored:
                file_cell += f" (authored: `{authored}`)"
            lines.append(f"| {file_cell} | `{find[:40]}` | `{repl[:60]}` | {reason} |")
        lines.append("")

    folders = plan.get("folders", [])
    if folders:
        lines.append("## Folders to scaffold")
        lines.append("")
        lines.append("| path | kind | role |")
        lines.append("|---|---|---|")
        for f in folders:
            role = str(f.get("role", "")).replace("|", "\\|")
            lines.append(f"| `{f['path']}` | {f['kind']} | {role} |")
        lines.append("")

    lines.append(
        footer
        or (
            # T6: the closing hint is a next.command — runnable as printed,
            # with apply's required <root> from the plan's own corpus_root.
            "_Approve by editing this plan's JSON, then: "
            f"`slopkeeper apply <plan.json> "
            f"{plan.get('corpus_root', '<root>')} --execute`_"
        )
    )
    return "\n".join(lines) + "\n"


# --- §5.6 approval provenance: rendered gate → executed-plan binding ---------

RENDER_SCHEMA = "slopkeeper/render@1"
RENDER_SIDECAR = "last-render.json"


def render_sidecar_path(root: Path) -> Path:
    return state_dir(root) / RENDER_SIDECAR


def write_render_sidecar(
    plan_path: Path, root: Path, artifacts: list[str] | None = None
) -> Path:
    """Record which plan bytes were rendered, for a later apply to check.

    Written by `plan render --out`: the human approves plan.md, and this
    sidecar remembers the hash of the plan JSON that produced it. A
    stdout render shows the same document but records nothing — there is
    no rendered file the gate was shown from, so no binding to make.

    artifacts (B1): the corpus-relative paths of THIS render's own
    outputs (the gate, the rewrites JSON) when they sit inside the
    corpus — apply excludes them from the rewrite scan, so a gate left
    in the tree is never rewritten by the apply it gates.
    """
    sidecar = render_sidecar_path(root)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "schema": RENDER_SCHEMA,
        "plan_sha256": sha256_file_exact(plan_path),
        "rendered_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    if artifacts:
        doc["artifacts"] = sorted(set(artifacts))
    sidecar.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return sidecar


def load_render_sidecar(root: Path) -> dict | None:
    """The last render's record, or None when absent/corrupt/foreign.

    Advisory by design (§5.6: warn, never refuse): no sidecar means
    nothing to compare; a corrupt or foreign-schema one must never block
    an apply, so it reads exactly like "no last render".
    """
    p = render_sidecar_path(root)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("schema") != RENDER_SCHEMA:
        return None
    return data


PLAN_TEMPLATE = {
    "schema": PLAN_SCHEMA,
    "corpus_root": "/absolute/path/to/corpus",
    "strategy": "role-bots|topics|projects|hybrid",
    "based_on": {
        "corpus_fingerprint": "<from inventory.fingerprint>",
        "inventory_at": "<iso timestamp>",
    },
    "moves": [
        {
            "from": "loose_file.md",
            "to": "folder/loose_file.md",
            "reason": "one line, shown to the human",
        }
    ],
    # Empty, not an example: a template rule would fire plan_bad_rules
    # (matched no visible files) on every corpus. The shape is documented
    # in AGENT_GUIDE.md ("Rule-based moves").
    "rules": [],
    # Empty, not an example: a prefix rewrite renames a directory the
    # plan empties file by file (B8). Documented in AGENT_GUIDE.md
    # ("Rule-based moves"). An entry may carry "acknowledge_broken":
    # true to consciously accept a glob/brace citation the rename
    # cannot fix coherently (validate's prefix_rewrite_dead_glob gate).
    "prefix_rewrites": [],
    # D12: opt in to journaled, undoable removal of directories the
    # plan empties file by file. Absent/false = advisory suggestions
    # only; the gate lists exactly what would be removed.
    "remove_emptied": False,
    "edits": [
        {
            "file": "stay_put_file.md",
            "find": "loose_file.md",
            "replace": "folder/loose_file.md",
            "reason": "rewrite reference to the moved file",
        }
    ],
    "rewrites": {"mode": "auto"},
    "external_roots": [],
    "scan_paths": [],
    "skip": [],
    "keep": ["<name>", {"path": "<name>", "reason": "<why it stays>"}],
    "unresolved": [{"path": "ledger_wb.html", "reason": "no signal — human decides"}],
    "folders": [
        {
            "path": "research/dossiers",
            "kind": "case",
            "role": "Market research dossiers.",
            "boundaries": ["finished dossiers move to archive/<year-month>"],
            "state_md": False,
        }
    ],
    "roster": {"root_agents_md": "optional: full text for the root AGENTS.md roster"},
}
