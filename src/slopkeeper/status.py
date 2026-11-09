# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The tending loop: remember the last tended state, diff against it.

"Tend the garden" is recurring, not one-shot. Agents keep sloppping files
into the corpus; each tending pass should cost tokens proportional to what
changed, not the whole corpus. The manifest records the file list at the
last executed apply; `status` diffs the disk against it and hands the
agent exactly the new/changed files (with heads) plus folder health.

Manifest lives at .slopkeeper/manifest.json — same dir as journals, so it
is skipped by every walk. Zero dependencies; no LLM, no network.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .ignore import corpus_walk, load_ignore, nested_corpora
from .models import (
    CURRENT_FP_PREFIX,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    SlopKeeperError,
    corpus_fingerprint,
    sha256_file,
    state_dir,
    tend_lock,
)
from .inventory import HEAD_BYTES, HEAD_CHARS, _kind  # noqa: F401 (kind reuse)

CHARS_PER_TOKEN = 4
# Phase 7 rebaseline threshold: drift past max(20% of the manifest's
# file count, this floor) means the manifest no longer describes the
# corpus — re-record instead of hand-building a zero-move plan.
DRIFT_FLOOR = 2000


def manifest_path(root: Path) -> Path:
    return state_dir(root) / MANIFEST_NAME


def _bad_unresolved(u: object) -> bool:
    """True when an unresolved entry lacks a non-empty string `path`.

    A null path (JSON null from an older writer, a hand edit) would
    crash the status walk (`root / None` TypeError) the moment it is
    trusted. Such entries are data, not crashes: they are skipped and
    counted wherever the state file is read or written.
    """
    return not (
        isinstance(u, dict) and isinstance(u.get("path"), str) and u["path"].strip()
    )


def load_manifest(root: Path) -> dict | None:
    p = manifest_path(root)
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise SlopKeeperError(
            "manifest_unreadable",
            f"manifest at {p} is unreadable: {e}",
            "Delete it and run `slopkeeper status --record` to rebuild.",
        )
    if not isinstance(m, dict) or m.get("schema") != MANIFEST_SCHEMA:
        # Foreign JSON (`echo '[]' > manifest.json`) must surface as the
        # loud schema refusal, not an internal crash on m.get().
        got = m.get("schema") if isinstance(m, dict) else type(m).__name__
        raise SlopKeeperError(
            "manifest_bad_schema",
            f"manifest schema {got!r} != {MANIFEST_SCHEMA!r}",
            "Delete the manifest and re-record.",
        )
    return m


def write_manifest(
    root: Path,
    unresolved: list | None = None,
    plan_path: str | None = None,
    plan_sha256: str | None = None,
) -> dict:
    """Snapshot the corpus. Called by `apply --execute`, `status --record`,
    scaffold --execute, and undo.

    §5.6: plan_sha256 binds the manifest to the exact plan bytes that
    were executed (None when the snapshot comes from status --record,
    scaffold, or an undo — no plan involved there).

    Carry-forward: a re-baseline snapshot — scaffold --execute or
    `status --record`, which pass no unresolved — keeps the previous
    manifest's unresolved backlog and last_plan instead of wiping them,
    so the documented apply → scaffold loop does not empty
    still_unresolved every tend. Callers that pass unresolved
    explicitly own the plan provenance block: apply passes the plan's
    fields; undo passes the surviving backlog and clears last_plan by
    omitting it. An absent or corrupt manifest carries nothing forward
    (empty), as before.

    Skip-and-count: entries without a non-empty string `path` are
    dropped from the backlog (whatever its source) and recorded as
    `skipped_bad_unresolved` — a null path stored here would TypeErrors
    every later `status` walk, so garbage never survives a write.
    """
    root = root.resolve()
    if unresolved is None:
        try:
            prev = load_manifest(root)
        except SlopKeeperError:
            prev = None  # absent or unreadable: nothing to carry forward
        if prev is not None:
            carried = prev.get("unresolved")
            if isinstance(carried, list):
                unresolved = list(carried)
            if plan_path is None:
                carried_plan = prev.get("last_plan")
                if isinstance(carried_plan, str):
                    plan_path = carried_plan
    if unresolved is None:
        unresolved = []
    if not isinstance(unresolved, list):
        unresolved, skipped_bad_unresolved = [], 1
    else:
        skipped_bad_unresolved = sum(1 for u in unresolved if _bad_unresolved(u))
        unresolved = [u for u in unresolved if not _bad_unresolved(u)]
    files: dict[str, list] = {}
    for dirpath, _dirnames, filenames in corpus_walk(root):
        dirpath = Path(dirpath)
        for fn in filenames:
            fp = dirpath / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            rel = fp.relative_to(root).as_posix()
            files[rel] = [st.st_size, st.st_mtime_ns, sha256_file(fp)]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "recorded_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": corpus_fingerprint(root),
        "file_count": len(files),
        "files": files,
        "unresolved": unresolved,
        "skipped_bad_unresolved": skipped_bad_unresolved,
        "last_plan": plan_path,
        "plan_sha256": plan_sha256,
    }
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _walk_now(root: Path) -> dict[str, tuple[int, int, str | None]]:
    """Current disk state per file: [size, mtime_ns, sha256_or_null].

    The hash slot mirrors the manifest (§5.2): null for files past
    sha256_file's 64 MiB cap, which then compare on size+mtime only.
    """
    out: dict[str, tuple[int, int, str | None]] = {}
    for dirpath, _dirnames, filenames in corpus_walk(root):
        dirpath = Path(dirpath)
        for fn in filenames:
            fp = dirpath / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            out[fp.relative_to(root).as_posix()] = (
                st.st_size,
                st.st_mtime_ns,
                sha256_file(fp),
            )
    return out


def _entry_changed(old: list, now: tuple) -> bool:
    """Manifest-vs-disk staleness for one path (§5.2).

    Entries are [size, mtime_ns, sha256_or_null]. When both sides carry
    a hash, the hash decides — so a bare `touch` is NOT modified and a
    same-size content edit IS. When either hash is null, the pre-v2
    rule applies: (size, mtime_ns) equality decides.
    """
    old_hash = old[2] if len(old) > 2 else None
    now_hash = now[2] if len(now) > 2 else None
    if old_hash is not None and now_hash is not None:
        return old_hash != now_hash
    return old[:2] != list(now[:2])


def _head_for(root: Path, rel: str) -> dict:
    """Title + head for a NEW file, so the agent can triage without opening it."""
    fp = root / rel
    kind = _kind(fp)
    if kind == "binary":
        return {"kind": kind, "title": None, "head": None}
    try:
        with fp.open("rb") as f:
            text = f.read(HEAD_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return {"kind": kind, "title": None, "head": None}
    title = None
    if kind == "doc":
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                title = s.lstrip("#").strip()[:120]
                break
    if title is None:
        for line in text.splitlines():
            s = line.strip()
            if s:
                title = s[:120]
                break
    return {"kind": kind, "title": title, "head": text[:HEAD_CHARS]}


def status(root: Path, record: bool = False) -> dict:
    """Diff the corpus against the last tend (§5.5 tending lock).

    Plain reads take NO lock — status must stay safe beside a running
    tend. Only --record (a manifest write) acquires .slopkeeper/tend.lock.
    """
    root = root.resolve()
    with tend_lock(root) if record else nullcontext():
        return _status(root, record)


def status_all(roots: list[str], record: bool = False) -> list[dict]:
    """One digest per registered root (`status --all`).

    Counts, not listings — a scheduled digest stays skimmable; full
    detail is one per-root `slopkeeper status <root>` away. Failures are
    per-root, never fatal: a deleted root or a corrupt manifest is a row
    with ok=false and an error, the other rows still report.
    """
    out: list[dict] = []
    for raw in roots:
        entry: dict = {"root": raw}
        root = Path(raw)
        if not root.is_dir():
            entry["ok"] = False
            entry["error"] = {
                "code": "root_missing",
                "message": f"{raw} is no longer a directory",
                "recovery": "Re-add its new path with `slopkeeper roots "
                "add`, or `slopkeeper roots remove` if it is gone.",
            }
            out.append(entry)
            continue
        try:
            s = status(root, record=record)
        except SlopKeeperError as e:
            entry["ok"] = False
            entry["error"] = e.as_dict()
            out.append(entry)
            continue
        except Exception as e:  # noqa: BLE001 — one poisoned root never kills the digest
            # State-file content is data, not a crash: a root whose
            # evaluation fails for any other reason is a contained row
            # (the same contract as cli's catch-all), and the rest of
            # the digest still reports.
            entry["ok"] = False
            entry["error"] = {
                "code": "internal",
                "message": f"{type(e).__name__}: {e}",
                "recovery": "Run `slopkeeper status <root>` for the full error; "
                "the other roots still reported.",
            }
            out.append(entry)
            continue
        entry.update(
            {
                "ok": True,
                "never_tended": s["never_tended"],
                "new_count": s.get("new_count", 0),
                "modified_count": len(s.get("modified", [])),
                "removed_count": len(s.get("removed", [])),
                "stale_folders": len(s.get("stale_folders", [])),
                "unresolved_count": len(s.get("still_unresolved", [])),
            }
        )
        out.append(entry)
    return out


def _status(root: Path, record: bool = False) -> dict:
    root = root.resolve()
    if record:
        # --record is the documented recovery for a corrupt manifest
        # (manifest_unreadable's own hint says "delete it and run
        # `slopkeeper status --record` to rebuild"), so an unreadable manifest
        # must not block the rebuild it promises. A wrong-schema manifest
        # still refuses loudly: re-recording over the schema gate would
        # hide a version mismatch. write_manifest carries unresolved and
        # last_plan forward from the readable manifest.
        try:
            load_manifest(root)
        except SlopKeeperError as e:
            if e.code != "manifest_unreadable":
                raise
        manifest = write_manifest(root)
    else:
        manifest = load_manifest(root)

    now = _walk_now(root)
    if manifest is None:
        return {
            "never_tended": True,
            "file_count": len(now),
            "nested_corpora": nested_corpora(root),
            "advice": [
                "No manifest: either this corpus was never organized, or "
                "tending happened without SlopKeeper.",
                "Run `slopkeeper doctor`, then a full inventory, plan, apply. "
                "The manifest is written automatically on `apply --execute`.",
            ],
        }

    old: dict = manifest.get("files", {})
    old_keys = set(old)
    now_keys = set(now)

    added = sorted(now_keys - old_keys)
    removed = sorted(old_keys - now_keys)
    modified = sorted(k for k in old_keys & now_keys if _entry_changed(old[k], now[k]))

    # §6.6 honesty: a file that was tended and is now ignored neither
    # vanished nor changed — the tool stopped looking at it. Report it
    # under removed_now_ignored instead of crying `removed`.
    ignore = load_ignore(root)
    removed_now_ignored = sorted(
        k for k in removed if ignore.is_ignored(k, (root / k).is_dir())
    )
    if removed_now_ignored:
        removed = [k for k in removed if k not in set(removed_now_ignored)]

    # Phase 7 honesty, same shape: a subtree that became its own corpus
    # (its own .slopkeeper/manifest.json) left the parent's walk — its
    # files are neither vanished nor broken, they are another corpus's
    # business now. Reported under removed_now_nested and named in
    # nested_corpora, never a fake removal storm.
    nested = nested_corpora(root)

    def _under_nested(rel: str) -> bool:
        return any(rel == n or rel.startswith(n + "/") for n in nested)

    removed_now_nested = sorted(k for k in removed if _under_nested(k))
    if removed_now_nested:
        removed = [k for k in removed if k not in set(removed_now_nested)]

    new_files = []
    est_chars = 0
    for rel in added[:2000]:
        h = _head_for(root, rel)
        new_files.append({"path": rel, **h})
        est_chars += 180 + len(str(h.get("head") or ""))

    # Folder health: AGENTS.md presence + STATE.md staleness.
    stale_folders = []
    agents_missing = []
    top_dirs = sorted({k.split("/")[0] for k in now_keys if "/" in k})
    for d in top_dirs:
        dp = root / d
        if not dp.is_dir():
            continue
        if not (dp / "AGENTS.md").exists():
            agents_missing.append(d)
        state = dp / "STATE.md"
        if state.exists():
            state_mtime = state.stat().st_mtime
            newest = max(
                (
                    now[k][1] / 1e9
                    for k in now_keys
                    if k.startswith(d + "/") and k != f"{d}/STATE.md"
                ),
                default=0.0,
            )
            if newest > state_mtime:
                stale_folders.append(
                    {
                        "folder": d,
                        "state_older_than_newest_file": True,
                        "newest_file_mtime": datetime.fromtimestamp(
                            newest, tz=timezone.utc
                        ).strftime("%Y-%m-%d"),
                    }
                )

    # Unresolved files from the last plan that still exist. Entries
    # with a null/missing path (state written before the plan gate, or
    # a hand edit) are skipped and counted — never a root / None crash.
    raw_unresolved = manifest.get("unresolved", [])
    skipped_bad_unresolved = 0
    if not isinstance(raw_unresolved, list):
        raw_unresolved, skipped_bad_unresolved = [], 1
    still_unresolved = []
    for u in raw_unresolved:
        if _bad_unresolved(u):
            skipped_bad_unresolved += 1
        elif (root / u["path"]).exists():
            still_unresolved.append(u)

    # Walk-contract staleness (fp4): a manifest recorded before a walk-set
    # change compares apples to oranges — its file set was collected under
    # different skip rules, so its diff is silently wrong. Name the problem
    # and the one-command fix instead of serving a misleading diff.
    fp_then = str(manifest.get("fingerprint") or "")
    fp_then_prefix = fp_then.split(":", 1)[0]
    walk_contract_stale = (
        fp_then_prefix.startswith("fp") and fp_then_prefix != CURRENT_FP_PREFIX
    )

    out = {
        "never_tended": False,
        "last_tended": manifest.get("recorded_at"),
        "walk_contract_stale": walk_contract_stale,
        "files_then": manifest.get("file_count"),
        "files_now": len(now),
        "new_files": new_files,
        "new_count": len(added),
        "modified": modified,
        "removed": removed,
        "removed_now_ignored": removed_now_ignored,
        "removed_now_nested": removed_now_nested,
        "nested_corpora": nested,
        "estimated_triage_tokens": est_chars // CHARS_PER_TOKEN,
        "fingerprint_now": corpus_fingerprint(root),
        "fingerprint_then": manifest.get("fingerprint"),
        "folders_missing_agents_md": agents_missing,
        "stale_folders": stale_folders,
        "still_unresolved": still_unresolved,
        "skipped_bad_unresolved": skipped_bad_unresolved,
    }
    if walk_contract_stale:
        out["advice"] = [
            f"Manifest was recorded under an older walk contract "
            f"({fp_then_prefix}); dot-directories became content in "
            f"{CURRENT_FP_PREFIX}, so this diff is missing them. Re-record "
            f"with `slopkeeper status --record`."
        ]
    # Phase 7 rebaseline: heavy drift means the manifest no longer
    # describes reality (an out-of-band change, a sync, a bulk import).
    # The agent's move is ONE command, not a hand-built zero-move plan.
    drift = len(added) + len(modified) + len(removed)
    baseline = manifest.get("file_count") or 0
    threshold = max(int(baseline) * 20 // 100, DRIFT_FLOOR)
    if drift > threshold:
        out["heavy_drift"] = {
            "changed": drift,
            "threshold": threshold,
            "new": len(added),
            "modified": len(modified),
            "removed": len(removed),
        }
        out.setdefault("advice", []).append(
            f"Heavy drift: {drift} files changed since the last tend "
            f"(threshold {threshold}). The manifest no longer describes "
            f"the corpus — rebaseline with `slopkeeper status --record` "
            f"instead of hand-building a zero-move plan."
        )
    return out
