# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The verify loop (P6, Slice C): prove the corpus still works after the tend.

Why this exists: everything SlopKeeper guaranteed before 0.3.0 was
TOPOLOGICAL. Moves were all-or-nothing, citations were rewritten, the
fingerprint bound the plan to the corpus, and undo could put the bytes
back. None of it answered the only question that matters when the corpus
is a codebase or a working directory of scripts: *does it still run?* A
move that breaks an import, a `package.json` path, or a Makefile target
leaves a perfectly journaled, perfectly undone-shaped corpus that no
longer builds.

So a plan (and any folder it declares) may carry shell commands, and
`apply --execute` runs them AFTER the moves, rewrites and edits:

  - every command runs with cwd at the corpus root, or at the folder's
    own directory for a folder-level command;
  - stdout/stderr tails are journaled as `verify` ops, so the recovery
    record says what the corpus looked like, not just where the files were;
  - the first non-zero exit stops the run, and — unless `--no-auto-undo`
    — the whole tend is rolled back through the journal, so a failed check
    leaves the tree byte-identical to before the apply;
  - the failure surfaces as `verify_failed` (exit 2) with the captured
    output in the envelope's `data`, because "the check failed" without
    the check's own words is an error the agent cannot act on.

Two decisions worth stating plainly:

  * The command runs in a SHELL (`shell=True`), because that is what a
    verify line in a plan looks like in the wild — `pytest -q`,
    `npm test && tsc --noEmit`. That makes a verify command arbitrary code
    execution, which is exactly why the human gate prints every one of
    them verbatim (`## 🧪 Verification`) BEFORE approve, and why they run
    only under `--execute`: a dry run reports `would_verify` and touches
    nothing. The plan is untrusted input; the gate is the trust boundary.
  * A command whose folder does not exist yet is SKIPPED with a loud
    warning, not failed. Folders are scaffolded after apply, so a
    folder-level check on a brand-new folder could never run — refusing
    the tend for that would make the feature unusable on the first tend of
    a new layout, and silently passing would be a lie. The warning names
    it: `verify_skipped_missing_cwd`.

Zero dependencies beyond stdlib; no network; nothing here knows about
LLMs. The tool runs the check the human approved — deciding WHAT to check
stays with the agent and the owner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .models import SlopKeeperError

# The defaults for a verify command. `timeout` bounds the whole tend:
# apply holds the corpus's tend lock while these run, so an unbounded
# check is an unbounded lock. A command that needs longer says so per
# entry; nothing runs past the hard ceiling.
DEFAULT_VERIFY_TIMEOUT = 600
MAX_VERIFY_TIMEOUT = 3600
# How much of each stream is kept — in the journal, the payload and the
# error alike. The TAIL is the useful part: a pytest failure ends with the
# summary and the failing assertions, and a build error ends with the
# error. The head of a 10k-line log is noise.
VERIFY_TAIL_CHARS = 4000

# The keys a verify entry may carry — strictly, like `rules` and `dedup`:
# a typo'd key that validate let through is a check that silently never
# runs, which is the one outcome a verification feature cannot afford.
# `folder` is the odd one: it is AUTHORABLE-nowhere (the tool stamps it
# from the folder entry that owns the check) but it is ACCEPTED so a
# `plan validate --out` artifact, which carries the normalized form, can
# be fed straight back to validate without tripping the key check.
VERIFY_KEYS = {"command", "cwd", "name", "timeout", "folder"}


def _bad(message: str, recovery: str = "") -> SlopKeeperError:
    return SlopKeeperError(
        "plan_bad_verify",
        message,
        recovery
        or (
            "A verify entry is a command string, or an object with "
            '"command" (required), "cwd" (corpus-relative directory), '
            '"name" (label) and "timeout" (seconds, <= %d). Run '
            "`slopkeeper plan schema` for the shape." % MAX_VERIFY_TIMEOUT
        ),
    )


def _one(raw: object, root: Path, where: str, folder: str | None) -> dict | None:
    """Normalize one verify entry, or None when it carries nothing to run."""
    if isinstance(raw, str):
        command, cwd, name, timeout = raw, None, None, None
    elif isinstance(raw, dict):
        extra = set(raw) - VERIFY_KEYS
        if extra:
            raise _bad(
                f"{where}: unknown key(s) {sorted(extra)} in a verify entry",
                "Remove them; a verify entry carries command, cwd, name, "
                "timeout (and folder, which the tool stamps).",
            )
        command = raw.get("command")
        cwd = raw.get("cwd")
        name = raw.get("name")
        timeout = raw.get("timeout")
    else:
        raise _bad(f"{where}: a verify entry must be a string or an object")
    if command is None:
        return None  # an entry with no command is nothing to run
    if not isinstance(command, str) or not command.strip():
        raise _bad(f"{where}: verify command must be a non-empty string")
    out: dict = {"command": command.strip(), "name": None, "cwd": None, "timeout": None}
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise _bad(f"{where}: verify name must be a non-empty string")
        out["name"] = name.strip()
    else:
        out["name"] = out["command"][:80]
    if cwd is not None:
        if not isinstance(cwd, str) or not cwd.strip():
            raise _bad(f"{where}: verify cwd must be a corpus-relative directory")
        if cwd.strip().rstrip("/") in {".", "./"}:
            # "run at the corpus root" is the most common thing an author
            # means, and the most natural way to write it is ".". Treating
            # that as a malformed plan would be pedantry that pushes people
            # to omit cwd — so it means root, and an explicit "." also wins
            # over the folder default for a folder-level entry.
            out["cwd"] = None
        else:
            from .models import safe_relpath  # local: models imports nothing here

            try:
                out["cwd"] = safe_relpath(cwd, root).as_posix()
            except SlopKeeperError as e:
                # A verify cwd that walks outside the corpus is a malformed
                # PLAN, not a path-safety incident mid-tend: refusing it under
                # plan_bad_verify keeps every "your plan is wrong" answer in
                # one exit class (2) with one recovery, instead of leaking the
                # generic path_escape (exit 1) for a typo in a plan field.
                raise _bad(
                    f"{where}: verify cwd {cwd!r} is not a path inside the corpus"
                    f" ({e.message})",
                    "cwd is corpus-relative — a directory inside the corpus, "
                    "usually the folder the check belongs to.",
                ) from None
    elif folder:
        out["cwd"] = folder
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise _bad(f"{where}: verify timeout must be a positive number of seconds")
        if timeout > MAX_VERIFY_TIMEOUT:
            raise _bad(
                f"{where}: verify timeout {timeout}s exceeds the "
                f"{MAX_VERIFY_TIMEOUT}s ceiling",
                "apply holds the tend lock while checks run; keep a single "
                f"check under {MAX_VERIFY_TIMEOUT}s, or split it.",
            )
        out["timeout"] = timeout
    else:
        out["timeout"] = DEFAULT_VERIFY_TIMEOUT
    if folder:
        out["folder"] = folder
    return out


def normalize_verify(
    raw: object,
    root: Path,
    *,
    where: str = "plan.verify",
    folder: str | None = None,
) -> list[dict]:
    """The plan's (or a folder's) verify commands, in run order.

    Accepts one command string, or a list of strings and/or objects;
    returns a list of {command, name, cwd, timeout[, folder]} with
    defaults filled. A plan with no verify at all normalizes to [] — and
    stays absent from the plan, so no envelope or journal grows a key for
    a corpus that has no checks.
    """
    if raw is None or raw == [] or raw == {} or raw == "":
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    for item in items:
        entry = _one(item, root, where, folder)
        if entry is None:
            continue
        key = (entry["command"], entry["cwd"])
        if key in seen:
            continue  # the same check declared twice would run (and fail) twice
        seen.add(key)
        out.append(entry)
    return out


def _tail(text: object) -> tuple[str, bool]:
    """(tail, truncated) for a captured stream; None and bytes-tolerant."""
    if text is None:
        return "", False
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= VERIFY_TAIL_CHARS:
        return text, False
    return text[-VERIFY_TAIL_CHARS:], True


def run_verify(commands: list[dict], root: Path) -> tuple[list[dict], dict | None]:
    """Run each verify command in order; stop at the first that does not pass.

    Returns (rows, failure): rows is one dict per command —
    {name, command, cwd, timeout, status, returncode, stdout, stderr,
    duration_ms, note} where status is passed | failed | error |
    timeout | skipped_missing_cwd — and failure is the row that stopped
    the run (or None). The caller decides what a failure costs the tend;
    this function only reports what happened, and never raises on a
    command's behavior.
    """
    rows: list[dict] = []
    failure: dict | None = None
    for spec in commands:
        cwd = root / spec["cwd"] if spec.get("cwd") else root
        base = {
            "name": spec["name"],
            "command": spec["command"],
            "cwd": spec.get("cwd") or ".",
            "timeout": spec["timeout"],
        }
        if spec.get("folder"):
            base["folder"] = spec["folder"]
        if not cwd.is_dir():
            # Folders are scaffolded after apply, so a brand-new folder's
            # check has nowhere to run yet. Skipping beats failing a tend
            # that is creating the very directory the check points at —
            # and beats passing silently.
            rows.append({**base, "status": "skipped_missing_cwd", "returncode": None})
            continue
        try:
            proc = subprocess.run(
                spec["command"],
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=spec["timeout"],
                env=None,
            )
        except subprocess.TimeoutExpired as e:
            out, out_cut = _tail(e.stdout)
            err, err_cut = _tail(e.stderr)
            row = {
                **base,
                "status": "timeout",
                "returncode": None,
                "stdout": out,
                "stderr": err,
                "note": f"timed out after {spec['timeout']}s",
            }
            if out_cut or err_cut:
                row["output_truncated"] = True
            rows.append(row)
            failure = row
            break
        except OSError as e:
            row = {
                **base,
                "status": "error",
                "returncode": None,
                "stdout": "",
                "stderr": f"could not run: {e}",
            }
            rows.append(row)
            failure = row
            break
        out, out_cut = _tail(proc.stdout)
        err, err_cut = _tail(proc.stderr)
        row = {
            **base,
            "status": "passed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": out,
            "stderr": err,
        }
        if out_cut or err_cut:
            row["output_truncated"] = True
        rows.append(row)
        if proc.returncode != 0:
            failure = row
            break
    return rows, failure
