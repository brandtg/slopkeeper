# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Guard: every code raised in src has a row in AGENT_GUIDE.md.

DESIGN §4.1 — "new code plus a row in the guide, or it doesn't ship."
This test makes that rule enforceable instead of aspirational: it scans
src/ for `SlopKeeperError(` first-argument string literals (stdlib re, no
deps) and asserts each appears in the first column of AGENT_GUIDE.md's
"Error codes you will see" table. Deleting a row fails the suite; a new
code without a row fails the suite.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "src"
GUIDE = REPO / "AGENT_GUIDE.md"
SECTION_HEADER = "## Error codes you will see"

# `SlopKeeperError("code", ...)` — first arg is always a bare literal.
_SLOPKEEPER_ERROR = re.compile(r"""SlopKeeperError\(\s*(['"])([^'"]*)\1""")
# First cell of a markdown table row.
_TABLE_ROW = re.compile(r"^\|([^|]+)\|")
_BACKTICK = re.compile(r"`([^`]+)`")
# A backticked token qualifies as a code only if it is bare snake_case
# (excludes markers like `status --all` that share the cell).
_CODE_LIKE = re.compile(r"^[a-z][a-z0-9_]*$")

# Documented but never raised via SlopKeeperError: data markers, not
# raised errors.
# - root_missing: a data marker in `status --all` rows.
# - rewrite_skipped: a data row in the rewrite summary (rewrites.skipped) —
#   Phase 6; the line was ambiguous, so the pass left it for hand edits.
TABLE_ONLY_CODES = {"root_missing", "rewrite_skipped"}


def raised_codes() -> set[str]:
    codes = set()
    for path in sorted(SRC_DIR.rglob("*.py")):
        for match in _SLOPKEEPER_ERROR.finditer(path.read_text(encoding="utf-8")):
            codes.add(match.group(2))
    return codes


def documented_codes() -> set[str]:
    text = GUIDE.read_text(encoding="utf-8")
    section = text.split(SECTION_HEADER, 1)[1].split("\n## ", 1)[0]
    codes = set()
    for line in section.splitlines():
        row = _TABLE_ROW.match(line)
        if row is None:
            continue
        for token in _BACKTICK.findall(row.group(1)):
            if _CODE_LIKE.match(token):
                codes.add(token)
    return codes


class TestErrorCodeContract(unittest.TestCase):
    def test_every_raised_code_is_documented(self):
        missing = raised_codes() - documented_codes()
        self.assertEqual(
            missing,
            set(),
            "codes raised in src/ but missing a row in AGENT_GUIDE.md's "
            "'Error codes you will see' table (DESIGN §4.1): "
            + ", ".join(sorted(missing)),
        )

    def test_every_documented_code_is_raised(self):
        ghost = documented_codes() - raised_codes() - TABLE_ONLY_CODES
        self.assertEqual(
            ghost,
            set(),
            "rows in AGENT_GUIDE.md's table that no src/ code raises — "
            "stale rows, delete them (or restore the code): "
            + ", ".join(sorted(ghost)),
        )


if __name__ == "__main__":
    unittest.main()
