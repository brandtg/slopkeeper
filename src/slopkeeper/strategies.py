# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Partitioning strategies: rubric packs the calling agent applies.

A strategy is not code — it's a rubric the agent reads, plus folder templates
the scaffold step renders. Ships as JSON so agents can read it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import SlopKeeperError

_STRATEGY_DIR = Path(__file__).parent / "strategies"


def list_strategies() -> list[dict]:
    out = []
    for p in sorted(_STRATEGY_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": d["id"], "name": d["name"], "summary": d["summary"]})
    return out


def get_strategy(strategy_id: str) -> dict:
    p = _STRATEGY_DIR / f"{strategy_id}.json"
    if not p.exists():
        raise SlopKeeperError(
            "unknown_strategy",
            f"no strategy {strategy_id!r}",
            f"Available: {[s['id'] for s in list_strategies()]}",
        )
    return json.loads(p.read_text(encoding="utf-8"))
