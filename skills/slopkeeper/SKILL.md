---
# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

name: slopkeeper
description: "Use when the user asks to organize, clean up, or tend a messy directory/corpus (slopkeeper). Drive the `slopkeeper` CLI: you classify, the human approves the rendered plan, slopkeeper moves."
---

# SlopKeeper — agent-slop corpus tending

slopkeeper is installed as the `slopkeeper` command (pipx or pip). Your surface is
its JSON subcommands — **not** the interactive REPL (bare `slopkeeper` is the
human's front door).

Get the full protocol — inventory → plan → render → apply → scaffold,
with every error code and recovery hint — by running:

    slopkeeper guide

…and follow it exactly. The non-negotiables, so you never get them wrong
even before you've read the guide:

1. **No plan, no moves.** Every write goes through a validated plan JSON.
   There is no other write path.
2. **Dry-run is the default.** `apply`, `scaffold`, and `undo` touch
   nothing without `--execute`.
3. **The human gate.** `slopkeeper plan render` markdown IS the approval
   document — show it to the human and wait. Never execute a plan the
   human hasn't seen rendered. Re-validate after any edit to the JSON.
4. **Unresolved is honest.** Files you can't classify go in `unresolved`
   with a reason — never guess.
5. **Corpus changed?** On `corpus_changed`, re-inventory and regenerate
   the plan. `--force` only if the changes were yours and expected.

This file is deliberately thin. `slopkeeper guide` is the canonical protocol,
shipped inside the package — the skill never drifts from the binary's
version of it.
