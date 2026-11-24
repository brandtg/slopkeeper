<!--
SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>

SPDX-License-Identifier: MIT
-->

# SlopKeeper

Someone has to clean up after your agents. SlopKeeper does: inventory,
partition and scaffold messy file corpora — the tool side of
"each folder is a bot". A calling agent (Claude Code, Hermes, anything
that shells out) does the classification; SlopKeeper does everything else:
inventory, plan validation, reference-integrity checking, all-or-nothing
execution with a journal and undo, and the AGENTS.md/STATE.md layer that
makes the result agent-navigable.

**Zero dependencies. Python 3.11+ (oldest supported CPython) stdlib only.
No LLM calls. No network.**

## Install

Requires Python 3.11+ (what Debian 12 / RHEL 9 ship; anything newer
works). Zero dependencies, so the install is tiny.

With [pipx](https://pipx.pypa.io) (recommended — puts `slopkeeper` on your
PATH in its own venv):

```bash
pipx install slopkeeper
```

Also fine: `uv tool install slopkeeper`, or `pip install slopkeeper`
(in a virtualenv, or `pip install --user` if you prefer).

Verify:

```bash
slopkeeper --version
```

## Quickstart

SlopKeeper's whole model is *agent classifies, human approves, SlopKeeper
moves* — so the other half of installing it is making sure your agent
knows how to drive it. The canonical protocol ships **inside the
package**; the one-time setup is to install the skill and let your agent
read the guide:

```bash
slopkeeper skill install claude    # or: opencode, hermes — or a skills-dir path
slopkeeper guide                   # the full agent protocol (AGENT_GUIDE.md)
```

Each skill target is a known agent name (`claude`, `opencode`, `hermes`)
or a path to a skills directory. An already-present skill is left alone
with a warning; `--force` refreshes it. The skill is packaged inside the
wheel, so it always matches the version that installed it.

Then just ask your agent to organize a directory. It will read `slopkeeper
guide` and drive the rest; the loop looks like:

```bash
slopkeeper doctor ~/workspace                        # preflight: size, mess, strategy
slopkeeper inventory ~/workspace --fast --summary    # what's there (token-conscious)
slopkeeper draft ~/workspace                         # or: the agent authors plan.json itself
slopkeeper plan validate plan.json ~/workspace
slopkeeper plan render plan.json ~/workspace          # the human gate — read this
slopkeeper apply plan.json ~/workspace --execute     # all-or-nothing, journaled
slopkeeper scaffold plan.json ~/workspace --execute  # writes AGENTS.md/STATE.md
```

Humans: the rendered plan (`plan render`) is your approval gate — read
it, edit the JSON, or say no. If you'd rather drive by hand, run bare
`slopkeeper` for the interactive REPL (below).

## How it works

```
inventory → agent drafts plan → validate → render (HUMAN GATE)
          → apply --execute → scaffold → done (undo possible)
```

**Tending a corpus that's already organized** (the usual case after the
first sweep) is the cheap loop: `status` diffs against the last tend's
manifest, the agent triages only the new slop, and a delta plan moves
just that. See AGENT_GUIDE.md — "Re-tending".

The commands:

- **doctor** — preflight: size, messiness, strategy suggestion, token
  estimate, advice. Start here.
- **inventory** — structured JSON of the corpus: files, heads, titles,
  dir stats, date/prefix/ext clusters, and a cross-reference index of
  which files mention which filenames. Token-conscious modes for big
  corpora (`--fast --summary`, or `--out` to land the full JSON in a
  file with counts alone on stdout).
- **clusters** — similarity hints for the planning agent: TF-IDF content
  clusters, scrape pairs, byte-identical copies, hub files. Hints, never
  a plan. `--out` lands the full payload in a file, counts on stdout.
- **refs** — every site that cites a given filename; the
  grep-before-move check behind reference integrity.
- **propose** — the layout fork as data: mechanical candidate partitions
  (by-project, by-kind, hybrid), each priced with move counts,
  citation-break counts, and unresolved estimates. Read-only; the agent
  compares, then authors the real plan.
- **status** — diff since the last tend (manifest in `.slopkeeper/`):
  new/modified/removed files, folder health, still-unresolved items.
  `--all` returns one count-level digest over every registered root.
- **roots** — registry of tended corpora at
  `$XDG_CONFIG_HOME/slopkeeper/roots.json` (`SLOPKEEPER_HOME` overrides):
  `roots add/remove/list`, feeding `status --all` for scheduled tending.
- **draft** — auto-drafts a delta plan for the new slop. A starting
  point for the agent to reclassify, not a decision.
- **plan** — the only write path. A JSON document of `moves` (from → to +
  reason), `keep`, `unresolved`, `folders` (to scaffold), and the corpus
  fingerprint it was authored against — required: validate and apply
  refuse an unbound plan (`plan_no_based_on`; `plan render` warns, never
  fabricates). Validated hard: path escapes,
  stale sources, collisions, nested moves, cycles, incomplete coverage.
  `plan validate` / `plan render` / `plan schema` are the subcommands.
- **apply** — all-or-nothing execution. Dry-run by default; refuses a
  corpus whose fingerprint changed since planning, and one that churned
  under the walk (`corpus_unstable` — files that vanished or denied
  `stat` mid-scan are counted in `stats.skipped_unstatable`, never
  skipped silently); refuses moves that
  cross symlinked paths (`symlink_escape`; `--allow-symlinks` for a
  deliberate traversal); pre-flights every
  destination before touching anything and re-checks it inside the
  write itself (an atomic no-replace link/unlink — a destination created
  mid-run is refused, not clobbered); journals every move with a
  content hash, one journal per tend, and undo VERIFIES those hashes
  (a file edited after the tend comes back flagged `content_changed`);
  flags every move cited from stay-put files.
- **move** — one bulk move (src dst root): builds the one-move plan,
  runs it through the same validate → gate → journal machinery, and
  renders the human gate. Dry-run by default; `--execute` runs it.
- **scaffold** — writes the folder-bot layer: root `AGENTS.md` roster,
  per-folder `AGENTS.md` and `STATE.md`. Never overwrites — existing
  files get a `*.slopkeeper-suggested` sibling instead.
- **undo** — reverses one journal exactly (restores, prunes empty dirs).
  One journal per tend — same-second tends get `-2`, `-3` siblings, so
  undo never reverts a neighbor tend. The journaled sha256s are verified
  on restore: a file edited after the tend still comes back, flagged
  `content_changed`. Takes one journal; name it (see
  `.slopkeeper/journal-*.jsonl`).
- **strategies** — partitioning rubric packs the agent reads: `role-bots`
  (folder-as-bot), `topics`, `projects`, `by-kind` (mechanical baseline),
  `hybrid` (default).
- **guide** — prints the full agent protocol (also readable via the
  installed skill and AGENT_GUIDE.md, shipped in the wheel).
- **skill** — `skill install <agent|dir>` copies the packaged SKILL.md
  into an agent's skills directory (`claude`, `opencode`, `hermes`, or a
  path); existing files are left alone unless `--force`. The one-time
  setup that teaches your agent the protocol.

### The REPL (bare `slopkeeper`)

Run `slopkeeper` with no subcommand and you get an interactive
`/command` prompt — the human front door for exploring a corpus and
tending it by hand. It drives the exact same functions as the JSON
subcommands (no new write path); every `--execute` stops for an explicit
y/N, and the default is no.

```text
$ slopkeeper
> /use ~/Documents/notes     # point it at a corpus
> /status                    # what changed since the last tend
> /new                       # the new slop, with titles and heads
> /draft                     # auto-draft a delta plan
> /moves  /show 1  /no 1  /defer 1  /dest 1 api/   # decide per move
> /plan                      # compile decisions, validate, save
> /render                    # read the human gate
> /apply                     # dry-run → [y/N] → execute
> /scaffold                  # [y/N] → write the AGENTS.md layer
> /undo                      # list journals, dry-run, execute
```

`/load <plan.json>` brings in a plan your agent wrote for move-by-move
review. Readline history persists, and Tab completes command names,
filesystem paths, and move/journal numbers. `/help` lists everything.

First-tend authoring (naming a taxonomy from a full inventory) is agent
work — point your agent at AGENT_GUIDE.md.

## Design notes

- The hard parts of corpus organization are *judgment* (the agent's),
  *reference integrity* (grep-before-move, baked into validate/apply),
  and *the human gate* (rendered markdown of every move with reasons and
  risks). The tool makes the gate cheap; it does not try to remove it.
- Fingerprint gating means a plan can never act on a corpus it wasn't
  written against — the "silent drift" failure mode is closed.
- Journals are JSONL — one file per tend, each line fsync'd whole, every
  move carrying its sha256 — and undo VERIFIES those hashes on the way
  back: exact, reversible, `content_changed` when a file was edited after
  the tend, empty dirs it created pruned. Nothing is ever deleted.
- Strategies are JSON rubric packs, not code: the agent reads them like
  documentation, and new strategies are new files, not releases.

## For developers

Everything above is the released package. To work on SlopKeeper itself:

From a local clone (or `uv tool install .` / `python3 -m pip install .`):

```bash
git clone https://github.com/brandtg/slopkeeper && cd slopkeeper
pipx install .
```

Install straight from GitHub without cloning (pin a tag as needed):

```bash
pipx install git+https://github.com/brandtg/slopkeeper.git@v0.2.0
```

Run from a checkout without installing anything:

```bash
make run ARGS="doctor ~/workspace"
# or, without uv:
PYTHONPATH=src python3 -m slopkeeper doctor ~/workspace
```

The canonical gate — lint, tests, license headers; what CI runs:

```bash
make check
```

The agent protocol lives in AGENT_GUIDE.md — shipped in the wheel and
printable via `slopkeeper guide`. Design rationale lives in module
docstrings and comments; audit reports land under docs/.

## License

MIT.
