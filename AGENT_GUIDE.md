<!--
SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>

SPDX-License-Identifier: MIT
-->

# AGENT_GUIDE — how to drive SlopKeeper

You are the calling agent. The user said something like "organize my
directory" or "clean up my workspace". This tool is your hands and your
seatbelt: it inventories, validates, moves and scaffolds. You classify.
The human approves. **Never generate or execute a plan without showing the
rendered markdown to the human first.**

(Humans can also drive slopkeeper themselves: bare `slopkeeper` with no subcommand
starts an interactive `/command` REPL. That is not your surface — drive
the JSON subcommands documented here; they are scriptable and stable.)

## Hard rules

1. **No plan, no moves.** Every file operation goes through a validated
   plan JSON. There is no other write path.
2. **Dry-run is the default.** `apply` and `scaffold` and `undo` do nothing
   destructive without `--execute`.
3. **The human gate.** Render the plan (`plan render`) and present it.
   The human edits the plan JSON or approves. Re-validate after any edit.
4. **Respect reference risks.** Every apply runs the citation rewrite
   pass by default (see "Reference rewrites"): citations of every moved
   path are rewritten to their post-move form — full paths, suffixes
   after `./`/`../`, directory prefixes, declared `prefix_rewrites`,
   Markdown links (recomputed when the relationship changes; a relative
   link whose old text still names the target after the move is left
   alone), wikilinks — across the corpus, declared
   `external_roots`, and declared `scan_paths`; every rewrite is
   journaled and undo-able. FORM IS PRESERVED (C2): a root-path
   citation is rewritten to the new root path (never a bare basename —
   a truncated path like `_org/x.md` that is a unique suffix of a
   moved path gets the full new path; a stale path that merely
   basename-matches is skipped and reported as `unresolved_path`),
   relative links stay relative, and tokens with a URL scheme or host
   (`https://docs.acme.example/...`) are never touched. Opt out only
   deliberately with
   `"rewrites": {"mode": "off"}` (globally) or `"rewrite": false` on a
   single move (a vendored subtree whose references travel with it —
   its citations are still SHOWN, with file/line/token, under
   `rewrite_off` (C6)). `reference_risks` then lists moves STILL
   keeping citations the plan does not fix — a DICT KEYED BY MOVED
   PATH: `{path, mentions, external_references, uncovered_references,
   referenced_by, citations}` where each citation carries `{file,
   line, text, token, status, why}` (B2). Cover them in the plan
   (`edits: [{file, find, replace}]` — applied after the moves,
   journaled, undo-able; `edit.file` is the move-source path if the
   file moves, else its current path, and `find` must occur exactly
   once in the post-rewrite view — see the edits section), move the
   citing files too, or get explicit human sign-off on
   the breakage. Never silently accept. Spans the pass could not safely
   rewrite (two live readings of one citation, a shared basename, a
   stale path that only basename-matches) come back as
   `rewrite_skipped` rows with {file, line, reason, token,
   text} — one row per skipped token, the rest of the line still
   rewrites (C5) — and `skip_rewrites` in the same summary lists the
   subset that cite a MOVING path ({file, line, token, reason}), the
   single bad lines to hand-edit. Fix those with `edits`. Citations
   from OUTSIDE the corpus (skills, cron scripts, scratch harnesses)
   are rewritten only when their directory is declared in the plan —
   sweep the directories that cite your corpus with `refs --scan-path`
   (see "Reference rewrites"). Recall limits: a bare word is never a
   citation (path context is required); a bare basename mention
   matches only names of 3+ characters containing a dot, and an
   ambiguous basename (several corpus files share it) is never
   attributed; extensionless files and non-text extensions are never
   read.
5. **Unresolved is honest.** Files you can't classify go in
   `unresolved` with a reason — never guess. `keep` is for top-level
   entries that stay (a watchlist, a folder that is already right);
   `skip` names top-level entries you deliberately did not examine.
   The sweep is recursive: loose files INSIDE existing folders (a
   docs/ full of slop, say) are candidates for moves too — nested
   sources are normal, not exceptions.
6. **Corpus changed?** If apply raises `corpus_changed`, re-inventory and
   regenerate the plan. `--force` only if the changes were yours and
   expected. Staleness is content-based: a bare `touch` won't force a
   re-plan, but a real edit — even one that keeps the byte size — will.
7. **Respect `.slopkeeperignore`.** A corpus may carry a
   `.slopkeeperignore` at its root (gitignore-style: `dir/` for a
   whole subtree at any depth, `*.ext` for basenames everywhere,
   `/path` anchored to the root). Ignored paths are invisible to
   inventory, draft, plans, reference scans and the fingerprint — the
   tool will refuse any move source or edit target inside them
   (`move_source_ignored` / `edit_target_ignored`) even if your plan
   predates the ignore file. Ignored directories remain legal move
   DESTINATIONS. `doctor` reports when the file is active. Do not
   propose plans that touch ignored paths, and do not cite files
   inside them as reference risks — their citations are invisible to
   `--strict-references` by design.
   `.slopkeeperignore` is NOT a gitignore: it shapes what SlopKeeper
   sees, not what git tracks — copying a project's `.gitignore` over
   strands co-located files (doctor warns when ignored and
   non-ignored files share a folder). The override: declare an ignored
   dir via the plan's `scan_paths` and its files join the citation
   rewrite pass and accept edit targets (undeclared,
   `edit_target_ignored` stands) — moves inside it stay refused.
   Dot-directories are CONTENT as of fp4 (`.claude/skills`, cron
   prompts): only true tool layer (`.git`, `.slopkeeper`, `.venv`,
   `node_modules`, ...) is ever invisible — per-corpus noise
   (`.cache`, transcript dumps) is this file's job, and doctor
   suggests the lines.

## Fixture hygiene (repo tests)

Everything this repo commits as test data — fixtures in tests/, the
golden files, and the examples in this guide and in src docstrings —
carries GENERIC placeholder content only: fictional companies (acme,
globex), neutral filler (minutes, memos, itineraries), placeholder
paths (/home/agent). Never commit anything real about the author or
their life: no colleagues' names, no real companies or products, no
personal research topics (politics, health, hobbies), no contact
details, no home-directory paths. The SPDX license header is the only
intentional name. When you add a fixture or an example, invent an
"ACME CO" — never reach for something real.

## The protocol

```
1. slopkeeper doctor <root>
     → strategy suggestion, token estimate, advice. Read it.
       Diagnostics included: per-top-dir file counts (and which dir
       ate the inventory walk cap), nested git repos and node_modules
       with copy-paste .slopkeeperignore lines (copying them
       silences the warning), ignore-stranding folders (.slopkeeperignore
       is NOT a gitignore), cache-like dot dirs with suggested lines,
       nested corpora, code-anchored directories (file:line evidence),
       non-regular files (devices/FIFOs — sandbox deny-list mounts;
       pruned from every walk and the fingerprint so sandboxed and
       unsandboxed runs agree), and a dangling-citation count
       (`refs --dangling` lists them).

2. slopkeeper inventory <root> [--fast --summary | --out inv.json]
     → <800 files: full JSON with heads + cross-references (read it whole)
       >800 files: --fast --summary, then per-dir inventories as needed
       --fields path,title,mtime (B11): the lean per-file projection —
       the WHOLE document is shaped (dirs, clusters and references are
       dropped; C7), written compact. Size scales with the corpus:
       roughly 150 bytes per file (~90KB for a 630-file corpus,
       ~23k tokens) — titles ride along without opening anything.
       Combine with --fast.
       --out inv.json: the SHAPED payload (--summary/--fields are
       honored in the file, not ignored) lands there; stdout carries
       only counts (stats + out_path) — read the file, not the terminal.

3. slopkeeper clusters <root> [--out clusters.json]
     → similarity HINTS: content clusters, scrape pairs, dupes, hubs,
       stale-version pairs (same basename, different content, older
       mtime — "likely older version"), and identical subtrees
       collapsed to one entry naming every copy.
       Name each cluster, merge/split by judgment. Never apply a cluster
       as-is; and check `hubs` before moving a file cited by many others.
       --out clusters.json: full payload in the file, counts on stdout.

4. slopkeeper strategies show <suggested-id>   (or your choice)
     → the partitioning rubric. Apply it. Strategies:
       role-bots | topics | projects | hybrid (default for mixed corpora)

4b. slopkeeper propose <root> [--scan-path <dir>] [--external-root <dir>]
     → the layout fork as data: 2-3 candidate partitions (by-project,
       by-kind, hybrid — each naming its strategy), each with
       move_count, citation_break_count, code_anchor_count and
       unresolved_estimate. Read-only; never authors a plan. Declare
       citing dirs (--scan-path for an ignored subtree, --external-root
       for a skills store) or the break counts ignore them. See
       "Choosing a layout: propose" below.

5. Write the plan JSON. Get the template:
     slopkeeper plan schema
   Record the inventory's fingerprint as based_on.corpus_fingerprint.
   Cover every top-level entry: moves | keep | unresolved | skip.
   (keep accepts "name" or {"path": "name", "reason": "..."}; skip is
   a plain list of names.)
   The corpus is treated recursively: don't stop at the root level.
   A folder full of loose slop (docs/, scratch/, whatever the owner
   dumped) gets its files planned like any other — name nested files
   directly in moves. Only use skip for a folder you genuinely did
   not examine, and say so to the owner.

6. slopkeeper plan validate <plan.json> <root>
     → fix every error code it gives you. Then:
   slopkeeper plan render <plan.json> <root> --out plan.md
     → HUMAN GATE: show plan.md. Wait for approval.

7. slopkeeper apply <plan.json> <root>            (dry-run)
   slopkeeper apply <plan.json> <root> --execute  (for real)
     → the citation rewrite pass runs by default (see "Reference
       rewrites"): `would_rewrite` (dry-run) / `rewrites` (executed)
       carry per-file counts AND per-line {line, old, new} detail (B1);
       `rewrite_skipped` rows carry {file, line, reason, token, text}.
       `--rewrites-out rewrites.json` writes the full line-level
       detail. `reference_risks` is a dict keyed by moved path, and
       lists ONLY moves keeping citations the plan does not fix — each
       row carries `citations` ({file, line, text, status, why}) and
       `uncovered_references` (B2). A plan whose
       bytes differ from the last `plan render --out`
       (e.g. hand-edited JSON) gets a `render_warning` in the result —
       informational only: re-render or get explicit sign-off if the
       human approved the earlier render.

8. slopkeeper scaffold <plan.json> <root> --execute
     → writes AGENTS.md roster + per-folder AGENTS.md/STATE.md.
       Existing files are never overwritten; you get *.slopkeeper-suggested
       to merge by hand.

9. Report: moved counts, risk resolutions, suggested-file merges needed,
   journal path (for undo). Offer: slopkeeper undo <journal> <root>
   (dry-run is the default; add --execute to actually restore)

   PARTIAL APPLY: `apply --execute` can fail MID-RUN (an OSError on one
   hop — permissions, disk full). The envelope then carries
   "partial": true, the process exits non-zero, and next.command is
   the exact undo for this journal. Read `errors`, fix the cause, run
   the suggested `slopkeeper undo <journal> <root>` (dry-run is the default;
   add --execute to actually restore), then re-apply. A source that
   vanishes mid-apply is the same path: it lands in `errors` and the
   run exits non-zero — never silent. Earlier hops stay journaled and
   undoable throughout.
```

Classification signals: work from the inventory's fields before opening
anything — `title` and `head` (what the file says it is), `mtime` (how
fresh it is), cluster membership and `hubs` (what cites it, and what it
cites). A file with no usable signal is `unresolved` with a reason —
never a guess.

## Choosing a layout: propose

Before authoring a plan, `slopkeeper propose <root>` turns the layout
fork into data — one candidate per strategy's mechanical core:

- **by-project** (the projects rubric): a directory carrying a project
  marker (README.md, pyproject.toml, package.json, Cargo.toml, go.mod,
  setup.py, requirements.txt, pom.xml) keeps its whole subtree in
  place — never split. Everything without a project signal is counted
  as unresolved, not guessed.
- **by-kind** (the mechanical floor): every file sorts into docs/,
  data/, scripts/ or media/ (misc for the unknown). Structure
  flattens; nothing is left unresolved.
- **hybrid**: project subtrees stay whole, and the leftovers kind-sort
  into hybrid's shared buckets (scripts/, data/, research/).

Per candidate: `move_count` (files whose dest differs from today),
`citation_break_count` (moved files cited from a file that does not
move — the same rule apply's `reference_risks` uses, so the counts and
the gate agree), `code_anchor_count` (anchored dirs the layout
disturbs — `__file__` math no pass can rewrite), and
`unresolved_estimate` (files the layout cannot place).

Declare citing dirs or the break counts ignore them:
`--scan-path <dir>` for an ignored subtree inside the corpus,
`--external-root <dir>` for a skills store or cron dir outside it
(both repeatable; a typo refuses with scan_path_missing, a skip-list
member with scan_path_skip_list). Verify any count with
`refs <root> <path> --scan-path <dir>`.

Read-only: no fingerprint effect, no writes, and it never authors a
plan. Pick a layout, write the plan (rules or moves), then `plan
validate` — the human still gates.

## Re-tending (the recurring loop)

After the first sweep, `apply --execute` and `scaffold --execute` both
write `.slopkeeper/manifest.json`. From then on, a tending pass costs
tokens proportional to the slop shot in, not the corpus size:

```
1. slopkeeper status <root>
     → new_files (with title+head: triage them without opening anything),
       modified, removed, stale folders, still-unresolved items.
       HEAVY DRIFT (the envelope carries "heavy_drift" and next.command
       names it): the manifest no longer describes the corpus — an
       out-of-band change, a sync, a bulk import. Rebaseline with
       `slopkeeper status --record <root>` (it carries the unresolved
       backlog and last_plan forward) instead of hand-building a
       zero-move plan; then draft/triage against the fresh manifest.
       A subtree that became its own corpus is reported under
       nested_corpora (its files move to removed_now_nested, never
       counted as removed).

2. slopkeeper draft <root> --out draft-plan.json
     → OPTIONAL but usually the fastest path: auto-drafts a PARTIAL plan
       by nearest filed neighbor (TF-IDF cosine). The sweep is
       RECURSIVE: new files at any depth are draftable (a file dumped
       into docs/ gets routed wherever its nearest filed neighbor
       lives); `--depth N` bounds the sweep to N directory levels
       (1 = root + immediate subdirs). Confident files get a
       move with the neighbor + score as the reason; everything else is
       unresolved WITH the reason ("best neighbor only 0.03"). Draft is
       a guess with a citation — review every move, then edit.
       The stdout envelope carries COUNTS only ({drafted_count,
       unresolved_count, new_count, undrafted, out_path}) — the plan
       body lives only in the --out file. The default path resolves
       against your CWD and is REFUSED if it lands inside the corpus
       (`draft_out_inside_root`) — a draft written into the root would
       shift its fingerprint and the next apply would refuse with
       corpus_changed. When new files exceed what the draft could see
       (status's 2000-new_files cap), `undrafted` carries the count and
       the reason, and the next-step hint says so. A zero-move draft
       (everything went unresolved) is still a valid plan: validate it
       and apply --execute — the backlog is recorded for the human.

3. Triage. Fix the draft: reclassify wrong moves, resolve the
   unresolved ones (they're the actual judgment work), add folder
   entries for any new folder the roster needs.

4. apply --execute → scaffold --execute  (both record the manifest)
```

Notes:
- A delta plan should carry `"partial": true` at the top level (draft
  plans already do) — no --allow-partial flags needed.
- `still_unresolved` in the status output is the human-decides backlog;
  surface it every tend until it's empty.
- `skipped_bad_unresolved` counts backlog entries whose `path` was
  null or missing — they are skipped (state-file content is data,
  never a crash); the next re-record drops them from the manifest and
  records the same count there.
- `modified` files are leads, not chores: a changed STATE.md or AGENTS.md
  is the owner tending by hand — leave it alone; a changed runbook may
  deserve a fresh look in your triage note.
- A directory too big to plan (tens of thousands of entries) is
  `slopkeeper move`'s job — one journaled move, gate intact, no
  61k-entry plan. See "Bulk moves".

## Tending several corpora on a schedule

Register the corpora you tend once; the registry is the loop's to-do
list, so a scheduled job is one call instead of a shell script. It lives
at `$XDG_CONFIG_HOME/slopkeeper/roots.json` (set `SLOPKEEPER_HOME` to
relocate it, e.g. for tests); content is sorted, unique, absolute paths.

```
slopkeeper roots add <dir>     → register a corpus (idempotent; must exist)
slopkeeper roots remove <dir>  → unregister (the corpus itself is untouched)
slopkeeper roots list          → what is registered

slopkeeper status --all        → ONE digest over every registered root:
  per root {root, ok, never_tended, new_count, modified_count,
  removed_count, stale_folders, unresolved_count}
```

The digest is counts, not listings — keep it skimmable, then drill into
whichever roots are nonzero with the ordinary per-root `status`. A
deleted root, a never-tended corpus, or a root whose status fails is a
row (`ok: false` carries the error), never a failed run — the other
roots always report. `--all` with an empty registry is a clean error:
add roots first.

## Bulk moves: a directory too big to plan

When a directory is too big to enumerate — a 61k-file data dir, a sync
target — neither a bare `mv` (no audit trail) nor a 61k-entry plan
(unwritable, unreviewable) is right. `slopkeeper move` is the third
thing:

```
slopkeeper move <src> <dst> <root>            (gate: renders the
                                               approval markdown, moves nothing)
slopkeeper move <src> <dst> <root> --execute  (for real)
```

- The gate is the render on stdout — the real path, reference rewrite
  counts, the same human approval as any plan. Nothing moves without
  --execute.
- One move, one journal entry: `undo` reverses it, the manifest is
  re-recorded, and the citation rewrite pass fixes references to the
  moved path exactly like any plan.
- The result carries an integrity check — `{"src_files": n,
  "dst_files": m, "match": bool}`, a raw subtree walk count on both
  sides (the `find | wc -l` replacement; ignored and nested-corpus
  files count, because bytes on disk are what must survive). A
  mismatch means the move dropped something — read it, never ship it.
- The usual refusals apply: `path_escape`, `move_source_ignored`,
  `dst_exists`, `corpus_changed` (the plan is built against a fresh
  fingerprint; if the corpus changes between gate and execute, re-run).
  The generated plan is materialized under `.slopkeeper/` and bound
  into the journal header — the audit trail is complete without any
  hand-written JSON.

## Rule-based moves: one entry, concrete paths at the gate

"Everything left in reports/_org goes to reports/org" is one plan
entry, not one move per file. A plan may declare rules:

    "rules": [{"match": "reports/_org/", "to": "reports/org"}]

- `match` is a corpus-relative path prefix (everything under it,
  recursively) or a `/`-aware glob: `*`/`?` stay inside one path
  segment, a whole-segment `**` crosses directories, and a slash-free
  glob anchors to the corpus root (`**/*.md` matches at any depth).
  The same glob grammar as `.slopkeeperignore`.
- Expansion happens at validate, BEFORE the human gate: every visible
  file under the match becomes a concrete move. The render lists the
  concrete moves and summarizes the rule ("expanded to N moves") —
  the human approves real paths, never a pattern.
- **Precedence** (0.1.6 field test B3): explicit moves beat rules —
  a rule never overrides a recorded explicit move, whatever the
  destination — and among rules the FIRST match wins, so order
  carve-outs before their catch-all
  (`[{"match": "reports/_org/*actions*", "to": "projects/actions"},
  {"match": "reports/_org/", "to": "reports/org"}]`). A rule whose
  matches were taken by an earlier rule reports `superseded: N` in
  the validate output; a rule whose matches all sit inside
  keep/unresolved entries warns at the gate ("keep shadows rules") —
  B4, never silent. A rule that matches no visible files still
  refuses with plan_bad_rules.
- By default a rule FLATTENS: every matched file lands as
  `<to>/<basename>`, and subdirectory paths are lost (the render says
  how many files were flattened). `"preserve": "tree"` keeps each
  file's path below the match prefix instead —
  `{"match": "reports/_org/", "to": "reports/org", "preserve": "tree"}`
  sends `reports/_org/2026-09-07_lib/img/x.webp` to
  `reports/org/2026-09-07_lib/img/x.webp` (B5).
- Every expanded move takes the rule's rewrite opt-out: a rule with
  `"rewrite": false` stamps `"rewrite": false` on each move it
  expands (C1 — a rule without the key, or with true, expands to the
  default pass; adding `"rewrite": true` to a rule is a no-op).
- keep and unresolved entries win over rules (recorded decisions), and
  files inside an explicitly moved directory travel with that dir.
- **Straggler report**: every validate and render lists the files left
  behind in each folder the plan removes files from — count plus
  examples in the render, the full list in `plan validate --out` JSON.
  Advisory, never fatal: stranding files may be the point (list them
  as unresolved), but the gate shows it either way.
- `plan validate` stdout carries counts (moves, straggler_dirs,
  straggler_files, emptied_dirs); pass `--out plan.validated.json` for
  the full rule-expanded move list plus the straggler report.

## Directories emptied file by file (prefix rewrites)

A directory whose files all move INDIVIDUALLY has no single move
covering it, so convention strings naming it
(`reports/_org/<date>_<topic>.md`) are not rewritten by the pass and
would dangle (0.1.6 field test B8). `plan validate` and `plan render`
list every such directory with its destination breakdown, and a
ready-to-paste suggestion when one destination takes a clear majority.
Declare the rename explicitly — never inferred, because a mechanical
rewrite can be wrong (a convention string may have moved somewhere
else semantically):

    "prefix_rewrites": [{"from": "reports/_org/", "to": "reports/org/",
                         "reason": "directory renamed"}]

The rewrite pass then swaps the prefix on every citation naming the
old one (convention strings, exact paths, dir forms), and the render
lists every rewrite site by line. Glob and brace citations under the
prefix are corpus-checked first (D2): every file the pattern names
must move into ONE destination for the rewrite to fire — otherwise
validate refuses with `prefix_rewrite_dead_glob` unless the entry
sets `"acknowledge_broken": true` (the agent's own out-of-corpus
references — `/tmp`, `~/Downloads` — are the agent's problem,
accepted consciously). For pattern renames
(`misc/<customer>-<context>/` -> `projects/<customer>/<context>/`) or
semantic retargets, use hand `edits` instead — each `find` must be
unique per file.

## Removing directories the plan emptied (D12)

The moves often leave directories empty (`misc/`, `reports/_org/`,
empty `.claude/.cc-writes` subdirs). The suggestions stay advisory —
but with `"remove_emptied": true` in the plan, apply REMOVES every
directory the plan emptied: the gate lists exactly what will be
removed before approval, each removal is a journaled `rmdir` op, undo
recreates the dirs, and a directory that still holds any file
(including ignored ones the validate walk cannot see) refuses loudly.
A partial run removes nothing. Without the field, nothing is ever
deleted — the tool's no-delete invariant holds.

## Reference rewrites

When a move changes a path, citations of it must change too. The
citation rewrite pass does this on every apply — you do not hand-write
find/replace pairs for it:

- **Default ON.** Every apply scans eligible TEXT files — the corpus,
  declared `external_roots`, declared `scan_paths` — and rewrites every
  citation of every moved path to its post-move form: full corpus
  paths, path suffixes after `./`/`../`, moved directory prefixes (so
  convention strings like `reports/_org/<date>_<topic>.md` come free),
  declared `prefix_rewrites`, Markdown links RECOMPUTED whenever
  either end moves, and Obsidian wikilinks (`[[x]]`, `[[x|alias]]`,
  `[[x#anchor]]`). Nothing else on a line is touched; zero-hit files
  are not written; every rewrite is journaled and undo-able. Opting
  out is explicit: `"rewrites": {"mode": "off"}` — exactly the old
  behavior. A single move (or rule) may also carry `"rewrite": false`
  (B1): a vendored subtree whose internal references travel with it
  has its internal links NOT rewritten — and since C6 its citations
  are still SHOWN (record-only scan) under `rewrite_off` with full
  file/line/token detail; anything citing it from outside still gets
  the pass.

- **Form is preserved (C2).** The replacement keeps the citation's own
  shape: a root-path citation gets the new root path (never a bare
  basename — a bare name only names the file from the target's own
  folder); a truncated path that is a UNIQUE suffix of the moved path
  (`_org/x.md` for `reports/_org/x.md`) is resolved and written as the
  full new path; a relative link is recomputed only when the citing
  file itself moves, and left alone when its old text still names the
  target from the post-move location (no `./x.md` → `x.md` churn); and
  a token with a URL scheme or host (`https://docs.acme.example/...`,
  `docs.acme.example/...`) is never a citation and never rewritten. A
  multi-segment token that merely basename-matches a moved file (a
  stale path like `misc/email.md` for `misc/acme/email.md`) is
  skipped and reported — `skip_rewrites` in the summary carries the
  single bad lines ({file, line, token, reason}) to hand-edit.

- **The gate shows the actual edits (B1).** `plan render` and the
  apply dry-run list every rewrite site as `file:line` with the old
  and new text — not just counts. For the full machine-readable list
  pass `--rewrites-out rewrites.json` (on either command): 
  `{file: [{"line", "old", "new"}]}`. Per-move `rewrite: false`
  subtrees show as `[rewrite: off]` in the move table.

- **Ambiguity is never guessed.** A span with two live readings (two
  moved paths cited such that both could be meant, or a bare mention
  of a basename several corpus files share — where at least one member
  moves) is left untouched and reported as a `rewrite_skipped` row —
  now carrying `{file, line, reason, token, text}` (B9), one row per
  skipped token (C5: the rest of the line still rewrites) — fix those
  by hand with edits. A path-qualified mention of a file that does not
  move (`slack/AGENTS.md` where some other AGENTS.md moves) is NOT a
  skip: the 720-row noise of the 0.1.6 report is gone. Binary files
  (PDFs, images) are never listed as skipped — they are simply not
  scanned, like every reference scan. Glob and brace tokens are
  corpus-checked first (D2/D11): every match must move into ONE
  destination for a rewrite; otherwise the token is skipped with a
  `glob_split` / `glob_unmatched` row instead of being rewritten into
  a pattern that names nothing. The hand-edit list (`skip_rewrites`)
  carries only actionable rows (D3): a BARE ambiguous mention stays in
  the informational `skipped` list only — it names no corpus file, so
  no edit can be written for it. A citation on an edit-owned line the
  edit will not fix is `edit_owned_uncovered` (D1) — a skip row plus
  a risk-row status, never silently dropped.

- **One engine behind `refs`, `reference_risks` and the rewrites
  (B2).** The apply-time `reference_risks` rows carry per-citation
  detail: `{path, mentions, external_references, referenced_by,
  uncovered_references, citations}` where each citation is
  `{file, line, text, token, status, why}` — status is `rewrite`
  (the pass fixes it), `no_change` (post-move form identical),
  `edit_covered`, `edit_owned_uncovered` (D1: the line is an edit's,
  but the edit's find does not cover this citation), `rewrite_off`
  (inside a "rewrite": false subtree — now WITH the site's own
  file/line/text/token, C6), `not_utf8`, `ambiguous_basename`,
  `prefix_conflict`, `unresolved_path` (a stale or foreign path that
  only basename-matches a moved file — C2) or `not_recorded` (now
  WITH the line and token when a targeted read finds them — D6), and
  `why` spells out the reason in English. Citations are deduped per
  (file, line, span, token) — a mention shared by K moving members is
  ONE row (D5). A move whose every citation is covered is NOT a risk
  at all — the report only lists moves with
  `uncovered_references > 0`, and `--strict-references` refuses on
  exactly those. `refs <root> <dir>` now lists citations of any path
  UNDER the directory too (it used to miss exact-file citations —
  the disagreement the field report hit).

- **`refs` is the verification tool** — before authoring and after
  applying:

      slopkeeper refs <root> <path-or-basename>
        → citation sites {file, line, text, kind} for a corpus path,
          directory, path suffix, or unambiguous basename (an
          ambiguous basename returns nothing — never attributed).
          A directory name includes citations of paths under it.
          Citations flagged "external": true sit outside the corpus
          (absolute file path). A slashed name that exists nowhere in
          the corpus comes back with "nonexistent_path": true and
          only EXACT citations of the string — never a different
          same-basename file's citations (D7).

      slopkeeper refs <root> <name> --scan-path ~/skills --scan-path ~/cron
        → also sweep directories OUTSIDE the corpus — or ignored ones
          inside it (explicitly naming an ignored dir is the override;
          the per-dir result marks it "ignored": true). Repeatable. A
          --scan-path inside the tool-layer skip list (.git,
          .slopkeeper, .venv, node_modules, ...) is never read and
          reported per-dir as scan_path_skip_list — never silently
          empty. Non-text files are skipped; the per-file read cap is
          512 KiB; scan dirs have no size ceiling.

      slopkeeper refs <root> --dangling [--limit N] [--prefix <dir>]
        → path-ish citations that resolve to NOTHING (already-broken
          links, half-updated references): {file, line, token, text}.
          Informational — fix by hand with edits; the rewrite pass
          never touches them. Bare words are never counted (they are
          prose), and URL-scheme tokens are skipped. The default is
          UNBOUNDED (D10 — the old 200-row cap made a clean-move
          verification impossible); --limit caps explicitly, --prefix
          lists every citation still naming a removed directory, and
          --per-file (default 5) applies here too. Slash-words (no
          extension, no trailing slash, no matching corpus dir) are
          prose, not citations — wikilink and markdown-link forms are
          always listed.

- **`edits` remain for surgical single-occurrence fixes** — a
  `rewrite_skipped` span, or wording you want different from the
  mechanical swap:

      "edits": [{"file": "<citing file (move-source path if it moves)>",
                 "find": "<exact text from the citation line, unique>",
                 "replace": "<new path after the move>",
                 "reason": "fix citation"}]

    One rule (C3/C4), checked identically by `plan validate` and
    `apply`: the edit runs AFTER the rewrite pass and OWNS its line —
    the pass never rewrites a line an edit covers (no double
    application), so `find` must occur exactly once in the file once
    the pass has run. The edit replaces exactly ONE occurrence (its
    first in the file), so a citation on the owned line is covered
    only when it overlaps that occurrence — anything else on the line
    is reported as `edit_owned_uncovered` (a skip row plus a risk-row
    status, D1): it will survive the edit, so fix it with another
    edit. Author `find` from the citation line as it
    stands today when the pass will not touch it (a skipped span, a
    convention string); author it from the POST-rewrite text when the
    pass itself produces that text (e.g. a `prefix_rewrites` rename
    you want to extend). Wrong form → edit_stale at validate, before
    anything moves. Edits and rewrites coexist in one file (C3): the
    pass rewrites the file's other lines, the edit lands on its own.
    find occurring more than once → edit_ambiguous. The target must
    be valid UTF-8 (edit_not_utf8) — non-UTF-8 files are refused,
    never re-encoded. Edits run after the moves and after the rewrite
    pass, are journaled, and undo reverses them.

- **Citations OUTSIDE the corpus** are rewritten when their directory
  is declared in the plan:

      "external_roots": [{"path": "/abs/dir", "expect": 3}],
      "scan_paths": ["/abs/corpus/.claude"]

    external_roots: absolute dirs OUTSIDE the corpus (cron scripts,
    sibling projects, skill stores). The optional expect asserts the
    rewrite count — a mismatch refuses (rewrite_expect_mismatch)
    before ANY write. scan_paths: absolute dirs the corpus walk does
    not see (ignored subtrees) — they join the rewrite pass the same
    way and `plan render` marks them. Edit/rewrite targets inside
    ignored dirs are allowed ONLY when declared via external_roots or
    scan_paths — otherwise edit_target_ignored stands. Undeclared
    paths are never touched. External edits and rewrites run after the
    in-root ones, and `plan render` marks their rows (writes outside
    the corpus).

- **Writability preflight (B7).** On `--execute`, apply checks every
  rewrite and edit target for write access BEFORE the first move and
  refuses with `target_unwritable` naming the paths — a sandbox that
  denies `.claude/skills` can no longer produce a half-applied run.
  Run the apply outside the sandbox, or grant the paths.

**"Cited but 0 rewrites" is explained (B10).** When the render names a
move whose citations produced no rewrite, each citing line is listed
with its reason (`no_change`: both ends moved and the relative form is
identical; `ambiguous_basename`; `not_utf8`; ...) — no more
unexplained silence.

## Error codes you will see

| code | meaning | your move |
|---|---|---|
| `plan_incomplete` | top-level entries uncovered | add to moves/keep/unresolved |
| `plan_no_moves` | `moves` is not a list, or is empty on a non-partial plan | a plan needs at least one move — unless it carries `"partial": true` (a zero-move partial plan is valid: the draft filed everything as unresolved; validate it and `apply --execute` records the backlog) |
| `plan_stale_src` | a source vanished | re-inventory, regenerate |
| `corpus_changed` | fingerprint mismatch | re-inventory; `--force` only if yours |
| `dst_exists` | destination occupied | resolve collision in plan |
| `referenced_moves` | strict mode: refs would break | rewrite refs first |
| `plan_nested_moves` | dir + its contents both moved | move dir whole OR its files |
| `plan_bad_keep` | keep entry is neither string nor {path} object | use `"name"` or `{"path": "name", "reason": "..."}` |
| `plan_bad_skip` | `skip` is not a list of plain names, or names a nonexistent top-level entry | `skip` is `["docs", ...]` — plain names (no slashes) of corpus entries that exist; the plan documents what it ignored |
| `plan_bad_unresolved` | an unresolved entry has no non-empty string `path` (null, missing, or blank) | every entry is `{"path": "<file>", "reason": "<why it waits>"}` — fix or drop it |
| `edit_stale` | edit target missing, or find-text absent from the file's post-rewrite view (the pass leaves each edit's own line untouched — author find from the line as it stands, or post-rewrite when the pass produces that text) | update the edit in the plan |
| `edit_ambiguous` | find text occurs more than once | make find unique; split the edit |
| `edit_not_utf8` | edit target is not valid UTF-8 (e.g. latin-1); editing would corrupt it | convert the file to UTF-8 first, or rewrite the citation by hand |
| `plan_bad_external_root` | external_roots entry is not an absolute, existing directory | create or fix the directory, or omit external_roots |
| `scan_path_missing` | `refs --scan-path` directory does not exist | create it or fix the path |
| `no_manifest` | draft with no prior tend | run a first tend first |
| `draft_out_inside_root` | `draft --out` resolves inside the corpus root — a draft written into the corpus shifts its fingerprint (the next apply would fail `corpus_changed`) | write the draft outside the corpus: run from another directory (the default `draft-plan.json` resolves against your CWD) or pass an explicit path elsewhere |
| `path_escape` | path outside root, contains a null byte, or is the root itself | use relative paths to real files; for edits outside the root, declare `external_roots` |
| `manifest_unreadable` / `manifest_bad_schema` | `.slopkeeper/manifest.json` corrupt | delete it; `status --record` rebuilds |
| `tending_in_progress` | another tend holds `.slopkeeper/tend.lock` (execute-mode apply/scaffold/undo, `status --record`) | wait for the other tend, or check for a dead agent's session; the OS frees the lock on process death |
| `root_not_dir` | `roots add` target is not an existing directory | register corpora that exist; SlopKeeper doesn't create them |
| `root_not_registered` | `roots remove` for a path not in the registry | check `slopkeeper roots list` |
| `root_missing` (in `status --all` data) | a registered root is no longer a directory | re-add its new path, or `roots remove` if it's gone |
| `roots_unreadable` / `roots_bad_schema` | roots registry file corrupt/foreign | delete it; re-add roots |
| `no_roots` | `status --all` with an empty registry | `roots add` first, or run `status <root>` directly |
| `no_root` | `status` with neither a root nor `--all` | name a root or use `--all` |
| `similarity_cap` | clusters handed more than 3000 files (the O(n²) matcher would crawl) | partition per-directory: `slopkeeper clusters <root>/<subdir>`, or organize smaller scopes |
| `plan_not_json` | plan file is not valid JSON | fix the JSON syntax and re-validate |
| `plan_unreadable` | plan file cannot be read | check the path exists and is readable |
| `plan_not_object` | plan JSON parses but is not an object | the plan is a JSON object with `schema` and `moves` — not a bare list |
| `plan_missing_field` | plan is missing a required field | add it; see `slopkeeper plan schema` |
| `plan_bad_schema` | `plan.schema` is not the current schema | regenerate from the current template: `slopkeeper plan schema` |
| `plan_wrong_root` | the plan's `corpus_root` names a different corpus | re-target or regenerate the plan for this root |
| `plan_bad_move` | a move lacks `from`/`to` | every move is `{"from": "...", "to": "..."}` |
| `move_source_ignored` | a move source is ignored by `.slopkeeperignore` | the tool never touches ignored paths — drop the move, or remove the pattern from the ignore file |
| `edit_target_ignored` | an edit target is ignored by `.slopkeeperignore` and its directory is not declared in `external_roots`/`scan_paths` | declare the dir via scan_paths (or external_roots), drop the edit, or remove the pattern from the ignore file |
| `plan_noop_move` | a move's `from` equals its `to` | drop the no-op move |
| `plan_dup_src` | two moves share a source | one move per file |
| `plan_dup_dst` | two moves share a destination | destinations are full file paths — a `to` naming a directory means the filename is missing; two files cannot land on one path (chains a→b, b→c are fine) |
| `plan_dst_under_src` | a destination is inside another move's source | move the directory whole, or flatten the plan |
| `plan_bad_edit` | an edit lacks `file`/`find`/`replace` | every edit carries all three; see `slopkeeper plan schema` |
| `plan_bad_folder` | a folder entry has no `path` | every folder entry names a path |
| `plan_bad_kind` | folder `kind` is unknown | kind is one of bot, case, topic, project, archive, scratch |
| `move_cycle` | the moves form a cycle (a→b→a) | remove the circular moves from the plan |
| `symlink_escape` | a path resolves outside the root via a symlink | remove the symlink or move its target explicitly |
| `no_corpus` | the argument to `doctor` is not a directory | pass the directory you want to organize |
| `no_plan` | REPL: no compiled plan to act on (`/apply`/`/render`/`/scaffold` before `/plan`, or decisions changed since) | `/plan` first — it compiles decisions, validates and saves; `/load` + `/plan` for an agent-authored plan |
| `no_such_move` | REPL: not a move number | `/moves` lists the working plan's moves |
| `no_basename` | REPL `/refs`, or CLI `refs` without a basename and without `--dangling` | `/refs <basename>` / `refs <root> <basename>` — or `refs <root> --dangling` |
| `inventory_bad_fields` | `inventory --fields` names a field that does not exist | fields are: bytes, ext, head, kind, mtime, path, title |
| `no_path` | REPL: a required path argument is missing or empty | see the usage line in the message |
| `no_such_journal` | REPL: not a journal index (or a non-numeric one) | `/undo` lists journals; `/undo <n> [!]` |
| `cat_unreadable` | REPL: the `/cat` target cannot be read | check the path (it is relative to the corpus root) |
| `edit_unreadable` | an edit target cannot be read | check the file exists and is readable |
| `journal_unreadable` | the journal file cannot be read | check the journal path |
| `journal_empty` | the journal contains no entries | nothing to undo |
| `corpus_too_big` | draft's filed-file count exceeds its 3000 cap (clusters refuses with `similarity_cap`) | partition per-directory, or narrow the draft |
| `plan_bad_rewrites` | plan.rewrites is not `{"mode": "auto"}` or `{"mode": "off"}` | set `"mode"` to `auto` (default) or `off`; it is the only key |
| `rewrite_expect_mismatch` | an external root's `expect` count differs from the computed rewrite count | verify with `refs --scan-path`, fix `"expect"`, or drop the assertion |
| `plan_bad_scan_path` | a `scan_paths` entry is not an absolute, existing directory | declare absolute directories that exist — or drop scan_paths |
| `scan_path_skip_list` | the named directory is tool-layer skip list (`.git`, `.slopkeeper`, `.venv`, `node_modules`, ...) — it is never read | scan or declare a directory outside the skip list; tool layer is invisible by design |
| `rewrite_skipped` | a citation span was left untouched (two live readings of one citation, a shared basename, or a stale/foreign path that only basename-matches a moved file — `unresolved_path`) — a data row in the rewrite summary / `rewrites.skipped`, one per skipped token, never a raised error | fix the listed file:line by hand with `edits` — exact-once find/replace; `skip_rewrites` carries the subset that cites a MOVING path |
| `plan_bad_rules` | a `rules` entry is malformed, its `to` is an existing file or outside the root, or its `match` hits no visible files | every rule is `{"match": "<prefix or glob>", "to": "<dest dir>"}` — optional `"preserve": "tree"` and `"rewrite": false`; check the path (or the corpus already moved) |
| `plan_bad_prefix_rewrite` | a `prefix_rewrites` entry is malformed, `from == to`, or `from` is not an existing directory | every entry is `{"from": "<old dir>/", "to": "<new dir>/"}` renaming a directory that exists at plan time (usually one the plan empties file by file) |
| `prefix_rewrite_dead_glob` | a `prefix_rewrites` entry's glob/brace citation cannot be rewritten coherently post-move (matches stay put, split, or name no corpus file) — the swap would point at nothing (D2) | fix the plan's moves so every named file lands in one destination, drop the entry, or set `"acknowledge_broken": true` on the entry to accept the broken citation consciously |
| `plan_bad_remove_emptied` | `plan.remove_emptied` is present but not a boolean | set `"remove_emptied": true` to have apply remove directories the plan emptied (journaled, undoable), or drop the field |
| `merge_collision` | a directory move's destination is created by another move of the same plan, and a child name collides with an existing path or another move's destination (E1) | rename one destination or drop one of the two moves — the merge refuses a name collision instead of nesting the directory |
| `target_unwritable` | a rewrite or edit target denies writes (sandbox deny-list, read-only mount) — refused before ANY move | run apply outside the sandbox or grant write access; nothing was moved |
| `unknown_strategy` | no strategy with that id | `slopkeeper strategies` for the available ids |
| `skill_unknown_agent` | `skill install` target is neither a known agent nor a path-like string | use a known agent (claude, opencode, hermes) or pass a skills directory path — the skill lands in `<dir>/slopkeeper/` |
| `skill_target_not_dir` | `skill install` destination exists as a file | remove it or pass a different skills directory |

## Sizing

- Inventory: at ≤ 800 files the full JSON (heads + cross-references)
  is one read — read it whole. Above that a full read is too big
  (~1000 files ≈ 150k tokens): use `--fast --summary`, then
  `inventory <subdir>` for the dirs you plan to reorganize, plus
  title/head from the summary. The lean classification view is
  `--fields path,title,mtime` (~150 bytes/file — about 90KB for a
  630-file corpus, whole document shaped) — titles
  without opening anything; it shapes `--out` too. Past 8000 files
  the inventory is truncated (`stats.truncated`) — per-top-dir counts
  are still taken, and doctor names the directory that ate the cap.
- Binary heads: PDF/DOCX/PPTX (and any ext with a registered
  extractor) carry extracted first lines when the system has the
  extractor (`pdftotext`, `docx2txt`/`pandoc`, `unzip` — probed from
  PATH, never installed by us). Without one the head is the literal
  marker `signal: none (binary, no extractor)` — loud, never silent.
- Reference scanning: un-capped and streaming — apply and `refs` scan
  corpora of any size linearly (the 2000-file cap is gone); apply's
  `reference_scan` reports `files_scanned`/`elapsed_ms`, never a
  silent skip. Per-file read cap 512 KiB. (The inventory's embedded
  cross-reference scan only runs at ≤ 800 files; clusters/draft keep
  their 3000 cap.)
- Clusters and draft: 3000 files. Past it `clusters` refuses with
  `similarity_cap` (draft: `corpus_too_big`) — partition
  per-directory instead.
