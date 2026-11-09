# Scaffolding plan

- Strategy: **hybrid**
- Corpus: `/corpus`
- External edit roots: `/home/agent/harness`
- Based on inventory: `0123456789abcdef` (2026-11-02T00:00:00+00:00)
- Moves: 3 · Folders: 1 · Unresolved: 1

## Summary

| destination | files |
|---|---|
| `research/dossiers/` | 2 |
| `spark-1.2.3/` | 1 |

## → `research/dossiers/`

| from | final | refs | reason |
|---|---|---|---|
| `acme_dossier.md` | `research/dossiers/acme_dossier.md` | ⚠️ 1 | dossier research |
| `globex_dossier.md` | `research/dossiers/globex_dossier.md` | — | dossier research |

## → `spark-1.2.3/`

| from | final | refs | reason |
|---|---|---|---|
| `handoff-tp2.md` | `spark-1.2.3/handoff-tp2.md` | — | spark handoff |

## 🔁 Citation rewrites (automatic): 0 across 0 file(s)

## ⚠️ Unresolved (left in place)

- `ledger_wb.html` — no signal — human decides

## ✏️ Reference rewrites (applied after moves)

| file | find | replace | reason |
|---|---|---|---|
| `WATCHLIST.md` | `acme_dossier.md` | `research/dossiers/acme_dossier.md` | rewrite citation to the moved dossier |
| `/home/agent/harness/nightly.sh` ⚠️ external | `tmp/docs/acme_dossier.md` | `docs/acme_dossier.md` | external citation (cron harness) |

## Folders to scaffold

| path | kind | role |
|---|---|---|
| `research/dossiers` | case | Market research dossiers. |

_Approve by editing this plan's JSON, then: `slopkeeper apply <plan.json> /corpus --execute`_
