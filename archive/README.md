# archive/ — moved, not deleted

**The rule: nothing in this repo is deleted.** A file that no longer belongs where it was is moved
here, keeping its original basename, together with its origin and the reason. That way a stale
artifact stops misleading tooling and future agents without any information being lost.

This directory holds no live inputs. Nothing in the running system — no module, script, workflow,
test or CI step — reads anything under `archive/`.

## Contents

| Path here | Original path | Tracked? | Why moved | Date | Safe to restore? |
|---|---|---|---|---|---|
| `commit_messages/commit_msg.txt` | `commit_msg.txt` (repo root) | no (untracked, and not gitignored) | Stale scratch: the draft commit message of `ca00b16` ("feat: 2026-07 group improve campaign (12 groups, v2 pipeline)"), already used. Zero references anywhere in the repo. Written by `group-improve-v2.js`'s final commit-message phase. | 2026-09-08 | Yes — nothing reads it. `mv` back if wanted. Note it is untracked *and* un-ignored, so it shows in `git status` in either location. |
| `commit_messages/commit_msg_round2.txt` | `commit_msg_round2.txt` (repo root) | **yes** (`git mv`) | Stale scratch accidentally committed in `c7f846e` — it holds that same commit's own message. Zero references. | 2026-09-08 | Yes — `git mv` back. |
| `claude_workflow_runs/modules-v3.run.json` | `.claude/workflows/modules-v3.run.json` | **yes** (`git mv`) | Run-state config for panel batch **B5** (`model_v2` / `model_lgb` / `blend_fit`, 5 reviewers). The batch never ran: `research/module_improve_v3_batchB5.md` does not exist, none of its test files exist, and its `tree_note` freezes a 2026-08 tree description that is no longer true. Tracked, it would have been the default "working config" for the next `/panel-improve` run. | 2026-09-08 | Yes, but prefer starting from `.claude/workflows/modules-v3.example.json`. Its absence is safe: `module-improve-v3.js` defaults `CFG_PATH` to that path and returns `{error: 'missing modules config'}` with guidance — the workflow fails loudly, it never silently runs the wrong batch. |
| `local_residue/backtest_report.json` | `backtest_report.json` (repo root) | no (gitignored, `.gitignore` `backtest_*report.json`) | Stale dev-Mac output dated 2026-07-11 — a zero-trade crypto replay (`n_trades: 0`). The Jetson keeps its own. | 2026-09-08 | Yes. Every consumer is existence-guarded (`gui.py` freshness strip; `backtest._patch_report_gate_block` returns early when the path is absent) and a `backtest.py` rerun regenerates it. |
| `local_residue/oi_history.json` | `oi_history.json` (repo root) | no (gitignored) | Stale dev-Mac output dated 2026-06-10 — a single `BTC/USD` row (`oi_archive._LIVE_HISTORY_FILE`). | 2026-09-08 | Yes. `oi_archive` treats a missing file as "no live history" and refetches (live cache TTL 900 s). |
| `local_residue/adaptive_state_testq1.json` | `adaptive_state_testq1.json` (repo root) | no (gitignored, `adaptive_state*.json`) | Test residue: `tests/test_c26_Q1.py` calls `adaptive_config.record_trials('testq1', …)`, which writes `adaptive_state_{asset_type}.json` into `BASE_DIR`. | 2026-09-08 | Yes, and unnecessary — the next suite run recreates it at the repo root. Seeing it reappear there is expected, not a regression. |

All three `local_residue/` patterns are depth-agnostic in `.gitignore` (no leading slash), so those
files stay ignored under `archive/` exactly as they were at the root.

## Convention for future moves

1. Add a row to the table above: destination, original path, tracked?, why, date, restore note.
2. `git mv` for tracked files (so `git status` shows `R` and history follows the file);
   plain `mv` for untracked ones.
3. Keep the original basename. Group by kind in a subdirectory
   (`commit_messages/`, `claude_workflow_runs/`, `local_residue/`, …) rather than by date.
4. Before moving, confirm nothing reads the path — grep `*.py`, `*.sh`, `*.md`, `*.js`, and the
   workflow/CI configs — and record that check in the "safe to restore" column.
5. Never delete. If the owner decides something should truly go, that is an explicit owner action,
   not a cleanup pass.
