---
name: decision-queue
description: Render the open owner-decision items from the completed 2026-07 module review (research/module_review_2026-07.json, 90 items — 1 P0, 21 P1, 68 P2 across 41 modules). Use when asked what is open, what to prioritize, or before planning model-facing work.
---

# /decision-queue — module-review owner decision queue

The 2026-07 review campaign is **DONE** (69 modules, 600 functions, 280 safe fixes
auto-applied). What remains is a 90-item queue of **deliberately deferred** items —
model-facing or behavior-changing, so they are OWNER decisions, not a to-do list.
**Never auto-implement a queue item.**

## Render it
```bash
python3 .claude/skills/decision-queue/render.py                 # all, grouped P0 -> P2
python3 .claude/skills/decision-queue/render.py --severity P1   # one severity
python3 .claude/skills/decision-queue/render.py --module base_loop
python3 .claude/skills/decision-queue/render.py --full          # + where/fix_sketch
```
Run from the repo root (paths are anchored, so any cwd works).

## Reading the output
- Each line: [severity] module: description. `--full` adds the exact code location
  (`where`) and the reviewer's `fix_sketch`.
- The single P0 (events_calendar trading-day windows) has a placeholder desc in the
  ledger; its full spec (np.busday_count) lives in the session-state memory and the
  campaign journals.
- When the user picks an item: follow its fix_sketch, and remember model-facing changes
  ship ONLY via the challenger -> shadow -> DM-HLN gate (see CLAUDE.md Conventions).
- Wave memory files hold each domain's KILL list — check them before proposing
  alternatives research already rejected.
