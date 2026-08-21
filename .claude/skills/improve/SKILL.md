---
name: improve
description: Run a module-improvement campaign with the proven group-improve-v2 workflow (tiered Design with evidence packs -> Sonnet implement -> Fable verify with one repair loop -> Haiku commit message). Use when the user asks for a repo-wide or multi-module improvement/cleanup campaign.
---

# /improve — group improvement campaigns

The repeatable 4-phase pipeline that produced the committed 2026-07 campaign (36/36 agents,
12/12 groups approved). The orchestration lives at `.claude/workflows/group-improve-v2.js`;
invoke it as the `group-improve-v2` workflow. This skill is the operating manual.

## Phases (per group)
1. **Design+Scout** — ONE read pass. Tier A groups get Fable, tier B Opus. Emits: a spec
   (behavior-neutral changes + unambiguous bug fixes ONLY), an EVIDENCE PACK (verbatim anchor
   snippets so the implementer barely reads files), and deferred_to_owner (every model-facing
   idea, with a fix sketch). Tier C = one agent designs AND implements; Sonnet verifies.
2. **Implement** — Sonnet, fed spec+evidence inline; STRICT ownership (the group's modules +
   its tests/test_grp_*.py); self-verifies with py_compile + the group's tests.
3. **Verify** — Fable reads diff+report only; polishes, or ONE guided repair round, or reverts.
4. **CommitMsg** — Haiku writes `commit_msg.txt` (the USER commits; agents never do).

Worker pool (default 3) runs group chains concurrently; no round barriers.

## Args contract (what the script reads)
- `groups_path` (or inline `groups`) — REQUIRED. JSON config:
  `{ "campaign_title", "baseline_note", "tree_note",
     "groups": [ { "id", "tier": "A|B|C", "mods": "space-separated files",
                   "test": "tests/test_grp_X.py", "seed": "known items + safe/defer hints",
                   "reuse": "spec|summary (optional)" } ] }`
  Template: `.claude/workflows/groups-2026-07.example.json` (the completed campaign).
- `repo` — repo root (default /Users/kywwilson/Desktop/Projects/trader).
- `reuse_path` — JSON file holding paid prior designs; a group with `reuse: "spec"` reads
  `.reuse.<id>.spec` from it, `reuse: "summary"` reads `.reuse.<id>.summary`.
- `round1_lines` — commit lines from an earlier round to merge into commit_msg.txt.
- `workers` — concurrent chains (default 3).

## Rules baked into the prompts (never relax them)
- MODEL-FACING changes (feature values, thresholds, entry/exit/sizing semantics, model
  artifacts) are never implemented — they go to deferred_to_owner with a sketch.
- policy_exits.exit_walk is sacred; heavy modules are READ, never imported on the Mac.
- Regressions are judged by NEW failure NAMES in owned files (see /regression-ab).
- Nothing is committed; the deliverable is commit_msg.txt for user review.

## Before launching
1. Build a fresh groups JSON (seeds from research/module_review_2026-07.json ledger entries +
   session-state; do NOT recycle stale seeds blindly).
2. Confirm tree state with the user -> tree_note; confirm the suite baseline -> baseline_note.
3. After completion: run /regression-ab, then hand the user commit_msg.txt.
