# research/campaign_2026-08/ — the 2026-08 comprehensive campaign

Ten waves in August 2026 — understand → research → fix → build → self-hunt — producing eight
numbered phase documents. Every model-facing change from the campaign shipped behind a
**default-OFF flag**, with the flag-OFF path byte-pinned by a test; only safety fixes, measurement
instruments, and failure-path hardening went live on ship.

**This directory is never renamed.** More than twenty references point into it from production
code, tests, and `../KILL_LIST.md`.

**Frozen research, live activation.** The eight documents are dated records — their `file.py:NNN`
citations are working-tree snapshots as of each file's own date, and later work moved those lines;
do not "fix" them. The exception is the activation material: `03` + `06` §4 + `08` IA-4 together
are the reference the owner actually works from on the Jetson.

| # | File | What it is |
|---|---|---|
| 01 | `01_state_map.md` | **The defect map.** Campaign ground truth, synthesized 2026-08-18 from 15 subsystem reader reports covering every production module, the tooling and the research corpus. Defect IDs **D01–D40** and build IDs **B01–B24** are assigned here and used by every later phase. |
| 02 | `02_research.md` | **The literature parameters.** 12 round-1 research reports (24 seed topics) plus 8 round-2 branch dives, written so a reader who never saw the source reports can act on it. Every recommendation is tagged **model-facing** (default-OFF flag / owner ruling / challenger-shadow path) or **measurement-only** (ships directly). Cited from `liquidity.py`, `shadow.py`, `validation.py`, `meta_curve.py` and their tests. |
| 03 | `03_jetson_runbook.md` | **The activation sequence — read this before flipping anything.** Owner-facing, phase by phase, with the evidence gate each flag must clear first. Covers the campaign's first-wave flags; `06` §4 and `08` IA-4 are its continuation for the later R2-C and IA-4 flags. |
| 04 | `04_commit_messages.md` | Commit-message drafts for the campaign tree (the edits interleave across ~60 files, so one commit was the honest unit). A historical record of what landed as `20a41db`. |
| 05 | `05_frontier_research.md` | **Frontier research (round 2).** 6 round-1 reports across 12 topics — time-series foundation models, tabular/sequence architectures, objectives and labels, ensembling, cross-sectional ML, anomaly decay, regime detection, nonstationarity, equity and crypto market structure, top-journal economics, LLM signal extraction — plus 6 deep dives. Yields **FR-01..FR-20** candidates and **N1–N10 dated negatives**, all kill-list screened. |
| 06 | `06_signal_model_plan.md` | **The signal-model defects + the 12-step Jetson experiment sequence.** Adjudicated from three independent lens reviews of the signal path (`model_v2.py`, `model_lgb.py`, `blend_fit.py`, `scripts/hypersearch_v2.py`, `predict_now.py`); every defect re-verified line-by-line, read-only. This is the panel that the model modules never got in the 2026-07 campaign (batch B5 never ran). §4's 12 steps are the pre-registered experiment order. |
| 07 | `07_decision_influences.md` | **The decision-influence audit** — "there are many influences to a decision; do all belong, and to what degree?" Explicitly **analysis only**: no code changed, no flag flipped. Every "remove"/"merge"/"retire" is a verdict on paper awaiting an owner ruling and, in nearly every case, Jetson evidence. Cited by `strategy_config.py` and the `tests/test_ia*.py` family. |
| 08 | `08_removed_code.md` | **The removed-code archive.** Every block removed by the influence-implementation wave, preserved verbatim with its original `file:line` span, the `07` verdict that removed it, the rationale, and **restoration instructions**. Cited from 7 production sites and 14 tests — the single most-referenced file in this directory. (A whole removed module would go to `research/code_archive/<name>.py.archived`; none this wave.) |

## Open asks, and one that closed

The runbook's Phase 5 lists four kill-list asks awaiting owner rulings. **Ask (3) — pseudo-CAPE
removal — was RULED and closed on 2026-08-22** and is implemented (`macro_indicators.py` now
always returns `cape=None`; the ruling is recorded in `../KILL_LIST.md`'s PENDING OWNER ASKS
appendix and the removed block is archived in `08_removed_code.md` IA-1.1). The runbook's original
2026-08-20 text is left intact with a dated annotation beside it — history is annotated, not
rewritten. Asks (1), (2) and (4), and the BTC-dominance data-feed ask, remain open.

## Reading order

New to the campaign: `01` (what was wrong) → `02` and `05` (what the literature says) → `06`
(what to do about the signal path) → `03` (how to turn any of it on). `07` and `08` are the
influence-pruning pass and its archive. `04` is history.
