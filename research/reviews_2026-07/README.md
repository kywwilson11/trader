# research/reviews_2026-07/ — the 2026-07 review corpus · FROZEN

Two review programs ran in July 2026 and left their records here: the **module-improve-v3 panel
campaign** (deep, per-module, N independent Opus reviewers each) and the **GUI review**.

**Frozen.** These are dated records. Their `file.py:NNN` citations are working-tree snapshots from
July 2026; the 2026-08 campaign edited many of those files afterwards. Read them for the reasoning
and the verdicts, not as current line pointers.

> The **90-item owner decision queue** from the separate 2026-07 *module review* (69 modules,
> 600 functions) is **not** here — it is live tool input and stays flat at
> `../module_review_2026-07.json`. Render it with `/decision-queue`. Queue items are owner
> decisions; do not auto-fix them.

## The module-improve-v3 panel campaign

`panel_campaign_plan.md` is the plan and the running status board: batches of **3 modules × 5 Opus
reviewers** (~29 agents per batch), each batch ending in a serialized full-suite `ab_check`, run
strictly sequentially. Each batch produced a report:

| Batch | Modules | Report |
|---|---|---|
| 0 — measurement contracts | `beta_ledger`, `decision_report`, `llm_eval` | `module_improve_v3_report.md` |
| A — live engine & order path | `base_loop`, `order_utils`, `execution_policy`, `risk_budget`, `bet_sizing` | `module_improve_v3_batchA.md` |
| B1 — validation core | `validation`, `sample_weights`, `calibration` | `module_improve_v3_batchB1.md` |
| B2 — promotion gate | `meta_label`, `backtest`, `policy_exits` | `module_improve_v3_batchB2.md` |
| B3 — cost model | `fees`, `liquidity`, `cost_regime` | `module_improve_v3_batchB3.md` |
| B4 — portfolio risk | `portfolio`, `portfolio_backtest` (+ `drawdown`, which died mid-run) | `module_improve_v3_batchB4.md` |

**Batch B5 never ran.** The plan's status board still shows B5 (`model_v2`, `model_lgb`,
`blend_fit`) as "running" and B6–B11 as "queued" — none of them executed. There is no
`module_improve_v3_batchB5.md` and no `tests/test_{model_v2,model_lgb,blend_fit}_v3.py`. The
signal-path modules that B5 would have covered were instead reviewed through a different lens by
the 2026-08 campaign — see `../campaign_2026-08/06_signal_model_plan.md`, which opens by noting
"the panel that model_v2.py and model_lgb.py never received". Treat the plan as **CLOSED at B4**.

Each report follows the same shape: raw findings per module, what was accepted and shipped, what
was rejected and why, and the residue promoted to **owner-decision items** rather than auto-fixed
(anything that changes estimator definitions, sample semantics, or cross-file / model-facing
behavior).

## The GUI review

`gui_review_2026-07.md` — a full review of `gui.py` and its data sources, organized as six clusters
plus an §11 phased roadmap. All six phases were subsequently implemented. It is the origin
document for the pure GUI-support modules that now cite it in their docstrings (`design_tokens.py`,
`journal_stats.py`, `tax_lots.py`) and for the shadow-status producer pinned by
`tests/test_shadow_status_persist.py`. Its recurring theme — "data produced and never read" — is
still the best inventory of instrumentation the GUI does not yet surface.

## Before acting on anything here

Read `../KILL_LIST.md` first, including its PENDING OWNER ASKS appendix. Several findings in these
reports deliberately stop at "owner must rule" precisely because they brush against a killed item.
