# Jetson Activation Runbook — 2026-08 Comprehensive Campaign

**Audience:** the owner, running on the Jetson (prod). **Nothing below is active yet** — every
model-facing change in the campaign shipped behind a default-OFF flag; only safety fixes,
measurement instruments, and failure-path hardening are live on ship. The dev-Mac suite gates
this tree at ~3,376 passed / 23-name baseline (`bash scripts/ab_check.sh`).

**Rules that govern this document**
- Flip order matters. Several flips are **evidence-gated**: the instrument that produces the
  evidence ships first and the flip happens only after you read its output.
- **Gotcha #2 events** (objective/feature/cost changes) are bundled into ONE study-reset retrain
  each — never flip those piecemeal.
- Flags live in two places by campaign convention: `strategy_config.py` constants (edit the file)
  and `TRADER_*` environment variables (set in the service env). All default OFF.

---

## Phase 0 — Commit, pull, baseline reads (no flips)

1. Review + commit the campaign tree on the Mac (commit-message drafts are in the campaign
   report). Pull on the Jetson.
2. `python -c "import bidask"` — if absent, STOP before any harvest: every stamped
   `Eff_Spread_Pct` so far came from the AR fallback and installing bidask mid-stream is a
   model-facing change (see 01_state_map D04 notes).
3. `bash scripts/ab_check.sh` on the Jetson — expect green (full-dep suite; the 23-name baseline
   is dev-Mac-only and must NOT be ported).
4. Restart the bots. Verify in logs, first hour:
   - stock predictions non-null (`grep -c "Not enough data for sequence" stock_bot_output.log`
     BEFORE restart to document the outage span — standing P0);
   - `[DAILY-FEATURES]` line reports how many live stock feature columns are warmup-filled
     constants (expected ~13 until the restore flip);
   - `[VOL]` sigma-source counters show `garch_fallback` for every symbol (expected until the
     HAR flip);
   - per-book flatten flags respond (`flatten_crypto.flag` / `flatten_stock.flag`);
   - the legacy-vs-calendar effective-n side-by-side line prints during the next weekly gate.
5. One-look measurement reads (all direct-shipped, no flips needed):
   - `python beta_ledger.py --days 90 --json beta_report.json` — read `period.obs_per_year_grid`
     (settles 252-vs-grid), `contamination_delta` (deposit fabrication size),
     `alpha_t_corrected`, and the `trend_conditional_lagged` block.
   - `python decision_report.py --days 30` — verdict-first output; note the banner: all
     pre-campaign decision_report figures are void.
   - `python llm_eval.py --days 30` — the NEW Driscoll-Kraay estimator is the primary b2 verdict;
     `legacy_b2` prints alongside for this release. The spend ledger shows implied bps/trade vs
     actual dollars. At n≥60 clustered timestamps this is the keep/kill-LLM-spend read.
   - `backtest_report.json` → `n_eff_clustered` vs `n_trades` (the D02 one-look: a collapse to
     <n/5 means the legacy clustering, not the model, decides promotions).

## Phase 1 — Instruments to run once (direct, no flips)

- `python scripts/sizing_cofire_report.py` — the B7 answer: per-multiplier bind rates, co-fire
  matrix, worst composed product, and the v2-shadow comparison (already journaling on every entry).
- The Stage-0 dump lands automatically on the next weekly backtest as `{slot}_stage0_preds.json`.
  Then: `python scripts/ic_by_name.py --in stage0_preds.json --time-key ts` and
  `python scripts/rank_gradient_report.py --preds stage0_preds.json --fwd-bars 1
  --cost-pct <fees> --extra-cols meta_p,pred_thresh_ratio` — these are the wave-9 activation
  gates for breadth / concentration / edge-Kelly.
- `python scripts/meta_learning_curve.py` then `--prefix stock` (~minutes) — read
  `floor.honest_floor` per book; required before the META_OOF_PRED flip.
- `python scripts/crypto_spread_census.py` (~a day of background polling) — required before any
  crypto spread-stamp decision; also answers the kill-list ask with data.
- `python scripts/llm_qualify.py` then `--shadow` for ~a week — free-endpoint verdicts +
  score-agreement vs the production analyst.
- `python scripts/train_lexicon.py` (uses `sentiment_cache.db` + stock parquet) — the learned
  lexicon's IC verdict vs the static lexicon and the LLM (dark artifact; expect possibly
  "no incremental IC" — that is a valid, useful verdict).
- GUI visual pass: 10 themes × the new panels (verdict gate box, freshness strip, meta panel,
  dm_v2/policy-gate segments, sizing v2-beside-legacy).

## Phase 2 — Evidence-gated flips (each independent; do in this order)

1. **`PROMOTION_GATE_V2 = True`** (strategy_config) after reading the side-by-side n_eff logs
   from ≥1 weekly cycle. Optionally `KISH_NEFF_ENABLED=True` for the stock book. From this flip
   on, the gate fails closed below 10 effective trades and prints MinTRL on failures.
2. **`GATE_TARGETS_CHALLENGER = True`** after one shadow-mode weekly build logs the D03 warning
   and a challenger cycle writes `{slot}_policy_gate.json`. Gate failures now HOLD the
   challenger; the champion is never rolled back by a challenger gate.
3. **`TRADER_SHADOW_DM_V2=1`** (env) after one full cycle of paired `dm_v2_*` fields in
   `shadow_status.json` agrees directionally with expectations. Stock shadows under v2 need
   ~56 days (or promote-at-max-duration).
4. **Calibration chain — strict order:**
   a. `CALIBRATION_V2 = True` only after `python scripts/reliability_report.py` before/after
      shows the tie-fix/Platt changes are sane;
   b. then `META_CALIBRATION_MODE='purged_oof'` (wave-9 activation #1 — only interpretable
      AFTER the tie fix);
   c. then `META_OOF_PRED = True` only after (i) `meta_curve_report.json` says the row count
      survives the ~55-60% OOF cut above `honest_floor`, and (ii) an honest-holdout veto-precision
      A/B (p<0.30) ≥ the leaked variant;
   d. `META_REPLAY_POLICY_PARITY = True` after reviewing the always-on rows_legacy/rows_parity
      drop counters (population shift visibility first).
5. **`DERISK_STACK_V2 = True`** after the cofire report + accumulated v2-shadow journal rows
   show the expected recovery (modal entries → ~1.0×) with hard floors intact. Delete
   `funding_history.json` once when also enabling **`TRADER_FUNDING_Z_TIME_THINNING=1`**.
6. **Execution set:** `TRADER_STOP_CLASSIFY_V2=1` after journals show the hard/trail split
   (`server_stop_kind` counts); `TRADER_MAKER_SHARE_NOTIONAL=1` after the notional split
   accrues; `TRADER_IOC_ENTRY_CAP=1` only with a paper A/B week (fill-rate vs slippage);
   `TRADER_STREAM_STOP_DETECT=1` only where `TRADER_ORDER_STREAM` is already on.
7. **`EVENTS_TRADING_DAY_WINDOWS = True`** — only ever blocks MORE (Friday→Monday earnings
   protection). Low ceremony.
8. **`TRADER_DAILY_FEATURE_RESTORE=1`** ONLY after the bit-parity check: compute the daily
   blocks via the live path and the harvest path over the same window and diff — parity holds ⇒
   flip (no retrain needed; semantics-restoring); any mismatch ⇒ STOP and report (the known
   hazard is Alpaca daily-bar official close vs harvest `resample('1D')` semantics).
9. **`TRADER_HAR_DAILY_FEED=1`** — gap: the offline QLIKE comparison script was scoped out;
   until built, gate this flip on (a) the `[VOL]` har-vs-garch counters staying stable for a
   week with the flag on in a SHADOW sense is not possible (it changes sizing directly), so
   either build the QLIKE script first or accept the wave-4 certification + the sizing-journal
   before/after as the evidence. Deliberately conservative: do this AFTER DERISK_STACK_V2 so
   vol-scope ownership is already settled.
10. **`TRADER_CLOSED_BARS_V2=1`** — changes live feature/sizing values → challenger/shadow
    path only, not a plain flip.

## Phase 3 — THE bundled retrain (ONE gotcha-#2 event)

Flip together, then delete `v2_study.db` + `stock_v2_study.db`, reset the adaptive
`best_score`, and reset `cum_trials` (the B-1 sanctioned reset):

- `HYPERSEARCH_V3 = True` (final refit, LGB-before-gate, blend fit → `lstm_weight`, blend-level
  holdout certification)
- `OBJECTIVE_V3 = True` (per-ticker reset, cost-anchored threshold range — the study reset is
  MANDATORY: the Optuna distribution changes)
- Owner-optional in the same event: `OBJECTIVE_LONG_ONLY = True`, `--preset stationary_lean`
- Data-store event (same retrain or its own): `TRADER_RAW_SIDECAR=1` + `TRADER_YF_WINDOW_SLICE=1`
  with the sidecar absent ⇒ forced full refetch rebuilds the D39-lost head and kills the D08
  Yahoo overwrite
- Cost flags IF pre-gates passed: `TRADER_SPREAD_FILL_V2=1`; `TRADER_CRYPTO_SPREAD_STAMP=1`
  only after the census AND the explicit kill-list ruling (KILL_LIST:90 ask);
  `TRADER_STOCK_MINUTE_EDGE=1` after confirming Basic-plan minute bars are SIP-sourced;
  `TRADER_COST_REGIME_FEATURES=1` — **prerequisite gap:** live-serve injection parity in
  predict_now is an unresolved handoff; do not flip until that lands.

After the retrain: the challenger passes the (now-honest) holdout blend gate → the
challenger-targeted policy gate → the v2 shadow test. That chain IS the promotion path.

## Phase 4 — LLM economics

1. Read the llm_eval spend ledger verdict (Phase 0.5). If "keep":
2. `llm_qualify_report.json` verdict `qualified` + a week of shadow agreement ⇒ **prerequisite:**
   add budget rows for the qualified model ids in `llm_client`'s budget tables (hardcoded — the
   50-RPD unknown-model default exhausts a 288-call/day analyst mid-morning; report's
   `live_budget_note`), then flip `selection_mode` to free-first in `llm_config.json`.
3. Enable the analyst dedup cache (config TTL ~1800s) — journals now mark `dedup_hit` so
   llm_eval collapses re-serves correctly. Prompt caching on Anthropic is already live in the
   client (watch `cache_read_input_tokens` in the cost lines).

## Phase 5 — Strategic follow-ons (owner decisions, evidence in hand by now)

- **SPY hedge (B20)**: with clean betas (`alpha_*_clean`, `trend_conditional_lagged`,
  down-semibeta idea) — size the trend-conditional ETB short; challenger/shadow discipline.
- **Crypto breadth (B17)**: after honest crypto costs land — CRYPTO_POOL 6→10 with as-of
  membership; needs the Stage-0 rank-gradient evidence.
- **Wave-8 activations (B23)**: UNIQUENESS_WEIGHTS_ENABLED, IMPACT_COST_ENABLED (+ DV30 stamp,
  now vol-scale under TRADER_IMPACT_VOLSCALE), PREDICTION_CACHE_ENABLED (after memo-key
  extension), CSCV-PBO warn field, GATE-1 realized correlation.
- **Kill-list asks (owner rulings, full text in the campaign report):** (1) re-open ONLY the
  minute-bar half of the EDGE-inflation kill; (2) bless hedge-trigger vs short-alpha-trigger
  distinction (new commonly-confused-survivors entry); (3) rule on pseudo-CAPE removal when
  adjudicating DERISK_STACK_V2 (min-aggregation does NOT launder it); (4) split the BTC-spillover
  survivor-#10 ruling three ways (alt-alpha lags: NO; crypto_trend gate: wire per B23;
  contemporaneous BTC context columns: preset repair).
- **Data-feed ask:** BTC-dominance regime input — blocked on your new-data-dependency ruling.

---

*Generated by the 2026-08 comprehensive campaign (Waves A, B-1/2/3, C-1/2X/3, D-1, V, X).
Source docs: `01_state_map.md` (defect map), `02_research.md` (parameters + citations),
the campaign report (owner-decision ledger + commit drafts).*
