# SIGNAL — ROUND 1 (2026-09-27) — DRAFT (sections A/B adjudicated; C/D/E pending handbacks)

## A. Research (SCOUT-1) — LANDED as a research note
- research/campaign_2026-09_jetson/research_signal.md (252 lines, 16 cited findings; 10 applicable, ranked; conformal left on kill list).
- General's spot-check of the code facts: LGB row cap scripts/hypersearch_v2.py:1287,1360-1368 (120k / byte budget, most-recent rows) — TRUE;
  TPESampler at :2371 relies on Optuna defaults; requirements*.txt pin `optuna>=4.0`, 4.7.0 installed; Optuna 5.0 flips multivariate/constant_liar — OWNER/ops item (pin `<5` or pass kwargs explicitly).
- Top experiments with pre-registered rules: S1-01 LGB lag-subset (weekly-block bootstrap CI(ΔIC)>0 AND blend DSR ≥ A−0.02 AND RSS ≤ A+100 MB; flag LGB_LAG_SUBSET),
  S1-02 DSR-gate power block (measurement-only certificate field; owner remedy if power(SR=0.10)<0.20 on 3/4 certs),
  S1-04 Beta-calibration arm for meta p (paired OOF log-loss CI excl. 0 both books, ECE not worse, veto flip <10%),
  S1-05 LGB-only serving question (w_raw−2se ≤ 0 on 2 retrains AND LGB-only DSR ≥ blend−0.02 AND λ* not lower) — owner item only.

## B. BARS_PER_YEAR (BPY-1) — measurement LANDED, fix STAGED (landing/SIG-R1-BPY)
- Landed directly: scripts/bars_per_year_census.py (measurement-only; pyarrow projection; RSS 175 MB crypto / 492 MB stock) + tests/test_bars_per_year_census_2026_09.py (7 passed, single file).
- MEASURED (objective): stock store = extended session (ET open-times 04–19); bars/ticker-day median 15, mean 13.66 (whole) / 15.185 (trailing 365d);
  pooled bars/yr 3443 whole-store, 3827 trailing year → ratio vs 1638 = 2.336, sqrt 1.528 (the "≈1.57×" in D/G6 assumed 16 bars every day — overstated). RTH-only subset = 7 bars/day = 1764 (1638 counts hours, not bars). Crypto 8751/8766 vs 8760 — fine.
- Blast radius (verified by me): backtest.py:89 dict has ZERO readers (backtest Sharpe/DSR/--gate unaffected); the only real reader is hypersearch_v2.compute_sharpe :654 →
  changes the regime penalty threshold crossing (:1249 `min < -0.5 ⇒ ×0.7`) and the save ratchet vs a best_score on the old scale (:2424-2475) ⇒ gotcha #2; holdout gate :2834 checks only sign(Sharpe) + per-trade DSR ⇒ unchanged.
- Staged patch: new stdlib `bars_calendar.py` (LEGACY/MEASURED tables, flag read at call time via getattr(strategy_config,'BARS_PER_YEAR_MEASURED',False)), hypersearch_v2 :654 + portfolio_backtest DEFAULT_PERIODS_PER_YEAR consult it; OFF path = the verbatim pre-flag expression (tests/test_sig_r1_bpy.py: 5 fail-before / 9 pass-after; existing test_hypersearch_v2 15, test_portfolio_backtest_v3 31, test_portfolio_backtest 9, test_review_b17 24 pass on the patched overlay).
- CROSS-DEPT (report, not made): ENGINE must add `BARS_PER_YEAR_MEASURED` (env TRADER_BARS_PER_YEAR_MEASURED, 1/true/yes idiom as liquidity.py:69) to strategy_config.py; and volatility.py:734 GARCH-path per-bar vol target is 1.53× too high for stocks with 1638 (HAR path cancels via BARS_PER_DAY=6.5 → must move to 15.19 in lockstep or HAR-path sizes drop 1.53×).
- NEW OWNER ITEM (from the census, needs a KILL_LIST/08 check next round): hypersearch's objective scores entries on ALL rows — ~49% of stock rows are pre/post-market — while live entries are RTH + entry windows only (stock_loop.py:125-141, strategy_config.py:113-117). Same family as the short-leg mismatch the CEO just fixed with OBJECTIVE_LONG_ONLY.
