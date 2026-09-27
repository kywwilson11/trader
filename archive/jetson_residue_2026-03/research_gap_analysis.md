# Research Gap Analysis — March 2026

What the research recommends vs what the code actually does, plus new findings.

## Already Implemented (Research Validated)

These items from the research files are already in the codebase and working:

| Recommendation | Status | File |
|---|---|---|
| Kelly criterion position sizing | Half-Kelly implemented | trading_utils.py |
| GARCH(1,1) volatility targeting | Implemented, cached 1hr | volatility.py |
| HMM regime detection | Implemented (hmmlearn) | regime_detector.py |
| Correlation-aware sizing (Markowitz) | Implemented | portfolio.py |
| VIX/STLFSI2/CAPE macro regime | Implemented | macro_indicators.py |
| Stablecoin peg monitoring | Implemented (emergency flatten) | macro_indicators.py |
| Hurst exponent feature | Implemented (rolling 100-bar) | indicators.py |
| Calendar features (month, day-of-week) | Implemented (sin/cos encoding) | indicators.py |
| Transaction cost in Sharpe (5bps) | Implemented in backtester | hypersearch_v2.py |
| Regime-aware backtesting | Fold Sharpe computed per regime | hypersearch_v2.py |
| LightGBM ensemble stacking | Implemented | model_lgb.py |
| Template Method (DRY trading loops) | Implemented | base_loop.py |
| Spread-aware order pricing | Implemented | order_utils.py |
| Circuit breaker (5% daily drawdown) | Implemented | order_utils.py |
| Winner's curse filter | Implemented (SMA20 + 2*ATR) | base_loop.py |
| Sentiment gate + LLM gate | Implemented | sentiment.py, llm_analyst.py |
| FP16 mixed precision training | Implemented | hypersearch_v2.py |
| Walk-forward CV (3-fold expanding) | Implemented | hypersearch_v2.py |
| ATR-based trailing stops | Implemented (1.5x ATR) | base_loop.py |
| Hard stop with 24hr lockout | Implemented, now persisted | base_loop.py |
| Dataclasses for Position/Quote | Implemented | types_mod.py |
| Atomic file writes | Implemented | trade_memory.py, run_pipeline.py |

## Gaps — High Priority (Clear research support, significant expected impact)

### 1. Hybrid Kelly-VIX Sizing
**Research**: arXiv 2508.16598 (2025) — scale Kelly fraction by VIX regime
**Current**: Kelly fraction is fixed (half-Kelly). GARCH and macro regime scale the RESULT but not the Kelly fraction itself.
**Gap**: Kelly fraction should shrink in high-vol regimes. Currently Kelly says "bet 10%" regardless of VIX — then macro scales it down. Better to have Kelly itself be regime-aware.
**Fix**: In `kelly_position_size()`, multiply fraction by VIX-based scaler (0.5 calm → 0.15 crisis).

### 2. Drawdown-Based Position Scaling
**Research**: Risk of ruin literature — reduce size during drawdowns, not just after circuit breaker.
**Current**: Circuit breaker at 5% daily drawdown → flatten all. Nothing between "normal" and "emergency flatten."
**Gap**: No gradual scaling. Account drops 3% intraday and bot trades at full size until 5% triggers.
**Fix**: Track peak equity. At 10% drawdown from peak → reduce all sizing 50%. At 15% → reduce 75%. At 20% → halt new entries.

### 3. Combinatorial Purged Cross-Validation (CPCV)
**Research**: de Prado (2018), confirmed by 2024 study as "markedly superior" for detecting overfitting.
**Current**: Walk-forward 3-fold expanding window with embargo.
**Gap**: Walk-forward is good but doesn't measure Probability of Backtest Overfitting (PBO). CPCV gives multiple backtest paths and quantifies overfitting risk.
**Fix**: Add PBO computation after Optuna search completes. If PBO > 0.5, flag the model as potentially overfit.

### 4. Stop Loss Width
**Research**: Trailing stops at 15-20% or 2-2.5 ATR improve returns. Very tight stops (3-5%) hurt via whipsaw.
**Current**: ATR_STOP_FLOOR_PCT = 3%, ATR_STOP_MULTIPLIER = 2.0, trail = 1.5x ATR. Floor of 3% is too tight per research.
**Gap**: The 3% floor is aggressive. Research says wider is better for hourly bars.
**Fix**: Consider raising ATR_STOP_FLOOR_PCT to 5% and ATR_TRAIL_MULTIPLIER to 2.0. Make these searchable in Optuna.

### 5. Re-entry Rules After Stops
**Research**: Stops without re-entry rules just lock in losses. Critical for profitability.
**Current**: 24-hour hard stop lockout. No re-entry logic — just waits 24h then treats symbol as fresh.
**Gap**: After a stop-out, the bot should track whether the thesis still holds (prediction still positive, fundamentals unchanged). If so, re-entry at a better price is the right move.
**Fix**: After hard stop lockout expires, require prediction > 1.5x threshold for re-entry (higher bar).

### 6. Separate Crypto vs Stock Parameters
**Research**: Crypto has higher momentum persistence, wider spreads, different volatility structure. Bitcoin ETFs reduced BTC volatility 55%.
**Current**: Same ATR multipliers, same stop floors, same cooldowns for both.
**Gap**: Crypto should have wider stops, longer momentum lookback, different threshold sensitivity.
**Fix**: Make stop parameters, cooldown, and threshold multipliers asset-type-specific in the config.

## Gaps — Medium Priority (Good evidence, moderate complexity)

### 7. Ledoit-Wolf Shrinkage for Correlation Matrix
**Research**: Nobel (Markowitz) + practical: raw correlation matrices are error-prone with limited data.
**Current**: `np.corrcoef()` on raw returns in portfolio.py.
**Gap**: Raw correlation is noisy. Ledoit-Wolf shrinkage is the standard fix.
**Fix**: `from sklearn.covariance import LedoitWolf` — drop-in replacement.

### 8. EGARCH for Asymmetric Volatility
**Research**: Engle (Nobel 2003) — negative returns increase vol more than positive returns.
**Current**: GARCH(1,1) — symmetric, treats +5% and -5% identically.
**Gap**: In practice, crashes increase vol more than rallies. EGARCH captures this.
**Fix**: `arch` package already supports EGARCH. Change model spec in volatility.py.

### 9. Hurst-Gated Signal Selection
**Research**: Hurst < 0.5 → mean-reversion works. Hurst > 0.5 → momentum works.
**Current**: Hurst is a feature for the LSTM but not used to gate which signals to trust.
**Gap**: When Hurst < 0.5, momentum-based entries are fighting mean reversion.
**Fix**: When Hurst < 0.45, require higher prediction threshold (market is mean-reverting, trend signals are less reliable).

### 10. Monte Carlo Robustness Test
**Research**: Reshuffle trade returns 10k times, compute distribution of outcomes.
**Current**: No Monte Carlo validation.
**Gap**: No way to know if Sharpe of 10 is robust or luck.
**Fix**: After Optuna completes, run MC simulation on validation returns. Report confidence interval.

### 11. Jump Penalty in HMM Transitions
**Research**: Statistical Jump Model (2024) — reduces false regime switches.
**Current**: Standard HMM with no transition penalty.
**Gap**: HMM may flip between bull/bear too frequently, causing position sizing whipsaw.
**Fix**: Add transition cost to regime_detector.py — require N consecutive bars in new regime before switching.

### 12. Ensemble Regime Voting
**Research**: Ensemble HMM + ML voting (2025) — better than either alone.
**Current**: HMM regime, GARCH sigma, and macro regime are applied independently as separate multipliers.
**Gap**: No joint regime decision. Could disagree (HMM says bull, macro says crisis).
**Fix**: Combine into single regime vote. When they disagree, reduce sizing (uncertainty).

## Gaps — Lower Priority (Interesting but complex or uncertain impact)

### 13. Temporal Fusion Transformer
**Research**: 40-50% MAE improvement over LSTM. Best for multi-horizon prediction.
**Current**: LSTM + attention + LightGBM ensemble.
**Gap**: TFT would be better but requires significant architectural change.
**Timeline**: Next major version. LAMFormer is a lighter alternative for Jetson.

### 14. Online Bayesian Change-Point Detection (BOCPD)
**Research**: Doesn't require pre-specifying regime count (unlike HMM).
**Current**: HMM with fixed states.
**Gap**: Novel regimes (unprecedented events) won't be detected by fixed-state HMM.
**Fix**: Add as complementary signal. Package: bayesian_changepoint_detection.

### 15. Factor Regression (Fama-French)
**Research**: Understand if alpha is real or just factor exposure.
**Current**: No factor decomposition.
**Gap**: Don't know if the bot's returns are from skill or from being long small-cap momentum.
**Fix**: Post-trade analysis tool. Not critical for live trading.

### 16. Anti-Martingale Scaling
**Research**: Increase position size after wins, decrease after losses (aligns with Kelly math).
**Current**: Position sizing based on current prediction and risk factors, no streak awareness.
**Gap**: After a losing streak, the bot trades at the same size. After winning, same size.
**Fix**: Track recent win/loss ratio in rolling window. Scale sizing by recent hit rate.

## Key Numbers from Research

- **Most published Sharpe > 1.5 results are overfit** (2025 walk-forward study with costs: Sharpe 0.33)
- **Half-Kelly gives 75% of full Kelly return with 50% variance**
- **Trailing stops at 2-2.5 ATR optimal** (current bot: 1.5x ATR trail, 2.0x hard stop)
- **CPCV is "markedly superior" to standard k-fold** for detecting overfitting
- **Bitcoin ETFs reduced BTC volatility by 55%** and concentrated liquidity in US hours
- **Raw OHLCV features can outperform technical indicators** for ML (counterintuitive)
- **RSI showed "inconsistent or minimal impacts"** as an ML feature (but may help as gate)
- **Spread costs can destroy sub-1% edge strategies** — spread-aware ordering is critical
- **3% stops are too tight** — research favors 5%+ for hourly bars
