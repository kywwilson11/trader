# Comprehensive Code Review — March 2026

## Module Compatibility: CLEAN
No incompatibilities found. All imports resolve, function signatures match, dataclass definitions consistent.

---

## Critical & High Priority Fixes

### 1. RISK: Quote None check missing in sell logic
**File:** `base_loop.py:531-559`
`_execute_sells()` calls `self.get_quote(symbol)` but doesn't check for None before accessing `quote['midpoint']`. Will KeyError on API failure.
**Fix:** Add `if quote is None: continue` after `get_quote()`.

### 2. RISK: Benchmark None not validated
**File:** `base_loop.py:182-183`
`get_benchmark_close()` can return None, passed to prediction without validation.
**Fix:** Guard with fallback or skip cycle.

### 3. RISK: Hard stop lockout not persisted across restarts
**File:** `base_loop.py:632-640`
`hard_stop_lockout` dict lost on restart — symbol that hit hard stop could be re-entered.
**Fix:** Persist to JSON like trade_memory.

### 4. RISK: Confidence scaling fragile if trade_threshold near zero
**File:** `base_loop.py:578-581`
`pred_return / self.trade_threshold` crashes on zero.
**Fix:** Guard `if self.trade_threshold > 0.001`.

### 5. RISK: Max exposure cap not verified post-fill
**File:** `stock_loop.py:303-314`
Exposure checked before order, never after fill. Concurrent fills can exceed cap.
**Fix:** Re-check exposure after fill confirmation.

### 6. PIPELINE: IPC file race condition (TOCTOU)
**File:** `run_pipeline.py:57-95`
`retrain_trigger.json` and `pipeline_command.json` read then deleted — another process could read between.
**Fix:** Atomic rename-then-read-then-delete.

### 7. PIPELINE: Unprotected global state
**File:** `run_pipeline.py:239-681`
`_suspend_requested`, `_manually_stopped` modified across threads without locks.
**Fix:** Use `threading.Lock()` around all mutations.

### 8. PIPELINE: Zombie processes on signal
**File:** `run_pipeline.py:684-702`
Signal handler terminates child processes but doesn't wait — creates zombies.
**Fix:** Add `proc.wait(timeout=5)` after terminate.

### 9. DATA: Deduplication logic inconsistency
**File:** `data_sources.py:180` vs `data_utils.py:128`
`fetch_with_fallback()` uses `keep='first'`, `append_ticker_data()` uses `keep='last'`. Could overwrite good data with stale during incremental harvest.
**Fix:** Standardize on `keep='last'` (newest wins).

### 10. DATA: Timezone handling gap in data source merging
**File:** `data_sources.py:167-169`
CryptoCompare returns UTC but no explicit tz normalization before merge with Alpaca/yfinance.
**Fix:** Normalize all frames to UTC before concat.

### 11. LLM: Race condition in quota reset
**File:** `llm_client.py:315-327`
`_daily_cost` modified inside lock but read outside lock in `get_recommended_model()`.
**Fix:** Read under same lock.

### 12. LLM: 429 Retry-After parsing incomplete
**File:** `llm_client.py:273-284`
Unparseable 429 responses get ignored instead of backed off.
**Fix:** Default to exponential backoff if regex fails.

---

## Performance Quick Wins

### P1. Remove redundant `.clone()` on state dict
**File:** `scripts/hypersearch_v2.py:624`
`.cpu()` already copies — `.clone()` doubles the memory spike (~20-30MB).
**Fix:** `best_state = {k: v.cpu() for k, v in model.state_dict().items()}`

### P2. Remove excessive `gc.collect()` from hot paths
**File:** `base_loop.py:441`, `hypersearch_v2.py` (6 locations)
Each call takes 50-200ms. In trading loop, adds 50-100ms/cycle.
**Fix:** Only keep in OOM recovery and between major phases.

### P3. Cache LEVERAGED_ETFS import at class init
**File:** `base_loop.py:625`
`from stock_config import LEVERAGED_ETFS` called per symbol per cycle.
**Fix:** Cache at `__init__`.

### P4. Cache LLM analysis JSON load
**File:** `base_loop.py:252, 476-501`
`load_analysis()` reads/parses JSON from disk every 30-second cycle.
**Fix:** Cache with 60s TTL.

### P5. Trade memory — cache + batch writes
**File:** `trade_memory.py:16-76`
Full JSON read+write on every trade (~50-100ms each).
**Fix:** In-memory cache with 30s write-back.

### P6. Pipeline status write throttle too aggressive
**File:** `run_pipeline.py:114-132`
Writes every 2 seconds = 600-1200 writes/hour (SD card wear on Jetson).
**Fix:** Increase throttle to 5-10s.

### P7. Double sorting in hypersearch data loading
**File:** `scripts/hypersearch_v2.py:166, 178`
Data sorted twice (in capping loop and feature extraction loop).
**Fix:** Sort once before loops.

### P8. Parquet column projection not used
**File:** `data_utils.py:35-72`
Loads all columns from Parquet even when subset needed. Wastes 200-500MB.
**Fix:** Pass `columns=` parameter to `pd.read_parquet()`.

---

## Code Organization Issues

### Large files that should be split:
- `gui.py` (4,290 lines) — extract into tab modules
- `sentiment.py` (1,215 lines) — split fetcher/scorer/history
- `run_pipeline.py` (1,146 lines) — extract process management

### Logging inconsistency:
25 files use `print()`, 13 use `get_logger()`. Should standardize on logger.

### Scattered config/constants:
Risk parameters (ATR multipliers, stop thresholds, Kelly fraction, LLM veto threshold) are hardcoded across base_loop.py, trading_utils.py, macro_indicators.py, volatility.py.

### Duplicate functions:
- `sentiment._score_text()` vs `sentiment_history._keyword_score()` — slightly different keyword scoring
- `indicators.compute_features()` vs `indicators.compute_stock_features()` — no unified interface
- `market_data.fetch_bars_alpaca()` vs `market_data.fetch_stock_bars_alpaca()` — separate but similar

### Missing test coverage:
- base_loop.py, crypto_loop.py, stock_loop.py (core trading logic)
- portfolio.py, regime_detector.py, volatility.py (risk management)

### No unified abstractions for:
- Caching (5 different TTL-cache implementations)
- Error handling (inconsistent across modules)
- Config loading (5 separate loaders)

---

## GARCH cache staleness
**File:** `volatility.py:102-107`
1-hour cache. During volatile events, stale forecast could be dangerous.
**Fix:** Log cache age, warn if stale during high-vol.

## Position dict race condition
**File:** `base_loop.py:393-406`
`del self.positions[symbol]` without checking key exists first.
**Fix:** `if symbol in self.positions: del self.positions[symbol]`

## GUI thread safety
**File:** `gui.py:3280-3330`
`on_positions()` slot modifies Qt table widgets from worker thread.
**Fix:** Marshal to main thread via `QMetaObject.invokeMethod`.

## GUI blocking API calls
**File:** `gui.py:1971-2024`
`self.api.submit_order()` blocks main GUI thread.
**Fix:** Move to worker thread.
