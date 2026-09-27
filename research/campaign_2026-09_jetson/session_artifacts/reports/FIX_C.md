# FIX_C: stock SIP end-clamp (C_data blocker 1)

Agent FIX_C, 2026-09-26/27, on the Jetson. I edited **only** `market_data.py` and added
`tests/test_market_data_sip_clamp_2026_09.py`. `data_sources.py` needed no change. Nothing was
committed or staged. There were no harvest writes and no store writes, and every API call was a
read-only bar fetch. The pre-fix copy is saved at `scratchpad/market_data.py.orig_fixC` (also
`scratchpad/fixC_orig/market_data.py`).

## Diff summary (market_data.py, +52/-3)
- `import logging` and a module logger `log = logging.getLogger(__name__)` (`market_data.py:8,21`).
- New constant **`SIP_RECENT_DELAY_MIN = 16`** (`market_data.py:548`). The 15-min Basic-plan SIP
  delay plus 1 min of margin, with a comment block citing the defect.
- New helper `_clamp_sip_end(end_dt, asset_type, now=None)` (`:551-563`):
  - crypto: returned unchanged;
  - stock: `min(end_dt, now - 16 min)`;
  - an `end` already at or before the clamp: returned unchanged, so bounded gap-repair windows
    keep their exact bounds.
  - The stock/crypto test is `asset_type == 'crypto'`, the same one `_fetch_chunk` uses.
- `fetch_historical_bars` applies the clamp right after `end_dt` is resolved (`:646-649`), so the
  chunk builder's final chunk ends at `now-16min` for stocks. This also covers the harvest's
  sidecar gap-repair caller (`scripts/harvest_stock_data.py:473`), which passes `end_date`; in
  that case the clamp is a no-op unless the window reaches the last 16 min.
- Honest failure path:
  - `_fetch_chunk`'s subscription branch (`:599-603`) now logs a **WARNING** with symbol, exact
    `start..end` bounds and the API error before returning None. It is still not retried, because
    the error is permanent.
  - `fetch_historical_bars` (`:691-696`) logs a WARNING for any chunk that finally failed, with
    its `i/N` index, bounds and `TAIL` vs `interior window`.
  - Earlier chunks' rows were already kept and still are (tested).
  - The misleading comment "yfinance will cover it" was corrected: it does not for stocks.
  - The "no retry on the last chunk" policy itself is unchanged.
- Sibling stock fetchers: `fetch_stock_bars_alpaca` (`:301`) and `refresh_daily_bars` (`:460`)
  send **no** `end=` (the server-default style `probe_noend.out` showed works), so they need no
  clamp.

## Item 3: data_sources.fetch_with_fallback stock branch
Policy unchanged: yfinance runs for stocks only when `not alpaca_ok` (`data_sources.py:187`).
With the clamp, the normal weekly incremental (one chunk from last-bar-48h to now) returns Alpaca
bars, so `alpaca_ok=True` and **yfinance is not called for stocks**. The live proof shows
`src={'alpaca': 112}` for the weekly-incremental shape, with no `[YF] Fetching AMD` line and
every bar at minute :00. Before the fix, the same single-chunk call returned None, so the code
fell to the :30-aligned yfinance path (the pre-fix A/B below).

## Tests
`tests/test_market_data_sip_clamp_2026_09.py`, 18 tests. It is Mac-safe: the API is a plain stub
class and alpaca is never imported.
- (a) Clamp math:
  - future, now, 5 min ago and 15 min ago are clamped to `now-16m`;
  - 1 day ago, exactly `now-16m` and 2021-04-01 are unchanged;
  - crypto is never clamped;
  - the default `now` is the UTC wall clock.
- (b) A stub whose `get_bars` raises `subscription does not permit querying recent SIP data`
  for any `end` within 15 min of now, and returns hourly bars otherwise:
  - the 7-day single-chunk (incremental) shape returns bars to within 2 h of now;
  - the 400-day, 3-chunk (rebuild) shape returns the full range with **zero** missing hours;
  - every stock request's `end` respects the delay;
  - crypto's last `end` stays at now;
  - an old `end_date` keeps its exact bound;
  - a forced denial of the last chunk emits both WARNINGs with bounds (`2021-07-01..2022-01-01`,
    `TAIL ... MISSING`) and keeps the first chunk's rows.
  - The stub was also checked against the **pre-fix** module. It reproduces the defect: the
    single chunk returns `None`, and the 3-chunk fetch ends one chunk early (2026-08-22).
- (c) Source pins:
  - `fetch_historical_bars` source contains `_clamp_sip_end(end_dt, asset_type`;
  - the helper uses `SIP_RECENT_DELAY_MIN` and the crypto guard;
  - the module has `SIP_RECENT_DELAY_MIN = 16`.

VERIFY (jenv, `CUDA_VISIBLE_DEVICES=''`):
```
$JPY -m py_compile market_data.py tests/test_market_data_sip_clamp_2026_09.py   -> OK
$JPY -m pytest tests/test_market_data.py tests/test_data_sources.py \
   tests/test_market_data_sip_clamp_2026_09.py tests/test_grp_data.py tests/test_c26_T2.py -q -p no:cacheprovider
======================== 90 passed, 2 warnings in 9.35s ========================
(new file alone: 18 passed in 1.15s)
```
I added `tests/test_c26_T2.py` because it holds `TestFetchHistoricalEndDate`, which monkeypatches
`_fetch_chunk` and pins `end_date` and default-end behaviour. It still passes: `now-16min` is
within its 3600 s tolerance. The 2 warnings are pre-existing (`data_utils.py:90` dateutil parse,
`data_sources.py:203` SettingWithCopy).

## Live proof (read-only; scripts `scratchpad/fixC_liveproof{,2}.py`, outputs `*.out`)
now = 2026-09-27 01:39 UTC (Saturday). The last trading day was Fri 2026-09-25.
```
POST-FIX fetch_with_fallback AMD from 2026-06-01: n=1312 first=2026-06-01 08:00 UTC
    last=2026-09-25 23:00 UTC (ET 2026-09-25 19:00, the final extended-hours bar) src={'alpaca': 1312} minutes={0: 1312}
POST-FIX fetch_with_fallback AMD from 2026-09-17 (weekly-incremental shape): n=112
    last=2026-09-25 23:00 UTC src={'alpaca': 112}   <- no yfinance call
POST-FIX fetch_with_fallback AMD from 2025-09-01 (3 chunks): n=4298 last=2026-09-25 23:00 UTC src={'alpaca': 4298}
PRE-FIX  fetch_historical_bars AMD from 2025-09-01: n=4010 last=2026-08-31 19:00 ET   <- final chunk silently dropped
PRE-FIX  fetch_historical_bars AMD from 2026-06-01: None  <- Alpaca empty -> stock yfinance :30 fallback
POST-FIX fetch_historical_bars BTC/USD from 2026-06-01 (Alpaca only): n=2834 last=2026-09-26 21:00 ET (= 01:00 UTC, current hour)  <- crypto unaffected
```
(For comparison, `C_probe/probe_fullfetch.out` shows the full 2016 rebuild ending at
2026-06-30 23:00 UTC before the fix.) In the post-fix run, `fetch_with_fallback('BTC-USD',
'2026-06-01', crypto)` returned `src={yfinance:14518, alpaca:2834}` starting 2024-09-27. That is
the pre-existing D08 `period='max'` behaviour (`TRADER_YF_WINDOW_SLICE` is OFF). It is unrelated
to this fix, and the crypto code path is untouched.

## What this changes
**This alters what a stock rebuild stores.** It recovers the lost newest ~3 months (the whole
final 6-month chunk) as Alpaca :00 SIP bars. It also stops weekly incrementals from falling to
:30-aligned yfinance bars. This is a data-store content change, so it belongs to the **clean
full rebuild** (C_data checklist step 0a). It must not land mid-stream against the old,
contaminated stores. Crypto fetches are byte-identical in behaviour, and so are the live
`fetch_stock_bars_alpaca` and `refresh_daily_bars` paths (no `end=`).

## Out-of-scope sibling (not fixed; not my file)
`llm_eval.py:218,135` builds `end = latest_entry + (max_h+6) h` and passes it to stock
`get_bars`. For journal entries from the last few hours, that end lands inside the SIP delay
window, or in the future. The same denial would then blank that symbol's outcome lookup.
Measurement-only, and moot today because no bots have run since 2026-05-07. The same clamp
(`market_data._clamp_sip_end`) would fix it.
