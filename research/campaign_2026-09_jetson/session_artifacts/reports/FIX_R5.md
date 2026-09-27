# FIX_R5 — raw-sidecar second-run crash (`'Index' object has no attribute 'hour'`)

## Root cause (one line)
The sidecar interior-gap repair (`harvest_crypto_data.py:~393` / `harvest_stock_data.py:~620`) appends a patch from
`market_data.fetch_historical_bars`, whose index is **America/New_York**-stamped (the Alpaca SDK's tz), onto the **UTC**
sidecar slice via `data_utils.append_ticker_data`; `pd.concat` of two different tzs yields an **object `Index`**, which
flowed through the 0-new-bars merge into `compute_features` → `indicators.py:532 idx.hour`.

The first run had no sidecar, so no gap repair ran — which is why it worked. The sidecar LOAD itself was fine.

## Reproduction (on a copy: scratchpad/fixr5/raw_ohlcv.parquet, `raw_sidecar_path` monkeypatched; script fixr5/repro.py)
```
load:                      DatetimeIndex dtype=datetime64[ns, UTC] n=265226
slice (BTC-USD):           DatetimeIndex dtype=datetime64[ns, UTC] n=50266
patch (gap-repair fetch):  DatetimeIndex dtype=datetime64[ns, America/New_York] n=22
after gap-repair append:   Index dtype=object n=50266          <-- DatetimeIndex lost HERE
new (fetch_with_fallback): DatetimeIndex dtype=datetime64[ns, UTC]
merged (0 new bars):       Index dtype=object n=50266          -> compute_features crash
merged without gap-repair: DatetimeIndex dtype=datetime64[ns, UTC]
```
(`fetch_with_fallback` normalises to UTC via `data_sources._to_utc`; the raw `fetch_historical_bars` call used by gap
repair does not.) Side note: every gap-repair window refetched 18–36 bars already present and added 0 rows — these are
real venue holes, so each run re-fetches the same ≤5 windows per ticker (cheap, bounded; not changed).

## Diff summary (my changes only, vs pre-edit backups in scratchpad/fixr5/*.orig; per-file diffs in fixr5/*.diff)
- `data_utils.py` (+75/−9):
  - new `_to_utc_index(index)` — any timestamp-like index → tz-aware UTC DatetimeIndex (naive taken AS UTC, aware
    converted, mixed-tz object Index healed via `pd.to_datetime(..., utc=True)`), name kept.
  - new `normalize_utc_index(df)` — no-op when already UTC, else shallow copy with a UTC index.
  - new `_ensure_utc_index(df, where)` — fail-loud guard: `TypeError` on a non-DatetimeIndex (message names the ticker,
    element types and tzs, and points at the mismatched-tz merge), `ValueError` on naive / non-UTC tz, unsorted, or
    duplicated index. Returns df unchanged.
  - `append_ticker_data`: both sides normalised to UTC before concat (and in the empty-side early returns); dedup
    keep='last' + sort unchanged. This is the source fix.
  - `load_raw_ohlcv`: contract is now UTC DatetimeIndex, de-duplicated on (timestamp, Ticker) keep='last'
    (merge_raw_ohlcv's rule), stable-sorted (mergesort, only when not already monotonic). Also handles a sidecar whose
    timestamp came back as a `Datetime` column.
  - `merge_raw_ohlcv`: both sides normalised to UTC; sorts are stable (mergesort).
- `scripts/harvest_crypto_data.py` (+5/−1): import `_ensure_utc_index`; `_ensure_utc_index(ohlcv, ticker)` right before
  `compute_features` in `prepare_data`.
- `scripts/harvest_stock_data.py` (+4/−1): identical gap-repair path exists (it would have crashed on the stock harvest's
  second sidecar run the same way); same import + guard right before `compute_stock_features` in `prepare_stock_data`.
  The running stock harvest imported the old modules and was not touched; its output was not touched.
- `tests/test_raw_sidecar_reload_2026_09.py` (new, 12 tests, Mac-safe; parquet round trips skip without an engine).

## SAVE side (item 4)
`save_raw_ohlcv` → `_atomic_to_disk` → `df.to_parquet(tmp, compression='snappy')` + `os.replace`. The index is written as
the pandas-metadata index column `Datetime` with type `timestamp[ns, tz=UTC]` (verified on raw_ohlcv.parquet before and
after this run). The stock sidecar `stock_raw_ohlcv.parquet` (to be written at the end of the running stock harvest)
uses the exact same code path (`harvest_stock_data.py:~656` `merge_raw_ohlcv` loop → `save_raw_ohlcv(raw, 'stock')`);
its first run is a full refetch with no gap repair, and all its frames come from `fetch_with_fallback` (UTC), so it
will save UTC correctly with the old code too. The bug only bites on a SECOND sidecar run — fixed now.

## Verification
- `py_compile` data_utils.py, both harvests, new test: OK.
- `pytest tests/test_raw_sidecar_reload_2026_09.py tests/test_data_utils.py tests/test_tb_restamp_2026_09.py
  tests/test_improve_harvest.py tests/test_c26_T3.py tests/test_grp_data.py tests/test_c26_T2.py -q -p no:cacheprovider`
  (C ext disabled via PYTHONPATH=noc): **141 passed, 2 warnings in 4.26s** (12 new).
- Extra: every other test file referencing data_utils / harvest scripts / prepare_* (17 files: test_asof_universe,
  test_backtest_v3, test_c26_Q2, test_c26_T4, test_grp_training, test_imports, test_gate_protocol, test_liquidity_v3,
  test_oi_inf_2026_09, test_r2c_fee_sweep, test_r2c_measurement_kernels, test_review_b05/b09/b20/b21,
  test_sentiment_pit_2026_09, test_wave4): **387 passed**. Full suite not run (per brief).

## Real crypto harvest (TRADER_RAW_SIDECAR=1 TRADER_YF_WINDOW_SLICE=1; log phase5/harvest_crypto3.log)
- **rc=0**, 0 `TB-GUARD`, 0 `Inf:`, 0 Traceback. (A first attempt died with rc=127 before Python started — my
  `/usr/bin/time` wrapper does not exist on this box; rerun without it.)
- Gap repair ran for all 6 tickers with NY-tz patches; each ticker "0 new bars" (the exact crash path) and passed the guard.
- `[SIDECAR] Saved 265226 raw rows to raw_ohlcv.parquet (9.6 MB)`; `[DATA] Saved 263889 rows to training_data.parquet (99.8 MB)` (+ CSV).

### Store validation (scratchpad/fixr5/validate_store.py → fixr5/validate_out.txt)
- training_data.parquet: 263,889 rows × 71 cols, index `Datetime timestamp[ns, tz=UTC]`, 2021-01-05 09:00 → 2026-09-25 03:00 UTC.
  Per ticker: BTC 50119, DOGE 50074, ETH 50117, LINK 49945, SOL 39790, XRP 23844.
- Families: TB_* 15 (Bars/Ret/Reason × 12/18/24/32/48), Funding_* 3 (Chg_24h, Rate_Ann, Z), OI_* 2 (Chg_24h, Z),
  Taker_Imb_24h. Eff_Spread_Pct / CS_* absent — expected, their stamps are dark flags not set in this run.
- Inf in any numeric column: **NONE**. OI_Chg_24h: finite share 1.0000, NaN 0, inf 0.
- Daily_Sentiment nonzero share 0.9690 overall (0.968–0.976 per ticker); log: 255696/263889 bars have sentiment.
- TB_Bars positional validity (restamp test's method — re-walk `policy_exits.compute_tb_labels` over the STORED rows per
  ticker and require exact equality wherever both walks resolve inside the stored rows; the last 10–31 rows per fb whose
  walk continues into the unstored bar tail are excluded): **DOGE / LINK / SOL / BTC / ETH / XRP all 0 mismatches** on
  Bars, Ret and Reason for every fb; 0 NaN; 0 values outside [0, fb]. LINK has 17 and SOL 12 interior >3h gaps and
  still re-walk exactly, so spans are positional over the stored rows.
- raw_ohlcv.parquet after the run: DatetimeIndex UTC, 265,226 rows, monotonic.
