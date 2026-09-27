# IMPL_g5 — G5 risk / non-model-gate fixes (2026-09-27, Jetson)

Files edited (owned only): volatility.py (after the IMPL_g4b gate opened), macro_indicators.py, events_calendar.py,
edgar_events.py, funding.py, oi_archive.py (on top of FIX_R1), short_flow.py, funding_archive.py, stock_config.py,
research/campaign_2026-08/08_removed_code.md (append), tests/test_new_modules.py (two tests retired),
tests/test_c26_W1.py (one line, coordinator-authorised), and a new tests/test_g5_fixes_2026_09.py (61 tests).
There were no git operations, no full suite run and no network calls; CUDA_VISIBLE_DEVICES was '' throughout.

## Changes
- **G5-1** `volatility._merge_complete_day_rrvs`: added `first_day`/`head_truncated`. The frame's first calendar day
  is now skipped when the frame starts after 00:00 UTC, as well as the forming last day. A frame that starts exactly
  at 00:00 keeps its head day, as before.
- **G5-2** `macro_indicators.fetch_vix`: the yfinance close is accepted only if `math.isfinite(val) and val > 0`.
  Otherwise it falls through to FRED, and FRED rows that are not finite and >0 are skipped. NaN is never cached.
  `import math` was added.
- **G5-3** `events_calendar._last_attempt = -float('inf')`.
- **G5-4** `short_flow` and `funding_archive`:
  - `load_archive()` is memoized on `(path, st_mtime_ns, st_size)` under a module lock. A failed read is never
    memoized, so it still warns and retries on every call.
  - `svr_series` and `get_funding_series` memoize per symbol on top of that. An entry is stored only when the frame
    used is the memoized frame (`_memo['df'] is arc`). The math moved verbatim into
    `_svr_series_from` / `_funding_series_from`.
  - `sync()`'s `os.replace` and repointing `ARCHIVE_FILE` both invalidate the memo.
- **G5-5** `funding.get_funding_rate`: a failed fetch stores `_cache[symbol] = (now, None)`, honoured for
  `_NEG_TTL = 300`, the same value as `oi_archive._NEG_TTL`; a test pins the two equal.
  - I kept it inside `_cache` on purpose. The existing fixtures in `test_c26_P5.py` and `test_review_b08.py` reset
    `funding._cache`, so negative entries are reset with it and cannot leak between tests. A separate dict would
    have needed edits to those fixtures, which I don't own.
- **G5-6** The `except` clause changes from `json.JSONDecodeError` to `ValueError` at events_calendar `_load_cache`,
  edgar_events `_load_cache`, funding `_load_history`, oi_archive `_load_live_history` and stock_config
  `load_stock_universe`.
- **G5-7** `forecast_volatility`: `if not np.isfinite(variance) or variance <= 0: return None`. There is also a
  backstop in `compute_vol_adjusted_size`: `if not np.isfinite(sigma) or sigma <= 0: return base_notional`.
- **G5-8** `get_garch_stop` was removed and a 3-line pointer comment left in its place.
  - The verbatim function and the two `TestVolatility` tests are archived in 08_removed_code.md under a new heading,
    "## 2026-09-26 — Jetson test & improvement campaign removals" → "### G5-8". The zero-caller grep proof is included.
  - The two tests were removed from tests/test_new_modules.py and replaced by a comment. test_wave4.py had no
    get_garch_stop test and is untouched.
- **G5-9** `oi_archive`: a `_hist_memo` keyed on `(path, mtime_ns, size)` of oi_history.json.
  - The memo is dropped before any in-place append, so a failed persist leaves no in-memory sample that is not on
    disk.
  - After a successful write it is refreshed from the tmp file's stamp, taken before `os.replace`, which keeps the
    inode. A foreign rewrite changes the stamp, so the file is re-read.
- **Coordinator add-on** `volatility._rv_save`: per-writer tmp `{path}.{pid}.{thread_ident}.tmp`, then `os.replace`,
  and the tmp is removed on any failure. This mirrors `_har_rrv_save`.
  - tests/test_c26_W1.py:510 now asserts that no `*.tmp` sibling is left:
    `assert not list(Path(volatility._HAR_RRV_FILE).parent.glob('*.tmp'))`.

## Evidence
- Hunter repros on the fixed tree:
  - rrv_headday stored/true is 1.000/1.000/1.000 for both min_bars=None (was 0.042) and min_bars=20 (was 0.833).
  - rrv_state_e2e, last 10 days: {'normal':113,'high':108,'crisis':19}, median percentile 44.9, mult 1.0 (normal).
    Before, it was {'crisis':240} with mult 0.3.
  - vix_nan: 30.0 from FRED, regime 'defensive', sizing 0.5.
  - funding_outage: 6 attempts over 3 cycles (was 36).
  - garbage_json: every loader returns defaults and nothing raises.
  - earn_boot: first call fetches.
  - garch_nan: zeros gives sigma None (was NaN, giving a 1.5x boost).
- G5-4 on the real Jetson archives (scratchpad/impl_g5/memo_vs_head.py): compared against the HEAD modules loaded
  side by side, 220 feature arrays plus live dicts/series are **bit-identical**. The comparison covered
  `live_svr_features` ×46, `svr_features_for_index` ×46, `get_funding_series` and `funding_features_for_index` ×6,
  over 2 passes, where pass 2 is served from the memo.
- Timing (memo_real.py): short_flow `live_svr_features` ×46 went from 856 ms to 4.5 ms per cycle; `svr_series` from
  951 ms to 0.4 ms; funding `get_funding_series` ×6 from 134 ms to 0.2 ms.
- Fail-before check: I ran the new test file against pre-fix copies of all 9 modules (git HEAD, oi_archive = HEAD +
  FIX_R1, volatility = pre-G5 backup; scratchpad/impl_g5/orig + g5orig_plugin).
  - Result: 28 failed + 20 errors. The errors are memo tests whose fixture patches the new `_memo`/`_hist_memo`.
  - 13 passed. These are equality/regression pins, the cases old code already handled, the 00:00-head kept case, and
    the G5-3 test (it loads the file from the repo root by design; earn_boot proves the pre-fix failure).
  - On the fixed tree: 59 passed, 2 skipped. The skips are param-specific cases that apply to the other module only.

## Verify
- py_compile of all 9 modules and the 3 edited/new tests: OK.
- Requested set: test_g5_fixes_2026_09, test_new_modules, test_wave4, test_edgar_events, test_oi_archive,
  test_oi_inf_2026_09, test_stock_config, test_grp_risk, test_grp_macro, test_grp_deriv and test_c26_P1 gave
  **238 passed, 2 skipped**. No test_volatility*/test_macro*/test_events_calendar*/test_funding*/test_short_flow*
  files exist.
- Adjacent set: every other test importing an owned module, 37 files including test_c26_P5, review_b06/07/08/09/17,
  c26_W1/S3, g4b, grp_loops/sentiment/ops, predict_now, ia1/ia3/ia4 and imports, gave **970 passed**.
  test_chart_core gave 96 passed.

## Notes for the orchestrator/owner
- **Behaviour deltas** occur only on degenerate input:
  - A NaN VIX close now goes to FRED or `None`, which triggers the blind-warning and degraded clamp.
  - A NaN or 0 GARCH variance now gives `None`, i.e. vol_mult 1.0.
  - `compute_vol_adjusted_size` with **+inf** sigma now returns base, 1.0x. It used to give 0.5x, because the ratio
    was 0 and then clamped. No producer emits inf after the G5-7 guard.
  - A funding outage now returns `None` for up to 300 s after OKX recovers.
- **G5-1 changes the stored RRV history**, which feeds the DERISK_STACK_V2 shadow evidence and the
  TRADER_HAR_DAILY_FEED path (both default OFF). Neither crypto_rv_history.json nor har_rrv_history.json exists on
  this box, so nothing needs resetting.
- **Doc drift, not mine to fix:** docs/MODULES.md:643 still describes `get_garch_stop` as a known issue; it should now
  say removed (G5-8). The volatility section of docs/MODULES.md should drop the function.
- **Not done:** oi_archive's harvest-time `load_archive()` memo (the hunter's secondary G5-4 note) was out of scope.
- **Pre-existing:** test_c26_P1::test_llm_outage_backoff_and_attempt_stamp emits an urllib3 `ssl.match_hostname`
  DeprecationWarning, which suggests it may open a real TLS connection. That test is not mine; worth a look.
