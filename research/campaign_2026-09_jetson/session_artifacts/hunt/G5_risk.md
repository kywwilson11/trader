# G5 — risk, sizing kernels, non-model gates: indisputable-improvement hunt (2026-09-26/27, Jetson)

Scope: portfolio, drawdown, risk_budget, volatility, macro_indicators, macro_calendar, regime_detector,
events_calendar, edgar_events, funding, oi_archive (post-FIX_R1), funding_archive, short_flow, basis_archive,
cost_regime, liquidity, fees, bet_sizing, crypto_trend, squeeze_features, borrow_proxy, short_cost,
strategy_config, stock_config, types_mod. This hunt was read-only and edited nothing in the repo.

Before hunting I read the brief and CAMPAIGN_BRIEF, research/AGENT_CONTEXT, FIX_R1 and the other FIX_* titles,
FLAGS.md §2 and §6, and the 2026-07 owner decision queue. The queue entries for these modules are dumped to
`hunt/G5/known_queue.txt`. Known and queued items are **not** re-reported as findings; they appear in the
appendix only where a finding touches them.

Every repro is in `<scratchpad>/hunt/G5/`. They run with
`source jenv.sh; CUDA_VISIBLE_DEVICES='' $JPY hunt/G5/<file>.py`. Each run took under 1 minute except
`rrv_state_e2e.py`, which takes about 2 minutes. RSS stayed under 250 MB.

Honesty note: my first run of `vix_nan.py` had not stubbed `requests`, so `fetch_financial_stress` made **one
read-only GET to fred.stlouisfed.org**. The script now stubs `requests`, and the re-run made no network call.
No other network calls were made.

---

## Findings, ranked by severity

### G5-1 · volatility.py:328-358 (`_merge_complete_day_rrvs`), consumed at :610 and :376 · class A
**Defect.** The "complete-day" RRV merge excludes only the frame's *last* (forming) calendar day. It never
excludes the *first* calendar day, which the rolling 250-bar live frame always truncates. Every day eventually
reaches the head position. From there, the dict-overwrite "self-correcting refresh" replaces the correct 24-bar
value with ever-shorter partial sums until the day leaves the frame. The result is that every settled day in
`crypto_rv_history.json` stores **1/24** of its true RRV, and every settled day in the HAR feed store stores
**20/24**. `min_bars=20` only stops the overwrites below 20 bars.

**Proof.**
- `hunt/G5/rrv_headday.py` replays the live cadence (`fetch_bars_alpaca(limit=250)` is `.tail(250)`,
  market_data.py:198/244) one new hourly bar at a time, over bars with an identical range every hour, so the
  true day RRV is exactly 24 times the per-bar value:
  ```
  B06 crypto_rv_history (min_bars=None): 28 settled days, stored/true RRV: min 0.042 median 0.042 max 0.042
  HAR feed store       (min_bars=20)  : 28 settled days, stored/true RRV: min 0.833 median 0.833 max 0.833
  with head day excluded (proposed)   : stored/true min 1.000 median 1.000 max 1.000
  ```
- `hunt/G5/rrv_state_e2e.py` runs the real `update_crypto_rv_state` for 110 days of **stationary iid** hourly
  ranges on the live-shaped rolling frame. The correct state here should be mostly 'normal':
  ```
  CURRENT CODE     : last-10-day states {'crisis': 240}, median pctile 95.3, get_crypto_rv_mult -> 0.3 (crisis)
  head day excluded: last-10-day states {'normal': 113, 'high': 110, 'crisis': 17}, median pctile 44.3, -> 1.0
  ```
  Once the 90-day history fills, the B06 BTC-RV state machine is pinned in 'crisis' (0.3x) on a calm,
  stationary market. This happens because "today" (a full trailing-24-bar sum) is compared against history
  values that are about 1/24 of their true size.

**Fix.** Skip the head day when the frame provably starts mid-day, placed next to the existing `last_day` skip:
```python
first_day = bars.index[0].normalize()
head_truncated = bars.index[0] != first_day          # frame begins after the day's 00:00 bar
...
if day != last_day and not (head_truncated and day == first_day) and np.isfinite(val) and val > 0:
```
Every day is interior to some 250-bar frame (250 > 24), so skipping the truncated head day never loses a day.
The fix only removes the contamination.

**Blast radius.**
- Callers:
  - `update_crypto_rv_state` (base_loop.py:956-962, every regime refresh). It feeds `get_crypto_rv_mult`, which
    is journaled as the DERISK_STACK_V2 shadow and applied to sizing only when that flag is ON.
  - `_har_feed_sigma` (TRADER_HAR_DAILY_FEED, default OFF).
  - `_har_rrv_load` (:297-311) seeds the HAR BTC store from `crypto_rv_history.json`, so the 1/24 values
    would also poison HAR-feed BTC sigma.
- **No live sizing decision changes today**, because both flags are OFF. What changes is the shadow evidence
  the owner will use to decide the DERISK_STACK_V2 flip.
- Neither `crypto_rv_history.json` nor `har_rrv_history.json` exists on this box yet, so nothing needs
  resetting if this lands before the bots start.
- Tests: `tests/test_c26_W1.py::test_merge_min_bars_none_matches_b3_inline` and `::test_merge_min_bars_skips_thin_head_day`
  use a head day that starts at 00:00 (24 bars), or a 5-bar head day at 19:00-23:00. Both still pass under this
  rule. The `tests/test_c26_S3.py` frames are whole days from 00:00 and are unaffected.
- Add one rolling-frame regression test, which is `rrv_headday.py` in essence.

**Why indisputable.** The function's own docstring says "Merge COMPLETE-day ... RRVs". A 1-bar partial is
provably not a complete day, and the measured stored/true ratio is 0.042.

### G5-2 · macro_indicators.py:107-121 (`fetch_vix`) · class A
**Defect.** A non-finite yfinance close is accepted as the VIX level. For example, today's `^VIX` row can come
back unpopulated with `Close=NaN`. When that happens:
- The FRED fallback is skipped and the NaN is cached for the full 1 h TTL.
- Every VIX ladder compares False and passes at 1.0x: `get_macro_regime`, `vix_tier_mult_v2`,
  `should_halt_stocks`, `should_block_risky_entries`, and base_loop's `f_vix`.
- The regime is labelled **'normal' without the "computed WITHOUT VIX" warning**, because `vix is not None`.
- base_loop's degraded-mode clamp is **bypassed**, because it counts only `vix is None` (base_loop.py:2516-2517).
- `detail['vix'] = round(nan, 1)` writes a NaN token into the journal.

**Proof.** `hunt/G5/vix_nan.py` stubs yfinance, FRED and requests, with no network:
```
fetch_vix -> nan | FRED fallback called: False
regime: normal sizing_mult 1.0 should_block_risky_entries False should_halt_stocks False vix_tier_mult_v2 1.0
cached for TTL: nan
```
With a finite guard, the same stub would return FRED's 30.0, giving a 'defensive' regime, `sizing_mult` 0.5 and
risky-entry blocking. How often yfinance actually returns a NaN last row was **not measured** (no network).
The finding is that the guard is missing.

**Fix.** In the yfinance branch:
```python
val = float(hist['Close'].iloc[-1])
if math.isfinite(val) and val > 0:
    _set_cached(...); return val
```
Otherwise fall through to FRED. Apply the same `isfinite and > 0` check to the FRED `val`.

**Blast radius.** The only change is on non-finite input: NaN becomes FRED's value, or `None` if that also
fails, which then triggers the documented blind-warning and degraded clamp. Existing tests still hold:
`tests/test_review_b07.py:164-185` (happy path 17.0, FRED fallback 17.5, total failure `None`), and
`test_ia1_removals` monkeypatches `fetch_vix` and is unaffected.

**Why indisputable.** The module's own docstring says a blind VIX must reach the fallback or read as `None`.
NaN evades both while also silently producing a neutral value on a live sizing path, which is the in-scope class.

### G5-3 · events_calendar.py:35 + :151 (`refresh_if_stale` throttle) · class A
**Defect.** `_last_attempt = 0.0` is compared against `time.monotonic()`, which on Linux counts seconds since
**boot**. For the first 1800 s of uptime, the process's very first refresh is therefore treated as "a failed
attempt < 30 min ago" and skipped. The stale on-disk calendar, or none at all, is served until uptime passes
30 min. `edgar_events.py:48` already fixed exactly this trap (`-float('inf')  # monotonic clock starts near 0
at boot`).

**Proof.** `hunt/G5/earn_boot.py` uses a temp cache that is 3 days old and lacks NVDA's print today; the fetch
is stubbed:
```
uptime     120s: fetch attempted=False  available=True  blocks_overnight_hold(NVDA)=False  earnings_within_days(NVDA)=False
uptime    1799s: fetch attempted=False  available=True  blocks_overnight_hold(NVDA)=False  earnings_within_days(NVDA)=False
uptime    1801s: fetch attempted=True   available=True  blocks_overnight_hold(NVDA)=True   earnings_within_days(NVDA)=True
```
The intended production start is a systemd unit at boot (scripts/setup_jetson_system.sh), which is exactly when
a post-outage calendar is stale. Consequences inside that window:
- The overnight sleeve (stock_loop.py:256/265, fail-closed by design) can keep a name that reports
  after the close.
- The entry block (stock_loop.py:1045) misses prints.

**Fix.** `_last_attempt = -float('inf')`, a one-token change.

**Blast radius.** Only the first call of a process started within 30 min of boot changes. No test pins the
`0.0` value. The only test references are `tests/test_grp_sentiment.py:107/122`, which monkeypatch
`_last_attempt = time.monotonic()` explicitly to throttle, so they are unaffected.

**Why indisputable.** The comment says "Throttle **failed** attempts"; no attempt has been made at that point.

### G5-4 · short_flow.py:81-89/150-171, funding_archive.py:101-121 (and oi_archive.py:130/238/312/345) · class C
**Defect.** `load_archive()` does a full `pd.read_parquet` on **every** call. The live feature injectors call it
once per symbol per 30 s cycle, and `PREDICTION_CACHE_ENABLED=False` means every cycle re-predicts every symbol:
- predict_now.py:413-421 calls `short_flow.live_svr_features` for each of the 46 stock names.
- predict_now.py:358-369 calls `funding.live_funding_features` for each of the 6 crypto names, which reaches
  `funding_archive.get_funding_series`.
- `funding_tilt` also reaches it when TRADER_FUNDING_Z_TIME_THINNING is set.

The rebuilt stores now carry `SVR_21/SVR_Z` and `Funding_*` (checked the schema of the current
`*training_data.parquet`), so both paths go hot as soon as the retrained models load.

**Measurement.** On the real archives on this Jetson, `hunt/G5/sf_cost.py`, `fa_cost.py` and `memo_proof.py`
give:
```
short_flow:      live_svr_features x46 = 0.94 s/cycle (0.56 s of it re-reading short_flow.parquet 46x)
funding_archive: get_funding_series x6 = 150-165 ms/cycle (~8 CPU-min/day)
memo on (mtime_ns,size) of ARCHIVE_FILE:              short_flow 855 -> 350 ms, funding 136 -> 54 ms
+ per-(stamp,symbol) memo of svr_series/get_funding_series: short_flow 834 -> 3.5 ms, funding 127 -> 0.1 ms
outputs asserted bit-identical (pd.testing.assert_series_equal(check_exact=True) / ==) for all 46 + 6 symbols
```

**Fix.** In each archive module:
- Memoize `load_archive()` on `(st_mtime_ns, st_size)` of `ARCHIVE_FILE`, returning the cached frame.
- Memoize the per-symbol derived series (`svr_series`, `get_funding_series`) in a dict keyed by symbol, cleared
  whenever the stamp changes. This keeps memory bounded to one stamp's worth.
- Callers are read-only on the returned objects. I verified this for `.iloc`, `.values[-90:]`, `.shift`/`.map`,
  arithmetic, and the `pd.concat` in `sync`.
- A harvest `sync()` rewrites the file through `os.replace`, which changes the stamp, so the memo self-invalidates
  across processes.

**Blast radius.**
- Readers: predict_now's live injectors, `funding_tilt`, the harvests' `*_features_for_index`, and `sync`.
- The oi_archive harvest path currently reads the 16 MB, 327k-row archive 3x per symbol, about 61 MB each time.
  It gets the same benefit, but only at harvest time, so it is secondary.
- Existing tests monkeypatch `get_funding_series` / `svr_series` or write fresh temp archives. The stamp changes
  on each write, so they are unaffected.
- Add an equality test that is `memo_proof.py` in essence.

**Why indisputable.** It is bit-identical, with a measured 0.8 s of CPU per stock cycle removed, on the Jetson
whose perf is the owner's #1 priority.

### G5-5 · funding.py:98-116 (`get_funding_rate`) · class D
**Defect.** There is no negative cache. During an OKX outage, each call re-pays the full 10 s `urlopen`
timeout. The sibling fetchers in oi_archive negative-cache for `_NEG_TTL=300` s precisely because "an OKX
outage would otherwise re-pay a 10s timeout per symbol per endpoint per prediction cycle" (oi_archive.py:372-377).
`get_funding_rate` is reached by every crypto prediction (6 per cycle, through the 5-worker pool) and by
`funding_tilt` for every sized candidate (serial, on the loop thread).

**Proof.** `hunt/G5/funding_outage.py` stubs `urlopen` to raise `socket.timeout`:
```
funding: 36 fetch attempts over 3 cycles x 6 symbols, timeout=10s each -> up to 360s blocked
oi_archive (neg-cached): 6 fetch attempts over the same 3 cycles
```
The repro calls `funding_tilt` for all 6 names, so it overstates slightly. In production the floor is 6
attempts per cycle through predictions, which is about 20 s of wall time on the 5-worker pool, plus 10 s for
each sized crypto candidate. A 30 s loop turns into a 50-60 s+ loop for the length of a hanging outage.

**Fix.** Mirror `oi_archive._neg_cached`: on exception, record `_fail_ts[symbol] = now`, and before fetching
return `None` while `now - _fail_ts < 300`.

**Blast radius.** Only the failure path changes. During an outage the result is still `None`, which callers
already map to tilt 1.0 and features 0.0. The one possible difference is up to 5 min of `None` after OKX
recovers. Tests: `tests/test_c26_P5.py` and `test_review_b08.py` monkeypatch `get_funding_rate` or its fetch;
the fixtures (`funding_env`) should also clear the new `_fail_ts` dict, so that a failure recorded in one test
cannot leak into a later test for the same symbol.

**Why indisputable.** This is the established repo pattern, already applied to the three sibling OKX fetchers,
and it turns an unbounded per-cycle stall into a bounded one.

### G5-6 · events_calendar.py:95, edgar_events.py:159, funding.py:78, oi_archive.py:417, stock_config.py:48 · class D
**Defect.** These JSON cache loaders catch `(OSError, json.JSONDecodeError)`. A non-UTF-8 byte in the file
raises **UnicodeDecodeError**, which is a `ValueError` but *not* a `JSONDecodeError`, so it escapes. risk_budget
already documents and handles this: `read_registry` catches `ValueError` because "the UnicodeDecodeError a
binary-garbage file raises (power loss mid-write)".

The escapes have these consequences:
- **events_calendar.** `calendar_available()` and `blocks_overnight_hold()` raise inside the unguarded
  `flatten_before_close → _select_overnight_keepers` (stock_loop.py:320/256/265). Every flatten-window cycle
  errors before any sell, so the **whole stock book is held overnight** until the file is removed. The module's
  own comment at :97-102 claims to guard this path.
- **stock_config.** `stock_loop.get_symbol_universe` (stock_loop.py:103) raises every cycle, so the stock loop
  stops working. The module's own comment says a bad file must fall back with a loud warning.
- **edgar_events.** `entry_blocked` raises, which the caller swallows, so the veto silently fails open forever.
  The file never self-heals.
- **oi_archive.** Live OI features stay at 0.0 forever. The raise happens before the rewrite, so the file never
  self-heals.
- **funding.** `get_funding_rate` raises after every successful fetch, because `_history` stays `None`.

**Proof.** `hunt/G5/garbage_json.py` uses temp files only:
```
UnicodeDecodeError is JSONDecodeError? False | is ValueError? True
events_calendar.calendar_available RAISES UnicodeDecodeError   (also blocks_overnight_hold, earnings_within_days)
edgar_events.entry_blocked RAISES UnicodeDecodeError
funding._load_history RAISES UnicodeDecodeError
oi_archive._load_live_history RAISES UnicodeDecodeError
stock_config.load_stock_universe RAISES UnicodeDecodeError
risk_budget.read_registry (reference guard: except (ValueError, OSError)) -> {}
```
**Fix.** At each of the 5 sites, replace `json.JSONDecodeError` with `ValueError`, which is its superclass. That
keeps each loader's existing corrupt-file semantics: reset to `{}` or defaults, with the existing warning.

**Blast radius.** Behaviour changes only on a non-UTF-8 file. Valid and JSON-corrupt files behave identically,
and no test pins the narrower tuple.

**Why indisputable.** The repo already made this exact fix in risk_budget. All writers are atomic, so the input
is improbable, but the failure modes are severe and the fix is a superclass swap.

### G5-7 · volatility.py:79-99 (`forecast_volatility`) · class A (low reachability)
**Defect.** The documented contract is "PER-BAR sigma ... or None". A zero-variance return series (≥100 bars of
identical closes, i.e. a frozen or halted feed) makes arch forecast `variance = NaN`, and `NaN <= 0` is False,
so **NaN** is returned. `compute_vol_adjusted_size` then evaluates `max(0.5, min(1.5, NaN))`, which gives
**1.5**. That is the *maximum boost*, reached on the default legacy path (base_loop.py:2396-2401 and :2601).
The NaN is also stored as `Position.garch_sigma` (base_loop.py:3262/3289) and serialized by `to_dict` into
journals as a NaN token.

**Proof.** `hunt/G5/garch_nan.py` and `garch_nan2.py` run the real arch 1.x:
```
zeros fit= ARCHModelResult sigma= nan vol_adj= 1.5
nonzero= 0 sigma= nan   (1, 2, 5, 10, 20 nonzero returns -> finite)
```
**Fix.** `if not np.isfinite(variance) or variance <= 0: return None`. Also guard `compute_vol_adjusted_size`
with `if not np.isfinite(sigma) or sigma <= 0: return base_notional`.

**Blast radius.** Only the exactly-flat-series case changes, from 1.5x to the documented neutral path (sigma
`None` gives `vol_mult` 1.0). Tests `tests/test_new_modules.py::TestVolatility::*` use random returns and are
unaffected.

**Why indisputable.** NaN is not "a sigma or None", and the NaN turns into the maximum size boost.

### G5-8 · volatility.py:102-122 (`get_garch_stop`) · class B (low value)
**Defect.** Zero production callers. `docs/graphs/import_graph.json` lists base_loop as the only non-test
importer of `volatility`, and base_loop's imports are `compute_vol_adjusted_size` plus a local `get_sigma`.
Repo-wide grep finds it only in volatility.py and `tests/test_new_modules.py:96-105`.
`tests/test_grp_loops.py::test_dead_imports_pruned` asserts base_loop no longer imports it. The code already
carries a "DEAD" comment. The 2026-07 review deferred removing it *only* because base_loop still imported it
then ("must be done together with a base_loop-owning change"), and that blocker is gone.

**Fix.** Move it verbatim to `research/campaign_2026-08/08_removed_code.md` (repo rule: delete nothing, archive
the block). Move its two tests along with it, or re-point them at the archived text.

**Blast radius.** Only those two tests. **Why indisputable.** Zero callers, proven, and the deferral reason no
longer holds.

### G5-9 · oi_archive.py:410-424/443-445 (`live_oi_features`) · class C (small)
**Defect.** `oi_history.json` is fully re-parsed on every call: 6 crypto symbols per 30 s cycle, under
`_live_lock`. Once filled (6 × 840 samples, about 207 KB), each parse takes **5.75 ms**, so **34.5 ms/cycle**,
about 100 CPU-s/day. There is exactly one writer (this function, in the crypto-bot process), and funding.py
already keeps its sibling history in memory (`_history`).

**Proof.** `hunt/G5/oi_hist_cost.py` uses a synthetic file of the steady-state size.

**Fix.** Keep `hist_all` in a module-level memo keyed on `(st_mtime_ns, st_size)`, re-read when the stamp
changes, and update the memo after the successful `os.replace`. Outputs are unchanged, including the FIX_R1
`usable` filter.

**Blast radius.** The live OI path only; tests in `test_oi_archive.py` / `test_oi_inf_2026_09.py` write files
between calls, which changes the stamp and so stays correct. **Why indisputable.** It is bit-identical, but the
gain is small, so it is ranked last.

---

## Judgment calls (not proposed) and already-known items touched

Known and queued (2026-07 owner queue or 01_state_map), not re-reported:
- **portfolio.**
  - Tail-position (not timestamp) return alignment.
  - Self-pair on pyramiding.
  - Fail-open 0.0 defaults.
- **funding.**
  - D28 z-baseline spans ~2.8 days (TIME_THINNING flag shipped dark).
  - Archive staleness with no freshness check.
- **volatility.**
  - HAR structurally dead live (D30).
  - Partial *forming*-day RRV in the legacy path. G5-1 is the head-day defect in the separate complete-day
    stores and is a different bug.
  - EGARCH has no asymmetry term.
- **macro_indicators.**
  - STLFSI2 is discontinued.
  - The VIX ladder is duplicated with base_loop (0.8 vs 0.7 at 15-25).
- **regime_detector.** Inverted smoothing (KILL-recommended layer).
- **events_calendar.** D07 calendar-day windows.
- **edgar_events.** No backoff on outage.
- **oi_archive.**
  - Coin vs notional unit mismatch.
  - The L/S begin-only query truncates the last 24 h.
- **short_flow.**
  - Up-to-a-week live staleness.
  - Universe-growth NaN hole.
- **basis_archive.**
  - `f/8` funding math.
  - Module unwired.
- **squeeze_features, crypto_trend, bet_sizing, borrow_proxy, short_cost, basis_archive.** Production-dead,
  declared-ahead research modules (01_state_map §7 activation backlog). Archiving them is an owner call.
- **risk_budget.** GATE-1 stale-drop bias (queued). `allocate_book_caps`, `scale_for_account_cap` and
  `simulate_two_books` have zero production callers; they are declared GATE-2 tooling.
- **stock_config.** TRADABLE_* have no consumers; they are listed in FLAGS §2.
- **strategy_config.** The zero-reader constants (TILT_MIN, IOC_EXIT_CAP_BPS, CRYPTO_CS_*, CONVICTION_*,
  TIER_*) are all in FLAGS §2 or read by portfolio_backtest.
- **SSOT.** The FLAGS §6 exceptions are not re-reported. Same-named constants re-typed outside strategy_config
  (`CORR_SANITY_MAX`, `TILT_MAX`, etc.) were checked by AST and none has diverged. `BARS_PER_YEAR` is
  duplicated in backtest, portfolio_backtest and hypersearch but has identical values, so it is not a
  divergence.
- **bidask missing on the Jetson.** The upward-biased AR spread fallback is in use; this was reported in
  C_data and FIX_A.

New observations deliberately not proposed. Each is model-facing, gate-facing, or arguable:
1. **funding_archive has non-8h prints.** SOL/USD has 98 two-hour and 3 four-hour funding intervals.
   `rate*3*365` and `shift(3)` "= 24h" assume 8 h prints, so SOL's Funding_Rate_Ann and Funding_Chg_24h are
   mis-scaled on those rows. This is a model-facing feature value. Measured with a read-only groupby over
   `funding_archive.parquet`.
2. **Funding live/train z parity.** Live z uses `pstdev` over the 90 archive prints *excluding* the current
   rate; training uses the rolling sample std (ddof 1) over 90 prints *including* it. In the local-history
   fallback, `samples[-3]` is about 30 min back, not 24 h, and the window is about 22 h. This is model-facing.
3. **OI live z window.** The live z uses up to 840 samples (35 d, `_LIVE_MAX_SAMPLES`) while training and the
   docstring say 30 d (720). This is model-facing.
4. **`check_stablecoin_pegs` on a one-sided book.** A zero bid or ask gives a mid of about 0.5 and a 50%
   "deviation", which halts crypto entries for 5 min. It fails in the conservative direction, so it is gate
   policy.
5. **`blocks_overnight_hold` over-blocks.** It also blocks on a *bmo* print that happened this morning. This is
   the conservative direction and a gate rule.
6. **Wall-clock TTLs.** `macro_indicators._get_cached`, `volatility._model_cache` and `regime_detector._hmm_cache`
   use `time.time()`; events uses `datetime.now()`. An NTP backward step on the RTC-less Jetson lengthens those
   TTLs. Switching to monotonic is arguable, since events persists across processes.
7. **Dead, deliberately deferred with an owner-visible pin.** `types_mod.Quote` and `MacroRegime.is_defensive`
   have zero production callers. `tests/test_review_b03.py::test_deferred_surface_still_present` pins them, and
   base_loop's docstring cites Quote as a shape reference.
8. **`volatility._har_cache` is shared by two key types.** The legacy path uses `datetime.date` keys and the
   feed path uses `pd.Timestamp` keys. They only thrash each other under TRADER_HAR_DAILY_FEED, which is OFF.
9. **`macro_indicators.get_spy_trend_ok` does not cache failures.** During an outage there is one Alpaca
   daily-bars GET per stock cycle. It fails open, and the cost is minor.
10. **Harvest-only helpers re-read the oi archive 3× per symbol** (about 61 MB each). This is covered by
    G5-4's memo if extended to oi_archive; it is training-path, so it was not measured under load.

## No issues found (beyond the known items above), per file
- **drawdown.py.** The ladder, HWM restore and `update_peak_equity` are NaN/inf-total on reachable inputs.
- **risk_budget.py.** The account formula, bisection, `book_risk_budget` closed form (re-derived), flock
  registry and gate-1 report all check out.
- **portfolio.py.** `diversified_book_risk` and `book_risk_budget` were re-derived and are correct. The EWMA
  diagnostics, `_timestamp_diag` and cache TTLs are fine.
- **macro_calendar.py.** The 2026 FOMC dates were verified. CPI dates were not verifiable offline. The ET
  conversion and naive guard are fine.
- **regime_detector.py.** Nothing beyond the known queue item.
- **edgar_events.py.** Only G5-6 (plus the known backoff item).
- **funding_archive.py.** Only G5-4.
- **short_flow.py.** Only G5-4.
- **cost_regime.py.** PIT lag, per-day collapse and FRED parse are all fine.
- **liquidity.py.** Clip/fill V1/V2, the vectorized cost twin and the impact term are all fine.
- **fees.py.** Only `maker_share` blindness under journal rotation, which is known and default OFF.
- **bet_sizing.py, crypto_trend.py, squeeze_features.py, borrow_proxy.py, short_cost.py.** These are unwired;
  their math matches their docstrings.
- **strategy_config.py.** No divergent duplicates.
- **stock_config.py.** Only G5-6.
- **types_mod.py.** Only appendix item 7.
