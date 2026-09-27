# FIX_R2 — REVIEW_fixes M1, M2, L6 (2026-09-26)

Files touched (only): market_data.py, sentiment_history.py, beta_ledger.py,
tests/test_market_data_sip_clamp_2026_09.py, tests/test_sentiment_pit_2026_09.py,
tests/test_measurement_fixes_2026_09.py. Harvest scripts, training stores and live DB untouched.

## M1 — market_data._clamp_sip_end (market_data.py:547-585)
- The stock end is still clamped to now-16min. New: when that limit falls inside the extended
  session (04:00-20:00 America/New_York, Mon-Fri), it is floored to `floor_hour(limit) - 1s`.
  Every returned bar then satisfies open+1h <= limit, so no forming bar reaches the store.
- The floor is session-gated, so after 20:00 ET, pre-04:00 ET and on weekends the clamp is
  byte-identical to before. All pre-existing tests (NOW is a Saturday) pass unchanged.
- Crypto is untouched. A caller end older than the cap is untouched.
- Constants added: `_SIP_SESSION_TZ` and `_SIP_SESSION_OPEN_H` / `_SIP_SESSION_CLOSE_H`. Uses
  stdlib zoneinfo; tzdata is present at /usr/share/zoneinfo.
- Tests added (class TestInSessionHourFloor):
  - 15:37 ET Tuesday gives 14:59:59 ET (= 18:59:59 UTC, EDT).
  - Edge and EST cases: clamp at exactly 15:00:00, 04:04 pre-market, 19:54, January EST.
  - Outside the session the result is unchanged: 20:16 (clamp exactly 20:00), 21:37, 03:30,
    Saturday, Sunday.
  - Crypto in-session is unchanged.
  - End-to-end `fetch_historical_bars` with a frozen RTH clock: the last request end is
    18:59:59 UTC and the last bar is 18:00 UTC (the old code returned the forming 19:00 bar).
- Real-clock tests adjusted to stay correct whatever hour the suite runs. They now assert
  against the clamp's exact bound: newest bar within [bound-1h, bound], and default-now equals
  the explicit-now result. The lag ceiling went from 2h to 2h20m, because in-session the
  newest complete bar can be 2h15m old. This was paired with the stronger exact-bound pin, so
  no test was weakened.

## M2 — sentiment_history FnG migration (sentiment_history.py:201-440)
- **Reordered.** `fetch_crypto_sentiment_history` now does this on an un-migrated DB:
  1. Fetches the full series first (limit=0).
  2. Only if the fetch yields rows, calls `_migrate_fng_date_basis(db, fetched)`. That runs
     copy-aside + verify + DELETE + INSERT refill + marker in ONE `BEGIN IMMEDIATE` transaction.
- **Why "return legacy values" and not "raise".** `scripts/harvest_crypto_data.py:361-372`
  wraps the call in `except Exception`, sets `Daily_Sentiment = 0.0`, and still saves the
  store. A raise would therefore write zeros. The function never raises on a fetch or
  migration failure.
- **On failure** (network error, empty `data`, non-JSON response, or the migration refusing):
  - the legacy table stays in place and the marker is not set;
  - a `log.warning` says the UTC re-dating was NOT done and that the LEGACY values served still
    carry the 1-day look-ahead leak (not zero);
  - the legacy cached values are returned.
- **New guard.** `_migrate_fng_date_basis` refuses with ValueError, touching nothing, if legacy
  rows exist and the refill is missing or covers fewer than 90% of the legacy row count. This
  covers a truncated API response. A fresh or empty DB still just gets the marker.
- **Coverage check.** The harvest has no warning, only a `filled` print, so the check lives in
  sentiment_history as `_warn_fng_coverage`, called on every return path. It logs a WARNING
  when nonzero-score days are below 90% of the requested days.
- Added `import logging` and `log = logging.getLogger(__name__)`. It deliberately does not use
  log_config. Pipeline subprocesses send stderr to stdout, so the lastResort handler's output
  lands in the harvest log.
- **Tests changed:**
  - (a) now passes a refill and asserts fng_daily == refill right after the migration. It is
    never observed empty.
  - The idempotence assertion is now snapshot-exact.
  - The conflicting-legacy test passes a refill so it still reaches the RuntimeError path.
  - The source pin (d) was replaced by a stronger one: `requests.get(` precedes the single
    `_migrate_fng_date_basis(db, fetched)` call.
- **Tests added:**
  - Refusal without a full refill (None, [], and 1 of 3 rows).
  - Failed fetch in 4 modes (raise / empty / garbage / bad JSON). Each serves legacy values
    (nonzero), leaves no marker and no legacy table, and logs the warning. The next successful
    fetch then migrates.
  - Migration conflict inside the fetch path serves legacy values and does not raise.
  - Fresh DB plus failed fetch: no marker, and the coverage WARNING shows `0/10`.
  - Coverage threshold: 18/20 gives no warning, 17/20 warns.
  - An already-migrated DB is a no-op: no network, no writes.
  - `test_live_db_copy_idempotent`: an sqlite backup-API copy of the real sentiment_cache.db
    (source opened mode=ro). It skips if the file is absent or unmigrated.
- **Live-DB proof** (scratchpad/m2_proof.py on a backup copy, network stubbed offline):
  - state is fng_date_basis=utc_publication; fng_daily has 2089 rows (2021-01-05..2026-09-25);
    the legacy table has 1876 rows.
  - `migrate()` returns 0 twice. A fetch over 2025-02-24..2026-02-24 makes no network call and
    returns 366 days, 358 of them nonzero (97.8%, so no warning).
  - Table and state are byte-identical before and after (hash 742489143f49).
  - Live sentiment_cache.db and -wal mtimes are unchanged (the -shm is touched by read opens and
    the running harvest).

## L6 — beta_ledger.drop_glitch_days (beta_ledger.py:148-240)
- **First-day glitch.** Day 0 is dropped when it is positive, outside the band of day 1, and
  day 2 is inside day 1's band ("compare to the next good day"). Day 1 then becomes the anchor.
- **Why only the first day.** A leading run of 2+ days has exactly the shape of the existing
  pinned case `[100000, 100500, 40000, ...]` (a real level change on day 2, which must be
  kept). The review's median-of-3 seed has the same limitation.
- **Accepted ambiguity (documented in the docstring).** A genuine >50% move between day 0 and
  day 1 is indistinguishable from a day-0 bad print, so it is dropped. That is one point.
- Pre-funding zeros on day 0 are left alone.
- **NaN profit_loss on a dropped day** now carries forward as 0 instead of NaN, with a stderr
  `[beta_ledger] WARNING: non-finite profit_loss on N dropped glitch day(s) carried forward as
  0: <dates>`. A kept day's own NaN pl is still left NaN.
- **Tests added:**
  - First-day drop, with the pl fold and clean returns < 1%; an upward first-day spike is
    also dropped.
  - A first-day glitch plus a mid-window glitch are both dropped.
  - Five cases where the rule must not fire: level change on day 2, pre-funding zeros, too
    short, unconfirmed day 1, and normal data.
  - NaN pl on a dropped day gives finite pl plus the warning; finite pl gives no warning.
  - A kept day's NaN pl stays NaN.

## Verification
- py_compile passes for all 6 files.
- `CUDA_VISIBLE_DEVICES='' $JPY -m pytest tests/test_market_data_sip_clamp_2026_09.py
  tests/test_market_data.py tests/test_data_sources.py tests/test_sentiment_pit_2026_09.py
  tests/test_sentiment_history.py tests/test_measurement_fixes_2026_09.py
  tests/test_beta_ledger.py tests/test_beta_ledger_v3.py tests/test_c26_T2.py -q -p
  no:cacheprovider` gives **192 passed** (1 pre-existing pandas SettingWithCopyWarning from
  data_sources.py:203).
- No full suite was run and no real network call was made. The M2 proof used a stubbed-offline
  `requests.get`, so the read-only alternative.me fetch was not needed.
