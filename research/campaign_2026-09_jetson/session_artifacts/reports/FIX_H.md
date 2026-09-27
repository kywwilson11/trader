# FIX_H — Daily_Sentiment PIT repair (applied 2026-09-26, Jetson)

Status: **model-facing, applied unflagged, and takes effect at the rebuild.** Per the orchestrator's
ruling, it ships inside the clean re-harvest and retrain (a gotcha-#2 event). Nothing changes model
inputs until a harvest runs. Nothing was committed. The live `sentiment_cache.db` was never opened for
writing.

## Diff summary (`git diff --stat`: sentiment_history.py +107/-1, scripts/harvest_stock_data.py +9/-5)

### sentiment_history.py
- **New `_migrate_fng_date_basis(db)`** plus the constants `_FNG_BASIS_KEY='fng_date_basis'`,
  `_FNG_BASIS_UTC='utc_publication'` and `_FNG_LEGACY_TABLE='fng_daily_legacy_localtz'`. It is called at
  the top of `fetch_crypto_sentiment_history` (right after `db = _get_db()`).
  - It returns immediately if `state[fng_date_basis]=='utc_publication'`. Otherwise it runs one
    `BEGIN IMMEDIATE` transaction that does four things:
    1. Re-checks the marker under the write lock.
    2. Copies `fng_daily` into `fng_daily_legacy_localtz`, which has an explicit PK schema and is filled
       with `INSERT OR IGNORE`.
    3. **Verifies that every row is preserved** (LEFT JOIN on date and value). If any row is missing it
       raises `RuntimeError` and rolls back, so `fng_daily` is never cleared unpreserved.
    4. Runs `DELETE FROM fng_daily` and sets the marker, plus an audit key `fng_date_basis_migrated_at`.
  - On a fresh or empty DB it only sets the marker, so no empty legacy table is created.
  - On any error it rolls back.
  - It is idempotent and runs at most once per DB.
- **`limit={total_days}` → `limit=0`** (full history). The old value counted back from TODAY.
- The refill path was already correct: it dates by `datetime.fromtimestamp(ts, tz=utc).date()`
  (unchanged, now at the `date_str = …` line in `fetch_crypto_sentiment_history`). After the
  migration `cached_dates` is empty, so every in-range day is re-inserted with its UTC date.
- **New `stock_sentiment_lookup_dates(index)`** implements the stock key `((t_utc - 6h).date() - 1 day)`.
  A naive index is treated as UTC and a tz-aware index is converted to UTC.

### scripts/harvest_stock_data.py (sentiment block only, ~536-552)
- Imports `stock_sentiment_lookup_dates` alongside `fetch_stock_sentiment_history` and uses it as the key.
  The unused `import datetime as _dt` is dropped.
- The cached read window's start moved from `min - 1d` to `min - 2d`, because the earliest 00:00 UTC bar
  now reads D-2.
- On the current store, the only bars whose key changes are those at hours 00 and 01 UTC:
  18,551 + 4 = 18,555 of 1,236,699 rows (1.5 %).
- Stub-module tests (test_c26_T2) that lack the helper fall into the existing `except` branch, which
  gives `Daily_Sentiment = 0.0`. They stay green.

### tests/test_sentiment_pit_2026_09.py (new, Mac-safe: sqlite3, pandas and a temp DB; requests.get stubbed)
- **(a) Migration.** Legacy rows are preserved verbatim, `fng_daily` is emptied and the marker is set.
  A second call returns 0 and leaves both post-migration rows and the legacy table untouched. A fresh
  DB gets the marker only. If a conflicting legacy row exists, the migration raises and `fng_daily` and
  the marker stay untouched.
- **(b) Refill.** A stubbed API payload of known (D 00:00 UTC, value) pairs is loaded over a leaked
  (D+1) cache. After the refill, `fng_daily[D]` equals the value published at D 00:00 UTC for every D,
  the returned dict matches, and the legacy table is intact. A second test pins that the fixture can
  tell the rules apart: America/Chicago `fromtimestamp` gives D-1 for both CST and CDT dates, and the
  shipped UTC rule appears in the function body.
  - The conda python has no `time.tzset`, so a forced-TZ run was not possible. The host is
    America/Chicago anyway.
- **(c) Stock key.** 00:00 → D-2, 05:59 → D-2, 06:00 → D-1, 14:30 → D-1, 23:00 → D-1. This holds for
  UTC-aware, naive and Chicago-aware indexes. An exhaustive check over Jan–Aug 2026, every hour, CST and
  CDT, shows the Chicago-dated bucket always closes at or before the bar. A source pin checks that
  harvest uses the helper and that the old `zip(final_df['Ticker'], final_df.index.date)` is gone.
- **(d) Source pin.** `fng/?limit=0&format=json` is present, `limit={total_days}` is absent, and the
  migration call comes before `if start_date is None`.

## Verification
```
$JPY -m py_compile sentiment_history.py scripts/harvest_stock_data.py tests/test_sentiment_pit_2026_09.py  -> OK
$JPY -m pytest tests/test_sentiment_pit_2026_09.py tests/test_sentiment_history.py tests/test_improve_harvest.py \
    tests/test_c26_P5.py tests/test_c26_T2.py tests/test_c26_V2.py tests/test_c26_P2.py -q -p no:cacheprovider
======================= 149 passed, 1 warning in 12.50s ========================
```
- The new file alone gives 13 passed.
- No `tests/test_harvest*.py` or `tests/test_sentiment_history*.py` files beyond `test_sentiment_history.py` exist.
- The full suite was not run.

## Copy-DB proof (`h_fix/proof.py`, output in `h_fix/proof_out.txt`)
- **How the copy was made.** It is a consistent snapshot taken with the sqlite **backup API** from a
  `mode=ro` connection, not `cp`. The live DB is in WAL mode and was being written by `--fetch-stocks`,
  so a raw copy of the three files could tear.
- **How the module was pointed at it.** `sh._DB_PATH` was set to the copy and `_db_local` was reset.
  Then `fetch_crypto_sentiment_history('2021-01-05', <UTC today>)` ran against the real alternative.me
  API, which is read-only.

| | before | after |
|---|---|---|
| tables | articles, daily_sentiment, fng_daily, sqlite_sequence, state | + `fng_daily_legacy_localtz` |
| fng_daily rows / min / max | 1876 / 2021-01-05 / 2026-02-24 | **2091 / 2021-01-05 / 2026-09-27** |
| fng_daily_legacy_localtz | — | 1876 / 2021-01-05 / 2026-02-24 (== pre-migration fng_daily: **True**) |
| state | live_mode=1 | + fng_date_basis=utc_publication, fng_date_basis_migrated_at |
| articles / daily_sentiment | 96048 / 14394 | 96048 / 14394 (untouched) |

- Second `_migrate_fng_date_basis` call: **0 (no-op)**.
- **Sample of 30 refilled dates** (seeded random): `fng_daily[D] ==` the API value stamped D 00:00 UTC in
  **30/30**. Across all refilled dates it is **2091/2091**. For example, 2021-03-01 has fng_daily=38,
  api[D]=38, api[D+1]=78, legacy[D]=78.
- Legacy rows: == api[D] in 228/1876 (chance) and == api[D+1] in **1876/1876**. This reconfirms the leak
  the migration removes.
- **Earliest date preserved.** The refill starts at 2021-01-05, which equals the requested start. The
  API's own earliest date is 2018-02-01.
- **What limit=0 fixes.** For a range ending before today, 2021-01-05..2021-12-31, the old
  `limit=total_days` (361) keeps **0** days. `limit=0` keeps 361.

## Serve-vs-harvest parity statement
- **Crypto: serve == harvest.**
  - Harvest: a bar on UTC day D gets `fng_daily[D]`, which is now truly the value published at
    D 00:00 UTC (`harvest_crypto_data.py:360-368`).
  - Serve: `predict_now.py:314` calls `get_live_daily_sentiment`, which calls `sentiment.get_fear_greed()`
    (`limit=1`, the latest publication, i.e. today-UTC's value).
  - The only edge is the 5-minute cache TTL just after 00:00 UTC. It can serve the previous day, which
    is stale-safe.
- **Stock: serve == harvest for every bar hour that exists in the store.** The store has bars at
  00, 01 and 07–23 UTC, and none at 02–06.
  - Serve: `predict_now.py:314` calls `sentiment_history.get_live_daily_sentiment`, which returns
    `daily_sentiment[(date.today() - 1 day)]`. The host is **America/Chicago**, so this is
    Chicago-date − 1.
  - Harvest: `(t_utc - 6h).date() - 1`. `t - 6h` is exactly Chicago wall-clock in CST, and one hour
    behind it in CDT. So they are equal except at **05:00–05:59 UTC during CDT**, where harvest gives
    D-2 and serve gives D-1. Serve's D-1 bucket is complete at 05:00 UTC in CDT, so serve is still
    point-in-time.
  - No bars exist at that hour. For example, a 00:00 UTC bar gets D-2 from both sides, and an RTH bar
    gets D-1 from both.
  - **No predict_now change is needed.**
  - **Residual caveat, not changed:** the serve rule depends on the host timezone. The line is
    `sentiment_history.py`, `get_live_daily_sentiment`:
    `yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()`. To make parity
    host-independent, that line would become
    `yesterday = stock_sentiment_lookup_dates([pd.Timestamp.now(tz='UTC')])[0]`.
  - I did **not** make that change. `tests/test_c26_P2.py:86-120` pin `date.today()-1`, and on UTC CI
    runners they would fail between 00:00 and 06:00 UTC if the rule changed. That is left as an owner
    decision.

## Notes for the orchestrator
- **The migration fires on the live DB the first time `fetch_crypto_sentiment_history` runs**, which is
  the crypto harvest. It takes `BEGIN IMMEDIATE`, and the connection has `timeout=60`, so it waits
  rather than failing if `--fetch-stocks` holds the write lock. If the migration raises, the harvest's
  try/except stamps `Daily_Sentiment=0.0` with a WARNING. That is fail-closed: it never falls back to
  leaked values.
- **Out of scope but relevant: mixed date basis in the stock article cache.** The running
  `--fetch-stocks` inserts **UTC-dated** articles (current code), incrementally from the newest cached
  day (2026-02-21, which is Chicago-dated). So the article cache now mixes Chicago-dated rows
  (≤ 2026-02-21) with UTC-dated rows. Boundary-day headlines may land under both dates, because the
  UNIQUE key includes the date.
  - The −6h key is PIT-safe for both bases, since a UTC bucket closes earlier than a Chicago bucket.
  - The boundary-day duplication is the owner's call.
