# C: Re-harvest readiness and what the harvest does to the stale stores

Agent C, 2026-09-26, on the Jetson (jetson env: py3.10, numpy 1.26.1, pandas 2.3.3, pyarrow 23).
**Nothing was written to the training stores, the archives or any production file.** All probe
scripts and their raw outputs are in `scratchpad/C_probe/` (`*.py` / `*.out`), and the store
inspection output is in `scratchpad/C_store_ranges.txt`. Every network call was read-only:
Alpaca bars, yfinance, HEAD requests to the archive CDNs, and the PyPI JSON API.

---

## VERDICT: NOT READY. Three blockers, and a fourth issue to settle before the first run.

1. **BLOCKER (new defect, stock book): a stock harvest today silently loses its newest ~3 months,
   and the weekly incrementals that follow would re-contaminate the store.**
   `market_data._fetch_chunk` sends `end=now`. On this account's plan the SIP feed rejects that
   with `APIError: subscription does not permit querying recent SIP data`. `_fetch_chunk`
   classifies it as a "subscription" error and returns `None` (`market_data.py:565-567`), and
   `fetch_historical_bars` skips the retry for the last chunk (`market_data.py:636-640`). The
   stock branch of `fetch_with_fallback` only calls yfinance when Alpaca returned *nothing*
   (`data_sources.py:187`), so nothing covers the gap. Measured through the harvest's own call:
   - A full rebuild from 2016 ends at **2026-06-30 23:00 UTC** for AMD, SPY and JPM
     (`C_probe/probe_fullfetch.out`). The 2026-07-01 → now chunk is dropped.
   - The same request with `end` omitted (the live `fetch_stock_bars_alpaca` style), or with
     `end=now-16min`, returns data through 2026-09-25 (`C_probe/probe_stock_recent.out`,
     `probe_noend.out`).
   - Worse: once the store exists, every weekly incremental is a single chunk ending at `now`.
     That chunk fails, so Alpaca returns nothing, so the stock yfinance fallback runs and appends
     **:30-aligned bars**. This is exactly the cross-run grid interleave `data_sources.py:184-186`
     warns about, and it is what the old store already contains (item 1c below).

   Crypto is unaffected: BTC/USD returns bars through the current hour. **Fix before any stock
   harvest.** It is an owner decision because it changes store contents. The one-liner is to clamp
   `end_dt` to `now - 16 min` in `fetch_historical_bars` (or omit `end` on the final chunk).

2. **BLOCKER: `bidask` is absent, and on hourly bars the fallback is a volatility gauge, not a
   spread.** The fallback `liquidity._abdi_ranaldo_rolling` is roughly the median hourly
   (H−L)/C. It stamps AMD, NVDA and TSLA at the **1.50 % cap** (median; 65-77 % of bars
   cap-clipped), GLD at 1.07 %, COST at 0.92 %. Real quoted spreads on these names are about
   1-5 bp. `Eff_Spread_Pct` feeds the per-bar cost in the `backtest.py` promotion gate (`:330-338`)
   and the `meta_label` labels (`:726-733`). A store stamped this way would reject essentially
   everything at the gate and label everything negative. This confirms runbook Phase 0 step 2:
   **STOP**.
   - `bidask` 2.1.0 is a pure-python `py3-none-any` wheel requiring `numpy` and `pandas` with no
     version bounds and Python ≥3.6. It is **compatible** with numpy 1.26.1 / pandas 2.3.3 /
     py3.10. Install with `--no-deps` so pip can't touch the pinned stack.
   - One nuance for the runbook's rationale: the stores on this device have **never** carried
     `Eff_Spread_Pct`, so installing bidask *before* the first rebuild is not a "mid-stream"
     change here. The rebuild is the first stamp.

3. **BLOCKER (procedure): the default incremental path must not be used on these stores.** It
   keeps and propagates known contamination:
   - the crypto D08 Yahoo overwrite (2023Q3-2026Q1 volume is in the wrong units);
   - the stock :00/:30 grid interleave (about 3,450 yfinance rows per ticker,
     2024-02-23..2026-02-19);
   - the D39 head loss;
   - and it would **refuse** the B15 merge for 8 of 46 stock names, freezing them at Feb 2026.

   A clean full rebuild is required, and the old stores must be moved away first (procedure
   below).

4. **Settle before the first run: the first harvest reaches only partial archive history, so
   several features stay neutral-filled in older rows.** `funding_archive.parquet`,
   `oi_archive.parquet` and `short_flow.parquet` **do not exist** on this device.
   - OI is capped at 2,000 day-files per sync against about 13,600 needed (10 symbols × days since
     2023-01-01). The first crypto harvest therefore gets OI/TT/Taker only for about the last 200
     days. Everything older is 0.0-filled (`harvest_crypto_data.py:224-237`).
   - Short-flow is capped at 150 files per run against about 428 weekdays wanted, so roughly the
     last 7 months arrive.
   - Pre-sync them standalone (commands below), or accept that the older rows stay 0.0-filled
     until later harvests back-fill.

**History depth is NOT a problem** as long as Alpaca is used. A fresh store is as long as or
longer than the old one at the head: crypto from 2021-01-01 (XRP from 2024-01-01), stocks from
2016-01-01. The tail is limited only by blocker 1. yfinance alone could not rebuild anything
multi-year: `period='max'` at 1 h returns **728-730 days** (first bar 2024-09-27).

---

## Recommended clean full re-harvest (numbered checklist)

Prerequisites (owner):

0a. **Land the stock SIP end-clamp fix** (blocker 1). Verify it with a stock
    `fetch_with_fallback('AMD','2026-06-01',api,'stock')` whose last bar is the last completed
    session.

0b. `/home/kyle/miniforge3/envs/jetson/bin/python -m pip install --no-deps 'bidask==2.1.0'`,
    then `python -c "import bidask"`. Afterwards, re-run `C_probe/probe_spread.py`: the
    `bidask EDGE failed … AR fallback` warning must disappear, and the medians must drop to the
    few-bp range.

0c. Make sure nothing else is running: no bots (none run since 2026-05-07), no other agent jobs.
    The stock concat is the RAM peak (see Risks).

0d. Optional but recommended: fill the stock sentiment gap without any LLM spend.
    `python sentiment_history.py --fetch-stocks` makes Finnhub `company_news` calls and
    keyword-scores them (`sentiment_history.py:959,973-975`), universe names only. `--backfill`
    is the LLM-spend path; do not use it for this.

Archive the old stores (delete-nothing). Use plain `mv`: the files are untracked and gitignored,
and the basename patterns in `.gitignore:15-18,26` still match under `archive/`.

1. `D=archive/training_stores/2026-02-28_pre_reharvest && mkdir -p $D`

2. `mv training_data.parquet training_data.csv stock_training_data.parquet stock_training_data.csv $D/`
   **Move BOTH formats.** `load_training_data` falls back to the CSV when the parquet is missing
   (`data_utils.py:117-121`). Leaving the CSV behind turns the "full" harvest into an incremental
   one from the CSV.

3. For the retrain event that follows (gotcha #2), move rather than delete
   `v2_study.db stock_v2_study.db` into the same `$D`. Add one row per moved file to
   `archive/README.md` (origin = repo root, untracked/gitignored, reason = "pre-2026-09 re-harvest;
   old 51/53-col schema, D08/grid-contaminated", safe to restore = yes, nothing reads `archive/`).
   Disk is not a concern: 1.77 GB moved, 382 GB free on `/`.

Pre-sync the archives. These write the three archive parquets, not the training stores, and each
step can be interrupted and re-run:

4. `python funding_archive.py`
   One-time burst of about 700 monthly zips (10 symbols from their `LISTING_MONTH`). Afterwards,
   about one new zip per symbol per month.

5. `python oi_archive.py --max-files 14000`
   Full back-fill 2023-01-01 → yesterday, all 10 symbols. The module's own estimate is about
   10-15 min per 2,000 files, so plan roughly 1-1.5 h. Without this step the harvest fetches
   only 2,000 files.

6. `python short_flow.py --max-files 450`
   About 428 weekday files covering `START_DAYS_BACK=600`; a few minutes.

Run the harvests. Use the jenv wrapper with `CUDA_VISIBLE_DEVICES=''`. Optionally add
`PYTHONPATH=<scratchpad>/noc` to disable the leaky C extension; Agent A showed it is numerically
equivalent on long frames.

7. `TRADER_RAW_SIDECAR=1 TRADER_YF_WINDOW_SLICE=1 python -u scripts/harvest_crypto_data.py 2>&1 | tee harvest_crypto_2026-09.log`
   With no `raw_ohlcv.parquet` present this is the documented forced full refetch
   (`harvest_crypto_data.py:269-281`, docstring `:13-24`). It also creates the raw sidecar, so
   later weekly incrementals no longer eat the head (D39). The slice flag stops Yahoo from
   overwriting Alpaca on later incrementals (D08). **Both flags are model-facing and belong to
   runbook Phase 3.** This IS that data-store event, so the owner must set them in the service
   environment too, or the next weekly incremental reverts to the old store-derived path. If the
   owner declines the flags, step 2 alone still forces a full harvest ("No existing data — full
   harvest", `:289-291`), but without D39/D08 protection going forward.

8. `TRADER_RAW_SIDECAR=1 TRADER_YF_WINDOW_SLICE=1 python -u scripts/harvest_stock_data.py 2>&1 | tee harvest_stock_2026-09.log`
   Covers 92 tickers (46 universe + 46 `TRAINING_CANDIDATE_POOL`, candidates from 2021-01-01).

9. Read each log for:
   - `[MERGED] … src={alpaca:N}` with no `yfinance` rows for stocks;
   - `[SPREAD] … median` in basis points, not about 1.5 %;
   - `[AS-OF]` drop counts;
   - **`[TB-GUARD]`**: expected to fire (Risks §4);
   - `Daily_Sentiment … bars have sentiment`;
   - `Date range … to` near today.

   Then confirm the schema: stock columns should include `Eff_Spread_Pct`, 20 `CS_*`,
   `SVR_21`/`SVR_Z` and 21 `TB_*`; crypto should include `Funding_*`/`OI_*`/`TT_LS_Z`/
   `Taker_Imb_24h` and `TB_*`.

10. Only after that: the Phase-3 retrain (reset `best_score` and `cum_trials`, and the study DBs
    moved in step 3).

**Expected duration.** There is no timing evidence in `pipeline_output.log`. It starts
2026-03-22 and contains no harvest phase, because the harvests last ran on 2026-02-28 and their
output is gone. From this session's probes of the exact fetch path
(`C_probe/probe_fullfetch.out`):
- Crypto: 57-77 s per coin for a full 2021→now Alpaca + yfinance fetch, plus about 60 s for the
  BTC benchmark. That is about 8 min of fetch; features take about 0.1 s per coin and TB labels
  about 1 s. With the archive syncs already done (steps 4-6), expect **~10-15 min** in total.
- Stock: about 60 s per 2016-start name and about 22 s per 2021-start candidate, so
  47 × 60 + 46 × 22 ≈ **65 min of fetch**. The SDK's own 429 back-off ("sleep 3 seconds and
  retrying") fires repeatedly. Add features, the concat/mask/rank passes and a multi-GB CSV
  write: expect **~1.5 h** in total.
- Steps 4-6: about 1.5 h, dominated by OI.

---

## Evidence

### 1. Incremental vs full: what the harvest does to the stale stores

**What is on disk** (`C_store_ranges.txt`). File mtimes are all 2026-02-28, i.e. about 5,000 h
old.

| store | rows | cols | span | notes |
|---|---|---|---|---|
| `training_data.parquet` | 234,822 | 52 (incl. Datetime) | 2021-01-13 → 2026-02-24 | 6 coins; XRP only from 2024-01-13; SOL has a 236-day hole ending 2024-02-23 |
| `stock_training_data.parquet` | 1,236,699 | 54 | 2016-01-21 → 2026-02-19 | 46 names, **no** candidates, **no** as-of masks, 158,729 rows (12.8 %) at **:30** |

No `raw_ohlcv.parquet` or `stock_raw_ohlcv.parquet` exists.

**a) The incremental path does NOT merge features, so there is no schema problem.**
- It extracts only `['Open','High','Low','Close','Volume']` from the old store:
  crypto `harvest_crypto_data.py:282-287, 318-322`; stock `harvest_stock_data.py:430-435, 482-487`.
- It appends new bars with `append_ticker_data` (keep='last', `data_utils.py:184-194`).
- It **recomputes every feature over the full OHLCV history**: `compute_features`
  (`harvest_crypto_data.py:139`) and `compute_stock_features` (`harvest_stock_data.py:188`).
- The new columns (`Eff_Spread_Pct`, `CS_*`, `Funding_*`/`OI_*`, `TB_*`, SVR, …) are therefore
  computed for every row, and the old feature columns are simply discarded. There is no NaN
  back-fill of new columns, no schema-mismatch error and no silent drop.
- `save_training_data` atomically replaces both files (`data_utils.py:152-181`).
- The problem is not the schema but the **OHLCV content carried forward**:
  - **Crypto D08 contamination is already in the store.** Median hourly BTC volume by quarter:
    2021-2023Q2 is 27-287 (coin units, Alpaca); 2023Q3-2024Q1 is 0.007-0.013; 2024Q2-Q4 is
    **0.9-18 million** (Yahoo USD volume); 2025 is **0** (Yahoo hourly crypto volume is mostly
    zero). ETH shows the same pattern. An incremental run makes this worse: yfinance
    `period='max'` ignores `start_date` (`data_sources.py:176-183`), so about 730 days of Yahoo
    rows re-enter via the store-level keep='last'. `TRADER_YF_WINDOW_SLICE` is the fix
    (`data_sources.py:196-201`).
  - **The stock grid interleave is already in the store.** Every name carries about 3,450 :30
    rows from 2024-02-23 to 2026-02-19 (the yfinance 730-day window as of February). Example
    AMD 2025-06-10: 13:30, 14:00, 14:30, … alternating. Post-market :00 bars there have
    Volume 0. It also inflates DV30: the median ratio of dv30 on all rows to dv30 on :00 rows
    only is 1.27. Incremental mode keeps all of it forever.
  - **D39 head creep.** Crypto starts 2021-01-13 rather than 2021-01-01. Each incremental run
    drops about 100 more warmup bars (`dropna`, `harvest_crypto_data.py:212`,
    `harvest_stock_data.py:259`).

**b) The 48 h overlap guard.**
- `_get_incremental_start` = last bar − 48 h (crypto `:80-92`, stock `:72-84`). That gives
  2026-02-22 for crypto and 2026-02-17 for stocks.
- The crypto path has **no** guard (`harvest_crypto_data.py:110-112`).
- Stocks run the B15 guard (`harvest_stock_data.py:150-163`,
  `data_utils.overlap_close_divergence` / `OVERLAP_DIVERGENCE_MAX=0.01`). I dry-ran it against
  the old store with today's `adjustment='all'` Alpaca bars (`C_probe/probe_mergeguard.out`).
  **8 of 46 are REFUSED**, and a refused ticker keeps its existing rows and gets **zero** new
  bars (`:163`):

  | name | divergence |
  |---|---|
  | PPLT | 90.0 % |
  | PALL | 80.1 % |
  | CRWD | 75.0 % |
  | META | 1.95 % |
  | FSLR | 1.77 % |
  | OXY | 1.48 % |
  | ABNB | 1.44 % |
  | AVGO | 1.22 % |

  PPLT, PALL and CRWD look like splits; the others are dividend/adjustment drift.
- Another 11 names sit at 0.5-0.96 %. They merge, and in doing so splice two adjustment bases.
- Only 4-9 overlap timestamps exist per name, because the old store's tail is mostly :30 rows.

**c) The "< 24 h old → skip" rule** (`run_pipeline._build_harvest_phases`, `run_pipeline.py:945-976`,
using file-mtime age from `_get_data_age_hours`, `:931-942`) **does not matter today**. The files
are about 5,000 h old, so a pipeline start would schedule both harvests. Running the scripts
directly bypasses the rule entirely. `_needs_force_harvest` (`:1125-1156`) also stays False,
because both old stores already carry `Target_Return_96` (forward bars for both books =
[12,18,24,32,48,64,96]).
- Historical note: the harvests stopped at 2026-02-28 because the weekly branch used to pass
  `skip_harvest=not force_harvest`. The fix comment is at `run_pipeline.py:1756-1761`. This
  explains the 7-month staleness.

**d) Forcing a clean full rebuild.** There is no CLI flag: neither harvest script uses argparse.
There are two documented mechanisms:
1. **Remove the store:** `load_training_data` returns empty → "No existing data — full harvest"
   (`harvest_crypto_data.py:289-291`, `harvest_stock_data.py:436-438`). Both .parquet and .csv
   must go (`data_utils.py:105-123`).
2. **`TRADER_RAW_SIDECAR=1` with the sidecar absent:** the old store is never read
   (`existing = pd.DataFrame()`), and there is a full refetch from `ALPACA_START` / `CANDIDATE_START`
   (`harvest_crypto_data.py:269-281`, `harvest_stock_data.py:421-428`, `data_utils.py:205-212`).
   The docstrings (`:13-24` / `:10-17`) and runbook Phase 3 name this the head-rebuild mechanism.

Mechanism 2 alone does not preserve the old store: `save_training_data` overwrites it. Hence the
`mv` in the checklist.

**What a full rebuild fetches** (dry-run of the exact `fetch_with_fallback` call, no features,
no writes, RSS 222 MB):

| ticker | rows | first | last | yfinance rows admitted |
|---|---|---|---|---|
| BTC | 50,264 | 2021-01-01 06:00 | now | 2 |
| ETH | 50,262 | 2021-01-01 | now | 2 |
| DOGE | 50,256 | 2021-01-01 | now | 4 |
| LINK | 50,256 | 2021-01-01 | now | 6 |
| SOL | 40,187 | 2021-01-01 | now | 7 (about 10,089 missing hours: an Alpaca hole, 2023-06 → about 2024-08) |
| XRP | 23,989 | **2024-01-01** | now | 30 |
| AMD | 40,493 | 2016-01-01 | **2026-06-30** | — |
| SPY | 41,488 | 2016-01-01 | **2026-06-30** | — |
| JPM | 19,585 | 2021-01-01 | **2026-06-30** | — |

- In a full crypto rebuild the yfinance rows are only Alpaca-hole fills, not D08-scale. The
  slice flag is a no-op when start = 2021, and matters for later incrementals.
- All stocks came back at :00 alignment with `src=alpaca` only.

### 2. Source health

(`C_probe/probe_sources.out`, `probe_stock_recent.out`, `probe_noend.out`)

- **Alpaca crypto, BTC/USD 9 d:** 218 bars, all at :00, no missing hours, median volume 0.042 BTC
  (venue units). The SDK returns `America/New_York` timestamps; `fetch_with_fallback._to_utc`
  converts them (`data_sources.py:144-149`). The store is UTC.
- **Alpaca stock, AMD:** fails with `end=now` (blocker 1).
  - `end=now-16min` returns 160 bars over 14 days, 04:00-19:00 ET. That is the **extended-hours**
    grid at about 16 bars per day, :00-aligned. The default feed is **SIP**: identical to
    `feed='sip'`, volume 256 M.
  - `feed='iex'` gives only 75 bars and 7.0 M volume, about 2.7 % of SIP. So DV30 would be
    unusable on IEX.
- **yfinance:**
  - BTC-USD: UTC, :00, close within a median 0.6 bp of Alpaca, but **58 % zero volume**.
  - AMD: **:30-aligned** (13:30-19:30 UTC, RTH only, 7 bars per day), per `docs/MAP.md` §4a and
    `data_sources.py:168-173`.
  - `period='max'` at 1 h returns **728-730 days** (first bar 2024-09-27, 3,480 AMD bars and
    17,331 BTC bars). A yfinance-only multi-year store is impossible, and the cap is exactly the
    D39/D08 window.
- **Alpaca depth probes** (earliest bar returned):

  | symbol | window requested | result |
  |---|---|---|
  | BTC/USD | 2016, 2019, 2020-06 | nothing |
  | BTC/USD | 2020-12-15 → 2021-01-15 | first bar **2021-01-01 01:00 ET** |
  | XRP/USD | 2021, 2022-06, 2023-06 | nothing |
  | XRP/USD | 2023-12 → 2024-01 | first bar **2024-01-01** |
  | AMD, SPY | 2015-06 | nothing |
  | AMD, SPY | 2015-12-20 → 2016-01-15 | first bar **2015-12-31 19:00 ET** (= 2016-01-01 00:00 UTC) |

  Account depth therefore matches the harvest constants `ALPACA_START` (2021-01-01 crypto,
  2016-01-01 stock). A fresh store is **not shorter** than the old one: old crypto started
  2021-01-13 after D39 creep, and old stock 2016-01-21.
- **Free archives.** No module offers a dry or no-write mode. Each `sync()` writes its parquet in
  the repo root, and the CLIs have only `--start` / `--max-files` / `--days-back`, so none were
  run. HEAD probes: Binance funding monthly zip → 200, Binance OI daily (2026-09-24 and
  2023-01-01) → 200, FINRA `CNMSshvol20260925.txt` → 200, alternative.me F&G → 200.
  CryptoCompare → **401**, as documented; the harvest still spends one call plus `sleep` per coin
  on it.

  | archive | target | cadence / cap |
  |---|---|---|
  | `funding_archive.sync` (`:124-184`) | `data.binance.vision/.../monthly/fundingRate/{SYM}/…-{YYYY-MM}.zip`, 10 perps, from `LISTING_MONTH` | complete months only; idempotent per (symbol, month); first run is a one-time burst |
  | `oi_archive.sync` (`:151-231`) | `…/daily/metrics/{SYM}/…-{day}.zip`, 10 perps from `OI_START=2023-01-01` | newest-first; `MAX_FILES_PER_SYNC=2000` (404s count); aborts after 10 consecutive failures; about 7 harvests to back-fill if left to the default |
  | `short_flow.sync` (`:92-147`) | `cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`, 92-name panel filter | `START_DAYS_BACK=600`, `MAX_FILES_PER_SYNC=150` newest-first |

  Neutral fills keep `dropna` from eating the rows these archives don't cover: crypto
  `_fill_archive_features` fills 0.0 (`harvest_crypto_data.py:224-237`); SVR is filled via
  `fill_warmup_features` before `dropna` (`harvest_stock_data.py:258-259`,
  `indicators.py:923-942`). Missing archive history therefore means neutral 0.0 features, not
  lost rows.

### 3. bidask and the fallback

(`C_probe/probe_spread.out`, last year of old-store :00 bars)

- `edge_spread_series` tries `from bidask import edge_rolling`. On any exception it logs a warning
  and uses `_abdi_ranaldo_rolling` (`liquidity.py:181-190`).
- That function computes `2·sqrt(mean_35((ln C − (ln H + ln L)/2)²))` (`liquidity.py:215-248`).
  This is the *same-bar* squared form, which its own docstring calls "several-fold UPWARD-biased,
  a spread UPPER BOUND". The result is clipped to [0.02 %, 1.50 %] (`:56-57`).
- Measured:

  | name | fallback median | cap-hit | true AR cross-product (reference only) | RTH-only fallback | median RTH (H−L)/C |
  |---|---|---|---|---|---|
  | AMD | **1.50 %** | 65 % | 0.58 % | 0.84 % | 1.05 % |
  | NVDA | **1.50 %** | 73 % | 0.96 % | 0.68 % | 0.88 % |
  | TSLA | **1.50 %** | 77 % | 0.72 % | | |
  | GLD | 1.07 % | 28 % | 0.14 % | 0.25 % | 0.30 % |
  | COST | 0.92 % | | | 0.38 % | 0.48 % |
  | SOFI | 0.88 % | | | | |
  | PRME | 1.50 % | 81 % | | | |

  The fallback tracks the bar range: on hourly bars it is a volatility measure. Extended-hours
  bars push it to the cap.
- bidask cannot be evaluated without installing it (not present; `ModuleNotFoundError`).
- PyPI (`pypi.org/pypi/bidask/json`): latest **2.1.0** (2024-12-22), `py3-none-any` wheel,
  `requires_python <4.0,>=3.6`, `requires_dist ['pandas','numpy']` with no bounds. It satisfies
  `requirements-jetson.txt:25 bidask>=2.1` alongside `:14 numpy>=1.26,<2` / `:15 pandas>=2.0,<3`
  (installed: numpy 1.26.1, pandas 2.3.3, scipy 1.15.3). Use `--no-deps`.
- **The runbook requires** (Phase 0.2): `python -c "import bidask"`; if absent, STOP before any
  harvest.
- Caveat: KILL_LIST:90 records that hourly-EDGE level claims were once wrong by 70-95×. After the
  install, check that the stamped medians are plausible (a few bp for megacaps) before trusting
  the gate.

### 4. PIT masks and the panel

(`C_probe/probe_membership.out`)

- `_asof_tradability_mask` (`harvest_stock_data.py:374-395`) stamps `_DV30 = panel_ranks.dv30`.
  That is the trailing 30-day median of the **daily sum of hourly Close·Volume, extended hours
  included** (`panel_ranks.py:59-68`). It keeps rows with `_DV30 ≥ $5M` and `Close ≥ $3`.
- `_asof_membership_mask` (`:339-359`) keeps per-day ranks ≤ `AS_OF_TOP_K=60` over the whole
  92-name panel.
- `add_panel_ranks` (`panel_ranks.py:79-115`) then signed-ranks 17 base columns per timestamp,
  plus CS_Dispersion, CS_Breadth and MS_Interact (20 columns). `neutral_fill_cs` fills them, and
  `_DV30`/`DV30` are dropped (`harvest_stock_data.py:527-531`).
- **Old store (46 names, around 2026-02-19):** all 46 pass the $5M / $3 floors. Top-60 is vacuous
  on a 46-name panel. The old store's DV30 is also inflated 1.27× by the :30 duplicate rows.
- **Today (Alpaca SIP daily bars, 30-day median $-volume, all 92 names returned):**
  - Floors: 91/92 pass. Only **PRME** fails ($2.90, $9.6 M).
  - **Top-60 membership keeps only 22 of the 46 traded universe names.** The 60th-rank cutoff is
    about $866 M/day, and the 46 megacap candidates dominate.
  - Out today, with their ranks: ABNB 63, DASH 67, SOFI 68, IONQ 69, MARA 72, ASTS 73, OXY 74,
    RBLX 75, FSLR 76, ARKK 78, ROKU 79, AFRM 80, QBTS 81, COPX 82, SNAP 83, ENPH 84, RDW 85,
    CRSP 86, QS 87, POET 88, PPLT 89, PALL 90, SERV 91, PRME (floor).
  - Consequence: in the rebuilt store, about half the traded universe contributes **no recent
    rows**, and several names hover near the cut (ranks 61-70).
  - Daily bars include extended-hours volume just as the hourly sum does, so the ranking is the
    same order of magnitude as the harvest's own. The harvest's exact value needs the hourly
    fetch.

### 5. Sentiment source

(`C_probe/sentiment_db.out`; read with `mode=ro`)

- Opening the WAL-mode DB created an empty `-wal` and a `-shm` sidecar file. No data was written.
- `sentiment_cache.db` contents:

  | table | rows | range / detail |
  |---|---|---|
  | `articles` | 63,743 | dates 2020-07-22 → 2026-02-21, all fetched in one pass on 2026-02-21; 6,350 LLM-scored 2026-02-28 → 2026-04-04 |
  | `daily_sentiment` | 10,401 | **2025-02-26 → 2026-02-21**, 46 universe symbols only; 9,822 keyword / 578 llm / 1 mixed; per-symbol 58 (PRME) to 329 (COST) days |
  | `fng_daily` | 1,876 | 2021-01-05 → 2026-02-24, complete |
  | `state` | 1 | `live_mode=1` |

  Only 50 `daily_sentiment` rows exist after 2026-02-19.
- **What the stock harvest stamps.** It uses
  `fetch_stock_sentiment_history(..., cached_only=True)` (`harvest_stock_data.py:543-548`,
  `sentiment_history.py:311-322`) with a D−1 lag and `.get(…, 0.0)`.
  - So `Daily_Sentiment` is **0.0 (neutral) for the whole 7-month gap**, for everything before
    2025-02-27, and for all 46 candidates at every date.
  - Estimate: only about 5-8 % of the rebuilt stock rows carry a nonzero value, all in one
    12-month window.
  - `Daily_Sentiment` is in the `stationary` preset (`indicator_config.py`), while live serves
    real D−1 scores (`sentiment_history.py:533-537`). That is a train/serve distribution gap.
  - Step 0d (Finnhub keyword fetch, no LLM spend) closes the recent gap for universe names.
- **Crypto.** `fetch_crypto_sentiment_history` (`:188-265`) makes one network call to
  alternative.me for the missing days, **writes** them into `fng_daily`, and fills the gap with
  real F&G. It is unlagged by design.

---

## Risks

1. **Tail loss and grid re-contamination (stock).** Blocker 1 applies both to a full rebuild
   (ends 2026-06-30) and to every later single-chunk incremental: Alpaca fails, yfinance :30 bars
   come in, and D08(b) recurs even with the sidecar on. The same `end=now` pattern also affects
   the sidecar's `find_interior_gaps` repairs only if they reach `now` (they don't; they are
   bounded).
2. **Schema.** Not a merge risk, since features are always recomputed. The risk is (a) old
   contaminated OHLCV carried forward by incremental mode, and (b) columns that exist but are
   mostly neutral-filled until the archives back-fill: OI and SVR, plus sentiment as above.
3. **bidask.** Without it, `Eff_Spread_Pct` about equals the bar range, pinned at the 1.5 % cap
   for megacaps. The backtest gate and meta labels become meaningless. With it, confirm the
   levels (KILL_LIST:90 history).
4. **The L7 TB-span guard will likely fire.** The as-of masks now remove **interior** rows: names
   crossing rank 60 back and forth (ABNB, DASH, SOFI, IONQ at ranks 63-69), and the $3 floor
   (PRME, and QS/SERV/SNAP near $5). `_warn_tb_span_violation` / `_tb_membership_guard` will
   print `[TB-GUARD]`. That is the documented trigger for deferred item 8 (re-stamp after
   filtering, `06_signal_model_plan.md:333`). TB_Bars spans for those names would be invalid
   until the fix lands.
5. **RAM.** The stock rebuild is roughly 92 tickers × 20-41k rows ≈ 2.0-2.5 M pre-mask rows ×
   about 125 columns. Measured: 70 feature columns plus 21 TB columns per ticker, about 0.93 KB
   per row, so about 2 GB for `final_df`. `pd.concat` doubles that transiently, and the mask and
   rank passes copy again: a peak of perhaps 4-6 GB on a 7.4 GB box with 12 GB swap. The old
   harvest never did this (1.24 M rows × 54 columns). Run it alone. The CSV write of about
   2-3 GB is slow.
6. **yfinance alignment.** In a full rebuild, the stock branch uses no yfinance (verified
   `src={alpaca}`, :00 only). Crypto admits only 2-30 hole-fill rows per coin, mostly with zero
   volume. The D08 overwrite occurs only in incremental mode without `TRADER_YF_WINDOW_SLICE`.
7. **Crypto quirks carried into any rebuild.**
   - The SOL Alpaca hole: about 10k hours in 2023-24, over which features and returns jump.
   - XRP has only 2024+ history on Alpaca.
   - The untracked C extension leaks about 23 KB per `compute_*features` call. That is harmless
     at harvest scale. Its heap overflow needs frames shorter than 100 bars, which a full harvest
     never produces (Agent A).
8. **Model-facing flags.** Steps 7-8 turn on two Phase-3 flags. They must persist in the service
   environment, and the rebuild must be followed by the gotcha-#2 reset (study DBs moved,
   `best_score`/`cum_trials` reset) before any training.
