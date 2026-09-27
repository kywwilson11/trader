# H — `Daily_Sentiment` point-in-time audit (2026-09-26, Jetson, read-only)

**Verdict: LEAK CONFIRMED in both old training stores. The cause is not the current code. The
crypto leak will survive a re-harvest with today's code unless the SQLite cache is repaired.**

- **Crypto store** (`training_data.parquet`, built 2026-02-28). Every bar on UTC day D carries the Fear & Greed
  value that alternative.me publishes at **(D+1) 00:00 UTC**. That is 1–24 h of lookahead on 100 % of rows.
  The stamped value is largely a function of day D's own price move: corr(ΔX_D, BTC ret_D) = **0.66**.
- **Stock store** (`stock_training_data.parquet`, built 2026-02-28). Bars on UTC day D carry the **same-day,
  unlagged** aggregate. Articles in it are bucketed by America/Chicago date, so it includes news up to
  D+1 ~06:00 UTC. The rest of the session, the close, and after-hours/earnings news are all in the value.
  About 9.8 % of rows are non-zero.
- **Both deployed models** (`model_v2.pth` 2026-04-11 and `stock_model_v2.pth` 2026-04-25) have
  `Daily_Sentiment` in their 23 feature columns (`feature_cols_v2.pkl`, `stock_feature_cols_v2.pkl`, preset
  `stationary`). The store mtime (2026-02-28) predates both models, and no later harvest wrote the stores.
  So they were almost certainly trained on the leaked values.

Artifacts: scripts `<scratch>/h_sent/align.py`, `<scratch>/h_sent/corr.py`; full table
`<scratch>/h_sent/corr_results.csv`, raw output `<scratch>/h_sent/corr_out.txt`.

---

## 1. How the feature is stamped: code paths and alignment rules

### 1a. Code that built the stores on disk (2026-02-28 = commit `a2e8788`, 2026-02-21)
The PIT fixes arrived later, in `5e23096` (2026-06-10, "honest model validation"). `git show a2e8788:…`:

| Path | Code at a2e8788 | Alignment rule actually applied |
|---|---|---|
| FnG cache insert | `sentiment_history.py:232` `datetime.date.fromtimestamp(ts)` (**local tz**, box = America/Chicago) | alternative.me stamps a value `00:00 UTC day D+1` = `18:00/19:00 Chicago day D`. It is cached as **date D** |
| Crypto harvest | `harvest_crypto_data.py:181-183` maps `index.date` (UTC; `Datetime` is `timestamp[ns, tz=UTC]`) → `fng_daily[D]` | bar at UTC t on day D → **value published at D+1 00:00 UTC** (future) |
| Article insert | `sentiment_history.py:406` `datetime.date.fromtimestamp(article_ts)` (**local tz**) | article dated by its Chicago calendar date |
| Stock harvest | `harvest_stock_data.py:175-178` `sentiment.get((ticker, str(date)))`, **no lag** | bar on UTC day D → aggregate of all articles with Chicago date D, i.e. published up to **D+1 ~05:00–06:00 UTC** |
| Live crypto | `get_live_daily_sentiment` → `sentiment.get_fear_greed()` → `data[0]` (`sentiment.py:979`) | serve at t → value published at **D 00:00 UTC** (PIT, but **one day behind what training taught**) |
| Live stock | `get_daily_sentiment(clean, date.today())` (a2e8788:492) | serve → the same-day aggregate, but the cache stopped at 2026-02-21 (see §3), so it served **0.0** |

### 1b. Current code (HEAD `438f56a`)
| Path | Line | Rule |
|---|---|---|
| FnG insert | `sentiment_history.py:239-246` | UTC bucketing: `fng_daily[D]` = value published at D 00:00 UTC. **Applies only to new inserts**. Cached dates are skipped (`:249-250`) and never rewritten |
| Crypto harvest | `scripts/harvest_crypto_data.py:360-368` | bar on UTC day D → `fng_daily[D]` (unlagged, correct **if** the cache is UTC-dated) |
| Article insert | `sentiment_history.py:433-441` | UTC date of Finnhub `datetime`. Timestamp **not stored**, only the date string |
| Stock harvest | `scripts/harvest_stock_data.py:533-550` | bar on UTC day D → `daily_sentiment[(ticker, D-1)]` (calendar D-1: Monday gets Sunday) |
| Live injection | `predict_now.py:298-318` | one scalar written to **every row of the window** (`df['Daily_Sentiment'] = value`) |
| Live crypto | `sentiment_history.py:519-530` + `sentiment.py:965-988` (5-min TTL, `limit=1`) | latest published FnG = D 00:00 UTC value |
| Live stock | `sentiment_history.py:531-535` | `daily_sentiment[(sym, local_today-1)]`. During RTH, Chicago date = UTC date, so this equals training's D-1 |

**Parity (current code).** Harvest and serve rules agree for both books: crypto uses day-D FnG, stocks use
D-1. **But the data under the crypto rule is still wrong** (§2a), so a fresh crypto harvest today would
reproduce the leak. There are three smaller parity gaps:
1. **Constant-window injection.** Live injects one scalar across all `seq_len` rows (`predict_now.py:314`).
   Training windows that span 00:00 UTC hold two different daily values, and the LSTM sees that step. This
   is a small mismatch, and it goes in the safe direction.
2. **Crypto partial-history fetch.** `sentiment_history.py:229` requests `limit={total_days}`, and
   alternative.me counts that back **from today**, not from `end_date`. A historical range ending before
   today silently loses its earliest days. This is secondary, but relevant to any cache rebuild.
3. **Stock cache is frozen at 2026-02-21.** Every one of the 63,743 articles has `fetched_at` = 2026-02-21.
   Harvest reads `cached_only=True`, and only the manual `python sentiment_history.py --fetch-stocks`
   CLI (`:973-976`) fetches. Consequences:
   - Live stock serving returned **0.0** for every day from 2026-02-22 through the 2026-05-07 bot stop.
   - A re-harvest would stamp 0.0 on every bar after 2026-02-22.
   - Non-zero coverage in the stock store spans only **53 weeks** (the Finnhub free-tier ~1 y).

   That is a train/serve skew even without any lookahead.

## 2. Measurements on the real stores

### 2a. Direct alignment test (decisive, no statistics needed)
| Check | Result |
|---|---|
| Crypto store `Daily_Sentiment[D]` == `fng_daily[D]` | **1.0000** (234,678 rows). D-1 gives 0.119, D+1 gives 0.119 (≈ chance: day-to-day identical-value rate is 0.118) |
| `fng_daily[D]` (cache) == alternative.me value stamped `D 00:00 UTC` | 228/1875 (≈ chance) |
| `fng_daily[D]` (cache) == alternative.me value stamped **`(D+1) 00:00 UTC`** | **1875/1875** |
| alternative.me stamp times | all `00:00` UTC. `time_until_update`=80578 s at 01:36 UTC, i.e. the value rolls at 00:00 UTC, as documented |
| Stock store (non-zero rows) == `daily_sentiment[D]` (same day) | **0.9763** (121,180 rows). D-1 gives 0.0005, D+1 gives 0.0005. The 2.4 % residual is post-build LLM re-aggregation (`llm_scored_at` runs 2026-02-28 → 2026-04-04) |
| Stored article date vs Finnhub `datetime` (NVDA/AAPL/TSLA, 2026-02-09..20, re-queried read-only) | 422 headlines matched: stored == **Chicago date 416**, == UTC date 357. Of the 61 articles whose UTC and Chicago dates differ, ~59 are stored under the Chicago date |
| Day boundary: UTC hour where the stamped value changes | crypto: 8,605 changes at 00 UTC, ~90 elsewhere (first bar after a data gap), 0 days with >1 value. Stock: first bar of the UTC day (00 / 08–13 UTC), 0 ticker-days with >1 value. The change hour matches the documented 00:00 UTC rule. **The content is a day early** (crypto) or same-day (stock) |

### 2b. Forward vs backward correlation by UTC hour (pooled Pearson)
Returns are clock-hour, per ticker, winsorized at 0.5/99.5 %. 95 % CI from a **weekly-block bootstrap**
(2,000 reps over 268 crypto weeks and 527 stock weeks; 53 weeks for the stock non-zero subset).
- `X_asis` = the stored value.
- `X_lag1` = the value one day earlier. For crypto, `fng_daily[D-1]` = the correctly-PIT value published
  at D 00:00 UTC. For stocks, the current harvest rule `daily_sentiment[D-1]`.

**Crypto** (6 coins, 234,810 rows):

| bucket | X_asis fwd4 | fwd12 | fwd24 | bwd24 | X_lag1 fwd4 | fwd12 | fwd24 | bwd24 |
|---|---|---|---|---|---|---|---|---|
| 00–05h | .079 [.053,.104] | .122 | **.157** [.124,.190] | .151 | .026 [−.000,.053] | .030 | .041 [.002,.078] | .154 |
| 06–11h | .055 | .124 | .128 | .157 | .008 | .021 | .041 | .126 |
| 12–17h | .082 | .103 | .096 | .171 | .015 | .026 | .040 | .096 |
| 18–23h | .054 [.025,.083] | .059 | **.054** [.016,.094] | .181 | .020 | .041 | .040 | .056 |
| all | .068 [.054,.083] | .103 | .109 [.075,.142] | .165 | .017 [.002,.032] | .029 | .040 [.002,.076] | .108 |

This is the lookahead signature on both tests:
- The as-is forward-24h correlation falls **monotonically** from .157 for bars early in the UTC day to
  .054 for bars late in it. Early bars have more of "their" day ahead of them.
- After the one-day shift it is **flat at ~.04** across buckets, with CIs touching 0 at 4–12h.
- The as-is 4h IC range (+.054…+.082) reproduces D_measurement's +.041…+.083.

**BTC daily-level test** (1,868 days). This is the cleanest view:

| | ret D−1 | ret D (same UTC day) | ret D+1 |
|---|---|---|---|
| Δ stamped value (ΔX_asis) | −.068 | **.659 [.623,.690]** | −.045 [−.102,.009] |
| Δ PIT value (ΔX_lag1) | **.659 [.625,.691]** | −.044 | .013 |
| level X_asis | .181 | .201 | .013 [−.034,.059] |
| level X_lag1 | .201 | .013 | .026 [−.025,.074] |

The stamped change tracks the return of the day it is stamped on, which is unknowable until 24:00 UTC. The
correctly-dated series tracks the *previous* day's return, which is what F&G legitimately encodes
(momentum/volatility inputs). Neither series predicts day D+1 (both CIs span 0).

**Stock** (46 names, 1,236,699 rows; the zeros dominate the all-rows view, so the non-zero subset is shown too):

| subset / bucket | X_asis fwd4 | fwd12 | bwd24 | X_lag1 fwd4 | fwd12 | bwd24 |
|---|---|---|---|---|---|---|
| all rows | .015 [.009,.022] | .019 | .042 | .002 [−.003,.008] | .003 | .003 |
| non-zero, all hours | .055 [.034,.077] | .080 [.050,.110] | .141 | .011 [−.005,.027] | .027 [.006,.048] | .021 |
| non-zero, RTH 13–16 UTC | .031 | **.084** [.054,.115] | .156 | −.014 | .038 [.016,.061] | .037 |
| non-zero, RTH 16–20 UTC | .079 [.049,.107] | .085 | .142 | .054 [.033,.076] | .057 [.034,.081] | .011 |

- As-is forward IC of .055–.085 on the covered rows collapses to ~.01–.04 after the lag.
- As-is bwd24 of ~.14–.16 is the same-day news/price co-movement baked into the value.
- One residual: `X_lag1` keeps a real-looking forward IC of ~.055 in late RTH (16–20 UTC, 4–12 h, which
  crosses the close/overnight). That is not a leak: the D-1 Chicago-dated aggregate ends by D ~06:00 UTC,
  hours before those bars. It is plausibly post-news drift. It rests on 53 weeks, so treat it as a
  hypothesis, not a signal.

## 3. `sentiment_cache.db` (read-only, `mode=ro`)
- **Schema** (`sentiment_history.py:35-73`):
  - `articles(id, symbol, date TEXT, headline, summary, url, keyword_score, llm_score, fetched_at, llm_scored_at, UNIQUE(symbol,date,headline))`
  - `daily_sentiment(symbol, date, score, article_count, llm_count, score_type, PK(symbol,date))`
  - `fng_daily(date PK, value, score)`
  - `state(key, value)`
- **Counts.** 63,743 articles across 46 symbols, dated 2020-07-22 → 2026-02-21, **all fetched
  2026-02-21T21:09** with the old code. 6,350 have LLM scores (Gemini backfill 2026-02-28 → 2026-04-04).
  There are 10,401 daily rows (9,822 keyword, 578 llm, 1 mixed) and 1,876 `fng_daily` rows (2021-01-05 →
  2026-02-24). `state` holds only `live_mode=1`.
- **Article timestamps are NOT stored.** Only a calendar `date` string, and every existing row is
  Chicago-dated. The per-article time cannot be recovered from the DB. It can be recovered only by
  re-querying Finnhub, which covers only ~1 year back on the free tier.
- **Sample.** NVDA 2026-02-13 has 99 articles, the 14th 77, the 15th 71, then nothing on 02-16..18 (a
  coverage hole), the 19th 61, the 20th 139. The daily aggregate for date D includes every article whose
  **Chicago** date is D. So yes: a D-stamped value contains articles published after every bar of UTC day D
  that it was joined to in the old store. Under the current D-1 rule, it contains articles up to D ~06:00
  UTC. That is PIT-safe for pre-market and RTH bars. It is **not** safe for the 18,551 stock bars stamped
  00:00 UTC (the prior US evening's extended-hours bar), which can see up to ~6 h of later articles.
- **Second, non-temporal lookahead channel (not measured).** `llm_score` was assigned in Feb–Apr 2026 to
  articles from 2025. The scoring LLM may already know how those stories resolved (Glasserman & Lin 2023,
  *Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis*). Only 578
  daily rows are LLM-typed, so the exposure is small. Record it; do not act on it now.

## 4. Verdict and fix

**LEAK CONFIRMED** in both stores that are on disk and that trained the deployed models.
- Crypto: 100 % of rows carry the next-day-published F&G, and ΔX correlates 0.66 with the same-day return.
- Stock: the non-zero rows carry the same-day aggregate, Chicago-bucketed into the next UTC morning.

The `5e23096` code fixes are logically correct but **do not reach the data**:
- **Crypto.** The FnG cache keeps its local-tz dates forever: `fetch_crypto_sentiment_history` returns
  cached dates and only inserts missing ones. The next harvest would still leak for every date
  ≤ 2026-02-24, and would add a seam (a duplicated value) at 02-24/02-25.
- **Stock.** The one-day lag neutralises the Chicago dating for every bar except the 00:00 UTC bar.

### Fix (model-facing: feature values change)
It belongs in the upcoming clean re-harvest + retrain, which is a gotcha-#2 event anyway: delete the
`*_study.db` files and reset `best_score`. It needs a PIT test. Owner decision; not implemented here.

**(1) Crypto: repair the FnG cache once, and fetch enough history** (`sentiment_history.py`, in
`fetch_crypto_sentiment_history`, right after `db = _get_db()` at line 200, and at line 229):
```diff
     db = _get_db()
+    # One-time repair (H audit 2026-09-26): rows cached before 5e23096 were
+    # bucketed in LOCAL time, so fng_daily[D] holds the value alternative.me
+    # published at (D+1) 00:00 UTC (verified 1875/1875). The UTC fix only
+    # governs NEW inserts; cached dates are skipped, so re-harvest re-leaks.
+    basis = db.execute("SELECT value FROM state WHERE key='fng_date_basis'").fetchone()
+    if not basis or basis[0] != 'utc_publication':
+        db.execute("CREATE TABLE IF NOT EXISTS fng_daily_legacy_localtz AS SELECT * FROM fng_daily")
+        db.execute("DELETE FROM fng_daily")
+        db.execute("INSERT OR REPLACE INTO state (key, value) VALUES ('fng_date_basis','utc_publication')")
+        db.commit()
@@
-            f'https://api.alternative.me/fng/?limit={total_days}&format=json',
+            # limit counts back from TODAY, not from end_date
+            f'https://api.alternative.me/fng/?limit=0&format=json',
```
The legacy table is kept, not dropped (delete-nothing). Also write the `fng_date_basis` marker whenever a
fresh DB is created, so the repair runs exactly once. An equivalent exact repair is
`UPDATE … SET date = date(date,'+1 day')`, because the shift is verified to be exactly one day. Refetching
from the API is simpler and authoritative. The harvest's same-day map (`harvest_crypto_data.py:366-368`)
is then correct as written, and matches live serving (`sentiment.py:979`, `data[0]`).

**(2) Stock: cover the legacy Chicago-dated articles for pre-06:00-UTC bars** (`scripts/harvest_stock_data.py:545-548`):
```diff
-        final_df['Daily_Sentiment'] = [
-            sentiment.get((ticker, str(date - _dt.timedelta(days=1))), 0.0)
-            for ticker, date in zip(final_df['Ticker'], final_df.index.date)
-        ]
+        # Legacy articles are America/Chicago-dated, so day D-1's aggregate can
+        # include articles up to D ~06:00 UTC. Key on (t - 6h) so a bar before
+        # 06:00 UTC (the 00:00 UTC extended-hours bar) falls back one more day.
+        _key = (final_df.index - pd.Timedelta(hours=6)).date
+        final_df['Daily_Sentiment'] = [
+            sentiment.get((ticker, str(date - _dt.timedelta(days=1))), 0.0)
+            for ticker, date in zip(final_df['Ticker'], _key)
+        ]
```
The serve side needs no change (live stock inference runs in RTH, where the rule is D-1). About 1.5 % of
stock rows change.

**(3) Before the re-harvest, decide the stock feature's fate.** The article cache has been frozen since
2026-02-21.
- If `--fetch-stocks` is not run, a re-harvest stamps 0.0 on every recent bar. That includes the entire
  holdout, and the live value is 0.0.
- In that case the feature is dead weight with train/serve skew, and it is better dropped from the stock
  preset.
- The alternative is a scheduled refresh (owner decision; kill-list check needed).

(Also recommended: store the article `published_at` epoch going forward, and add a PIT unit test asserting
`fng_daily[D]` == the API value stamped D 00:00 UTC on a fixture.)

### Were the old models trained on leaked values? Yes, almost certainly.
Both deployed artifacts list `Daily_Sentiment` among their 23 features. They were trained in April on the
only stores that exist, the Feb-28 builds; the store mtimes show no later harvest. For crypto the leak is
strong:
- Every row sees a value whose day-over-day change correlates 0.66 with the same day's return.
- An 18-bar LSTM window that crosses midnight sees that step directly.
- The 64-bar label window overlaps the leaked remainder of the day.

At serve time the crypto model received the PIT value, which is exactly the series training had
shifted by one day. So the learned mapping could not transfer. The stock model got the same-day aggregate
on about 10 % of rows in training and 0.0 live.

This plausibly explains **part** of the spring's optimistic in-sample/holdout scores. The holdout is cut
from the same leaked store, so a DSR/holdout gate could not catch it. The size of the contribution is
unmeasured. The cheap way to measure it (Jetson, not run here, because training is out of scope) is a
LightGBM ablation on the old store: X_asis vs X_lag1 vs dropped, same folds, compare holdout IC/DSR.
