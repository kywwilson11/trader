# REVIEW_fixes — independent review of the 2026-09-26 fix round (FIX_A..H)

Reviewer: read-only. No edits, nothing staged. Scope: `git diff` of every production file plus
the untracked `tests/test_*_2026_09.py`. Method: I read every hunk against the FIX reports and
traced callers and consumers. Probes ran in the jetson env with `CUDA_VISIBLE_DEVICES=''`:
- import side-effects (torch / indicators_c in `sys.modules`);
- the `llm_client` model-gate helpers over 18 ids;
- the non-secret `llm_config.json` routing fields;
- `.env` key names only;
- `bash -n`.

I made no network calls. The full suite was not run.

Bottom line:
- **No order path, live flag or gate-decision logic changed.**
- **No tracked file was deleted.** `git status` shows no `D`/`R`, and the residue moves were
  untracked files.
- **No test was weakened in substance.** The two LLM price pins were value corrections; the
  fingerprint-split caveat is under L8.
- There are **three MEDIUM** findings. None blocks the Jetson rebuild, but M1 and M3 should be
  fixed before it runs.

---

## MEDIUM

### M1. The SIP clamp makes the stock store's last bar a partial, still-forming bar, which can trip the B15 merge guard on the next incremental
- **Where:** `market_data.py:646-649` (the clamp), `scripts/harvest_stock_data.py:153` (the guard),
  `data_utils.py:31,269` (`OVERLAP_DIVERGENCE_MAX = 0.01`).
- **Before the fix, this could not happen for stocks.** The final chunk was denied, and the weekly
  incremental fell back to yfinance.
- **What happens now:** stock requests end at `now-16min`. If the harvest runs during 04:00-20:00 ET
  on a weekday, Alpaca returns the bar whose open is ≤ end. That bar is still forming, and on the
  Basic plan it is also 15-min delayed. FIX_C's own crypto probe shows the same thing:
  `last=… current hour`.
- **Nothing drops it.** No harvest code removes an in-progress bar (`grep` finds
  `drop_forming_bar` only on the live path).
- **Consequence for the next incremental:** the 48h overlap compares that partial close with the
  final close. A >1% gap makes the guard **REFUSE the whole merge**, so the ticker keeps stale rows
  and prints "Full refetch required".
  - An hourly move above 1% within one bar is routine for SOXL/TQQQ and small caps.
  - The crypto harvest has no merge guard, so this is a stock-only, newly reachable path.
  - The same partial row also goes into the raw sidecar.
- **Status:** inferred from code plus the crypto live proof. Stocks could not be observed because
  the market is closed now (Saturday). To verify, run one read-only `fetch_historical_bars(api,'AMD',…,'stock')`
  during Monday RTH and look at the last bar's volume.
- **Fix:** in `fetch_historical_bars`, for stocks only, drop rows whose `open + 1h > clamp_limit`
  after the fetch. Or equivalently, floor the clamped `end` to `hour(now-16m) - 1s`. This keeps
  closed bars only, matching the `drop_forming_bar` contract for training windows.

### M2. After the one-time FnG migration, a failed alternative.me fetch silently writes Daily_Sentiment = 0.0 for every crypto training bar
- **Where:** `sentiment_history.py:278` (the migration runs before the fetch), `:303-314` (fetch
  error → `return result`, which is now empty), `scripts/harvest_crypto_data.py:362-372`
  (`.map(sentiment).fillna(0.0)`, then only a `filled` count is printed).
- **Before:** if the fetch failed, the legacy cache still served values. They were leaky, but the
  feature was not zero.
- **Now:** the first harvest after the fix empties `fng_daily` and then needs one 15-second HTTP
  call. If that call fails, the rebuild (a gotcha-#2 retrain) trains on an all-zero model-facing
  feature. The only sign is a log line.
- **Fix, either of:**
  - (a) Reorder so the migration is atomic with the refill. Fetch first; only if the fetch returns
    data, run move-aside + DELETE + INSERT + marker in one transaction.
  - (b) Make the crypto harvest fail the phase (exit non-zero) when sentiment coverage is below
    ~90% of bars.

### M3. D1 fixed the GUI order pagination, which now spends a large share of the shared Alpaca rate limit
- **Where:** `gui.py:1584-1628` (`fetch_orders`), default cadence `gui.py:391-396`
  (`orders` = 30 s, minimum 10 s).
- **Load:** every tick now walks up to 10 pages (`PAGE=100`, `HARD_ORDER_CAP=1000`). That is about
  20 req/min at the default cadence and up to about 60 req/min at the 10 s minimum. The account and
  positions timers add about 18 more.
- **Why it matters:** this is the same key the bots use, and Alpaca's limit is 200/min. Before the
  fix the walk died after page 1, so this load is new. FIX_F flagged it, but it is still unmitigated.
- **Fix:** do the full walk only at boot and then every ~10 min. In between, fetch page 1 only and
  merge it into the cached list by `id`.

---

## LOW

### L1. The llm_client auto-path validation retry ignores budget, cost cap and rate limit, and under-counts RPD
- **Where:** `llm_client.py:1550-1587`. The retry call is `_anthropic_post` at `:1565`.
- **What it skips:**
  - `_cost_ok()`, `get_budget()` and `_rate_limit_ok()`;
  - RPD counting: `record_call(model)` runs once per *returned* result (`:1751`), so a
    discard+retry cycle bills two requests and counts zero or one RPD. Discarded billed responses
    never count toward RPD at all.
- **Latency:** the analyst call is synchronous inside the trading cycle (`base_loop.py:389`,
  timeout `llm_analyst.py:40` = 45 s). The worst case is 2×45 s, plus the `call_llm` fallback, which
  can hit the same model again.
- **Exposure:** only Opus ≥5.5, Fable/Mythos ≥5.1 and unparseable ids take this path. None is
  configured (`provider=gemini`, `claude=claude-opus-4-6`, `openai=gpt-4.1`).
- **Fix:** check `_cost_ok()` before the retry, and count requests per call (return a request count
  so the caller can `record_call` n times).
- **Item-5 answers:**

  | Question | Answer |
  |---|---|
  | Fail-open on every error branch? | Yes. The first-post exception is caught by `call_claude` / `call_llm` → None. A retry-post exception returns `(None, usage)`. `max_tokens`/`refusal` → None. Two failed validations → None. Any extraction exception is caught by the caller → None, though usage is lost in that case. |
  | Double-billing? | No. The two requests' usage is summed and recorded exactly once, via either `_record_cost` or `_charge_discarded`. |
  | Budget respected? | No (above). |
  | Anything newer-only sent to older models? | No. Forced-capable ids get a byte-identical body. `strict:true`, the `auto` tool choice and the trailing instruction block go only to the auto path. |

  I verified the gates against the claude-api skill:
  - sampling is removed on Opus 4.7+, Sonnet 5 and Fable/Mythos;
  - forced tool use is removed on Opus 5.5 and Fable/Mythos 5.1;
  - prices and the 2× 1-hour cache write match the skill tables.

### L2. Non-canonical Claude ids are mis-routed and mis-priced
- **Where:** `llm_client.py:405-407` (`_CLAUDE_ID_RE`), `_provider_for` at `:100`. Probe results:
  - `us.anthropic.claude-opus-4-6` → provider **gemini**, priced $1.25/$10 (true price $5/$25, so
    the cap trips late), and routed to the auto path with temperature dropped.
  - `claude-haiku-4-5-latest` → auto path, priced at the $10/$50 ceiling.
- **Fix:** let the regex tolerate an optional platform prefix and `-latest`/`@date` suffixes, and
  have `_provider_for` match `claude` anywhere in the id.
- **Unverified pricing still open:**
  - The Gemini family ceiling (1.25/10) is below `gemini-3.1-pro-preview` (2/12); FIX_G flagged this.
  - Cache reads are over-billed at 0.10× for Opus 5.5 (0.05×) and Fable 5.1 (0.025×). That errs in
    the conservative direction.

### L3. Cost-accounting changes can move routing brackets (no gate logic changed)
- **Where:** `llm_client.py:694-697`. `get_recommended_model` picks a bracket from `_daily_cost`.
- **Why:** D6 now charges truncated or blocked Gemini responses, and the price table changed. Both
  move when the analyst steps down a tier or hits the $1 cap. This is more accurate accounting,
  not new gate logic, but it can change which model scores on heavy days.
- **Fix:** none needed. Record it in the owner review note.

### L4. setup_jetson_system.sh step 0 imports torch as root
- **Where:** `scripts/setup_jetson_system.sh:59-64` (runs after the EUID check).
- **Risk:** root-owned `__pycache__` entries in `/home/kyle/miniforge3/envs/jetson/...` wherever a
  `.pyc` is missing, plus `/root` caches.
- **Fix:** `sudo -u "$SUDO_USER" env PYTHONDONTWRITEBYTECODE=1 LD_PRELOAD=… "$PYBIN" -c …`.

### L5. Unit EnvironmentFile semantics
- **Where:** `setup_jetson_system.sh`, unit `EnvironmentFile=-…/.env`.
- **Change:** systemd now exports `.env` to the parent, and `trading_utils.py:20`
  `load_dotenv()` does not override existing variables. So edits to `.env` need a unit restart, and
  any future `TRADER_*` flag placed in `.env` reaches every child, including ones that never load
  dotenv.
- **Today:** `.env` holds only the 4 secrets. No behaviour change.
- **Fix:** add a line to the docs (runbook/FLAGS).

### L6. beta_ledger.drop_glitch_days edge cases
- **Where:** `beta_ledger.py:148-204`.
- **Edge cases:**
  - A glitch on the **first** observation becomes the reference. Every later normal day then
    "fails to revert" and is kept, so the glitch day is never dropped.
  - A NaN `profit_loss` on a dropped day turns `carry` into NaN, which poisons the next kept day's
    clean return.
- **Fix:** seed `last_good` from the median of the first 3 days; treat non-finite pl as 0 and add a
  warning.
- Measurement-only.

### L7. sizing_cofire_report `--json PATH` can leak its temp file on failure
- **Where:** `scripts/sizing_cofire_report.py:362-365`.
- **Issue:** if `json.dump` raises, `<PATH>.<pid>.tmp` is left behind. The other two new atomic
  writers use `finally` cleanup.
- **Fix:** wrap in try/finally with an unlink.
- No caller parses stdout apart from `tests/test_c26_S3.py`, and the bare `--json` behaviour is
  preserved.

### L8. Golden fingerprint split: CI now asserts less, and the numba overrides are unverified there
- **Where:** `tests/test_indicators_parity.py:402-480`.
- **The hash:** it is now asserted only on `(darwin, pure, pandas 3)` and `(linux, aarch64,
  pandas 2, numba)`. Both CI legs are x86_64 and always skip it. The column-sum (6 dp) and warm-up
  pins still run there, so the loss is modest, but it is a small coverage drop on CI.
- **The overrides:** the numba column-sum overrides were measured only on aarch64 and pandas 2.3.
  The py3.12 "modern" CI leg (pandas 3, unpinned numba) may disagree.
- **Fix:** record CI fingerprints from the first CI run, or loosen the numba sums to `rel=1e-9`.

### L9. Serve-side stock sentiment depends on the host timezone (pre-existing, unchanged)
- **Where:** `sentiment_history.py:639` uses `date.today()` (local time).
- **Risk:** the new harvest key `(t_utc-6h).date()-1` equals serve only on an America/Chicago host.
  If the Jetson's TZ changes (a UTC image, or a container), serve becomes up to one day newer than
  training for RTH bars.
- **Fix (owner, see FIX_H):** `stock_sentiment_lookup_dates([pd.Timestamp.now(tz='UTC')])[0]`, with
  `test_c26_P2` re-pinned.

---

## Item 3 — train/serve parity (explicit statement)

**Sentiment:**
- **Crypto:** harvest `fng_daily[UTC date of bar]` is, after the migration, the value published at
  that day's 00:00 UTC. Serve `get_fear_greed()` (`limit=1`) returns the same value. **Parity
  holds.** The only lag is a few minutes after 00:00 UTC, when serve may still return the previous
  day's value; that direction is safe.
- **Stock:** stock trading and prediction happen only in RTH (13:30-20:00 UTC).
  - For bars at 07-23 UTC the harvest key is unchanged (UTC date − 1), and it equals serve (Chicago
    date − 1).
  - The only rows whose key changed are the 00/01 UTC extended-hours bars (1.5% of rows), now D-2.
    That is correct and PIT-safe, and those hours are never served.
  - **Parity holds on the Chicago host** (caveat L9).

**SIP clamp:**
- **The live loops do not use `fetch_historical_bars`.** `predict_now.py:194`,
  `stock_loop.py:1566` and `base_loop.py:2386,3254` call `fetch_stock_bars_alpaca`. It sends no
  `end=`, starts at now−45 d, uses `.tail(320)`, and has `drop_forming_bar` behind `closed_only`.
  It is untouched.
- **Live features never read the training store,** so the clamp cannot create a harvest→live gap.
  The weekly incremental re-fetches from last bar − 48 h, so no interior hole is created either.
- **The only parity-relevant side effect is M1:** a partial final bar in the store, and the merge
  refusal that follows.
- **Pre-existing, unchanged note:** on the Basic plan, a live bar more than 1 h old can still be
  incomplete by up to 15 min of delayed prints, which `drop_forming_bar` does not consider.

## Item 4 — Jetson memory / CPU
- **No new heavy imports at module scope.** Checked by probe: after importing `sentiment_history`,
  `run_pipeline`, `llm_client`, `fundamentals`, `market_data` and `indicators`, `torch` is absent
  from `sys.modules`. `indicators_c` is not loaded and `_HAS_C` is False.
- **Hot paths:**
  - `base_loop` and `predict_now` were not touched.
  - `indicators` now always uses numba. That matches the C ext (A report: 0 NaN-placement
    mismatches, ≤1e-12 relative), so there is no value change and no speed loss.
  - `stock_sentiment_lookup_dates` over 1.24 M rows costs the same as the old list comprehension
    (~1-2 s, transient strings of order 100 MB, harvest-only).
- **GPU:** removing `CUDA_VISIBLE_DEVICES=` from the unit exposes the GPU to the parent and its
  non-training children (sentiment fetch/backfill, shadow's meta_label relaunch). None of them
  touch CUDA:
  - `hw_monitor.is_gpu_available` has no production caller;
  - backtest loads with `map_location='cpu'`;
  - importing `meta_label` does not import torch.

  Bots keep `BOT_ENV` `''`, so there is no new CUDA context in the live processes.
- **GUI:** the only new load is M3 (network, not memory).

## Item 2 — invariant audit
- **indicators.py:** backend C→numba by default. Not model-facing (verified parity above).
  Training that ran with the C ext and serving that now runs numba agree.
- **SIP clamp and sentiment PIT:** applied unflagged, and recorded as applied-in-the-rebuild
  (FIX_C, FIX_H). They are data-content changes that take effect only when a harvest runs.
- **llm_client:** the request body changes only for models that previously returned 400. For
  configured models the body is byte-identical (probe: `claude-opus-4-6` keeps sampling and forced
  tool use). Scores, parsing, prompts and fallback chains are unchanged. Accounting can move routing
  (L3).
- **run_pipeline:**
  - `_training_env` changes only an inherited *empty* `CUDA_VISIBLE_DEVICES`. Today it is unset, so
    this is a no-op.
  - `_update_per_bot_status` is status-only. Consumers: `run_pipeline.py:106` (a Telegram string)
    and GUI display.
- **gui.py:** display and measurement only. It places no orders. The gap-audit pool mirrors
  `stock_loop.py:263`'s `LEVERAGED_ETFS > 1` rule.
- **Tests:**
  - The two price-pin edits (`test_llm_claude.py:215`, `test_llm_providers.py:170-171`) correct
    the values to the published list prices; they are not weakenings.
  - The fingerprint split: see L8.
- **Nothing** places orders, flips a live flag, or deletes a tracked file.

---

## No issues found (per file)

| File | Checked | Result |
|---|---|---|
| indicators.py | opt-in gate, env var read at import only, `_HAS_C` false by default, no kernel change | no issues |
| market_data.py | clamp tz math (aware/aware), end_date bounds kept, crypto untouched, empty-chunk path, logger name has no shadowing, WARNING reaches stderr | only M1 |
| sentiment_history.py | migration txn (re-check under BEGIN IMMEDIATE, verify-before-DELETE, rollback), idempotency, `limit=0`, UTC bucketing, lookup-date tz handling | only M2, L9 |
| scripts/harvest_stock_data.py | key helper use, `min-2d` window, exception path → 0.0 | no issues |
| llm_client.py | lock order (`_quota_lock` → flock, no nested flock), midnight rollover, `_charge_discarded` single-charge at all 6 sites, `_fallback_price`, budget rows, 1-hour cache arithmetic reduces to the old formula when absent | L1-L3 only |
| gui.py | `NumericTableItem` (no recursion, None-safe), `_alpaca_until` (round-up plus the existing `prev_until` no-progress guard stops loops), `_engine_env`, ATR fill show/hide incl. `band=None` short-circuit, `_scroll_wrap`/`indexOf`, gap-audit pool/rank/cap | only M3 |
| run_pipeline.py | `_training_env` purity, `TRAIN_ENV` only at `run_phase`, `BOT_ENV` unchanged, `_BOT_SCOPE` global exists (`:1342`) | no issues |
| fundamentals.py | `_safe_float` never raises (bool/NaN/inf/str), numeric output byte-identical, 52wk skipped unless both are numeric | no issues |
| decision_report.py | `_dropped_null_pred` on every return path, `horizon_pending` stays non-negative, `representative` consumed for display only (gui/chart_core) | no issues |
| beta_ledger.py | CLI default-ON is measurement-only, `data_quality` keys, argparse validation | L6 only |
| execution_report.py | unknown-tactic split, atomic write with pid tmp + cleanup | no issues |
| llm_eval.py | mkstemp same-dir + replace + cleanup, `MIN_POWER_*` constants exist, verdict still starts `insufficient_power` | no issues |
| scripts/sizing_cofire_report.py | nargs='?' semantics, exit codes, no GUI caller | L7 only |
| chart_core.py | `api_available` None-vs-False split matches `decision_report._write_stale_report` (`:794`/`:805`) | no issues |
| scripts/setup_jetson_system.sh | `bash -n` OK, step 0 before any change, unit values = `run_pipeline.ENV`, OOMPolicy, swap message | L4, L5 |
| .gitignore | two anchored `archive/local_residue/` patterns, no broad rule | no issues |
| llm_config.py | docstring-only | no issues |
