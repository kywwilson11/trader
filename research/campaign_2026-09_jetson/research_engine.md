# ENGINE research scout — 2026-09-27 (append-only)

**Purpose.** What is NEW (2025–2026) or was MISSED by `docs/MAP.md` §9 and
`research/campaign_2026-08/05_frontier_research.md` for the live ENGINE — venue facts, execution,
sizing/vol, crypto microstructure, Jetson inference memory, and process supervision — turned into
experiments with pre-registered decision rules. Nothing here is shipped; model-facing items are owner
asks. Every web claim cites a source row below (all accessed 2026-09-27); every device claim cites a
local measurement (M-rows) run on the prod Jetson today, read-only (no orders; Alpaca calls were
GET-only). `research/KILL_LIST.md` and `08_removed_code.md` were checked per topic. Items marked
**unverified** could not be confirmed from a primary source or a local run.

## Sources (accessed 2026-09-27)

| id | title / venue / author | date | URL |
|---|---|---|---|
| S1 | Alpaca Docs, "Crypto Spot Trading" (fee tiers, order types, TIF) | undated page | https://docs.alpaca.markets/docs/crypto-trading |
| S2 | Alpaca Docs, "Crypto Orders" | undated page | https://docs.alpaca.markets/docs/crypto-orders |
| S3 | Alpaca Docs, "Real-time Crypto Data" (locations us / us-1 / eu-1) | undated page | https://docs.alpaca.markets/docs/real-time-crypto-pricing-data |
| S4 | Alpaca changelog, "Add support for us-1 crypto market data location" (fetch failed; search snippet only) | unverified | https://docs.alpaca.markets/changelog/add-support-for-us-1-crypto-market-data-location |
| S5 | Alpaca forum, "Crypto market data from v1beta3 is way to slow" | 2024-12-30; staff reply 2025-01-13 | https://forum.alpaca.markets/t/crypto-market-data-from-v1beta3-is-way-to-slow-what-to-replace-with/15752 |
| S6 | Alpaca forum, "Avg Entry and Cost Basis not calculated on a particular crypto position [PAPER]" | 2025-08-12 → fixed 2025-08-28 | https://forum.alpaca.markets/t/avg-entry-and-cost-basis-not-calculated-on-a-particular-crypto-position-paper/17618 |
| S7 | Alpaca forum, "Crypto positions disappeared from my paper trading account 5 Nov 2025" | 2025-11-05 … 2026-09-16 | https://forum.alpaca.markets/t/crypto-positions-disappeared-from-my-paper-trading-account-5-nov-2025-at-404am-et-position-does-not-exist/18104 |
| S8 | Alpaca-py, "Migration from alpaca-trade-api-python" | undated | https://alpaca.markets/sdks/python/migration.html |
| S9 | Albers, Cucuringu, Howison, Shestopaloff, "The Market Maker's Dilemma", arXiv 2502.18625 | v1 2025-02-25, v2 2025-11-23 | https://arxiv.org/abs/2502.18625 |
| S10 | Li, Laryea, Ihlamur, "Optimal Stop-Loss and Take-Profit Parameterization for Autonomous Trading Agent Swarm", arXiv 2604.27150 | 2026-04-29 | https://arxiv.org/abs/2604.27150 |
| S11 | Ryan, "Conformal Kelly", arXiv 2608.01494 | 2026-08-02 | https://arxiv.org/abs/2608.01494 |
| S12 | Marshall (Amberdata), "The Rhythm of Liquidity" | 2025-09-02 | https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth |
| S13 | "A Predictive Framework Integrating Multi-Scale Volatility Components…", arXiv 2507.22409 | 2025-07-30 | https://arxiv.org/html/2507.22409v1 |
| S14 | PyTorch 2.10 Release Blog ("Torchscript is now Deprecated") | released 2026-01; page upd. 2026-07-23 | https://pytorch.org/blog/pytorch-2-10-release-blog/ |
| S15 | LWN, "Systemd 254 released" (RestartSteps= / RestartMaxDelaySec=) | 2023-07-28 | https://lwn.net/Articles/939511/ |
| S16 | Ubuntu 22.04 (jammy) manpage loginctl(1), enable-linger | jammy | https://manpages.ubuntu.com/manpages/jammy/man1/loginctl.1.html |

## Local measurements (prod Jetson, 2026-09-27; scripts + raw output in the ENGINE w3 scratch dir)

| id | what | result |
|---|---|---|
| M1 | `torch_mem.py` (hwlock): smaps_rollup after each import | numpy+pandas 85 MB RSS → +lightgbm 144 → **+torch 475 MB** (+331 MB RSS, +208 MB private-dirty anon) → + model built/traced 505 MB. torch 2.8.0, `quantized.supported_engines=['qnnpack']`, `mkldnn` unavailable on this aarch64 wheel |
| M2 | `np_lstm.py`: numpy replica of `model_v2.RegressionLSTM.forward` (2-layer LSTM h=128 + 4-head MHA + LayerNorm + mean-pool + MLP), 200 random windows, T=48, D=40 | 293,761 params (~1.2 MB fp32). max abs Δ vs torch: **1.6e-7 (float32), 1.3e-7 (float64)**. Latency: torch eager 22.5 ms, jit 18.6 ms, numpy-fp32 53.8 ms per forward. `quantize_dynamic` qint8 (qnnpack): max Δ **3.6e-3**, median 1.1e-3 |
| M3 | `quote_age_probe.py` (24 polls × 10 s, 05:24–05:28 UTC Sun) + `loc_probe.py` (3 rounds, us vs us-1) | loc=us quote-timestamp age >180 s in **ETH 12/24, SOL 18/24, DOGE 2/24** polls (max 486 s, SOL); BTC/LINK/XRP 0/24. ETH age 282→303→323 s with an unchanged 1.94 bps spread. loc=us-1 (Kraken) ages 0.1–13.5 s for all six. Median quoted spread us vs us-1 (bps): BTC 2.4 vs 0.01, ETH 1.9 vs 0.04, SOL 8.7 vs 0.83, LINK 19 vs 2.6, XRP 35 vs 0.07, DOGE 35 vs 0.01 |
| M4 | `asset_probe.py` GET /v2/assets (paper) | all six active; `price_increment` and `min_trade_increment` = 1e-9; `min_order_size` ≈ $1 notional (e.g. DOGE 10.37) |
| M5 | `wick_census.py` / `badprint.py` on `training_data.parquet` (263,889 rows, 2021-01-05 → 2026-09-25) | P(intra-hour Open→Low drop ≥2 / ≥4 / ≥8 %) post-2025-10-10 per name-hour: BTC 1.57/0.20/0.012 %, ETH 2.33/0.29/0.012, SOL 3.36/0.41/0.048, LINK 3.14/0.46/0.060, XRP 2.21/0.41/0.048, DOGE 3.51/0.55/0.155 (all lower than pre-cutoff). ≥4 % drops fall on weekends 19.3 % vs 28.5 % of hours. **≥40 wick-only bars (Low ≥15 % below Open, |Close/Open−1| <3 %)** incl. BTC Low $8,200 vs Open $65,606 (2021-10-21 11:00), a run of BTC lows ~$34.5k vs ~$42–43k opens (2023-12 → 2024-01), DOGE $0.0975 vs $0.219 (2025-10-10 21:00); 11 of the listed bars carry `TB_Reason_24 = 1` (hard_stop) |
| M6 | host facts | systemd **249** (Ubuntu 22.04.5); `loginctl show-user kyle` → `Linger=no`, user manager running; polkit `org.freedesktop.login1.set-self-linger` defaults **allow_any=yes** (`/usr/share/polkit-1/actions/org.freedesktop.login1.policy:127-134`; `/etc/polkit-1/localauthority` unreadable → override status unverified); cgroup2, `user@1000.service` delegated controllers = **memory pids**; supervisord/pm2/onnxruntime not installed |
| M7 | log census `pipeline_output.log` | **1,778** "Stock bot crashed (exit 1), restarted" lines (lines 5387–9269), every one the same `ValueError: Unknown format code 'f' for object of type 'str'` at `fundamentals.format_fundamentals_for_llm` in `stock_bot_output.log` (source line since fixed: `fundamentals.py:367` uses `_fmt_num`) |

---

## E1 — Alpaca venue facts 2025–26

**E1.1 Fee schedule and order types (confirms code).** S1 lists 8 volume tiers from 15/25 bps
(maker/taker, 0–100k 30-day USD) down to 0/10 bps (>100M); crypto order types are market, limit and
stop_limit, TIF gtc and ioc only (S1, S2); no trailing-stop order type is listed for crypto (S1).
- Applies: yes. `fees.py:40-42` (25/15 tier 1) matches S1; the $100k paper book never leaves tier 1
  at current turnover (judgment). `crypto_loop.py:181-196` GTC stop_limit is a supported type (S2).
  M4: paper `price_increment` 1e-9, so `crypto_loop.py:172-173` `_round_px` (4–6 dp) is never
  rejected on increment grounds. No experiment needed. Kill-list: nothing related.

**E1.2 Crypto quote publication — the 180 s staleness rule rejects live, quiet books (NEW).**
Alpaca documents no publication cadence (S3: silent); S3 says Alpaca executes crypto orders "in its
own exchange, and also supports Kraken", with loc `us` = Alpaca and `us-1`/`eu-1` = Kraken; a forum
user reported 5 BTC trades in 20 minutes on v1beta3 and staff pointed to quotes instead (S5). M3 shows
the `us` quote timestamp only advances when the BBO changes: in a Sunday 05:24 UTC window the ETH and
SOL quotes were older than 180 s in 50 % and 75 % of polls while the spread was unchanged, and Kraken
`us-1` quotes were never older than 14 s. "On-change" semantics are **inferred from M3, unverified in
Alpaca docs**; the sample is one 4-minute window.
- Applies: yes. `order_utils.py:164` returns None when age >180 s; in `base_loop.py:1293-1295` a
  None quote `continue`s past the whole local exit stack (trailing, TP, vertical, signal) for that
  symbol for the cycle; `base_loop.py:1979-1981` skips signal sells; `base_loop.py:3016` blocks
  entries (fail-closed, correct). The resting GTC stop_limit still guards the hard stop.
- **Experiment X1 (measurement-only, runbook Phase 0).** Add a quote-age column to
  `scripts/crypto_spread_census.py` (it polls the same endpoint, `:30`, but records no age;
  args `:122-130`) and run it 24 h on a weekday and 24 h on a weekend at `--interval 10`, `--loc us`
  and `--loc us-1`. Also count `[QUOTE] … stale, ignoring` lines per symbol in `logs/trader.log`
  once bots run. Pre-registered rule: if any of the six names has >5 % of `us` polls aged >180 s in
  either window → owner item for a default-OFF liveness rule (proposed name
  `TRADER_CRYPTO_QUOTE_LIVENESS`: treat a `us` quote as live when its bid/ask equal the previous poll
  and the `us-1` quote is <60 s old); if every name is <1 % in both windows → close X1. Note: today's
  `logs/trader.log` stale lines (2026-09-26/27) are test-fixture output (round 300/600 s ages), not
  production evidence.
- Kill-list: checked. Not "Order-book imbalance filters" [wave-2] (no book-state signal is
  proposed); this is feed hygiene, not alpha.

**E1.3 Which venue's spread is the true cost (NEW).** M3: Alpaca `us` spreads are 35 bps on XRP/DOGE
and 19 bps on LINK against Kraken `us-1` spreads under 3 bps. Whether Alpaca paper/live crypto orders
fill against the `us` book or are routed to Kraken liquidity is **unverified** (S3 wording is ambiguous).
- Applies: yes — `fees.py:184`/`:243-246` and `liquidity.py` build crypto costs from `us` quotes;
  kill-list pending ask #1 (`TRADER_CRYPTO_SPREAD_STAMP`) depends on it.
- **Experiment X2 (measurement-only, Phase 1 once n≥30 crypto fills).** At each crypto entry also
  fetch the `us-1` quote (one extra GET) and journal `mid_us`, `mid_us1`, `spread_us`, `spread_us1`
  next to the existing `quote_age_s` (`base_loop.py:1899-1913`). Rule: over ≥30 fills, compute
  MAE of realized entry slippage (`execution_report.py:174-188`) against ½·spread_us and against
  ½·spread_us1; the predictor with ≥25 % lower MAE is the one the crypto spread stamp must use. If
  neither wins by 25 %, no change. Kill-list: checked; stays inside ask #1's "stamp" half, builds no
  passive-fill simulator ([wave-7] Honest-OHLC entry).

**E1.4 Paper-venue incidents that corrupt engine state (NEW, 2025–26).** S6: Aug 12–28 2025, paper-only,
some crypto positions (BTC, DOT) had no avg entry / cost basis (two asset ids for BTC/USD). S7: paper
crypto positions vanished 2025-11-05 (fixed 2025-11-17), again 2026-01-28, and new reports 2026-09-16
(a batch sync job between internal databases).
- Applies: yes, paper-only. (a) `base_loop.py:3270` sets `fill_price = float(pos.avg_entry_price)`
  with no fallback to the order's `filled_avg_price`; if the broker returns 0, the position is stored
  with `entry_price=0` (`:3323`), `tp_price` stays None (`:3275`), and `crypto_loop.py:205` places a
  resting stop at 0·(1−d) = 0. The API value during S6 (0 vs null) is **unverified**; the comment at
  `base_loop.py:2693` records a paper avg_entry_price=0 quirk on inherited positions. (b) A vanished
  position hits the DESYNC path `base_loop.py:1452-1466`, which writes an `estimated=True`
  `exit_reason='desync'` trade; Kelly already drops estimated rows (`trading_utils.py:230-240`) —
  verified clean.
- **Experiment X3 (Phase 1, measurement-only).** Per week, count `exit_reason=='desync'` rows in
  `trade_memory.json` and decision journals `journals/*.jsonl`, and entry rows with `fill_price<=0`.
  Rule: any non-zero count → cross-check the Alpaca forum/status for that date; if confirmed, flag
  those rows `venue_incident` in reports (never deleted). The (a) fallback is an ENGINE class-A candidate
  for the general (repro: fake `verify_position` returning `avg_entry_price='0'`).
- Kill-list: checked, nothing related.

**E1.5 SDK status.** alpaca-trade-api is deprecated in favor of alpaca-py (S8; maintenance end-2022
per search summary, **unverified wording**). The bots use the legacy SDK with an alpaca-py fallback
(`trading_utils.py:62-98`); alpaca-py is not installed here (CAMPAIGN_BRIEF). Nothing new for 2025–26
beyond the known `websockets<11` pin conflict. Websocket connection limits: searched, no crypto-specific
limit documented (S3 silent) — nothing new.

## E2 — Retail execution: maker/taker, ladders, stop gaps, stop placement

**E2.1 Maker fills are adversely selected (2025, live experiment).** S9 ran hundreds of thousands of
random maker orders on Binance BTC perpetuals and found a negative correlation between maker fill
probability and post-fill returns; profitable maker placement often required posting against the
book imbalance.
- Applies: partly. The crypto maker ladder (`order_utils.py:404-470`, `MAKER_ENTRIES_ENABLED=True`
  `strategy_config.py:80`) assumes the 10 bps fee saving plus half-spread outweighs selection;
  entry fill attribution already exists (`order_utils.py:378-396` → `execution_report.py:145-164`),
  but no post-fill markout is journaled. Venue differs (perps vs Alpaca spot) — transfer unverified.
- **Experiment X4 (measurement-only, Phase 1).** Journal the `us` mid at +5, +30, +60 min after
  every crypto entry fill, tagged by tactic (`maker*` vs `taker_fallback`). Rule: over ≥50 maker and
  ≥20 taker fills, if the 60-min markout of maker fills is worse than taker fills by more than the
  realized saving (10 bps fee + measured half-spread) with a 95 % bootstrap CI excluding zero → owner
  item to set `MAKER_ENTRIES_ENABLED=False` (existing flag); otherwise keep.
- Kill-list: checked. **"Adverse-selection contrarian-posting guard" [wave-7] is killed** — S9's
  contrarian-posting result is exactly that idea; nothing here re-opens it (no ask filed; X4 only
  measures, it posts nothing differently).

**E2.2 Stop-limit gap-through on the resting crypto stop (NEW angle).** The resting stop is a GTC
stop_limit with the limit 2 % below the trigger (`crypto_loop.py:168`, `:186-189`). A move that trades
through trigger−2 % inside the stop's reaction time leaves it unfilled. M5 (hourly proxy, upper bound
only): post-2025-10-10, 0.20–0.55 % of name-hours fall ≥4 % from the open and 0.012–0.155 % fall ≥8 %;
DOGE is highest. Hourly data cannot show fill order — minute data is required.
- Applies: yes. The live loop's software stop then sells at market (`base_loop.py:1441-1447`).
- **Experiment X5 (Phase 1 live + offline).** (a) Journal each resting-stop outcome (filled / expired
  unfilled / canceled by the loop), fill price vs trigger. Rule: over ≥20 triggers, if ≥10 % are
  unfilled or fill >1 % below trigger → owner review of `RESTING_STOP_LIMIT_GAP` (policy, not
  auto-changed). (b) Offline: pull Alpaca `us` 1-min bars for the FR-17 crypto stress windows and count
  minutes where Low < trigger·0.98 within 1 min of Low < trigger for ATR-distance stops. Measurement-only.
- Kill-list: checked; FR-17's reverting-wick audit is adjacent and complementary (it asks "was the
  stop-out real", X5 asks "did the stop fill").

**E2.3 Bad prints in the crypto training store (MISSED; cross-department).** M5 found ≥40 wick-only
bars whose Low is ≥15 % below Open while Close ≈ Open, including a BTC Low of $8,200 on 2021-10-21 and
repeated BTC lows ~20 % under price through Dec-2023/Jan-2024; 11 of the listed bars carry a
hard_stop triple-barrier label. Source attribution (Alpaca vs yfinance merge, `scripts/harvest_crypto_data.py:1-19`)
is **unverified**.
- Applies: yes, to labels (`policy_exits` hard stop fires on Low) and to HAR sizing
  (`volatility.py:158-164` sums ln(H/L)² per day; the clamp at `:216` bounds forecasts to the
  window max, which a bad print sets). SIGNAL-path files — reported, not touched.
- **Experiment X6 (offline, before the next retrain, Phase 0).** For every flagged bar, fetch the
  same hour from a second source (yfinance for the last 730 days; Alpaca minute bars otherwise) and
  mark it a bad print when the second source's Low is within 3 % of min(Open, Close). Rule: if ≥10
  confirmed bad prints exist, open an owner item (SIGNAL) for a harvest-side wick filter behind a
  default-OFF flag; report how many TB labels flip (gotcha #2: harvest + retrain + study reset).
- Kill-list: checked; not "Round-number / reference-level microstructure" [wave-4]; data hygiene.

**E2.4 Stop/TP parameter tuning (2026).** S10 replays >900 trades under alternative exits and finds
tighter stops / earlier profit-taking better, but abandoned the chronological split and used a
randomized split.
- Applies: no. A randomized split on time-ordered trades is not out-of-sample under this repo's
  purged walk-forward standard; exit semantics are shared by labels/backtest/live (`policy_exits.py`,
  MAP §9 #2 trailing-denominator owner decision). Nothing to adopt. Kill-list: "Shipping
  concentration/sizing changes on literature priors alone" [wave-5] applies by analogy.

## E3 — Vol targeting, drawdown control, Kelly

Searched: 2025–26 crypto realized-vol forecasting, volatility-managed crypto portfolios, drawdown
de-leveraging, fractional Kelly under estimation error. **Nothing new that changes the engine.**
- S13 (2025-07) still uses HAR {1,5,22}-day components on 5-min crypto RV; a claim that crypto HAR
  should use 7/30-day windows appeared only in a search summary and is **unverified** —
  `volatility.py:188-189` (5/22) stays; no experiment proposed.
- S11 (2026-08) is a pre-registered conformal-interval Kelly scaler: development Sharpe 1.34, sealed
  2022+ OOS 7.0–8.5 %/yr, **below** buy-and-hold. This reinforces the kill of **"Conformal abstention
  (CQR/ACI)" [wave-4]**; `KELLY_CAP=0.25` (`strategy_config.py:53`) and the B09 Kelly repair stand.
- Strategy-level vol targeting remains killed [wave-6, econ-07]; HAR-RV sizing is survivor #8.
  `drawdown.py:23` ladder: no 2025–26 primary source found that tests a drawdown ladder OOS.

## E4 — Crypto microstructure 2025–26 for a 30 s long-only spot bot

- Hour-of-day depth (S12, Binance BTC/FDUSD, Jul–Aug 2025): best 11:00 UTC ($3.86M within 10 bps),
  worst 21:00 UTC ($2.71M, −42 %), and Saturday 17:00 UTC was the deepest window. The 11:00/21:00
  pattern is already in 05 lesson 7; FR-18 already audits session-of-week. **Nothing new.**
- NEW local numbers: M3's `us`-venue spreads (XRP/DOGE ≈35 bps, LINK ≈19 bps on a Sunday night)
  exceed the 25 bps taker fee. That feeds X2 and the already-built census; no new experiment.
- M5: large intra-hour drops are **under**-represented on weekends (19 % of ≥4 % drops vs 29 % of
  hours) and cluster at 00, 14–15, 17, 20 UTC. This argues against a weekend-specific stand-down;
  no experiment proposed (FR-18 covers session effects).
- Funding/perp spillover: nothing new beyond 05 lesson 6 / FR-03. Kill-list: basis-timing tilt and
  carry rank stay dead [wave-7].

## E5 — Jetson/ARM inference memory

**E5.1 Where the bot's RAM goes (M1).** `import torch` alone costs ~331 MB RSS (≈208 MB private) on
this box; the model itself is 293,761 parameters (≈1.2 MB). Weight quantization therefore cannot move
RAM: qint8 dynamic quantization saves ~1 MB and changes outputs by up to 3.6e-3 (M2), and torch's
eager quantization API now prints a deprecation notice pointing to torchao (M2 stderr). ONNX Runtime
is not installed and cannot be installed without pip; untested. TorchScript is deprecated as of 2.10
(S14); the Jetson is pinned to 2.8 (`predict_now.py:160` `torch.jit.trace`), so this is informational.
- Applies: yes. Torch is imported on the bot path only by `predict_now.py:16`
  (`base_loop.py:31` imports it); `hw_monitor.py:140` short-circuits under `CUDA_VISIBLE_DEVICES=''`.
- **Experiment X7 (Jetson, class-C candidate; serving path ⇒ default-OFF).** Build a numpy forward
  for `RegressionLSTM` (M2 prototype: 1.6e-7 max Δ on random weights, 54 ms/forward — bar-keyed
  cache means ~1 forward per name per hour). Pre-registered acceptance, all three required:
  (1) on the real champion(s) and ≥1,000 real feature windows from `training_data.parquet` /
  `stock_training_data.parquet`: max |Δpred| ≤ 1e-6 and zero changes in any gate decision
  (threshold side, q10 veto, rank) versus torch; (2) the combined `run_bots.py` process RSS
  (smaps_rollup) drops ≥250 MB with torch never imported; (3) full-universe cycle time stays within
  the 30 s budget. Flag name proposed: `TRADER_NUMPY_LSTM_SERVE` (default OFF). Reject if any fails.
  Training stays on torch; the replica reads the same `state_dict`.
- Kill-list: checked. **"Halve Jetson inference via pruned feature core" [wave-6]** is a different
  claim (LGB column count); X7 changes no features and no model, only the runtime that evaluates it.

## E6 — Process supervision without root

**E6.1 A user-level systemd unit is installable here without sudo (NEW for this box).** M6: the user
manager already runs; `Linger=no`; this box's polkit default for `set-self-linger` is `yes` for any
session, so `loginctl enable-linger` for one's own user needs no sudo (S16 describes linger: a user
manager is spawned at boot and kept after logout). Any local polkit override is **unverified**
(localauthority unreadable). cgroup v2 delegates `memory pids` to the user manager, so `MemoryMax=`
is enforceable in a `--user` unit. systemd 249 has `StartLimitIntervalSec/Burst` but **not**
`RestartSteps=`/`RestartMaxDelaySec=` (added in 254, S15).
- Applies: yes. `scripts/setup_jetson_system.sh:193-217` builds a system unit that needs sudo; the
  same directives (Type=notify, WatchdogSec=900, Restart=on-failure, RestartSec=30, OOMPolicy=continue,
  MemoryMax=6G, positive OOMScoreAdjust) are valid in `~/.config/systemd/user/trader.service` with
  `WantedBy=default.target`; `run_pipeline.py:80-100` `_sd_notify` works against a user manager.
- **Experiment X8 (ops, owner runs it; runbook Phase 0 "bots up").** Owner runs
  `loginctl enable-linger` and installs the user unit. Pre-registered acceptance: after a reboot with
  no login, `systemctl --user is-active trader` = active within 180 s; `kill -9` of the main PID →
  back within 60 s; `kill -STOP` of the main PID → watchdog restart within `WatchdogSec` + 60 s. Any
  failure → keep the manual launch and file the failing step.

**E6.2 The internal bot supervisor restarts forever at a fixed 60 s (measured).** M7: 1,778 restarts
of one deterministic crash. `run_pipeline.py:1032-1076` restarts any exited bot on every 60 s
monitor pass (`:1712-1721`, `:1832-1844`) with no backoff, no crash-signature check and no give-up;
the alert is deduped only to once per 10 min (`notify.py:14`), i.e. ~178 alerts for that run.
Standard practice (S15's exponential restart delay; systemd's start-limit) is backoff plus a limit.
- **Experiment X9 (class-D candidate for the ENGINE general, not model-facing).** Proposed rule:
  per-bot delay doubling 60→120→…→960 s, reset after 30 min alive; after 5 crashes within 60 min
  whose last traceback line is identical, stop auto-restarting that bot and send one `level='error'`
  alert. Pre-registered acceptance: a unit test with a fake `Popen` replaying M7's sequence yields
  ≤6 restarts and exactly 1 error alert; a crash with differing signatures keeps restarting (bounded
  by backoff); a `_manually_stopped` bot is untouched (`run_pipeline.py:1037-1044`). JUDGMENT: the
  thresholds (5 / 60 min / 960 s) are choices, not derived.
- Kill-list: checked, nothing related. supervisord/pm2: not installed, not needed given E6.1.

## Ranking (expected value / cost)

1. X1 quote-age census — S, measurement; decides whether local exits are silently skipped on quiet books.
2. X9 supervisor backoff + crash-signature give-up — S, ops; M7 is direct evidence.
3. X3 + E1.4(a) avg_entry_price=0 fallback — S; documented paper incident path to a $0 stop.
4. X8 user systemd unit + linger — S, owner action; removes the sudo blocker for supervision.
5. X6 bad-print verification in the crypto store — M, offline; touches labels and HAR (SIGNAL owner).
6. X2 us vs us-1 spread-as-cost — S to build, needs ≥30 fills; feeds kill-list ask #1.
7. X7 numpy LSTM serving — M; ~330 MB RSS per bot process if acceptance holds.
8. X5 stop_limit gap audit — S live journal + M offline; low base rate (M5).
9. X4 maker markout — S; needs ≥50 maker fills; outcome may only confirm the status quo.

## Trailing-denominator divergence table (W9, 2026-09-27)

Owner item from `CLAUDE.md` § Shared kernels. No code changed. Script and raw output:
ENGINE `w9/trail_table.py` and `w9/trail_table.out` in the scratchpad.

**The three sites.**
- `policy_exits.py:153-157` (the kernel; used for labels, backtest, meta rows and decision_report):
  `td = clamp((a*atr_trail_mult)/entry, stop_floor, stop_ceil)`. It is fixed at entry and enforced
  as `ts = hwm*(1-td)` at `:181`. The file's own divergence note is at `:39-45`.
- `stock_loop.py:1484-1485` (the live stock exit): the same entry denominator. It is sent as a
  server `trailing_stop` with `trail_percent=round(trail_pct*100, 1)` at `:1506-1509`. The broker
  trails from its own HWM, which starts at the upgrade, and the percentage is rounded to 0.1 %.
- `base_loop.py:1076-1079` (the live crypto exit, through `_manage_stops` at `:1320` and
  `:1330-1331`; `crypto_loop._manage_stops` at `:236` only prunes and then calls super) uses
  `raw_trail_dist = entry_atr*ATR_TRAIL_MULTIPLIER / hwm`. It is re-derived every cycle.
  `_book_stop_risks` (`:1225`) also calls it for **both** books, because StockLoop does not override
  it. So for stocks the HWM convention reaches risk accounting only, never the exit.

**(a) Closed form.** Symbols: E = entry, H = rE (r ≥ 1+act once armed), a = ATR/E, m = trail mult,
C(x) = min(c, max(f, x)).
- Entry convention (kernel, stock): T_E = rE·[1 − C(m·a)].
- HWM convention (base/crypto): T_H = rE·[1 − C(m·a/r)].
- Gap: **T_H − T_E = rE·[C(m·a) − C(m·a/r)] ≥ 0**. C is monotone, so the HWM convention is
  **always tighter or equal**.
- Inside the clamps: gap = **m·ATR·(r−1)** in price. In bps of the HWM that is 1e4·m·a·(r−1)/r.
- Partial floor (m·a/r < f ≤ m·a): gap = rE·(m·a − f).
- Both floored (m·a < f): gap = 0.
- Partial ceiling (m·a/r < c < m·a): gap = rE·(c − m·a/r).
- Macro tightening k = `stop_mult` < 1 (`base_loop.py:1083-1086`) scales the gap by k. The kernel
  has no macro multiplier; that is a separate, already-documented divergence.
- The effective stop is max(E·(1 − C(stop_mult·a)), trail). In every cell below the trail is above
  the hard stop, so the effective gap equals the trail gap.
- In the HWM form the trail is H − m·ATR, a classic Chandelier exit. The entry form is a fixed
  percentage set at entry. Which definition is right is a JUDGMENT call; which one the backtest
  validated is OBJECTIVE: the entry form.

Policies: CRYPTO m=2.0, f=1.5 %, c=15 %, act=1.5 %. STOCK m=2.0, f=1 %, c=10 %, act=1 %.

**(b) Table.** Entry = 100. The gap is in bps of the HWM, and the HWM convention is the tighter one
in every non-zero cell. The gaps are the same for both books (m is equal); only the a = 0.5 % levels
differ, because of the floor.

| r | a=0.5 % (crypto / stock) | a=1 % | a=2 % | a=4 % |
|---|---|---|---|---|
| 1.015 | 99.977 / 100.485 both, 0 | 99.470 vs 99.500, 3.0 | 97.440 vs 97.500, 5.9 | 93.380 vs 93.500, 11.8 |
| 1.03 | 101.455 / 101.970 both, 0 | 100.940 vs 101.000, 5.8 | 98.880 vs 99.000, 11.7 | 94.760 vs 95.000, 23.3 |
| 1.05 | 103.425 / 103.950 both, 0 | 102.900 vs 103.000, 9.5 | 100.800 vs 101.000, 19.0 | 96.600 vs 97.000, 38.1 |
| 1.10 | 108.350 / 108.900 both, 0 | 107.800 vs 108.000, 18.2 | 105.600 vs 106.000, 36.4 | 101.200 vs 102.000, 72.7 |
| 1.20 | 118.200 / 118.800 both, 0 | 117.600 vs 118.000, 33.3 | 115.200 vs 116.000, 66.7 | 110.400 vs 112.000, 133.3 |

Cells read "entry-convention stop vs HWM-convention stop, gap in bps". At a = 0.5 %, m·a = 1 %, so
both conventions sit on the floor (crypto 1.5 %; stock 1 %, with m·a/r below the floor) and the gap
is 0. The largest gap on the grid is 133 bps: a run to +20 % on a 4 %-ATR name.

**(c) Positions affected tonight.** None.
- The live account holds only the six crypto positions (BTC, ETH, SOL, LINK, XRP, DOGE).
- All six are zero-basis (avg_entry_price 0; ENGINE W1). At `base_loop.py:1073` the `entry_price > 0`
  test is false, so the ATR branch is skipped. Neither convention applies: stop = STOP_LOSS_PCT,
  trail = TRAIL_PCT = 5 %, and `trailing_active = hwm >= 0` is always True. That makes a pure
  hwm·0.95 trail, pinned by `tests/test_engine_r1_inherited_positions.py:245`.
- There are no stock positions and no open orders.
- Bots have not run since 2026-05-07.

**(d) Two candidate resolutions.** Both are exit-rule changes, so both are owner decisions behind a
default-OFF flag.
1. **Loop → entry.** In `base_loop.py:1077`, `/ hwm` becomes `/ entry_price` (with the existing
   `hwm > 0` guard becoming `entry_price > 0`). This is the **only** option that keeps
   labels == backtest == live without a retrain:
   - labels and backtest do not move;
   - crypto live loosens by the table's gap;
   - stock `_book_stop_risks` stops disagreeing with the stock server trail.
   Blast radius: `_desired_stop_for`, which feeds the crypto resting-stop ratchet at `:1323`, the
   software trail at `:1330`, and book risk at `:1225`.
2. **Kernel → HWM.** In `policy_exits.py:153`, td would be re-clamped per bar as C(m·a/hwm_j), and
   `stock_loop.py:1484` would have to follow. A fixed-percentage broker `trailing_stop` cannot
   express this. `trail_price = m·ATR` would work unclamped, but the clamps break it, so stocks would
   need cancel/replace churn. The kernel change also changes every label: full harvest, retrain,
   and gotcha #2 on both books. It is SIGNAL's file.

**Pin for the chosen one.** Add a new test (for example `tests/test_trail_denominator_parity.py`)
that imports both `policy_exits` and `base_loop`. Tests may import the kernel;
`tests/test_ia4_flagged.py::test_vertical_is_loop_layer_only` only forbids the **loop files** from
importing it, so the fix stays a mirror. The test covers the r × a grid above:
- Build a gap-free bar path that rises to H = rE, then falls through the trail.
- Assert that the kernel's trailing-exit price (reason 3) equals
  `base_loop._desired_stop_for(pos)[0]`, and equals `hwm*(1 - stock trail_pct)` before the 0.1 %
  broker rounding.
- Cover the floor and ceiling cells.

## X7 spike results (W8, 2026-09-27)

**What was run.** `lstm_numpy_serve.py` (new, root; numpy-only at import, torch lazy inside
`export_from_pth` / the `__main__` harness; nothing imports it — `tests/test_lstm_numpy_serve.py::
test_serving_path_does_not_use_module_yet` pins that) replicates `model_v2.RegressionLSTM.forward`
op-for-op (mapping table in its docstring). Harness: `python lstm_numpy_serve.py parity|memory`.
**Artifacts:** the only real champions on the box, `archive/models/2026-04_pre_rebuild/{,stock_}*`
(root copies were moved there 2026-09-26; `archive/README.md:39`), copied read-only to the W8 scratch.
Crypto: H=288, heads=2, T=18, F=23, **1,378,497 params (5.5 MB fp32)**; stock: H=160, heads=8, T=20,
**438,209 params**, th 0.51 / 1.31. (E5.1's "293,761 params" was the M2 synthetic H=128 config, not
the champion.) Neither book has an LGB/q10 leg on disk, so `predicted_return == lstm_pred`
(`predict_now.py` blend only runs when `lgb_model.txt` exists) and the q10 veto is N/A.
**Windows:** 1,575 crypto (300 random timestamps × 6 names, 2021-01 → 2026-09) and 1,632 stock
(40 timestamps × ≤46 names) from the current stores via `data_utils.load_training_data`, built
exactly as `get_live_prediction` does (dropna on consumed cols → last `seq_len` rows → float64
`scaler_X.transform` → float32 tensor, B=1); 0 non-finite windows. Torch reference = the real
serving object, `predict_now.load_model('cpu', prefix)` (jit-traced), `TORCH_NUM_THREADS=2`.

| metric (max / p99 / median \|Δpred\|, pred in %) | crypto | stock |
|---|---|---|
| numpy-fp32 vs torch-fp32 jit (the serving comparison) | **5.72e-6** / 3.8e-6 / 4.8e-7 | **5.96e-6** / 4.0e-6 / 4.8e-7 |
| numpy-fp64 vs torch-fp32 jit | 8.00e-6 / 4.3e-6 / 5.8e-7 | 1.17e-5 / 4.4e-6 / 7.7e-7 |
| numpy-fp64 vs torch-fp64 eager (op-mapping correctness) | **1.6e-14** | **1.2e-14** |
| torch-fp32 vs torch-fp64 (torch's own fp32 error) | 8.00e-6 | 1.17e-5 |
| numpy-fp32 B=1 vs numpy-fp32 batched (BLAS shape noise) | 3.8e-6 | 1.9e-6 |
| sign / threshold-verdict (BUY/HOLD/SELL) / within-timestamp rank changes, fp32 and fp64 | 0 / 0 / 0 | 0 / 0 / 0 |
| closest window to a gate: min \|\|pred\|−th\|, min \|pred\| | 4.8e-3, 1.5e-3 | 3.0e-3, 3.0e-3 |
| VmRSS after weights + 1 forward, fresh process, median of 3 — minimal (numpy only vs torch) | 45 vs 400 MB → **−355 MB** | 43 vs 392 → **−349 MB** |
| same, bot-like process (pandas+joblib scaler+lightgbm+indicators+market_data first) | 218 vs 567 MB → **−349 MB** | 214 vs 560 → **−346 MB** |
| lightgbm import alone (in the bot-like process) | +53 MB | +53 MB |
| ms/forward B=1: torch jit (2 thr) / numpy fp32 (OpenBLAS default 6 thr) / numpy fp32 (OPENBLAS_NUM_THREADS=1) | 13.4 / 18.8 / **9.7** | 12.6 / 12.8 / **5.9** |
| per 30 s cycle (6 crypto / 46 stock forwards, every forward a memo miss) | torch 0.08 s, numpy-1thr 0.06 s | torch 0.58 s, numpy-1thr 0.27 s |

Timings were taken while the CEO's hypersearch held the box (load 2–3); the 6-thread numpy figure
spiked to 74.6 ms in one probe (thread contention), which is why single-thread BLAS is the relevant
setting. Numerics are identical under `OPENBLAS_NUM_THREADS=1` (re-run: same maxima). The M2
"54 ms/forward" was an artifact of the prototype re-casting every weight per call.

**Verdict against the pre-registered rule: NOT FEASIBLE (partial — fails criterion 1a only).**
(1a) max |Δpred| ≤ 1e-6: **FAIL** — 5.7e-6 / 6.0e-6 in fp32, 8.0e-6 / 1.17e-5 in fp64. (1b) zero sign,
threshold-verdict and rank changes: PASS (0 of 3,207). (2) ≥ 250 MB RSS saving: PASS (346–355 MB).
(3) cycle time within 30 s: PASS. Diagnosis (OBJECTIVE, measured above): the misses are float32
rounding, not a mapping error — numpy-fp64 matches torch-fp64 to 1.6e-14, and torch-fp32 itself sits
8.0e-6 / 1.17e-5 from the fp64 result, so a 1e-6 absolute bar on outputs of median magnitude ~3.1 %
(≈3 fp32 ulps) cannot be met by an independent fp32 implementation; mirroring torch's bias-add
order (`(xW_ih+b_ih)+(hW_hh+b_hh)`) did not help (LSTM-output Δ 1.37e-6 vs 1.15e-6). The goalpost is
not moved here: re-registering criterion 1a (e.g. "max |Δ| ≤ torch's own fp32-vs-fp64 error on the
same windows, and zero decision changes") is an **owner decision** (JUDGMENT). Integration design is
therefore not proposed; for the record, any future flip must also make `predict_now.py:16`
`import torch` / `:110` `device = torch.device(...)` / the `torch.tensor` + `inference_mode` calls (`:484`, `:487`) in
`get_live_prediction` conditional (SIGNAL's file) — the saving is realised only if no module in the
bot process imports torch (today: `predict_now` via `base_loop.py:31`; `hw_monitor.py:140`
short-circuits under `CUDA_VISIBLE_DEVICES=''`). Under the combined `run_bots.py` process the saving
is ~350 MB once; under `run_pipeline`'s default one-process-per-bot it is ~350 MB per process.

## R3 scout — E2/E4 follow-ups: X2 venue-of-fill and X4 post-fill returns (W12, 2026-09-27)

Measurement-only; no production code, flag or posting behaviour changed; scripts and raw pulls in the
ENGINE `w12` scratch dir; every broker/market-data call a read-only GET on the paper account. **Main
caveat:** every fill below is a PAPER fill — Alpaca paper matches against quotes with no queue or
impact (W12-S2) — so nothing here says how LIVE orders would fill.

| id | source | date | URL (accessed 2026-09-27) |
|---|---|---|---|
| W12-S1 | Alpaca Docs "Crypto Spot Trading": orders trade on the "Alpaca Exchange"; pair object `"exchange": "ALPACA"`; maker/taker definitions | undated | https://docs.alpaca.markets/docs/crypto-trading |
| W12-S2 | Alpaca Docs "Paper Trading": matched "against the best available current market price (NBBO)"; limit buy fills only when limit ≥ best ask; random partial fills 10 % of the time; no impact, latency slippage or queue position; "simulates crypto trading as well" | undated | https://docs.alpaca.markets/docs/paper-trading |
| W12-S3 | Alpaca Docs "Real-time Crypto Data": `us` = Alpaca US, `us-1` = Kraken US, `eu-1` = Kraken EU; "executes your crypto orders in its own exchange, and also supports Kraken"; "us-1 represents the states listed below" (23 states/territories) and gives no reason for the mapping | undated | https://docs.alpaca.markets/docs/real-time-crypto-pricing-data |
| W12-S4 | Alpaca forum, staff (D. Whitnable): "All crypto trades on Alpaca fill against other orders on Alpaca … the 'Alpaca Exchange'" (2024-03-30); a follow-up on 2025-10-25 asking about book composition got no answer | 2024-03-30 / 2025-10-25 | https://forum.alpaca.markets/t/which-crypto-exchanges-does-alpaca-trading-api-uses/13960 |
| W12-S5 | Kraken blog, "Alpaca integrates Kraken Embed": the integration is for the Broker API (B2B) and says nothing on routing for Trading-API accounts | 2025-06-17 | https://blog.kraken.com/product/kraken-embed/alpaca-integrates-kraken |
| W12-S6 | Zhai, "Public Trader Identity: Adverse Selection and Return Predictability", arXiv 2608.04373 (Hyperliquid, wallet-level taker toxicity persists, ρ=0.52 across 10-day windows) | 2026-08-05 | https://arxiv.org/abs/2608.04373 |

**X2(a) — what Alpaca documents about the venue.** Crypto orders trade on Alpaca's own exchange (S1,
S4); S3's "also supports Kraken" and per-state `us-1` list are never explained, and no document ties a
Trading-API account's routing to its state (**unverified**). No order or fill field names a venue: the
36 order keys and 13 FILL-activity keys pulled today (`/v2/orders`, `/v2/account/activities/FILL`) have
no `venue`/`exchange`; the only venue string is the pair object's `exchange` (S1). Fees post as
separate `CFEE` activities. No 2025–26 routing-doc change found (S5 is Broker-API only).

**X2(b) — our journals: n = 0, stop.** `journals/*.jsonl` (51 files, 2026-02-23 → 05-07): 11,818
skip, 264 buy, 31 short, **0 sell** rows; all 264 buy rows carry only `symbol, pred_return,
sentiment_*, llm_*, final_notional, skip_reason` — no `decision_price`/`fill_price`/`slippage_bps`/
`entry_tactic`/bid/ask (the fill keys in `trade_journal.py:9-21`, emitted at `base_loop.py:3494-3509`,
postdate these files). Even today's buy row stores only the `us` mid — no bid/ask, `us-1` quote or
broker `order_id`. **Broker reconstruction instead (real paper data):** 907 crypto FILL activities =
686 filled orders, 2026-02-07 → 04-26 (buys 199 market / 213 limit; sells 111 / 163), metadata for all
686; arrival quote = last quote at/before `submitted_at` from historical v1beta3 quotes on **both**
`us` and `us-1` (arrival, not our decision quote; decision→submit latency unmeasured). Arrival-quote
age: `us` median 13.5 s (p90 88), `us-1` 0.2 s (p90 1.8). Slippage = side·(fill VWAP − mid)/mid, bps.

| taker (market) fills, both sides | n | MAE vs ½spread `us` | MAE vs ½spread `us-1` | median slip vs `us` mid | median ½spread `us` / `us-1` |
|---|---|---|---|---|---|
| all six + 4 legacy alts | 310 | **10.2** | 16.2 | — | — |
| BTC/USD | 24 | 3.7 | 6.9 | 3.8 | 5.0 / 0.01 |
| ETH/USD | 38 | 4.1 | 6.5 | 6.1 | 4.9 / 0.05 |
| SOL/USD | 28 | 18.9 | **12.6** | 6.0 | 18.9 / 0.62 |
| LINK/USD | 52 | 6.0 | 14.7 | 15.0 | 10.2 / 1.21 |
| XRP/USD | 66 | 18.6 | 24.7 | 13.1 | 19.8 / 0.26 |
| DOGE/USD | 69 | 7.6 | 21.7 | 17.2 | 16.2 / 0.61 |

Market-buy VWAP sits at the `us` touch (median −0.4 bps vs the `us` arrival ask; p10 −12, p90 +7) and
**+12.7 bps beyond the Kraken ask** (p10 +4.0; 199/199 at/through it); sells mirror (−1.0 / +12.6).
113 of 213 limit buys were marketable vs the Kraken ask but not the `us` ask at arrival and waited a
median 5.5 s (p90 26) vs 0.0 s for the 76 marketable on both — S2's rule applied to an Alpaca-venue
quote. **Reading (paper only):** paper crypto fills are priced off the Alpaca `us` book, not Kraken;
`us` was the better predictor in every month (MAE Feb 7.4 vs 11.8, Mar 12.6 vs 21.5, Apr 23.1 vs 31.1).
SOL is the exception: its `us` quote overstated cost (½spread 18.9, realized 6.0), consistent with W3
M3's stale SOL `us` quotes (mechanism **unverified**). Fees: 121 of 843 CFEE rows carry an `order_id`
(71 orders); they imply median 22 bps on market and 12–13.5 on limit fills, not the published 25/15
(`fees.py:41-42`) — **unexplained, unverified**, not used above.

**X2(c) — live probe, Sun 2026-09-27 07:30–07:32 UTC** (6 rounds × 2 locs, 15 s apart; this session
ran on a Sunday, so weekday nights come from historical last quotes at 9 instants, Tue–Thu 2026-09-22..24
× 02:00/04:00/07:30 UTC). Median full spread, bps, `us` / `us-1` / gap:

| | BTC | ETH | SOL | LINK | XRP | DOGE |
|---|---|---|---|---|---|---|
| Sun 07:30 live | 3.8 / 0.01 / 3.8 | 2.4 / 0.04 / 2.4 | 6.4 / 0.82 / 5.6 | 20.3 / 3.8 / 16.5 | 38.5 / 0.07 / 38.5 | 35.3 / 0.34 / 35.0 |
| Tue–Thu nights (hist.) | 2.7 / 0.01 / 2.6 | 2.9 / 0.04 / 2.9 | 7.8 / 0.87 / 6.9 | 19.6 / 5.6 / 14.1 | 39.2 / 1.1 / 38.0 | 32.5 / 0.8 / 31.7 |

The gap is **structural, not a weekend effect** (weekday nights within ~4 bps of Sunday on every name).
Max `us` quote age in the probe 181 s (ETH; over the 180 s rule) vs < 10 s on `us-1`. Live quotes and cost
inputs use `loc='us'` (alpaca-trade-api `rest.py:912-917` default; `order_utils.get_quote` passes no `loc`).

**X2(d) — pre-registered rule (supersedes round-1 X2's wording; same 25 % margin).** Data:
`python scripts/fill_venue_slippage_report.py --pull B.json`, then `--replay B.json`, on ≥ 30 taker
(`type=market`) crypto fills. Number: mean |slip vs `us-1` arrival mid − ½spread_loc|, loc ∈ {`us`,
`us-1`}; the winner must be ≤ 0.75 × the other (per-symbol reported where n ≥ 10). Retrospective paper
result: **`us` wins** (10.2 vs 16.2); per symbol BTC/ETH/LINK/DOGE → `us`, XRP → no_change (18.6 vs the
18.5 cut), SOL → `us-1`.
- `us` wins (today): no execution change. The kill-list ask-#1 census (`TRADER_CRYPTO_SPREAD_STAMP`,
  `liquidity.py:245-330`) must use `--loc us` (already the default, `scripts/crypto_spread_census.py:126`);
  a `us-1` census would understate taker crossing cost by ~12 bps/side. The placeholder tiers
  (`liquidity.py:254`, LINK/XRP/DOGE 0.10 %) and flat `FLAT_SPREAD_PCT['crypto']=0.10` (`fees.py:55`) sit
  below realized paper round-trip crossing (2 × median taker slip: LINK 30, XRP 26, DOGE 34 bps). The
  owner ruling on ask #1 and gotcha #2 are still required.
- `us-1` wins on LIVE fills: owner item for a default-OFF cost-quote-location flag (proposed
  `TRADER_CRYPTO_COST_QUOTE_LOC`, default `us`; not built). no_change: keep. A paper verdict never
  licenses a live-cost change; re-run on ≥ 30 live fills if the account goes live.
- Ask #1 relation: X2 informs only its "stamp" half and builds no passive-fill simulator (the [wave-7]
  Honest-OHLC simulator half stays dead). Runbook: Phase 1 (instrument, no flip; can run now); the
  verdict feeds the Phase 5 ask-#1 ruling.

**X4 — post-fill returns (adverse-selection MEASUREMENT only).** Retrospective **n_maker = 0**: no
historical fill came from the bid-join ladder. `place_maker_buy` first appears in `f67f710`
(2026-06-10), the bots last ran 2026-05-07, every historical `client_order_id` is a bare UUID with no
`maker-` tag (`order_utils.py:409-416`), and legacy limit buys filled a median **+3.7 bps above** the
`us` mid (`compute_limit_price`'s mid+offset, not a bid-join). Descriptive only (confounded; tactic
not randomized): legacy limit buys (n=212) minus market buys (n=181), mean `us-1` 1-min-close move
after the fill, 95 % bootstrap CI — +1 min −1.35 bps [−3.1, +0.3]; +5 min −4.2 [−9.4, +0.9]; +30 min
−6.8 [−16.1, +2.7]. All CIs include 0 (mean +30 min moves: +9.3 limit, +16.2 market).
- Forward mids: the journals have none; hourly harvest bars are too coarse. Alpaca still serves `us-1`
  historical quotes and 1-min bars for Feb 2026 today (7+ months back), and client_order_id tags now
  identify tactic, so the CLI reconstructs markouts **without any loop change**. Optional in-loop row,
  only if the owner wants independence from Alpaca data retention: `{"action": "fill_markout", symbol,
  order_id, entry_tactic, fill_ts, fill_price, h_min ∈ {1,5,30}, mid_us, mid_us1, age_us, age_us1}`.
  Cost: ≤ 3 rows (~200 B) per entry, ≤ 2 batched latest-quote GETs per 30 s cycle only when a markout
  is due, pending list ≤ 3 × open entries. Judgment: not worth building while the CLI works.
- Toxic rule: net markout_H = side·(px_us-1(t_fill+H) − VWAP)/VWAP·1e4 − fee (15 maker / 25 taker);
  primary H = 30 min, +1/+5 descriptive (supersedes round 1's 60 min). **TOXIC** iff n_maker
  (`maker-*`) ≥ 50 and n_taker (market) ≥ 20 buys, and the 95 % bootstrap CI of mean(net_maker) −
  mean(net_taker) lies wholly below 0. TOXIC → owner item to set the EXISTING
  `MAKER_ENTRIES_ENABLED=False` (`strategy_config.py:80`); otherwise keep. A paper verdict is not live
  evidence: S2's simulator fills a resting bid exactly when the ask reaches it, with no queue.
- Kill-list boundary: nothing here changes where, when or how orders post. The [wave-7]
  "Adverse-selection contrarian-posting guard" and [wave-2] "Order-book imbalance filters" stay killed;
  X4 can only keep the ladder or switch it off through its existing flag.

**Survey (2025–26).** No new primary evidence on Alpaca crypto execution quality beyond S1–S5 (the
2025-10-25 book-composition question in S4 is unanswered). S6: taker toxicity is persistent on a perps
DEX at 1 s horizons — does not transfer to a 30 s-cycle Alpaca spot bot (venue, horizon, instrument).
**Tool:** `scripts/fill_venue_slippage_report.py` (stdlib-only; `--replay` offline, `--pull` read-only
GETs, `--json`) + `tests/test_fill_venue_slippage_report.py` (13 synthetic tests, socket-blocked
replay); replaying today's bundle reproduces the numbers above (X2 `us`, X4 insufficient_n).

## R4 scout — E6 crash-loop backoff and E3 drawdown-ladder review (W14, 2026-09-27)

This section is measurement and design only. Nothing in production changed. Scripts and outputs are in the ENGINE `w14` scratch dir. Log parsing ran under hwlock; the only broker call was a read-only GET of portfolio history.

Checks: `KILL_LIST.md` has no supervision entry. The [wave-6] vol-targeting kill and the [wave-5] "sizing changes on literature priors alone" kill are respected, because nothing here ships from the literature. `08_removed_code.md` has no restart, backoff or ladder removal (the grep hits only IA-2.2, :360-372, unrelated). `07_decision_influences.md:105,:197` hold that dd_mult stays, that its rungs are "owner risk preference", and that account health gets "never a fourth" mechanism. Nothing here adds one.

| id | source (accessed 2026-09-27) | date | URL |
|---|---|---|---|
| W14-S1 | systemd.service(5): `RestartSteps=`/`RestartMaxDelaySec=` "Added in version 254"; restarts are "subject to unit start rate limiting" | undated | https://man7.org/linux/man-pages/man5/systemd.service.5.html |
| W14-S2 | KEP-4603: CrashLoopBackOff 10 s ×2 "capped at five minutes", reset after "2x the maximum backoff -- that is, 10 minutes"; alpha 1 s→60 s gate (release **unverified**) | living | https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/4603-tune-crashloopbackoff/README.md |
| W14-S3 | KEP-5593 "Configure the max CrashLoopBackOff delay", beta target 1.35 | 2025-09-30 | https://github.com/kubernetes/enhancements/issues/5593 |
| W14-S4 | AWS Well-Architected REL05-BP03 "Control and limit retry calls" (page date **unverified**) | undated | https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_mitigate_interaction_failure_limit_retries.html |
| W14-S5 | Ryan, "Conformal Kelly", arXiv 2608.01494 (= E3 S11; new angle: its §3 "drawdown dial") | 2026-08-02 | https://arxiv.org/abs/2608.01494 |
| W14-S6 | Landolfi, "Drawdown Risk Beyond Brownian Motion", arXiv 2608.00127 | 2026-07-31 | https://arxiv.org/abs/2608.00127 |

### E6(a) — evidence

**Logs.** `pipeline_output.log` (1.15 MB) has no timestamps on its crash lines and no rotations. `stock_bot_output.log` (115 MB) and `crypto_bot_output.log` (60 MB) are timestamped. `logs/trader.log{,.1..5}` (about 46 MB) contain 0 crash tracebacks. Crash time is the last timestamp before each `Traceback`; a start is `Loading prediction models`.

| metric (stock bot; crypto: 28 starts, 0 tracebacks) | value |
|---|---|
| starts / crashes | 1,808 / 1,778 (= `pipeline_output.log:5387-9269`) |
| distinct signatures (innermost frame + exception) | **1** (`fundamentals.py:format_fundamentals_for_llm` ValueError). All share the *first* frame `stock_loop.py:574`, so the signature must use the innermost frame |
| same as previous / longest identical run | 100 % / 1,778 |
| lifetime start→crash | p10 27 s, p50 42 s, p90 83 s; 7 lifetimes ≥ 30 min (overnight survivals) |
| crash→restart / crash spacing | p50 30 s, max 96 s / median 74 s |
| when | 7 RTH days, 08:30–15:00 CT. It **self-healed** at 11:59 on 04-23; 04-30 began at 10:00 |
| alerts | `run_pipeline.py:1073-1075` with the 600 s dedupe (`notify.py:32`) → 232 would-be alerts. **0 were sent**: no channel in `.env`, so `notify.py:259` returns early |
| side cost (sample hour, 44 restarts) | 97 LLM, 88 FMP-403 and 43 news calls: about 5 external calls per restart (scaling to all 1,778 is **unverified**) |

**Changed since April.** `base_loop.py:347-363` (edf3151, 2026-06-10) now contains in-cycle errors. What still crash-loops is startup: `base_loop.py:332-345` (`_load_models`, the scoped cancel, `_reconstruct_positions`), plus imports. In combined mode a stock-thread crash exits the whole process (`run_bots.py:163-175,:228`). Crypto then restarts too, with a cancel plus a restart stop: 18 writes per zero-basis restart (W11). Six zero-basis crypto positions are held today.

### E6(b) — pure-function design, replayed (`w14/backoff_policy.py`, stdlib only)

**Functions.**
- `crash_signature(tail)` → (innermost `file:func`, masked exception line).
- `next_restart_delay(k, base=60, cap=960)` = 0 if k = 0, else min(cap, 60·2^(k−1)).
- `consecutive_same(history)`; `should_give_up(history, window_sec=3600, n=5)`.
- ≥ 1,800 s alive clears the history (K8s-style, S2). An explicit launch clears the latch.

**Replay model.** A bot alive since t crashes at the first observed crash ≥ t + 21 s, then restarts at crash + 30 s + delay. Episodes end at the retrain stops (04-18, 04-25, 05-02). The current policy replays to 1,616 restarts vs 1,778 observed (−9 %).

| policy | restarts | avoided | give-up after | warn / critical alerts |
|---|---|---|---|---|
| current (next 60 s pass) | 1,616 | — | never | 230 / 0 |
| backoff 60→960 s only | 169 | **89.5 % — fails** | never | 144 / 0 |
| backoff + give-up 5 in 60 min (latch) | **12** | **99.3 %** | 14.6 / 16.9 / 15.5 min | 6 / 3 |
| backoff + give-up 3 in 30 min | 6 | 99.6 % | 4.9 / 6.6 / 4.6 min | 3 / 3 |
| 5/60, then parked probe every 3,600 s | 148 | 90.8 % | same | 127 (mute while parked) / 3 |
| same times, signatures alternating A/B | 1,616 | 0 %, max added delay **0 s** | never | 230 / 0 |

**Acceptance: PASS (latch).** 99.3 % of restarts are avoided. Distinct signatures add 0 s, so the first restart is bounded by the 60 s monitor pass (`run_pipeline.py:1712-1721,:1832-1844`), not by the functions.

**Decision-neutral:** the functions and the latch act only on an exited process. They never signal a running bot and never touch orders; OFF is byte-identical to today. **Caveat:** before June, a 42 s restart still ran the breaker and `_manage_stops` (`base_loop.py:394,:409`). A startup crash never reaches the loop, so delaying it costs nothing.

**Policy (owner):** the numbers, and latch vs parked probe. A transient startup failure (an Alpaca outage = 5 identical `ConnectionError`s) would latch the bots down after the outage ends.

**Flag:** the house ops pattern is a run_pipeline constant read from `TRADER_*`, as with `EOD_DIGEST_ENABLED` at `run_pipeline.py:144` (`FLAGS.md:155,:459` still cite a stale `:118`). Proposal: `BOT_RESTART_BACKOFF` ← `TRADER_BOT_RESTART_BACKOFF='0'`, facing = ops. Tests: the fake-Popen M7 replay gives ≤ 6 restarts and 1 critical; A/B adds 0 s; `_manually_stopped` stays untouched (`:1037-1044`).

**Companions:**
- (i) Alert-only: one `critical` after 20 identical consecutive cycle errors. Today `base_loop.py:356-360` only sends a deduped `warning`, forever.
- (ii) Owner: configure a notify channel.

### E6(c) — practice vs this box

- **K8s:** capped exponential backoff (10→300 s) with a 10 min healthy reset (S2); the cap became node-configurable in 2025-26 (S3).
- **S4:** all four of its anti-patterns are present: no backoff or maximum; retrying errors that "predictably will not resolve without manual intervention"; no alerting on repeated failures; and "retrying at multiple layers".
- **systemd:** the defaults (10 s / burst 5, W5) never trip at about 74 s spacing. `StartLimitBurst=5` over 3,600 s would trip but is signature-blind. `RestartSteps` needs version ≥ 254; this box runs 249 (S1). Either unit sees only `run_pipeline`, and `mark_progress()` (`run_pipeline.py:1034`) keeps `WATCHDOG=1` flowing, so the fix belongs in `_check_restart_bots`.

### E3(a) — the ladder as it is

| element | as-is | cite |
|---|---|---|
| rungs | dd ≥ 20 % → 0.25, ≥ 15 % → 0.50, ≥ 10 % → 0.75, else 1.0 | `drawdown.py:23,:74-80` |
| dd / peak / reset | (peak−eq)/peak floored at 0; monotone max; reset **only by a new high** | `drawdown.py:28-33,:62-71` |
| equity | account `get_account().equity`, cycle 1 then every 10 cycles; **stocks in RTH only** | `base_loop.py:1160-1173,:375-388,:417-419` |
| seed / persistence | $100k seed, dropped at the first real read (D15); peak saved **per book** in `{prefix}_position_state.json` | `drawdown.py:25`; `base_loop.py:220-223,:529-530,:560` |
| restart | `_update_equity` first, then `restore_peak_equity(saved, cur, seed=cur)` (cur = 0 if the fetch failed) | `base_loop.py:593,:602-612`; `drawdown.py:36-59` |
| use | f_dd × 11 other factors → clip to [0.1, 1.30] → base·kelly·vol_mult·tilt; v2 keeps dd_mult separate | `base_loop.py:2625-2629,:2699,:2753-2755,:2793` |
| base / breaker | risk $ = equity × 0.005, so base is ∝ equity. Breaker: 5 % below `last_equity`, account-wide; flattens own book, halts to ~16:05 ET; first each cycle; fails closed | `base_loop.py:2534,:158,:394,:729-880`; `order_utils.py:1265-1293` |

### E3(b) — findings

- **B1. OBJECTIVE arithmetic; JUDGMENT remedy.** The 0.1 floor masks the deepest rung. At April's regime (macro 0.56×, vix_tilt 0.7): 0.7 × 0.56 × 0.25 = 0.098 → floored to 0.1, so the 20 % rung acts as 0.255. With VIX > 25 it acts as 0.36. The floor is intended (`strategy_config.py:59-62`).
- **B2. OBJECTIVE; size unmeasured.** Two peaks track one account, and the stock book samples only in RTH. The books can sit on different rungs for the same account drawdown.
- **B3. OBJECTIVE, already an owner decision.** If the startup account read fails, `_equity` stays at $100k (`:221`) while the saved peak is adopted. If the breaker's read succeeds and cycle 1's refresh fails, sizing uses $100k for up to 10 cycles: about 18 % dd → 0.5× on $121.9k. `base_loop.py:1175-1179` calls fail-closed "an owner decision".
- **B4. JUDGMENT.** Base ∝ equity, dd_mult and the EWMA book-vol scalar (`portfolio.py:573`) all read one equity path. At 20 % dd, notional is 0.8 × 0.25 = 0.20 of the peak-sized amount.
- **B5. JUDGMENT.** The breaker is a 1-day, account-wide drawdown that flattens only its own book (`order_utils.py:1268-1270`). The ladder uses the cumulative HWM. There is no re-risk rule except a new high.

### E3(c) — survey (what differs from round 1)

Round 1 looked for OOS tests of ladders, crypto vol and Kelly. This round asked instead about placebo tests of leverage dials, and about how deep drawdowns *should* run (rung calibration).

- **S5:** a [0.25, 1] dial "pays in drawdown, not growth" and "truncates the drawdown tail without reducing typical drawdown". Its trigger is model miscoverage, not equity.
- **S6:** a Gaussian depth table "mis-warns" under skew and fat tails, so fixed 10/15/20 % rungs mean different things at different book vol and Sharpe.

### E3(d) — real equity replay and X11

**Data.** The hourly series was rejected: 71 of 173 day-aligned points differ by more than 2 %, and 04-17→06-16 reads about $94–10k (cause **unverified**). The daily series was used: 172 points, Δequity = `profit_loss` within $1 (no D29 deposits), with the 08-13 $93.63 outlier dropped. Close-to-close moves give only a lower bound on breaker trips.

| window | days | max DD | ladder days 0.75/0.5/0.25 | close-to-close ≥ 5 % | both |
|---|---|---|---|---|---|
| bots running 02-24→05-07 | 52 | 6.13 % | 0/0/0 | 0 (worst 3.44 %) | 0 |
| no bot 05-08→09-26 (untended) | 97 | 30.42 % | 5/1/52 | 4 (06-03/04/06, 09-16) | 3 |

The bot logs show 0 trips and 26 fail-closed API-error lines. No `*position_state.json` exists, so the ladder starts at 1.0. **Verdict:** neither mechanism was ever active while the bots ran.

**X11 (measurement-only; needs the bots running).**
- Inputs: buy-row `sizing` (`base_loop.py:2801,:3617-3619`), trip rows, and both files' daily `peak_equity`.
- After ≥ 60 RTH days, report: (i) dd < 1 days per book; (ii) the share of dd < 1 buys with `tilt_raw` < 0.1; (iii) the maximum peak divergence; (iv) days halted with dd < 1.
- Rules: (ii) ≥ 25 % → owner item "exempt dd_mult from the floor?"; (iii) ≥ 1 % → one account peak behind `DD_PEAK_ACCOUNT_SHARED` (default OFF, gate-facing); dd never < 1 → "ladder untested", no change.

## R6 scout — E1 follow-up: Alpaca paper-account quirks and their engine consequences (W19, 2026-09-27)

Read-only GETs only (stdlib urllib, paper host asserted): `/v2/account`, `/v2/account/activities` (all 2,276 rows),
`/v2/orders?status=all` (1,320 orders, 3-day windows), `/v2/positions`, `/v2/positions/{sym|asset_id}`,
`/v2/assets/{sym|id}`, `/v2/account/portfolio/history` (1D), daily crypto bars. Scripts and raw JSON:
ENGINE `w19/ro_history.py`, `ro_orders2.py`, `ro_assets.py`, `ro_bars.py` (+ `.json`/`.out`). Local sources:
`crypto_bot_output.log` (Feb 23 → May 5), `trade_memory.json`. New web sources (accessed 2026-09-27; S6/S7 above re-read): D1 Alpaca "Crypto Spot Trading", upd. 2025-09-24, https://docs.alpaca.markets/docs/crypto-trading · D2 "Paper Trading", upd. 2026-07-07, https://docs.alpaca.markets/docs/paper-trading ·
D3 "Placing Orders", upd. 2026-08-10, https://docs.alpaca.markets/docs/orders-at-alpaca · D4 API ref "Replace Order", undated, https://docs.alpaca.markets/reference/patchorderbyorderid-1 ·
D5 API ref "Get an open position", undated, https://docs.alpaca.markets/reference/getopenposition-1 · F1 forum "Crypto order with order_class", 2023-01-30 / 2023-07-17, https://forum.alpaca.markets/t/crypto-order-with-order-class/11596

### Q1 — where the six zero-basis positions came from (OBJECTIVE facts, one inference flagged)
Account `created_at` 2026-01-18; one JNLC +$100,000 on 2026-01-20; activity types in the whole history are only FILL 1,408 · CFEE 843 · FEE(TAF) 24 · JNLC 1, so there was no reset (D2: resets no longer exist, a new account is created instead) and no transfer or adjustment activity. Every client_order_id in 1,320 orders is a bare UUID
except the 12 `cstop-` of 2026-09-27. The `trader-`/`maker-` tags do not appear, so no tactic is recoverable from tags.
| sym | first fill | last fill | Σfills+Σin-kind CFEE | broker qty (log from 04-11; now) | Δ vs ledger | entry at 05-03 restart |
|---|---|---|---|---|---|---|
| BTC | 02-07 | 04-08 | 0.278415065 | 0.249291739 | −10.5 % | 69,998.61 |
| ETH | 02-07 | 04-08 | 7.908339351 | 7.904614454 | −0.05 % | 2,156.84 |
| SOL | 02-07 | 04-26 | 199.791801742 | 132.38 → 212.082911929 (05-03) | +6.2 % | 83.585 |
| LINK | 02-07 | 04-26 | 1678.703400196 | 1257.91 → 1576.529633666 (05-03) | −6.1 % | 9.0725 |
| XRP | 02-07 | 04-08 | 13848.618804276 | 13641.540903233 | −1.5 % | 1.354763 |
| DOGE | 02-07 | 04-08 | 118629.018064669 | 114913.12732127 | −3.1 % | 0.092206 |
(Broker qty: `crypto_bot_output.log:729312` 2026-04-11 22:28 CDT, `:1306382` 05-03. Entry: `trade_memory.json` 05-03 rows. Orders-net equals activities-net exactly for all six. The SOL running ledger is negative (−12.41) on 04-01.)
1. **The size is ours: a buy → false-DESYNC → re-buy loop (Apr 5 – May 3).** There are 67 `[DESYNC] … Position gone at broker
   (bracket stop filled?)` lines: 03-16 ×2, 03-31 ×4, 04-05 ×6, 04-06 ×12, 04-07 ×12, 04-08 ×6, 04-10, 04-11 ×6, 04-19 ×6,
   04-25 ×6, 04-26 ×2, 05-03 ×6. Buys each cycle at `pred=0.0000` (`:571640-571765`). From 04-01 the six names took
   $95.8k of buy fills and **zero** sell fills. The 04-11 restart listed all six positions 3 s before cycle 1 flagged all six
   "gone" (`:729312-729326`). The same happened at every restart, so this is not a broker disappearance. F-S7 reports
   no April-2026 event. The emitting code was never committed (`git log -S` is empty). The only trace is the March
   `__pycache__/base_loop.cpython-312.pyc` (`_verify_broker_position` → `get_position(symbol)`). The exact false-positive
   mechanism is **unverified**. Today's code has no per-cycle qty==0 DESYNC. It desyncs only after a rejected sell (`base_loop.py:1618-1649`).
   Consequence still live: the 67 phantom exits sit in `trade_memory.json` as `exit_reason='broker_stop'` with no
   `estimated` flag. Their pnl is up to +17.1 %, and they enter the Kelly sample (`trading_utils.py:239` drops only `estimated`).
2. **The quantity drift is the broker's.** For all six names, broker qty ≠ Σfills+ΣCFEE (−10.5 % … +6.2 %). The drift was
   already present at the 04-11 restart, and no non-trade activity explains it. It is a paper-ledger inconsistency of S6/S7 type (cause **unverified**).
3. **The zero basis is the broker's (S6 mechanism, strong circumstantial).** All six positions now carry an `asset_id` that
   none of our 1,320 orders ever used. E.g. the BTC position is on `64bbff51…` while every BTC order, including today's
   stops, is on `276e2673…`. Both ids resolve to an **active, tradable** `BTC/USD` (`w19/ro_assets.out`). S6 staff root
   cause: "Two CUSIPS for the same asset (BTC/USD)… purchased non-current CUSIP, preventing price lookup". On 05-03 the basis
   was still > 0 (trade_memory entries above). Since then the only broker-visible anomaly is the 1D portfolio-history print
   **2026-08-13 equity = $93.63 = cash**, back to $82,776.95 on 08-14. Equity on the other 100 days since 05-04 equals
   today's qty × close + $93.63 (median error 0.29 %, max 2.0 %). So the quantities did not change across that event.
   Inference (**unverified**): the S7 sync job re-created the book under the old asset ids around 2026-08-13 and dropped the basis.
   Measured today: `avg_entry_price`/`cost_basis` are the string `"0"` (not null); `GET /v2/positions/BTCUSD` and
   `/{asset_id}` → 200; `/v2/positions/BTC%2FUSD` → **404**.
- **Unverified and consequential:** whether a sell on the current asset id depletes a position held under the old id (known: placing the 6 stops dropped `qty_available` to 0, so the reservation links).

### Q2 — documented paper semantics that touch the engine
- **Reset:** D2 says "create and delete paper accounts, rather than resetting them… generate new API keys". A "reset" is now a new
  account. `position_state.json`/`trade_memory.json`/peak equity would then describe a different account. The engine has no
  account-id check (grep of `base_loop.py` for `account_number`: none). That is an owner/ops item, not a code fix.
- **Fields:** D5: `qty_available` = "Total number of shares available minus open orders". Measured: with the full-qty GTC
  stop_limit resting, `qty_available` = "0" on all six (`w19/ro_history.json`). This confirms the J1 mechanism: an add-on stop for
  old+new qty can only be covered by the add-on's own qty. `asset_marginable` = false (D1: "Cryptocurrencies cannot be bought
  on margin"). D1/D5 say nothing about how a zero or null basis arises.
- **Replace:** D4 says "The new Order object with the new order ID", and "A success return code from a replaced order does NOT
  guarantee the existing open order has been replaced. If the existing open order is filled before the replacing (new)
  order reaches the execution venue, the replacing (new) order is rejected". Crypto/stop_limit replace support is **not
  documented**. D3 lists replace only for bracket/OCO/trailing. By the docs, a REPLACE is still cancel-then-new at the venue.
  Whether it would shrink our measured 0.67–0.68 s client-side gap (W16) is **unverified** and needs a write experiment (owner).
- **Cancel:** D3 says `pending_cancel`, and cancellable "up until the point it reaches a state of either filled, canceled, or expired". Idempotency and latency are undocumented. Today's six sequential cancels carry `canceled_at` stamps 1.02 s apart (`w16/ro_orders.out`), so per-call latency is not separable.
- **Order classes / intent:** D1 lists only "Market, Limit and Stop Limit… `gtc`, and `ioc`" with no order classes. F1 (2023) error:
  "crypto orders not allowed for advanced order_class: oto". Staff: "Currently, OTO orders are not supported for crypto".
  2026 status **unverified**. `position_intent` is not mentioned in D3. So an entry-attached stop is not available, and
  `crypto_loop._after_entry_protection` (`crypto_loop.py:204-206`) stays necessary.

### Q3 — failure modes with no handler (today's behaviour → consequence → pre-registered harness test, NOT written)
H1 **Position hidden for ≥1 cycle (S7), equity intact.** The loop does not reconcile broker positions mid-run (grep: no
adopt/orphan path in `base_loop.py`). If an exit fires in the hidden cycle, `_execute_stop_exit` first cancels the
resting stop (`base_loop.py:1605`). The sell is then rejected, `verify_position` returns None (`order_utils.py:914-941`),
and the position gets an estimated `desync` row, a pop and a cooldown (`base_loop.py:1626-1648`). When the position
reappears it has **no resting stop and no tracking until restart**. Test T-H1: add `FakeBroker.vanish(sym, cycles=1)`
(404 "position does not exist" from get_position/list, orders kept). Drive a trailing breach in that cycle. Assert that
after reappearance, for every symbol with broker qty>0, a resting stop exists **or** the symbol is tracked within 2 cycles.
Expected today: FAIL. The fix changes exit behaviour, so it is an owner item.
H2 **Position hidden with equity = cash (the 2026-08-13 print).** dd = (123,095.98−93.63)/123,095.98 = 99.9 % trips
`check_circuit_breaker` (`order_utils.py:1274-1284`, called `base_loop.py:763`). `emergency_flatten` finds no targets
(`order_utils.py:1347`) and returns [] (`:1367,:1413`), then `self.positions.clear()` (`base_loop.py:854`) writes six estimated
`circuit_breaker` rows and halts. The stale resting stops survive (not in targets), but nothing ratchets them, and the
restored positions stay untracked until restart. Test T-H2: `vanish(..., equity_mode='cash')`. Assert (a) today's pin:
tracking cleared, 6 estimated rows, 6 stops still resting. (b) An owner-rule variant: no clear when the `list_positions`
set is empty but tracked stops are live. Which behaviour is right is an owner decision.
H3 **Filled buy, `filled_avg_price` None, broker avg "0".** `ofp` becomes 0 (`base_loop.py:3572`), entry stays 0, and the resting
stop lands at 0·(1−d) (`crypto_loop.py:205`, rejected). The buy row has fill 0 with no `estimated` flag. Test T-H3: the fake
returns a filled order with `filled_avg_price=None` plus `report_zero_basis`. Assert that the buy row carries `estimated=True`
or a `basis_unknown` marker and that no stop_limit is submitted at ≤ 0. Expected today: FAIL.
H4 **Add-on with a resting stop (qty_available < qty).** The J1 path: `_after_entry_protection` places a full-qty stop without cancelling
(`crypto_loop.py:204-206`) → "insufficient balance". `_place_resting_stop` then sets `stop_order_id=None` (`:201`), which blinds
server-fill detection (`base_loop.py:1418`) for one cycle. The exit path is clean: it cancels before selling (`:1605`), and
"position EXISTS but shares unavailable" only retries (`:1650-1652`). Test T-H4 (the fake already models `_reserved_qty`,
`tests/fake_alpaca_broker.py:429,472`): after an add-on fill, assert 0 rejected submits and exactly one live stop for the
total qty at end of cycle. Expected today: FAIL (W1 item 6).
H5 **`avg_entry_price` null (not "0").** Legacy SDK: `float(None)` raises in `reconstruct_positions._entry`
(`order_utils.py:1224`), and the position is **dropped** with "bad position payload" (`:1254-1256`). In `_place_and_track_buy`,
`float(pos.avg_entry_price)` (`base_loop.py:3563`) raises after a fill. alpaca-py shim: `_shim_position` raises
(`alpaca_compat.py:65`) inside `get_position`. `verify_position` swallows that and returns None, so every exit rejection
becomes a **false DESYNC of a live position** (`base_loop.py:1626`). Test T-H5: the fake's `report_null_basis`. Assert that
reconstruct keeps the position (entry 0 + warning) and that `verify_position` returns it. Expected today: FAIL. This is a
decision-neutral robustness candidate (class D) for the general. The live API returns "0" today, and the value it returned during S6 is **unverified**.

### Q4 — experiment X12 "paper-quirk census" (measurement-only, ENGINE, `scripts/paper_quirk_census.py`)
Stdlib + `.env` read, GET-only, paper-host assert, < 30 MB. Owner-run with `--interval 900`. Nothing is scheduled (gotcha #5).
Each snapshot appends one JSONL row to `logs/paper_census/YYYY-MM-DD.jsonl` (new STATE_FILES row). The row holds: per position
`symbol, asset_id, qty, qty_available, avg_entry_price, cost_basis, current_price`; the current `/v2/assets/{sym}.id`;
account `equity, cash, last_equity`; open orders `id, symbol, asset_id, type, qty, stop_price`; FILL/CFEE activities since the
last row; `position_state.json` keys. Detections:
V vanish (a symbol in the previous row or position_state is absent, with no sell FILL in between) · B basis (avg >0 → ≤0/null,
or qty>0 with cost_basis ≤0) · Q qty drift (|Δqty − Σfills − ΣCFEE| > max(1e-8, 1e-6·qty)) · E equity (equity − cash < 5 % of
Σqty·price while positions exist) · A asset split (position asset_id ≠ current asset id) · R phantom reserve (qty_available <
qty with no open order, or an open stop qty > position qty).
**Rule (pre-registered):** V, B, Q or E seen on 2 consecutive rows, or a single Q with |Δ|·price ≥ $50, triggers an owner alert via
`notify.notify(level='warning')` plus a row flag. A is logged once per day: 6/6 is the known standing state today, so A alone never
alerts. After 30 days: ≥1 V or E event → owner item for a runtime guard (H1/H2). 0 events other than A → close X12 and
keep only X3. The first real exit of a name with an A flag must show broker qty going to 0 on the old asset id; otherwise
file a venue item with Alpaca. The 63 phantom `broker_stop` rows since 04-05 go to the owner for a `venue_incident`/`estimated`
re-flag. This changes the Kelly sample, so it is not shipped.
Kill-list: checked. No reconciliation/census/desync entry. `08_removed_code.md`: no desync/reconcile/basis block (grep). Not a rebuild of `scripts/crypto_spread_census.py` or `crypto_quote_staleness_census.py` (quote-side only).
