# CLAUDE.md — trader

Autonomous **paper-trading** system (Alpaca) for **crypto** (24/7) and **US stocks** (market hours).
One **RegressionLSTM + LightGBM blend per book** (the old dual bear/bull ensemble is gone —
"bear/bull" survives only as regime diagnostics and the champion/challenger shadow slots)
+ meta-labeling, Numba TA features, honest validation (purged walk-forward + Deflated Sharpe),
cost-aware gating, and a policy backtester that replays *real* exits before any model is promoted. Runs in production on an **NVIDIA Jetson
Orin Nano (8 GB)**. Research is organized into numbered "waves" (see `research/waveN_research.json`).

> **The README is stale** — it describes old internals. Trust the code and this file, not `README.md`.
> Deep architecture invariants live in the `trader-architecture-truth` memory; current rolling
> state (what's uncommitted, what's next) lives in the `session-state` memory. **Read those two first.**

---

## ⚠️ Two-machine reality (the #1 operational fact)

Work happens across two machines with **different installed dependencies**:

| | **This dev Mac** | **Jetson Orin Nano (prod)** |
|---|---|---|
| Python | 3.13.5, framework `python3` (no venv) | 3.10, `/home/kyle/miniforge3/envs/jetson/bin/python` |
| Installed | numpy, pandas, scipy, statsmodels, bidask, yfinance, bs4, yaml, **pytest** | full stack incl. torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, CUDA |
| **NOT installed** | **torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, finnhub, alpaca, PySide6** | — |
| Can do | pure-algorithm code, synthetic-data unit tests, web research | training, harvest, Stage-0 measurement, live trading, GUI |
| **Cannot do** | model training, data harvest, anything importing the heavy deps, run the bots/GUI | — |

**Implication for how I plan work:** on the Mac, only build/verify things provable with
numpy/pandas/scipy/bidask + synthetic data. Anything needing torch/lightgbm/joblib/dotenv/real
journals/parquet is **Jetson-gated** — write it, unit-test the pure parts, and flag it for the
user to run on the Jetson. The Mac↔Jetson sync is user-driven and **not encoded in the repo** —
do not invent ssh/rsync steps.

---

## Running tests

**Canonical dev-Mac command** (12 modules can't import their heavy deps, so always continue past
collection errors):

```bash
python3 -m pytest tests/ --continue-on-collection-errors -q
```

**Current baseline (verified 2026-07-02): `780 passed, 34 failed, 1 skipped, 14 errors`.**
The 34 failures + 14 errors are **ALL pre-existing missing-dependency failures** (dotenv / torch /
lightgbm / optuna / joblib / numba / sklearn), **not regressions**. On the full Jetson stack the
suite is green.

**Verify a change introduced no regressions (the standard method):** A/B with `git stash` —
run the command above, `git stash`, run again, compare the failed/errored set. Identical set ⇒
zero regressions. Never treat the 34/14 baseline as "broken."

- `pytest tests/ --collect-only -q` reports **653 collected, 12 collection errors** (the collected
  count counts tests inside modules that then error at import).
- `tests/test_sentiment_headlines.py` is a **standalone runner** (~1060 assertions), not a pytest
  module — run it with `python tests/test_sentiment_headlines.py` (needs deps → Jetson/CI).
- CI (`.github/workflows/ci.yml`): Ubuntu, py3.10+3.12, full deps, `py_compile` syntax check →
  sentiment runner → `pytest tests/ -v --tb=short -x`.

---

## Running the system

All entry points are plain `python <file>.py`. On this Mac most require the Jetson stack; listed
for reference (exact flags verified from the source):

| Command | What it does |
|---|---|
| `python run_pipeline.py` | Orchestrator: harvest → train → gate → launch bots → weekly retrain (hot-reload, bots never stop). Flags: `--no-retrain`, `--bot-only`, `--skip-harvest`, `--combined-bots`, `--crypto-only`, `--stock-only`, `--trials N`, `--retrain-trials N` |
| `python run_bots.py [--combined-bots\|--crypto-only\|--stock-only]` | Live bots only (combined = both loops in one process, saves ~0.5–0.8 GB RAM on Jetson) |
| `python scripts/harvest_crypto_data.py` / `harvest_stock_data.py` | 1Y hourly OHLCV + features → `*training_data.{csv,parquet}` |
| `python scripts/hypersearch_v2.py --trials N [--prefix stock] [--data F] [--fresh] [--shadow] [--preset stationary]` | Optuna TPE search (LSTM + LightGBM leg), holdout DSR gate |
| `python backtest.py --prefix {crypto\|stock} --days N [--gate]` | Policy replay (real entries/exits/fees); `--gate` rolls back to `.prev` on Sharpe/DSR fail |
| `python decision_report.py --days N` | Per-trade gate attribution + conviction calibration (Stage-0 measurement) |
| `python beta_ledger.py --days N` | Realized-beta ledger: daily equity vs SPY+BTC (lagged AKL betas, HAC alpha t-stat, up/down + trend-conditional betas). Measurement-only |
| `python indicator_leadlag.py --data F [--preset P]` | Per-feature leading/lagging diagnostic: predictive IC vs reactive coupling at 1–48h (overlap-adjusted, FDR), redundancy clusters + exact dupes. Measurement-only |
| `python gui.py` | PySide6 dashboard (8 tabs, 9 themes); reads `pipeline_status.json` + logs |

---

## Architecture in brief

**Pipeline:** data → features → dual LSTM (bear/bull) → cost gate → meta-label gate → sentiment/LLM
gate → order → ATR-based exits → cross-book risk cap.

**Single source of truth — `strategy_config.py`.** `CRYPTO_POLICY`/`STOCK_POLICY` (ATR mults, stop
floors, TP RR, cooldowns), `RISK_PCT_PER_TRADE=0.005`, `MAX_BOOK_RISK_PCT=0.025`, `KELLY_CAP=0.25`,
vol targets, entry windows, overnight sleeve, execution tactics, IOC caps. **Both the live loops AND
the backtester read it** — drift here means the backtest validates a different policy than trades.

**Shared kernels (one implementation, many consumers — keep them in sync):**
- `policy_exits.py` — Numba exit-stack kernel (hard/trailing/TP/signal/EOD/vertical). Used by
  `backtest.py`, the harvest triple-barrier labels, AND `meta_label.py`. Guarantees
  label semantics == backtest == live. `exit_walk(side=+1)` is the live long path (untouched);
  `side=-1` is the offline short mirror.
- `fees.py` + `liquidity.py` (`bidask` EDGE per-name spread) — the cost model every gate shares.
- `base_loop.py` — Template-Method base for `crypto_loop.py` + `stock_loop.py`.

**Validation/promotion:** `validation.py` (Deflated Sharpe, `DSR_MIN=0.60`; CSCV-PBO; Lo-2002 serial
factor), `sample_weights.py` (avg-uniqueness → effective-n), `backtest.py` (policy-replay promotion
gate), `meta_label.py` (secondary classifier; veto p<0.30).

**Module map (one line each):** signals/features → `indicators.py`, `sentiment*.py`,
`fundamentals.py`, `volatility.py`, `squeeze_features.py`, `*_archive.py` (funding/oi/basis);
execution → `order_utils.py`, `execution_policy.py`, `liquidity.py`; costs/risk → `fees.py`,
`short_cost.py`, `borrow_proxy.py`, `cost_regime.py`, `risk_budget.py`, `portfolio*.py`; models →
`model_v2.py` (LSTM), `model_lgb.py`, `predict_now.py`; LLM → `llm_client.py`/`llm_analyst.py`
(Gemini + Anthropic/Claude, both schema-enforced — Gemini responseSchema, Claude forced tool use;
provider switch + per-role overrides + pricing corrections in `llm_config.json`; cross-provider
fallback; `ANTHROPIC_API_KEY` env accepted; backfill role stays Gemini — Batch API);
ops → `gui.py`, `gpu_lock.py`, `hw_monitor.py`, `monitor_drift.py`.
For full detail see the `trader-architecture-truth` memory.

**PIT discipline (do not break):** all features strictly trailing; sentiment/short-interest lagged
to publication date; borrow cost regime-dated; universe membership as-of (no survivorship).

---

## Conventions

- **Commit/push ONLY when the user explicitly asks.** Currently a large body of work is uncommitted
  — see `session-state`. The user reviews before anything is committed.
- **Commit style:** conventional prefixes seen in history — `feat:`, `fix:`, `docs:`, `test:`,
  imperative mood, often `feat: wave-N <scope> — <detail>`. Branch is `master`; remote `origin`.
- **Deployment gate:** every **model-facing** change ships only through the
  **challenger → shadow → DM-HLN** promotion path. **Instrumentation/measurement-only** changes
  (journals, reports, offline research kernels) are safe to ship directly.
- **Research → ship:** completed wave research is saved to `research/waveN_research.json` (committed).
  Implement `mac_now` items on the Mac, defer `jetson_later` items. Each wave's memory file holds the
  survivors + **kill list** (things research rejected — do NOT rebuild them).
- **Effort:** the user wants real max-effort, literature-grounded, multi-agent work and accepts long
  runs. Priorities, in order: **Jetson 8 GB memory/perf › financial soundness › LLM utilization ›
  trading strategy.** See `user-working-style` memory.

---

## Environment & secrets

- **Secrets** (in `.env`, gitignored): `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`,
  `FINNHUB_API_KEY`.
- **Feature flags / env:** `TRADER_USE_ALPACA_PY` (use alpaca-py adapter), `TRADER_SHADOW_MODE`,
  `TRADER_ORDER_STREAM`, `TORCH_NUM_THREADS` (=2 for bots), `CUDA_VISIBLE_DEVICES` (='' → bots CPU-only).
  Notifications: `TRADER_TELEGRAM_*`, `TRADER_WEBHOOK_URL`, `TRADER_HEALTHCHECK_URL`.
- **Config flags in `strategy_config.py`** (not env): `HAR_VOL_ENABLED`, `CONVICTION_JOURNAL_ENABLED`,
  `MAKER_ENTRIES_ENABLED`, `ENTRY_WINDOWS_ENABLED`, `OVERNIGHT_SLEEVE_ENABLED`;
  `OBJECTIVE_LONG_ONLY` (default off — score only the deployable long side in hypersearch; flipping
  = objective change ⇒ gotcha #2). In `indicator_config.py`: `HURST_ON_RETURNS` (default off — correct
  R/S input; model-facing, flip only with harvest+retrain ⇒ gotcha #2).

## What's committed vs generated

**Committed:** all `*.py`, `research/*.json`, `requirements*.txt`, `stock_universe.json`,
`strategy_config.py` & friends, `tests/`.
**Gitignored (generated/secret):** `.env`, `.claude/`, `*.pth`/`*.pkl`/`models/`, `*_study.db`,
`*.log`, training `*.csv`/`*.parquet`, `*_archive.parquet`, `journals/`, `*_predictions.json`,
`llm_config.json`, `indicator_config.json`, `sentiment_cache.db*`, `pipeline_status.json`, `*.prev`.
So: code + research + policy config are versioned; all data/models/runtime state/secrets are local.

## Gotchas

1. **README is stale.** Verify against code.
2. **First Jetson retrain after objective/feature changes:** delete `v2_study.db` + `stock_v2_study.db`
   (old Optuna scores incomparable) and reset the adaptive `best_score`.
3. **numpy pin:** `requirements.txt` says `numpy<2` but this Mac has 2.4.6 (py3.13 forces it). Fine for
   the pure modules; just don't trust the pin as reality on the Mac.
4. **Don't stack variance corrections:** `validation.serial_correlation_factor` (Lo-2002) is OFF by
   default and is *separate* from the label-overlap uniqueness effective-n — using both double-counts.
5. **No background automation is scheduled.** A prior timed workflow hop-chain is dead; do **not**
   create scheduled wakeups for this project unless the user asks.

---

*Memory (this index, `session-state`, `user-working-style`, `trader-architecture-truth`, the wave
archives) lives outside the git repo at
`~/.claude/projects/-Users-kywwilson-Desktop-Projects-trader/memory/`.*
