<p align="center">
  <img src="logos/circuit_bull.png" alt="Trader" width="200">
</p>

<h1 align="center">Trader</h1>

<p align="center">
  <a href="https://github.com/kywwilson11/trader/actions/workflows/ci.yml">
    <img src="https://github.com/kywwilson11/trader/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

<p align="center">
  Autonomous paper-trading system for US stocks (market hours) and crypto (24/7). One RegressionLSTM
  + LightGBM blend per book, meta-labeling veto, cost-aware gates, honest validation (purged
  walk-forward + Deflated Sharpe), and a policy backtester that replays real exits before any model
  is promoted. Runs in production on an NVIDIA Jetson Orin Nano (8 GB); trades via the
  <a href="https://alpaca.markets/">Alpaca</a> paper API.
</p>

> **`CLAUDE.md` is the operational source of truth** (how to run/test, the two-machine reality,
> conventions). This README is orientation, rewritten 2026-07-15 from `CLAUDE.md` + source. If this
> file and the code disagree, trust the code.

## What it is

- Two independent books — crypto and stocks — sharing one Template-Method engine (`base_loop.py`).
- **One** `RegressionLSTM` (`model_v2.py`) + LightGBM (`model_lgb.py`) blend per book, producing
  multi-horizon hourly return forecasts. The old dual bear/bull 3-class ensemble is **retired** —
  "bear/bull" survives only as regime diagnostics and as the champion/challenger shadow-slot names.
- A meta-label veto (`meta_label.py`, secondary classifier, vetoes trades with p < 0.30).
- A shared cost model (`fees.py` + `liquidity.py`, `bidask` EDGE per-name spread) every gate uses.
- Honest validation: purged walk-forward with embargo, an untouched holdout, Deflated Sharpe
  (`DSR_MIN=0.60`), CSCV-PBO, label-overlap effective-n (`sample_weights.py`).
- A policy-replay promotion gate (`backtest.py`) that rolls a bad promotion back to `.prev`.
- Every **model-facing** change ships only through challenger → shadow → DM-HLN deployment.
- Fail-closed execution and strict point-in-time (PIT) discipline throughout.

## Architecture

```
run_pipeline.py                    Orchestrator: harvest -> train -> gate -> launch bots -> weekly
                                    retrain (hot-reload; bots never stop)
|-- scripts/harvest_crypto_data.py    1Y hourly OHLCV + features -> training_data.{csv,parquet}
|-- scripts/harvest_stock_data.py     1Y hourly OHLCV + features -> stock_training_data.{csv,parquet}
|-- scripts/hypersearch_v2.py         Optuna TPE search: RegressionLSTM + LightGBM leg, holdout DSR gate
|-- backtest.py                       Policy-replay promotion gate (real entries/exits/fees)
|-- crypto_loop.py                    24/7 crypto trading
|-- stock_loop.py                     Market-hours stock trading
+-- run_bots.py                       Live bots only — default runs BOTH loops in one process
                                       (saves ~0.5-0.8 GB RAM on the Jetson)
```

Shared kernels and support modules:

- `base_loop.py` — Template-Method base both trading loops are built on.
- `strategy_config.py` — **single source of truth** for policy (ATR mults, stop floors, TP RR,
  risk sizing, entry windows, execution tactics). Both the live loops **and** the backtester read
  it — drift here means the backtest validates a different policy than what trades.
- `policy_exits.py` — shared Numba exit-stack kernel (hard/trailing/TP/signal/EOD/vertical). Used
  by `backtest.py`, the harvest triple-barrier labels, and `meta_label.py`, guaranteeing label
  semantics == backtest == live.
- `model_v2.py` / `model_lgb.py` / `predict_now.py` — model definitions and live inference.
- `meta_label.py` / `validation.py` / `sample_weights.py` — meta-label veto, validation, and
  sample-weighting.
- `fees.py` + `liquidity.py` — the cost model every gate shares.
- `llm_client.py` / `llm_analyst.py` — multi-provider LLM client (Gemini / Claude / OpenAI, all
  schema-enforced) used for pre-trade risk overlay.
- `gui.py` — PySide6 dashboard, 8 tabs, 10 themes.

## How a trade happens

data → features → RegressionLSTM+LightGBM blend → cost gate → meta-label gate → sentiment/LLM
gate → order → ATR-based exits → cross-book risk cap.

Sizing uses `RISK_PCT_PER_TRADE=0.005`, `KELLY_CAP=0.25`, `MAX_BOOK_RISK_PCT=0.025` as of this
writing — **current values live in `strategy_config.py`; trust it, not this paragraph.**

## Two-machine reality

Work happens across two machines with **different installed dependencies**:

| | **This dev Mac** | **Jetson Orin Nano (prod)** |
|---|---|---|
| Python | 3.13.5, framework `python3` (no venv) | 3.10, `/home/kyle/miniforge3/envs/jetson/bin/python` |
| Installed | numpy, pandas, scipy, statsmodels, bidask, yfinance, bs4, yaml, **pytest** | full stack incl. torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, CUDA |
| **NOT installed** | **torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, finnhub, alpaca, PySide6** | — |
| Can do | pure-algorithm code, synthetic-data unit tests, web research | training, harvest, Stage-0 measurement, live trading, GUI |
| **Cannot do** | model training, data harvest, anything importing the heavy deps, run the bots/GUI | — |

The Mac↔Jetson sync is user-driven and **not encoded in the repo**.

## Quick start

### 1. Install

```bash
git clone git@github.com:kywwilson11/trader.git
cd trader

# Desktop
./scripts/setup.sh

# Jetson Orin Nano (JetPack 6.x)
./scripts/setup.sh --jetson
```

Or install manually:
```bash
# Desktop
pip install -r requirements.txt
pip install torch torchvision

# Jetson (PyTorch from Jetson AI Lab wheels)
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -r requirements-jetson.txt
```

> **Note:** PyTorch 2.9.1 is broken on Jetson (missing `libcudss.so.0`). Use 2.8.0.

### 2. Configure

Create a `.env` file (or let `scripts/setup.sh` create the template):
```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_finnhub_key
```

- **Alpaca** — sign up at [alpaca.markets](https://alpaca.markets/) for a free paper trading account.
- **Finnhub** — sign up at [finnhub.io](https://finnhub.io/) for a free API key (optional, stock
  news sentiment).
- **LLM analysis** (optional) — configure via the GUI Settings tab, or edit `llm_config.json`
  directly (gitignored — it holds API keys).

### 3. Verify connectivity
```bash
python scripts/connection_test.py
```

### 4. Run

```bash
python run_pipeline.py     # full pipeline: harvest -> train -> trade -> weekly retrain
python gui.py               # dashboard, separate terminal
```

**Jetson one-time system setup** (headless, NVMe swap, cuDSS/cuSPARSELt installed system-wide):
```bash
sudo bash scripts/setup_jetson_system.sh   # flags: --skip-headless / --skip-swap
sudo reboot
python run_pipeline.py --combined-bots
```

## Entry points

| Command | What it does |
|---|---|
| `python run_pipeline.py` | Orchestrator: harvest → train → gate → launch bots → weekly retrain (hot-reload, bots never stop). Flags: `--no-retrain`, `--bot-only`, `--skip-harvest`, `--combined-bots`, `--crypto-only`, `--stock-only`, `--trials N` (default 200), `--retrain-trials N` (default 100), `--retrain-day N` (default 5), `--retrain-hour N` (default 2) |
| `python run_bots.py [--crypto-only\|--stock-only]` | Live bots only — default runs BOTH loops in one process (saves ~0.5–0.8 GB RAM on Jetson) |
| `python scripts/harvest_crypto_data.py` / `scripts/harvest_stock_data.py` | 1Y hourly OHLCV + features → `*training_data.{csv,parquet}` |
| `python scripts/hypersearch_v2.py --trials N [--prefix stock] [--data F] [--fresh] [--shadow] [--preset P] [--max-rows N]` | Optuna TPE search (RegressionLSTM + LightGBM leg), holdout DSR gate |
| `python backtest.py --prefix {''\|stock} --days N [--gate] [--min-sharpe X] [--min-dsr X]` | Policy replay (real entries/exits/fees); `--gate` rolls back to `.prev` on Sharpe/DSR fail |
| `python decision_report.py --days N` | Per-trade gate attribution + conviction calibration (Stage-0 measurement) |
| `python beta_ledger.py --days N [--json F]` | Realized-beta ledger: daily equity vs SPY+BTC (lagged AKL betas, HAC alpha t-stat, up/down + trend-conditional betas). Measurement-only |
| `python indicator_leadlag.py --data F [--preset P] [--json F]` | Per-feature leading/lagging diagnostic: predictive IC vs reactive coupling at 1–48h (overlap-adjusted, FDR), redundancy clusters + exact dupes. Measurement-only |
| `python llm_eval.py --days N [--asset {crypto,stock}]` | Measures whether the LLM gate predicts returns |
| `python gui.py` | PySide6 dashboard (8 tabs, 10 themes); reads `pipeline_status.json` + logs |

## Testing & CI

Canonical dev-Mac command (some test modules can't import their heavy deps, so always continue
past collection errors):
```bash
python3 -m pytest tests/ --continue-on-collection-errors -q
```

Baseline (verified 2026-07-15): `1887 passed, 21 failed, 15 skipped, 7 errors` — the failures + errors are
**all** pre-existing missing-dependency noise (no torch/lightgbm/optuna/joblib/numba/sklearn/dotenv
on the Mac); the suite is green on the full Jetson stack. The standard regression check is a
git-stash A/B, diffing failure **names** (not counts) — see `CLAUDE.md` and the `/regression-ab`
skill.

`tests/test_sentiment_headlines.py` is a standalone runner (~1060 assertions), not a pytest module —
run it with `python tests/test_sentiment_headlines.py` (needs deps → Jetson/CI).

CI (`.github/workflows/ci.yml`, Ubuntu, Python 3.10 + 3.12): `py_compile` of all top-level and
`scripts/` `.py` files → the sentiment runner → `pytest tests/ -v --tb=short -x`.

The suite is 133 test files / ~1890 passing tests and moving — trust `python3 -m pytest --collect-only`
over any number quoted here.

## Repo map

Signals/features → `indicators.py`, `sentiment*.py`, `fundamentals.py`, `volatility.py`,
`squeeze_features.py`, `*_archive.py` (funding/oi/basis); execution → `order_utils.py`,
`execution_policy.py`, `liquidity.py`; costs/risk → `fees.py`, `short_cost.py`, `borrow_proxy.py`,
`cost_regime.py`, `risk_budget.py`, `portfolio*.py`; models → `model_v2.py` (LSTM), `model_lgb.py`,
`predict_now.py`; LLM → `llm_client.py` / `llm_analyst.py` (Gemini + Anthropic/Claude, both
schema-enforced, cross-provider fallback); ops → `gui.py`, `gpu_lock.py`, `hw_monitor.py`,
`monitor_drift.py`.

~80 top-level modules, 133 test files. Per-function documentation lives in the code, not here.

## Generated / local files

Models (`*.pth`/`*.pkl`), Optuna `*_study.db`, training CSV/parquet, `journals/`,
`pipeline_status.json`, `*_predictions.json`, `sentiment_cache.db*`, `llm_config.json` (holds API
keys), `indicator_config.json`, and logs are all **gitignored** — generated or secret, local only.

Committed: source `.py`, `tests/`, `research/*.json`, `requirements*.txt`, `stock_universe.json`,
and `strategy_config.py` & friends.

## Research process

Completed research is organized into numbered "waves" and saved to `research/waveN_research.json`
(committed). The 2026-07 function-by-function module review (69 modules, 600 functions) applied
280 safe fixes directly and left a 90-item owner decision queue in
`research/module_review_2026-07.json` — render it with the `/decision-queue` skill.
