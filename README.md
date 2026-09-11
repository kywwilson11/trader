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

> This README is orientation. **`CLAUDE.md` is the operational source of truth** (how to run and
> test, the two-machine reality, conventions, gotchas), and **`docs/MAP.md`** is the architecture
> narrative — repo layout, the full lifecycle, and the goal of each piece. If any doc and the code
> disagree, trust the code.

**Documentation map** — [`docs/README.md`](docs/README.md) indexes it all:
[`docs/MAP.md`](docs/MAP.md) (how it fits together) · [`docs/MODULES.md`](docs/MODULES.md)
(per-module reference + the complete CLI flag census) · [`docs/FLAGS.md`](docs/FLAGS.md) (every
behaviour flag, default and evidence gate) · [`docs/STATE_FILES.md`](docs/STATE_FILES.md) (every
generated file, its writer and reader) · [`docs/GLOSSARY.md`](docs/GLOSSARY.md) ·
[`CLAUDE.md`](CLAUDE.md) (running, testing, conventions) ·
[`research/README.md`](research/README.md) (the research record).

## What it is

- Two independent books — crypto and stocks — sharing one Template-Method engine (`base_loop.py`).
- **One** `RegressionLSTM` (`model_v2.py`) + LightGBM (`model_lgb.py`) blend per book, producing
  multi-horizon hourly return forecasts. The old dual bear/bull 3-class ensemble is **retired** —
  "bear/bull" survives only as regime diagnostics and as the champion/challenger shadow-slot names.
- A meta-label veto (`meta_label.py`, secondary classifier, vetoes trades with p < 0.30) and a
  shared cost model (`fees.py` + `liquidity.py`, `bidask` EDGE per-name spread) every gate uses.
- Honest validation: purged walk-forward with embargo, an untouched holdout, Deflated Sharpe
  (`DSR_MIN=0.60`), CSCV-PBO, label-overlap effective-n (`sample_weights.py`).
- A policy-replay promotion gate (`backtest.py`) that rolls a bad promotion back to `.prev`; every
  **model-facing** change ships only through challenger → shadow → DM-HLN deployment.
- Fail-closed execution and strict point-in-time (PIT) discipline throughout.

## Architecture

```
run_pipeline.py                    Orchestrator: harvest -> train -> gate -> launch bots -> weekly
                                    retrain (bots are STOPPED for the retrain, restarted after)
|-- scripts/harvest_crypto_data.py    1Y hourly OHLCV + features -> training_data.{csv,parquet}
|-- scripts/harvest_stock_data.py     1Y hourly OHLCV + features -> stock_training_data.{csv,parquet}
|-- scripts/hypersearch_v2.py         Optuna TPE search: RegressionLSTM + LightGBM leg, holdout DSR gate
|-- backtest.py                       Policy-replay promotion gate (real entries/exits/fees)
|-- crypto_loop.py                    24/7 crypto trading
|-- stock_loop.py                     Market-hours stock trading
+-- run_bots.py                       Live bots only — BOTH loops in one process
                                       (saves ~0.5-0.8 GB RAM on the Jetson)
```

Two hand-offs are easy to confuse. The **weekly retrain is a cold restart**: `run_pipeline` calls
`_stop_bots`, retrains, then `_restart_bots`, and the fresh loops read the champion slot in
`BaseTradingLoop._load_models`. The **daily shadow promotion is the hot-reload**:
`shadow.promote_challenger` writes `{prefix}model_v2.manifest.json` last, and
`base_loop._hot_reload_check` (keyed on that mtime via `trading_utils.model_reload_key`) swaps
models on the next 30 s cycle while the bots keep trading. Bot layout is a second either/or:
`run_pipeline`'s default is **one process per bot**, and `--combined-bots` (what the systemd unit
in `scripts/setup_jetson_system.sh` passes, so what production runs) hosts both loops in
`run_bots.py` instead.

Three shared kernels keep the system honest, and every consumer must use them:
`strategy_config.py` (the **single source of truth** for policy — both the live loops and the
backtester read it, so drift means the backtest validates something other than what trades),
`policy_exits.py` (one Numba exit stack shared by `backtest.py`, the harvest triple-barrier labels
and `meta_label.py`, guaranteeing label semantics == backtest == live), and `fees.py` +
`liquidity.py` (the cost model every gate shares). Around them sit the models
(`model_v2.py`, `model_lgb.py`, `predict_now.py`), the validation stack (`meta_label.py`,
`validation.py`, `sample_weights.py`), the multi-provider LLM overlay (`llm_client.py` /
`llm_analyst.py` — Gemini / Anthropic-Claude / OpenAI and OpenAI-compatible endpoints, all
schema-enforced, with cross-provider fallback), and `gui.py`, the PySide6 dashboard (8 tabs,
12 themes). Full narrative: [`docs/MAP.md`](docs/MAP.md); per-module reference:
[`docs/MODULES.md`](docs/MODULES.md).

## How a trade happens

data → features → RegressionLSTM+LightGBM blend → cost gate → meta-label gate → sentiment/LLM
gate → order → ATR-based exits → cross-book risk cap. Sizing uses `RISK_PCT_PER_TRADE=0.005`,
`KELLY_CAP=0.25`, `MAX_BOOK_RISK_PCT=0.025` as of this writing — **current values live in
`strategy_config.py`; trust it, not this paragraph.**

## Two-machine reality

Development happens on a Mac that deliberately lacks the heavy dependencies (no torch, lightgbm,
optuna, joblib, numba, sklearn, dotenv, alpaca, PySide6, pyarrow, arch, hmmlearn); training,
harvest, live trading and the GUI happen on the Jetson. The authoritative table is in
[`CLAUDE.md`](CLAUDE.md) § Two-machine reality; the Mac↔Jetson sync is user-driven and **not
encoded in the repo**.

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
pip install -r requirements.txt && pip install torch torchvision          # desktop
# Jetson (PyTorch from Jetson AI Lab wheels)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -r requirements-jetson.txt
```

> **Note:** PyTorch 2.9.1 is broken on Jetson (missing `libcudss.so.0`). Use 2.8.0.
> Parquet is a first-class data path but `pyarrow` is in none of the requirements files — install
> it explicitly (`pip install pyarrow`) on any machine that harvests or reads archives.

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
Read the output: the script prints failures but still exits 0.

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

The daily commands with their verified flags; **every** entry point, flag and research CLI is
catalogued in [`docs/MODULES.md`](docs/MODULES.md) and [`scripts/README.md`](scripts/README.md).

| Command | What it does |
|---|---|
| `python run_pipeline.py` | Orchestrator: harvest → train → gate → launch bots → weekly retrain (the retrain **stops** the bots and restarts them; the daily shadow promotion is the hot-reload). Flags: `--trials N` (200), `--retrain-trials N` (100), `--retrain-day N` (5), `--retrain-hour N` (2), `--no-retrain`, `--bot-only`, `--skip-harvest`, `--combined-bots`, `--crypto-only`, `--stock-only` |
| `python run_bots.py [--crypto-only\|--stock-only]` | Live bots only — BOTH loops in one process (saves ~0.5–0.8 GB RAM on Jetson). `run_pipeline` uses this only under `--combined-bots`; its own default is one process per bot |
| `python scripts/harvest_crypto_data.py` / `scripts/harvest_stock_data.py` | 1Y hourly OHLCV + features → `*training_data.{csv,parquet}` (no flags) |
| `python scripts/hypersearch_v2.py --trials N [--prefix stock] [--data F] [--fresh] [--shadow] [--preset P] [--max-rows N] [--mode M] [--no-status]` | Optuna TPE search (RegressionLSTM + LightGBM leg), holdout DSR gate |
| `python backtest.py --prefix {''\|stock} --days N [--gate] [--min-sharpe X] [--min-dsr X] [--model-prefix P] [--trials N] [--no-stage0-dump] [--fee-mult X] [--fee-sweep L]` | Policy replay (real entries/exits/fees); `--gate` rolls back to `.prev` on Sharpe/DSR fail |
| `python decision_report.py --days N` | Per-trade gate attribution + conviction calibration (Stage-0 measurement) |
| `python beta_ledger.py --days N [--lags N] [--equity-csv F] [--benchmarks-csv F] [--json F]` | Realized-beta ledger: daily equity vs SPY+BTC (lagged AKL betas, HAC alpha t-stat, up/down + trend-conditional betas). Measurement-only |
| `python indicator_leadlag.py --data F [--preset P] [--features L] [--horizons H] [--fdr-q Q] [--json F]` | Per-feature leading/lagging diagnostic: predictive IC vs reactive coupling at 1–48h (overlap-adjusted, FDR), redundancy clusters + exact dupes. Measurement-only |
| `python llm_eval.py --days N [--asset {crypto,stock}] [--advisor]` | Measures whether the LLM gate predicts returns |
| `python gui.py` | PySide6 dashboard (8 tabs, 12 themes); reads `pipeline_status.json` + logs |

## Testing & CI

Canonical dev-Mac command (some test modules can't import their heavy deps, so always continue
past collection errors):
```bash
python3 -m pytest tests/ --continue-on-collection-errors -q
```

The current baseline — the only place a suite count is quoted — is [`CLAUDE.md`](CLAUDE.md)
§ Running tests; its failures and errors are all pre-existing missing-dependency noise on the Mac,
and the suite is green on the full Jetson stack. The standard regression check is
**`bash scripts/ab_check.sh`**, which judges by failure *names* against
`tests/baseline_failures.txt` — never by counts; a `git stash` A/B is the fallback for when that
baseline may itself be stale (see `CLAUDE.md` and the `/regression-ab` skill).
`tests/test_sentiment_headlines.py` is a standalone runner, not a pytest module — run it with
`python tests/test_sentiment_headlines.py`. Suite conventions: [`tests/README.md`](tests/README.md).

CI (`.github/workflows/ci.yml`, Ubuntu) runs two legs — **py3.10 jetson-parity** (`requirements-ci.txt`,
the production pins) and **py3.12 modern** (unpinned forward-compat) — each doing `py_compile` of the
top-level and `scripts/` files → the sentiment runner → `pytest tests/ -v --tb=short -x`.

## Repo map

Every module — signals/features, execution, costs/risk, models, LLM, ops — with its goal, public
API, callers and the files it reads and writes: [`docs/MODULES.md`](docs/MODULES.md); how they
compose: [`docs/MAP.md`](docs/MAP.md). Per-function docs live in the code.

## Generated / local files

Models (`*.pth`/`*.pkl`), Optuna `*_study.db`, training CSV/parquet, `journals/`,
`pipeline_status.json`, `*_predictions.json`, `sentiment_cache.db*`, `llm_config.json` (holds API
keys), `indicator_config.json`, and logs are all **gitignored** — generated or secret, local only;
source, tests, `docs/`, `research/`, `.claude/`, `archive/`, requirements and policy config are
committed. Authoritative per-file table: [`docs/STATE_FILES.md`](docs/STATE_FILES.md). Nothing here
is deleted — files that outlive their place are **moved** to `archive/` or a dated `research/`
subdirectory and recorded in [`archive/README.md`](archive/README.md).

## Research process

Research is a record, not a backlog — every round is dated and frozen, and indexed by
[`research/README.md`](research/README.md). In order: **waves 1–9** (2026-06, `research/waves/`,
each with its survivors and its kill list) → the **2026-07 module review** (69 modules / 600
functions; 280 safe fixes applied, a 90-item owner decision queue left in
`research/module_review_2026-07.json` — render it with `/decision-queue`; panel reviews frozen in
`research/reviews_2026-07/`) → the **2026-08 comprehensive campaign** (10 waves in
`research/campaign_2026-08/`, including the Jetson activation runbook; every model-facing change
behind a default-OFF flag) → **R2 / R3** (2026-08, still in owner review: the R2 signal-model round
and the R3 econ/Nobel literature round in `research/literature/`). Two rules bind every round:
`research/KILL_LIST.md` is the consolidated do-not-rebuild list, and `research/AGENT_CONTEXT.md`
is the brief every spawned agent reads first.
