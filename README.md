# Trader

Autonomous paper trading system for stocks and crypto, powered by dual bear/bull LSTM models with Numba-accelerated technical indicators and sentiment-gated risk management.

Runs on an NVIDIA Jetson Orin Nano. Trades via the [Alpaca](https://alpaca.markets/) paper trading API.

## Architecture

```
run_pipeline.py                   Orchestrator (train → trade → weekly retrain)
├── harvest_crypto_data.py        Crypto training data (10 symbols, 1yr hourly)
├── harvest_stock_data.py         Stock training data (~45 symbols, 1yr hourly)
├── hypersearch_dual.py           Optuna hyperparameter search (bear + bull models)
├── crypto_loop.py                24/7 crypto trading (10 symbols)
└── stock_loop.py                 Market-hours stock trading (top 10 of ~45)

gui.py                            PySide6 dashboard (positions, P&L, news, markets, models, logs)
stock_config.py                   Market universe config (stocks + crypto → stock_universe.json)
watchdog.py                       Standalone process supervisor (alternative to pipeline)
evolve.py                         3-day retraining orchestrator (alternative to pipeline)
```

## Features

- **Dual bear/bull LSTM models** — separate models optimized per direction, hot-reloadable without downtime
- **ATR-based adaptive risk** — stop-loss, trailing stops, and take-profit scaled to current volatility
- **Sentiment gating** — Fear & Greed Index + Finnhub news scores modulate position sizing (0x–1.5x)
- **Confidence-based sizing** — trade notional scaled by prediction strength
- **Numba JIT indicators** — RSI, MACD, ATR, Bollinger Bands, Stochastic, OBV (~1.8x speedup)
- **Cross-asset features** — BTC prices for crypto correlations, SPY for stock relative strength
- **Circuit breaker** — auto-flattens all positions on 5% daily drawdown
- **Persistent Optuna studies** — Bayesian hyperparameter search with SQLite memory across cycles
- **Weekly auto-retrain** — Saturday 2 AM: harvest fresh data, retrain models, bots hot-reload improvements
- **Position reconstruction** — survives crashes by syncing state from Alpaca API on restart
- **PySide6 dashboard** — live positions, P&L, orders, news (filterable), markets (stocks + crypto heatmap/chart), model status, hardware gauges, 7 themes

---

## Module Reference

### Trading Loops

#### `crypto_loop.py` — Crypto Trading Bot

24/7 cryptocurrency trading loop for 10 symbols.

| Parameter | Value |
|---|---|
| Schedule | 24/7, 30-second cycle |
| Universe | BTC, ETH, XRP, SOL, DOGE, LINK, AVAX, DOT, LTC, BCH |
| Position size | $250/trade, confidence-scaled |
| Stop-loss | ATR × 2.0 (fallback: 4%) |
| Trailing stop | ATR × 1.5 (fallback: 3%) |
| Take-profit | 2:1 risk-reward (fallback: 12% cap) |
| Cooldown | 30 min per symbol |

- Loads dual bear/bull models, runs parallel predictions via `ThreadPoolExecutor` (5 workers)
- Writes `crypto_predictions.json` each cycle for GUI Markets tab (bear/bull/score/signal per symbol)
- Sentiment multiplier (0.3x–1.3x) gates all trades based on Fear & Greed + symbol-specific news
- Circuit breaker: 5% daily equity drawdown flattens all positions + 1-hour halt
- Hot-reload: detects `.pth` file changes and reloads models without stopping

#### `stock_loop.py` — Stock Trading Bot

Market-hours stock trading loop with dynamic top-N selection.

| Parameter | Value |
|---|---|
| Schedule | 9:30 AM – 4:00 PM ET, weekdays only |
| Universe | ~45 high-beta stocks → top 10 by bull signal |
| Position size | $2,500/position, $25k max exposure |
| Stop-loss | ATR × 2.0 (fallback: 3%) |
| Trailing stop | ATR × 1.5, upgrades at 1.5% profit (fallback: 2%) |
| Take-profit | 2:1 risk-reward (fallback: 10%, cap: 15%) |
| Flatten time | 3:50 PM ET (avoid overnight gap risk) |

- Scores all ~45 stocks with both models, trades only the top 10 by bull confidence
- Writes `stock_predictions.json` each cycle for GUI Markets tab (bear/bull/score/signal per symbol)
- Dynamic stock universe: reloads `stock_universe.json` each cycle so GUI edits take effect live
- Bracket orders: buy with stop-loss + take-profit children
- End-of-day: cancels all stop orders and market-sells all positions at 3:50 PM ET
- SPY relative strength features for cross-market context

### Pipeline & Training

#### `run_pipeline.py` — Pipeline Orchestrator

End-to-end system: initial training → bot startup → weekly retrain with hot-reload.

**Phases:**
1. **Harvest** — download 1-year hourly bars (skips if data < 24h old)
2. **Train** — Optuna hypersearch for bear + bull models (crypto and/or stock)
3. **Launch bots** — start crypto and stock loops as background processes
4. **Weekly retrain** — Saturday 2 AM: re-harvest, retrain with fewer trials, bots hot-reload

Bots are never interrupted for retraining. They run continuously and detect new model files via mtime checking. If a retrain produces a better model, the `.pth` file is overwritten and bots pick it up automatically.

```
Flags:
  --trials N          Trials per model, initial training (default: 250)
  --retrain-trials N  Trials per model, weekly retrain (default: 50)
  --retrain-day N     Day of week 0=Mon..6=Sun (default: 5=Saturday)
  --retrain-hour N    Hour 0-23 (default: 2)
  --bot-only          Skip training, start bots immediately
  --skip-harvest      Use existing CSVs
  --crypto-only       Crypto models + bot only
  --stock-only        Stock models + bot only
  --no-retrain        Train once, run bots forever
```

Writes `pipeline_status.json` every 2 seconds for GUI monitoring (phase, trial count, best scores, bot health, next retrain time).

#### `hypersearch_dual.py` — Hyperparameter Search

Optuna TPE sampler with MedianPruner for efficient bear/bull model optimization.

**Search space:**

| Parameter | Range |
|---|---|
| seq_len | 12, 18, 24 |
| hidden_dim | 64, 96, 128 |
| num_layers | 1–2 |
| dropout | 0.05–0.45 (step 0.05) |
| learning_rate | 1e-4 – 3e-3 (log scale) |
| batch_size | 128, 256 |
| bull_threshold | 0.10–0.35 (step 0.01) |
| weight_decay | 0 – 1e-3 |
| scheduler | cosine, plateau, none |

**Objective function:** `composite_score = target_F1 × 0.5 + balanced_accuracy × 0.2 − catastrophic_rate × 0.3`

- **Target F1**: precision × recall for the target class (bear or bull)
- **Balanced accuracy**: mean per-class recall across bear/neutral/bull
- **Catastrophic rate**: bear↔bull confusion (the expensive mistakes in trading)

**Performance optimizations:**
- Pre-allocates full sequence tensors on GPU — eliminates per-batch CPU→GPU transfers
- `cudnn.benchmark = True` — cuDNN auto-tunes LSTM kernels for fixed input shapes
- Saves best-epoch predictions to skip redundant final validation pass
- `optimizer.zero_grad(set_to_none=True)` — deallocates gradients instead of zeroing

**Safety mechanisms:**
- MedianPruner reports `composite_score` each epoch (same metric as final objective), 20-epoch warmup, 15-trial startup matching TPE
- Early stopping with patience=15 epochs
- Hard timeout of 600 seconds per trial
- Rejects degenerate models: any class accuracy below 10% returns score 0
- Persistent SQLite storage: studies survive restarts, Bayesian memory accumulates

```bash
python hypersearch_dual.py --target bear --data training_data.csv
python hypersearch_dual.py --target bull --data stock_training_data.csv --prefix stock
```

#### `harvest_crypto_data.py` — Crypto Data Pipeline

Downloads 1 year of hourly OHLCV bars for 10 cryptocurrencies via yfinance, computes 40+ technical features, and saves to `training_data.csv`.

- BTC-USD used as benchmark for cross-asset features (BTC return, correlation, RSI divergence)
- Target: next-bar return as percentage
- Output sorted chronologically for proper time-series split during training

#### `harvest_stock_data.py` — Stock Data Pipeline

Downloads 1 year of hourly OHLCV bars for stocks via yfinance, computes stock-specific features, and saves to `stock_training_data.csv`.

- Stock universe loaded dynamically from `stock_universe.json` (editable via GUI Markets tab)
- SPY used as benchmark for relative strength features (ratio, correlation, RSI divergence)

#### `evolve.py` — 3-Day Retraining Orchestrator (Alternative)

Standalone 3-day model improvement cycle with model versioning and promotion gates.

1. Harvests fresh data
2. Runs alternating 50-trial batches for bear/bull (300 trials total per cycle)
3. Evaluates new model against currently deployed model
4. Promotes only if new model beats current (versioned copies in `models/`)

Useful as a standalone retrainer outside the pipeline. `run_pipeline.py` is the preferred orchestrator.

### ML & Inference

#### `model.py` — CryptoLSTM Architecture

3-class market regime classifier (bear / neutral / bull).

```
Input: (batch, seq_len, input_dim)
  → LSTM (hidden_dim, num_layers, dropout, batch_first)
  → Final hidden state
  → FC: hidden_dim → 64 → ReLU → num_classes
Output: logits → softmax probabilities
```

JIT-traceable via `torch.jit.trace` for ~30% faster inference on Jetson GPU.

#### `predict_now.py` — Live Inference Engine

Loads dual bear/bull models, fetches live bars, computes features, and returns a prediction score.

- `load_dual_models()` — loads both bear and bull models (falls back to single if one missing)
- `get_live_prediction()` — end-to-end: bars → features → scale → model → score
- Score = `(bull_prob − bear_prob) × bull_threshold` — positive = bullish, negative = bearish
- JIT trace attempted on load for faster repeated inference
- Handles prefixed models: `stock_bear_model.pth`, `stock_scaler_X.pkl`, etc.

### Technical Indicators

#### `indicators.py` — Numba JIT Indicators

40+ features computed with Numba `@njit` acceleration (~1.8x speedup over pure pandas).

**Indicators:**
- RSI (14), MACD (12/26/9), ATR (14), Bollinger Bands (20/2σ), Stochastic (14/3/3)
- OBV, Rate of Change (12), Rolling Percentile (100), Linear Regression Slope (5)

**Derived features:**
- Trend: SMA 20/50/100, EMA 12, price-to-SMA ratios
- Volume: volume ratio (20-bar), OBV, ROC
- Time: hour/day cyclical encoding (sin/cos)
- Advanced: ATR percentile, RSI divergence, volume-price confirmation
- Cross-asset: BTC return 1h + correlation + RSI divergence (crypto), SPY relative strength (stocks)

Two entry points:
- `compute_features(df, btc_close)` — crypto feature set
- `compute_stock_features(df, spy_close)` — stock feature set with SPY relative strength

Falls back to pure pandas if Numba is unavailable.

### Market Data & Utilities

#### `market_data.py` — Bar Fetching & ATR

Fetches hourly OHLCV bars from Alpaca or yfinance with ATR computation for adaptive stops.

- `fetch_bars_alpaca()` — crypto bars via Alpaca (6-day, 120 bars)
- `fetch_stock_bars_alpaca()` — stock bars via Alpaca
- `fetch_bars_yfinance()` — fallback yfinance download
- `get_live_atr()` — latest ATR value for stop-loss distance
- `flatten_yfinance_columns()` — handles yfinance MultiIndex columns

#### `trading_utils.py` — Shared Trading Utilities

Common code shared between crypto and stock loops.

- `get_api()` — constructs Alpaca REST client from `.env` credentials
- `get_model_mtime()` — file modification time for hot-reload detection
- `choose_inference_device()` — GPU with CPU fallback on OOM
- `cooldown_ok()` — 30-minute per-symbol trade cooldown
- `predict_symbol()` — wrapper running both bear/bull predictions

#### `order_utils.py` — Order Lifecycle & Risk Management

Complete order management: quoting, placement, fill polling, circuit breaker.

- `get_quote()` / `get_crypto_quote()` / `get_stock_quote()` — bid/ask/spread/midpoint
- `place_limit_order()` / `place_stock_limit_order()` — limit orders with notional-to-qty conversion
- `manage_order_lifecycle()` — polls fill status, cancels stale orders, optional market fallback
- `reconstruct_positions()` — rebuilds position state from API (survive crashes)
- `check_circuit_breaker()` — 5% daily equity drawdown detection
- `emergency_flatten()` — market-sells everything immediately

### Sentiment

#### `sentiment.py` — Sentiment Analysis & Trade Gating

Rule-based NLP sentiment scoring with Fear & Greed Index integration.

**Data sources:**
- [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) — free API, 5-min cache
- [Finnhub](https://finnhub.io/) — symbol-specific + market news (optional, 60 calls/min free tier)
- Full article scraping via BeautifulSoup (30-min cache)

**Scoring:**
- 150+ positive/negative keywords with phrase matching and word boundaries
- Negation-aware: flips sentiment within 3 words ("not bullish" → bearish)
- Article weighting: headline 25% + summary 25% + full text 50%
- Tanh smoothing scaled by √(word_count)

**Trade multiplier (`sentiment_gate`):**

| Condition | Multiplier |
|---|---|
| Extreme fear (FnG ≤ 10) | 0.3x |
| Fear (FnG ≤ 25) | 0.5x |
| Cautious (FnG ≤ 40) | 0.8x |
| Normal (40–75) | 1.0x |
| Greed (FnG ≥ 75) | 0.85x |
| Extreme greed (FnG ≥ 90) | 0.6x |
| Symbol news ≤ −0.3 | 0.0x (block) |
| Symbol news ≥ 0.3 | 1.2x (boost) |

Final multiplier clamped to [0.0, 1.5].

### Monitoring & Operations

#### `gui.py` — PySide6 Dashboard

Desktop monitoring app with live updates (2-second polling of `pipeline_status.json`).

**Tabs:**
- **Positions** — open positions, P&L, exposure, recent fills
- **Performance** — equity curve, daily P&L, drawdown
- **News** — Fear & Greed Index, Finnhub headlines (My Universe / All News / Global Macro filters)
- **Markets** — combined stocks + crypto universe with heatmap, price chart (1Y/3M/1M/1W/1D zoom), metrics table with live model predictions (bear/bull/score/signal from both bots), add/remove symbols
- **Models** — model scores (C-Bear, C-Bull, S-Bear, S-Bull), trial progress, pipeline phase, next retrain time
- **Hardware** — GPU temp, RAM usage, CUDA status
- **Logs** — tailing pipeline, crypto bot, and stock bot log files

**Themes:** Bubblegum Goth, Batman, Joker, Harley Quinn, Dark, Space, Money, Two-Face, Black Metal

#### `watchdog.py` — Process Supervisor (Alternative)

Standalone process monitor for trading bots outside the pipeline.

- Checks bot health every 30 seconds
- Auto-restarts crashed processes (max 3 restarts per hour)
- Emergency liquidation via `emergency_flatten()` on repeated failures
- Use `run_pipeline.py` for the integrated version with auto-restart built in

#### `hw_monitor.py` — Hardware Monitor

Jetson Orin Nano hardware telemetry (no sudo required).

- `get_gpu_temp()` — GPU temperature from tegrastats / thermal zones
- `get_ram_usage()` — used/total MB from `/proc/meminfo`
- `is_gpu_available()` — tests CUDA allocation (detects OOM)
- `wait_for_cool_gpu()` — blocks until GPU drops below threshold

#### `connection_test.py` — API Connectivity Check

Verifies Alpaca API credentials, reports account equity, PDT status, and trading mode.

```bash
python connection_test.py
```

### Testing

#### `test_sentiment.py` — Sentiment Test Suite

1,035 test cases validating `_score_text()` accuracy.

- 160+ hand-written edge cases (earnings, macro, analyst, negation, mixed signals)
- 900+ template-generated permutations (stocks × verbs × reasons × patterns)
- Validates text preprocessing (`_validate_text`) and article scoring (`_score_articles`)
- Target: 99%+ accuracy

---

## Quick Start

### 1. Install
```bash
pip install alpaca-trade-api python-dotenv torch numpy pandas numba \
    yfinance finnhub-python optuna joblib scikit-learn PySide6 pyqtgraph
```

### 2. Configure
Create a `.env` file:
```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_finnhub_key
```

### 3. Verify connectivity
```bash
python connection_test.py
```

### 4. Run the full system
```bash
# Full pipeline: harvest → train → trade → weekly retrain
python run_pipeline.py

# Or skip training if models already exist
python run_pipeline.py --bot-only

# GUI dashboard (separate terminal)
python gui.py
```

### 5. Manual steps (if not using pipeline)
```bash
# Harvest training data
python harvest_crypto_data.py    # → training_data.csv
python harvest_stock_data.py     # → stock_training_data.csv

# Train models
python hypersearch_dual.py --target bear --data training_data.csv
python hypersearch_dual.py --target bull --data training_data.csv
python hypersearch_dual.py --target bear --data stock_training_data.csv --prefix stock
python hypersearch_dual.py --target bull --data stock_training_data.csv --prefix stock

# Run bots individually
python crypto_loop.py
python stock_loop.py
```

## Output Files

| File | Source | Purpose |
|---|---|---|
| `training_data.csv` | harvest_crypto_data.py | Crypto training dataset |
| `stock_training_data.csv` | harvest_stock_data.py | Stock training dataset |
| `bear_model.pth` | hypersearch_dual.py | Crypto bear model weights |
| `bull_model.pth` | hypersearch_dual.py | Crypto bull model weights |
| `stock_bear_model.pth` | hypersearch_dual.py | Stock bear model weights |
| `stock_bull_model.pth` | hypersearch_dual.py | Stock bull model weights |
| `*_config.pkl` | hypersearch_dual.py | Model hyperparameters |
| `*scaler_X.pkl` | hypersearch_dual.py | Feature scaler (MinMaxScaler) |
| `*_study.db` | hypersearch_dual.py | Optuna SQLite study (Bayesian memory) |
| `stock_universe.json` | stock_config.py / GUI | Market universe (stocks + crypto symbols) |
| `stock_predictions.json` | stock_loop.py | Live stock model predictions for GUI |
| `crypto_predictions.json` | crypto_loop.py | Live crypto model predictions for GUI |
| `pipeline_status.json` | run_pipeline.py | Live pipeline state for GUI |
| `pipeline_output.log` | run_pipeline.py | Pipeline log |
| `crypto_bot_output.log` | run_pipeline.py | Crypto bot log |
| `stock_bot_output.log` | run_pipeline.py | Stock bot log |
