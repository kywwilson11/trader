# Trader

Autonomous paper trading bot for stocks and crypto, powered by dual LSTM models with Numba-accelerated technical indicators and sentiment-gated risk management.

Runs on an NVIDIA Jetson Orin Nano. Trades via the [Alpaca](https://alpaca.markets/) paper trading API.

## Architecture

```
watchdog.sh
├── evolve.py          3-day retraining cycle (harvest → hypersearch → promote)
├── stock_loop.py      Market-hours stock trading (top 10 of ~50 stocks)
└── crypto_loop.py     24/7 crypto trading (10 symbols)
```

**Inference stack**: `predict_now.py` → `model.py` (CryptoLSTM) → `indicators.py` (Numba JIT)

**Risk stack**: `order_utils.py` (order lifecycle) → `sentiment.py` (F&G + Finnhub gating)

## Features

- **Dual bear/bull LSTM models** — separate models optimized per direction, hot-reloadable without downtime
- **ATR-based adaptive risk** — stop-loss, trailing stops, and take-profit scaled to current volatility
- **Sentiment gating** — Fear & Greed Index + Finnhub news scores modulate position sizing (0x–1.5x)
- **Confidence-based sizing** — trade notional scaled by prediction strength
- **Numba JIT indicators** — RSI, MACD, ATR, Bollinger Bands, Stochastic, OBV (~1.8x speedup)
- **Cross-asset features** — BTC prices for crypto correlations, SPY for stock relative strength
- **Circuit breaker** — auto-flattens all positions on 5% daily drawdown
- **Persistent Optuna studies** — Bayesian hyperparameter search with SQLite memory across cycles
- **Position reconstruction** — survives crashes by syncing state from Alpaca API on restart
- **PySide6 dashboard** — live positions, P&L, orders, news (filterable), model status, hardware gauges, 7 themes

## Trading Loops

### Crypto (`crypto_loop.py`)
- **Schedule**: 24/7, 30-second loop
- **Universe**: BTC, ETH, XRP, SOL, DOGE, LINK, AVAX, DOT, LTC, BCH
- **Sizing**: $250/trade, confidence-scaled
- **Risk**: ATR trailing stops, 30-min per-symbol cooldown, sentiment gate

### Stocks (`stock_loop.py`)
- **Schedule**: Market hours only (9:30 AM – 4:00 PM ET), auto-flatten at 3:50 PM
- **Universe**: ~50 stocks (tech, commodities, space, quantum) → top 10 by bull signal
- **Sizing**: $2,500/position, $25k max exposure, confidence-scaled
- **Risk**: Bracket orders (stop + TP), trailing stop upgrade at 1.5% profit, sentiment gate

## Model Training

### Pipeline (`evolve.py`)
Every 3 days:
1. **Harvest** — download 1 year of hourly bars (`harvest_crypto_data.py`, `harvest_stock_data.py`)
2. **Hypersearch** — Optuna TPE sampler, alternating 50-trial batches for bear/bull models
3. **Evaluate** — validate on held-out 20% split
4. **Promote** — copy improved models to `models/` with version numbers (keeps last 5)

### Hyperparameters (`hypersearch_dual.py`)
Searched: seq_len, hidden_dim, num_layers, dropout, LR, batch_size, bull_threshold, weight_decay, scheduler. Median pruning with 8-epoch warmup. 300s per-trial timeout.

## GUI Dashboard (`gui.py`)

PySide6 desktop app with tabs:
- **Positions** — live P&L, exposure, open orders, recent fills
- **Performance** — equity curve, daily P&L, max drawdown
- **News** — Fear & Greed Index, Finnhub headlines (My Universe / All News / Global Macro filters)
- **Models** — model age, config, evolve.py progress bar
- **Hardware** — GPU temp, RAM usage, CUDA availability
- **Logs** — tailing evolve, hypersearch, pipeline, and monitor logs

Themes: Bubblegum Goth, Batman, Joker, Harley Quinn, Dark, Space, Money

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU (optional, falls back to CPU)
- Alpaca paper trading account
- Finnhub API key (optional, for news sentiment)

### Install
```bash
pip install alpaca-trade-api python-dotenv torch numpy pandas numba \
    yfinance finnhub-python optuna PySide6 pyqtgraph
```

### Configure
Create a `.env` file:
```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_finnhub_key
```

### Run
```bash
# Full system (trading + retraining)
./watchdog.sh

# GUI dashboard
python gui.py

# Individual loops
python crypto_loop.py
python stock_loop.py

# One-off hypersearch
python hypersearch_dual.py --target bull --trials 100
```

## Monitoring

- `watchdog.py` — process supervisor, restarts crashed loops, emergency flatten on repeated failures
- `hw_monitor.py` — GPU temp, RAM, thermal throttling (pauses at 75°C)
- `connection_test.py` — Alpaca connectivity and PDT status check
- `monitor_search.sh` — hypersearch health (stall detection, trial progress)

## File Overview

| File | Purpose |
|---|---|
| `crypto_loop.py` | 24/7 crypto trading loop |
| `stock_loop.py` | Market-hours stock trading loop |
| `indicators.py` | Numba JIT technical indicators |
| `predict_now.py` | Live inference with torch.jit.trace |
| `model.py` | CryptoLSTM architecture |
| `order_utils.py` | Order placement and lifecycle |
| `sentiment.py` | Fear & Greed + Finnhub sentiment |
| `gui.py` | PySide6 dashboard |
| `evolve.py` | 3-day retraining orchestrator |
| `hypersearch_dual.py` | Optuna hyperparameter search |
| `harvest_crypto_data.py` | Crypto training data pipeline |
| `harvest_stock_data.py` | Stock training data pipeline |
| `watchdog.py` | Process supervisor |
| `hw_monitor.py` | Hardware monitoring |
| `connection_test.py` | API connectivity test |
