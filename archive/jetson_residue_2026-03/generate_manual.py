#!/usr/bin/env python3
"""Generate algorithm-focused PDF manual for the Trader system."""

from fpdf import FPDF
import datetime


class Manual(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "Trader System Manual", align="L")
            self.cell(0, 6, f"v3.0 -- {datetime.date.today()}", align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _clean(text):
    """Replace unicode chars with latin-1 safe equivalents."""
    reps = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
        "\u2190": "<-", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x",
        "\u2248": "~=", "\u2260": "!=", "\u00b2": "^2", "\u00b3": "^3",
        "\u03c3": "sigma", "\u03b1": "alpha", "\u03b2": "beta",
        "\u03c9": "omega", "\u2605": "*", "\u2022": "-", "\u00b7": "-",
        "\u2212": "-", "\u00b1": "+/-", "\u221a": "sqrt",
    }
    for old, new in reps.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def s(pdf, text, size=10):
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(1)


def sb(pdf, text, size=10):
    """Bold paragraph."""
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(1)


def chapter(pdf, num, title):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, _clean(f"Chapter {num}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)


def section(pdf, title):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def subsection(pdf, title):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def code(pdf, text):
    pdf.set_font("Courier", "", 7.5)
    pdf.set_fill_color(240, 240, 240)
    for line in _clean(text).split("\n"):
        pdf.cell(0, 3.8, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)


def bullet(pdf, text, size=10):
    pdf.set_font("Helvetica", "", size)
    pdf.cell(5, 5, "-")
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(0.5)


def table_row(pdf, cells, widths, bold=False):
    style = "B" if bold else ""
    pdf.set_font("Helvetica", style, 7.5)
    for cell, w in zip(cells, widths):
        pdf.cell(w, 4.5, _clean(str(cell)), border=1)
    pdf.ln(4.5)


# ============================================================
# PAGES
# ============================================================

def title_page(pdf):
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 32)
    pdf.cell(0, 15, "Trader System Manual", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Alpaca Paper Trading Bot", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "LSTM + LightGBM Ensemble with 12-Layer Risk Management",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    lines = [
        "Platform: NVIDIA Jetson Orin Nano Super (ARM64, 8GB)",
        "JetPack 6.2.1 | PyTorch 2.8.0 | CUDA 12.6 | cuDNN 9.3",
        "Broker: Alpaca Markets (Paper Trading, $100k equity)",
        "LLM: Google Gemini (tiered routing: Pro/Flash/Lite)",
        f"Generated: {datetime.date.today()}",
    ]
    for line in lines:
        pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def build_toc(pdf):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    toc = [
        ("1", "System Overview"),
        ("2", "Prediction Engine -- LSTM + LightGBM"),
        ("3", "Feature Engineering"),
        ("4", "Training Pipeline & Optuna Search"),
        ("5", "Trading Algorithm -- Main Loop"),
        ("6", "Entry Gates (Buy Logic)"),
        ("7", "Exit Logic (Sell, Short, Cover)"),
        ("8", "Position Sizing -- 12-Layer Risk Stack"),
        ("9", "Stop-Loss & Risk Controls"),
        ("10", "Horizon-Adaptive Scaling"),
        ("11", "LLM Integration"),
        ("12", "Sentiment Analysis"),
        ("13", "GARCH Volatility Forecasting"),
        ("14", "Macro Regime Detection"),
        ("15", "Correlation-Aware Sizing"),
        ("16", "HMM Regime Detection"),
        ("17", "Data Pipeline"),
        ("18", "Research Foundations"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"  {num}.  {title}", new_x="LMARGIN", new_y="NEXT")


# ---- CHAPTER 1 ----
def ch1(pdf):
    chapter(pdf, 1, "System Overview")

    section(pdf, "1.1 Purpose")
    s(pdf, "This system is an automated paper trading bot for stocks and cryptocurrencies. "
      "It uses an LSTM neural network with multi-head self-attention, ensembled with LightGBM "
      "gradient-boosted trees, to predict forward returns on hourly bars. Positions are sized "
      "through a 12-layer risk management stack drawing on portfolio theory, volatility modeling, "
      "behavioral economics, and regime detection. The system trades autonomously on Alpaca's "
      "paper trading platform with $100k equity.")
    s(pdf, "The bot runs two independent trading loops: CryptoLoop (24/7, all crypto pairs) and "
      "StockLoop (market hours only, top-N by signal strength). Both inherit from a shared "
      "BaseTradingLoop via the Template Method pattern, ensuring identical risk management "
      "logic across asset classes.")

    section(pdf, "1.2 Hardware Platform")
    s(pdf, "The system runs on an NVIDIA Jetson Orin Nano Super, chosen for its GPU capability "
      "at low power consumption (15-25W). Key specifications:")
    bullet(pdf, "CPU: 6-core ARMv8 Cortex-A78AE @ 1728MHz")
    bullet(pdf, "GPU: Ampere SM_87 (compute capability 8.7), 8 SMs, 1024 CUDA cores")
    bullet(pdf, "Memory: 8GB LPDDR5 unified (shared CPU/GPU)")
    bullet(pdf, "Storage: 465GB NVMe (primary, 381GB free), 117GB SD card (secondary)")
    bullet(pdf, "OS: Ubuntu 22.04, JetPack 6.2.1, Linux 5.15.148-tegra")
    bullet(pdf, "Power mode: MAXN_SUPER (maximum performance)")
    s(pdf, "The unified memory architecture means GPU training and CPU inference compete for the "
      "same 8GB. This drives several design decisions: inference always runs on CPU to preserve "
      "GPU for training, a GPU lock prevents concurrent GPU access, training caps CUDA allocation "
      "at 40% (~3GB), and OOM retry logic halves batch size on failure.")

    section(pdf, "1.3 Software Stack")
    bullet(pdf, "Jetson env (Python 3.10): PyTorch 2.8.0, CUDA 12.6, cuDNN 9.3, Numba 0.63.1, "
           "LightGBM, Optuna, arch (GARCH), hmmlearn, alpaca-trade-api 3.2.0, fredapi")
    bullet(pdf, "Base env (Python 3.12): PySide6 for GUI dashboard only (PySide6 not available "
           "for ARM64 Python 3.10)")
    s(pdf, "PyTorch 2.9.1 exists but is BROKEN (missing libcudss.so.0) -- do NOT upgrade. "
      "The LD_LIBRARY_PATH must include nvidia/cusparselt/lib for libcusparseLt.so.0.")

    section(pdf, "1.4 Broker Integration")
    s(pdf, "The system connects to Alpaca Markets paper trading via their REST API. Account "
      "equity is $100k. API credentials (ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL) "
      "are loaded from .env via python-dotenv. The system supports both stock and crypto "
      "trading through Alpaca's unified API, with bracket orders for stocks and limit orders "
      "for crypto.")

    section(pdf, "1.5 LLM Integration")
    s(pdf, "Google Gemini provides supplementary analysis via a tiered routing system. The "
      "system auto-detects free vs paid tier from API rate limit headers and routes requests "
      "to the appropriate model (Pro for high-value analysis, Flash for mid-range, Lite when "
      "daily budget is nearly exhausted). Daily budget cap: $1.00. The LLM serves as a "
      "supplementary gate in the trading pipeline, not the primary signal.")


# ---- CHAPTER 2 ----
def ch2(pdf):
    chapter(pdf, 2, "Prediction Engine -- LSTM + LightGBM")

    section(pdf, "2.1 RegressionLSTM Architecture (model_v2.py)")
    s(pdf, "The neural network predicts continuous return percentages (regression, not "
      "classification). The architecture combines LSTM temporal modeling with multi-head "
      "self-attention for adaptive timestep weighting:")
    code(pdf, """Input: (batch, seq_len, input_dim)    # e.g., (32, 20, 22)

  -> LSTM(input_dim, hidden_dim, num_layers, dropout)
     - Unidirectional (causal -- no future leakage)
     - Dropout between layers if num_layers > 1
     - hidden_dim: 64 to 384 (tuned by Optuna)
     - num_layers: 1 or 2 (3+ overfits on this data)

  -> MultiHeadAttention(hidden_dim, n_heads, dropout, batch_first=True)
     - Self-attention: Q = K = V = lstm_output
     - n_heads: 2 to 4 (must evenly divide hidden_dim)
     - Learns which timesteps in the sequence are most informative
     - Allows the model to attend to distant bars (e.g., a spike 15 bars ago)

  -> Residual Connection + LayerNorm
     - output = LayerNorm(attn_out + lstm_out)
     - Stabilizes training, prevents attention from degrading LSTM signal
     - Residual ensures LSTM information is preserved even if attention
       learns nothing useful

  -> Mean Pooling (dim=1)
     - Average over the sequence dimension
     - Collapses (batch, seq_len, hidden_dim) -> (batch, hidden_dim)

  -> FC Head
     - Linear(hidden_dim, 64) -> ReLU -> Dropout -> Linear(64, 1)
     - Final scalar output: predicted return percentage

Output: predicted_return (%)""")

    section(pdf, "2.2 Hyperparameter Ranges")
    s(pdf, "All architecture hyperparameters are tuned via Optuna's TPE sampler:")
    widths = [42, 55, 93]
    table_row(pdf, ["Parameter", "Range", "Notes"], widths, bold=True)
    params = [
        ("hidden_dim", "[64, 384] step 32", "LSTM hidden state size. Adaptive boundaries."),
        ("num_layers", "[1, 2]", "LSTM depth. 3+ causes overfitting on hourly data."),
        ("n_heads", "[2, 4]", "Attention heads. Must evenly divide hidden_dim."),
        ("dropout", "[0.10, 0.40] step 0.05", "Applied in LSTM, attention, and FC head."),
        ("seq_len", "[8, 40] step 2", "Input window length in bars. Adaptive boundaries."),
    ]
    for p in params:
        table_row(pdf, p, widths)

    section(pdf, "2.3 LightGBM Stacking (model_lgb.py)")
    s(pdf, "LightGBM is trained on flattened LSTM input sequences. The (seq_len x n_features) "
      "tensor is reshaped into a 1D vector with lag-named columns (e.g., RSI, RSI_lag1, "
      "RSI_lag2, ...). This tree-based model captures non-linear feature interactions and "
      "threshold effects that the LSTM may miss. Literature shows 10-15% Sharpe improvement "
      "from LSTM+tree stacking.")
    code(pdf, """LightGBM hyperparameters:
  num_leaves=63, learning_rate=0.05, feature_fraction=0.7
  bagging_fraction=0.8, bagging_freq=5
  500 boosting rounds, early stopping after 20 rounds without improvement

Training data: Same walk-forward folds as LSTM (no data leakage)
Features: Flattened sequence (seq_len * n_features columns)""")

    section(pdf, "2.4 Ensemble Weighting")
    s(pdf, "The final prediction is a weighted average of both models:")
    code(pdf, """predicted_return = 0.6 * lstm_pred + 0.4 * lgb_pred

Rationale:
  - LSTM captures temporal dynamics and sequential patterns
  - LightGBM captures static feature interactions and thresholds
  - 60/40 weighting reflects LSTM's stronger performance on temporal data
  - If LGB model is unavailable: falls back to LSTM-only (graceful degradation)""")

    section(pdf, "2.5 Loss Function")
    s(pdf, "Training uses Huber loss with return-weighted samples. Huber loss is less sensitive "
      "to outliers than MSE (important in financial data with fat tails), while still providing "
      "gradient signal for small errors unlike MAE:")
    code(pdf, """raw_loss = HuberLoss(delta=huber_delta, reduction='none')
weights = clamp(|actual_return| + 1.0, max=50.0)
loss = (raw_loss * weights).mean()

huber_delta: tuned by Optuna in range [0.5, 2.0]

Effect of return-weighting:
  A 5% actual return gets weight 6.0 (5.0 + 1.0)
  A 0.1% actual return gets weight 1.1 (0.1 + 1.0)
  -> Model learns to predict large moves accurately (where profit is)
  -> Small noise near zero is de-emphasized""")

    section(pdf, "2.6 Inference Pipeline (predict_now.py)")
    s(pdf, "Inference runs on CPU (even when GPU is available) to preserve GPU memory for "
      "training. JIT tracing provides 10-20% speedup on CPU:")
    code(pdf, """For each symbol (parallel, 5 threads via ThreadPoolExecutor):
  1. Fetch recent bars (120 crypto / 200 stock hourly)
     - Primary: Alpaca API
     - Fallback: yfinance
  2. Compute technical features (indicators.py, 40+ indicators)
  3. Inject Daily_Sentiment (if model was trained with it)
     - Lazy import sentiment_history.get_live_daily_sentiment()
     - Fallback to 0.0 if unavailable
  4. Scale features with RobustScaler (fitted on training data)
  5. Extract last seq_len bars as input tensor
  6. LSTM inference (JIT-traced model, CPU, <10ms per symbol)
  7. LightGBM prediction on flattened features (if model available)
  8. Weighted ensemble -> predicted_return %
  9. Optional: return snapshot dict of latest indicator values""")


# ---- CHAPTER 3 ----
def ch3(pdf):
    chapter(pdf, 3, "Feature Engineering")

    section(pdf, "3.1 Implementation Stack")
    s(pdf, "Features are computed through a three-tier implementation stack, selected "
      "automatically at import time for maximum performance on ARM64:")
    bullet(pdf, "Tier 1 -- C extension (indicators_c.so): ARM64 NEON-optimized native binary "
           "compiled via Numba AOT. Fastest path, used when available.")
    bullet(pdf, "Tier 2 -- Numba JIT (@njit): Compiled on first call, 3-10x speedup over "
           "pure Python. Automatic fallback if C extension unavailable.")
    bullet(pdf, "Tier 3 -- Pure NumPy/Pandas: Always-available fallback. No compilation required.")

    section(pdf, "3.2 Momentum Indicators")
    bullet(pdf, "RSI(14): Relative Strength Index. Bounded [0, 100]. Measures speed and change "
           "of price movements. >70 overbought, <30 oversold.")
    bullet(pdf, "MACD(12,26,9): Moving Average Convergence Divergence. Three components: "
           "MACD line (fast EMA - slow EMA), signal line (9-period EMA of MACD), histogram "
           "(MACD - signal). Captures momentum shifts and trend direction.")
    bullet(pdf, "ROC(12): Rate of Change over 12 periods. Simple percentage return. "
           "Stationary by construction.")

    section(pdf, "3.3 Volatility Indicators")
    bullet(pdf, "ATR(14): Average True Range. Measures volatility as the average of true ranges "
           "(max of high-low, |high-prev_close|, |low-prev_close|). Used for stop-loss placement "
           "and position sizing.")
    bullet(pdf, "Bollinger Bands(20,2): Upper/lower bands at +/-2 standard deviations from "
           "20-period SMA. Features: BBP (percent B, position within bands, bounded 0-1) and "
           "BBB (bandwidth, measures volatility expansion/contraction).")

    section(pdf, "3.4 Oscillators")
    bullet(pdf, "Stochastic(14,3,3): %K and %D lines. Bounded [0, 100]. Compares closing price "
           "to high-low range over 14 periods. %K is raw, %D is 3-period SMA of %K. "
           "Overbought >80, oversold <20.")

    section(pdf, "3.5 Volume Indicators")
    bullet(pdf, "OBV: On Balance Volume. Cumulative volume where up days add volume, down days "
           "subtract. Confirms price trends (rising OBV + rising price = strong trend).")
    bullet(pdf, "Volume_Ratio: Current volume / rolling average volume. Stationary by construction. "
           "Spikes indicate unusual activity (earnings, news).")

    section(pdf, "3.6 Trend Indicators")
    bullet(pdf, "SMA_20, SMA_100: Simple Moving Averages. Used to compute ratios.")
    bullet(pdf, "Price_SMA20_Ratio, Price_SMA100_Ratio: Price divided by SMA. Stationary, "
           "centered around 1.0. Values >1 = above trend, <1 = below trend.")

    section(pdf, "3.7 Advanced Features")
    subsection(pdf, "Hurst Exponent (R/S Analysis)")
    s(pdf, "Measures persistence in the price series using Rescaled Range analysis with a "
      "rolling window of 100 bars. This is a key regime indicator that determines whether "
      "momentum strategies or mean-reversion strategies are appropriate:")
    code(pdf, """H > 0.5: Trending (persistent momentum -- favor momentum signals)
H ~ 0.5: Random walk (low predictability -- reduce confidence)
H < 0.5: Mean-reverting (anti-persistent -- momentum signals unreliable)

Implementation: _hurst_rs(arr, window=100)
  1. For each rolling window: compute mean, cumulative deviation
  2. R = max(cumdev) - min(cumdev), S = std(returns)
  3. H = log(R/S) / log(window)
  Bounded [0, 1], stationary""")

    subsection(pdf, "Calendar Features")
    s(pdf, "Cyclical time encoding captures intraday, weekly, and seasonal patterns without "
      "creating artificial discontinuities (e.g., hour 23 is close to hour 0):")
    code(pdf, """Hour_sin  = sin(2 * pi * hour / 24)      Hour_cos  = cos(2 * pi * hour / 24)
Day_sin   = sin(2 * pi * dayofweek / 7) Day_cos   = cos(2 * pi * dayofweek / 7)
Month_sin = sin(2 * pi * month / 12)    Month_cos = cos(2 * pi * month / 12)

Turn_of_Month: binary flag for last 2 and first 2 trading days of month.
  Captures the well-documented Turn-of-Month effect (Ariel 1987, Lakonishok
  & Smidt 1988) where returns are disproportionately concentrated around
  month boundaries due to institutional cash flows.""")

    section(pdf, "3.8 Cross-Asset Features")
    code(pdf, """Crypto-only (BTC as market leader):
  BTC_Return_1h   -- Bitcoin hourly return (altcoins lag BTC by 1-4 hours)
  BTC_SMA_Ratio   -- BTC price vs SMA (BTC trend state)
  BTC_RSI         -- BTC momentum (altcoin beta to BTC is ~1.3)

Stock-only:
  RS_vs_SPY       -- Relative strength vs S&P 500 (sector rotation signal)
  Price_VWAP_Ratio -- Intraday strength (price / VWAP, >1 = strong)
  Gap_Pct         -- Overnight gap: (open - prev_close) / prev_close
  ATR_Pct         -- Normalized volatility: ATR / close * 100""")

    section(pdf, "3.9 Feature Presets")
    s(pdf, "Four presets control which features the model trains on. The preset is selected "
      "in indicator_config.json and can be changed via the GUI Settings tab:")
    widths = [35, 18, 137]
    table_row(pdf, ["Preset", "Count", "Description"], widths, bold=True)
    presets = [
        ("minimal", "18", "Core oscillators + returns only. Fast training, lower accuracy."),
        ("stationary", "20", "All bounded/stationary features. No trending data (SMA levels)."),
        ("standard", "30+", "Stationary + trend ratios + volume. Recommended starting point."),
        ("full", "all", "Every computed feature including raw levels. Risk of overfitting."),
    ]
    for p in presets:
        table_row(pdf, p, widths)
    s(pdf, "The 'stationary' preset is recommended for regression models. All features are "
      "bounded or stationary by construction, preventing distribution drift that degrades "
      "model accuracy over time.")


# ---- CHAPTER 4 ----
def ch4(pdf):
    chapter(pdf, 4, "Training Pipeline & Optuna Search")

    section(pdf, "4.1 Pipeline Orchestrator (run_pipeline.py)")
    s(pdf, "Training is managed by a multi-phase pipeline orchestrator that handles data "
      "harvesting, hyperparameter search, model selection, and bot lifecycle:")
    code(pdf, """Phase A: Initial Training
  1. Harvest: Alpaca -> yfinance -> CryptoCompare (3-source fallback)
  2. Optuna search: 200 trials per model (crypto + stock)
  3. Save best model: model_v2.pth, config_v2.pkl, scaler_v2.pkl

Phase B: Trading (continuous)
  - Start crypto + stock bots as subprocesses
  - Monitor & auto-restart on crash
  - Bots hot-reload models when .pth file mtime changes

Phase C: Weekly Retrain (Saturday 2 AM or manual via GUI)
  - Stop bots (free GPU memory)
  - Adaptive mode: refine (70 trials) or explore (120 trials)
  - Incremental data harvest (only new bars since last run)
  - Search with potentially expanded hyperparameter space
  - Restart bots with new models (if score improved)""")

    section(pdf, "4.2 Optuna Study Configuration")
    code(pdf, """study = optuna.create_study(
  storage='sqlite:///{prefix}v2_study.db',  # Resumable across runs
  direction='maximize',                     # Maximize risk-adjusted Sharpe
  sampler=TPESampler(n_startup_trials=60),  # Bayesian optimization
  pruner=MedianPruner(n_startup_trials=60, n_warmup_steps=12)
)""")
    s(pdf, "TPE (Tree-structured Parzen Estimator) is a Bayesian optimization algorithm that "
      "models the relationship between hyperparameters and performance using kernel density "
      "estimation. After 60 random startup trials to build the density model, it focuses "
      "exploration on promising regions of the search space. MedianPruner terminates "
      "unpromising trials early -- those performing below the running median of completed "
      "trials -- after 12 warmup epochs, saving significant GPU time.")

    section(pdf, "4.3 Hyperparameter Search Space")
    widths = [42, 55, 93]
    table_row(pdf, ["Parameter", "Range", "Notes"], widths, bold=True)
    params = [
        ("forward_bars", "[12,18,24,32,48]", "Prediction horizon (bars ahead). Expandable to 96."),
        ("seq_len", "[8, 40] step 2", "LSTM input window. Adaptive low/high boundaries."),
        ("hidden_dim", "[64, 384] step 32", "LSTM hidden size. Adaptive boundaries."),
        ("num_layers", "[1, 2]", "LSTM depth. 3+ overfits on hourly data."),
        ("n_heads", "[2, 4]", "Attention heads. Must divide hidden_dim evenly."),
        ("dropout", "[0.10, 0.40] step 0.05", "Regularization strength."),
        ("learning_rate", "[5e-4, 3e-3] log", "Adam optimizer LR. Log-uniform sampling."),
        ("batch_size", "[512, 1024, 2048]", "Larger = faster but OOM risk on Jetson 8GB."),
        ("weight_decay", "[1e-5, 5e-4] log", "L2 regularization for Adam."),
        ("huber_delta", "[0.5, 2.0] step 0.1", "Huber loss robustness threshold."),
        ("trade_threshold", "[0.05, 1.0] step 0.01", "Minimum predicted return to trade."),
        ("scheduler", "[cosine, plateau]", "LR annealing strategy."),
    ]
    for p in params:
        table_row(pdf, p, widths)

    section(pdf, "4.4 Walk-Forward Cross-Validation (3-Fold)")
    s(pdf, "Expanding window strategy with embargo gap to prevent look-ahead bias. The embargo "
      "gap equals seq_len bars, ensuring no overlapping sequences between training and validation:")
    code(pdf, """Fold 0: train on first 60%  | embargo (seq_len bars) | val on next 14%
Fold 1: train on first 73%  | embargo (seq_len bars) | val on next 14%
Fold 2: train on first 86%  | embargo (seq_len bars) | val on next 14%

Key properties:
  - Expanding window: each fold sees MORE training data (mimics production)
  - Embargo gap = seq_len * EMBARGO_MULTIPLIER (default 1x) bars
    Prevents target leakage from overlapping sequences
  - Per-ticker chronological splits: within each ticker, data is ordered
    by timestamp. No future data leaks to the training set.
  - Validation set always follows training set in time (temporal ordering)""")

    section(pdf, "4.5 Training Loop Internals")
    code(pdf, """For each fold (up to MAX_EPOCHS=60):
  Loss: Return-weighted Huber loss (see Chapter 2)
  Precision: FP16 mixed precision (torch.amp.GradScaler on CUDA)
  Gradient clipping: max_norm=1.0 (prevents gradient explosion)
  Early stopping: 10 epochs patience on validation loss
  OOM retry: halves batch size (min 128), clears CUDA cache, retries
  Trial timeout: 900s per trial (15 minutes max)

GPU memory management:
  torch.cuda.set_per_process_memory_fraction(0.40) = ~3.2GB on 8GB Jetson
  This leaves headroom for system processes and prevents OOM kills""")

    section(pdf, "4.6 Sharpe Ratio Objective")
    s(pdf, "The optimization objective is a risk-adjusted Sharpe ratio computed from simulated "
      "trading on validation data:")
    code(pdf, """Per-fold Sharpe calculation:
  signal = +1 if pred > threshold, -1 if pred < -threshold, 0 otherwise
  trade_return = signal * actual_return - 5 bps (transaction cost per trade)
  annualized_sharpe = (mean(trade_returns) / std(trade_returns)) *
                      sqrt(annualized_trade_frequency)

Fold aggregation (risk-adjusted):
  score = mean(fold_sharpes) - 0.5 * std(fold_sharpes)
  -> Penalizes inconsistency across folds
  -> A model with Sharpe [2.0, 0.5, 1.0] scores lower than [1.2, 1.1, 1.0]

Regime-aware penalty:
  Regime detection: rolling 50-bar return > 2% = bull, < -2% = bear, else sideways
  If ANY regime (bull/bear/sideways) has Sharpe < -0.5: score *= 0.7
  -> 30% score reduction ensures model works in ALL market conditions
  -> Prevents models that only profit in bull markets""")

    section(pdf, "4.7 Adaptive Search Modes")
    s(pdf, "The search space and trial count adapt automatically based on previous results:")
    code(pdf, """Modes and trial counts:
  INITIAL:  200 trials (first run, widest exploration)
  REFINE:    70 trials (exploit local optimum, narrow ranges around best)
  EXPLORE:  120 trials (wider search after edge detection or stagnation)

Transition rules:
  -> EXPLORE: if ANY hyperparameter hits search space boundary
              OR 3+ cycles without >5% score improvement
  -> REFINE:  after explore completes (toggle back to exploitation)

State persisted to adaptive_state_{asset_type}.json:
  best_score, best_params, search_space, mode,
  cycles_without_improvement, expansion_history""")

    section(pdf, "4.8 Edge Detection & Space Expansion")
    s(pdf, "When the best hyperparameter value is at or near a search space boundary, the "
      "true optimum may lie outside the current range. The system detects this and expands:")
    code(pdf, """Detection: check if best value within 10% of boundary
  Example: seq_len range [8, 40], best=8
    (8-8)/(40-8) = 0.0 <= 0.10 -> LOW EDGE DETECTED

Expansion pools (next value added to boundary):
  forward_bars: low=[8], high=[64, 96]
  seq_len: low=[4], high=[48]
  hidden_dim: low=[32], high=[512]

Hard limits prevent unbounded growth:
  seq_len: [4, 64], hidden_dim: [32, 512], etc.

If categorical parameter changes: Optuna study DB is DELETED
  (incompatible with new search space)""")

    section(pdf, "4.9 Model Protection")
    s(pdf, "A new model is only saved to disk if its score exceeds the best_score recorded in "
      "adaptive_state. This prevents regression -- a bad retrain cycle cannot overwrite a "
      "good model. Files saved on improvement: model_v2.pth (weights), config_v2.pkl "
      "(hyperparams), scaler_v2.pkl (RobustScaler fitted on training data), "
      "feature_cols_v2.pkl (column names ensuring train/inference feature alignment).")
    s(pdf, "Current best: stock model score=11.009, forward_bars=96 (96-hour prediction horizon).")

    section(pdf, "4.10 Forward Bars")
    s(pdf, "The forward_bars parameter defines the prediction horizon -- how many bars ahead "
      "the model predicts returns. The default search set is [12, 18, 24, 32, 48] hours, "
      "expandable to 96 via edge detection. Longer horizons give the model more signal "
      "(larger moves to predict) but require wider stops and longer cooldowns. All risk "
      "parameters scale adaptively with forward_bars (see Chapter 10).")


# ---- CHAPTER 5 ----
def ch5(pdf):
    chapter(pdf, 5, "Trading Algorithm -- Main Loop")

    section(pdf, "5.1 Loop Cycle (every 30 seconds)")
    s(pdf, "The main trading loop runs a fixed sequence of operations every 30 seconds. "
      "Both CryptoLoop and StockLoop inherit this skeleton from BaseTradingLoop:")
    code(pdf, """run():
  while True:
    check_market_hours()           # Crypto: always True; Stocks: 9:30-4:00 ET
    _circuit_breaker_check()       # 5% daily drawdown -> flatten + 1h sleep
    flatten_before_close()         # Stocks: sell all at 3:50 PM ET
    _hot_reload_check()            # Reload model if .pth mtime changed

    [every 10 cycles = ~5 minutes]:
      _update_macro_regime()       # VIX, STLFSI2, CAPE, stablecoins
      _update_equity()             # Track peak equity for drawdown
      _update_correlations()       # 30-day rolling Pearson matrix
      log GPU temperature          # Thermal monitoring

    _manage_stops()                # Hard stop, trailing stop, take-profit
    preds = _get_predictions()     # Parallel LSTM+LGB inference (5 threads)
    _run_llm_analysis(preds)       # LLM scoring (throttled to 1 hour)
    _execute_sells(preds)          # Sell if bearish + LLM confirms
    _execute_llm_veto_sells()      # Sell if LLM score < 0.15 (catastrophic)
    _execute_covers(preds)         # Cover shorts if bullish or stopped
    _execute_shorts(preds)         # Short if strongly bearish
    _execute_buys(preds)           # Buy with full 12-layer risk stack
    _sleep(30s + jitter)           # Thermal throttle if GPU > 75C""")

    section(pdf, "5.2 Horizon-Adaptive Parameters")
    s(pdf, "All trading parameters scale with the prediction horizon (forward_bars) using "
      "sqrt(forward_bars/5) as the scaling factor. This ensures that stops, cooldowns, and "
      "thresholds are appropriately calibrated whether the model predicts 12 hours or 96 hours "
      "ahead. See Chapter 10 for full details.")

    section(pdf, "5.3 Min Return Floor")
    s(pdf, "Before any trade, the predicted return must exceed a minimum floor derived from "
      "the risk-free rate. This ensures the model predicts enough upside to justify the risk:")
    code(pdf, """min_return_floor = 10Y_Treasury_yield * (forward_bars / 252) * 2.0
  -> Must predict > 2x the risk-free rate for the holding period

min_short_floor  = 10Y_Treasury_yield * (forward_bars / 252) * 5.0
  -> Must predict > 5x risk-free rate to short (higher bar for shorts)

Example with 4.5% 10Y yield and forward_bars=96:
  min_return_floor = 0.045 * (96/252) * 2.0 = 3.43%
  min_short_floor  = 0.045 * (96/252) * 5.0 = 8.57%""")

    section(pdf, "5.4 Hot-Reload")
    s(pdf, "Every cycle checks if the model .pth file modification time has changed. If so, "
      "the model, scaler, feature_cols, and config are reloaded from disk. This allows the "
      "weekly retrain pipeline to update models while bots are running without any restart. "
      "The reload is atomic -- all four files are loaded together or none are updated.")


# ---- CHAPTER 6 ----
def ch6(pdf):
    chapter(pdf, 6, "Entry Gates (Buy Logic)")

    s(pdf, "Before any buy order is placed, the candidate symbol must pass ALL 11 sequential "
      "gates. If any gate rejects, the symbol is skipped for this cycle. The gates are ordered "
      "from cheapest to most expensive computation:")

    section(pdf, "Gate 1: Position Check")
    s(pdf, "The symbol must not already be held long, and must not have an active short position. "
      "Prevents doubling down and ensures clean position management.")

    section(pdf, "Gate 2: Cooldown")
    s(pdf, "The symbol must not have been traded within the cooldown window. Cooldown scales "
      "linearly with the prediction horizon to prevent whipsaw re-entry on longer timeframes:")
    code(pdf, """cooldown = base_cooldown * (forward_bars / base_forward_bars)

Examples:
  forward_bars=5:  cooldown = 60 minutes (base)
  forward_bars=24: cooldown = ~5 hours
  forward_bars=96: cooldown = ~19 hours""")

    section(pdf, "Gate 3: Hard-Stop Lockout")
    s(pdf, "If the symbol hit a hard stop-loss recently, it is locked out for 24-48 hours. "
      "This prevents 'revenge trading' -- the behavioral tendency to immediately re-enter "
      "after a loss, often with impaired judgment. Lockout is persisted to disk so it "
      "survives bot restarts.")

    section(pdf, "Gate 4: Prediction Exceeds Min Return Floor")
    s(pdf, "The predicted return must exceed the min_return_floor (2x risk-free rate for "
      "the holding period). This ensures the model sees enough upside to justify taking risk. "
      "A prediction of +0.5% is meaningless if the risk-free rate earns +0.4% over the "
      "same period.")

    section(pdf, "Gate 5: Quote Available + Spread Check")
    s(pdf, "A valid bid/ask quote must be available from the broker. The predicted return must "
      "exceed 2x the round-trip spread cost (spread-to-reward ratio). This ensures the edge "
      "exceeds market friction:")
    code(pdf, """Required: pred_return > 2 * spread_pct (round-trip)

Example: spread = 0.05% -> pred must be > 0.10%
  If pred = 0.08%, the edge is consumed by spread costs""")

    section(pdf, "Gate 6: Winner's Curse Filter")
    s(pdf, "Inspired by Milgrom & Wilson's auction theory (Nobel 2020). When price has extended "
      "beyond SMA_20 + 2*ATR, the buyer is likely overpaying for momentum that's about to "
      "reverse. In this condition, the prediction threshold is raised to 1.5x to require much "
      "stronger conviction before entering:")
    code(pdf, """if price > SMA_20 + 2 * ATR:
    effective_threshold = threshold * 1.5
    -> Requires 50% stronger signal to enter late-stage rallies""")

    section(pdf, "Gate 7: Correlation Check")
    s(pdf, "The average absolute correlation between the candidate and all currently held "
      "positions must be below 0.7. If above, entry is blocked entirely to prevent portfolio "
      "concentration. Based on Markowitz portfolio theory. See Chapter 15 for full details.")

    section(pdf, "Gate 8: Macro Regime")
    s(pdf, "Macro conditions can block entry entirely:")
    bullet(pdf, "VIX > 35: Halt ALL stock entries (crisis mode). Crypto unaffected.")
    bullet(pdf, "VIX 25-35: Block non-safe-haven stocks (only TLT, GLD, etc. allowed).")
    bullet(pdf, "Stablecoin depeg > 2%: Halt ALL crypto entries.")

    section(pdf, "Gate 9: Position Sizing (12-Layer Stack)")
    s(pdf, "The 12-layer risk stack (Chapter 8) computes the position size. If the final "
      "sized amount is below the minimum notional, entry is skipped. The sizing stack can "
      "effectively veto a trade by reducing size to zero.")

    section(pdf, "Gate 10: Sentiment Gate")
    s(pdf, "News-based sentiment scoring applies a multiplier to position size. Catastrophic "
      "news (score < -0.5) reduces size by 85%. Strong bullish news (score > 0.4) increases "
      "size by 35%. The multiplier range is 0.15x to 1.35x. See Chapter 12 for details.")

    section(pdf, "Gate 11: LLM Gate")
    s(pdf, "The LLM score must exceed BUY_LLM_MIN (approximately 0.60, adjusted by horizon). "
      "If the score is available, it also applies a sizing multiplier of (0.5 + score), "
      "ranging from 0.65x to 1.5x. A score below 0.15 is a hard veto that prevents entry "
      "regardless of all other signals.")


# ---- CHAPTER 7 ----
def ch7(pdf):
    chapter(pdf, 7, "Exit Logic (Sell, Short, Cover)")

    section(pdf, "7.1 Sell Hysteresis")
    s(pdf, "Selling requires BOTH a bearish prediction AND bearish LLM confirmation. This "
      "dual-confirmation prevents churn from noisy signals:")
    code(pdf, """Sell requires BOTH:
  pred < -(threshold * SELL_PRED_MULT)    # Model is bearish
  llm_score < SELL_LLM_MAX               # LLM confirms bearish view

Dead zone prevents oscillation:
  Buy zone:  LLM > 0.60 (approximately, adjusted by horizon)
  Dead zone: LLM 0.40 to 0.60 (no action -- hold current position)
  Sell zone: LLM < 0.40 (approximately, adjusted by horizon)

The gap between buy (0.60) and sell (0.40) thresholds ensures that
minor LLM score fluctuations do not trigger buy-sell-buy-sell churn.""")

    section(pdf, "7.2 Emergency Sell")
    s(pdf, "If the prediction is extremely bearish (beyond the short threshold, i.e., 5x "
      "risk-free rate), an emergency sell is triggered regardless of LLM score. This bypasses "
      "the normal hysteresis to protect against sharp drawdowns when the model has very high "
      "conviction in a downturn.")

    section(pdf, "7.3 LLM Veto Sell")
    s(pdf, "Every cycle, all held positions are checked against their LLM scores. If any "
      "position has LLM score < 0.15 (catastrophic zone), it is immediately liquidated. "
      "This overrides all other logic -- the LLM has identified a severe risk (fraud, hack, "
      "regulatory action) that the quantitative model cannot see in price data alone.")

    section(pdf, "7.4 Short Entry")
    s(pdf, "Short selling has a higher bar than long entry due to unlimited downside risk:")
    code(pdf, """Short entry requires ALL:
  pred < -min_short_floor              # 5x risk-free rate (very bearish)
  llm_score < 0.30                     # LLM strongly bearish
  Not a crypto symbol                  # No crypto shorting
  Not a leveraged ETF                  # No leveraged shorting
  Position size = half normal          # Half-sized for risk control

Short positions use the same stop-loss and take-profit logic as longs,
but inverted (stop above entry, take-profit below entry).""")

    section(pdf, "7.5 Short Cover")
    s(pdf, "Short positions are covered (closed) when any of these conditions are met:")
    bullet(pdf, "Prediction flips bullish (pred > threshold)")
    bullet(pdf, "LLM score rises above 0.65 (sentiment reversal)")
    bullet(pdf, "Stop-loss triggered (price rises above entry + stop_dist)")
    bullet(pdf, "Daily flatten for stocks at 3:50 PM ET")

    section(pdf, "7.6 Daily Flatten (Stocks Only)")
    s(pdf, "At 3:50 PM ET, all stock positions (both longs and shorts) are closed to avoid "
      "overnight gap risk. This protects against earnings announcements, geopolitical shocks, "
      "and overnight gap-downs that the intraday model cannot predict. Crypto never flattens "
      "since markets never close.")


# ---- CHAPTER 8 ----
def ch8(pdf):
    chapter(pdf, 8, "Position Sizing -- 12-Layer Risk Stack")

    s(pdf, "Position sizing passes through 12 multiplicative layers. Each layer can independently "
      "reduce (or modestly increase) the base notional. The design philosophy: multiple weak "
      "signals compound into strong risk management. No single layer dominates, and the stack "
      "is resilient to any individual layer failing.")

    section(pdf, "Step 1: Kelly Criterion (Half-Kelly)")
    s(pdf, "The Kelly Criterion (Kelly, 1956) computes the optimal fraction of capital to risk "
      "on each trade, maximizing long-term geometric growth rate. We use half-Kelly because: "
      "(1) it yields 75% of full-Kelly's return with only 50% of the variance, (2) full-Kelly "
      "drawdowns commonly exceed 50%, and (3) estimation error in win rate and payoff ratios "
      "makes full-Kelly dangerously aggressive:")
    code(pdf, """kelly_f = (win_rate * avg_win/avg_loss - (1 - win_rate)) / (avg_win/avg_loss)
half_kelly = kelly_f / 2, clamped to [0.05, 0.25]
position_size = half_kelly * equity

Data source: last 200 trades in trade_memory.json
Minimum: 50 trades before Kelly activates (else fixed NOTIONAL)""")

    section(pdf, "Step 2: VIX Scaling")
    code(pdf, """VIX > 35:  kelly_scale = 0.3  (crisis -- near halt)
VIX 25-35: kelly_scale = 0.5  (defensive)
VIX 15-25: kelly_scale = 0.7  (caution)
VIX < 15:  kelly_scale = 1.0  (normal -- full size)""")

    section(pdf, "Step 3: Drawdown-Based Reduction")
    s(pdf, "Tracks peak equity and reduces sizing when underwater. This prevents compounding "
      "losses during losing streaks:")
    code(pdf, """drawdown = (peak_equity - current_equity) / peak_equity
dd >= 20%: base *= 0.25 (75% reduction -- near halt)
dd >= 15%: base *= 0.50 (50% reduction)
dd >= 10%: base *= 0.75 (25% reduction)
dd < 10%:  no change""")

    section(pdf, "Step 4: Confidence Scaling")
    s(pdf, "Scales position size proportionally to prediction strength. The denominator is "
      "the min_return_floor (not the noise threshold), ensuring confidence reflects excess "
      "return over the risk-free hurdle:")
    code(pdf, """confidence = clamp(pred_return / min_return_floor, 0.5, 2.0)
sized = base * confidence

Effect: Stronger predictions get bigger positions.
  pred = 2x floor -> confidence = 2.0 -> double size
  pred = 0.5x floor -> confidence = 0.5 -> half size""")

    section(pdf, "Step 5: GARCH(1,1) Volatility Targeting")
    s(pdf, "Based on Robert Engle's GARCH model (Nobel Prize 2003). Each position targets "
      "the same daily volatility (2% default), regardless of the asset's actual volatility:")
    code(pdf, """sigma = GARCH/EGARCH 1-step-ahead forecast (decimal, e.g. 0.03)
ratio = target_vol / sigma = 0.02 / 0.03 = 0.67
sized *= clamp(ratio, 0.5, 2.0)

Effect: High-vol assets get smaller positions, low-vol get larger
  -> Equalizes dollar risk across portfolio (risk parity)""")

    section(pdf, "Step 6: Macro Regime Multiplier")
    s(pdf, "Combines VIX, STLFSI2, CAPE, and stablecoin peg data into a single sizing "
      "multiplier. See Chapter 14 for the full macro regime detection system.")

    section(pdf, "Step 7: Correlation Reduction (Markowitz)")
    code(pdf, """avg_corr = mean(abs(corr(candidate, each_held_position)))
sizing_factor = max(0.5, 1.0 - 0.5 * avg_corr)

Examples:
  corr = 0.0  -> factor = 1.0x  (uncorrelated, full size)
  corr = 0.35 -> factor = 0.825x (moderate reduction)
  corr = 0.7  -> REJECTED (blocked by Gate 7, never reaches here)""")

    section(pdf, "Step 8: HMM Regime Scaling")
    s(pdf, "3-state Gaussian HMM classifies the current market regime:")
    code(pdf, """Bull regime:    sizing = 1.2x (lean in)
Neutral regime: sizing = 1.0x (no change)
Bear regime:    sizing = 0.3x (70% reduction, very defensive)""")

    section(pdf, "Step 9: Ensemble Disagreement Penalty")
    s(pdf, "When the macro regime (VIX-based) and HMM regime disagree on market conditions, "
      "a 20% penalty is applied. Rationale: if two independent regime detectors see different "
      "things, uncertainty is high and exposure should be reduced:")
    code(pdf, """if macro_regime != hmm_regime:
    sized *= 0.8  (20% penalty for disagreement)""")

    section(pdf, "Step 10: Leveraged ETF Scaling")
    s(pdf, "Leveraged ETFs (e.g., TQQQ 3x, SOXL 3x) have their position size divided by "
      "the leverage factor. This ensures dollar risk is equivalent to the underlying asset. "
      "A $1,000 position in TQQQ has the same risk as $3,000 in QQQ.")

    section(pdf, "Step 11: Sentiment Gate Multiplier")
    code(pdf, """News-based sentiment multiplier:
  score <= -0.5: mult = 0.15  (catastrophic: hack, fraud, bankruptcy)
  score <= -0.3: mult = 0.35  (heavy bearish news)
  score <= -0.1: mult = 0.70  (mild caution)
  score >= 0.4:  mult = 1.35  (strong positive news)
  score >= 0.2:  mult = 1.20  (positive news)
  else:          mult = 1.00  (neutral)
  Final clamp: [0.15, 1.35]""")

    section(pdf, "Step 12: LLM Score Multiplier")
    code(pdf, """If LLM score available:
  sized *= (0.5 + llm_score)

  LLM score 0.15 -> mult = 0.65x (barely above veto)
  LLM score 0.50 -> mult = 1.00x (neutral)
  LLM score 0.70 -> mult = 1.20x (bullish confirmation)
  LLM score 1.00 -> mult = 1.50x (maximum conviction)
  Range: [0.65, 1.50]""")

    section(pdf, "Complete Sizing Example")
    code(pdf, """AAPL trade: pred=+3.5%, min_return_floor=3.43%, VIX=22, equity=$100k

  Step 1  (Kelly, half-kelly=0.12): 0.12 * $100k = $12k, cap $5k -> $5,000
  Step 2  (VIX=22, caution):        $5,000 * 0.7 = $3,500
  Step 3  (drawdown=8%, no DD):     $3,500 (unchanged)
  Step 4  (confidence=3.5/3.43):    $3,500 * 1.02 = $3,570
  Step 5  (GARCH sigma=0.018):      $3,570 * (0.02/0.018) = $3,967, cap 2x
  Step 6  (macro, VIX=22):          $3,967 * 0.8 = $3,173
  Step 7  (corr with MSFT=0.5):     $3,173 * 0.75 = $2,380
  Step 8  (HMM neutral, 1.0x):      $2,380
  Step 9  (macro=HMM, no penalty):  $2,380
  Step 10 (not leveraged):          $2,380
  Step 11 (sentiment=0.3, 1.2x):    $2,856
  Step 12 (LLM=0.7, 1.2x):         $3,427
  Final order notional:              $3,427""")


# ---- CHAPTER 9 ----
def ch9(pdf):
    chapter(pdf, 9, "Stop-Loss & Risk Controls")

    section(pdf, "9.1 ATR-Based Dynamic Stops")
    s(pdf, "Stop distances are computed from Average True Range (ATR), a volatility measure. "
      "This means volatile assets get wider stops (avoiding whipsaw) and calm assets get "
      "tighter stops (locking in gains sooner). All stop parameters scale with the prediction "
      "horizon via sqrt(forward_bars/5):")
    code(pdf, """stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / entry_price
stop_dist = clamp(stop_dist, FLOOR_PCT, CEIL_PCT)
stop_dist *= macro_regime.stop_mult  (tighten when stressed)

Horizon scaling (forward_bars=96 example):
  scale = sqrt(96/5) = 4.38
  FLOOR_PCT: base 5% * 4.38 = ~22%
  CEIL_PCT:  base 10% * 4.38 = ~40%
  -> Wider stops for longer horizons (more time = more volatility)""")

    section(pdf, "9.2 Trailing Stop")
    s(pdf, "The trailing stop activates after a position gains ATR_TRAIL_ACTIVATE_PCT (default "
      "approximately 1% of entry price, scaled by horizon). Once activated, it tracks the "
      "high-water-mark and sells if price drops below HWM * (1 - trail_dist):")
    code(pdf, """Every cycle, for each held position:
  1. Update high_water_mark = max(hwm, current_midpoint)

  2. Activation check:
     If hwm >= entry * (1 + ATR_TRAIL_ACTIVATE_PCT):
       trailing_activated = True

  3. Trailing stop check (only if activated):
     If price <= hwm * (1 - trail_dist):
       -> SELL, record exit_reason='trailing'
       -> Locks in profit as price rises, exits on pullback""")

    section(pdf, "9.3 Take-Profit")
    s(pdf, "Take-profit targets are set at a 3:1 risk-reward ratio relative to the stop "
      "distance, capped at TAKE_PROFIT_CEIL_PCT:")
    code(pdf, """tp_price = entry_price * (1 + stop_dist * TAKE_PROFIT_RR)
tp_price = min(tp_price, entry_price * (1 + TAKE_PROFIT_CEIL_PCT))

TAKE_PROFIT_RR = 3.0 (3:1 risk-reward)
TAKE_PROFIT_CEIL_PCT scales with horizon

Example with 5% stop:
  tp_price = entry * (1 + 0.05 * 3.0) = entry * 1.15 (15% gain target)""")

    section(pdf, "9.4 Macro Regime Tightening")
    s(pdf, "When the macro regime detects stress (elevated VIX, high STLFSI2, stablecoin "
      "instability), stop distances are tightened via regime.stop_mult. This reduces the "
      "maximum loss per position during volatile markets:")
    code(pdf, """stop_dist *= macro_regime.stop_mult

Normal:         stop_mult = 1.0  (no change)
STLFSI2 > 1.0: stop_mult = 0.8  (20% tighter)
Stablecoin depeg > 0.5%: stop_mult = 0.7 (30% tighter, crypto only)""")

    section(pdf, "9.5 Bracket Orders (Stocks)")
    s(pdf, "Stock positions use Alpaca bracket orders (server-side stops). When the parent "
      "buy order fills, child stop-loss and take-profit orders are automatically created by "
      "the broker. This provides protection even if the bot process crashes. After +1% gain, "
      "the server-side stop is upgraded to a trailing stop that adjusts dynamically.")

    section(pdf, "9.6 Circuit Breaker")
    s(pdf, "If the portfolio suffers a 5% daily drawdown, the circuit breaker triggers an "
      "emergency response:")
    code(pdf, """Trigger: (last_equity - current_equity) / last_equity >= 5%

Actions:
  1. Cancel all open orders immediately
  2. Market-sell all positions (longs and shorts)
  3. Sleep for 1 hour (no new trades)
  4. Resume monitoring after sleep

Rationale: Aligns with institutional risk management practice.
  Prevents cascading losses during flash crashes or black swan events.""")

    section(pdf, "9.7 Hard-Stop Lockout")
    s(pdf, "After a hard stop-loss fill, the symbol is locked out for 24-48 hours. This "
      "prevents 'revenge trading' -- the behavioral tendency (documented by Kahneman) to "
      "immediately re-enter after a loss, often increasing position size to 'make it back'. "
      "Lockout timestamps are persisted to hard_stop_lockout.json to survive bot restarts.")

    section(pdf, "9.8 Broker Desync Detection")
    s(pdf, "Before executing a sell, the system verifies the position actually exists at the "
      "broker. This prevents a dangerous failure mode where the bot's internal state shows a "
      "long position, but the broker has already filled a stop-loss. Without this check, "
      "selling a non-existent position would accidentally open a short.")


# ---- CHAPTER 10 ----
def ch10(pdf):
    chapter(pdf, 10, "Horizon-Adaptive Scaling")

    s(pdf, "All risk parameters derive from the forward_bars prediction horizon. This ensures "
      "that a model predicting 96 hours ahead uses appropriately wider stops, longer cooldowns, "
      "and higher return thresholds than a model predicting 12 hours ahead. The fundamental "
      "scaling law is based on the square root of time, which is how volatility scales under "
      "the random walk assumption.")

    section(pdf, "10.1 Core Scaling Formula")
    code(pdf, """scale = sqrt(forward_bars / 5)

forward_bars=5:  scale = 1.0   (baseline)
forward_bars=12: scale = 1.55
forward_bars=24: scale = 2.19
forward_bars=48: scale = 3.10
forward_bars=96: scale = 4.38""")

    section(pdf, "10.2 Stop Distances")
    s(pdf, "ATR multiplier, floor, and ceiling all scale with sqrt(time) because price "
      "volatility grows with the square root of the holding period:")
    code(pdf, """ATR_STOP_MULTIPLIER *= scale
FLOOR_PCT *= scale       # e.g., 5% base -> 22% at 96 bars
CEIL_PCT *= scale        # e.g., 10% base -> 44% at 96 bars

Rationale: A 96-hour position needs wider stops because price can
naturally fluctuate more over 4 days than over 5 hours. Tight stops
on long-horizon positions would cause constant stop-outs.""")

    section(pdf, "10.3 Cooldown Period")
    s(pdf, "Cooldown scales linearly (not sqrt) with horizon because it represents a minimum "
      "holding period, not a volatility measure:")
    code(pdf, """cooldown = base_cooldown * (forward_bars / base_forward_bars)

forward_bars=5:  cooldown = 60 minutes (base)
forward_bars=12: cooldown = ~2.4 hours
forward_bars=24: cooldown = ~5 hours
forward_bars=48: cooldown = ~10 hours
forward_bars=96: cooldown = ~19 hours""")

    section(pdf, "10.4 LLM Hysteresis Thresholds")
    s(pdf, "The gap between buy and sell LLM thresholds widens slightly with horizon to "
      "reduce churn on longer-term positions:")
    code(pdf, """Buy LLM threshold:  0.55 to 0.60 (higher for longer horizons)
Sell LLM threshold: 0.40 to 0.45 (higher for longer horizons)

Dead zone: the gap between buy and sell thresholds
  Short horizon (5 bars):  dead zone = 0.55 - 0.40 = 0.15
  Long horizon (96 bars):  dead zone = 0.60 - 0.45 = 0.15

Within the dead zone, no action is taken (hold current position).""")

    section(pdf, "10.5 Min Return Floor")
    s(pdf, "The minimum return floor scales naturally with holding period because it is "
      "derived from the annualized risk-free rate:")
    code(pdf, """min_return_floor = 10Y_yield * (forward_bars / 252) * 2.0
min_short_floor  = 10Y_yield * (forward_bars / 252) * 5.0

With 4.5% 10Y yield:
  forward_bars=5:  buy floor = 0.18%,  short floor = 0.45%
  forward_bars=24: buy floor = 0.86%,  short floor = 2.14%
  forward_bars=96: buy floor = 3.43%,  short floor = 8.57%""")

    section(pdf, "10.6 Confidence Scaling Denominator")
    s(pdf, "Confidence scaling uses min_return_floor as its denominator, not the noise "
      "threshold. This means confidence = (pred / min_return_floor), which correctly "
      "measures how much excess return the model predicts above the risk-free hurdle. "
      "A prediction of 7% with a 3.43% floor gives confidence 2.0 (maximum), while "
      "a prediction of 3.5% gives confidence 1.02 (barely above neutral).")


# ---- CHAPTER 11 ----
def ch11(pdf):
    chapter(pdf, 11, "LLM Integration")

    section(pdf, "11.1 Two-Stage Pipeline")
    s(pdf, "LLM analysis runs on a 1-hour cadence (LLM_INTERVAL_SEC=3600) and consists of "
      "two stages: global context generation and per-symbol analysis.")

    subsection(pdf, "Stage 1: Global Context (hourly)")
    s(pdf, "Fetches broad market data and asks the LLM to produce a structured market digest:")
    code(pdf, """Data fetched:
  Market indices: S&P 500, NASDAQ, VIX, DXY (dollar), Oil, Gold, 10Y yield, BTC
    Each with: current price, 52-week high/low, 1w/1m/3m/1y returns
  Sector ETFs (10): XLF, XLK, XLE, XLV, XLI, XLP, XLU, XLRE, XLB, XLC
    Each with: same metrics as indices
  Fear & Greed indices (CNN + crypto)
  General news: 20 recent headlines

LLM output (structured JSON):
  regime: risk-on / risk-off / rotational / crisis
  themes: list of {theme, description, sector_impact_tags}
  risk_factors: list of current risks with severity
  opportunities: list of potential opportunities""")

    subsection(pdf, "Stage 2: Per-Symbol Analysis (hourly)")
    s(pdf, "For each symbol in the candidate list (top predictions + held positions), the "
      "LLM receives the global context plus symbol-specific data and produces a structured "
      "analysis:")
    code(pdf, """System prompt defines 5 analysis dimensions:
  1. WHY: Explain the directional thesis in 1-2 sentences
  2. FUNDAMENTALS: Key financial metrics and valuation
  3. CATALYSTS: Upcoming events that could move the price
  4. RISKS: Specific risks to the thesis
  5. SYNTHESIS: Overall conviction with reasoning

Output: {s: score (0.0-1.0), m: multiplier, r: reasoning,
         bull: bull case, bear: bear case}""")

    section(pdf, "11.2 Score Interpretation")
    s(pdf, "The LLM score is a continuous value from 0.0 to 1.0 with defined zones:")
    code(pdf, """0.00 - 0.15: VETO (catastrophic -- do not trade under any circumstances)
0.15 - 0.35: Bearish (strong negative outlook)
0.35 - 0.48: Lean negative (mild concern)
0.48 - 0.52: Neutral (insufficient conviction either way)
0.52 - 0.65: Lean positive (mild bullish)
0.65 - 0.85: Bullish (strong positive outlook)
0.85 - 1.00: Strong conviction (very high confidence)""")

    section(pdf, "11.3 Time Horizon Alignment")
    s(pdf, "The forward_bars parameter is injected into the LLM prompt so the analysis "
      "matches the model's prediction horizon. A model predicting 96 hours ahead needs "
      "the LLM to evaluate 4-day catalysts, not intraday noise. The prompt explicitly "
      "states the holding period and asks the LLM to align its conviction accordingly.")

    section(pdf, "11.4 Smart Routing")
    s(pdf, "The system auto-detects free vs paid API tier from the x-ratelimit-limit-requests "
      "header on the first API call (RPM <= 15 = free tier). Cost-based routing then selects "
      "the appropriate model based on cumulative daily spend:")
    code(pdf, """Daily cost thresholds:
  < $0.10:     Pro for analyst, Flash for sentiment (best quality)
  $0.10-$0.25: Downgrade analyst to Flash
  $0.25-$0.50: Further downgrade to 2.5-Flash
  > $0.50:     Lite models throughout (cheapest)
  Hard cap:    $1.00/day (~$30/month)

Model pricing (per million tokens):
  Gemini 3.1 Pro:        $2.00 in / $12.00 out
  Gemini 3 Flash:        $0.50 in / $3.00 out
  Gemini 3.1 Flash Lite: $0.25 in / $1.50 out
  Gemini 2.5 Flash:      $0.15 in / $0.60 out

Config keys: analyst_model_override, sentiment_model_override,
  tier_override (null = auto-detect)""")


# ---- CHAPTER 12 ----
def ch12(pdf):
    chapter(pdf, 12, "Sentiment Analysis")

    section(pdf, "12.1 Two-Tier Architecture")
    s(pdf, "Sentiment analysis uses two complementary approaches: a fast deterministic keyword "
      "scorer for live trading decisions, and an LLM-based scorer for higher accuracy when "
      "API budget allows:")
    bullet(pdf, "Tier 1 -- Keyword scoring: Instant, no API call, deterministic. Used for all "
           "live trading decisions where latency matters.")
    bullet(pdf, "Tier 2 -- LLM batch scoring: Higher accuracy, tiered by article recency. "
           "Runs in background, results merged into daily aggregates.")

    section(pdf, "12.2 Keyword Scoring Algorithm")
    s(pdf, "The keyword engine uses a comprehensive dictionary of financial terms with "
      "negation-aware scoring:")
    code(pdf, """Dictionary size:
  70+ positive words, 100+ negative words
  20+ positive phrases, 25+ negative phrases

Phase 1: Phrase matching (word-boundary regex, negation-aware)
  Positive phrases: 'all-time high' (1.5), 'beat expectations' (1.5), ...
  Negative phrases: 'death cross' (-1.5), 'slashed price target' (-2.0), ...
  If negator within 3-word window: flip polarity * 0.7

Phase 2: Single-word scoring with bidirectional negation check
  Each positive keyword: +1.0 (or -0.7 if negated)
  Each negative keyword: -1.0 (or +0.7 if negated)
  Negators: 'not', 'no', 'never', 'neither', 'barely', 'hardly', etc.

Phase 3: Tanh normalization
  scale = 0.4 / sqrt(word_count / 10)
  score = tanh(raw * scale)  -> smooth to (-1.0, +1.0)

Combination: headline_score * 0.6 + summary_score * 0.4""")

    section(pdf, "12.3 LLM Tiering by Article Recency")
    s(pdf, "When LLM scoring is available, articles are prioritized by recency to allocate "
      "the best (most expensive) models to the most important content:")
    code(pdf, """Newest 20% of articles -> Pro model (highest accuracy)
Next 40% of articles   -> Flash model (good accuracy, lower cost)
Remaining 40%          -> Lite model (adequate accuracy, cheapest)

Rationale: Recent news has the most impact on near-term price action.
  Older articles are less actionable and don't justify premium model costs.""")

    section(pdf, "12.4 Sentiment Gate Integration")
    s(pdf, "The sentiment gate applies a multiplicative factor to position sizing based on "
      "symbol-specific news sentiment:")
    code(pdf, """Sentiment multiplier mapping:
  score <= -0.5: mult = 0.15  (catastrophic news: hack, fraud)
  score <= -0.3: mult = 0.35  (heavy bearish news)
  score <= -0.1: mult = 0.70  (mild caution)
  score >= 0.4:  mult = 1.35  (strong positive news)
  score >= 0.2:  mult = 1.20  (positive news)
  else:          mult = 1.00  (neutral)""")
    s(pdf, "Design principle: the ML model already sees Daily_Sentiment as a training feature, "
      "so broad market sentiment adjustments would double-count. Symbol-specific breaking news "
      "IS new information the model cannot see, so hard gates for catastrophic events (hack, "
      "fraud, bankruptcy) are applied as a safety net.")

    section(pdf, "12.5 Fear & Greed Index")
    s(pdf, "Crypto-specific bubble protection: when the Fear & Greed Index reaches extreme "
      "greed (>= 90), a 0.7x sizing reduction is applied. This is the only Fear & Greed "
      "integration point, because the ML model already captures sentiment via the "
      "Daily_Sentiment feature:")
    code(pdf, """Crypto: alternative.me API (free, no auth required)
  0-24:  Extreme Fear     25-49: Fear
  50:    Neutral           51-74: Greed
  75-89: Greed            90-100: Extreme Greed -> 0.7x bubble protection

Stocks: CNN Fear & Greed (includes VIX component, AAII survey, etc.)
Cache: 5 minutes""")


# ---- CHAPTER 13 ----
def ch13(pdf):
    chapter(pdf, 13, "GARCH Volatility Forecasting")

    s(pdf, "Based on Robert Engle's ARCH/GARCH framework (Nobel Prize 2003). GARCH captures "
      "volatility clustering -- the empirical observation that high-volatility periods tend to "
      "persist, and low-volatility periods tend to persist. This provides a forward-looking "
      "volatility estimate that is superior to backward-looking measures like ATR or simple "
      "standard deviation.")

    section(pdf, "13.1 GARCH(1,1) Model")
    code(pdf, """sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

Where:
  sigma_t^2: conditional variance at time t
  omega:     baseline variance (long-run average volatility)
  alpha:     reaction coefficient (how fast vol responds to shocks)
  beta:      persistence coefficient (how long high/low vol persists)
  r_{t-1}:   previous period return

Typical values: alpha ~ 0.05-0.15, beta ~ 0.80-0.95
  alpha + beta < 1.0 required for stationarity""")

    section(pdf, "13.2 EGARCH Preference")
    s(pdf, "The system tries EGARCH (Exponential GARCH) first, falling back to standard "
      "GARCH(1,1) if EGARCH fitting fails. EGARCH captures the asymmetric volatility response "
      "-- the empirical fact that market crashes increase volatility more than rallies of "
      "equivalent magnitude. This asymmetry is especially important for stop-loss calibration, "
      "as downside volatility is what triggers stops.")

    section(pdf, "13.3 Implementation Details")
    code(pdf, """Implementation (volatility.py):
  1. Try EGARCH first (captures crash asymmetry)
  2. Fall back to standard GARCH(1,1) if EGARCH fails
  3. Require >= 100 data points to fit (statistical reliability)
  4. Cache fitted model for 1 hour (vol is persistent, refit expensive)
  5. Forecast 1-step-ahead variance -> sqrt -> sigma

Used for:
  Position sizing: ratio = target_vol(2%) / sigma, clamped [0.5, 2.0]
  Effect: high-vol asset (sigma=4%) -> 0.5x size (half position)
          low-vol asset (sigma=1%) -> 2.0x size (double position)
  -> Equalizes dollar risk per position across the portfolio""")


# ---- CHAPTER 14 ----
def ch14(pdf):
    chapter(pdf, 14, "Macro Regime Detection")

    s(pdf, "The macro regime system combines multiple independent indicators to classify "
      "the overall market environment. Each indicator has specific thresholds and actions:")

    section(pdf, "14.1 VIX (CBOE Volatility Index)")
    code(pdf, """Source: yfinance (^VIX), 1-hour cache
Thresholds:
  VIX < 15:  Normal      -> sizing 1.0x, no restrictions
  VIX 15-25: Caution     -> sizing 0.8x
  VIX 25-35: Defensive   -> sizing 0.5x, block risky stock entries
                             (only safe-havens: TLT, GLD, etc. allowed)
  VIX > 35:  Crisis      -> sizing 0.3x, HALT ALL stock entries

The VIX is the primary macro signal because it reflects real-time options
pricing and represents aggregate market fear.""")

    section(pdf, "14.2 STLFSI2 (Financial Stress Index)")
    code(pdf, """Source: FRED CSV download, 1-day cache
Published by: St. Louis Federal Reserve (weekly)
Components: 18 financial indicators including yield spreads,
  volatility measures, and funding stress indicators

Thresholds:
  STLFSI2 <= 1.0: Normal -> no adjustment
  STLFSI2 > 1.0:  High stress -> sizing *= 0.5, stops *= 0.8 (tighter)

Captures systemic financial stress that VIX alone may miss
(e.g., credit market freeze, interbank lending stress).""")

    section(pdf, "14.3 CAPE Ratio (Cyclically Adjusted PE)")
    code(pdf, """Estimation: SPY trailing PE * 1.6 (approximation), 1-day cache
z-score = (CAPE - 25.0) / 8.0

Thresholds:
  z-score <= 1.5: Normal -> no adjustment
  z-score > 1.5:  Overvalued -> stock sizing *= 0.7

Based on Shiller's CAPE (Nobel 2013). Historically, CAPE > 35
signals overvaluation. The z-score normalization allows the system
to adapt to slowly rising market valuations.""")

    section(pdf, "14.4 Stablecoin Peg Monitoring")
    code(pdf, """Source: Alpaca quotes (USDT, USDC), 5-minute cache
Measures: deviation from $1.00 peg

Thresholds:
  depeg > 2.0%: EMERGENCY -> halt ALL crypto trading, flatten positions
                  Signals potential exchange insolvency or contagion
  depeg > 0.5%: WARNING -> tighten crypto stops *= 0.7 (30% tighter)
                  Signals stress in crypto infrastructure

Historical precedent: UST depeg in May 2022 preceded $40B collapse.
Early detection of stablecoin stress can prevent catastrophic losses.""")

    section(pdf, "14.5 Regime Output")
    code(pdf, """MacroRegime dataclass output:
  sizing_mult:     0.0 to 1.0+ (multiplicative position scaling)
  stop_mult:       0.7 to 1.0 (tighten stops when stressed)
  regime_label:    'normal', 'caution', 'defensive', 'crisis'
  stablecoin_alert: True/False (crypto-specific warning)

Properties:
  should_halt_stocks:       VIX > 35
  should_block_risky_entries: VIX > 25""")


# ---- CHAPTER 15 ----
def ch15(pdf):
    chapter(pdf, 15, "Correlation-Aware Sizing")

    s(pdf, "Based on Markowitz portfolio theory (Nobel 1990). The correlation system prevents "
      "concentrated portfolios where multiple positions move together, which amplifies "
      "drawdowns during market stress.")

    section(pdf, "15.1 Correlation Matrix Estimation")
    code(pdf, """Process:
  1. Fetch 30-bar rolling returns for all symbols in universe
  2. Compute correlation matrix:
     - Try Ledoit-Wolf shrinkage estimator (sklearn)
       Regularizes toward structured estimator, robust to small samples
     - Fallback to numpy corrcoef if sklearn unavailable
  3. Cache result for 1 hour (recalculated every 10 cycles = ~5 min)""")

    section(pdf, "15.2 Entry Blocking")
    s(pdf, "Before any buy, the system computes the average absolute correlation between "
      "the candidate symbol and all currently held positions:")
    code(pdf, """avg_corr = mean(abs(corr(candidate, each_held_position)))

If avg_corr > 0.7: REJECT entry entirely
  -> Symbol is too correlated with existing portfolio
  -> Adding it would increase concentration risk without diversification

Example: Holding AAPL, MSFT, GOOGL. Candidate: META
  corr(META, AAPL) = 0.82, corr(META, MSFT) = 0.78, corr(META, GOOGL) = 0.75
  avg_corr = 0.783 > 0.7 -> REJECTED (tech concentration)""")

    section(pdf, "15.3 Sizing Reduction")
    s(pdf, "For candidates that pass the 0.7 threshold, a sizing reduction is applied "
      "proportional to the correlation level:")
    code(pdf, """sizing_factor = max(0.5, 1.0 - 0.5 * avg_corr)

Examples:
  avg_corr = 0.0  -> factor = 1.0x  (fully uncorrelated, no reduction)
  avg_corr = 0.2  -> factor = 0.9x  (slight reduction)
  avg_corr = 0.4  -> factor = 0.8x  (moderate reduction)
  avg_corr = 0.6  -> factor = 0.7x  (significant reduction)
  avg_corr = 0.69 -> factor = 0.655x (near rejection threshold)

Floor of 0.5 prevents complete elimination -- even highly correlated
assets may have valid idiosyncratic signals.""")

    section(pdf, "15.4 Ledoit-Wolf Shrinkage")
    s(pdf, "The Ledoit-Wolf shrinkage estimator is critical for correlation estimation with "
      "small sample sizes (30 bars). Classical sample correlation matrices are notoriously "
      "unstable with fewer observations than dimensions. Ledoit-Wolf regularizes the sample "
      "covariance matrix toward a structured target (scaled identity), producing more stable "
      "and reliable correlation estimates. This addresses the well-known limitation Markowitz "
      "himself acknowledged: portfolio optimization is 'extremely sensitive to estimation "
      "errors in the input parameters.'")


# ---- CHAPTER 16 ----
def ch16(pdf):
    chapter(pdf, 16, "HMM Regime Detection")

    s(pdf, "A 3-state Gaussian Hidden Markov Model classifies the market into regimes based "
      "on return distributions. The key insight from Sargent and Sims (Nobel 2011): markets "
      "alternate between distinct states (trending, mean-reverting, volatile), and the current "
      "state is only probabilistically observable from price data.")

    section(pdf, "16.1 Model Specification")
    code(pdf, """Model: GaussianHMM (from hmmlearn library)
  n_states = 3
  covariance_type = 'full'
  Minimum data: 200 bars (statistical reliability)
  Input: 1D array of log returns

Fitting: Baum-Welch algorithm (Expectation-Maximization)
  Iteratively estimates: transition matrix, emission means, emission covariances
  Refit daily (HMM fitting is expensive, regimes change slowly)
  Cached for 24 hours between refits""")

    section(pdf, "16.2 State Classification")
    s(pdf, "After fitting, the three states are sorted by their mean return to assign "
      "semantic labels:")
    code(pdf, """State sorting by emission mean:
  Lowest mean return  -> Bear state
  Middle mean return  -> Neutral state
  Highest mean return -> Bull state

Additional classification:
  If neutral state vol > 1.5x median vol of all states:
    -> Reclassified as High-Volatility state""")

    section(pdf, "16.3 Trading Parameter Adjustments")
    code(pdf, """Bull regime:
  sizing = 1.2x (lean in, momentum likely to continue)
  threshold = 0.8x (lower bar for entry, more permissive)
  stops = 1.0x (normal)

Neutral regime:
  sizing = 1.0x (no change)
  threshold = 1.0x (no change)
  stops = 1.0x (normal)

Bear regime:
  sizing = 0.3x (70% reduction, very defensive)
  threshold = 1.5x (50% higher bar for entry)
  stops = 0.8x (20% tighter, protect capital)

High-volatility state:
  sizing = 0.5x (50% reduction)
  threshold = 1.2x (higher bar)
  stops = 1.3x (30% wider to avoid whipsaw)""")

    section(pdf, "16.4 Whipsaw Prevention (3-Bar Persistence)")
    s(pdf, "To prevent rapid oscillation between regimes on minor price fluctuations, the "
      "system requires 3 consecutive bars in a new regime before switching:")
    code(pdf, """_smooth_regime(current_state, state_history):
  If last 3 states are all the same new state:
    -> Switch to new regime (confirmed transition)
  Else:
    -> Stay in previous regime (insufficient confirmation)

This prevents:
  Cycle 1: Bull -> (noise) -> Bear -> (noise) -> Bull
  From causing rapid sizing changes (1.2x -> 0.3x -> 1.2x)
  Instead: stays in previous regime until new one persists""")


# ---- CHAPTER 17 ----
def ch17(pdf):
    chapter(pdf, 17, "Data Pipeline")

    section(pdf, "17.1 Storage Format")
    s(pdf, "All market data is stored in Parquet format with Snappy compression, with CSV "
      "files maintained for backward compatibility. Parquet provides significant advantages "
      "for time series data: columnar storage reduces I/O for feature computation, Snappy "
      "compression reduces disk usage by 3-5x, and read performance is 5-10x faster than CSV.")

    section(pdf, "17.2 Three-Source Fallback")
    s(pdf, "Data harvesting uses a cascading fallback strategy to maximize data availability:")
    code(pdf, """1. Alpaca (primary): Hourly bars
   Crypto: 2021+, Stock: 2016+
   Chunked 6-month fetches with exponential backoff (4, 8, 16, 32s)
   Rate limit: adaptive pacing (up to 30s between chunks on 429)

2. yfinance (secondary): Last 730 days hourly
   Pre/post market data included
   Flattens MultiIndex columns for yfinance 0.2.x compatibility

3. CryptoCompare (crypto only, tertiary): Free API, no key required
   2000-bar backward chunks
   Filters bars with close > 0 (removes invalid data)""")

    section(pdf, "17.3 Incremental Harvesting")
    s(pdf, "After initial data download, subsequent harvests only fetch bars newer than the "
      "last run, with a 48-hour overlap for safety (catches any bars that may have been "
      "delayed or corrected by the exchange):")
    code(pdf, """Process:
  1. Read existing Parquet file, find most recent timestamp per ticker
  2. Fetch new bars from (last_timestamp - 48 hours) to now
  3. Merge new bars with existing data, deduplicate on timestamp
  4. Write atomically (temp file then os.replace, crash-safe)

Row capping: If total rows > max_rows, keep most recent per ticker
  (balanced across symbols to prevent one ticker from dominating)""")

    section(pdf, "17.4 Multi-Horizon Targets")
    s(pdf, "Target variables are computed for multiple prediction horizons simultaneously:")
    code(pdf, """Target_Return_N = (close[t+N] - close[t]) / close[t] * 100

N in [12, 18, 24, 32, 48] bars ahead (configurable, expandable to 96)
Optuna searches across horizons to find the most predictable timeframe.
Stale targets (e.g., Target_Return_6 from old configs) are auto-removed.""")

    section(pdf, "17.5 Data Validation")
    bullet(pdf, "Gap detection: identifies missing bars in the time series")
    bullet(pdf, "NaN/Inf checks: removes rows with invalid numeric values")
    bullet(pdf, "Per-ticker coverage: ensures minimum bar count per symbol")
    bullet(pdf, "Atomic writes: write-then-rename pattern prevents partial file corruption "
           "on crash or power loss (important for Jetson which may lose power)")


# ---- CHAPTER 18 ----
def ch18(pdf):
    chapter(pdf, 18, "Research Foundations")

    s(pdf, "The system's design decisions are grounded in peer-reviewed research and Nobel "
      "Prize-winning economic theory. Below is a mapping of key academic contributions to "
      "specific implementation decisions in the trading system.")

    section(pdf, "18.1 Harry Markowitz (1952, Nobel 1990) -- Modern Portfolio Theory")
    s(pdf, "PAPER: 'Portfolio Selection', Journal of Finance, 1952. Markowitz demonstrated "
      "that portfolio risk depends not just on individual asset risk but on the correlations "
      "between assets. Diversification reduces total portfolio risk without proportionally "
      "reducing expected return.")
    s(pdf, "IMPLEMENTATION: portfolio.py uses correlation-aware position sizing (Chapter 15). "
      "Entry is blocked when avg correlation > 0.7. Sizing is reduced proportionally to "
      "correlation. The Sharpe ratio (derived from Markowitz's framework by his student "
      "William Sharpe) is the optimization objective for the Optuna search.")

    section(pdf, "18.2 Robert Engle (Nobel 2003) -- ARCH/GARCH")
    s(pdf, "DISCOVERY: Autoregressive Conditional Heteroskedasticity -- volatility itself is "
      "predictable. High-volatility periods cluster together, as do low-volatility periods. "
      "This is the most important insight for risk management: we can forecast tomorrow's "
      "volatility from today's.")
    s(pdf, "IMPLEMENTATION: volatility.py implements GARCH(1,1) and EGARCH for volatility "
      "forecasting (Chapter 13). Used for position sizing (target 2% daily vol per position), "
      "stop-loss calibration, and risk parity across the portfolio.")

    section(pdf, "18.3 Daniel Kahneman (Nobel 2002) -- Prospect Theory")
    s(pdf, "DISCOVERY: People feel losses approximately 2x as painfully as equivalent gains "
      "(loss aversion). This creates systematic behavioral biases: panic selling at bottoms "
      "(our bot buys), holding losers too long hoping for recovery (our mechanical stops "
      "cut losses), and selling winners too early to lock in gains (our trailing stops "
      "let winners run).")
    s(pdf, "IMPLEMENTATION: The entire mechanical execution system is the implementation. "
      "By removing human emotional decision-making, the bot exploits the behavioral biases "
      "of other market participants. The hard-stop lockout (Chapter 9) specifically prevents "
      "revenge trading -- a manifestation of loss aversion.")

    section(pdf, "18.4 Richard Thaler (Nobel 2017) -- Behavioral Economics")
    s(pdf, "DISCOVERY: Built on Kahneman's work to show that behavioral biases persist even "
      "among sophisticated investors. Calendar effects (Turn of Month, January effect) exist "
      "because institutional cash flows are behaviorally driven.")
    s(pdf, "IMPLEMENTATION: Calendar features (Hour/Day/Month sin/cos, Turn_of_Month flag) "
      "in Chapter 3. Sentiment gate design recognizes that news-driven panic creates "
      "predictable overreaction patterns.")

    section(pdf, "18.5 J.L. Kelly Jr. (1956) -- Kelly Criterion")
    s(pdf, "PAPER: 'A New Interpretation of Information Rate', Bell System Technical Journal. "
      "Kelly proved that betting a fraction f* = edge/odds maximizes the long-term geometric "
      "growth rate of capital. Half-Kelly (f*/2) sacrifices only 25% of growth rate while "
      "halving variance and maximum drawdown.")
    s(pdf, "IMPLEMENTATION: trading_utils.py computes half-Kelly from the last 200 trades, "
      "clamped to [0.05, 0.25]. Minimum 50 trades required before Kelly activates. "
      "VIX-scaled Kelly further adjusts based on market conditions (Step 2 of the risk stack).")

    section(pdf, "18.6 Baum-Welch Algorithm -- HMM Training")
    s(pdf, "The Baum-Welch algorithm (a special case of Expectation-Maximization) fits the "
      "Hidden Markov Model parameters from observed returns. It iteratively estimates the "
      "transition probabilities between states and the emission distributions within each "
      "state, converging to a local maximum of the likelihood function.")
    s(pdf, "IMPLEMENTATION: regime_detector.py fits a 3-state Gaussian HMM using hmmlearn's "
      "implementation of Baum-Welch (Chapter 16). Daily refit, 200+ bar minimum, 3-bar "
      "persistence hysteresis for whipsaw prevention.")

    section(pdf, "18.7 Hochreiter & Schmidhuber (1997) -- LSTM")
    s(pdf, "PAPER: 'Long Short-Term Memory', Neural Computation. LSTMs solved the vanishing "
      "gradient problem that prevented standard RNNs from learning long-range dependencies. "
      "The gated architecture (input, forget, output gates) allows the network to selectively "
      "remember or forget information over hundreds of timesteps.")
    s(pdf, "IMPLEMENTATION: model_v2.py uses LSTM as the primary temporal encoder (Chapter 2). "
      "The LSTM captures sequential patterns in hourly price data -- momentum, mean reversion, "
      "volatility clustering -- that are invisible to static feature-based models.")

    section(pdf, "18.8 Vaswani et al. (2017) -- Attention Mechanism")
    s(pdf, "PAPER: 'Attention Is All You Need', NeurIPS 2017. Multi-head self-attention allows "
      "the model to learn which parts of the input sequence are most relevant to the output, "
      "with different attention heads learning different aspects of relevance.")
    s(pdf, "IMPLEMENTATION: model_v2.py adds MultiHeadAttention on top of the LSTM output "
      "(Chapter 2). Self-attention with Q=K=V=lstm_output allows the model to learn which "
      "timesteps in the lookback window are most informative for predicting forward returns. "
      "The residual connection and LayerNorm ensure the attention layer enhances rather than "
      "degrades the LSTM's temporal signal.")


# ============================================================
# MAIN
# ============================================================

def main():
    pdf = Manual()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)

    title_page(pdf)
    build_toc(pdf)
    ch1(pdf)
    ch2(pdf)
    ch3(pdf)
    ch4(pdf)
    ch5(pdf)
    ch6(pdf)
    ch7(pdf)
    ch8(pdf)
    ch9(pdf)
    ch10(pdf)
    ch11(pdf)
    ch12(pdf)
    ch13(pdf)
    ch14(pdf)
    ch15(pdf)
    ch16(pdf)
    ch17(pdf)
    ch18(pdf)

    out = "/home/kyle/trader/Trader_System_Manual.pdf"
    pdf.output(out)
    print(f"Generated: {out}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
