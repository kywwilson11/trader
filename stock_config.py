"""Market universe configuration — load/save the list of traded symbols.

Reads from stock_universe.json, falling back to hardcoded defaults.
Supports both stock symbols (TSLA) and crypto pairs (BTC/USD).
No heavy imports (stdlib only) so it's safe for the GUI env.
"""

import json
import logging
import os
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "stock_universe.json"

_DEFAULTS = [
    'ABNB', 'AFRM', 'AMD', 'ARKK', 'ARM', 'ASTS',
    'AVAX/USD', 'BCH/USD', 'BTC/USD',
    'COIN', 'COPX', 'CRSP', 'CRWD',
    'DASH', 'DOGE/USD', 'DOT/USD',
    'ENPH', 'ETH/USD', 'FSLR', 'GLD', 'HOOD',
    'IONQ', 'LINK/USD', 'LTC/USD',
    'MARA', 'META', 'MRNA', 'MRVL', 'MSTR',
    'NET', 'NVDA', 'OXY', 'PALL', 'PLTR', 'POET', 'PPLT',
    'QBTS', 'QS',
    'RBLX', 'RDW', 'RKLB', 'ROKU',
    'SERV', 'SHOP', 'SLV', 'SMCI', 'SNAP', 'SOFI', 'SOL/USD', 'SOXL',
    'TQQQ', 'TSLA',
    'UBER',
    'XRP/USD',
]


def _clean(symbols):
    """Deduplicate and sort: stocks first (alphabetical), then crypto (alphabetical)."""
    unique = set(s.upper().strip() for s in symbols if s.strip())
    stocks = sorted(s for s in unique if '/' not in s)
    crypto = sorted(s for s in unique if '/' in s)
    return stocks + crypto


def load_stock_universe() -> list[str]:
    """Return the current market universe (stocks then crypto, sorted)."""
    try:
        with open(_FILE) as f:
            symbols = json.load(f)
        if isinstance(symbols, list) and symbols:
            return _clean(symbols)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        # AttributeError: _clean on non-string entries. The file is committed,
        # so ANY fallback silently swaps the traded universe — warn loudly.
        logging.getLogger(__name__).warning(
            "stock_universe.json unreadable (%r) — falling back to %d "
            "hardcoded defaults", exc, len(_DEFAULTS))
    return list(_DEFAULTS)


def save_stock_universe(symbols: list[str]) -> None:
    """Persist a new market universe to disk (sorted, deduplicated)."""
    clean = _clean(symbols)
    # Atomic replace: bot loops re-read this file every cycle while the GUI
    # writes it, so a truncate-then-rewrite (or crash mid-write) must never
    # be observable as torn JSON.
    tmp = _FILE.with_name(_FILE.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(clean, f, indent=2)
    os.replace(tmp, _FILE)


# Leveraged ETFs: ticker -> leverage multiplier
# Position sizes are divided by this factor to normalize risk
LEVERAGED_ETFS = {
    'TQQQ': 3,
    'SOXL': 3,
}

# Safe-haven symbols allowed to trade during VIX > 25 defensive regimes.
# Wave-9 #3: added consumer-staples + healthcare defensives so the book can
# rotate into low-beta names (not just metals) on risk-off days — the de-
# clustering that breadth promotion is meant to capture.
SAFE_HAVEN_SYMBOLS = {'GLD', 'SLV', 'PALL', 'PPLT', 'OXY', 'COPX',
                      'PG', 'KO', 'PEP', 'WMT', 'JNJ', 'MRK', 'T', 'VZ'}

# ETFs in the universe/panel — residual momentum is hard-zeroed for these
# (an index product's residual vs the index is degenerate noise)
ETF_TICKERS = {'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'PALL', 'PPLT', 'COPX',
               'ARKK', 'TQQQ', 'SOXL'}

# --- Training candidate pool (survivorship mitigation; NOT traded) ---
# The trading universe is ~50 hand-picked high-beta names — a winner-
# tilted panel. Training on it alone teaches patterns conditioned on
# eventual success (documented +1.6-4.9pp/yr backtest inflation). The
# harvest therefore trains on universe + this sector-diverse liquid pool,
# masked to the AS-OF top-K by trailing dollar volume, so each name
# contributes rows only from periods a mechanical rule would have
# selected it. Residual bias: names that faded BEFORE today are still
# absent (no free historical-membership data source).
TRAINING_CANDIDATE_POOL = [
    # Financials / payments
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB',
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'LLY', 'UNH', 'ABBV',
    # Consumer / staples
    'PG', 'KO', 'PEP', 'WMT', 'MCD', 'HD', 'LOW',
    # Industrials
    'CAT', 'DE', 'BA', 'GE', 'HON', 'UPS',
    # Telecom
    'T', 'VZ',
    # Megacap / legacy tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'INTC', 'CSCO', 'ORCL', 'IBM',
    'TXN', 'QCOM',
    # Index ETFs (always top-of-rank; regime diversity)
    'SPY', 'QQQ', 'IWM',
]

# As-of membership: keep a training row only when its name ranked in the
# top K of the harvested panel by trailing 30d median dollar volume AT
# THAT TIME
AS_OF_TOP_K = 60

# Candidates fetch from here (not 2016): bounds the Jetson concat spike,
# and the row-capped trainer mostly samples the recent era anyway
CANDIDATE_START = '2021-01-01'


# --- Sector buckets (factor-crowding caps) ---
# The ranked top-N clusters into one theme on exactly the days that theme
# runs hot; per-name correlation sizing helps but a hard bucket notional
# cap is the backstop. Unmapped symbols are uncapped. crypto_proxy is the
# tightest cap because the CRYPTO BOOK already carries spot beta to the
# same factor (MSTR is a leveraged BTC bet wearing an equity ticker).
SECTOR_BUCKETS = {
    # Crypto-beta equities
    'COIN': 'crypto_proxy', 'MSTR': 'crypto_proxy', 'MARA': 'crypto_proxy',
    'HOOD': 'crypto_proxy',
    # Semis / AI hardware (SOXL is 3x the same factor)
    'NVDA': 'semis', 'AMD': 'semis', 'ARM': 'semis', 'AVGO': 'semis',
    'MRVL': 'semis', 'SMCI': 'semis', 'SOXL': 'semis', 'POET': 'semis',
    # Speculative growth / pre-profit moonshots
    'IONQ': 'spec_growth', 'QBTS': 'spec_growth', 'RKLB': 'spec_growth',
    'ASTS': 'spec_growth', 'RDW': 'spec_growth', 'SERV': 'spec_growth',
    'QS': 'spec_growth', 'CRSP': 'spec_growth', 'PRME': 'spec_growth',
    'MRNA': 'spec_growth', 'ENPH': 'spec_growth', 'FSLR': 'spec_growth',
    'AFRM': 'spec_growth', 'SOFI': 'spec_growth', 'ARKK': 'spec_growth',
    # Metals / hard assets
    'GLD': 'metals', 'SLV': 'metals', 'PALL': 'metals', 'PPLT': 'metals',
    'COPX': 'metals',
    # --- wave-9 #3: map the previously-UNMAPPED (= uncapped) names so the
    # factor-crowding cap actually diversifies the top-K (a cap only truncates,
    # never enlarges, so this is strictly more conservative). ---
    'JPM': 'financials', 'BAC': 'financials', 'WFC': 'financials',
    'GS': 'financials', 'MS': 'financials', 'V': 'financials',
    'MA': 'financials', 'AXP': 'financials',
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy',
    'OXY': 'energy',
    'JNJ': 'healthcare', 'PFE': 'healthcare', 'MRK': 'healthcare',
    'LLY': 'healthcare', 'UNH': 'healthcare', 'ABBV': 'healthcare',
    'PG': 'staples', 'KO': 'staples', 'PEP': 'staples', 'WMT': 'staples',
    'MCD': 'staples', 'HD': 'staples', 'LOW': 'staples',
    'CAT': 'industrials', 'DE': 'industrials', 'BA': 'industrials',
    'GE': 'industrials', 'HON': 'industrials', 'UPS': 'industrials',
    'T': 'telecom', 'VZ': 'telecom',
    'AAPL': 'megacap_tech', 'MSFT': 'megacap_tech', 'GOOGL': 'megacap_tech',
    'AMZN': 'megacap_tech', 'INTC': 'megacap_tech', 'CSCO': 'megacap_tech',
    'ORCL': 'megacap_tech', 'IBM': 'megacap_tech', 'TXN': 'megacap_tech',
    'QCOM': 'megacap_tech', 'META': 'megacap_tech',
    'SPY': 'index', 'QQQ': 'index', 'IWM': 'index', 'TQQQ': 'index',
    'ABNB': 'growth_tech', 'DASH': 'growth_tech', 'UBER': 'growth_tech',
    'SHOP': 'growth_tech', 'NET': 'growth_tech', 'CRWD': 'growth_tech',
    'PLTR': 'growth_tech', 'RBLX': 'growth_tech', 'ROKU': 'growth_tech',
    'SNAP': 'growth_tech', 'TSLA': 'growth_tech',
}

# Bucket notional caps as a fraction of MAX_EXPOSURE
BUCKET_CAP_FRACTION = {
    'crypto_proxy': 0.20,
    'default': 0.35,
}

# --- Live tradable universe promotion (wave-9 #3) ---
# The model scores universe + TRAINING_CANDIDATE_POOL every hour, but the loop
# trades only the hand-list. OFF by default: promoting the as-of top-K of the
# full panel into the SELECTABLE set is MODEL-FACING (the model must actually
# predict the pool names live) and must clear the IC-by-name diagnostic on the
# Jetson first (promote only names whose individual OOS rank-IC is positive).
# K_HOLD > K_ENTER is the hysteresis band; held names are always included.
TRADABLE_POOL_ENABLED = False
AS_OF_TRADABLE_TOP_K = 20
TRADABLE_K_ENTER = 20
TRADABLE_K_HOLD = 28


# Top cryptos by market cap (USD pairs on Alpaca).
# Wave-9 #6: AVAX/BCH/DOT/LTC are in stock_universe.json but were dropped here —
# restoring them (6 -> 10) gives the crypto cross-sectional ranks + dispersion
# 67% more breadth. This is MODEL-FACING: the crypto model was trained on the
# 6-coin panel, so the 4 must be added HERE only together with a crypto
# harvest+retrain (train/serve parity), on the Jetson.
CRYPTO_SYMBOLS = [
    'BTC/USD',
    'ETH/USD',
    'XRP/USD',
    'SOL/USD',
    'DOGE/USD',
    'LINK/USD',
]

# The intended full coin set for wave-9 #6 — a DECLARATION ONLY, nothing reads
# it yet. The harvest does NOT consume this: scripts/harvest_crypto_data.py
# hardcodes its own 6-coin CRYPTO_TICKERS list (~line 30), which must be
# updated to this set in the SAME Jetson change (then harvest+retrain, then
# promote the coins into CRYPTO_SYMBOLS for live trading).
CRYPTO_POOL = CRYPTO_SYMBOLS + ['AVAX/USD', 'BCH/USD', 'DOT/USD', 'LTC/USD']
