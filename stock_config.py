"""Market universe configuration — load/save the list of traded symbols.

Reads from stock_universe.json, falling back to hardcoded defaults.
Supports both stock symbols (TSLA) and crypto pairs (BTC/USD).
No heavy imports (json, pathlib only) so it's safe for the GUI env.
"""

import json
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
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return list(_DEFAULTS)


def save_stock_universe(symbols: list[str]) -> None:
    """Persist a new market universe to disk (sorted, deduplicated)."""
    clean = _clean(symbols)
    with open(_FILE, 'w') as f:
        json.dump(clean, f, indent=2)


# Leveraged ETFs: ticker -> leverage multiplier
# Position sizes are divided by this factor to normalize risk
LEVERAGED_ETFS = {
    'TQQQ': 3,
    'SOXL': 3,
}

# Safe-haven symbols allowed to trade during VIX > 25 defensive regimes
SAFE_HAVEN_SYMBOLS = {'GLD', 'SLV', 'PALL', 'PPLT', 'OXY', 'COPX'}

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
}

# Bucket notional caps as a fraction of MAX_EXPOSURE
BUCKET_CAP_FRACTION = {
    'crypto_proxy': 0.20,
    'default': 0.35,
}


# Top cryptos by market cap (USD pairs on Alpaca)
CRYPTO_SYMBOLS = [
    'BTC/USD',
    'ETH/USD',
    'XRP/USD',
    'SOL/USD',
    'DOGE/USD',
    'LINK/USD',
]
