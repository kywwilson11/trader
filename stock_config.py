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


# Top cryptos by market cap (USD pairs on Alpaca)
CRYPTO_SYMBOLS = [
    'BTC/USD',
    'ETH/USD',
    'XRP/USD',
    'SOL/USD',
    'DOGE/USD',
    'LINK/USD',
]
