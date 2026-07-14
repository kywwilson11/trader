"""Close-out fixes from the 2026-07 module-review workflow (b23 P1s).

Two unambiguous bugs fixed during the final review pass:
1. Live stock bars fetched with the SDK-default adjustment='raw' while the
   harvest trains on adjustment='all' — a split/dividend inside the live
   window skewed every price-derived feature vs the trained distribution.
2. Circuit-breaker flatten-failure tracking compared BROKER-format failure
   symbols ('BTCUSD') against universe-format position keys ('BTC/USD'),
   so the keep-failed-symbols filter kept nothing for crypto and dropped
   live, unprotected positions from tracking.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_live_stock_fetch_uses_adjusted_bars():
    src = (REPO / "market_data.py").read_text()
    start = src.index("def fetch_stock_bars_alpaca")
    body = src[start:src.index("\ndef ", start + 10)]
    assert "adjustment='all'" in body, (
        "live stock bars must be split/dividend-adjusted like the harvest")


def test_market_data_module_still_imports():
    import market_data
    assert callable(market_data.fetch_stock_bars_alpaca)
    assert callable(market_data.drop_forming_bar)


def test_circuit_breaker_normalizes_failure_symbols():
    src = (REPO / "base_loop.py").read_text()
    start = src.index("failures = emergency_flatten(")
    body = src[start:start + 1600]
    # sentinel guard: unknown broker state keeps everything tracked
    assert "'<list_positions failed>' in failures" in body
    # normalized comparison: universe 'BTC/USD' matches broker 'BTCUSD'
    assert "f.replace('/', '')" in body
    assert "s.replace('/', '') in failed_norm" in body


def test_normalization_logic():
    # the exact comparison base_loop now performs, on representative data
    failures = ['BTCUSD', 'ETHUSD', 'TSLA']
    positions = {'BTC/USD': 1, 'ETH/USD': 2, 'SOL/USD': 3, 'TSLA': 4,
                 'NVDA': 5}
    failed_norm = {f.replace('/', '') for f in failures}
    kept = {s for s in positions if s.replace('/', '') in failed_norm}
    assert kept == {'BTC/USD', 'ETH/USD', 'TSLA'}  # failed stay tracked
    # flattened-OK positions (SOL, NVDA) are the ones released
