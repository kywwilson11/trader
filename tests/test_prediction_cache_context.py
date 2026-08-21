"""GUI review 2026-07 §5/§11 Phase 2.3 (producer side): prediction-cache
decision-context enrichment (meta_p, conviction, regime, llm_gate, rank).

crypto_loop.py / stock_loop.py cannot be imported directly on this Mac
(base_loop -> trading_utils -> dotenv, not installed here) — matching the
existing tests/test_review_b01.py convention, write_prediction_cache is
extracted by AST from the source text and exec'd with a stubbed `self`, so
this runs fully (no skip needed) without ever importing the real modules
or any heavy dependency.
"""
import ast
import datetime
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
CRYPTO_SRC = (REPO / "crypto_loop.py").read_text()
STOCK_SRC = (REPO / "stock_loop.py").read_text()


def _extract_method(src: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, ast.FunctionDef)
                        and item.name == method_name):
                    return textwrap.dedent(ast.get_source_segment(src, item))
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_method(src, class_name, method_name, glb):
    seg = _extract_method(src, class_name, method_name)
    ns = dict(glb)
    exec(compile(seg, f"<{method_name}>", "exec"), ns)
    return ns[method_name]


class _Log:
    def debug(self, *a): pass
    def info(self, *a): pass
    def warning(self, *a): pass
    def error(self, *a): pass


def _glb(cache_path):
    import os
    return {'json': json, 'datetime': datetime, 'os': os,
            'logger': _Log(), '_PRED_CACHE_FILE': cache_path}


def _conviction_stub(pred, meta_p):
    """Stand-in for BaseTradingLoop._conviction_tier: write_prediction_cache
    only needs to CALL it and use the return value. The tiering logic
    itself is covered by tests/test_conviction_journal.py."""
    return 'A' if meta_p is not None and meta_p >= 0.6 else 'B'


# ---------------------------------------------------------------------------
# crypto_loop.write_prediction_cache
# ---------------------------------------------------------------------------

def test_crypto_cache_adds_context_keys_when_available(tmp_path):
    cache = tmp_path / 'crypto_predictions.json'
    fn = _load_method(CRYPTO_SRC, 'CryptoLoop', 'write_prediction_cache',
                      _glb(cache))
    me = SimpleNamespace(
        trade_threshold=0.15,
        _last_meta_p={'BTC/USD': 0.72},
        llm_scores={'BTC/USD': {'s': 0.6}, 'ETH/USD': {'s': 0.05}},
        _veto_strikes={'ETH/USD': 1},
        macro_regime=SimpleNamespace(regime_label='risk_on'),
        _conviction_tier=_conviction_stub,
    )
    fn(me, {'BTC/USD': 0.5, 'ETH/USD': -0.3, 'SOL/USD': 0.01})
    data = json.loads(cache.read_text())

    # BTC/USD: meta_p on record, LLM scored and NOT struck -> pass
    assert data['BTC/USD']['meta_p'] == 0.72
    assert data['BTC/USD']['conviction'] == 'A'
    assert data['BTC/USD']['regime'] == 'risk_on'
    assert data['BTC/USD']['llm_gate'] == 'pass'
    assert data['BTC/USD']['signal'] == 'BULL'    # existing behavior intact
    assert data['BTC/USD']['pred'] == 0.5

    # ETH/USD: no meta_p on record, but LLM scored AND currently struck
    assert 'meta_p' not in data['ETH/USD']
    assert 'conviction' not in data['ETH/USD']
    assert data['ETH/USD']['llm_gate'] == 'veto'
    assert data['ETH/USD']['regime'] == 'risk_on'  # book-wide, not per-name

    # SOL/USD: never scored by the LLM at all -> key OMITTED, not None
    assert 'llm_gate' not in data['SOL/USD']
    assert 'meta_p' not in data['SOL/USD']

    # cost-gate/cost-bps have no cached home anywhere on the loop -> never emitted
    for row in data.values():
        assert 'cost_gate' not in row
        assert 'cost_bps' not in row
        # crypto has no entry-rank concept at all -> never emitted
        assert 'rank' not in row


def test_crypto_cache_omits_all_context_keys_when_nothing_available(tmp_path):
    """Symmetric to test_review_b01's exact-dict-equality DOGE/USD check:
    with no enrichment sources on self, entries stay exactly {pred, score,
    signal, updated} -- proves the new keys are additive-only, never
    null-padded filler (and that this change cannot break that test)."""
    cache = tmp_path / 'crypto_predictions.json'
    fn = _load_method(CRYPTO_SRC, 'CryptoLoop', 'write_prediction_cache',
                      _glb(cache))
    me = SimpleNamespace(trade_threshold=0.15)
    fn(me, {'BTC/USD': 0.5})
    data = json.loads(cache.read_text())
    assert set(data['BTC/USD']) == {'pred', 'score', 'signal', 'updated'}


# ---------------------------------------------------------------------------
# stock_loop.write_prediction_cache -- same context keys, plus entry-rank
# ---------------------------------------------------------------------------

def test_stock_cache_adds_context_keys_including_rank(tmp_path):
    cache = tmp_path / 'stock_predictions.json'
    fn = _load_method(STOCK_SRC, 'StockLoop', 'write_prediction_cache',
                      _glb(cache))
    me = SimpleNamespace(
        trade_threshold=0.15,
        top_symbols=['AAPL', 'MSFT'],
        _last_meta_p={'AAPL': 0.8},
        llm_scores={'AAPL': {'s': 0.7}},
        _veto_strikes={},
        macro_regime=SimpleNamespace(regime_label='neutral'),
        _conviction_tier=_conviction_stub,
    )
    fn(me, {'AAPL': 0.5, 'MSFT': 0.3, 'XOM': -0.4})
    data = json.loads(cache.read_text())

    assert data['AAPL']['rank'] == 1
    assert data['MSFT']['rank'] == 2
    assert 'rank' not in data['XOM']           # outside the ranked top-N

    assert data['AAPL']['meta_p'] == 0.8
    assert data['AAPL']['conviction'] == 'A'
    assert data['AAPL']['llm_gate'] == 'pass'
    assert data['AAPL']['regime'] == 'neutral'
    assert data['AAPL']['signal'] == 'BULL'        # existing behavior intact

    # MSFT: ranked (top-N) but never individually meta/LLM scored
    assert 'meta_p' not in data['MSFT']
    assert 'llm_gate' not in data['MSFT']
    assert data['MSFT']['regime'] == 'neutral'     # regime is book-wide


def test_stock_cache_omits_all_context_keys_when_nothing_available(tmp_path):
    cache = tmp_path / 'stock_predictions.json'
    fn = _load_method(STOCK_SRC, 'StockLoop', 'write_prediction_cache',
                      _glb(cache))
    me = SimpleNamespace(trade_threshold=0.15, top_symbols=['AAPL'])
    fn(me, {'AAPL': 0.5, 'XOM': -0.4})
    data = json.loads(cache.read_text())
    # AAPL is top-ranked -> gets `rank`; nothing else is available
    assert set(data['AAPL']) == {'pred', 'score', 'signal', 'updated', 'rank'}
    assert set(data['XOM']) == {'pred', 'score', 'signal', 'updated'}
