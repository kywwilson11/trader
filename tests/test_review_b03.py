"""2026-07 review batch b03: trading_utils / alpaca_compat / types_mod fixes.

alpaca_compat and types_mod import cleanly on the dev Mac (no heavy deps at
module scope). trading_utils needs dotenv, so (following the
tests/test_prediction_cache.py / tests/test_review_b01.py pattern) it is
covered by source guards plus extract-and-exec tests of the pure functions.

Covers:
  - trading_utils: dead hw_monitor import removed; model_reload_key TOCTOU
    (manifest stat via get_model_mtime, race-free); get_api loud
    missing-env logging + live-endpoint warning; compute_kelly_fraction
    docstring contract ([0.05, 0.25] clamp, 0.05 floor on losing edge);
    shared-constants comment no longer claims ORDER_TIMEOUT centralization.
  - alpaca_compat: stop_limit order support (P1 — crypto resting stop);
    account/position/order shim parity with the GUI's reads (P2);
    portfolio-history equity/timestamp/profit_loss pair filtering (P2);
    snapshot + raw-delete GUI surface (P2); _parse_dt blank/garbage
    degradation; _timeframe fail-loud; **_ignored kwargs warnings.
  - types_mod: LLMResult dead code removed; unused `field` import removed;
    to_dict now delegates to dataclasses.asdict (no silent field drift).
"""
import ast
import datetime
import json
import logging
import re
import textwrap
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import alpaca_compat
import types_mod
from alpaca_compat import (CompatREST, _shim_account, _shim_bar, _shim_order,
                           _shim_portfolio_history, _shim_position,
                           _shim_snapshot)

REPO = Path(__file__).resolve().parent.parent
UTILS_SRC = (REPO / "trading_utils.py").read_text()
COMPAT_SRC = (REPO / "alpaca_compat.py").read_text()
TYPES_SRC = (REPO / "types_mod.py").read_text()
CRYPTO_SRC = (REPO / "crypto_loop.py").read_text()


# ---------------------------------------------------------------------------
# Extraction helper (pattern: exec one module-level function, stubbed globals)
# ---------------------------------------------------------------------------

def _extract_func(src: str, name: str, replace: dict | None = None) -> str:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = textwrap.dedent(ast.get_source_segment(src, node))
            for old, new in (replace or {}).items():
                assert old in seg, f"{old!r} not in {name}"
                seg = seg.replace(old, new)
            return seg
    raise AssertionError(f"{name} not found")


def _load_func(src, name, glb, replace=None):
    ns = dict(glb)
    exec(compile(_extract_func(src, name, replace), f"<{name}>", "exec"), ns)
    return ns[name]


# ===========================================================================
# trading_utils
# ===========================================================================

def test_dead_hw_monitor_import_removed():
    assert 'is_gpu_available' not in UTILS_SRC
    assert 'from hw_monitor' not in UTILS_SRC


def test_constants_comment_no_longer_claims_order_timeout_centralized():
    # The old comment claimed centralization that base_loop/stock_loop never
    # adopted for ORDER_TIMEOUT (they define their own class attrs).
    assert 'centralizing here prevents drift' not in UTILS_SRC
    assert 'is NOT centralized here' in UTILS_SRC
    # The constant itself must survive: tests/test_new_modules.py pins it.
    assert 'ORDER_TIMEOUT = 30' in UTILS_SRC


class TestModelReloadKey:
    def _run(self, files, prefix=''):
        calls = []

        def get_model_mtime(path):
            calls.append(path)
            return files.get(path, 0)

        fn = _load_func(UTILS_SRC, 'model_reload_key',
                        {'get_model_mtime': get_model_mtime})
        return fn(prefix), calls

    def test_no_exists_getmtime_race(self):
        seg = _extract_func(UTILS_SRC, 'model_reload_key')
        # The TOCTOU pair is gone; all stats go through get_model_mtime.
        assert 'os.path.exists' not in seg
        assert 'os.path.getmtime' not in seg

    def test_manifest_preferred(self):
        m, calls = self._run({'model_v2.manifest.json': 111.5,
                              'model_v2.pth': 55.0})
        assert m == 111.5
        assert calls == ['model_v2.manifest.json']  # pth never consulted

    def test_pth_fallback_when_manifest_missing(self):
        m, calls = self._run({'model_v2.pth': 55.0})
        assert m == 55.0
        assert calls == ['model_v2.manifest.json', 'model_v2.pth']

    def test_both_missing_returns_zero(self):
        m, _ = self._run({})
        assert m == 0

    def test_prefix_paths(self):
        m, calls = self._run({'stock_model_v2.manifest.json': 7.0},
                             prefix='stock')
        assert m == 7.0
        assert calls == ['stock_model_v2.manifest.json']


def test_get_api_missing_env_is_loud():
    msgs = []
    fake_os = SimpleNamespace(getenv=lambda *a, **k: None, environ={})
    fn = _load_func(
        UTILS_SRC, 'get_api',
        {'os': fake_os, 'print': lambda m: msgs.append(str(m))},
        replace={
            # force the legacy-SDK-unavailable path on every machine
            'import alpaca_trade_api as tradeapi':
                "raise ImportError('stubbed')",
            # don't construct a real CompatREST (needs alpaca)
            'api = CompatREST(key, secret, base_url)':
                "api = ('compat', key, secret, base_url)",
            # the timeout installer isn't in the extracted namespace
            # (D01 wired it in; it needs a real REST client anyway)
            '_install_rest_timeouts(api)': 'pass',
        })
    result = fn()
    assert result == ('compat', None, None, None)
    assert any('ALPACA_API_KEY' in m and 'ERROR' in m for m in msgs)
    assert any('ALPACA_API_SECRET' in m and 'ERROR' in m for m in msgs)
    assert any('ALPACA_BASE_URL' in m and 'LIVE' in m for m in msgs)


class TestKellyFraction:
    """The docstring now documents the real contract; pin that behavior."""

    def _kelly(self, tmp_path, data, **kw):
        f = tmp_path / 'trade_memory.json'
        f.write_text(json.dumps(data))
        fn = _load_func(UTILS_SRC, 'compute_kelly_fraction',
                        {'json': json, 'np': np, '_TRADE_MEMORY_FILE': f})
        return fn(**kw)

    @staticmethod
    def _trades(n_win, n_loss, win=2.0, loss=-1.0, estimated=False):
        out = []
        for i in range(n_win + n_loss):
            out.append({'ts': f'2026-06-{(i % 28) + 1:02d}T{i % 24:02d}:00:00',
                        'pnl_pct': win if i < n_win else loss,
                        'estimated': estimated})
        return out

    def test_losing_edge_floors_at_005(self, tmp_path):
        # Raw Kelly is negative here; the documented contract is the 0.05
        # floor (base_loop maps it to a <1.0x multiplier), never 0/negative.
        data = {'BTC/USD': self._trades(10, 50, win=0.5, loss=-1.0)}
        assert self._kelly(tmp_path, data) == 0.05

    def test_hot_sample_capped_at_025(self, tmp_path):
        data = {'BTC/USD': self._trades(55, 5, win=3.0, loss=-0.5)}
        v = self._kelly(tmp_path, data)
        assert v is not None and 0.05 <= v <= 0.25

    def test_insufficient_history_returns_none(self, tmp_path):
        data = {'BTC/USD': self._trades(5, 5)}
        assert self._kelly(tmp_path, data) is None

    def test_asset_type_scopes_books(self, tmp_path):
        data = {'BTC/USD': self._trades(30, 30)}
        assert self._kelly(tmp_path, data, asset_type='stock') is None
        assert self._kelly(tmp_path, data, asset_type='crypto') is not None

    def test_estimated_trades_excluded(self, tmp_path):
        data = {'BTC/USD': self._trades(30, 30, estimated=True)}
        assert self._kelly(tmp_path, data) is None

    def test_docstring_documents_clamp_and_consumer_contract(self):
        doc = ast.get_docstring(next(
            n for n in ast.parse(UTILS_SRC).body
            if isinstance(n, ast.FunctionDef)
            and n.name == 'compute_kelly_fraction'))
        assert '[0.05, 0.25]' in doc
        assert '(0.0 to 1.0)' not in doc          # the rotted claim
        assert '0.125' in doc                     # base_loop mapping
        assert '50 pseudo-trades' in doc          # shrinkage prior


def test_kelly_position_size_still_present():
    # Deletion was DEFERRED: tests/test_new_modules.py::TestKelly pins this
    # function and that file is outside batch-b03 ownership. Guard against a
    # half-applied removal that would break Jetson/CI.
    assert 'def kelly_position_size' in UTILS_SRC


# ===========================================================================
# alpaca_compat — pure shims (Mac-runnable with synthetic namespaces)
# ===========================================================================

def _enum(v):
    return SimpleNamespace(value=v)


class TestShimOrder:
    def test_gui_surface_and_types(self):
        sub = datetime.datetime(2026, 7, 1, 12, 0)
        fil = datetime.datetime(2026, 7, 1, 12, 1)
        leg = SimpleNamespace(id='L1', client_order_id=None, symbol='AAPL',
                              qty=None, side=_enum('sell'),
                              order_type=_enum('limit'), status=_enum('new'),
                              filled_qty=None, filled_avg_price=None,
                              legs=None)
        o = SimpleNamespace(id='O1', client_order_id='cid', symbol='AAPL',
                            qty='1.5', side=_enum('buy'),
                            order_type=_enum('market'),
                            status=_enum('filled'), filled_qty='1.5',
                            filled_avg_price='100.25', notional='150.38',
                            submitted_at=sub, filled_at=fil, legs=[leg])
        s = _shim_order(o)
        assert (s.id, s.symbol, s.qty) == ('O1', 'AAPL', 1.5)
        assert (s.side, s.type, s.status) == ('buy', 'market', 'filled')
        assert s.filled_avg_price == 100.25
        # GUI fetch_orders reads these three (P2: were missing -> AttributeError)
        assert s.notional == 150.38
        assert s.submitted_at is sub and s.filled_at is fil
        # legs recursion + None handling
        assert s.legs[0].qty is None and s.legs[0].filled_qty == 0.0
        assert s.legs[0].submitted_at is None and s.legs[0].notional is None

    def test_gui_str_pattern_on_missing_timestamps(self):
        o = SimpleNamespace(id='O2', symbol='ETH/USD', qty=None, side='buy',
                            order_type=None, status=None)
        s = _shim_order(o)
        # gui.py: str(o.submitted_at) if o.submitted_at else ""
        assert (str(s.submitted_at) if s.submitted_at else "") == ""
        assert (str(s.filled_at) if s.filled_at else "") == ""


class TestShimPosition:
    def test_gui_surface(self):
        p = SimpleNamespace(symbol='BTCUSD', qty='0.5', side=_enum('long'),
                            avg_entry_price='50000', current_price='51000',
                            market_value='25500', unrealized_pl='500',
                            unrealized_plpc='0.02')
        s = _shim_position(p)
        # exactly what gui.fetch_positions reads
        for attr in ('symbol', 'qty', 'side', 'avg_entry_price',
                     'current_price', 'unrealized_pl', 'unrealized_plpc',
                     'market_value'):
            assert hasattr(s, attr), attr
        assert s.side == 'long'
        assert s.unrealized_plpc == 0.02

    def test_missing_optionals_default(self):
        p = SimpleNamespace(symbol='AAPL', qty='10', avg_entry_price='100')
        s = _shim_position(p)
        assert s.side == ''
        assert s.unrealized_plpc == 0.0 and s.unrealized_pl == 0.0


def test_shim_account_gui_surface():
    a = SimpleNamespace(equity='100000.5', last_equity='99000',
                        cash='5000.25', portfolio_value='100000.5',
                        buying_power='200000', status=_enum('ACTIVE'),
                        trading_blocked=False)
    s = _shim_account(a)
    # gui.fetch_account reads equity/cash/buying_power/last_equity/
    # portfolio_value (P2: cash + portfolio_value were missing)
    assert s.cash == 5000.25
    assert s.portfolio_value == 100000.5
    assert s.equity == 100000.5 and s.last_equity == 99000.0
    assert s.status == 'ACTIVE' and s.trading_blocked is False


class TestShimPortfolioHistory:
    def test_none_equity_dropped_pairwise(self):
        h = SimpleNamespace(timestamp=[1, 2, 3, 4],
                            equity=[100.0, None, '102.5', 103.0],
                            profit_loss=[0.0, None, 2.5, 0.5],
                            profit_loss_pct=[0.0, None, 0.025, 0.005])
        s = _shim_portfolio_history(h)
        assert s.equity == [100.0, 102.5, 103.0]
        assert s.timestamp == [1, 3, 4]           # was [1,2,3,4] -> misaligned
        assert s.profit_loss == [0.0, 2.5, 0.5]
        assert s.profit_loss_pct == [0.0, 0.025, 0.005]
        # beta_ledger contract: pd.Series(equity, index=timestamp)
        assert len(s.equity) == len(s.timestamp)

    def test_missing_pl_arrays_yield_empty_lists(self):
        h = SimpleNamespace(timestamp=[1, 2], equity=[10.0, 11.0])
        s = _shim_portfolio_history(h)
        assert s.profit_loss == [] and s.profit_loss_pct == []
        assert s.equity == [10.0, 11.0]
        # gui.fetch_history: list(hist.profit_loss) if hist.profit_loss else []
        assert (list(s.profit_loss) if s.profit_loss else []) == []

    def test_all_none_or_empty(self):
        s = _shim_portfolio_history(
            SimpleNamespace(timestamp=None, equity=None))
        assert s.equity == [] and s.timestamp == []


class TestShimSnapshot:
    def test_watchlist_surface(self):
        def bar(c):
            return SimpleNamespace(open=c - 1, high=c + 1, low=c - 2,
                                   close=c, volume=1000, timestamp=None)
        snap = SimpleNamespace(latest_trade=SimpleNamespace(price=123.4),
                               daily_bar=bar(120.0),
                               previous_daily_bar=bar(118.0))
        s = _shim_snapshot(snap)
        # gui watchlist reads latest_trade.p, daily_bar.o/h/l/c/v, prev_daily_bar.c
        assert s.latest_trade.p == 123.4
        assert (s.daily_bar.o, s.daily_bar.h, s.daily_bar.l) == (119.0, 121.0, 118.0)
        assert (s.daily_bar.c, s.daily_bar.v) == (120.0, 1000)
        assert s.prev_daily_bar.c == 118.0

    def test_missing_pieces_are_none(self):
        s = _shim_snapshot(SimpleNamespace(latest_trade=None, daily_bar=None,
                                           previous_daily_bar=None))
        assert s.latest_trade is None
        assert s.daily_bar is None and s.prev_daily_bar is None


class TestParseDt:
    def test_passthrough_and_iso(self):
        dt = datetime.datetime(2026, 7, 1, 9, 30)
        assert CompatREST._parse_dt(None) is None
        assert CompatREST._parse_dt(dt) is dt
        parsed = CompatREST._parse_dt('2026-07-01T09:30:00Z')
        assert parsed.tzinfo is not None and parsed.hour == 9
        assert CompatREST._parse_dt('2026-07-01').year == 2026

    def test_blank_returns_none(self):
        # empty/hand-edited .clean_slate must not crash list_orders
        assert CompatREST._parse_dt('') is None
        assert CompatREST._parse_dt('   \n') is None

    def test_garbage_warns_and_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING, logger='alpaca_compat'):
            assert CompatREST._parse_dt('not-a-date') is None
        assert any('unparseable' in r.message for r in caplog.records)


class TestRawDelete:
    def _fake(self, calls):
        return SimpleNamespace(_trading=SimpleNamespace(
            close_position=lambda sym: calls.append(sym) or 'closed'))

    def test_positions_path_unquotes_symbol(self):
        calls = []
        # gui close button: quote('BTC/USD', safe='') -> 'BTC%2FUSD'
        assert CompatREST.delete(self._fake(calls),
                                 '/positions/BTC%2FUSD') == 'closed'
        assert calls == ['BTC/USD']

    def test_plain_symbol(self):
        calls = []
        CompatREST.delete(self._fake(calls), '/positions/AAPL')
        assert calls == ['AAPL']

    def test_other_paths_fail_loud(self):
        with pytest.raises(NotImplementedError):
            CompatREST.delete(self._fake([]), '/orders/abc')


def test_compat_source_guards():
    # P1: the resting-stop order type crypto_loop actually submits
    assert "type='stop_limit'" in CRYPTO_SRC          # caller still uses it
    assert "elif type == 'stop_limit':" in COMPAT_SRC
    assert 'StopLimitOrderRequest' in COMPAT_SRC
    # _timeframe now fails loud instead of silently returning hourly bars
    assert 'return TimeFrame.Hour' not in COMPAT_SRC
    assert 'unsupported timeframe' in COMPAT_SRC
    # every **_ignored sink logs what it drops (submit_order, list_orders,
    # get_portfolio_history, get_bars, get_crypto_bars); count signature
    # occurrences only — one comment also mentions **_ignored
    assert COMPAT_SRC.count('**_ignored)') == 5
    assert COMPAT_SRC.count('if _ignored:') == 5
    # docstring names the raw-return exceptions
    doc = ast.get_docstring(ast.parse(COMPAT_SRC))
    for name in ('get_calendar', 'close_all_positions', 'cancel_all_orders'):
        assert name in doc, name


def test_compat_snapshot_methods_exist():
    # GUI markets tab (get_snapshots / get_crypto_snapshots) + manual close
    for method in ('get_snapshots', 'get_crypto_snapshots', 'delete'):
        assert callable(getattr(CompatREST, method)), method


# ===========================================================================
# types_mod
# ===========================================================================

def test_llmresult_removed():
    assert not hasattr(types_mod, 'LLMResult')
    assert 'LLMResult' not in TYPES_SRC
    assert 'LLM results' not in (types_mod.__doc__ or '')


def test_field_import_removed():
    assert re.search(r'\bfield\b', TYPES_SRC) is None
    assert 'from dataclasses import asdict, dataclass' in TYPES_SRC


def test_position_to_dict_is_asdict():
    pos = types_mod.Position(qty=1.0, entry_price=50.0, high_water_mark=52.0,
                             stop_order_id='abc', trailing_activated=True,
                             entry_atr=0.8, take_profit_price=55.0,
                             garch_sigma=0.02)
    d = pos.to_dict()
    assert d == asdict(pos)
    # exact key set + order the hand-rolled dict used to produce
    assert list(d) == ['qty', 'entry_price', 'high_water_mark',
                       'stop_order_id', 'trailing_activated', 'entry_atr',
                       'take_profit_price', 'garch_sigma']
    # tests/test_new_modules.py contract
    assert d['qty'] == 1.0 and d['trailing_activated'] is True


def test_quote_to_dict_is_asdict():
    q = types_mod.Quote(bid=99.0, ask=101.0, spread=2.0, midpoint=100.0,
                        spread_pct=2.0)
    assert q.to_dict() == asdict(q)
    assert q.to_dict()['midpoint'] == 100.0       # test_new_modules contract


def test_deferred_surface_still_present():
    # Quote and MacroRegime.is_defensive deletions were DEFERRED (their
    # removal requires editing tests/test_new_modules.py, outside b03
    # ownership). Guard against half-applied removals.
    assert hasattr(types_mod, 'Quote')
    assert isinstance(types_mod.MacroRegime.is_defensive, property)
