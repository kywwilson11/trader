"""Review batch b02 — order_utils / execution_policy / order_stream fixes.

Pins the reviewer-approved fixes:
  P0  manage_order_lifecycle no-fallback timeout returns the post-cancel
      order (partial fills survive), so the maker ladder chases only the
      remainder instead of re-buying the full notional.
  P2  market-fallback confirmation returns the FRESHEST fetched state, not
      the submit-time snapshot (slow fills no longer look unacquired).
  P2  start_order_stream check-and-set is serialized (one stream thread
      even under concurrent callers) and fails closed on unset base URL.
  P3  list_orders paging limit, naive-UTC quote timestamps, should_trade
      min_edge deferring to fees.MIN_EDGE_MULTIPLE, reconstruct/verify
      cleanup, NaN spread fail-closed in choose_entry_tactic, reconnect
      backoff reset, docstring rot pins.
"""

import datetime
import math
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fees
import order_stream
import order_utils as ou
from execution_policy import choose_entry_tactic
from order_utils import (cancel_all_open_orders, cancel_orders_for_symbol,
                         get_quote, manage_order_lifecycle, place_maker_buy,
                         reconstruct_positions, should_trade, verify_position)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)


# --- P0: no-fallback timeout keeps the partially-filled order object ---

class _StuckPartialAPI:
    """The realistic GTC bid-join sequence: the order sits partially_filled
    for the whole poll window (no spontaneous terminal state) and only goes
    canceled when WE cancel it — filled_qty survives the cancel."""

    def __init__(self, qty=10.0, filled=6.0, px=100.0, fetch_fail_from=None):
        self.order = SimpleNamespace(id='o1', symbol='BTC/USD', qty=qty,
                                     side='buy', status='partially_filled',
                                     filled_qty=filled, filled_avg_price=px)
        self.canceled = []
        self.submitted = []
        self.get_calls = 0
        self.fetch_fail_from = fetch_fail_from

    def get_order(self, oid):
        self.get_calls += 1
        if self.fetch_fail_from and self.get_calls >= self.fetch_fail_from:
            raise RuntimeError('broker unreachable')
        return self.order

    def cancel_order(self, oid):
        self.canceled.append(oid)
        if self.order.status not in ('filled',):
            self.order.status = 'canceled'

    def submit_order(self, **kw):
        self.submitted.append(kw)
        raise AssertionError('no fallback expected')


class TestLifecycleNoFallbackReturn:
    def test_timeout_returns_canceled_order_with_partial_fill(self):
        api = _StuckPartialAPI()
        result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                        fallback_to_market=False)
        assert result is not None                      # was None before fix
        assert result.status == 'canceled'
        assert float(result.filled_qty) == 6.0         # partial fill visible
        assert 'o1' in api.canceled                    # never left working
        assert api.submitted == []                     # and no fallback

    def test_none_when_post_cancel_fetch_fails(self):
        # Polling succeeds (stuck partial), but the post-cancel fetch dies:
        # per the docstring, None only when state could never be fetched.
        api = _StuckPartialAPI(fetch_fail_from=3)      # 2 polls, then fail
        result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                        fallback_to_market=False)
        assert result is None
        assert 'o1' in api.canceled


# --- P0 blast radius: maker ladder chases ONLY the remainder ---

class _LadderAPI:
    """Rung 1 fills 60% and sits until the timeout-cancel; rung 2 fills."""

    def __init__(self):
        self.submitted = []
        self.orders = {}
        self.canceled = []

    def submit_order(self, **kw):
        oid = f"o{len(self.submitted) + 1}"
        self.submitted.append(kw)
        if len(self.submitted) == 1:
            o = SimpleNamespace(id=oid, symbol=kw['symbol'], qty=kw['qty'],
                                side='buy', status='partially_filled',
                                filled_qty=6.0, filled_avg_price=100.0)
        else:
            o = SimpleNamespace(id=oid, symbol=kw['symbol'], qty=kw['qty'],
                                side='buy', status='filled',
                                filled_qty=kw['qty'], filled_avg_price=100.0)
        self.orders[oid] = o
        return SimpleNamespace(id=oid, status='new')

    def get_order(self, oid):
        return self.orders[oid]

    def cancel_order(self, oid):
        self.canceled.append(oid)
        o = self.orders[oid]
        if o.status not in ('filled',):
            o.status = 'canceled'


class TestMakerLadderPartialAtTimeout:
    QUOTE = {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
             'spread': 0.2, 'spread_pct': 0.2}

    def test_stuck_partial_rung_chases_only_remainder(self):
        api = _LadderAPI()
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000,
                                         lambda: dict(self.QUOTE),
                                         stage_timeout=4)
        assert tactic == 'maker_reprice'
        assert result.status == 'filled'
        # Rung 1 bought $600; rung 2 must chase ~$400, NOT the full $1000
        assert api.submitted[1]['qty'] == pytest.approx(4.0, abs=0.01)
        acquired = sum(float(o.filled_qty) * float(o.filled_avg_price)
                       for o in api.orders.values())
        assert acquired == pytest.approx(1000.0, rel=0.01)  # not 2.2x


# --- P2: market fallback returns the freshest fetched state ---

class _SlowMarketFillAPI:
    """Limit order never fills; market fallback fills slowly (never reaches
    'filled' inside the 3-poll confirmation window)."""

    def __init__(self):
        self.limit = SimpleNamespace(id='o1', symbol='BTC/USD', qty=10.0,
                                     side='buy', status='new',
                                     filled_qty=0, filled_avg_price=None)
        self.fresh = SimpleNamespace(id='o2', symbol='BTC/USD', qty=10.0,
                                     side='buy', status='partially_filled',
                                     filled_qty=3.5, filled_avg_price=100.0)
        self.snapshot = None
        self.canceled = []

    def get_order(self, oid):
        return self.limit if oid == 'o1' else self.fresh

    def cancel_order(self, oid):
        self.canceled.append(oid)
        self.limit.status = 'canceled'

    def submit_order(self, **kw):
        assert kw['type'] == 'market'
        self.snapshot = SimpleNamespace(id='o2', symbol=kw['symbol'],
                                        qty=kw['qty'], side=kw['side'],
                                        status='accepted', filled_qty=0,
                                        filled_avg_price=None)
        return self.snapshot


class TestMarketFallbackFreshState:
    def test_returns_fetched_state_not_submit_snapshot(self):
        api = _SlowMarketFillAPI()
        result = manage_order_lifecycle(api, 'o1', timeout=4, poll_interval=2,
                                        fallback_to_market=True)
        assert result is api.fresh                 # freshest fetched state
        assert result is not api.snapshot          # not filled_qty=0 snapshot
        assert float(result.filled_qty) == 3.5     # caller sees the coins


# --- P3: list_orders paging limit ---

class _LimitRecordingAPI:
    def __init__(self):
        self.limits = []

    def list_orders(self, status='open', limit=None):
        self.limits.append(limit)
        return []


class TestListOrdersLimit:
    def test_cleanup_passes_high_limit(self):
        api = _LimitRecordingAPI()
        cancel_all_open_orders(api)
        assert api.limits == [500]

    def test_cancel_for_symbol_passes_high_limit(self):
        api = _LimitRecordingAPI()
        assert cancel_orders_for_symbol(api, 'BTC/USD') is True
        assert api.limits == [500]

    def test_falls_back_for_shims_without_limit_kwarg(self):
        class _NoLimit:
            calls = 0

            def list_orders(self, status='open'):
                self.calls += 1
                return []

        api = _NoLimit()
        assert cancel_orders_for_symbol(api, 'X') is True
        assert api.calls == 1


# --- P3: naive quote timestamps are UTC, not machine-local ---

class _QuoteAPI:
    def __init__(self, t):
        self.t = t

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: SimpleNamespace(bp=100.0, ap=100.1, t=self.t)}


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class TestQuoteStaleness:
    def test_naive_stale_quote_rejected_regardless_of_local_tz(self):
        # Old code read a naive timestamp as machine-local: west of UTC the
        # age went negative and a 5-minute-stale quote passed the guard.
        naive = (_utc_now() - datetime.timedelta(seconds=300)).replace(tzinfo=None)
        assert get_quote(_QuoteAPI(naive), 'BTC/USD', 'crypto') is None

    def test_naive_fresh_quote_accepted(self):
        # ...and east of UTC every fresh quote would have been rejected.
        naive = (_utc_now() - datetime.timedelta(seconds=10)).replace(tzinfo=None)
        out = get_quote(_QuoteAPI(naive), 'BTC/USD', 'crypto')
        assert out is not None and out['bid'] == 100.0

    def test_aware_staleness_behavior_unchanged(self):
        stale = _utc_now() - datetime.timedelta(seconds=300)
        fresh = _utc_now() - datetime.timedelta(seconds=5)
        assert get_quote(_QuoteAPI(stale), 'BTC/USD', 'crypto') is None
        assert get_quote(_QuoteAPI(fresh), 'BTC/USD', 'crypto') is not None


# --- P3: should_trade default min_edge tracks fees.MIN_EDGE_MULTIPLE ---

class TestShouldTradeDefaultMinEdge:
    def test_default_tracks_fees_module(self, monkeypatch):
        # Baseline: clears the canonical 2x hurdle
        assert should_trade(0.5, 0.05, asset_type='stock') is True
        # Tuning the canonical multiple must move the default-gated callers
        # (the old literal min_edge=2.0 default kept the stale hurdle)
        monkeypatch.setattr(fees, 'MIN_EDGE_MULTIPLE', 50.0)
        assert should_trade(0.5, 0.05, asset_type='stock') is False

    def test_explicit_min_edge_still_honored(self, monkeypatch):
        monkeypatch.setattr(fees, 'MIN_EDGE_MULTIPLE', 50.0)
        assert should_trade(0.5, 0.05, asset_type='stock', min_edge=1.0) is True


# --- P3: reconstruct_positions / verify_position cleanup ---

class _PosAPI:
    def __init__(self, book):
        self.book = book  # spelled symbol -> qty

    def get_position(self, sym):
        if sym not in self.book:
            raise RuntimeError('position does not exist')
        return SimpleNamespace(symbol=sym, qty=self.book[sym],
                               avg_entry_price=100.0, current_price=110.0)


class TestPositionHelpers:
    def test_reconstruct_resolves_slashless_variant_and_shape(self):
        out = reconstruct_positions(_PosAPI({'BTCUSD': 2.0}),
                                    ['BTC/USD', 'ETH/USD'])
        assert set(out) == {'BTC/USD'}
        info = out['BTC/USD']
        # dead stop_order_id / trailing_activated keys are gone
        assert set(info) == {'qty', 'entry_price', 'high_water_mark'}
        assert info['qty'] == 2.0
        assert info['high_water_mark'] == 110.0

    def test_reconstruct_skips_shorts(self):
        assert reconstruct_positions(_PosAPI({'BTCUSD': -2.0}), ['BTC/USD']) == {}

    def test_verify_position_is_long_only(self):
        assert verify_position(_PosAPI({'BTCUSD': -1.0}), 'BTC/USD') is None
        assert verify_position(_PosAPI({'BTCUSD': 0.0}), 'BTC/USD') is None
        pos = verify_position(_PosAPI({'BTCUSD': 3.0}), 'BTC/USD')
        assert pos is not None and float(pos.qty) == 3.0


# --- P3: choose_entry_tactic fails closed on non-finite spread ---

class TestExecutionPolicyNonFinite:
    def test_non_finite_spread_fails_closed_to_cross(self):
        for bad in (float('nan'), float('inf'), float('-inf'), None):
            out = choose_entry_tactic('stock', bad, name_class='mid')
            assert out['tactic'] == 'cross', bad
            assert out['post_offset_pct'] == 0.0
            assert math.isfinite(out['post_offset_pct'])

    def test_finite_bands_unchanged(self):
        assert choose_entry_tactic('stock', 0.10, name_class='mid')['tactic'] == 'ladder'
        out = choose_entry_tactic('crypto', float('nan'))
        assert out['tactic'] == 'post' and out['post_offset_pct'] == 0.0


# --- P3: reconnect backoff resets after a healthy connection ---

class TestStreamBackoff:
    def test_ladder_doubles_to_cap_on_rapid_failures(self):
        b = order_stream._BACKOFF_INITIAL
        sleeps = []
        for _ in range(8):
            s, b = order_stream._next_backoff(b, 0.5)
            sleeps.append(s)
        assert sleeps == [5, 10, 20, 40, 80, 160, 300, 300]

    def test_healthy_connection_resets_ladder(self):
        assert order_stream._next_backoff(300, 3600.0) == (5, 10)
        assert order_stream._next_backoff(80, 61.0) == (5, 10)
        # boundary: exactly the threshold does NOT reset
        assert order_stream._next_backoff(80, 60.0) == (80, 160)


# --- P2/P3: start_order_stream serialization + fail-closed config ---

def _install_fake_alpaca(monkeypatch):
    stream_mod = types.ModuleType('alpaca.trading.stream')

    class FakeTradingStream:
        def __init__(self, key, secret, paper=None):
            pass

        def subscribe_trade_updates(self, cb):
            pass

        def run(self):
            threading.Event().wait()  # park forever (daemon thread)

    stream_mod.TradingStream = FakeTradingStream
    alpaca_mod = types.ModuleType('alpaca')
    trading_mod = types.ModuleType('alpaca.trading')
    alpaca_mod.trading = trading_mod
    trading_mod.stream = stream_mod
    monkeypatch.setitem(sys.modules, 'alpaca', alpaca_mod)
    monkeypatch.setitem(sys.modules, 'alpaca.trading', trading_mod)
    monkeypatch.setitem(sys.modules, 'alpaca.trading.stream', stream_mod)


def _stream_threads():
    return [t for t in threading.enumerate() if t.name == 'order-stream']


class TestStreamStartup:
    def _env(self, monkeypatch, base_url='https://paper-api.alpaca.markets'):
        monkeypatch.setenv('TRADER_ORDER_STREAM', '1')
        monkeypatch.setenv('ALPACA_API_KEY', 'k')
        monkeypatch.setenv('ALPACA_API_SECRET', 's')
        if base_url is None:
            monkeypatch.delenv('ALPACA_BASE_URL', raising=False)
        else:
            monkeypatch.setenv('ALPACA_BASE_URL', base_url)
        monkeypatch.setattr(order_stream, '_started', False)

    def test_concurrent_starts_spawn_exactly_one_thread(self, monkeypatch):
        _install_fake_alpaca(monkeypatch)
        self._env(monkeypatch)
        before = len(_stream_threads())
        n = 8
        barrier = threading.Barrier(n)
        results = []

        def call():
            barrier.wait(timeout=10)
            results.append(order_stream.start_order_stream())

        workers = [threading.Thread(target=call) for _ in range(n)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=10)
        assert results == [True] * n
        assert len(_stream_threads()) - before == 1

    def test_check_and_set_is_serialized_source_pin(self):
        # The behavioral race is timing-dependent; pin the structure too.
        body = (REPO / 'order_stream.py').read_text().split(
            'def start_order_stream', 1)[1]
        assert body.index('with _start_lock:') < body.index('if _started:')

    def test_fails_closed_when_base_url_unset(self, monkeypatch):
        _install_fake_alpaca(monkeypatch)
        self._env(monkeypatch, base_url=None)
        before = len(_stream_threads())
        assert order_stream.start_order_stream() is False
        assert len(_stream_threads()) == before
        assert order_stream._started is False  # a later fixed env may retry

    def test_missing_credentials_warns(self, monkeypatch, caplog):
        _install_fake_alpaca(monkeypatch)
        self._env(monkeypatch)
        monkeypatch.delenv('ALPACA_API_KEY', raising=False)
        monkeypatch.delenv('ALPACA_API_SECRET', raising=False)
        with caplog.at_level('WARNING', logger='order_stream'):
            assert order_stream.start_order_stream() is False
        assert any('ALPACA_API_KEY' in r.message for r in caplog.records)


# --- P3: docstring-rot pins ---

class TestDocPins:
    def test_ioc_docstring_admits_not_wired(self):
        doc = ' '.join(ou.place_marketable_ioc.__doc__.split())
        assert 'NOT YET WIRED' in doc

    def test_execution_policy_unit_misreading_fixed(self):
        src = (REPO / 'execution_policy.py').read_text()
        assert '10%-of-half-spread' not in src
        assert '20%-of-half-spread' in src

    def test_order_stream_docstring_single_connection_note(self):
        assert 'ONE trade_updates connection per account' in order_stream.__doc__

    def test_lifecycle_docstring_matches_return_contract(self):
        doc = manage_order_lifecycle.__doc__
        assert 'canceled order' in doc and 'filled_qty' in doc
