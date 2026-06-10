"""Fault-injection ("chaos") tests for the order seams.

The places trading systems actually lose money are the failure paths:
API timeouts mid-order, cancels that race fills, partial fills at
timeout, dead quote feeds. These tests script those faults against the
real order_utils code and assert the safety contracts:

  - an order is never left working after the code gives up on it
  - market fallbacks chase ONLY the unfilled remainder
  - a cancel that races a fill keeps the fill (no double-buy)
  - flatten failures are REPORTED, not swallowed
  - scoped flattens never touch the other book's positions
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from order_utils import (cancel_orders_for_symbol, emergency_flatten,
                         manage_order_lifecycle, place_maker_buy)


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)


class ChaosAPI:
    """Scriptable broker: per-method fault predicates + canned orders."""

    def __init__(self):
        self.orders: dict[str, SimpleNamespace] = {}
        self.calls = defaultdict(int)
        self.fail: dict[str, callable] = {}   # name -> predicate(call_no)
        self.submitted: list[dict] = []
        self.positions: list[SimpleNamespace] = []
        self.canceled: list[str] = []
        self.cancel_effect = 'canceled'       # or 'filled' (race) or 'ignore'
        self._next_id = 0

    def _chaos(self, name):
        self.calls[name] += 1
        pred = self.fail.get(name)
        if pred and pred(self.calls[name]):
            raise RuntimeError(f'chaos: {name} #{self.calls[name]}')

    def add_order(self, **kw):
        self._next_id += 1
        o = SimpleNamespace(id=f'o{self._next_id}', status='new',
                            filled_qty=0, filled_avg_price=None,
                            symbol='BTC/USD', qty=1.0, side='buy')
        for k, v in kw.items():
            setattr(o, k, v)
        self.orders[o.id] = o
        return o

    # --- broker surface ---

    def submit_order(self, **kw):
        self._chaos('submit_order')
        self.submitted.append(kw)
        o = self.add_order(symbol=kw.get('symbol', 'BTC/USD'),
                           qty=kw.get('qty', 0), side=kw.get('side', 'buy'))
        if kw.get('type') == 'market':
            o.status = 'filled'
            o.filled_qty = kw.get('qty', 0)
            o.filled_avg_price = 100.0
        return o

    def get_order(self, oid):
        self._chaos('get_order')
        return self.orders[oid]

    def cancel_order(self, oid):
        self._chaos('cancel_order')
        self.canceled.append(oid)
        o = self.orders.get(oid)
        if o is None or self.cancel_effect == 'ignore':
            return
        if self.cancel_effect == 'filled':       # cancel raced a fill
            o.status = 'filled'
            o.filled_qty = o.qty
            o.filled_avg_price = 100.0
        elif o.status not in ('filled',):
            o.status = 'canceled'

    def list_orders(self, status='open'):
        self._chaos('list_orders')
        return [o for o in self.orders.values()
                if o.status in ('new', 'partially_filled', 'pending_cancel')]

    def list_positions(self):
        self._chaos('list_positions')
        return self.positions


# --- manage_order_lifecycle ---

class TestLifecycleFaults:
    def test_get_order_errors_cancel_and_give_up(self):
        api = ChaosAPI()
        o = api.add_order()
        api.fail['get_order'] = lambda n: True  # broker unreachable
        result = manage_order_lifecycle(api, o.id, timeout=30, poll_interval=5)
        assert result is None
        assert o.id in api.canceled  # never left working

    def test_timeout_fallback_chases_only_remainder(self):
        api = ChaosAPI()
        o = api.add_order(qty=10.0)
        o.filled_qty = 6.0          # partial fill, then stuck
        result = manage_order_lifecycle(api, o.id, timeout=4, poll_interval=2,
                                        fallback_to_market=True)
        assert o.id in api.canceled
        markets = [s for s in api.submitted if s.get('type') == 'market']
        assert len(markets) == 1
        assert markets[0]['qty'] == pytest.approx(4.0)  # 10 - 6
        assert getattr(result, 'status', None) == 'filled'

    def test_cancel_fill_race_keeps_fill_no_double_buy(self):
        api = ChaosAPI()
        o = api.add_order(qty=10.0)
        api.cancel_effect = 'filled'  # fill lands during the cancel
        result = manage_order_lifecycle(api, o.id, timeout=4, poll_interval=2,
                                        fallback_to_market=True)
        assert result is o and result.status == 'filled'
        assert not [s for s in api.submitted if s.get('type') == 'market']

    def test_no_fallback_when_disabled(self):
        api = ChaosAPI()
        o = api.add_order(qty=10.0)
        manage_order_lifecycle(api, o.id, timeout=4, poll_interval=2,
                               fallback_to_market=False)
        assert o.id in api.canceled
        assert api.submitted == []


# --- place_maker_buy under faults ---

class TestMakerLadderFaults:
    QUOTE = {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
             'spread': 0.2, 'spread_pct': 0.2}

    def test_submit_exception_falls_through_to_marketable(self):
        api = ChaosAPI()
        api.fail['submit_order'] = lambda n: n == 1  # rung 1 explodes
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000,
                                         lambda: dict(self.QUOTE),
                                         stage_timeout=4)
        assert tactic == 'taker_fallback'
        # The fallback limit order exists and got managed
        assert any(s.get('type') == 'limit' for s in api.submitted)

    def test_quote_feed_dies_mid_ladder(self):
        api = ChaosAPI()
        quotes = iter([dict(self.QUOTE)])  # one quote, then feed dies

        def quote_fn():
            return next(quotes, None)

        result, tactic = place_maker_buy(api, 'BTC/USD', 1000, quote_fn,
                                         stage_timeout=4, max_reprices=0)
        assert tactic == 'unfilled'
        # The one placed rung was canceled, not orphaned
        assert api.canceled


# --- cancel_orders_for_symbol ---

class TestCancelFaults:
    def test_list_orders_error_returns_false(self):
        api = ChaosAPI()
        api.fail['list_orders'] = lambda n: True
        assert cancel_orders_for_symbol(api, 'BTC/USD', timeout=1) is False

    def test_uncancelable_order_returns_false(self):
        api = ChaosAPI()
        api.add_order(symbol='BTC/USD')
        api.cancel_effect = 'ignore'   # broker accepts cancel, ignores it
        assert cancel_orders_for_symbol(api, 'BTC/USD', timeout=1,
                                        poll_interval=0.5) is False

    def test_cancel_exception_but_order_gone_returns_true(self):
        api = ChaosAPI()
        o = api.add_order(symbol='BTC/USD')
        api.fail['cancel_order'] = lambda n: True  # 'already canceled' style

        # Re-poll shows it terminal anyway
        real_list = api.list_orders

        def list_then_empty(status='open'):
            o.status = 'canceled'
            return real_list(status)

        api.list_orders = list_then_empty
        assert cancel_orders_for_symbol(api, 'BTC/USD', timeout=2,
                                        poll_interval=0.5) is True


# --- emergency_flatten ---

class TestFlattenFaults:
    def _pos(self, symbol, qty=2.0):
        return SimpleNamespace(symbol=symbol, qty=qty)

    def test_scoped_flatten_never_touches_other_book(self):
        api = ChaosAPI()
        api.positions = [self._pos('BTC/USD'), self._pos('NVDA', 30)]
        failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures == []
        sold = {s['symbol'] for s in api.submitted if s.get('type') == 'market'}
        assert 'NVDA' not in sold
        assert sold & {'BTC/USD', 'BTCUSD', 'BTC-USD'}

    def test_submit_error_reported_as_failure(self):
        api = ChaosAPI()
        api.positions = [self._pos('BTC/USD')]
        api.fail['submit_order'] = lambda n: True
        failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures == ['BTC/USD']

    def test_unfilled_flatten_reported(self):
        api = ChaosAPI()
        api.positions = [self._pos('BTC/USD')]

        # Market orders submit fine but never fill
        real_submit = api.submit_order

        def submit_stuck(**kw):
            o = real_submit(**kw)
            o.status = 'new'
            o.filled_qty = 0
            return o

        api.submit_order = submit_stuck
        failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures == ['BTC/USD']

    def test_list_positions_error_is_loud(self):
        api = ChaosAPI()
        api.fail['list_positions'] = lambda n: True
        failures = emergency_flatten(api, symbols=['BTC/USD'])
        assert failures  # never silently 'all good'
