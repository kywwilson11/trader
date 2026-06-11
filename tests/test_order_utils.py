"""Tests for order_utils.py — pure computation functions."""

import pytest

from order_utils import compute_limit_price, should_trade


class TestComputeLimitPrice:
    def test_buy_above_midpoint(self):
        quote = {"midpoint": 100.0}
        price = compute_limit_price("buy", quote, offset_bps=5)
        assert price > 100.0

    def test_sell_below_midpoint(self):
        quote = {"midpoint": 100.0}
        price = compute_limit_price("sell", quote, offset_bps=5)
        assert price < 100.0

    def test_offset_magnitude(self):
        quote = {"midpoint": 10000.0}
        buy_price = compute_limit_price("buy", quote, offset_bps=10)
        # 10 bps = 0.1% = $10 offset on $10000
        expected = 10000.0 + 10000.0 * (10 / 10000.0)
        assert abs(buy_price - expected) < 0.01

    def test_zero_offset(self):
        quote = {"midpoint": 50.0}
        buy_price = compute_limit_price("buy", quote, offset_bps=0)
        sell_price = compute_limit_price("sell", quote, offset_bps=0)
        assert buy_price == 50.0
        assert sell_price == 50.0

    def test_rounds_to_four_decimals(self):
        quote = {"midpoint": 3.33333333}
        price = compute_limit_price("buy", quote, offset_bps=7)
        # Should be rounded to 4 decimal places
        assert price == round(price, 4)


class TestShouldTrade:
    """should_trade now prices the FULL round trip: venue fees + spread.

    Crypto taker round trip = 2 x 25bps + spread; the old gate compared
    against spread alone, admitting trades whose entire predicted move
    was smaller than the Alpaca fee bill.
    """

    def test_crypto_must_clear_fee_hurdle(self):
        # crypto taker RT = 0.50% + 0.1% spread = 0.60%; min_edge 2 -> 1.2%
        assert should_trade(predicted_return=2.0, spread_pct=0.1,
                            asset_type='crypto') is True
        assert should_trade(predicted_return=1.0, spread_pct=0.1,
                            asset_type='crypto') is False

    def test_stock_hurdle_is_much_lower(self):
        # stock RT ~= 0.063% + 0.05% spread = 0.113%; min_edge 2 -> ~0.226%
        assert should_trade(predicted_return=0.5, spread_pct=0.05,
                            asset_type='stock') is True
        assert should_trade(predicted_return=0.2, spread_pct=0.05,
                            asset_type='stock') is False

    def test_maker_pricing_lowers_crypto_hurdle(self):
        # Exits are market/stop -> always taker. maker entry RT =
        # (15+25)bps + 0.1% spread = 0.50%; min_edge 2 -> 1.0%.
        # taker entry RT = (25+25)bps + 0.1% = 0.60% -> 1.2%.
        assert should_trade(predicted_return=1.1, spread_pct=0.1,
                            asset_type='crypto', maker=True) is True
        assert should_trade(predicted_return=1.1, spread_pct=0.1,
                            asset_type='crypto', maker=False) is False

    def test_negative_return_uses_abs(self):
        assert should_trade(predicted_return=-2.0, spread_pct=0.1,
                            asset_type='crypto') is True

    def test_custom_min_edge(self):
        # stock hurdle at min_edge=1 is ~0.113%; at 10 it's ~1.13%
        assert should_trade(predicted_return=0.5, spread_pct=0.05,
                            asset_type='stock', min_edge=10.0) is False
        assert should_trade(predicted_return=0.5, spread_pct=0.05,
                            asset_type='stock', min_edge=1.0) is True


class _FakeOrder:
    def __init__(self, id, status='new', filled_qty=0, filled_avg_price=None,
                 symbol='BTC/USD', qty=1.0, side='buy'):
        self.id = id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.symbol = symbol
        self.qty = qty
        self.side = side


class _FakeAPI:
    """Scripted API: each submit_order returns the next scripted order;
    get_order returns its (possibly mutated) terminal state."""

    def __init__(self, script):
        self.script = list(script)   # list of _FakeOrder terminal states
        self.submitted = []          # kwargs of every submit_order call
        self.canceled = []
        self._orders = {}

    def submit_order(self, **kw):
        terminal = self.script.pop(0)
        self.submitted.append(kw)
        self._orders[terminal.id] = terminal
        return _FakeOrder(terminal.id, status='new',
                          symbol=kw.get('symbol', 'BTC/USD'),
                          qty=kw.get('qty', 0), side=kw.get('side', 'buy'))

    def get_order(self, order_id):
        return self._orders[order_id]

    def cancel_order(self, order_id):
        self.canceled.append(order_id)


class TestPlaceMakerBuy:
    QUOTE = {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
             'spread': 0.2, 'spread_pct': 0.2}

    def _run(self, api, monkeypatch, notional=1000):
        from order_utils import place_maker_buy
        import time as _t
        monkeypatch.setattr(_t, 'sleep', lambda s: None)
        return place_maker_buy(api, 'BTC/USD', notional,
                               lambda: dict(self.QUOTE), stage_timeout=4)

    def test_first_rung_fill_is_maker(self, monkeypatch):
        api = _FakeAPI([_FakeOrder('o1', 'filled', 10.0, 100.0)])
        result, tactic = self._run(api, monkeypatch)
        assert tactic == 'maker'
        assert result.status == 'filled'
        # Bid-join: limit placed AT the bid, never above
        assert api.submitted[0]['limit_price'] == 100.0
        assert api.submitted[0]['type'] == 'limit'

    def test_reprice_then_fill(self, monkeypatch):
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 0, None),       # rung 1 times out
            _FakeOrder('o2', 'filled', 10.0, 100.0),     # rung 2 fills
        ])
        result, tactic = self._run(api, monkeypatch)
        assert tactic == 'maker_reprice'
        assert result.status == 'filled'
        assert len(api.submitted) == 2

    def test_taker_fallback_after_rungs_fail(self, monkeypatch):
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 0, None),
            _FakeOrder('o2', 'canceled', 0, None),
            _FakeOrder('o3', 'filled', 9.98, 100.15),    # marketable fallback
        ])
        result, tactic = self._run(api, monkeypatch)
        assert tactic == 'taker_fallback'
        assert result.status == 'filled'
        # Fallback limit crosses toward the ask (mid + offset), above bid
        assert api.submitted[2]['limit_price'] > 100.0

    def test_partial_fill_chases_only_remainder(self, monkeypatch):
        # Rung 1 fills 60% then times out; rung 2 must chase ~40%
        api = _FakeAPI([
            _FakeOrder('o1', 'canceled', 6.0, 100.0),    # $600 of $1000
            _FakeOrder('o2', 'filled', 4.0, 100.0),
        ])
        result, tactic = self._run(api, monkeypatch)
        assert tactic == 'maker_reprice'
        qty2 = api.submitted[1]['qty']
        assert abs(qty2 - 4.0) < 0.01  # ~$400 remainder at bid=100

    def test_partial_to_dust_counts_as_maker(self, monkeypatch):
        # 99.5% filled -> remainder is dust, no chase, judged by qty
        api = _FakeAPI([_FakeOrder('o1', 'canceled', 9.95, 100.0)])
        result, tactic = self._run(api, monkeypatch)
        assert tactic == 'maker_partial'
        assert float(result.filled_qty) > 0
        assert len(api.submitted) == 1

    def test_no_quote_returns_unfilled(self, monkeypatch):
        from order_utils import place_maker_buy
        import time as _t
        monkeypatch.setattr(_t, 'sleep', lambda s: None)
        api = _FakeAPI([])
        result, tactic = place_maker_buy(api, 'BTC/USD', 1000, lambda: None,
                                         stage_timeout=4)
        assert result is None and tactic == 'unfilled'
        assert api.submitted == []
