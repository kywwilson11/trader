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
        # maker RT = 0.30% + 0.1% = 0.40%; min_edge 2 -> 0.8%
        assert should_trade(predicted_return=1.0, spread_pct=0.1,
                            asset_type='crypto', maker=True) is True
        assert should_trade(predicted_return=1.0, spread_pct=0.1,
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
