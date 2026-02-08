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
    def test_high_return_low_spread(self):
        assert should_trade(predicted_return=1.0, spread_pct=0.1) is True

    def test_low_return_high_spread(self):
        assert should_trade(predicted_return=0.1, spread_pct=0.5) is False

    def test_exact_threshold(self):
        # min_edge=2.0, spread=0.1 → threshold = 0.2
        # predicted_return must exceed (not equal) threshold
        assert should_trade(predicted_return=0.2, spread_pct=0.1) is False
        assert should_trade(predicted_return=0.21, spread_pct=0.1) is True

    def test_negative_return_uses_abs(self):
        # abs(-1.0) = 1.0 > 0.1 * 2.0 = 0.2
        assert should_trade(predicted_return=-1.0, spread_pct=0.1) is True

    def test_custom_min_edge(self):
        assert should_trade(predicted_return=0.5, spread_pct=0.1, min_edge=10.0) is False
        assert should_trade(predicted_return=0.5, spread_pct=0.1, min_edge=1.0) is True
