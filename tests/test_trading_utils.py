"""Tests for trading_utils.py — cooldown and model mtime helpers."""

import datetime
import os
import pytest

from trading_utils import cooldown_ok, get_model_mtime


class TestCooldownOk:
    def test_never_traded_returns_true(self):
        assert cooldown_ok({}, "BTC/USD") is True

    def test_recent_trade_returns_false(self):
        last = {
            "BTC/USD": datetime.datetime.now() - datetime.timedelta(minutes=5),
        }
        assert cooldown_ok(last, "BTC/USD", cooldown_minutes=30) is False

    def test_old_trade_returns_true(self):
        last = {
            "BTC/USD": datetime.datetime.now() - datetime.timedelta(minutes=60),
        }
        assert cooldown_ok(last, "BTC/USD", cooldown_minutes=30) is True

    def test_exactly_at_cooldown(self):
        last = {
            "TSLA": datetime.datetime.now() - datetime.timedelta(minutes=30),
        }
        # >= cooldown_minutes * 60, so exactly at boundary should be True
        assert cooldown_ok(last, "TSLA", cooldown_minutes=30) is True

    def test_different_symbols_independent(self):
        last = {
            "BTC/USD": datetime.datetime.now() - datetime.timedelta(minutes=5),
        }
        assert cooldown_ok(last, "ETH/USD") is True

    def test_zero_cooldown_always_ok(self):
        last = {
            "BTC/USD": datetime.datetime.now(),
        }
        assert cooldown_ok(last, "BTC/USD", cooldown_minutes=0) is True


class TestGetModelMtime:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "model.pth"
        f.write_text("test")
        mtime = get_model_mtime(str(f))
        assert mtime > 0

    def test_missing_file(self):
        assert get_model_mtime("/nonexistent/path/model.pth") == 0

    def test_returns_float(self, tmp_path):
        f = tmp_path / "model.pth"
        f.write_text("test")
        assert isinstance(get_model_mtime(str(f)), float)
