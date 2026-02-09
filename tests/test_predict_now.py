"""Tests for predict_now.py — path generation utility."""

import pytest

from predict_now import _prefixed_paths


class TestPrefixedPaths:
    def test_no_prefix(self):
        paths = _prefixed_paths("")
        assert paths["bear_model"] == "bear_model.pth"
        assert paths["bear_config"] == "bear_config.pkl"
        assert paths["bull_model"] == "bull_model.pth"
        assert paths["scaler_X"] == "scaler_X.pkl"
        assert paths["feature_cols"] == "feature_cols.pkl"

    def test_stock_prefix(self):
        paths = _prefixed_paths("stock")
        assert paths["bear_model"] == "stock_bear_model.pth"
        assert paths["bear_config"] == "stock_bear_config.pkl"
        assert paths["bull_model"] == "stock_bull_model.pth"
        assert paths["scaler_X"] == "stock_scaler_X.pkl"

    def test_default_model_path_unchanged(self):
        paths = _prefixed_paths("stock")
        assert paths["default_model"] == "stock_predictor.pth"

    def test_returns_all_required_keys(self):
        paths = _prefixed_paths("test")
        required = {"bear_model", "bear_config", "bull_model", "bull_config",
                     "scaler_X", "feature_cols", "default_model", "default_config"}
        assert required.issubset(paths.keys())
