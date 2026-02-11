"""Tests for indicator_config.py — preset management."""

import pytest

from indicator_config import (
    PRESETS,
    CRYPTO_ONLY_COLS,
    STOCK_ONLY_COLS,
    get_preset_features,
    get_all_preset_info,
)


class TestPresets:
    def test_all_presets_exist(self):
        assert set(PRESETS.keys()) == {"minimal", "standard", "stationary", "full"}

    def test_minimal_subset_of_standard(self):
        minimal = set(PRESETS["minimal"]["features"])
        standard = set(PRESETS["standard"]["features"])
        assert minimal.issubset(standard)

    def test_full_features_is_none(self):
        assert PRESETS["full"]["features"] is None

    def test_each_preset_has_description(self):
        for name, preset in PRESETS.items():
            assert "description" in preset
            assert len(preset["description"]) > 0


class TestGetPresetFeatures:
    def test_minimal_returns_list(self):
        features = get_preset_features("minimal")
        assert isinstance(features, list)
        assert len(features) > 0

    def test_full_returns_none(self):
        assert get_preset_features("full") is None

    def test_unknown_returns_none(self):
        assert get_preset_features("nonexistent") is None


class TestStationaryPreset:
    """Tests for the stationary feature preset."""

    _RAW_PRICE_VOLUME = {"Open", "High", "Low", "Close", "Volume",
                         "SMA_20", "ATR", "OBV", "Volume_SMA_20",
                         "BBL_20_2.0", "BBU_20_2.0", "VWAP"}

    def test_no_raw_price_volume_columns(self):
        features = set(PRESETS["stationary"]["features"])
        overlap = features & self._RAW_PRICE_VOLUME
        assert not overlap, f"Stationary preset contains raw price/volume: {overlap}"

    def test_stationary_features_are_list(self):
        features = PRESETS["stationary"]["features"]
        assert isinstance(features, list)
        assert len(features) > 0

    def test_stationary_subset_of_standard(self):
        """Every stationary feature should exist in either standard or full."""
        standard = set(PRESETS["standard"]["features"])
        stationary = set(PRESETS["stationary"]["features"])
        # All stationary features should be in the standard set
        assert stationary.issubset(standard), (
            f"Stationary features not in standard: {stationary - standard}")


class TestGetAllPresetInfo:
    def test_returns_all_presets(self):
        info = get_all_preset_info()
        assert set(info.keys()) == {"minimal", "standard", "stationary", "full"}

    def test_info_has_count_and_description(self):
        info = get_all_preset_info()
        for name, data in info.items():
            assert "description" in data
            assert "count" in data
            if name != "full":
                assert isinstance(data["count"], int)
                assert data["count"] > 0
            else:
                assert data["count"] is None
