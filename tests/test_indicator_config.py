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
    def test_three_presets_exist(self):
        assert set(PRESETS.keys()) == {"minimal", "standard", "full"}

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


class TestGetAllPresetInfo:
    def test_returns_all_presets(self):
        info = get_all_preset_info()
        assert set(info.keys()) == {"minimal", "standard", "full"}

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
