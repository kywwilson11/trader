"""Tests for indicator_config.py behavior-neutral improvements (2026-07):
atomic save, defensive-copy get_preset_features, and existing invariants
(dup-free presets, disjoint only-cols, lean exclusion) exercised against a
tmp_path-patched _FILE so the repo's real gitignored indicator_config.json
is never touched.
"""

import json

import pytest

import indicator_config
from indicator_config import (
    CRYPTO_ONLY_COLS,
    PRESETS,
    STOCK_ONLY_COLS,
    _LEAN_EXCLUDE,
    get_preset_features,
    load_indicator_config,
    save_indicator_config,
)


@pytest.fixture
def patched_file(tmp_path, monkeypatch):
    target = tmp_path / "indicator_config.json"
    monkeypatch.setattr(indicator_config, "_FILE", target)
    return target


class TestAtomicSave:
    def test_save_load_roundtrip(self, patched_file):
        save_indicator_config({"preset": "stationary"})
        loaded = load_indicator_config()
        assert loaded == {"preset": "stationary"}
        # file contents are valid JSON
        with open(patched_file) as f:
            on_disk = json.load(f)
        assert on_disk == {"preset": "stationary"}

    def test_save_is_atomic_no_leftover_tmp(self, patched_file):
        save_indicator_config({"preset": "minimal"})
        entries = list(patched_file.parent.iterdir())
        assert entries == [patched_file]

    def test_save_failure_cleans_tmp_and_raises(self, patched_file):
        # pre-existing valid content that must survive a failed save
        save_indicator_config({"preset": "standard"})
        with open(patched_file) as f:
            before = f.read()

        with pytest.raises(TypeError):
            save_indicator_config({"preset": object()})

        entries = list(patched_file.parent.iterdir())
        assert entries == [patched_file], "no *.tmp stragglers should remain"
        with open(patched_file) as f:
            after = f.read()
        assert after == before, "failed save must not touch existing content"


class TestGetPresetFeaturesCopy:
    def test_get_preset_features_returns_copy(self):
        f = get_preset_features("minimal")
        f.append("BOGUS")
        assert "BOGUS" not in PRESETS["minimal"]["features"]
        f2 = get_preset_features("minimal")
        assert "BOGUS" not in f2

    def test_full_and_unknown_contract_unchanged(self):
        assert get_preset_features("full") is None
        assert get_preset_features("nope") is None


class TestLoadFallbacks:
    def test_load_corrupt_file_falls_back(self, patched_file):
        patched_file.write_text("not json{{")
        assert load_indicator_config() == {"preset": "standard"}

    def test_load_unknown_or_unhashable_preset_falls_back(self, patched_file):
        patched_file.write_text(json.dumps({"preset": "bogus"}))
        assert load_indicator_config() == {"preset": "standard"}

        patched_file.write_text(json.dumps({"preset": ["x"]}))
        assert load_indicator_config() == {"preset": "standard"}

    def test_load_undecodable_binary_file_falls_back(self, patched_file):
        # UnicodeDecodeError path: ValueError but not JSONDecodeError
        patched_file.write_bytes(b"\xff\xfe\x00garbage")
        assert load_indicator_config() == {"preset": "standard"}


class TestPresetInvariants:
    def test_preset_lists_duplicate_free(self):
        for name, preset in PRESETS.items():
            features = preset["features"]
            if features is None:
                continue
            assert len(features) == len(set(features)), (
                f"preset {name!r} has duplicate features")

    def test_only_cols_disjoint_and_duplicate_free(self):
        crypto = set(CRYPTO_ONLY_COLS)
        stock = set(STOCK_ONLY_COLS)
        assert crypto & stock == set()
        assert len(CRYPTO_ONLY_COLS) == len(crypto)
        assert len(STOCK_ONLY_COLS) == len(stock)

    def test_lean_exclusion_invariant(self):
        stationary = set(PRESETS["stationary"]["features"])
        lean = set(PRESETS["stationary_lean"]["features"])
        assert _LEAN_EXCLUDE <= stationary
        assert stationary - lean == _LEAN_EXCLUDE
