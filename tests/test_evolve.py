"""Tests for evolve.py — version management and score I/O."""

import json
import os
import pytest
from unittest.mock import patch

from evolve import get_next_version, load_scores, save_scores, MODELS_DIR


class TestGetNextVersion:
    def test_empty_returns_1(self):
        scores = {"bear": {}, "bull": {}}
        assert get_next_version(scores, "bear") == 1

    def test_increments(self):
        scores = {"bear": {"1": {}, "2": {}, "3": {}}}
        assert get_next_version(scores, "bear") == 4

    def test_non_sequential_versions(self):
        scores = {"bear": {"1": {}, "5": {}, "3": {}}}
        assert get_next_version(scores, "bear") == 6

    def test_missing_target_key(self):
        scores = {"bear": {"1": {}}}
        assert get_next_version(scores, "bull") == 1

    def test_string_keys_handled(self):
        scores = {"bull": {"10": {}, "2": {}}}
        assert get_next_version(scores, "bull") == 11


class TestLoadScores:
    def test_missing_file_returns_defaults(self, tmp_path):
        fake_path = str(tmp_path / "scores.json")
        with patch("evolve.SCORES_FILE", fake_path):
            scores = load_scores()
        assert scores == {"bear": {}, "bull": {}, "current_bear": None, "current_bull": None}

    def test_loads_existing_file(self, tmp_path):
        fake_path = str(tmp_path / "scores.json")
        data = {"bear": {"1": {"score": 0.85}}, "bull": {},
                "current_bear": 1, "current_bull": None}
        with open(fake_path, "w") as f:
            json.dump(data, f)
        with patch("evolve.SCORES_FILE", fake_path):
            scores = load_scores()
        assert scores["bear"]["1"]["score"] == 0.85
        assert scores["current_bear"] == 1


class TestSaveScores:
    def test_writes_json(self, tmp_path):
        fake_path = str(tmp_path / "scores.json")
        data = {"bear": {"1": {"score": 0.9}}, "bull": {},
                "current_bear": 1, "current_bull": None}
        with patch("evolve.SCORES_FILE", fake_path):
            save_scores(data)
        with open(fake_path) as f:
            loaded = json.load(f)
        assert loaded["bear"]["1"]["score"] == 0.9

    def test_overwrites_existing(self, tmp_path):
        fake_path = str(tmp_path / "scores.json")
        with open(fake_path, "w") as f:
            json.dump({"old": True}, f)
        with patch("evolve.SCORES_FILE", fake_path):
            save_scores({"bear": {}, "bull": {}})
        with open(fake_path) as f:
            loaded = json.load(f)
        assert "old" not in loaded
