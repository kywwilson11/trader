"""Tests for run_pipeline.py — schedule and phase-building logic."""

import datetime
import os
import pytest
from unittest.mock import patch

from run_pipeline import _next_retrain_time, _build_harvest_phases, BASE_DIR


class TestNextRetrainTime:
    def test_future_same_week(self):
        # If today is Monday, ask for Saturday (day=5)
        now = datetime.datetime(2025, 6, 2, 10, 0)  # Monday 10 AM
        with patch("run_pipeline.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timedelta = datetime.timedelta
            result = _next_retrain_time(5, 2)  # Saturday 2 AM
        assert result.weekday() == 5
        assert result.hour == 2
        assert result > now

    def test_already_past_this_week(self):
        # If today is Sunday, and retrain is Saturday, it should be next Saturday
        now = datetime.datetime(2025, 6, 8, 10, 0)  # Sunday 10 AM
        with patch("run_pipeline.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timedelta = datetime.timedelta
            result = _next_retrain_time(5, 2)  # Saturday 2 AM
        assert result.weekday() == 5
        assert result > now
        assert (result - now).days >= 5  # next Sat is 6 days away

    def test_same_day_before_hour(self):
        now = datetime.datetime(2025, 6, 7, 0, 0)  # Saturday midnight
        with patch("run_pipeline.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timedelta = datetime.timedelta
            result = _next_retrain_time(5, 2)  # Saturday 2 AM
        assert result.day == now.day  # same day
        assert result.hour == 2

    def test_same_day_after_hour(self):
        now = datetime.datetime(2025, 6, 7, 15, 0)  # Saturday 3 PM
        with patch("run_pipeline.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = now
            mock_dt.timedelta = datetime.timedelta
            result = _next_retrain_time(5, 2)  # Saturday 2 AM
        assert result > now
        assert (result - now).days >= 6  # should be next Saturday


class TestBuildHarvestPhases:
    def test_skip_harvest(self):
        phases = _build_harvest_phases(skip_harvest=True, train_crypto=True, train_stock=True)
        assert phases == []

    def test_both_crypto_and_stock_no_csv(self, tmp_path):
        # With no CSVs, both phases should be generated
        with patch("run_pipeline.BASE_DIR", str(tmp_path)):
            phases = _build_harvest_phases(False, True, True)
        ids = [p["id"] for p in phases]
        assert "crypto_harvest" in ids
        assert "stock_harvest" in ids

    def test_crypto_only(self, tmp_path):
        with patch("run_pipeline.BASE_DIR", str(tmp_path)):
            phases = _build_harvest_phases(False, True, False)
        ids = [p["id"] for p in phases]
        assert "crypto_harvest" in ids
        assert "stock_harvest" not in ids

    def test_stock_only(self, tmp_path):
        with patch("run_pipeline.BASE_DIR", str(tmp_path)):
            phases = _build_harvest_phases(False, False, True)
        ids = [p["id"] for p in phases]
        assert "stock_harvest" in ids
        assert "crypto_harvest" not in ids

    def test_fresh_data_skipped(self, tmp_path):
        # Create a recent CSV
        csv = tmp_path / "training_data.csv"
        csv.write_text("test")
        with patch("run_pipeline.BASE_DIR", str(tmp_path)):
            phases = _build_harvest_phases(False, True, False)
        # Fresh data should be skipped
        assert len(phases) == 0
