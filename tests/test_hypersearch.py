"""Tests for hypersearch_dual.py — data split and class labeling."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypersearch_dual import get_indices_and_classes


class TestGetIndicesAndClasses:
    def _make_data(self, tickers_n):
        """Create fake returns and boundaries for n tickers with given sizes."""
        all_returns = []
        ticker_boundaries = {}
        tickers = []
        offset = 0
        for i, n in enumerate(tickers_n):
            ticker = f"T{i}"
            tickers.append(ticker)
            # Returns oscillate so we get all 3 classes
            returns = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
            all_returns.append(returns)
            ticker_boundaries[ticker] = (offset, offset + n)
            offset += n
        return np.concatenate(all_returns), np.array(tickers), ticker_boundaries

    def test_returns_three_values(self):
        returns, tickers, bounds = self._make_data([100, 100])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(classes, np.ndarray)

    def test_no_overlap(self):
        """Train and val indices should not overlap."""
        returns, tickers, bounds = self._make_data([200, 150])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        assert len(set(train) & set(val)) == 0

    def test_per_ticker_split(self):
        """Each ticker should contribute to both train and val."""
        returns, tickers, bounds = self._make_data([200, 200, 200])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        train_set = set(train)
        val_set = set(val)
        for ticker in tickers:
            start, end = bounds[ticker]
            ticker_range = set(range(start + 10, end))
            assert len(ticker_range & train_set) > 0, f"{ticker} has no train indices"
            assert len(ticker_range & val_set) > 0, f"{ticker} has no val indices"

    def test_temporal_ordering(self):
        """Within each ticker, train indices should come before val indices."""
        returns, tickers, bounds = self._make_data([300])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        start, end = bounds["T0"]
        # All train indices for this ticker should be < all val indices
        assert max(train) < min(val)

    def test_approximately_80_20_split(self):
        """Split should be close to 80/20 per ticker."""
        returns, tickers, bounds = self._make_data([500])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        total = len(train) + len(val)
        train_ratio = len(train) / total
        assert 0.78 < train_ratio < 0.82

    def test_three_classes_present(self):
        """With oscillating returns and threshold 0.15, all 3 classes should exist."""
        returns, tickers, bounds = self._make_data([500])
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, 10)
        unique = set(classes[train])
        assert 0 in unique  # bear
        assert 1 in unique  # neutral
        assert 2 in unique  # bull

    def test_seq_len_respected(self):
        """No index should be less than start + seq_len for its ticker."""
        returns, tickers, bounds = self._make_data([100, 100])
        seq_len = 24
        train, val, classes = get_indices_and_classes(
            returns, tickers, bounds, 0.15, seq_len)
        all_indices = train + val
        for ticker in tickers:
            start, end = bounds[ticker]
            ticker_indices = [i for i in all_indices if start <= i < end]
            if ticker_indices:
                assert min(ticker_indices) >= start + seq_len
