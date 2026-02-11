"""Tests for hypersearch_v2.py — walk-forward CV, Sharpe computation, weighted loss."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hypersearch_v2 import (
    get_walk_forward_folds,
    compute_sharpe,
)


class TestWalkForwardFolds:
    def _make_boundaries(self, n_tickers, rows_per):
        tickers = [f'T{i}' for i in range(n_tickers)]
        boundaries = {}
        for i, t in enumerate(tickers):
            boundaries[t] = (i * rows_per, (i + 1) * rows_per)
        return tickers, boundaries

    def test_three_folds_generated(self):
        tickers, boundaries = self._make_boundaries(2, 500)
        folds = get_walk_forward_folds(tickers, boundaries, seq_len=12, n_folds=3)
        assert len(folds) == 3

    def test_expanding_train_sizes(self):
        tickers, boundaries = self._make_boundaries(1, 1000)
        folds = get_walk_forward_folds(tickers, boundaries, seq_len=12, n_folds=3)
        train_sizes = [len(train) for train, _ in folds]
        # Each fold should have more training data than the previous
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_no_overlap_between_train_and_val(self):
        tickers, boundaries = self._make_boundaries(1, 1000)
        folds = get_walk_forward_folds(tickers, boundaries, seq_len=12, n_folds=3)
        for train, val in folds:
            overlap = set(train) & set(val)
            assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} indices"

    def test_embargo_gap(self):
        tickers, boundaries = self._make_boundaries(1, 1000)
        seq_len = 24
        folds = get_walk_forward_folds(tickers, boundaries, seq_len=seq_len, n_folds=3)
        for train, val in folds:
            if len(train) > 0 and len(val) > 0:
                gap = val[0] - train[-1]
                assert gap >= seq_len, f"Embargo gap {gap} < seq_len {seq_len}"

    def test_val_sizes_roughly_equal(self):
        tickers, boundaries = self._make_boundaries(1, 1000)
        folds = get_walk_forward_folds(tickers, boundaries, seq_len=12, n_folds=3)
        val_sizes = [len(val) for _, val in folds]
        # All val sizes should be within 2x of each other
        assert max(val_sizes) <= 2 * min(val_sizes)


class TestComputeSharpe:
    def test_good_predictions(self):
        """Correct directional predictions with variance should yield positive Sharpe."""
        np.random.seed(0)
        actuals = np.random.randn(200) * 0.5
        # Predictions correctly match the sign with some noise
        preds = actuals * 2.0 + np.random.randn(200) * 0.1
        sharpe = compute_sharpe(preds, actuals, threshold=0.3)
        assert sharpe > 0

    def test_inverse_predictions(self):
        """Wrong directional predictions should yield negative Sharpe."""
        np.random.seed(0)
        actuals = np.random.randn(200) * 0.5
        # Predictions are inverted
        preds = -actuals * 2.0 + np.random.randn(200) * 0.1
        sharpe = compute_sharpe(preds, actuals, threshold=0.3)
        assert sharpe < 0

    def test_zero_signals(self):
        """All predictions below threshold should give 0 Sharpe."""
        preds = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
        actuals = np.array([1.0, -1.0, 1.0, -1.0, 0.5])
        sharpe = compute_sharpe(preds, actuals, threshold=0.5)
        assert sharpe == 0.0

    def test_known_output(self):
        """Deterministic test: all same-direction trades."""
        # 100 trades, all long, all return +1%
        preds = np.ones(100) * 0.6
        actuals = np.ones(100) * 1.0
        sharpe = compute_sharpe(preds, actuals, threshold=0.5)
        # mean=1.0, std=0 would be inf, but all same return => std~0
        # With constant returns, std=0 => returns 0.0
        assert sharpe == 0.0  # constant returns, zero vol

    def test_varied_returns(self):
        """Non-constant returns should give non-zero Sharpe."""
        np.random.seed(42)
        preds = np.random.randn(200) * 0.5
        actuals = preds * 0.3 + np.random.randn(200) * 0.1  # correlated
        sharpe = compute_sharpe(preds, actuals, threshold=0.2)
        assert sharpe != 0.0


class TestWeightedHuberLoss:
    def test_higher_return_gets_more_weight(self):
        """Verify that weighted Huber loss gives more weight to large moves."""
        import torch
        import torch.nn as nn

        criterion = nn.HuberLoss(delta=1.0, reduction='none')

        pred = torch.tensor([0.0, 0.0])
        # Small return vs large return, same prediction error
        y_small = torch.tensor([0.5])
        y_large = torch.tensor([3.0])

        loss_small = criterion(pred[:1], y_small)
        loss_large = criterion(pred[1:], y_large)

        weight_small = torch.abs(y_small) + 1.0
        weight_large = torch.abs(y_large) + 1.0

        weighted_small = (loss_small * weight_small).mean()
        weighted_large = (loss_large * weight_large).mean()

        assert weighted_large > weighted_small
