"""Tests for hypersearch_v2.py — walk-forward CV, Sharpe computation, weighted loss."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hypersearch_v2 import (
    get_walk_forward_folds,
    get_holdout_boundary,
    get_holdout_indices,
    compute_sharpe,
    simulate_trades,
    FORWARD_BARS,
)


def _make_time_dataset(n_tickers=2, hours=24 * 365 * 2):
    """Synthetic multi-ticker hourly dataset with the time arrays the
    time-based fold builder requires."""
    base_times = np.arange(hours, dtype=np.int64) * 3600 + 1_600_000_000
    tickers = [f'T{i}' for i in range(n_tickers)]
    boundaries = {t: (i * hours, (i + 1) * hours) for i, t in enumerate(tickers)}
    all_times = np.concatenate([base_times] * n_tickers)
    max_fb = max(FORWARD_BARS)
    label_idx = np.minimum(np.arange(hours) + max_fb, hours - 1)
    all_label_times = np.concatenate([base_times[label_idx]] * n_tickers)
    return tickers, boundaries, all_times, all_label_times


class TestWalkForwardFolds:
    """Folds split by CALENDAR TIME with purge + embargo.

    The old positional split trained on the first tickers' entire history
    and validated on other tickers over the SAME calendar window — with 6
    BTC-correlated cryptos that was near-direct leakage.
    """

    def test_three_folds_generated(self):
        tickers, boundaries, t, lt = _make_time_dataset()
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=12, n_folds=3)
        assert len(folds) == 3

    def test_expanding_train_sizes(self):
        tickers, boundaries, t, lt = _make_time_dataset()
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=12, n_folds=3)
        train_sizes = [len(train) for train, _ in folds]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_val_strictly_after_train_in_time(self):
        tickers, boundaries, t, lt = _make_time_dataset()
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=12, n_folds=3)
        for train, val in folds:
            assert t[train].max() < t[val].min(), "train overlaps val in time"

    def test_purge_no_label_crosses_boundary(self):
        """No train row's LABEL window may complete inside the val period."""
        tickers, boundaries, t, lt = _make_time_dataset()
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=24, n_folds=3)
        for train, val in folds:
            assert lt[train].max() <= t[val].min(), "label leakage across boundary"

    def test_both_tickers_in_train_and_val(self):
        """Cross-sectional: every ticker appears on BOTH sides of the split
        (the old splitter put whole tickers on one side)."""
        tickers, boundaries, t, lt = _make_time_dataset(n_tickers=2)
        hours = boundaries[tickers[0]][1]
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=12, n_folds=3)
        for train, val in folds:
            assert (train < hours).any() and (train >= hours).any()
            assert (val < hours).any() and (val >= hours).any()

    def test_nothing_touches_holdout(self):
        tickers, boundaries, t, lt = _make_time_dataset()
        boundary = get_holdout_boundary(t)
        folds = get_walk_forward_folds(t, lt, tickers, boundaries,
                                       seq_len=12, n_folds=3)
        for train, val in folds:
            assert t[val].max() <= boundary
            assert t[train].max() <= boundary

    def test_holdout_after_everything(self):
        tickers, boundaries, t, lt = _make_time_dataset()
        boundary = get_holdout_boundary(t)
        ho = get_holdout_indices(t, tickers, boundaries, seq_len=12)
        assert len(ho) > 0
        assert t[ho].min() > boundary


class TestComputeSharpe:
    """Sharpe now simulates NON-OVERLAPPING holds net of REAL fees.

    The old simulator counted every signal bar as an independent trade
    earning the full overlapping fb-bar return (inflating scores ~sqrt(fb))
    and charged 5bps round trip vs ~60bps crypto reality.
    """

    def test_good_predictions(self):
        """Directional edge much larger than costs -> positive Sharpe."""
        np.random.seed(0)
        n = 1000
        preds = np.random.randn(n)
        # 3% moves aligned with the prediction sign — clears 0.6% costs
        actuals = np.sign(preds) * 3.0 + np.random.randn(n) * 1.0
        sharpe = compute_sharpe(preds, actuals, threshold=0.3,
                                forward_bars=24, asset_type='crypto')
        assert sharpe > 0

    def test_inverse_predictions(self):
        """Wrong directional predictions should yield negative Sharpe."""
        np.random.seed(0)
        n = 1000
        preds = np.random.randn(n)
        actuals = -np.sign(preds) * 3.0 + np.random.randn(n) * 1.0
        sharpe = compute_sharpe(preds, actuals, threshold=0.3,
                                forward_bars=24, asset_type='crypto')
        assert sharpe < 0

    def test_costs_make_noise_negative(self):
        """Random predictions must look DECISIVELY unprofitable once real
        fees are charged — the entire point of the cost-aware objective."""
        rng = np.random.default_rng(1)
        preds = rng.normal(0, 1, 5000)
        actuals = rng.normal(0, 2, 5000)
        sharpe = compute_sharpe(preds, actuals, threshold=0.5,
                                forward_bars=24, asset_type='crypto')
        assert sharpe < -1.0

    def test_zero_signals(self):
        """All predictions below threshold should give 0 Sharpe."""
        preds = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
        actuals = np.array([1.0, -1.0, 1.0, -1.0, 0.5])
        sharpe = compute_sharpe(preds, actuals, threshold=0.5)
        assert sharpe == 0.0

    def test_constant_returns_zero_vol(self):
        preds = np.ones(100) * 0.6
        actuals = np.ones(100) * 1.0
        sharpe = compute_sharpe(preds, actuals, threshold=0.5)
        assert sharpe == 0.0  # constant returns, zero vol

    def test_holds_are_non_overlapping(self):
        """A continuous signal across n bars opens ceil(n/fb) trades, not n."""
        n, fb = 1000, 24
        preds = np.full(n, 5.0)
        actuals = np.full(n, 2.0)
        trades = simulate_trades(preds, actuals, threshold=0.5,
                                 forward_bars=fb, txn_cost_pct=0.6)
        assert len(trades) == int(np.ceil(n / fb))
        assert np.allclose(trades, 2.0 - 0.6)

    def test_annualization_differs_by_asset(self):
        """Same trades annualize higher for 24/7 crypto (8760 bars/yr) than
        for stocks (1638 RTH bars/yr) — the old code applied the stock
        constant to crypto."""
        rng = np.random.default_rng(7)
        preds = rng.normal(0, 1, 4000)
        actuals = np.sign(preds) * 2.0 + rng.normal(0, 1.5, 4000)
        s_c = compute_sharpe(preds, actuals, 0.5, 24, 'crypto')
        s_s = compute_sharpe(preds, actuals, 0.5, 24, 'stock')
        assert s_c > s_s > 0


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
