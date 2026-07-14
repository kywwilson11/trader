"""Stage-3 improvement tests for portfolio_backtest.py:

1. conviction_gated fail-closed on NaN conviction fields (signal/meta_p/
   pred_thresh_ratio) so a NaN never sails through a set floor.
2. panel_from_frame's signal_lag path is immune to input row order (stable
   sort by timestamp before the per-ticker shift), so an unsorted frame
   cannot leak a future bar's signal."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_backtest as pb


def _period(*dicts):
    return list(dicts)


def test_nan_meta_p_fails_set_floor():
    cands = _period(
        {'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0, 'meta_p': float('nan')},
        {'symbol': 'B', 'signal': 0.8, 'fwd_return': 1.0, 'meta_p': 0.7},
    )
    policy = pb.conviction_gated(3, meta_floor=0.6)
    admitted = policy(pb._sorted_desc(cands))
    assert [c['symbol'] for c in admitted] == ['B']


def test_nan_signal_and_ratio_fail_set_floors():
    cands_signal = _period(
        {'symbol': 'A', 'signal': float('nan'), 'fwd_return': 1.0},
    )
    policy_signal = pb.conviction_gated(3, signal_floor=0.1)
    admitted_signal = policy_signal(cands_signal)
    assert admitted_signal == []

    cands_ratio = _period(
        {'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0,
         'pred_thresh_ratio': float('nan')},
    )
    policy_ratio = pb.conviction_gated(3, ratio_floor=0.5)
    admitted_ratio = policy_ratio(cands_ratio)
    assert admitted_ratio == []


def test_nan_field_passes_when_floor_unset():
    cands = _period(
        {'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0, 'meta_p': float('nan')},
    )
    policy = pb.conviction_gated(3, signal_floor=0.1)
    admitted = policy(cands)
    assert [c['symbol'] for c in admitted] == ['A']


def test_boundary_value_still_admitted():
    cands = _period(
        {'symbol': 'A', 'signal': 0.9, 'fwd_return': 1.0, 'meta_p': 0.6},
    )
    policy = pb.conviction_gated(3, meta_floor=0.6)
    admitted = policy(cands)
    assert [c['symbol'] for c in admitted] == ['A']


def _lag_frame():
    idx = pd.to_datetime(['2026-01-01 10:00', '2026-01-01 11:00', '2026-01-01 12:00'])
    df = pd.DataFrame({
        'Ticker': ['AAA', 'AAA', 'AAA'],
        'sig': [1.0, 2.0, 3.0],
        'fwd': [0.1, 0.2, 0.3],
    }, index=idx)
    return df


def test_signal_lag_immune_to_row_order():
    sorted_df = _lag_frame()
    shuffled_df = sorted_df.iloc[[2, 0, 1]]

    panel_sorted = pb.panel_from_frame(sorted_df, 'sig', 'fwd', ticker_col='Ticker',
                                        signal_lag=1)
    panel_shuffled = pb.panel_from_frame(shuffled_df, 'sig', 'fwd', ticker_col='Ticker',
                                         signal_lag=1)

    expected = [
        [{'symbol': 'AAA', 'signal': 1.0, 'fwd_return': 0.2}],
        [{'symbol': 'AAA', 'signal': 2.0, 'fwd_return': 0.3}],
    ]

    def _strip(panel):
        return [[{'symbol': c['symbol'], 'signal': c['signal'],
                   'fwd_return': c['fwd_return']} for c in period]
                for period in panel]

    assert _strip(panel_sorted) == expected
    assert _strip(panel_shuffled) == expected


def test_no_lag_path_unchanged():
    shuffled_df = _lag_frame().iloc[[2, 0, 1]]
    panel = pb.panel_from_frame(shuffled_df, 'sig', 'fwd', ticker_col='Ticker',
                                signal_lag=0)

    expected = [
        [{'symbol': 'AAA', 'signal': 1.0, 'fwd_return': 0.1}],
        [{'symbol': 'AAA', 'signal': 2.0, 'fwd_return': 0.2}],
        [{'symbol': 'AAA', 'signal': 3.0, 'fwd_return': 0.3}],
    ]

    def _strip(panel):
        return [[{'symbol': c['symbol'], 'signal': c['signal'],
                   'fwd_return': c['fwd_return']} for c in period]
                for period in panel]

    assert _strip(panel) == expected
