"""Tests for the as-of training universe (tradability + membership masks)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from harvest_stock_data import (_asof_membership_mask,
                                _asof_tradability_mask)


def _tdf(ticker, dv30, days=10, close=50.0, start='2026-01-05'):
    """Hourly frame with a constant _DV30 (post-tradability shape)."""
    idx = pd.date_range(start + ' 14:30', periods=days * 7, freq='h', tz='UTC')
    idx = idx[idx.indexer_between_time('14:30', '20:30')]
    df = pd.DataFrame({'Close': close, 'Volume': 1e6, '_DV30': dv30},
                      index=idx)
    df['Ticker'] = ticker
    return df


class TestMembershipMask:
    def test_keeps_topk_drops_rest(self):
        # 5 names, top_k=3: the two thinnest must be dropped entirely
        frames = [_tdf(f'S{i}', dv30=(i + 1) * 1e7) for i in range(5)]
        df = pd.concat(frames).sort_index()
        out = _asof_membership_mask(df, top_k=3)
        kept = set(out['Ticker'])
        assert kept == {'S2', 'S3', 'S4'}   # the 3 largest DV names

    def test_membership_is_per_day(self):
        # A's volume collapses halfway; B's surges — membership flips
        a1 = _tdf('A', 9e7, days=5, start='2026-01-05')
        a2 = _tdf('A', 1e6, days=5, start='2026-01-12')
        b1 = _tdf('B', 1e6, days=5, start='2026-01-05')
        b2 = _tdf('B', 9e7, days=5, start='2026-01-12')
        c = pd.concat([_tdf('C', 5e7, days=5, start='2026-01-05'),
                       _tdf('C', 5e7, days=5, start='2026-01-12')])
        df = pd.concat([a1, a2, b1, b2, c]).sort_index()
        out = _asof_membership_mask(df, top_k=2)
        week1 = out[out.index < pd.Timestamp('2026-01-12', tz='UTC')]
        week2 = out[out.index >= pd.Timestamp('2026-01-12', tz='UTC')]
        assert set(week1['Ticker']) == {'A', 'C'}
        assert set(week2['Ticker']) == {'B', 'C'}

    def test_late_listing_contributes_no_early_rows(self):
        old = _tdf('OLD', 5e7, days=10, start='2026-01-05')
        new = _tdf('NEW', 9e7, days=5, start='2026-01-12')  # lists later
        df = pd.concat([old, new]).sort_index()
        out = _asof_membership_mask(df, top_k=2)
        new_rows = out[out['Ticker'] == 'NEW']
        assert new_rows.index.min() >= pd.Timestamp('2026-01-12', tz='UTC')
        # OLD keeps its full history (pool smaller than k early on)
        assert len(out[out['Ticker'] == 'OLD']) == len(old)

    def test_small_pool_passes_everything(self):
        df = pd.concat([_tdf('A', 1e7), _tdf('B', 2e7)]).sort_index()
        out = _asof_membership_mask(df, top_k=60)
        assert len(out) == len(df)

    def test_missing_dv_column_is_noop(self):
        df = _tdf('A', 1e7).drop(columns=['_DV30'])
        out = _asof_membership_mask(df, top_k=1)
        assert len(out) == len(df)


class TestTradabilityStampsDV:
    def test_dv30_column_present_after_mask(self):
        idx = pd.date_range('2026-01-05 14:30', periods=24 * 30, freq='h',
                            tz='UTC')
        df = pd.DataFrame({'Close': np.full(len(idx), 50.0),
                           'Volume': np.full(len(idx), 1e6)}, index=idx)
        out = _asof_tradability_mask(df, 'TEST')
        assert '_DV30' in out.columns
        assert (out['_DV30'] > 0).all()
