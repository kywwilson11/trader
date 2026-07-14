"""Tests for the 2026-07 harvest_stock_data cleanup (spec stage 3)."""
import sys, types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
try:
    import dotenv  # noqa: F401
except ImportError:  # dev-Mac: stub load_dotenv so the module imports
    _m = types.ModuleType('dotenv')
    _m.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _m
import harvest_stock_data as h


def _tdf(ticker, dv30, days=10, close=50.0, start='2026-01-05'):
    """Hourly frame with a constant _DV30 (post-tradability shape).

    Copied locally from tests/test_asof_universe.py::_tdf per spec —
    do not import from another test module.
    """
    idx = pd.date_range(start + ' 14:30', periods=days * 7, freq='h', tz='UTC')
    idx = idx[idx.indexer_between_time('14:30', '20:30')]
    df = pd.DataFrame({'Close': close, 'Volume': 1e6, '_DV30': dv30},
                      index=idx)
    df['Ticker'] = ticker
    return df


def test_import_smoke():
    assert callable(h.main)


def test_time_import_removed():
    assert not hasattr(h, 'time')


def test_summary_split_excludes_tb_labels():
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR',
            'Eff_Spread_Pct', 'Ticker',
            'Target_Return_4', 'Target_Return',
            'TB_Ret_4', 'TB_Bars_4', 'TB_Reason_4']
    fc, tcols, tbcols = h._summary_feature_split(cols)
    assert fc == 7  # OHLCV(5) + ATR + Eff_Spread_Pct; Ticker + targets + TB_* excluded
    assert tcols == ['Target_Return_4', 'Target_Return']
    assert tbcols == ['TB_Ret_4', 'TB_Bars_4', 'TB_Reason_4']


def test_summary_split_no_tb_columns():
    cols = ['Close', 'Volume', 'Ticker']
    fc, tcols, tbcols = h._summary_feature_split(cols)
    assert fc == 2
    assert tcols == []
    assert tbcols == []


def test_membership_mask_no_shadowed_pandas():
    frames = [_tdf('BIG', dv30=9e7), _tdf('SMALL', dv30=1e6)]
    df = pd.concat(frames).sort_index()
    out = h._asof_membership_mask(df, top_k=1)
    assert set(out['Ticker']) == {'BIG'}
