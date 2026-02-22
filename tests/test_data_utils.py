"""Tests for data_utils.py — Parquet I/O, CSV fallback, append, validation."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Patch _BASE_DIR to use temp dir for all tests
_TEMP_DIR = None


@pytest.fixture(autouse=True)
def temp_base_dir(tmp_path):
    """Redirect data_utils to use a temp directory."""
    global _TEMP_DIR
    _TEMP_DIR = tmp_path
    with mock.patch('data_utils._BASE_DIR', tmp_path):
        yield tmp_path


def _make_df(n=100, tickers=('BTC-USD', 'ETH-USD')):
    """Create a sample training DataFrame."""
    dates = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    rows = []
    for t in tickers:
        for i, dt in enumerate(dates):
            rows.append({
                'Open': 100 + i * 0.1,
                'High': 101 + i * 0.1,
                'Low': 99 + i * 0.1,
                'Close': 100.5 + i * 0.1,
                'Volume': 1000 + i,
                'RSI': 50 + (i % 30),
                'Target_Return_12': 0.5 + i * 0.01,
                'Ticker': t,
            })
    df = pd.DataFrame(rows)
    idx = pd.DatetimeIndex(
        pd.date_range('2024-01-01', periods=len(df), freq='h', tz='UTC'),
        name=None
    )
    df.index = idx[:len(df)]
    return df


def test_save_load_parquet_roundtrip(tmp_path):
    """Save then load Parquet, verify shape and dtypes preserved."""
    from data_utils import save_training_data, load_training_data
    df = _make_df(50, ('BTC-USD',))

    save_training_data(df, 'crypto')
    loaded = load_training_data('crypto')

    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)
    # Numeric columns should match closely
    for col in ['Open', 'Close', 'Volume', 'RSI']:
        np.testing.assert_array_almost_equal(
            loaded[col].values, df[col].values, decimal=5)


def test_csv_fallback(tmp_path):
    """Load from CSV when no Parquet exists."""
    from data_utils import load_training_data

    df = _make_df(30, ('BTC-USD',))
    csv_path = tmp_path / 'training_data.csv'
    df.to_csv(csv_path)

    loaded = load_training_data('crypto')
    assert len(loaded) == len(df)


def test_append_ticker_data():
    """Dedup + sort when appending new data."""
    from data_utils import append_ticker_data

    dates1 = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
    dates2 = pd.date_range('2024-01-01 08:00', periods=10, freq='h', tz='UTC')

    df1 = pd.DataFrame({'Close': range(10)}, index=dates1)
    df2 = pd.DataFrame({'Close': range(100, 110)}, index=dates2)

    merged = append_ticker_data(df1, df2)

    # Should have unique timestamps, sorted
    assert merged.index.is_monotonic_increasing
    assert not merged.index.duplicated().any()
    # Overlap period (8:00-9:00) should prefer new data (keep='last')
    assert merged.loc[dates2[0], 'Close'] == 100


def test_validate_catches_gaps(tmp_path):
    """Detect gaps in hourly data."""
    from data_utils import validate_training_data

    # Create data with a 5-hour gap
    dates = list(pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC'))
    dates.append(pd.Timestamp('2024-01-01 15:00', tz='UTC'))  # 5h gap after 10th
    df = pd.DataFrame({
        'Close': range(len(dates)),
        'Ticker': 'BTC-USD',
    }, index=pd.DatetimeIndex(dates))

    report = validate_training_data(df, 'crypto')
    assert len(report['gaps']) > 0


def test_validate_catches_nans(tmp_path):
    """NaN columns should be reported."""
    from data_utils import validate_training_data

    df = _make_df(20, ('BTC-USD',))
    df.loc[df.index[:5], 'RSI'] = np.nan

    report = validate_training_data(df, 'crypto')
    nan_cols = [n['column'] for n in report['nan_columns']]
    assert 'RSI' in nan_cols


def test_migrate_csv_to_parquet(tmp_path):
    """Migration preserves row count."""
    from data_utils import migrate_csv_to_parquet, load_training_data

    df = _make_df(40, ('BTC-USD',))
    csv_path = tmp_path / 'training_data.csv'
    df.to_csv(csv_path)

    assert migrate_csv_to_parquet('crypto') is True
    assert (tmp_path / 'training_data.parquet').exists()

    loaded = load_training_data('crypto')
    assert len(loaded) == len(df)


def test_get_data_path_prefers_parquet(tmp_path):
    """get_data_path returns .parquet when both exist."""
    from data_utils import get_data_path

    # Create both files
    (tmp_path / 'training_data.csv').write_text('a,b\n1,2\n')
    df = pd.DataFrame({'a': [1], 'b': [2]})
    df.to_parquet(tmp_path / 'training_data.parquet')

    path = get_data_path('crypto')
    assert str(path).endswith('.parquet')


def test_get_data_path_csv_only(tmp_path):
    """get_data_path falls back to CSV when no Parquet."""
    from data_utils import get_data_path

    (tmp_path / 'training_data.csv').write_text('a,b\n1,2\n')

    path = get_data_path('crypto')
    assert str(path).endswith('.csv')


def test_load_empty_returns_empty_df(tmp_path):
    """Loading nonexistent data returns empty DataFrame."""
    from data_utils import load_training_data
    df = load_training_data('crypto')
    assert df.empty
