"""R5 (2026-09): raw-OHLCV sidecar reload must hand compute_features a
tz-aware UTC DatetimeIndex.

Root cause pinned here: the interior-gap-repair patch comes from
market_data.fetch_historical_bars stamped America/New_York (the Alpaca
SDK's tz) while the sidecar slice is UTC; pd.concat of two DIFFERENT tzs
yields an object Index, which reached indicators.compute_features and
crashed at `idx.hour` on the second sidecar run.

Mac-safe: pandas only; the parquet round trip skips without an engine.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

import data_utils as du


def _bars(start, n, tz='UTC', close0=100.0, src='alpaca', name='Datetime'):
    idx = pd.date_range(start, periods=n, freq='h', tz=tz, name=name)
    c = close0 + np.arange(n, dtype=float)
    return pd.DataFrame({'Open': c, 'High': c + 1, 'Low': c - 1,
                         'Close': c, 'Volume': np.ones(n), 'Src': src},
                        index=idx)


def _assert_utc(df):
    assert isinstance(df.index, pd.DatetimeIndex), type(df.index)
    assert str(df.index.tz) == 'UTC'
    assert df.index.is_monotonic_increasing


def _has_parquet():
    return (importlib.util.find_spec('pyarrow') is not None
            or importlib.util.find_spec('fastparquet') is not None)


# --- the reproduced failure ------------------------------------------------

def test_mixed_tz_concat_is_the_root_cause():
    """Documents the pandas behaviour behind the crash (raw concat)."""
    utc = _bars('2026-01-01', 10)
    ny = _bars('2026-01-01', 4).tz_convert('America/New_York')
    raw = pd.concat([utc, ny])
    assert not isinstance(raw.index, pd.DatetimeIndex)   # object Index


def test_gap_repair_patch_in_new_york_tz_keeps_utc_index():
    existing = _bars('2021-01-10 00:00', 48)
    patch = _bars('2021-01-10 10:00', 6, close0=500.0).tz_convert(
        'America/New_York')
    out = du.append_ticker_data(existing, patch)
    _assert_utc(out)
    assert len(out) == 48                 # full overlap: nothing new
    assert not out.index.has_duplicates
    # keep='last': the patch values won on the overlap
    assert out.loc[pd.Timestamp('2021-01-10 10:00', tz='UTC'),
                   'Close'] == 500.0
    du._ensure_utc_index(out, 'BTC-USD')  # the harvest guard passes


def test_second_run_sequence_load_slice_gaprepair_zero_new_bars():
    """Exact harvest sequence: sidecar slice -> NY gap-repair append ->
    0-new-bars incremental merge -> guard (previously AttributeError)."""
    raw = du.merge_raw_ohlcv(pd.DataFrame(), _bars('2026-09-20', 120),
                             'BTC-USD')
    raw = du.merge_raw_ohlcv(raw, _bars('2026-09-20', 120, close0=5.0),
                             'ETH-USD')
    t_raw = raw[raw['Ticker'] == 'BTC-USD']
    existing = t_raw[['Open', 'High', 'Low', 'Close', 'Volume', 'Src']]
    patch = _bars('2026-09-21', 5).tz_convert('America/New_York')
    existing = du.append_ticker_data(existing, patch)
    _assert_utc(existing)
    new = existing.iloc[-52:].copy()      # all overlap -> 0 new bars
    merged = du.append_ticker_data(existing, new)
    _assert_utc(merged)
    assert len(merged) == len(existing)
    du._ensure_utc_index(merged.drop(columns=['Src']), 'BTC-USD')


# --- merge helpers ---------------------------------------------------------

def test_zero_new_bars_merge_preserves_index():
    existing = _bars('2026-01-01', 30)
    empty = existing.iloc[:0]
    out = du.append_ticker_data(existing, empty)
    _assert_utc(out)
    assert out.index.equals(existing.index)
    assert out.index.name == 'Datetime'


def test_n_new_bars_merge_dedupes_overlap_keep_last():
    existing = _bars('2026-01-01 00:00', 30)
    new = _bars('2026-01-01 20:00', 20, close0=900.0)   # 10 overlap, 10 new
    out = du.append_ticker_data(existing, new)
    _assert_utc(out)
    assert len(out) == 40
    assert not out.index.has_duplicates
    assert out.loc[pd.Timestamp('2026-01-01 20:00', tz='UTC'),
                   'Close'] == 900.0


def test_naive_side_is_taken_as_utc():
    existing = _bars('2026-01-01', 10)
    new = _bars('2026-01-01 05:00', 10, tz=None)
    out = du.append_ticker_data(existing, new)
    _assert_utc(out)
    assert len(out) == 15


def test_merge_raw_ohlcv_mixed_tz_keeps_utc_and_ticker_keyed_dedup():
    raw = du.merge_raw_ohlcv(pd.DataFrame(), _bars('2026-01-01', 10), 'A')
    raw = du.merge_raw_ohlcv(raw, _bars('2026-01-01', 10), 'B')
    upd = _bars('2026-01-01 08:00', 4, close0=777.0).tz_convert(
        'America/New_York')
    out = du.merge_raw_ohlcv(raw, upd, 'A')
    _assert_utc(out)
    assert (out['Ticker'] == 'A').sum() == 12
    assert (out['Ticker'] == 'B').sum() == 10
    a = out[out['Ticker'] == 'A']
    assert not a.index.has_duplicates
    assert a.loc[pd.Timestamp('2026-01-01 09:00', tz='UTC'),
                 'Close'] == 778.0


# --- guard -----------------------------------------------------------------

def test_guard_raises_on_plain_index():
    df = _bars('2026-01-01', 5)
    mixed = pd.concat([df, df.tz_convert('America/New_York').iloc[:0],
                       _bars('2026-01-02', 2).tz_convert('Asia/Tokyo')])
    assert not isinstance(mixed.index, pd.DatetimeIndex)
    with pytest.raises(TypeError, match='not a DatetimeIndex'):
        du._ensure_utc_index(mixed, 'BTC-USD')
    plain = df.reset_index(drop=True)
    with pytest.raises(TypeError, match='BTC-USD'):
        du._ensure_utc_index(plain, 'BTC-USD')


def test_guard_raises_on_naive_non_utc_unsorted_or_duplicated():
    df = _bars('2026-01-01', 5)
    with pytest.raises(ValueError, match='tz'):
        du._ensure_utc_index(df.tz_localize(None))
    with pytest.raises(ValueError, match='tz'):
        du._ensure_utc_index(df.tz_convert('America/New_York'))
    with pytest.raises(ValueError, match='strictly increasing'):
        du._ensure_utc_index(df.iloc[::-1])
    with pytest.raises(ValueError, match='strictly increasing'):
        du._ensure_utc_index(pd.concat([df, df.iloc[-1:]]))
    assert du._ensure_utc_index(df) is df


def test_both_harvests_guard_before_feature_computation():
    """Source pin: the guard sits immediately before compute_*features."""
    import pathlib
    root = pathlib.Path(du.__file__).resolve().parent / 'scripts'
    for fname, call in (('harvest_crypto_data.py', 'compute_features(ohlcv'),
                        ('harvest_stock_data.py',
                         'compute_stock_features(ohlcv')):
        src = (root / fname).read_text()
        g = src.index('_ensure_utc_index(ohlcv, ticker)')
        c = src.index(call)
        assert 0 < c - g < 300, fname


# --- load / save round trip ------------------------------------------------

def _sidecar_at(monkeypatch, tmp_path):
    p = tmp_path / 'raw_ohlcv.parquet'
    monkeypatch.setattr(du, 'raw_sidecar_path', lambda prefix: p)
    return p


@pytest.mark.skipif(not _has_parquet(), reason='no parquet engine')
def test_round_trip_preserves_index_type_tz_order(monkeypatch, tmp_path):
    p = _sidecar_at(monkeypatch, tmp_path)
    raw = du.merge_raw_ohlcv(pd.DataFrame(), _bars('2026-01-01', 50), 'A')
    raw = du.merge_raw_ohlcv(raw, _bars('2026-01-01', 30, close0=7.), 'B')
    assert du.save_raw_ohlcv(raw, 'crypto')
    import pyarrow.parquet as pq
    ts = [f for f in pq.read_schema(p) if f.name == 'Datetime']
    assert ts and 'tz=UTC' in str(ts[0].type)   # index saved WITH tz
    back = du.load_raw_ohlcv('crypto')
    _assert_utc(back)
    assert back.index.name == 'Datetime'
    assert len(back) == 80
    pd.testing.assert_frame_equal(back, raw, check_freq=False,
                                  check_index_type=False)


@pytest.mark.skipif(not _has_parquet(), reason='no parquet engine')
def test_load_normalizes_column_index_and_dedupes_keep_last(monkeypatch,
                                                            tmp_path):
    p = _sidecar_at(monkeypatch, tmp_path)
    a = _bars('2026-01-01', 6).assign(Ticker='A')
    dup = _bars('2026-01-01 02:00', 1, close0=999.).assign(Ticker='A')
    b = _bars('2026-01-01', 6).assign(Ticker='B')
    df = pd.concat([a, b, dup]).tz_convert('America/New_York')
    df.reset_index().to_parquet(p)          # timestamp as a COLUMN
    back = du.load_raw_ohlcv('crypto')
    _assert_utc(back)
    assert len(back) == 12
    ra = back[back['Ticker'] == 'A']
    assert ra.loc[pd.Timestamp('2026-01-01 02:00', tz='UTC'),
                  'Close'] == 999.0
