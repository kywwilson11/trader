"""scripts/bars_per_year_census.py — exact numbers on a tiny synthetic store.

Store: ticker AAA = 10 weekdays x 16 extended-hours bars (04:00..19:00 ET open
times, Alpaca convention); ticker BBB = 5 weekdays x 7 bars (09:00..15:00 ET,
an RTH-only name). January => EST (UTC-5), no DST edge.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pyarrow')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import bars_per_year_census as bpc  # noqa: E402


def _frame():
    rows = []
    for d in pd.bdate_range('2026-01-05', periods=10):
        for h in range(4, 20):
            rows.append((d + pd.Timedelta(hours=h), 'AAA'))
    for d in pd.bdate_range('2026-01-05', periods=5):
        for h in range(9, 16):
            rows.append((d + pd.Timedelta(hours=h), 'BBB'))
    df = pd.DataFrame(rows, columns=['Datetime', 'Ticker'])
    df['Datetime'] = (df['Datetime'].dt.tz_localize('America/New_York')
                      .dt.tz_convert('UTC'))
    df['Close'] = 1.0
    return df.set_index('Datetime')


@pytest.fixture
def store(tmp_path):
    p = tmp_path / 'stock_training_data.parquet'
    _frame().to_parquet(p, engine='pyarrow')   # Datetime stored as the index
    return p


def test_schema_projection_finds_index_timestamp(store):
    assert bpc._pick_columns(store) == ('Datetime', 'Ticker')
    df = bpc.load_index(store)
    assert list(df.columns) == ['ts', 'ticker'] and len(df) == 195
    assert str(df['ts'].dt.tz) == 'UTC'


def test_census_exact_numbers(store):
    res = bpc.census(bpc.load_index(store))
    assert res['rows'] == 195 and res['tickers'] == 2
    assert res['asset'] == 'stock' and res['bar_interval_min'] == 60
    assert res['weekend_share'] == 0.0
    assert res['session_counts'] == {'pre': 50, 'open_straddle': 15,
                                     'rth_full': 90, 'close_straddle': 0,
                                     'post': 40, 'weekend': 0}
    assert res['minute_of_hour_hist'] == {0: 195}
    a = res['all']
    assert a['bars_per_ticker_day'] == {'median': 16.0, 'mean': 13.0,
                                        'p10': 7.0, 'p90': 16.0}
    assert a['bars_per_year_day_based'] == pytest.approx((16 + 7) / 2 * 252)
    assert a['bars_per_year_day_pooled'] == pytest.approx(13.0 * 252)  # 3276
    assert a['ratio_day_pooled_vs_legacy'] == pytest.approx(2.0)
    assert a['sqrt_ratio_day_pooled_vs_legacy'] == pytest.approx(math.sqrt(2))
    span_a = (11 + 16 / 24) / 365.25          # Jan05 04:00 -> Jan16 20:00
    span_b = (4 + 7 / 24) / 365.25            # Jan05 09:00 -> Jan09 16:00
    assert a['bars_per_year_span_mean'] == pytest.approx(
        (160 / span_a + 35 / span_b) / 2)
    r = res['rth_only']
    assert r['bars_per_ticker_day']['mean'] == 7.0
    assert r['bars_per_year_day_pooled'] == pytest.approx(7 * 252)   # 1764
    assert r['ratio_day_pooled_vs_legacy'] == pytest.approx(1764 / 1638)
    assert res['by_year'][2026]['rows_per_ticker'] == pytest.approx(97.5)


def test_rth_only_flag_filters_whole_census(store):
    res = bpc.census(bpc.load_index(store), rth_only=True)
    assert res['rows'] == 105
    assert res['session_counts']['pre'] == 0
    assert res['session_counts']['post'] == 0
    assert res['all']['bars_per_year_day_pooled'] == pytest.approx(1764)


def test_classify_half_hour_labels():
    om = np.array([8 * 60 + 30, 9 * 60 + 30, 15 * 60 + 30, 16 * 60, 9 * 60])
    cls = bpc._classify(om, 60, np.zeros(5, dtype=int))
    assert list(cls) == ['pre', 'rth_full', 'close_straddle', 'post',
                         'open_straddle']


def test_crypto_autodetect_and_24h():
    ts = pd.date_range('2026-01-05', periods=24 * 14, freq='h', tz='UTC')
    df = pd.DataFrame({'ts': ts, 'ticker': pd.Categorical(['BTC/USD'] * len(ts))})
    res = bpc.census(df)
    assert res['asset'] == 'crypto' and res['legacy_bars_per_year'] == 8760
    assert res['session_counts']['weekend'] == 24 * 4
    assert res['all']['bars_per_year_day_pooled'] == pytest.approx(
        len(ts) / len(np.unique(ts.tz_convert('America/New_York').date))
        * 365.25)


def test_main_exit_zero_on_missing_file(tmp_path, capsys):
    assert bpc.main(['--data', str(tmp_path / 'nope.parquet')]) == 0
    assert 'ERROR' in capsys.readouterr().out


def test_main_writes_json(store, tmp_path):
    out = tmp_path / 'c.json'
    assert bpc.main(['--data', str(store), '--json', str(out)]) == 0
    import json
    j = json.loads(out.read_text())
    assert j['rows'] == 195 and j['session_counts']['rth_full'] == 90
