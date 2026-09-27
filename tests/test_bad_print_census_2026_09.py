"""SIG-R2-X6 — scripts/bad_print_census.py (measurement-only wick census).

Synthetic parquet stores written with pyarrow; every count is exact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))

import bad_print_census as bp  # noqa: E402

N = 400


def _clean(seed, start='2025-01-01'):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=N, freq='h', tz='UTC')
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, N)))
    open_ = np.r_[close[0], close[:-1]]
    sp = 0.002 * close
    return pd.DataFrame({'Open': open_, 'High': np.maximum(open_, close) + sp,
                         'Low': np.minimum(open_, close) - sp, 'Close': close,
                         'Volume': 10.0}, index=idx)


def _wick(df, i, side, depth, body=0.001):
    o = df['Open'].iat[i]
    df.iloc[i, df.columns.get_loc('Close')] = o * (1 + body)
    if side == 'low':
        df.iloc[i, df.columns.get_loc('Low')] = o * (1 - depth)
        df.iloc[i, df.columns.get_loc('High')] = o * (1 + body) * 1.001
    else:
        df.iloc[i, df.columns.get_loc('High')] = o * (1 + depth)
        df.iloc[i, df.columns.get_loc('Low')] = o * 0.999
    return df


@pytest.fixture
def store(tmp_path):
    a = _clean(1)
    a = _wick(a, 50, 'low', 0.30)
    a = _wick(a, 120, 'high', 0.20)
    a = _wick(a, 390, 'low', 0.16)              # inside trailing window
    a = _wick(a, 200, 'low', 0.30, body=0.02)   # body 2 %: only at 0.03
    a = _wick(a, 300, 'low', 0.12)              # too shallow at 0.15
    b = _clean(2)
    b = _wick(b, 10, 'high', 0.50)
    frames = []
    for t, f in (('AAA', a), ('BBB', b)):
        f = f.copy()
        f['Ticker'] = t
        f['TB_Reason_24'] = 6.0
        f['TB_Reason_12'] = 1.0
        f.iloc[50, f.columns.get_loc('TB_Reason_24')] = 1.0
        f.iloc[-5:, f.columns.get_loc('TB_Reason_24')] = np.nan
        frames.append(f)
    df = pd.concat(frames)
    df.index.name = 'Datetime'
    p = tmp_path / 'store.parquet'
    pq.write_table(pa.Table.from_pandas(df), p, row_group_size=300)
    return p, a, b


def test_reason_names_mirror_policy_exits():
    pe = pytest.importorskip('policy_exits')
    assert bp.REASON_NAMES == pe.REASON_NAMES


def test_flag_wick_prints_exact():
    o = np.array([100, 100, 100, 100, 100, 0.0])
    c = np.array([100.5, 100.5, 102, 100.5, 100.5, 0.0])
    l = np.array([80, 90, 80, 99, 99, 0.0])
    h = np.array([101, 101, 101, 120, 114, 0.0])
    lo, hi = bp.flag_wick_prints(o, h, l, c, 0.15, 0.01)
    assert lo.tolist() == [True, False, False, False, False, False]
    assert hi.tolist() == [False, False, False, True, False, False]
    lo3, _ = bp.flag_wick_prints(o, h, l, c, 0.15, 0.03)
    assert lo3.tolist() == [True, False, True, False, False, False]
    _, hi2 = bp.flag_wick_prints(o, h, l, c, 0.15, 0.01, high_frac=0.10)
    assert hi2.tolist() == [False, False, False, True, True, False]


def test_census_counts_dates_reasons_trailing(store):
    p, a, b = store
    df, names = bp.load_store(p, ['Ticker', 'Open', 'High', 'Low', 'Close',
                                  'Volume', 'TB_Reason_24', 'TB_Reason_12'])
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == 'UTC'
    c = bp.census(df, 0.15, 0.01, None, holdout_days=5)
    A, B = c['per_ticker']['AAA'], c['per_ticker']['BBB']
    assert (A['low'], A['high'], A['any']) == (2, 1, 3)
    assert (B['low'], B['high'], B['any']) == (0, 1, 1)
    assert c['totals'] == {'rows': 2 * N, 'low': 2, 'high': 2, 'any': 4,
                           'in_trailing': 1}
    assert A['in_trailing'] == 1                     # row 390 of 400
    assert [d['ts'] for d in A['dates']] == [
        a.index[50].isoformat(), a.index[120].isoformat(),
        a.index[390].isoformat()]
    assert [d['side'] for d in A['dates']] == ['low', 'high', 'low']
    assert A['max_low_depth'] == pytest.approx(0.30)
    assert B['max_high_depth'] == pytest.approx(0.50)
    assert c['tb_reason_columns'] == ['TB_Reason_12', 'TB_Reason_24']
    assert A['tb_reason_counts']['TB_Reason_24'] == {'hard_stop': 1,
                                                     'vertical': 2}
    assert A['tb_reason_counts']['TB_Reason_12'] == {'hard_stop': 3}
    assert A['frac_any'] == pytest.approx(3 / N)
    c3 = bp.census(df, 0.15, 0.03, None, holdout_days=5)
    assert c3['per_ticker']['AAA']['low'] == 3       # + the 2 % body bar
    c10 = bp.census(df, 0.10, 0.01, None, holdout_days=5)
    assert c10['per_ticker']['AAA']['low'] == 3      # + the 12 % wick


def test_census_store_chunked_equals_whole(store):
    p, _, _ = store
    cols = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'TB_Reason_24']
    whole = bp.census(bp.load_store(p, cols)[0], 0.15, 0.01, None, 5)
    chunked = bp.census_store(p, cols, None, 0.15, 0.01, None, 5,
                              chunk_rows=10)     # one ticker per chunk
    assert json.dumps(whole, sort_keys=True) == json.dumps(chunked,
                                                           sort_keys=True)
    sub = bp.census_store(p, cols, ['BBB'], 0.15, 0.01, None, 5)
    assert list(sub['per_ticker']) == ['BBB']
    # the trailing window stays anchored on the STORE max, not BBB's
    assert sub['params']['store_max_ts'] == whole['params']['store_max_ts']


def test_repair_frame_rule():
    idx = pd.date_range('2025-01-01', periods=9, freq='h', tz='UTC')
    g = pd.DataFrame({'Open': 100.0, 'Close': 100.5, 'High': 101.0,
                      'Low': 99.5, 'Volume': 1.0}, index=idx)
    g = g[['Open', 'High', 'Low', 'Close', 'Volume']]
    g.iloc[7] = [100.0, 100.6, 80.0, 100.2, 1.0]      # O H L C V: low wick
    g.iloc[8] = [100.0, 130.0, 99.8, 100.0, 1.0]      # high wick
    lo, hi = bp.flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'])
    assert lo.tolist() == [False] * 7 + [True, False]
    assert hi.tolist() == [False] * 8 + [True]
    r = bp.repair_frame(g, lo, hi)
    # every normal bar's TR is 1.5 -> median prior TR 1.5 at both prints
    assert r['Low'].iat[7] == pytest.approx(100.0 - 1.5)
    assert r['High'].iat[8] == pytest.approx(100.0 + 1.5)
    assert len(r) == len(g)
    unchanged = [i for i in range(9) if i not in (7, 8)]
    pd.testing.assert_frame_equal(r.iloc[unchanged], g.iloc[unchanged])
    assert (r['Open'] == g['Open']).all() and (r['Close'] == g['Close']).all()


def test_footprint_reach_and_parity(store):
    p, a, _ = store
    g = a.copy()
    g['Ticker'] = 'AAA'
    tr = pd.Series(bp.true_range(g['High'], g['Low'], g['Close']),
                   index=g.index)
    g['ATR'] = tr.rolling(14).mean()
    fp = bp.footprint(g, 0.15, 0.01)
    f = fp['features']
    assert f['ATR']['max_reach_bars'] == 13           # 14-bar mean
    assert f['ATR']['in_store'] is True
    assert f['STOCHk_14_3_3']['max_reach_bars'] <= 15
    assert f['STOCHd_14_3_3']['max_reach_bars'] <= 17
    assert f['Pos_Range_60h']['max_reach_bars'] <= 59
    assert f['ATR_Percentile']['max_reach_bars'] <= 112
    assert f['Parkinson_RRV_day']['max_reach_bars'] == -1   # day-level
    assert fp['atr_store_vs_recomputed_median_absrel']['AAA'] < 1e-12
    assert len(fp['atr_at_print']) == 3
    assert f['ATR']['max_ratio_at_print'] > 3     # one 30 % TR in a 14-mean


def test_tb_flips_removes_wick_hard_stop():
    pytest.importorskip('policy_exits')
    idx = pd.date_range('2025-01-01', periods=80, freq='h', tz='UTC')
    close = np.full(80, 100.0) + np.linspace(0, 1, 80)
    g = pd.DataFrame({'Open': close, 'High': close + 0.3, 'Low': close - 0.3,
                      'Close': close, 'Volume': 1.0}, index=idx)
    g.iloc[40, g.columns.get_loc('Low')] = 70.0       # wick-only print
    g['Ticker'] = 'AAA'
    tr = pd.Series(bp.true_range(g['High'], g['Low'], g['Close']),
                   index=g.index)
    g['ATR'] = tr.rolling(14).mean()
    res = bp.tb_flips(g, 'crypto', 0.15, 0.01, None, fbs=[12])['AAA']['12']
    assert res['hard_stop_before'] > res['hard_stop_after']
    assert res['hard_stop_removed'] >= 1


def test_venue_verdicts():
    f = {'side': 'low', 'Open': 100.0, 'Close': 100.0, 'Low': 70.0,
         'High': 100.5}
    assert bp.venue_verdict(f, 99.0, 100.6)[0] == 'bad_print'
    assert bp.venue_verdict(f, 80.0, 100.6)[0] == 'real_move'
    assert bp.venue_verdict(f, 92.0, 100.6)[0] == 'partial'
    # Alpaca Close 25 % below the venue's whole range -> off-market body
    body = {'side': 'high', 'Open': 4.9, 'Close': 4.9, 'Low': 4.88,
            'High': 6.55}
    assert bp.venue_verdict(body, 6.5, 6.56)[0] == 'bad_body'
    hi = {'side': 'high', 'Open': 100.0, 'Close': 100.0, 'Low': 99.5,
          'High': 130.0}
    assert bp.venue_verdict(hi, 99.6, 100.8)[0] == 'bad_print'


def test_plan_windows_and_rescore():
    fl = [{'ticker': 'X', 'ts': '2025-01-01T00:00:00+00:00'},
          {'ticker': 'X', 'ts': '2025-01-02T00:00:00+00:00'},
          {'ticker': 'X', 'ts': '2025-03-01T00:00:00+00:00'},
          {'ticker': 'Y', 'ts': '2025-01-01T05:00:00+00:00'}]
    w = bp.plan_windows(fl, max_requests=2)
    assert len(w) == 2
    assert w[0][0] == 'X' and w[0][1] == pd.Timestamp('2025-01-01', tz='UTC')
    assert w[0][2] - w[0][1] == pd.Timedelta(hours=bp.VENUE_WINDOW_H)
    must = bp.plan_windows(fl, 1, must=[('Y', '2025-01-01T05:00:00+00:00')])
    assert [x[0] for x in must] == ['Y']
    chk = {'ticker': 'X', 'ts': 't', 'side': 'low', 'Open': 100.0,
           'Close': 100.0, 'Low': 70.0, 'High': 100.5, 'venue_low': 99.0,
           'venue_high': 100.6}
    vc = bp.rescore_venue_checks({'checks': [dict(chk), dict(chk),
                                             {'ticker': 'Z', 'ts': 't',
                                              'verdict': 'no_venue_bar'}]})
    assert vc['tally'] == {'bad_print': 1, 'no_venue_bar': 1}


def test_main_writes_reports_offline(store, tmp_path):
    p, _, _ = store
    out, js = tmp_path / 'r.txt', tmp_path / 'r.json'
    bp.main(['--data', str(p), '--footprint', '--out', str(out),
             '--json', str(js), '--holdout-days', '5'])
    d = json.loads(js.read_text())
    assert d['census']['totals']['any'] == 4
    assert d['footprint']['features']['ATR']['max_reach_bars'] == 13
    assert d['tb_flips'] is None
    txt = out.read_text()
    assert 'AAA: rows 400 low 2 high 1 any 3' in txt
    assert 'FOOTPRINT' in txt
