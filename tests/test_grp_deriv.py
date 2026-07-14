"""Behaviour-neutral scout fixes for the funding/OI/basis/short-flow group.

Runs on the dev Mac (no torch/parquet/network): parquet + network fetches are
monkeypatched. Pins the value-neutrality of the freshness-stamp logging and the
basis_archive corrupt-zip robustness fix.
"""
import json
import statistics
import time as _time

import pandas as pd
import pytest


def _fund_series(days_old, n=40, val=0.0001):
    end = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_old)
    idx = pd.date_range(end=end, periods=n, freq='8h', tz='UTC')
    return pd.Series([val] * n, index=idx)


def test_funding_stale_baseline_warns_and_value_neutral(monkeypatch, caplog):
    import funding
    import funding_archive
    monkeypatch.setattr(funding, 'get_funding_rate', lambda sym: 0.0002)
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda sym: _fund_series(10))
    funding._stale_warned.clear()
    with caplog.at_level('WARNING'):
        stale = funding.live_funding_features('BTC/USD')
    assert stale is not None
    assert set(stale) == {'Funding_Rate_Ann', 'Funding_Z', 'Funding_Chg_24h'}
    assert any('stale' in r.getMessage() for r in caplog.records)
    # Same rate + same sample VALUES with a FRESH index => identical features.
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda sym: _fund_series(0))
    funding._stale_warned.clear()
    fresh = funding.live_funding_features('BTC/USD')
    assert fresh == stale


def test_funding_fresh_baseline_no_warn(monkeypatch, caplog):
    import funding
    import funding_archive
    monkeypatch.setattr(funding, 'get_funding_rate', lambda sym: 0.0002)
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda sym: _fund_series(0))
    funding._stale_warned.clear()
    with caplog.at_level('WARNING'):
        out = funding.live_funding_features('BTC/USD')
    assert out is not None
    assert not any('stale' in r.getMessage() for r in caplog.records)


def _svr_pair(days_old, n=300):
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_old)
    idx = pd.date_range(end=end, periods=n, freq='D')
    return pd.Series(0.3, index=idx), pd.Series(0.1, index=idx)


def test_short_flow_stale_svr_warns_value_neutral(monkeypatch, caplog):
    import short_flow
    monkeypatch.setattr(short_flow, 'svr_series', lambda sym: _svr_pair(15))
    short_flow._svr_stale_warned.clear()
    with caplog.at_level('WARNING'):
        out = short_flow.live_svr_features('AAPL')
    assert out == {'SVR_21': 0.3, 'SVR_Z': 0.1}
    assert any('old' in r.getMessage() for r in caplog.records)


def test_short_flow_fresh_svr_no_warn(monkeypatch, caplog):
    import short_flow
    monkeypatch.setattr(short_flow, 'svr_series', lambda sym: _svr_pair(1))
    short_flow._svr_stale_warned.clear()
    with caplog.at_level('WARNING'):
        out = short_flow.live_svr_features('AAPL')
    assert out == {'SVR_21': 0.3, 'SVR_Z': 0.1}
    assert not any('old' in r.getMessage() for r in caplog.records)


def test_basis_sync_continues_past_corrupt_zip(monkeypatch, capsys):
    import basis_archive
    monkeypatch.setattr(basis_archive, 'load_archive',
                        lambda: pd.DataFrame(columns=['symbol', 'ts', 'premium']))
    monkeypatch.setattr(basis_archive, '_months',
                        lambda start: ['2025-01', '2025-02'])

    class _Resp:
        def read(self):
            return b'zip-bytes'

    monkeypatch.setattr(basis_archive.urllib.request, 'urlopen',
                        lambda req, timeout=30: _Resp())
    calls = {'n': 0}

    def fake_parse(data):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ValueError('corrupt zip')
        return pd.DataFrame({'ts': pd.to_datetime(['2025-02-01'], utc=True),
                             'premium': [0.0001]})

    monkeypatch.setattr(basis_archive, '_parse_zip', fake_parse)
    monkeypatch.setattr(pd.DataFrame, 'to_parquet',
                        lambda self, path, **k: None, raising=False)
    monkeypatch.setattr(basis_archive.os, 'replace', lambda a, b: None)
    ok = basis_archive.sync(symbols=['BTC/USD'], start='2025-01')
    assert ok is True                 # pre-fix this raises out of sync()
    assert calls['n'] == 2            # both months attempted
    assert 'parse failed' in capsys.readouterr().out


def test_basis_load_archive_corrupt_reports(monkeypatch, capsys, tmp_path):
    import basis_archive
    f = tmp_path / 'basis_archive.parquet'
    f.write_bytes(b'not a parquet file')
    monkeypatch.setattr(basis_archive, 'ARCHIVE_FILE', f)
    df = basis_archive.load_archive()
    assert list(df.columns) == ['symbol', 'ts', 'premium']
    assert df.empty
    assert 'corrupt' in capsys.readouterr().out.lower()


def test_oi_live_ls_age_stamp_value_neutral(monkeypatch, caplog):
    import oi_archive
    newest_ms = (_time.time() - 24 * 3600) * 1000     # 24h stale (the D5 case)
    rows = [[str(int(newest_ms - i * 3600 * 1000)), f'{1.0 + 0.001 * i}']
            for i in range(200)]
    payload = json.dumps({'code': '0', 'data': rows}).encode()

    class _Resp:
        def read(self):
            return payload

    monkeypatch.setattr(oi_archive.urllib.request, 'urlopen',
                        lambda req, timeout=10: _Resp())
    oi_archive._ls_cache.clear()
    oi_archive._fail_cache.clear()
    vals = [float(r[1]) for r in rows]
    exp = (vals[0] - statistics.fmean(vals)) / statistics.pstdev(vals)
    with caplog.at_level('DEBUG'):
        out = oi_archive.live_ls_features('BTC/USD')
    assert out is not None
    assert out['TT_LS_Z'] == pytest.approx(exp)     # value-neutral
    assert any('old' in r.getMessage() for r in caplog.records)
