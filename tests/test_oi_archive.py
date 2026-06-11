"""Tests for the open-interest archive + live features."""

import datetime as dt
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oi_archive
from oi_archive import (_parse_zip, live_oi_features,
                        oi_features_for_index)


def _mk_zip(day: str, n_5min: int = 288, oi0: float = 100_000.0,
            drift: float = 0.0):
    header = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
              "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
              "count_long_short_ratio,sum_taker_long_short_vol_ratio")
    rows = []
    t0 = dt.datetime.fromisoformat(day + 'T00:05:00')
    for i in range(n_5min):
        ts = t0 + dt.timedelta(minutes=5 * i)
        oi = oi0 * (1 + drift * i / n_5min)
        rows.append(f"{ts:%Y-%m-%d %H:%M:%S},BTCUSDT,{oi:.4f},"
                    f"{oi * 70000:.4f},1.5,1.36,1.52,1.37")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f'BTCUSDT-metrics-{day}.csv',
                    header + '\n' + '\n'.join(rows))
    return buf.getvalue()


def test_parse_zip_resamples_to_hourly():
    out = _parse_zip(_mk_zip('2026-06-01'))
    assert out is not None
    assert len(out) == 24  # 288 5-min rows -> 24 hourly
    assert {'ts', 'oi', 'oi_value', 'tt_ls_ratio', 'taker_ratio'} <= set(out.columns)
    assert out['ts'].dt.tz is not None  # UTC-aware


def test_parse_zip_garbage_returns_none():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('x.csv', 'foo,bar\n1,2')
    assert _parse_zip(buf.getvalue()) is None


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """Build a 20-day hourly archive with a level shift near the end."""
    monkeypatch.setattr(oi_archive, 'ARCHIVE_FILE', tmp_path / 'oi.parquet')
    idx = pd.date_range('2026-05-01', periods=24 * 20, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    oi_val = 7e9 * (1 + rng.normal(0, 0.002, len(idx))).cumprod()
    oi_val[-24:] *= 1.25  # 25% OI surge in the final day
    arc = pd.DataFrame({'symbol': 'BTC/USD', 'ts': idx,
                        'oi': oi_val / 70000, 'oi_value': oi_val,
                        'tt_ls_ratio': 1.4, 'taker_ratio': 1.3})
    arc.to_parquet(tmp_path / 'oi.parquet')
    return idx, oi_val


def test_oi_features_alignment_and_values(archive):
    idx, oi_val = archive
    bars = pd.date_range('2026-05-15', periods=48, freq='h', tz='UTC')
    feats = oi_features_for_index('BTC/USD', bars)
    assert feats is not None
    assert np.isfinite(feats['OI_Chg_24h']).all()
    # Last bars sit in the surge -> strongly positive change and z
    feats_end = oi_features_for_index('BTC/USD', idx[-3:])
    assert feats_end['OI_Chg_24h'][-1] > 10
    assert feats_end['OI_Z'][-1] > 1.5


def test_oi_features_none_for_unknown_or_short(archive, monkeypatch):
    assert oi_features_for_index('ETH/USD', pd.date_range(
        '2026-05-15', periods=4, freq='h', tz='UTC')) is None


# --- live features ---

@pytest.fixture
def live_env(tmp_path, monkeypatch):
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE',
                        tmp_path / 'oi_history.json')
    oi_archive._live_cache.clear()
    return tmp_path


def test_live_cold_start_zeros(live_env, monkeypatch):
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    out = live_oi_features('BTC/USD')
    assert out == {'OI_Chg_24h': 0.0, 'OI_Z': 0.0}
    # First sample persisted
    hist = json.loads((live_env / 'oi_history.json').read_text())
    assert len(hist['BTC/USD']) == 1


def test_live_chg_vs_24h_ago(live_env, monkeypatch):
    now = time.time()
    hist = {'BTC/USD': [[now - 86400, 4000.0], [now - 3600, 4900.0]]}
    (live_env / 'oi_history.json').write_text(json.dumps(hist))
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    out = live_oi_features('BTC/USD')
    assert out['OI_Chg_24h'] == pytest.approx(25.0)  # 4000 -> 5000


def test_live_z_needs_week_of_history(live_env, monkeypatch):
    now = time.time()
    rng = np.random.default_rng(1)
    vals = list(4000 + rng.normal(0, 40, 200))
    hist = {'BTC/USD': [[now - (200 - i) * 3600, v]
                        for i, v in enumerate(vals)]}
    (live_env / 'oi_history.json').write_text(json.dumps(hist))
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 4400.0)
    out = live_oi_features('BTC/USD')
    assert out['OI_Z'] > 3  # 10 sigma-ish surge clipped by sample std


def test_live_history_thinned_to_hourly(live_env, monkeypatch):
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    live_oi_features('BTC/USD')
    live_oi_features('BTC/USD')  # seconds later -> must NOT append again
    hist = json.loads((live_env / 'oi_history.json').read_text())
    assert len(hist['BTC/USD']) == 1


def test_live_none_when_okx_down(live_env, monkeypatch):
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: None)
    assert live_oi_features('BTC/USD') is None


# --- top-trader long/short ratio features ---

class TestLSFeatures:
    def _archive(self, tmp_path, monkeypatch, ratios):
        monkeypatch.setattr(oi_archive, 'ARCHIVE_FILE',
                            tmp_path / 'oi.parquet')
        idx = pd.date_range('2026-05-01', periods=len(ratios), freq='h',
                            tz='UTC')
        arc = pd.DataFrame({'symbol': 'BTC/USD', 'ts': idx,
                            'oi': 1.0, 'oi_value': 7e9,
                            'tt_ls_ratio': ratios, 'taker_ratio': 1.3})
        arc.to_parquet(tmp_path / 'oi.parquet')
        return idx

    def test_ls_z_detects_crowding_surge(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(2)
        ratios = list(1.4 + rng.normal(0, 0.05, 400))
        ratios[-5:] = [2.4] * 5  # top traders pile long
        idx = self._archive(tmp_path, monkeypatch, ratios)
        feats = oi_archive.ls_features_for_index('BTC/USD', idx[-3:])
        assert feats is not None
        assert feats['TT_LS_Z'][-1] > 3

    def test_ls_none_for_short_or_missing(self, tmp_path, monkeypatch):
        idx = self._archive(tmp_path, monkeypatch, [1.4] * 50)  # < 200 rows
        assert oi_archive.ls_features_for_index('BTC/USD', idx) is None
        assert oi_archive.ls_features_for_index('ETH/USD', idx) is None

    def test_live_ls_z_from_okx_history(self, monkeypatch):
        import io, json as _json, urllib.request
        oi_archive._ls_cache.clear()
        rng = np.random.default_rng(3)
        hist = list(1.5 + rng.normal(0, 0.05, 300))
        hist[0] = 2.5  # newest reading: extreme long crowding
        payload = {'code': '0',
                   'data': [[str(1781143200000 - i * 3600000), f'{v:.4f}']
                            for i, v in enumerate(hist)]}
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: io.BytesIO(
                                _json.dumps(payload).encode()))
        out = oi_archive.live_ls_features('BTC/USD')
        assert out is not None and out['TT_LS_Z'] > 3

    def test_live_ls_none_on_thin_history_or_error(self, monkeypatch):
        import io, json as _json, urllib.request
        oi_archive._ls_cache.clear()
        payload = {'code': '0', 'data': [['1781143200000', '1.5']] * 10}
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: io.BytesIO(
                                _json.dumps(payload).encode()))
        assert oi_archive.live_ls_features('BTC/USD') is None
        oi_archive._ls_cache.clear()

        def boom(req, timeout=10):
            raise OSError('down')

        monkeypatch.setattr(urllib.request, 'urlopen', boom)
        assert oi_archive.live_ls_features('BTC/USD') is None
