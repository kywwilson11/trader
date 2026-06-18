"""Wave-7 Finding 9: spot-perp basis archive (basis_archive).

Synthetic premiumIndexKlines zips (headered + headerless) verify the parser,
and a synthetic premium series verifies the PIT features (Bps/Z/Chg + the
sd==0 flat-premium mask), mirroring test_oi_archive's approach."""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import basis_archive as ba


def _mk_klines_zip(month='2026-01', n=200, premium=0.0005, header=False):
    """One premiumIndexKlines monthly zip: 12-col klines, premium in col 4."""
    t0 = pd.Timestamp('2026-01-01T00:00:00Z')
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,"
                      "quote_volume,count,taker_buy_volume,taker_buy_quote,ignore")
    for i in range(n):
        ot = int((t0 + pd.Timedelta(hours=i)).value // 10**6)  # epoch ms
        ct = ot + 3_600_000
        lines.append(f"{ot},{premium},{premium},{premium},{premium},0,"
                     f"{ct},0,1,0,0,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f'BTCUSDT-1h-{month}.csv', '\n'.join(lines))
    return buf.getvalue()


class TestParseZip:
    def test_headerless(self):
        out = ba._parse_zip(_mk_klines_zip(n=100, premium=0.0009, header=False))
        assert out is not None and len(out) == 100
        assert {'ts', 'premium'} <= set(out.columns)
        assert out['ts'].dt.tz is not None
        assert out['premium'].iloc[0] == pytest.approx(0.0009)

    def test_headered(self):
        # Binance added headers to vision CSVs in 2025 -> header row must drop
        out = ba._parse_zip(_mk_klines_zip(n=50, premium=0.0003, header=True))
        assert out is not None and len(out) == 50
        assert out['premium'].iloc[0] == pytest.approx(0.0003)

    def test_garbage_returns_none(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('x.csv', 'foo,bar\n1,2')
        # 2 cols < 5 -> skipped -> None
        assert ba._parse_zip(buf.getvalue()) is None


class TestFeatures:
    def _patch_series(self, monkeypatch, s):
        monkeypatch.setattr(ba, 'get_basis_series', lambda sym: s)
        # no funding series -> skip the residual column
        import funding_archive
        monkeypatch.setattr(funding_archive, 'get_funding_series',
                            lambda sym: None, raising=False)

    def test_bps_and_z(self, monkeypatch):
        idx = pd.date_range('2026-01-01', periods=800, freq='h', tz='UTC')
        # premium drifts up then spikes -> positive Z at the end
        prem = pd.Series(np.concatenate([np.full(790, 0.0002),
                                         np.full(10, 0.0020)]), index=idx)
        self._patch_series(monkeypatch, prem)
        out = ba.basis_features_for_index('BTC/USD', idx)
        assert out is not None
        assert out['Basis_Bps'][0] == pytest.approx(2.0, abs=1e-6)  # 0.0002*1e4
        assert out['Basis_Z'][-1] > 2.0   # the spike is a high z-score

    def test_flat_premium_z_is_zero_not_nan(self, monkeypatch):
        idx = pd.date_range('2026-01-01', periods=400, freq='h', tz='UTC')
        prem = pd.Series(np.full(400, 0.0001), index=idx)  # pinned flat
        self._patch_series(monkeypatch, prem)
        out = ba.basis_features_for_index('BTC/USD', idx)
        # sd==0 -> masked to 0.0, never NaN (would silently drop bars)
        assert np.nanmax(np.abs(out['Basis_Z'])) == pytest.approx(0.0)

    def test_too_short_returns_none(self, monkeypatch):
        idx = pd.date_range('2026-01-01', periods=10, freq='h', tz='UTC')
        self._patch_series(monkeypatch, pd.Series(np.full(10, 0.0001), index=idx))
        assert ba.basis_features_for_index('BTC/USD', idx) is None

    def test_basis_minus_funding_residual(self, monkeypatch):
        idx = pd.date_range('2026-01-01', periods=300, freq='h', tz='UTC')
        prem = pd.Series(np.full(300, 0.0004), index=idx)   # 4 bps basis
        monkeypatch.setattr(ba, 'get_basis_series', lambda sym: prem)
        import funding_archive
        # funding 0.0008 over 8h -> per-hour premium 0.0001 -> 1 bps; residual 3
        monkeypatch.setattr(funding_archive, 'get_funding_series',
                            lambda sym: pd.Series(np.full(300, 0.0008), index=idx),
                            raising=False)
        out = ba.basis_features_for_index('BTC/USD', idx)
        assert 'Basis_minus_Funding' in out
        assert out['Basis_minus_Funding'][-1] == pytest.approx(3.0, abs=1e-6)
