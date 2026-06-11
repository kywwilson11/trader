"""Tests for cross-sectional panel ranks + ROD/periodicity features."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panel_ranks
from panel_ranks import (add_panel_ranks, compute_live_panel_ranks,
                         neutral_fill_cs)


def _panel_df(values_by_symbol, ts='2026-06-10 15:00', cols=None):
    """One-timestamp panel frame in harvest layout (rows per ticker)."""
    cols = cols or ['Return_4h']
    frames = []
    for sym, vals in values_by_symbol.items():
        row = {c: v for c, v in zip(cols, vals if isinstance(vals, (list, tuple)) else [vals])}
        row['Ticker'] = sym
        frames.append(pd.DataFrame(row, index=[pd.Timestamp(ts, tz='UTC')]))
    return pd.concat(frames)


class TestHarvestRanks:
    def test_rank_bounds_and_ordering(self):
        df = _panel_df({'A': -2.0, 'B': 0.0, 'C': 1.0, 'D': 3.0})
        out = add_panel_ranks(df)
        r = out.set_index('Ticker')['CS_Rank_Return_4h']
        assert r['A'] == pytest.approx(-1.0)   # bottom of the panel
        assert r['D'] == pytest.approx(1.0)    # top of the panel
        assert r['B'] == pytest.approx(-1 / 3)
        assert r['A'] < r['B'] < r['C'] < r['D']
        assert (out['CS_Rank_Return_4h'].abs() <= 1.0).all()

    def test_ranks_are_per_timestamp(self):
        df1 = _panel_df({'A': 5.0, 'B': 1.0}, ts='2026-06-10 14:00')
        df2 = _panel_df({'A': 1.0, 'B': 5.0}, ts='2026-06-10 15:00')
        out = add_panel_ranks(pd.concat([df1, df2]))
        a = out[out['Ticker'] == 'A']['CS_Rank_Return_4h']
        assert a.iloc[0] == pytest.approx(1.0)    # top at 14:00
        assert a.iloc[1] == pytest.approx(-1.0)   # bottom at 15:00

    def test_single_member_neutral(self):
        out = add_panel_ranks(_panel_df({'A': 7.0}))
        assert out['CS_Rank_Return_4h'].iloc[0] == 0.0

    def test_context_columns(self):
        df = _panel_df({'A': [1.0, 1.2], 'B': [3.0, 0.8], 'C': [-1.0, 1.1],
                        'D': [1.0, 0.9]},
                       cols=['Return_4h', 'Price_SMA20_Ratio'])
        out = add_panel_ranks(df)
        assert out['CS_Dispersion'].iloc[0] == pytest.approx(
            np.std([1.0, 3.0, -1.0, 1.0], ddof=1))
        # 2 of 4 above trend -> centered breadth 0
        assert out['CS_Breadth'].iloc[0] == pytest.approx(0.0)

    def test_neutral_fill(self):
        df = _panel_df({'A': 1.0, 'B': 2.0})
        out = add_panel_ranks(df)
        out.loc[out.index[0], 'CS_Rank_Return_4h'] = np.nan
        filled = neutral_fill_cs(out)
        assert filled['CS_Rank_Return_4h'].isna().sum() == 0


class TestLiveParity:
    """The red team's survival conditional: live ranks must match what
    the harvest would produce on the same cross-section."""

    def _mk_live(self, monkeypatch, panel_vals, top_k=60):
        idx = pd.date_range('2026-06-08 14:30', periods=80, freq='h',
                            tz='UTC')

        def fake_bars(api, sym):
            base = pd.DataFrame({
                'Open': 100.0, 'High': 101.0, 'Low': 99.0,
                'Close': 100.0, 'Volume': panel_vals[sym]['dv'] / 100.0 / 7,
            }, index=idx)
            return base

        def fake_features(bars, spy_close=None):
            out = bars.copy()
            sym_dv = bars['Volume'].iloc[-1]
            for s, v in panel_vals.items():
                if abs(v['dv'] / 100.0 / 7 - sym_dv) < 1e-6:
                    for c in panel_ranks.CS_RANK_BASE_COLS:
                        out[c] = v.get(c, 0.0)
                    break
            return out

        import market_data, indicators
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca', fake_bars)
        monkeypatch.setattr(indicators, 'compute_stock_features',
                            fake_features)
        monkeypatch.setattr(panel_ranks, '_panel_symbols',
                            lambda: list(panel_vals))
        panel_ranks._live_cache = None
        return compute_live_panel_ranks(api=object(), top_k=top_k)

    def test_live_matches_harvest_rank_math(self, monkeypatch):
        vals = {f'S{i}': {'dv': (i + 1) * 1e7, 'Return_4h': float(i)}
                for i in range(12)}
        live = self._mk_live(monkeypatch, vals)
        # Harvest-side ranks on the identical cross-section
        hdf = _panel_df({s: v['Return_4h'] for s, v in vals.items()})
        hout = add_panel_ranks(hdf).set_index('Ticker')
        for s in vals:
            assert live[s]['CS_Rank_Return_4h'] == pytest.approx(
                hout.loc[s, 'CS_Rank_Return_4h']), s

    def test_live_topk_membership_mask(self, monkeypatch):
        # 12 names, top_k=8: the 4 thinnest must be excluded from the
        # ranked output entirely (they read neutral 0.0 at injection)
        vals = {f'S{i}': {'dv': (i + 1) * 1e7, 'Return_4h': float(i)}
                for i in range(12)}
        live = self._mk_live(monkeypatch, vals, top_k=8)
        assert set(live) == {f'S{i}' for i in range(4, 12)}

    def test_live_failure_returns_empty(self, monkeypatch):
        import market_data

        def boom(api, sym):
            raise OSError('feed down')

        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca', boom)
        monkeypatch.setattr(panel_ranks, '_panel_symbols',
                            lambda: ['A', 'B'])
        panel_ranks._live_cache = None
        assert compute_live_panel_ranks(api=object()) == {}


class TestRODFeatures:
    def _bars(self, closes_by_day):
        rows, idx = [], []
        for day, closes in closes_by_day.items():
            for h, c in enumerate(closes):
                idx.append(pd.Timestamp(f'{day} {14 + h}:30', tz='UTC'))
                rows.append(c)
        df = pd.DataFrame({'Close': rows}, index=pd.DatetimeIndex(idx))
        for c in ('Open', 'High', 'Low'):
            df[c] = df['Close']
        df['Volume'] = 1e6
        return df

    def test_rod_ret_vs_prior_session_close(self):
        from indicators import compute_stock_features
        df = self._bars({'2026-06-08': [100, 101, 102],
                         '2026-06-09': [99, 100.5, 103]})
        out = compute_stock_features(df)
        rod = out['ROD_Ret']
        # Day 2 bars measure vs day-1 close (102)
        assert rod.iloc[3] == pytest.approx((99 / 102 - 1) * 100)
        assert rod.iloc[5] == pytest.approx((103 / 102 - 1) * 100)
        # Day 1 has no prior session -> NaN (dropna/neutral handles it)
        assert rod.iloc[:3].isna().all()

    def test_same_hour_mean_excludes_today(self):
        from indicators import compute_stock_features
        days = {f'2026-05-{d:02d}': [100, 100 * (1 + 0.01 * (d % 2)), 100]
                for d in range(1, 30)}
        out = compute_stock_features(self._bars(days))
        shm = out['Same_Hour_Mean_40d']
        assert shm.notna().sum() > 0
        # The 15:30 bar alternates +1%/0% daily -> trailing mean ~0.5%
        bar2 = shm[shm.index.hour == 15].dropna()
        assert bar2.iloc[-1] == pytest.approx(0.5, abs=0.2)


class TestBouncedLoser:
    def test_detector_logic(self, monkeypatch):
        import stock_loop as sl
        loop = object.__new__(sl.StockLoop)  # no __init__ (no API)
        loop.api = object()

        def bars_for(closes_today, prev_close=100.0):
            idx = (list(pd.date_range('2026-06-09 14:30', periods=30,
                                      freq='h', tz='UTC'))
                   + [pd.Timestamp(f'2026-06-10 {14 + i}:30', tz='UTC')
                      for i in range(len(closes_today))])
            vals = [prev_close] * 30 + closes_today
            df = pd.DataFrame({'Close': vals}, index=pd.DatetimeIndex(idx))
            return df

        import market_data
        # Loser (-3% at 15:00 mark) that bounced +1.5% into the close
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s: bars_for([98.0, 97.0, 98.5]))
        assert loop._is_bounced_loser('XYZ') is True
        # Loser that kept falling: no bounce
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s: bars_for([98.0, 97.0, 96.5]))
        assert loop._is_bounced_loser('XYZ') is False
        # Winner all day: not a loser
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s: bars_for([102.0, 103.0, 104.0]))
        assert loop._is_bounced_loser('XYZ') is False
