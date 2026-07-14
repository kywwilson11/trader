"""Tests for gate attribution / conviction calibration replays."""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decision_report
from decision_report import replay_entry, gate_attribution, load_journal


def _bars(path_returns, start='2026-06-01', hours_per_day=24):
    """Crypto-style continuous hourly bars following path_returns."""
    closes = 100 * np.cumprod(1 + np.asarray(path_returns))
    idx = pd.date_range(start, periods=len(closes), freq='h', tz='UTC')
    df = pd.DataFrame({'Close': closes}, index=idx)
    df['Open'] = df['Close'].shift(1).fillna(100.0)
    df['High'] = df[['Open', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.999
    df['Volume'] = 1e6
    return df


class TestReplay:
    def test_winning_path_positive_net(self):
        # Steady +0.5%/bar -> TP or vertical exits well above costs
        bars = _bars([0.005] * 60)
        net = replay_entry(bars, bars.index[10], 'crypto')
        assert net is not None and net > 1.0

    def test_crashing_path_stopped_out(self):
        bars = _bars([0.0] * 20 + [-0.04] * 10 + [0.0] * 30)
        net = replay_entry(bars, bars.index[18], 'crypto')
        assert net is not None and net < -1.0

    def test_unresolved_at_edge_returns_none(self):
        bars = _bars([0.001] * 30)
        assert replay_entry(bars, bars.index[-1], 'crypto') is None

    def test_timestamp_after_data_none(self):
        bars = _bars([0.001] * 30)
        late = bars.index[-1] + pd.Timedelta(hours=5)
        assert replay_entry(bars, late, 'crypto') is None


class TestGateAttribution:
    def test_attribution_from_synthetic_journal(self, monkeypatch):
        # Veto on a name that would have crashed (gate saves) and one
        # that would have mooned (gate costs)
        crash = _bars([0.0] * 10 + [-0.05] * 8 + [0.0] * 40)
        moon = _bars([0.008] * 60)

        import market_data
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s: crash if s == 'BAD/USD' else moon)
        rows = [
            {'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'BAD/USD',
             'ts': str(crash.index[8])},
            {'action': 'skip', 'skip_reason': 'llm_veto', 'symbol': 'GOOD/USD',
             'ts': str(moon.index[8])},
        ]
        out = gate_attribution(rows, api=object())
        assert out['meta_veto']['counterfactual_mean_net_pct'] < -1
        assert out['meta_veto']['saved_total_pct'] > 1     # earned its keep
        assert out['llm_veto']['counterfactual_mean_net_pct'] > 1
        assert out['llm_veto']['saved_total_pct'] < -1     # charged admission

    def test_journal_loader_tolerates_garbage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', tmp_path)
        today = dt.date.today().isoformat()
        (tmp_path / f'{today}.jsonl').write_text(
            '{"action": "skip"}\nnot json\n{"action": "buy"}\n')
        rows = load_journal(1)
        assert len(rows) == 2
