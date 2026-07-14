"""Tests for gate attribution / conviction calibration replays.

2026-07 additions cover the review fixes: fetch-failure vs horizon-pending
split, per-episode dedup, deterministic bootstrap CIs, qty_zero pricing,
the always-write-stale-JSON behavior, the stock truncated-horizon bug fix,
and the cost_floor flat-spread caveat.
"""

import contextlib
import datetime as dt
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decision_report
from decision_report import (
    replay_entry, gate_attribution, load_journal, signal_exit_audit,
    conviction_calibration, _replay_grouped, _dedup_first_per_day,
    _bootstrap_ci, run_report, GATE_REASONS,
)


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


# ===========================================================================
# 2026-07 review additions
# ===========================================================================

def _stub_dotenv(monkeypatch):
    """dotenv isn't installed on this dev Mac (see CLAUDE.md); stub it in
    sys.modules so trading_utils / decision_report's lazy `from dotenv
    import load_dotenv` succeeds and we can reach the get_api() call."""
    fake = types.ModuleType('dotenv')
    fake.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'dotenv', fake)


@contextlib.contextmanager
def _isolated_trading_utils(monkeypatch):
    """Import trading_utils with dotenv stubbed, then evict it from
    sys.modules on exit so a successful import here doesn't leak into later
    test MODULES in the same session — other files (test_imports,
    test_gpu_lock, test_new_modules) deliberately exercise the real
    dev-Mac ModuleNotFoundError path (dotenv missing) and must see it
    regardless of what ran first. monkeypatch alone won't do this: it only
    reverts sys.modules['dotenv'], but the plain `import trading_utils`
    statement caches the successfully-imported module under its OWN key,
    which monkeypatch never touched and so never restores."""
    pre_existing = 'trading_utils' in sys.modules
    _stub_dotenv(monkeypatch)
    import trading_utils
    try:
        yield trading_utils
    finally:
        if not pre_existing:
            sys.modules.pop('trading_utils', None)


class TestReplayGrouped:
    """FETCH-FAILURE (raising fetcher / too-short bars) must be counted
    separately from HORIZON-PENDING (bars are fine, but the exit horizon
    hasn't resolved yet against them) — folding both into one bucket, the
    pre-2026-07 behavior, hides API outages behind an innocuous number."""

    def test_fetch_failure_vs_horizon_pending(self, monkeypatch):
        good = _bars([0.001] * 60)     # plenty of bars, calm path
        short = _bars([0.001] * 10)    # too short to be tradeable (<30 bars)

        import market_data

        def fetcher(api, sym):
            if sym == 'RAISE/USD':
                raise RuntimeError('api outage')
            if sym == 'SHORT/USD':
                return short
            return good

        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', fetcher)
        rows = [
            {'symbol': 'RAISE/USD', 'ts': str(good.index[5])},
            {'symbol': 'SHORT/USD', 'ts': str(short.index[2])},
            # near the very end of a long, calm frame -> the horizon can't
            # resolve yet (replay_entry's own edge guard returns None)
            {'symbol': 'OK/USD', 'ts': str(good.index[-2])},
        ]
        samples, n_fetch_failed, n_horizon_pending = decision_report._replay_grouped(
            rows, api=object())
        assert n_fetch_failed == 2      # RAISE/USD (raised) + SHORT/USD (too short)
        assert n_horizon_pending == 1   # OK/USD
        assert samples == []


class TestDedup:
    """A symbol skipped for the same reason every cycle of a day otherwise
    emits ~24 overlapping 24-bar replays/day, double-counting the same P&L
    into saved_total_pct — dedupe to one episode per
    (symbol, skip_reason, calendar-day) BEFORE replay."""

    def test_dedup_keeps_first_per_symbol_reason_day(self):
        day1 = pd.Timestamp('2026-06-01T09:00', tz='UTC')
        rows = ([
            {'symbol': 'AAA', 'skip_reason': 'meta_veto',
             'ts': str(day1 + pd.Timedelta(hours=h))}
            for h in range(5)
        ] + [
            {'symbol': 'AAA', 'skip_reason': 'meta_veto',            # next day
             'ts': str(day1 + pd.Timedelta(days=1))},
            {'symbol': 'AAA', 'skip_reason': 'llm_veto', 'ts': str(day1)},   # diff reason
            {'symbol': 'BBB', 'skip_reason': 'meta_veto', 'ts': str(day1)},  # diff symbol
        ])
        out = _dedup_first_per_day(rows, ['symbol', 'skip_reason'])
        assert len(out) == 4    # 5 same-day AAA/meta_veto rows -> 1 + the 3 distinct keys
        aaa_meta_day1 = [r for r in out if r['symbol'] == 'AAA'
                        and r['skip_reason'] == 'meta_veto'
                        and pd.Timestamp(r['ts']).date() == day1.date()]
        assert len(aaa_meta_day1) == 1
        assert aaa_meta_day1[0]['ts'] == str(day1)   # earliest-in-day kept

    def test_gate_attribution_dedups_before_replay(self, monkeypatch):
        bars = _bars([0.002] * 200)
        import market_data
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', lambda api, s: bars)
        day1 = bars.index[5]
        # same symbol/reason skipped every hour of the same day -> one episode
        rows = [{'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'AAA/USD',
                 'ts': str(day1 + pd.Timedelta(hours=h))} for h in range(6)]
        out = gate_attribution(rows, api=object())
        assert out['meta_veto']['vetoes_raw'] == 6
        assert out['meta_veto']['vetoes_priced'] == 1


class TestBootstrapCI:
    def test_deterministic_and_brackets_mean(self):
        vals = [1.0, -0.5, 2.0, 0.3, -1.2, 0.8, 1.5, -0.2, 0.6, 2.2]
        lo1, hi1 = _bootstrap_ci(vals, seed=0)
        lo2, hi2 = _bootstrap_ci(vals, seed=0)
        assert (lo1, hi1) == (lo2, hi2)          # deterministic
        assert lo1 <= float(np.mean(vals)) <= hi1  # CI brackets the sample mean

    def test_single_value_degenerate_ci(self):
        assert _bootstrap_ci([1.234]) == (1.234, 1.234)

    def test_empty_returns_nan(self):
        lo, hi = _bootstrap_ci([])
        assert np.isnan(lo) and np.isnan(hi)


class TestQtyZero:
    def test_qty_zero_in_gate_reasons(self):
        assert 'qty_zero' in GATE_REASONS

    def test_qty_zero_rows_now_priced(self, monkeypatch):
        bars = _bars([0.002] * 60)
        import market_data
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', lambda api, s: bars)
        rows = [{'action': 'skip', 'skip_reason': 'qty_zero', 'symbol': 'AAA/USD',
                 'ts': str(bars.index[5])}]
        out = gate_attribution(rows, api=object())
        assert 'qty_zero' in out
        assert out['qty_zero']['vetoes_priced'] == 1


class TestStockHorizonFix:
    def test_stock_truncated_horizon_now_pending(self):
        # short, flat, single-calendar-day frame: only 3 bars remain after
        # the entry point, well under the 24-bar vertical barrier. Pre-2026-07
        # the unresolved-horizon guard only fired `and asset_type == 'crypto'`,
        # so this STOCK replay would have been silently priced instead of
        # counted horizon-pending.
        bars = _bars([0.0005] * 12)
        net = replay_entry(bars, bars.index[-3], 'stock')
        assert net is None

    def test_crypto_truncated_horizon_still_pending(self):
        # same shape, crypto — must be unaffected by the fix
        bars = _bars([0.0005] * 12)
        net = replay_entry(bars, bars.index[-3], 'crypto')
        assert net is None


class TestStaleReport:
    def test_stale_json_written_on_api_failure(self, tmp_path, monkeypatch):
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        today = dt.date.today().isoformat()
        (journal_dir / f'{today}.jsonl').write_text(json.dumps(
            {'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'AAA/USD',
             'ts': '2026-06-01T00:00:00+00:00'}) + '\n')
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            def _raise_get_api():
                raise RuntimeError('no credentials')
            monkeypatch.setattr(trading_utils, 'get_api', _raise_get_api)

            report = run_report(days=1)

        assert report['stale'] is True
        assert report['api_available'] is False

        out_path = tmp_path / 'decision_report.json'
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text())
        assert on_disk['stale'] is True
        assert on_disk['api_available'] is False

    def test_stale_json_written_on_empty_journal(self, tmp_path, monkeypatch):
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        report = run_report(days=1)
        assert report['stale'] is True
        out_path = tmp_path / 'decision_report.json'
        assert out_path.exists()


class TestCostFloorCaveat:
    def test_cost_floor_flat_spread_caveat_printed(self, tmp_path, monkeypatch, capsys):
        bars = _bars([0.001] * 200)
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        today = dt.date.today().isoformat()
        row = {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
               'ts': str(bars.index[5])}
        (journal_dir / f'{today}.jsonl').write_text(json.dumps(row) + '\n')
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        import market_data
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', lambda api, s: bars)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca', lambda api, s: bars)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            run_report(days=1)

        captured = capsys.readouterr()
        assert 'cost_floor priced at FLAT spread' in captured.out
        assert 'structurally unreliable' in captured.out
