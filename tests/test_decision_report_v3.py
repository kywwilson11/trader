"""2026-07b decision_report.py hardening tests.

Covers: the tz-naive dedup sentinel crash fix, out-of-window replay
exclusion (vs. the old silent-price-from-bar-0 bug), the full-frame ATR
stop fix (entry-bar-slice ATR was always NaN), verdict/bucket floors
(MIN_VERDICT_N / MIN_BUCKET_N), the meta_p null-fallback fix, run_report's
per-section error isolation + quality/journal_flags block, atomic writes,
the shared bars_cache fetch-once behavior, admitted_k_distribution's
malformed-value tolerance, gate-attribution drift counters
(_gates_seen_unpriced / _unclassified_skip_reasons), load_journal's
action-filter + field projection, and cost_floor spread coverage.

Pure numpy/pandas/stdlib — no torch/lightgbm/joblib/dotenv anywhere in
this module or the code paths it exercises directly (dotenv is stubbed
into sys.modules, exactly like tests/test_decision_report.py, only for
the handful of tests that go through run_report's get_api() call).
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
import market_data
from decision_report import (
    replay_entry, gate_attribution, conviction_calibration,
    signal_exit_audit, admitted_k_distribution, load_journal, run_report,
    _replay_grouped, _dedup_first_per_day, _gate_verdict,
    _signal_exit_verdict, _atr14, _eod_mask, GATE_REASONS, UNPRICED_GATES,
)

REPO = Path(__file__).resolve().parent.parent


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


# ===========================================================================
# Dedup timestamp handling (tz-naive sentinel crash fix)
# ===========================================================================

class TestDedupTz:
    def test_mixed_naive_aware_no_raise(self):
        rows = [{'symbol': 'A', 'skip_reason': 'm', 'ts': '2026-07-05T09:00:00'},
                {'symbol': 'A', 'skip_reason': 'm',
                 'ts': '2026-07-20T09:00:00-07:00'}]
        out = _dedup_first_per_day(rows, ['symbol', 'skip_reason'])
        assert len(out) == 2

    def test_missing_and_garbage_ts_tolerated(self):
        aware = {'symbol': 'A', 'skip_reason': 'm',
                 'ts': '2026-07-05T09:00:00+00:00'}
        missing = {'symbol': 'B', 'skip_reason': 'm'}
        garbage = {'symbol': 'C', 'skip_reason': 'm', 'ts': 'garbage'}
        out = _dedup_first_per_day([aware, missing, garbage],
                                   ['symbol', 'skip_reason'])
        assert len(out) == 3
        syms = [r['symbol'] for r in out]
        assert syms.index('A') < syms.index('B')
        assert syms.index('A') < syms.index('C')

    def test_all_aware_ordering_unchanged(self):
        # replicates tests/test_decision_report.py's
        # test_dedup_keeps_first_per_symbol_reason_day, but with -07:00
        # offsets instead of UTC — the rewritten int64 sort key must give
        # the identical shape/ordering for uniform-tz input.
        day1 = pd.Timestamp('2026-06-01T09:00:00-07:00')
        rows = ([
            {'symbol': 'AAA', 'skip_reason': 'meta_veto',
             'ts': str(day1 + pd.Timedelta(hours=h))}
            for h in range(5)
        ] + [
            {'symbol': 'AAA', 'skip_reason': 'meta_veto',            # next day
             'ts': str(day1 + pd.Timedelta(days=1))},
            {'symbol': 'AAA', 'skip_reason': 'llm_veto', 'ts': str(day1)},
            {'symbol': 'BBB', 'skip_reason': 'meta_veto', 'ts': str(day1)},
        ])
        out = _dedup_first_per_day(rows, ['symbol', 'skip_reason'])
        assert len(out) == 4
        aaa_meta_day1 = [r for r in out if r['symbol'] == 'AAA'
                        and r['skip_reason'] == 'meta_veto'
                        and pd.Timestamp(r['ts']).date() == day1.date()]
        assert len(aaa_meta_day1) == 1
        assert aaa_meta_day1[0]['ts'] == str(day1)

    def test_day_bucket_row_wall_clock(self):
        # same UTC instant-ish window, but different OWN wall-clock dates
        # (2026-07-01 vs 2026-07-02) -> the day bucket must use each row's
        # own tz, not a UTC-normalized date, so both survive dedup.
        rows = [{'symbol': 'A', 'skip_reason': 'm',
                 'ts': '2026-07-01T23:00:00-04:00'},
                {'symbol': 'A', 'skip_reason': 'm',
                 'ts': '2026-07-02T01:00:00+00:00'}]
        out = _dedup_first_per_day(rows, ['symbol', 'skip_reason'])
        assert len(out) == 2


# ===========================================================================
# Out-of-window replay exclusion
# ===========================================================================

class TestOutOfWindow:
    def test_replay_before_frame_none(self):
        bars = _bars([0.004] * 250, start='2026-07-15')
        assert replay_entry(bars, bars.index[0] - pd.Timedelta(days=5),
                            'crypto') is None

    def test_replay_grouped_counts(self, monkeypatch):
        bars = _bars([0.001] * 60)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'symbol': 'AAA/USD',
                 'ts': str(bars.index[0] - pd.Timedelta(days=5))}]
        result = _replay_grouped(rows, api=object())
        assert result == ([], 0, 0, 1)

    def test_gate_attribution_excludes(self, monkeypatch):
        bars = _bars([0.001] * 200, start='2026-07-01')
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'skip', 'skip_reason': 'meta_veto',
                 'symbol': 'AAA/USD',
                 'ts': str(bars.index[0] - pd.Timedelta(days=20 + i))}
                for i in range(12)]
        out = gate_attribution(rows, api=object())
        assert 'meta_veto' not in out
        assert out['_out_of_window'] == 12
        assert out['_unresolved'] == 12
        assert out['_fetch_failed'] == 0


# ===========================================================================
# Full-frame ATR stop fix
# ===========================================================================

class TestAtrFix:
    def test_atr_stop_not_fallback(self):
        bars = _bars([0.001] * 30 + [-0.04] * 6 + [0.0] * 30)
        net = replay_entry(bars, bars.index[30], 'crypto')
        assert net is not None and -3.5 < net < -1.0
        # documents the old bug mechanism: ATR computed on the entry-bar
        # SLICE has no history to roll over -> NaN head
        assert np.isnan(_atr14(bars.iloc[30:])[0])

    def test_hoisted_equals_default(self):
        bars = _bars([0.003] * 100 + [-0.01] * 50 + [0.002] * 150)
        for asset in ('crypto', 'stock'):
            af = _atr14(bars)
            ef = _eod_mask(bars.index, asset)
            for i in range(5, 280, 13):
                a = replay_entry(bars, bars.index[i], asset)
                b = replay_entry(bars, bars.index[i], asset,
                                 atr_full=af, eod_full=ef)
                assert a == b or (a is None and b is None)


# ===========================================================================
# Verdict floor (MIN_VERDICT_N)
# ===========================================================================

class TestVerdictFloor:
    def test_gate_verdict_floor(self):
        for n in (1, 2, 3, 8):
            v = _gate_verdict(n, (0.4, 0.4))
            assert 'insufficient n' in v
            assert 'REVIEW' not in v
        assert _gate_verdict(9, (0.3, 0.5)) == 'REVIEW (charging admission — CI excludes zero)'
        assert _gate_verdict(9, (-0.5, -0.3)) == 'OK (earning its keep — CI excludes zero)'
        assert _gate_verdict(9, (-0.1, 0.1)) == 'cannot conclude (CI spans zero)'

    def test_signal_exit_verdict_floor(self):
        for n in (1, 2, 3, 8):
            v = _signal_exit_verdict(n, (0.4, 0.4))
            assert 'insufficient n' in v
            assert 'CHANGE' not in v
        assert _signal_exit_verdict(9, (0.3, 0.5)) == (
            'CHANGE — apply 2-reading confirmation to signal exits (CI excludes zero)')
        assert _signal_exit_verdict(9, (-0.5, -0.3)) == (
            'NO CHANGE — the flip is saving money (CI excludes zero)')
        assert _signal_exit_verdict(9, (-0.1, 0.1)) == 'cannot conclude (CI spans zero)'

    def test_single_row_gate_flagged(self, monkeypatch):
        bars = _bars([0.001] * 60)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'skip', 'skip_reason': 'meta_veto',
                 'symbol': 'AAA/USD', 'ts': str(bars.index[5])}]
        out = gate_attribution(rows, api=object())
        gate = out['meta_veto']
        assert gate['insufficient_n'] is True
        assert 'insufficient n' in gate['verdict']
        for key in ('vetoes_priced', 'counterfactual_mean_net_pct', 'ci90'):
            assert key in gate


# ===========================================================================
# Rank-bucket MIN_BUCKET_N suppression + coverage disclosure
# ===========================================================================

class TestRankBuckets:
    def test_min_n_suppression(self, monkeypatch):
        bars = _bars([0.001] * 200)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'buy', 'symbol': 'AAA/USD', 'pred_return': 0.01,
                 'ts': str(bars.index[10 + h]),
                 'entry_rank': 1 if h < 12 else 6} for h in range(13)]
        out = conviction_calibration(rows, api=object())
        assert 'rank_1_3' in out and out['rank_1_3']['n'] == 12
        assert 'rank_6_7' not in out
        assert out['_rank_buckets_suppressed'] == {'rank_6_7': 1}

    def test_rank_coverage(self, monkeypatch):
        bars = _bars([0.001] * 200)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'buy', 'symbol': 'AAA', 'pred_return': 0.01,
                 'ts': str(bars.index[10 + h]), 'entry_rank': (h % 7) + 1}
                for h in range(9)]
        rows += [{'action': 'buy', 'symbol': 'BTC/USD', 'pred_return': 0.01,
                  'ts': str(bars.index[30 + h])} for h in range(5)]
        out = conviction_calibration(rows, api=object())
        assert out['rank_coverage'] == {
            'n_total': 14, 'n_with_rank': 9,
            'stock_with_rank': 9, 'crypto_with_rank': 0,
        }

    def test_rank_coverage_excludes_malformed_pred(self, monkeypatch):
        # a malformed-pred_return row carrying a rank must NOT count toward
        # per-asset rank coverage: stock_with_rank + crypto_with_rank ==
        # n_with_rank always (coverage is counted post pred-validation).
        bars = _bars([0.001] * 200)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'buy', 'symbol': 'AAA/USD', 'pred_return': 0.01,
                 'ts': str(bars.index[10 + h]), 'entry_rank': 1}
                for h in range(9)]
        rows.append({'action': 'buy', 'symbol': 'AAA/USD',
                     'pred_return': 'garbage', 'ts': str(bars.index[40]),
                     'entry_rank': 2})
        out = conviction_calibration(rows, api=object())
        rc = out['rank_coverage']
        assert rc['n_with_rank'] == rc['stock_with_rank'] + rc['crypto_with_rank'] == 9
        assert rc['crypto_with_rank'] == 9
        assert out['_malformed_pred_return'] == 1


# ===========================================================================
# meta_p None -> meta_prob fallback
# ===========================================================================

class TestMetaFallback:
    def test_meta_p_null_falls_back(self, monkeypatch):
        bars = _bars([0.001] * 200)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'buy', 'symbol': 'AAA/USD', 'pred_return': 0.01,
                 'ts': str(bars.index[10 + h]),
                 'meta_p': None, 'meta_prob': 0.5} for h in range(9)]
        out = conviction_calibration(rows, api=object())
        assert 'meta_0.45_0.60' in out
        assert out['meta_0.45_0.60']['n'] == 9


# ===========================================================================
# run_report: per-section error isolation, quality block, journal_flags
# ===========================================================================

class TestRunReportFailClosed:
    def _seed(self, tmp_path, monkeypatch, rows):
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        today = dt.date.today().isoformat()
        with open(journal_dir / f'{today}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

    def test_analysis_error_writes_stale(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch, [
            {'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'AAA/USD',
             'ts': '2026-06-01T00:00:00+00:00'}])

        def _raise(*a, **k):
            raise RuntimeError('boom')
        monkeypatch.setattr(decision_report, 'gate_attribution', _raise)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            report = run_report(days=1)

        assert report['stale'] is True
        assert 'gates' in report['errors']
        assert 'boom' in report['errors']['gates']
        on_disk = json.loads((tmp_path / 'decision_report.json').read_text())
        assert on_disk['stale'] is True
        assert 'gates' in on_disk['errors']

    def test_zero_priced_marked_stale(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch, [
            {'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'AAA/USD',
             'ts': '2026-06-01T00:00:00+00:00'}])

        def _raise(*a, **k):
            raise RuntimeError('down')
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', _raise)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca', _raise)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            report = run_report(days=1)

        assert report['stale'] is True
        assert report.get('stale_reason')
        assert report['quality']['fetch_failed'] == 1
        assert report['quality']['representative'] is False
        on_disk = json.loads((tmp_path / 'decision_report.json').read_text())
        assert on_disk['stale'] is True

    def test_healthy_quality_and_flags(self, tmp_path, monkeypatch):
        bars = _bars([0.001] * 200)
        self._seed(tmp_path, monkeypatch, [
            {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
             'ts': str(bars.index[5])}])
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s, **k: bars)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            report = run_report(days=1)

        assert 'stale' not in report
        assert report['quality']['representative'] is True
        # journal_flags must TRUTHFULLY reflect the config sources: the
        # conviction flag mirrors strategy_config (whatever the owner set
        # it to — hardcoding True here would fail spuriously on a config
        # flip); the llm flag defaults True because BASE_DIR is tmp_path,
        # so no llm_config.json exists.
        import strategy_config
        assert report['journal_flags'] == {
            'conviction_journal_enabled': bool(strategy_config.CONVICTION_JOURNAL_ENABLED),
            'llm_journal_enabled': True}
        assert dt.datetime.fromisoformat(report['generated']).tzinfo is not None

    def test_days_clamped(self, tmp_path, monkeypatch):
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        report = run_report(days=-5)
        assert report['stale'] is True
        assert report['days'] >= 0


# ===========================================================================
# Atomic writes
# ===========================================================================

class TestAtomicWrite:
    def test_no_tmp_left(self, tmp_path, monkeypatch):
        bars = _bars([0.001] * 200)
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        today = dt.date.today().isoformat()
        row = {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
               'ts': str(bars.index[5])}
        (journal_dir / f'{today}.jsonl').write_text(json.dumps(row) + '\n')
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda api, s, **k: bars)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            run_report(days=1)

        assert list(tmp_path.glob('*.tmp')) == []
        json.loads((tmp_path / 'decision_report.json').read_text())   # no raise

    def test_torn_write_preserves_previous(self, tmp_path, monkeypatch):
        (tmp_path / 'decision_report.json').write_text('{"ok": 1}')
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        def _raise(*a, **k):
            raise RuntimeError('boom')
        monkeypatch.setattr(decision_report.json, 'dumps', _raise)

        with pytest.raises(RuntimeError):
            decision_report._write_stale_report(1, api_available=False)

        on_disk = json.loads((tmp_path / 'decision_report.json').read_text())
        assert on_disk == {'ok': 1}


# ===========================================================================
# Shared bars_cache: fetch once per symbol per report
# ===========================================================================

class TestFetchOnce:
    def test_symbol_fetched_once(self, tmp_path, monkeypatch):
        bars = _bars([0.001] * 200)
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        today = dt.date.today().isoformat()
        rows = [
            {'action': 'skip', 'skip_reason': 'meta_veto', 'symbol': 'AAA/USD',
             'ts': str(bars.index[5])},
            {'action': 'buy', 'symbol': 'AAA/USD', 'pred_return': 0.01,
             'ts': str(bars.index[5])},
            {'action': 'sell', 'exit_reason': 'signal_sell', 'symbol': 'AAA/USD',
             'pnl_pct': 1.0, 'ts': str(bars.index[5])},
        ]
        with open(journal_dir / f'{today}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)

        calls = {'n': 0}

        def counting_fetch(api, sym, **k):
            calls['n'] += 1
            return bars
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', counting_fetch)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca', counting_fetch)

        with _isolated_trading_utils(monkeypatch) as trading_utils:
            monkeypatch.setattr(trading_utils, 'get_api', lambda: object())
            run_report(days=1)

        assert calls['n'] == 1


# ===========================================================================
# admitted_k_distribution: malformed-value tolerance + fail-closed surfacing
# ===========================================================================

class TestAdmittedK:
    def test_malformed_tolerated(self):
        rows = [
            {'action': 'entry_window', 'asset_type': 'stock', 'admitted_k': None},
            {'action': 'entry_window', 'asset_type': 'stock', 'admitted_k': 'abc'},
            {'action': 'entry_window', 'asset_type': 'stock', 'admitted_k': 5},
        ]
        out = admitted_k_distribution(rows)
        assert out['stock']['mean_admitted_k'] == 5.0
        assert out['stock']['_malformed_admitted_k'] == 2

    def test_fail_closed_and_candidates(self):
        row = {'action': 'entry_window', 'asset_type': 'stock', 'admitted_k': 1,
               'n_candidates': 4,
               'veto_counts': {'no_pred': 3, 'no_quote': 1, 'trade_budget': 2}}
        out = admitted_k_distribution([row])
        assert out['stock']['fail_closed'] == {'no_pred': 3, 'no_quote': 1}
        assert out['stock']['mean_n_candidates'] == 4.0
        assert 'pct_windows_zero_note' in out['stock']


# ===========================================================================
# Gate drift counters
# ===========================================================================

class TestGateCounters:
    def test_unclassified_counted(self, monkeypatch):
        bars = _bars([0.001] * 60)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [{'action': 'skip', 'skip_reason': 'brand_new_gate',
                 'symbol': 'AAA/USD', 'ts': str(bars.index[5])}]
        out = gate_attribution(rows, api=object())
        assert out['_unclassified_skip_reasons'] == {'brand_new_gate': 1}

    def test_gates_seen_unpriced(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError('down')
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca', _raise)
        rows = [{'action': 'skip', 'skip_reason': 'meta_veto',
                 'symbol': 'AAA/USD', 'ts': '2026-06-01T00:00:00+00:00'}]
        out = gate_attribution(rows, api=object())
        assert out['_gates_seen_unpriced'] == {'meta_veto': 1}
        assert 'meta_veto' not in out

    def test_unpriced_gates_matches_producers(self):
        src = ((REPO / 'base_loop.py').read_text()
               + (REPO / 'stock_loop.py').read_text())
        for name in UNPRICED_GATES:
            assert f"vc['{name}']" in src, f"UNPRICED_GATES has a phantom key: {name}"
        assert 'budget' not in UNPRICED_GATES
        assert 'trade_budget' in UNPRICED_GATES


# ===========================================================================
# load_journal: action filter + field projection
# ===========================================================================

class TestLoadJournalFilter:
    def test_filters_and_projects(self, tmp_path, monkeypatch):
        journal_dir = tmp_path / 'journals'
        journal_dir.mkdir()
        monkeypatch.setattr(decision_report, 'JOURNAL_DIR', journal_dir)
        today = dt.date.today().isoformat()
        lines = [
            json.dumps({'action': 'skip', 'skip_reason': 'meta_veto',
                       'symbol': 'AAA', 'ts': '2026-06-01T00:00:00+00:00'}),
            json.dumps({'action': 'buy', 'symbol': 'AAA',
                       'ts': '2026-06-01T00:00:00+00:00',
                       'llm_reasoning': 'x' * 2000}),
            json.dumps({'action': 'sell', 'exit_reason': 'signal_sell',
                       'symbol': 'AAA', 'ts': '2026-06-01T00:00:00+00:00'}),
            json.dumps({'action': 'entry_window', 'asset_type': 'stock',
                       'admitted_k': 2}),
            json.dumps({'action': 'llm_analysis', 'symbol': 'AAA'}),
            json.dumps({'action': 'account_risk', 'foo': 'bar'}),
            'null', '123', 'not json',
        ]
        (journal_dir / f'{today}.jsonl').write_text('\n'.join(lines) + '\n')

        rows = load_journal(1)
        assert len(rows) == 4
        assert not any('llm_reasoning' in r for r in rows)
        buy = [r for r in rows if r.get('action') == 'buy'][0]
        assert 'symbol' in buy and 'ts' in buy


# ===========================================================================
# cost_floor spread coverage
# ===========================================================================

class TestCostFloorCoverage:
    def test_mixed_coverage(self, monkeypatch):
        bars = _bars([0.001] * 200)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows = [
            {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
             'ts': str(bars.index[5]), 'spread_pct': 0.25},
            {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
             'ts': str(bars.index[29])},
            {'action': 'skip', 'skip_reason': 'cost_floor', 'symbol': 'AAA/USD',
             'ts': str(bars.index[53])},
        ]
        out = gate_attribution(rows, api=object())
        assert out['_cost_floor_spread_coverage'] == pytest.approx(1 / 3, abs=1e-3)
        assert out['_cost_floor_flat_spread'] is True
