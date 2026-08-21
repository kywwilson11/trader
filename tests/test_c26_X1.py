"""c26 final bug-hunt wave (X1) — cross-packet fixes.

Covers (Mac-runnable, stubs/fakes only):
  F1  run_pipeline._build_training_phases omits --trials under
      PROMOTION_GATE_V2 (cumulative pool resolution); OFF literals pinned.
  F2  backtest.aggregate_metrics gate_v2 degraded path: calendar estimator
      unavailable -> IID null WITH the closed floor, never legacy clustered.
  F6  backtest.main gate-failure: fallback_champion with .prev present holds
      the established champion instead of rolling back to a stale .prev.
  F3  market_data.refresh_daily_bars: network fetch outside the cache lock,
      in-flight dedup, failure keeps the previous entry.
  F12 meta_label._write_refusal unlinks its pid tmp on failure.
  F9  base_loop._rebuild_prediction_pool rate limit (one per 10 min).
  F10 stock_loop._recover_external_exit closed-order fallback time filter.
  F11 order_utils.place_marketable_ioc: malformed quote -> None, no raise.
  F13 shadow._append_promotion_ledger nulls placeholder stats.
  C-3 panel_ranks daily-restore for the 4 constant CS ranks (flag-gated).
"""
import ast
import concurrent.futures
import datetime as dt
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import backtest
import market_data as md
import meta_label
import order_utils as ou
import panel_ranks
import run_pipeline as rp
import sample_weights
import shadow as sh
import strategy_config
import validation as V

BASE_SRC = (REPO / "base_loop.py").read_text()
STOCK_SRC = (REPO / "stock_loop.py").read_text()


# ---------------------------------------------------------------------------
# extraction helpers (pattern: tests/test_c26_P1.py)
# ---------------------------------------------------------------------------

def _extract_method(src, class_name, method_name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, ast.FunctionDef)
                        and item.name == method_name):
                    return textwrap.dedent(ast.get_source_segment(src, item))
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_method(src, class_name, method_name, glb):
    seg = _extract_method(src, class_name, method_name)
    ns = dict(glb)
    exec(compile(seg, f"<{method_name}>", "exec"), ns)
    return ns[method_name]


# ---------------------------------------------------------------------------
# F1 — run_pipeline gate cmds under PROMOTION_GATE_V2
# ---------------------------------------------------------------------------

def _gate_phases(phases):
    return {ph['id']: ph for ph in phases
            if ph['id'].endswith('_backtest_gate')}


class TestGateCmdTrialsOmission:
    def test_flag_on_omits_trials(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'PROMOTION_GATE_V2', True)
        monkeypatch.setenv('TRADER_SHADOW_MODE', '1')
        monkeypatch.setattr(rp, '_print', lambda *a, **k: None)
        phases = rp._build_training_phases(50, True, True, shadow=True)
        gates = _gate_phases(phases)
        assert gates['crypto_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--days', '44', '--gate']
        assert gates['stock_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--prefix', 'stock',
            '--days', '60', '--gate']
        for g in gates.values():
            assert '--trials' not in g['cmd']

    def test_flag_off_keeps_exact_literals(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'PROMOTION_GATE_V2', False)
        monkeypatch.setenv('TRADER_SHADOW_MODE', '1')
        monkeypatch.setattr(rp, '_print', lambda *a, **k: None)
        phases = rp._build_training_phases(50, True, True, shadow=True)
        gates = _gate_phases(phases)
        # Redundant with test_c26_Q2 but pins the conditional itself.
        assert gates['crypto_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--days', '44',
            '--trials', '50', '--gate']
        assert gates['stock_backtest_gate']['cmd'] == [
            rp.PYTHON, '-u', 'backtest.py', '--prefix', 'stock',
            '--days', '60', '--trials', '50', '--gate']


# ---------------------------------------------------------------------------
# F2 — gate_v2 degraded path (calendar estimator unavailable)
# ---------------------------------------------------------------------------

def _chain_trades(n=100, seed=11, name='N0'):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2026-06-01', tz='UTC')
    trades = []
    for k in range(n):
        entry = base + pd.Timedelta(hours=4 * k)
        trades.append({
            'ticker': name, 'entry_time': str(entry),
            'exit_time': str(entry + pd.Timedelta(hours=5)),
            'entry': 100.0, 'exit': 101.0, 'bars_held': 5,
            'gross_pct': float(rng.normal(0.05, 1.0)),
            'net_pct': float(rng.normal(0.05, 1.0)),
            'reason': 'take_profit',
        })
    return trades


def _raise_calendar(*a, **k):
    raise RuntimeError('calendar estimator down')


class TestGateV2CalendarUnavailable:
    def test_v2_falls_back_to_iid_with_closed_floor(self, monkeypatch):
        monkeypatch.setattr(backtest._strategy_config,
                            'PROMOTION_GATE_V2', True)
        monkeypatch.setattr(sample_weights, 'calendar_effective_n',
                            _raise_calendar)
        trades = _chain_trades()
        m = backtest.aggregate_metrics(trades, 'crypto', 90.0,
                                       n_search_trials=10)
        assert m['gate_v2_active'] is True
        assert m['n_eff_source'] == 'calendar_unavailable_iid'
        assert m['n_eff_calendar'] is None
        # IID null: the DSR consumed n_trades, not the clustered/floored 10
        assert m['dsr_n_eff_used'] == float(len(trades))
        assert m['dsr'] == m['dsr_iid']
        rets = np.array([t['net_pct'] for t in trades])
        ref = V.dsr_from_trade_returns(rets, n_trials=10, n_eff=None,
                                       fail_closed_floor=True)
        assert m['dsr_raw'] == pytest.approx(ref['dsr'])

    def test_flag_off_same_input_keeps_legacy_clustered(self, monkeypatch):
        monkeypatch.setattr(backtest._strategy_config,
                            'PROMOTION_GATE_V2', False)
        monkeypatch.setattr(sample_weights, 'calendar_effective_n',
                            _raise_calendar)
        trades = _chain_trades()
        m = backtest.aggregate_metrics(trades, 'crypto', 90.0,
                                       n_search_trials=10)
        assert m['gate_v2_active'] is False
        assert m['n_eff_source'] == 'clustered'
        assert m['n_eff_calendar'] is None
        rets = np.array([t['net_pct'] for t in trades])
        ref = V.dsr_from_trade_returns(rets, n_trials=10, n_eff=1.0,
                                       n_eff_source='clustered')
        assert m['dsr_raw'] == pytest.approx(ref['dsr'])
        assert m['dsr_n_eff_used'] == float(ref['n_eff'])   # floored legacy


# ---------------------------------------------------------------------------
# F6 — fallback_champion + .prev present => hold, not rollback
# ---------------------------------------------------------------------------

CORE = backtest.ARTIFACT_SUFFIXES[:4]
FAIL_METRICS = {'n_trades': 0, 'sharpe': -1.0, 'dsr': 0.0}


def _touch_core(tmp_path, slot):
    p = f'{slot}_' if slot else ''
    for s in CORE:
        (tmp_path / f'{p}{s}').write_text('x')


def _touch_prev(tmp_path, slot):
    p = f'{slot}_' if slot else ''
    for s in CORE:
        (tmp_path / f'{p}{s}.prev').write_text('x')


def _wire_main(monkeypatch, tmp_path, metrics, restore_calls):
    monkeypatch.setattr(backtest, 'BASE_DIR', tmp_path)

    def rb(prefix, days, trials, **kw):
        return dict(metrics)
    monkeypatch.setattr(backtest, 'run_backtest', rb)

    def spy(prefix):
        restore_calls.append(prefix)
        return True
    monkeypatch.setattr(backtest, 'restore_previous_model', spy)


class TestFallbackChampionHold:
    def test_prev_present_holds_champion(self, monkeypatch, tmp_path,
                                         capsys):
        calls = []
        notes = []
        _touch_core(tmp_path, '')        # champion core, challenger absent
        _touch_prev(tmp_path, '')        # established champion (.prev exists)
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        import notify as notify_mod
        monkeypatch.setattr(notify_mod, 'notify',
                            lambda msg, **kw: notes.append((msg, kw)))
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 3
        assert calls == []               # restore NOT called
        out = capsys.readouterr().out
        assert 'CHAMPION HELD' in out
        assert 'falling back' in out
        # A deliberate hold pages at warning, never critical, and says HELD
        (msg, kw), = notes
        assert kw['level'] == 'warning'
        assert 'HELD' in msg and 'no rollback available' not in msg

    def test_no_prev_still_rolls_back(self, monkeypatch, tmp_path):
        # mirror of the existing Q2 pin: genuine first deploy (no .prev)
        calls = []
        _touch_core(tmp_path, '')
        _wire_main(monkeypatch, tmp_path, FAIL_METRICS, calls)
        monkeypatch.setattr(sys, 'argv',
                            ['backtest.py', '--gate',
                             '--model-prefix', 'challenger'])
        assert backtest.main() == 3
        assert calls == ['']             # legacy restore path unchanged


# ---------------------------------------------------------------------------
# F3 — refresh_daily_bars lock discipline
# ---------------------------------------------------------------------------

def _wire_daily(monkeypatch, tmp_path, symbols=None):
    monkeypatch.setattr(md, '_DAILY_CACHE_FILE',
                        str(tmp_path / 'daily_cache.json'))
    monkeypatch.setattr(md, '_daily_cache',
                        {'loaded': True, 'symbols': dict(symbols or {})})
    monkeypatch.setattr(md, '_daily_inflight', set())


def _bar(day):
    return SimpleNamespace(t=day, o=1.0, h=2.0, l=0.5, c=1.5, v=100.0)


class TestRefreshDailyBarsLock:
    def test_fetch_runs_outside_lock_and_updates_cache(self, monkeypatch,
                                                       tmp_path):
        _wire_daily(monkeypatch, tmp_path)
        yesterday = (dt.datetime.now(dt.timezone.utc)
                     - dt.timedelta(days=1)).date()

        def get_bars(sym, tf, start=None, adjustment=None):
            acquired = md._daily_cache_lock.acquire(blocking=False)
            assert acquired, 'cache lock held across the network fetch'
            md._daily_cache_lock.release()
            return [_bar(yesterday)]

        api = SimpleNamespace(get_bars=get_bars)
        assert md.refresh_daily_bars(api, 'AAPL') is True
        entry = md._daily_cache['symbols']['AAPL']
        assert yesterday.isoformat() in entry['bars']
        assert 'AAPL' not in md._daily_inflight

    def test_inflight_symbol_skipped(self, monkeypatch, tmp_path):
        _wire_daily(monkeypatch, tmp_path)
        md._daily_inflight.add('AAPL')

        def get_bars(*a, **k):
            pytest.fail('get_bars must not be called while in-flight')

        api = SimpleNamespace(get_bars=get_bars)
        assert md.refresh_daily_bars(api, 'AAPL') is False
        assert 'AAPL' in md._daily_inflight    # not discarded by the skip

    def test_fetch_failure_keeps_previous_entry(self, monkeypatch, tmp_path):
        stale = time.time() - 2 * 86400
        prev = {'fetched_at': stale, 'bars': {'2026-08-01': [1, 2, 0.5,
                                                             1.5, 100]}}
        _wire_daily(monkeypatch, tmp_path, {'AAPL': dict(prev)})

        def get_bars(*a, **k):
            raise OSError('feed down')

        api = SimpleNamespace(get_bars=get_bars)
        assert md.refresh_daily_bars(api, 'AAPL') is False
        assert md._daily_cache['symbols']['AAPL']['bars'] == prev['bars']
        assert md._daily_cache['symbols']['AAPL']['fetched_at'] == stale
        assert 'AAPL' not in md._daily_inflight


# ---------------------------------------------------------------------------
# F12 — refusal sidecar tmp cleanup
# ---------------------------------------------------------------------------

class TestWriteRefusalTmpCleanup:
    def test_replace_failure_leaves_no_tmp_residue(self, monkeypatch,
                                                   tmp_path, capsys):
        def boom(src, dst):
            raise OSError('disk gone')
        monkeypatch.setattr(meta_label.os, 'replace', boom)
        meta_label._write_refusal(
            {'refused': tmp_path / 'meta_refused.json'}, ['r1'])
        assert list(tmp_path.glob('*.tmp.*')) == []
        assert 'refusal sidecar write failed' in capsys.readouterr().out

    def test_failure_before_tmp_assignment_is_safe(self, tmp_path, capsys):
        # paths dict whose values explode on str() -> failure precedes tmp
        class Bad:
            def __str__(self):
                raise RuntimeError('no path')

            def get(self, k):
                return None

            def __getitem__(self, k):
                return Bad()
        meta_label._write_refusal(Bad(), ['r1'])
        assert 'refusal sidecar write failed' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# F9 — prediction-pool rebuild rate limit
# ---------------------------------------------------------------------------

class TestPoolRebuildRateLimit:
    def test_first_swaps_second_rate_limited(self):
        fn = _load_method(
            BASE_SRC, 'BaseTradingLoop', '_rebuild_prediction_pool',
            {'ThreadPoolExecutor': concurrent.futures.ThreadPoolExecutor})
        old = SimpleNamespace(
            shutdown=lambda wait, cancel_futures: None)
        me = SimpleNamespace(_prediction_pool=old, MAX_PREDICTION_WORKERS=1,
                             get_asset_type=lambda: 'crypto')
        fn(me)                                     # first call: swaps
        first = me._prediction_pool
        assert first is not old
        assert isinstance(me._pool_rebuild_ts, float)
        try:
            fn(me)                                 # immediate second call
            assert me._prediction_pool is first    # NOT swapped
        finally:
            first.shutdown(wait=False)


# ---------------------------------------------------------------------------
# F10 — external-exit recovery time filter
# ---------------------------------------------------------------------------

def _recover_fn():
    return _load_method(STOCK_SRC, 'StockLoop', '_recover_external_exit',
                        {'datetime': dt})


def _sell(fill_dt, price='99.0'):
    return SimpleNamespace(status='filled', side='sell',
                           filled_avg_price=price, filled_at=fill_dt)


class TestRecoveryTimeFilter:
    def test_stale_sell_skipped_fresh_chosen(self):
        fn = _recover_fn()
        entry = dt.datetime.now() - dt.timedelta(hours=2)
        stale = _sell(entry - dt.timedelta(hours=5), price='90.0')
        fresh = _sell(entry + dt.timedelta(hours=1), price='99.0')
        recorded = []
        me = SimpleNamespace(
            api=SimpleNamespace(get_order=lambda oid: pytest.fail('no probes'),
                                list_orders=lambda **k: [stale, fresh]),
            _tp_order_ids={}, llm_scores={},
            last_trade_time={'AAPL': entry},
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)))
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0)
        assert fn(me, 'AAPL', info) is True
        (args, kwargs), = recorded
        assert args[2] is fresh                     # the postdating fill
        assert kwargs['exit_reason'] == 'external_close'

    def test_all_stale_falls_back_to_estimated(self):
        fn = _recover_fn()
        entry = dt.datetime.now() - dt.timedelta(hours=2)
        stale = _sell(entry - dt.timedelta(hours=5))
        me = SimpleNamespace(
            api=SimpleNamespace(get_order=lambda oid: pytest.fail('no probes'),
                                list_orders=lambda **k: [stale]),
            _tp_order_ids={}, llm_scores={},
            last_trade_time={'AAPL': entry},
            _record_confirmed_exit=lambda *a, **k: pytest.fail('recorded'))
        assert fn(me, 'AAPL', SimpleNamespace(stop_order_id=None)) is False

    def test_no_entry_proxy_rejects_30h_old_fill(self):
        fn = _recover_fn()
        old = _sell(dt.datetime.now() - dt.timedelta(hours=30))
        me = SimpleNamespace(
            api=SimpleNamespace(get_order=lambda oid: pytest.fail('no probes'),
                                list_orders=lambda **k: [old]),
            _tp_order_ids={}, llm_scores={},
            last_trade_time={},
            _record_confirmed_exit=lambda *a, **k: pytest.fail('recorded'))
        assert fn(me, 'AAPL', SimpleNamespace(stop_order_id=None)) is False

    def test_no_entry_proxy_accepts_recent_fill(self):
        fn = _recover_fn()
        recent = _sell(dt.datetime.now() - dt.timedelta(hours=3))
        recorded = []
        me = SimpleNamespace(
            api=SimpleNamespace(get_order=lambda oid: pytest.fail('no probes'),
                                list_orders=lambda **k: [recent]),
            _tp_order_ids={}, llm_scores={},
            last_trade_time={},
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)))
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0)
        assert fn(me, 'AAPL', info) is True
        assert len(recorded) == 1

    def test_tp_probe_path_untouched(self):
        # confirmed via order-id probe: no time filter applies
        fn = _recover_fn()
        tp = SimpleNamespace(status='filled', filled_avg_price='103.5',
                             side='sell')
        recorded = []
        me = SimpleNamespace(
            api=SimpleNamespace(get_order=lambda oid: tp),
            _tp_order_ids={'AAPL': 'tp1'}, llm_scores={},
            last_trade_time={},
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)))
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0)
        assert fn(me, 'AAPL', info) is True
        (args, kwargs), = recorded
        assert kwargs['exit_reason'] == 'take_profit'


# ---------------------------------------------------------------------------
# F11 — marketable IOC malformed-quote guard
# ---------------------------------------------------------------------------

class TestIocQuoteGuard:
    def test_malformed_quote_returns_none_no_submit(self):
        submitted = []
        api = SimpleNamespace(
            submit_order=lambda **k: submitted.append(k))
        out = ou.place_marketable_ioc(api, 'AAPL', 'buy', 5,
                                      {'spread_pct': 0.1}, 20.0)
        assert out is None
        assert submitted == []

    def test_well_formed_quote_still_submits(self):
        submitted = []

        def submit_order(**k):
            submitted.append(k)
            return SimpleNamespace(id='o1', **k)

        api = SimpleNamespace(submit_order=submit_order)
        quote = {'bid': 100.0, 'ask': 100.1, 'midpoint': 100.05}
        out = ou.place_marketable_ioc(api, 'AAPL', 'buy', 5, quote, 20.0)
        assert out is not None
        assert len(submitted) == 1
        assert submitted[0]['type'] == 'limit'
        assert submitted[0]['time_in_force'] == 'ioc'
        assert submitted[0]['limit_price'] >= 100.1   # marketable through ask


# ---------------------------------------------------------------------------
# F13 — promotion ledger placeholder-stat nulling
# ---------------------------------------------------------------------------

class TestLedgerPlaceholderNulling:
    def test_placeholder_stats_nulled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sh, 'BASE_DIR', tmp_path)
        report = {'n': 3, 'age_days': 12.0, 'p': 1.0, 'dm': 0.0,
                  'mean_d': 0.0, 'hit_champ': None, 'hit_chall': None,
                  'fb_max': None}
        sh._append_promotion_ledger('stock', 'stock', 'discarded', report)
        row = json.loads((tmp_path / 'stock_promotion_ledger.jsonl')
                         .read_text().splitlines()[0])
        assert row['p'] is None
        assert row['dm'] is None
        assert row['mean_d'] is None
        assert row['stats_computed'] is False
        assert row['n'] == 3                        # filtering key preserved

    def test_computed_stats_preserved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sh, 'BASE_DIR', tmp_path)
        report = {'n': 40, 'age_days': 12.0, 'p': 0.5, 'dm': -0.2,
                  'mean_d': -0.01, 'hit_champ': 0.5, 'hit_chall': 0.5,
                  'fb_max': 1.0}
        sh._append_promotion_ledger('stock', 'stock', 'discarded', report)
        row = json.loads((tmp_path / 'stock_promotion_ledger.jsonl')
                         .read_text().splitlines()[0])
        assert row['p'] == 0.5 and row['dm'] == -0.2
        assert row['mean_d'] == -0.01
        assert row['stats_computed'] is True


# ---------------------------------------------------------------------------
# C-3 — panel_ranks daily-restore (TRADER_DAILY_FEATURE_RESTORE)
# ---------------------------------------------------------------------------

DAILY4 = ['RM_252_21', 'ON_Mom_252', 'RR_5', 'MA_Dist_50d']


def _panel_df(values_by_symbol, ts='2026-06-10 15:00', cols=None):
    cols = cols or ['Return_4h']
    frames = []
    for sym, vals in values_by_symbol.items():
        row = {c: v for c, v in zip(
            cols, vals if isinstance(vals, (list, tuple)) else [vals])}
        row['Ticker'] = sym
        frames.append(pd.DataFrame(row, index=[pd.Timestamp(ts, tz='UTC')]))
    return pd.concat(frames)


class TestPanelDailyRestore:
    """Wave C-3: the 4 daily-window CS rank bases restored from the
    daily-bars cache under TRADER_DAILY_FEATURE_RESTORE (flag OFF pinned
    byte-identical)."""

    def _run_live(self, monkeypatch, n=12, top_k=60):
        idx = pd.date_range('2026-06-08 14:30', periods=80, freq='h',
                            tz='UTC')
        panel_vals = {f'S{i}': {'dv': (i + 1) * 1e7,
                                'Return_4h': float(i)} for i in range(n)}

        def fake_bars(api, sym, **k):
            return pd.DataFrame({
                'Open': 100.0, 'High': 101.0, 'Low': 99.0, 'Close': 100.0,
                'Volume': panel_vals[sym]['dv'] / 100.0 / 7,
            }, index=idx)

        def fake_features(bars, spy_close=None, symbol=None):
            out = bars.copy()
            for c in panel_ranks.CS_RANK_BASE_COLS:
                if c in DAILY4:
                    out[c] = np.nan          # live-frame reality: all-NaN
                else:
                    out[c] = panel_vals[symbol].get(c, 0.0)
            return out

        import indicators
        monkeypatch.setattr(md, 'fetch_stock_bars_alpaca', fake_bars)
        monkeypatch.setattr(indicators, 'compute_stock_features',
                            fake_features)
        monkeypatch.setattr(panel_ranks, '_panel_symbols',
                            lambda: list(panel_vals))
        panel_ranks._live_cache = None
        return panel_ranks.compute_live_panel_ranks(api=object(),
                                                    top_k=top_k)

    def test_flag_off_byte_identity_no_cache_reads(self, monkeypatch):
        monkeypatch.delenv('TRADER_DAILY_FEATURE_RESTORE', raising=False)
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        control = self._run_live(monkeypatch)

        def raiser(sym):
            pytest.fail('load_daily_bars must not be called with flag OFF')

        monkeypatch.setattr(md, 'load_daily_bars', raiser)
        monkeypatch.setattr(md, 'daily_bars_fetched_at', raiser)
        out = self._run_live(monkeypatch)
        assert out == control
        for s, feats in out.items():
            for c in DAILY4:
                assert feats[f'CS_Rank_{c}'] == 0.0    # NaN base -> neutral

    def test_flag_on_restores_and_matches_harvest_math(self, monkeypatch):
        monkeypatch.setenv('TRADER_DAILY_FEATURE_RESTORE', '1')
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        daily = pd.DataFrame(
            {'Open': 1.0, 'High': 2.0, 'Low': 0.5, 'Close': 1.5,
             'Volume': 100.0},
            index=pd.DatetimeIndex([pd.Timestamp('2026-06-09', tz='UTC')]))
        monkeypatch.setattr(md, 'load_daily_bars', lambda sym: daily)
        monkeypatch.setattr(md, 'daily_bars_fetched_at',
                            lambda sym: time.time())

        def fake_restore(df, daily_bars, spy_daily, sym):
            i = int(sym[1:])
            for j, c in enumerate(DAILY4):
                df[c] = float(i * (j + 1))     # distinct per symbol+col
            return df, 4, 0

        import indicators
        monkeypatch.setattr(indicators, 'apply_daily_restore', fake_restore)
        out = self._run_live(monkeypatch)
        # The four CS ranks vary across names now
        for c in DAILY4:
            vals = {feats[f'CS_Rank_{c}'] for feats in out.values()}
            assert len(vals) > 1, c
        # Parity with the harvest rank math on the same injected values
        hdf = _panel_df({f'S{i}': float(i) for i in range(12)},
                        cols=['RM_252_21'])
        hout = panel_ranks.add_panel_ranks(hdf).set_index('Ticker')
        for i in range(12):
            s = f'S{i}'
            assert out[s]['CS_Rank_RM_252_21'] == pytest.approx(
                hout.loc[s, 'CS_Rank_RM_252_21']), s

    def test_flag_on_fail_open_cache_error(self, monkeypatch):
        monkeypatch.setenv('TRADER_DAILY_FEATURE_RESTORE', '1')
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)

        def boom(sym):
            raise OSError('cache broken')

        monkeypatch.setattr(md, 'load_daily_bars', boom)
        monkeypatch.setattr(md, 'daily_bars_fetched_at',
                            lambda sym: time.time())
        out = self._run_live(monkeypatch)
        assert len(out) == 12                       # symbols not dropped
        for s, feats in out.items():
            for c in DAILY4:
                assert feats[f'CS_Rank_{c}'] == 0.0

    def test_flag_on_fail_open_stale_cache(self, monkeypatch):
        monkeypatch.setenv('TRADER_DAILY_FEATURE_RESTORE', '1')
        monkeypatch.delenv('TRADER_CLOSED_BARS_V2', raising=False)
        def guarded_load(sym):
            if sym == 'SPY':
                return None       # flag-gated pre-load is allowed
            pytest.fail('stale cache must not be loaded')

        monkeypatch.setattr(md, 'load_daily_bars', guarded_load)
        monkeypatch.setattr(md, 'daily_bars_fetched_at',
                            lambda sym: time.time() - 5 * 86400)
        out = self._run_live(monkeypatch)
        assert len(out) == 12
        for s, feats in out.items():
            for c in DAILY4:
                assert feats[f'CS_Rank_{c}'] == 0.0
