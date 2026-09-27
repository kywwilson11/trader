"""G6 measurement-shelf fixes (2026-09-26 hunt, hunt/G6_measurement.md).

Pins:
  G6-1  llm_eval.realize_scored_rows: stock bar window is a calendar allowance
        (max t0 + max_h//7 + 5 days) clamped by market_data._clamp_sip_end;
        crypto window byte-identical.
  G6-2  decision_report / sizing_cofire_report read a naive (legacy) journal
        ts as the writer's LOCAL wall clock; offset-aware rows unchanged.
  C-1   llm_qualify: timeouts count against schema validity and enter the
        latency sample censored at the budget -> the gate can fail.
  C-1b  horizon_transfer_report / wave6_stage0 / funding_drift_audit load
        only the columns they read (parquet schema projection), output
        sha-identical to the full load (pyarrow-gated test).
  C-2   train_lexicon buckets stock bars by New York session date.
  B-2   rank_gradient_report: empty --preds dump -> "no rows", exit 2.
  G6A-1 execution_report: explicit "shortfall section skipped" line, no
        compare footer without fills.
  B-1   reliability_report labels scored on the gate's jointly-finite rows.
  C-3/4/5 sizing_cofire: "N buy rows, M with sizing" header, chronological
        flip ordering across DST, no tmp leak on a failed replace.

Mac-safe: numpy/pandas + the pure paths of these modules; the parquet
round-trip test importorskips pyarrow; zones come from zoneinfo, never the
process TZ.
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import decision_report  # noqa: E402
import execution_report  # noqa: E402
import llm_eval  # noqa: E402
import market_data  # noqa: E402

ET = ZoneInfo('America/New_York')
_REAL_CLAMP = market_data._clamp_sip_end
CHI = ZoneInfo('America/Chicago')


# --------------------------------------------------------------------------
# G6-1  llm_eval stock realization window
# --------------------------------------------------------------------------

def _ext_bars(a, b):
    """Alpaca-shaped stock hourly bar opens: 04:00-20:00 ET, weekdays."""
    t = a.astimezone(ET).replace(minute=0, second=0, microsecond=0)
    out = []
    while t <= b:
        if t.weekday() < 5 and 4 <= t.hour < 20:
            out.append(t)
        t += timedelta(hours=1)
    return out


class _FakeSip:
    """Rejects the WHOLE request when end is inside the 15-min SIP delay,
    exactly like the real feed (hunt/G6/r_sip_future_end.out)."""

    def __init__(self, now):
        self.now = now
        self.ends = []

    def get_bars(self, sym, tf, start, end, adjustment=None):
        e = datetime.fromisoformat(end)
        s = datetime.fromisoformat(start)
        self.ends.append(e)
        if e > self.now - timedelta(minutes=15):
            raise RuntimeError('subscription does not permit querying '
                               'recent SIP data')
        return [SimpleNamespace(t=t, c=100.0 + i)
                for i, t in enumerate(_ext_bars(s, e))]


def _stock_panel():
    """SPY scored every RTH hour 10:05..15:05 ET for 10 trading days, fb=24."""
    rows, d, days = [], datetime(2026, 8, 10, tzinfo=ET), 0
    while days < 10:
        if d.weekday() < 5:
            for h in range(10, 16):
                rows.append({'symbol': 'SPY', 'asset_type': 'stock',
                             't0': d.replace(hour=h, minute=5).timestamp(),
                             'horizon': 24, 's': 0.6, 'pred': 0.1})
            days += 1
        d += timedelta(days=1)
    return rows


def _truly_realizable(rows, now):
    bars = _ext_bars(
        datetime.fromtimestamp(rows[0]['t0'], timezone.utc) - timedelta(hours=2),
        _REAL_CLAMP(now, 'stock', now=now))
    ts = [b.timestamp() for b in bars]
    n = 0
    for r in rows:
        i0 = next((i for i, t in enumerate(ts) if t >= r['t0']), None)
        n += i0 is not None and i0 + r['horizon'] < len(ts)
    return n


def _freeze_clamp(monkeypatch, now):
    monkeypatch.setattr(market_data, '_clamp_sip_end',
                        lambda e, a, now=now: _REAL_CLAMP(e, a, now=now))


@pytest.mark.parametrize('lag,expected', [(timedelta(hours=2), 48),
                                          (timedelta(days=3), 54)])
def test_stock_panel_realizes_every_realizable_row(monkeypatch, lag, expected):
    rows = _stock_panel()
    now = datetime.fromtimestamp(rows[-1]['t0'], timezone.utc) + lag
    _freeze_clamp(monkeypatch, now)
    api = _FakeSip(now)
    out = llm_eval.realize_scored_rows(rows, api=api)
    got = sum(o[1] is not None for o in out)
    truth = _truly_realizable(rows, now)
    assert truth == expected
    assert got == truth                     # was 0/48 and 49/54 before G6-1
    assert all(e <= now - timedelta(minutes=16) for e in api.ends)


def test_stock_end_is_calendar_allowance_when_old(monkeypatch):
    seen = {}

    def fake_lookup(api, sym, asset, start, end):
        seen[asset] = (start, end)
        return np.array([]), np.array([])
    monkeypatch.setattr(llm_eval, '_bars_lookup', fake_lookup)
    t0 = datetime(2026, 5, 4, 14, 5, tzinfo=timezone.utc).timestamp()
    llm_eval.realize_scored_rows(
        [{'symbol': 'AAPL', 'asset_type': 'stock', 't0': t0, 'horizon': 24,
          's': 0.5, 'pred': 0.0},
         {'symbol': 'BTC/USD', 'asset_type': 'crypto', 't0': t0,
          'horizon': 24, 's': 0.5, 'pred': 0.0}], api=object())
    base = datetime.fromtimestamp(t0, timezone.utc)
    assert seen['stock'][1] == base + timedelta(days=24 // 7 + 5)
    # crypto request byte-identical to the pre-fix window
    assert seen['crypto'][1] == base + timedelta(hours=24 + 6)
    assert seen['crypto'][0] == seen['stock'][0] == base - timedelta(hours=2)


def test_stock_end_clamped_for_recent_rows(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_eval, '_bars_lookup',
                        lambda api, sym, asset, s, e: (seen.append(e) or
                                                       (np.array([]),
                                                        np.array([]))))
    now = datetime(2026, 9, 22, 18, 37, tzinfo=timezone.utc)  # Tue 14:37 ET
    _freeze_clamp(monkeypatch, now)
    t0 = (now - timedelta(hours=3)).timestamp()
    llm_eval.realize_scored_rows([{'symbol': 'AAPL', 'asset_type': 'stock',
                                   't0': t0, 'horizon': 12, 's': 0.5,
                                   'pred': 0.0}], api=object())
    assert seen == [_REAL_CLAMP(now + timedelta(days=30), 'stock', now=now)]
    assert seen[0] <= now - timedelta(minutes=16)


# --------------------------------------------------------------------------
# G6-2  legacy naive ts = LOCAL wall clock
# --------------------------------------------------------------------------

@pytest.fixture
def chicago(monkeypatch):
    import sizing_cofire_report as scr
    monkeypatch.setattr(decision_report, '_LOCAL_TZ', CHI)
    monkeypatch.setattr(scr, '_LOCAL_TZ', CHI)
    return scr


def test_decision_report_naive_ts_is_local(chicago):
    f = decision_report._naive_local_to_utc
    assert f(pd.Timestamp('2026-05-05T10:00:00')) == pd.Timestamp(
        '2026-05-05T15:00:00', tz='UTC')                       # CDT
    assert f(pd.Timestamp('2026-01-05T10:00:00')) == pd.Timestamp(
        '2026-01-05T16:00:00', tz='UTC')                       # CST
    # offset-aware rows are unchanged, whatever the local zone
    assert f(pd.Timestamp('2026-05-05T10:00:00-05:00')) == pd.Timestamp(
        '2026-05-05T15:00:00', tz='UTC')
    assert f(pd.Timestamp('2026-05-05T10:00:00+00:00')) == pd.Timestamp(
        '2026-05-05T10:00:00', tz='UTC')


def test_decision_report_replay_uses_local_reading(chicago, monkeypatch):
    idx = pd.date_range('2026-05-01', periods=24 * 10, freq='h', tz='UTC')
    bars = pd.DataFrame({'Open': 100.0, 'High': 101.0, 'Low': 99.0,
                         'Close': 100.0, 'Volume': 1.0}, index=idx)
    monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                        lambda api, sym, closed_only=True: bars)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, sym, limit=None, closed_only=True: bars)
    seen = []

    def fake_replay(b, ts, asset, **k):
        seen.append(ts)
        return 0.1
    monkeypatch.setattr(decision_report, 'replay_entry', fake_replay)
    rows = [{'symbol': 'AAPL', 'ts': '2026-05-05T10:00:00'},
            {'symbol': 'AAPL', 'ts': '2026-05-05T10:00:00-05:00'}]
    samples, n_fail, _, n_oow = decision_report._replay_grouped(
        rows, api=None)
    assert (n_fail, n_oow, len(samples)) == (0, 0, 2)
    want = pd.Timestamp('2026-05-05T15:00:00', tz='UTC')
    assert seen == [want, want]


def test_decision_report_dedup_orders_naive_as_local(chicago):
    rows = [{'symbol': 'X', 'skip_reason': 'r', 'ts': '2026-05-05T10:00:00',
             'tag': 'naive'},                                   # 15:00Z
            {'symbol': 'X', 'skip_reason': 'r',
             'ts': '2026-05-05T14:00:00+00:00', 'tag': 'aware'}]
    out = decision_report._dedup_first_per_day(rows, ['symbol', 'skip_reason'])
    assert [r['tag'] for r in out] == ['aware']   # truly first in time


def test_sizing_cofire_naive_ts_is_local(chicago):
    scr = chicago
    got = scr._parse_ts('2026-05-05T10:00:00')
    assert got.astimezone(timezone.utc) == datetime(2026, 5, 5, 15, 0,
                                                    tzinfo=timezone.utc)
    aware = scr._parse_ts('2026-05-05T10:00:00+00:00')
    assert aware == datetime(2026, 5, 5, 10, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# C-1  llm_qualify: timeouts are failures
# --------------------------------------------------------------------------

def _q_ok(latency=1.0):
    return {'ok': True, 'text': None, 'status': None, 'retry_after': None,
            'latency_s': latency, 'error': None}


def _q_timeout(budget):
    return {'ok': False, 'text': None, 'status': None, 'retry_after': None,
            'latency_s': budget, 'error': 'timed out'}


def _qualify(monkeypatch, pattern):
    """pattern: list of 'ok' / 'to' per attempt; ok answers are valid."""
    import llm_qualify as lq
    calls = []

    def stub(cand, prompt, system, schema, max_tokens, timeout):
        i = len(calls)
        calls.append(i)
        if pattern[i] == 'to':
            return _q_timeout(timeout)
        r = _q_ok()
        syms = list((schema.get('properties') or {}).keys())
        r['text'] = json.dumps({s: {'s': 0.6, 'bull': 'b', 'bear': 'b',
                                    'r': 'r'} for s in syms})
        return r
    monkeypatch.setattr(lq, '_transport_call', stub)
    monkeypatch.setattr(lq, '_ledger_spent', lambda: None)
    res = lq.run_qualification(
        [{'name': 'c', 'model': 'm', 'provider_kind': 'openai-compatible'}],
        {}, n_calls=len(pattern), spacing_s=0.0, sleep_fn=lambda s: None)
    return lq, res['c/m']


def test_qualify_half_timeouts_not_qualified(monkeypatch):
    lq, q = _qualify(monkeypatch, ['ok', 'to'] * 5)
    assert q['n_attempts'] == 10 and q['n_completed'] == 5
    assert q['n_timeouts'] == 5
    assert q['schema_valid_pct'] == pytest.approx(50.0)   # was 100.0
    assert q['p95_latency_s'] >= lq.BUDGET_S
    assert q['p95_censored'] is True
    assert q['verdict'] == 'failed'                       # was 'qualified'


def test_qualify_latency_ceiling_can_fail_on_its_own(monkeypatch):
    # 2/20 timeouts: schema 90% alone would be 'marginal'; the p95 lands on a
    # censored no-answer (> budget, true latency unknown) -> failed.
    lq, q = _qualify(monkeypatch, ['ok'] * 18 + ['to'] * 2)
    assert q['schema_valid_pct'] == pytest.approx(90.0)
    assert q['p95_censored'] is True
    assert q['verdict'] == 'failed'
    assert lq.verdict_for(dict(q, p95_censored=False)) == 'marginal'


def test_qualify_all_answered_unchanged(monkeypatch):
    lq, q = _qualify(monkeypatch, ['ok'] * 10)
    assert q['schema_valid_pct'] == pytest.approx(100.0)
    assert q['n_timeouts'] == 0 and q['p95_censored'] is False
    assert q['p95_latency_s'] == pytest.approx(1.0)
    assert q['verdict'] == 'qualified'


# --------------------------------------------------------------------------
# C-1b  column-projected store loads (sha-identical)
# --------------------------------------------------------------------------

def _synthetic_store():
    rng = np.random.default_rng(7)
    frames = []
    for k, tick in enumerate(['AAA/USD', 'BBB/USD', 'CCC/USD']):
        idx = pd.date_range('2025-10-01', periods=24 * 150, freq='h',
                            tz='UTC', name='Datetime')
        n = len(idx)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        f = pd.DataFrame({
            'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
            'Close': close, 'Volume': rng.random(n),
            'RSI': rng.random(n), 'MACD': rng.normal(size=n),
            'Target_Return_4': rng.normal(size=n),
            'Target_Return_12': rng.normal(size=n),
            'Target_Return': rng.normal(size=n),
            'TB_Bars_12': rng.integers(1, 13, n).astype(float),
            'TB_Bars_24': rng.integers(1, 25, n).astype(float),
            'TB_Reason_12': np.where(rng.random(n) < 0.5, 'tp', 'sl'),
            'Funding_Rate_Ann': rng.normal(size=n) + (k * 0.1),
            'Funding_Z': rng.normal(size=n),
            'Ticker': tick}, index=idx)
        frames.append(f)
    return pd.concat(frames)


@pytest.fixture
def store(tmp_path, monkeypatch):
    pytest.importorskip('pyarrow')
    import data_utils
    df = _synthetic_store()
    df.to_parquet(tmp_path / 'training_data.parquet')
    monkeypatch.setattr(data_utils, '_BASE_DIR', tmp_path)
    requested = []
    real = data_utils.load_training_data

    def spy(prefix, columns=None):
        requested.append(None if columns is None else list(columns))
        return real(prefix, columns=columns)
    monkeypatch.setattr(data_utils, 'load_training_data', spy)
    return SimpleNamespace(path=tmp_path, requested=requested, df=df)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=str).encode()).hexdigest()


def _no_data_lines(text):
    return [ln for ln in text.splitlines() if not ln.startswith('[DATA]')]


def test_horizon_transfer_projection_identical(store, monkeypatch):
    import horizon_transfer_report as htr

    def digest():
        per, hz = htr.load_per_name('crypto')
        h = hashlib.sha256(json.dumps(hz).encode())
        for name in sorted(per):
            fwd, t = per[name]
            h.update(name.encode())
            h.update(np.asarray(t).tobytes())
            for k in sorted(fwd):
                h.update(fwd[k].tobytes())
        return h.hexdigest()
    projected = digest()
    assert sorted(store.requested[-1]) == ['Target_Return_12',
                                           'Target_Return_4', 'Ticker']
    monkeypatch.setattr(htr, '_projected_columns', lambda *a, **k: None)
    full = digest()
    assert store.requested[-1] is None
    assert projected == full


def test_wave6_projection_identical(store, monkeypatch, capsys):
    import wave6_stage0 as w6
    a = w6.measure_book('crypto')
    out_a = _no_data_lines(capsys.readouterr().out)
    assert sorted(store.requested[-1]) == ['TB_Bars_12', 'TB_Bars_24',
                                           'Ticker']
    monkeypatch.setattr(w6, '_projected_columns', lambda *a, **k: None)
    b = w6.measure_book('crypto')
    out_b = _no_data_lines(capsys.readouterr().out)
    assert a is not None and sorted(a['horizons']) == [12, 24]
    assert _sha(a) == _sha(b) and out_a == out_b


def test_funding_drift_projection_identical(store, monkeypatch, capsys):
    import funding_drift_audit as fda

    def run(out):
        monkeypatch.setattr(sys, 'argv', ['funding_drift_audit.py',
                                          '--split', '2026-01-15',
                                          '--trailing-days', '30',
                                          '--out', str(out)])
        assert fda.main() == 0
        payload = json.loads(out.read_text())
        payload.pop('generated')
        return payload, _no_data_lines(capsys.readouterr().out)
    pa, oa = run(store.path / 'a.json')
    assert sorted(store.requested[-1]) == [
        'Funding_Rate_Ann', 'Funding_Z', 'Target_Return_12',
        'Target_Return_4', 'Ticker']
    monkeypatch.setattr(fda, '_projected_columns', lambda *a, **k: None)
    pb, ob = run(store.path / 'b.json')
    assert store.requested[-1] is None
    assert _sha(pa) == _sha(pb)
    assert [ln.replace('a.json', '') for ln in oa] == [
        ln.replace('b.json', '') for ln in ob]
    assert len(pa['rows']) == 2


def test_projection_falls_back_to_full_when_csv_fresher(store):
    import wave6_stage0 as w6
    csv = store.path / 'training_data.csv'
    csv.write_text('x\n')
    pq_mtime = (store.path / 'training_data.parquet').stat().st_mtime
    os.utime(csv, (pq_mtime + 30 * 86400, pq_mtime + 30 * 86400))
    assert w6._projected_columns('crypto', lambda c: True) is None


def test_projection_none_without_store(tmp_path, monkeypatch):
    import data_utils
    import horizon_transfer_report as htr
    monkeypatch.setattr(data_utils, '_BASE_DIR', tmp_path)
    assert htr._projected_columns('stock', lambda c: True) is None


# --------------------------------------------------------------------------
# C-2  train_lexicon: New York session dates
# --------------------------------------------------------------------------

def _lexicon_prices(monkeypatch, tmp_path, extra_args=()):
    import learned_lexicon as ll
    import train_lexicon as tl
    # Fri 2023-11-10 and Mon 2023-11-13 (EST), 04:00-19:00 ET opens.
    opens = [t for t in _ext_bars(datetime(2023, 11, 10, 3, tzinfo=ET),
                                  datetime(2023, 11, 13, 20, tzinfo=ET))]
    idx = pd.DatetimeIndex([t.astimezone(timezone.utc) for t in opens],
                           name='Datetime')
    bars = pd.DataFrame({'Ticker': 'AAPL',
                         'Open': np.arange(len(idx), dtype=float),
                         'Close': np.arange(len(idx), dtype=float) + 0.5},
                        index=idx)
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: bars.copy())
    monkeypatch.setattr(ll, 'load_articles', lambda *a, **k: pd.DataFrame(
        {'symbol': ['AAPL'], 'headline': ['h'], 'summary': ['s']}))
    seen = {}

    def fake_train(articles, prices, **k):
        seen['prices'] = prices
        return ({'meta': {'n_docs': 0, 'n_terms_screened': 0,
                          'lambda': None}}, {'horizons': {}, 'verdict': 'x'})
    monkeypatch.setattr(ll, 'train_lexicon', fake_train)
    monkeypatch.setattr(ll, 'write_json_atomic', lambda *a, **k: None)
    tl.main(['--no-novelty', '--out', str(tmp_path / 'l.json'),
             '--report', str(tmp_path / 'r.json'), *extra_args])
    return seen['prices'], bars


def test_train_lexicon_session_dates_are_new_york(monkeypatch, tmp_path):
    prices, bars = _lexicon_prices(monkeypatch, tmp_path)
    dates = list(prices['date'])
    assert dates == [dt.date(2023, 11, 10), dt.date(2023, 11, 13)]
    assert all(d.weekday() < 5 for d in dates)        # no phantom Saturday
    fri = prices[prices['date'] == dt.date(2023, 11, 10)].iloc[0]
    mon = prices[prices['date'] == dt.date(2023, 11, 13)].iloc[0]
    assert fri['close'] == bars['Close'].iloc[15]     # Fri 19:00 ET bar
    assert mon['open'] == bars['Open'].iloc[16]       # Mon 04:00 ET bar


def test_train_lexicon_session_tz_utc_reproduces_old_buckets(monkeypatch,
                                                             tmp_path):
    prices, _ = _lexicon_prices(monkeypatch, tmp_path,
                                ('--session-tz', 'UTC'))
    assert dt.date(2023, 11, 11) in set(prices['date'])   # the old Saturday


# --------------------------------------------------------------------------
# B-2  rank_gradient_report empty dump
# --------------------------------------------------------------------------

_RGR = str(REPO / 'scripts' / 'rank_gradient_report.py')


@pytest.mark.parametrize('name,body', [('d.json', '[]'), ('d.csv', ''),
                                       ('h.csv', 'ts,symbol,signal,fwd_return\n')])
def test_rank_gradient_empty_dump_exit_2(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    r = subprocess.run([sys.executable, _RGR, '--preds', str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert 'no rows' in r.stderr
    assert 'Traceback' not in r.stderr


def test_rank_gradient_help_documents_exit_codes():
    r = subprocess.run([sys.executable, _RGR, '--help'],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert 'exit status:' in r.stdout and 'empty --preds dump' in r.stdout


# --------------------------------------------------------------------------
# G6A-1  execution_report skipped-shortfall notice
# --------------------------------------------------------------------------

def _exec_journal(tmp_path, monkeypatch, rows):
    jd = tmp_path / 'journals'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', jd)
    monkeypatch.setattr(execution_report, 'BASE_DIR', tmp_path)


def test_execution_report_says_shortfall_skipped(tmp_path, monkeypatch,
                                                 capsys):
    _exec_journal(tmp_path, monkeypatch, [
        {'action': 'buy', 'symbol': s, 'final_notional': 100.0}
        for s in ('BTC/USD', 'ETH/USD', 'AAPL')])
    rep = execution_report.run_report(days=1)
    out = capsys.readouterr().out
    assert 'shortfall section skipped: 0/3 buys carry slippage_bps' in out
    assert 'IMPLEMENTATION SHORTFALL' not in out
    assert 'Compare against the backtest' not in out
    assert 'overall_mean_bps' not in rep


def test_execution_report_footer_kept_with_fills(tmp_path, monkeypatch,
                                                 capsys):
    _exec_journal(tmp_path, monkeypatch, [
        {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 3.0}])
    execution_report.run_report(days=1)
    out = capsys.readouterr().out
    assert 'IMPLEMENTATION SHORTFALL' in out
    assert 'Compare against the backtest' in out
    assert 'shortfall section skipped' not in out


# --------------------------------------------------------------------------
# B-1  reliability_report labels on the gate's row set
# --------------------------------------------------------------------------

def test_reliability_labels_use_jointly_finite_rows(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(float)
    p_true = np.where(y == 1, 0.7, 0.3)
    p_legacy = p_true.copy()
    p_purged = np.clip(p_true + rng.normal(0, 0.08, n), 0.01, 0.99)
    p_purged[:100] = np.nan                    # unscored OOF rows
    p_legacy[:100] = 1.0 - y[:100]             # legacy maximally wrong there
    path = tmp_path / 'calib_holdout.json'
    path.write_text(json.dumps({'p_legacy': p_legacy.tolist(),
                                'p_purged': p_purged.tolist(),
                                'y': y.tolist()}))
    r = subprocess.run([sys.executable,
                        str(REPO / 'scripts' / 'reliability_report.py'),
                        '--in', str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    brier_line = next(ln for ln in r.stdout.splitlines()
                      if ln.strip().startswith('Brier'))
    assert '(worse)' in brier_line             # was "(better)"
    assert 'keep legacy' in r.stdout or 'no calibration improvement' in r.stdout


# --------------------------------------------------------------------------
# C-3/C-4/C-5  sizing_cofire_report
# --------------------------------------------------------------------------

_SC = str(REPO / 'scripts' / 'sizing_cofire_report.py')


def _sc_dir(tmp_path, rows):
    jd = tmp_path / 'j'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return jd


def test_sizing_header_counts_buys_without_sizing(tmp_path):
    ts = datetime.now(timezone.utc).isoformat()
    jd = _sc_dir(tmp_path, [{'ts': ts, 'action': 'buy', 'symbol': 'BTC/USD'}
                            for _ in range(3)])
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert '3 buy rows, 0 with sizing' in p.stdout
    assert 'no rows' in p.stdout                     # test_c26_S3 pin kept
    assert 'pre-date the sizing producer' in p.stdout
    j = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd),
                        '--json'], capture_output=True, text=True)
    rep = json.loads(j.stdout)
    assert rep['n_buy_rows'] == 0                    # JSON meaning unchanged
    assert rep['n_buy_rows_without_sizing'] == 3


def test_sizing_header_mixed(tmp_path):
    ts = datetime.now(timezone.utc).isoformat()
    jd = _sc_dir(tmp_path, [
        {'ts': ts, 'action': 'buy', 'symbol': 'BTC/USD'},
        {'ts': ts, 'action': 'buy', 'symbol': 'BTC/USD',
         'sizing': {'vix_tilt': 0.7, 'tilt_raw': 0.7, 'tilt': 0.7}}])
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd)],
                       capture_output=True, text=True)
    assert '2 buy rows, 1 with sizing' in p.stdout
    assert 'per-multiplier' in p.stdout


def test_sizing_flip_count_is_chronological_across_dst():
    import sizing_cofire_report as scr

    def row(ts, vix):
        return {'ts': ts, 'symbol': 'BTC/USD', 'action': 'buy',
                'sizing': {'tilt': 1.0, 'tilt_raw': 1.0, 'stack': 'legacy',
                           'v2': {'tilt': 1.0, 'family': {'vix': vix}}}}
    rows = [row('2026-11-01T01:50:00-05:00', 'calm'),     # 06:50Z
            row('2026-11-01T01:10:00-06:00', 'stress'),   # 07:10Z
            row('2026-11-01T01:30:00-06:00', 'calm')]     # 07:30Z
    rep = scr.build_report(rows, [], 0, 1, 30, 'all')
    assert rep['v2_shadow']['flip_counts']['vix_tier_total'] == 2   # was 1


def test_sizing_json_failed_replace_leaves_no_tmp(tmp_path):
    ts = datetime.now(timezone.utc).isoformat()
    jd = _sc_dir(tmp_path, [{'ts': ts, 'action': 'buy', 'symbol': 'BTC/USD',
                             'sizing': {'tilt_raw': 0.7, 'tilt': 0.7}}])
    target = tmp_path / 'target_is_dir'
    target.mkdir()
    p = subprocess.run([sys.executable, _SC, '--journal-dir', str(jd),
                        '--json', str(target)], capture_output=True, text=True)
    assert p.returncode == 1
    assert not list(tmp_path.glob('target_is_dir.*.tmp'))
