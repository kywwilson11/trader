"""scripts/trade_memory_phantom_audit.py — owner #22 phantom-exit audit (DRY-RUN ONLY).

Synthetic trade memories + broker caches through the pure core and ``--replay``;
no network (sockets blocked for the CLI test). The Kelly mirror is pinned to the
real trading_utils.compute_kelly_fraction where that module imports.
"""
import ast
import datetime as dt
import hashlib
import json
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import trade_memory_phantom_audit as pa  # noqa: E402

SCRIPT = os.path.join(ROOT, 'scripts', 'trade_memory_phantom_audit.py')
T0 = dt.datetime(2026, 4, 5, 16, 0, 0, tzinfo=dt.timezone.utc)
COV = '2026-01-18T00:00:00Z'


def _iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%S.%f') + '123Z'      # 9 fractional digits


def _rec(ts, reason='broker_stop', action='sell', exit_px=100.0, pnl=1.0, **kw):
    r = {'ts': ts.isoformat(timespec='seconds'), 'action': action, 'entry': 99.0,
         'exit': exit_px, 'pnl_pct': pnl, 'llm_score': None, 'reasoning': '',
         'news': '', 'exit_reason': reason}
    r.update(kw)
    return r


def _fill(t, sym='BTC/USD', side='sell', qty=1.0, px=100.0, oid=None):
    return {'activity_type': 'FILL', 'transaction_time': _iso(t), 'symbol': sym,
            'side': side, 'qty': str(qty), 'price': str(px),
            'order_id': oid or f'o-{sym}-{side}-{t.timestamp()}'}


def _cache(fills, orders=(), cov=COV):
    return {'fills': list(fills), 'orders': list(orders), 'coverage_start': cov}


def _one(data, cache, **kw):
    return pa.classify(data, pa.build_executions(cache), coverage_start=cache.get('coverage_start'),
                       **kw)


# ------------------------------------------------------------------ parsing / ids

def test_parse_ts_fraction_widths_and_offsets():
    a = pa.parse_ts('2026-04-05T16:16:09.171190123Z')
    b = pa.parse_ts('2026-04-05T16:16:09.17119+00:00')
    c = pa.parse_ts('2026-04-05T11:16:09.17119-05:00')
    assert a == b == c and a.tzinfo is not None
    assert pa.parse_ts('2026-04-05T16:16:09+00:00').microsecond == 0


def test_record_ids_disambiguate_duplicate_timestamps():
    data = {'BTC/USD': [_rec(T0), _rec(T0), _rec(T0 + dt.timedelta(seconds=1))]}
    ids = [rid for rid, _, _, _ in pa.iter_records(data)]
    ts = T0.isoformat(timespec='seconds')
    assert ids == [f'BTC/USD|{ts}|0', f'BTC/USD|{ts}|1',
                   f'BTC/USD|{(T0 + dt.timedelta(seconds=1)).isoformat(timespec="seconds")}|0']


# ------------------------------------------------------------------ classification boundaries

@pytest.mark.parametrize('offset_min,verdict,why', [
    (0.0, 'MATCHED', ''),
    (15.0, 'MATCHED', ''),                               # boundary inclusive
    (-15.0, 'MATCHED', ''),
    (15.5, 'AMBIGUOUS', 'dt_outside_match_window'),
    (1440.0, 'AMBIGUOUS', 'dt_outside_match_window'),    # ambig boundary inclusive
    (-1440.0, 'AMBIGUOUS', 'dt_outside_match_window'),
    (1441.0, 'PHANTOM', 'no_fill_within_ambig_window'),
    (-7707.0, 'PHANTOM', 'no_fill_within_ambig_window'),  # the April 2026 geometry
])
def test_window_boundaries(offset_min, verdict, why):
    data = {'BTC/USD': [_rec(T0)]}
    cache = _cache([_fill(T0 + dt.timedelta(minutes=offset_min))])
    r = _one(data, cache)[0]
    assert (r['verdict'], r['why']) == (verdict, why)
    assert r['nearest_dt_min'] == pytest.approx(offset_min, abs=1e-3)


def test_no_fill_for_symbol_is_phantom_and_before_coverage_is_ambiguous():
    data = {'ETH/USD': [_rec(T0)]}
    r = _one(data, _cache([_fill(T0, sym='BTC/USD')]))[0]
    assert r['verdict'] == 'PHANTOM' and r['nearest_dt_min'] is None
    r = _one(data, _cache([], cov=_iso(T0 - dt.timedelta(hours=1))))[0]
    assert (r['verdict'], r['why']) == ('AMBIGUOUS', 'before_history_coverage')


def test_side_mapping_sell_vs_cover_and_symbol_normalisation():
    data = {'BTC/USD': [_rec(T0)], 'META': [_rec(T0, reason='short_cover', action='cover')]}
    cache = _cache([_fill(T0, sym='BTCUSD', side='buy'),              # wrong side for a sell
                    _fill(T0, sym='META', side='buy', oid='m1')])
    rows = {r['symbol']: r for r in _one(data, cache)}
    assert rows['BTC/USD']['verdict'] == 'PHANTOM'
    assert rows['META']['verdict'] == 'MATCHED' and rows['META']['match_order_id'] == 'm1'
    cache = _cache([_fill(T0 + dt.timedelta(minutes=1), sym='BTCUSD')])  # slashless symbol
    assert _one({'BTC/USD': [_rec(T0)]}, cache)[0]['verdict'] == 'MATCHED'


def test_one_execution_backs_at_most_one_record_closest_wins():
    data = {'BTC/USD': [_rec(T0), _rec(T0 + dt.timedelta(minutes=10))]}
    cache = _cache([_fill(T0 + dt.timedelta(minutes=9), oid='only')])
    a, b = _one(data, cache)
    assert (b['verdict'], b['match_order_id']) == ('MATCHED', 'only')
    assert (a['verdict'], a['why']) == ('AMBIGUOUS', 'fill_claimed_by_closer_record')


def test_partial_fills_aggregate_per_order_and_use_closest_fill_time():
    data = {'BTC/USD': [_rec(T0, exit_px=101.0)]}
    cache = _cache([_fill(T0 - dt.timedelta(minutes=40), qty=1, px=100, oid='x'),
                    _fill(T0 + dt.timedelta(minutes=2), qty=3, px=101.333333, oid='x')])
    r = _one(data, cache)[0]
    assert r['verdict'] == 'MATCHED' and r['match_dt_min'] == pytest.approx(2.0, abs=1e-3)
    assert r['match_qty'] == 4 and r['match_vwap'] == pytest.approx(101.0, rel=1e-6)


def test_price_and_qty_tolerance_downgrade_to_ambiguous_only():
    data = {'BTC/USD': [_rec(T0, exit_px=100.0)]}
    r = _one(data, _cache([_fill(T0, px=106.0)]))[0]
    assert (r['verdict'], r['why']) == ('AMBIGUOUS', 'price_outside_tol')
    assert _one(data, _cache([_fill(T0, px=104.9)]))[0]['verdict'] == 'MATCHED'
    data = {'BTC/USD': [_rec(T0, qty=2.0)]}
    r = _one(data, _cache([_fill(T0, qty=1.0)]))[0]
    assert (r['verdict'], r['why']) == ('AMBIGUOUS', 'qty_outside_tol')
    assert _one(data, _cache([_fill(T0, qty=1.99)]))[0]['verdict'] == 'MATCHED'


def test_filled_order_without_fill_activity_is_a_fallback_execution():
    order = {'id': 'ord1', 'symbol': 'BTC/USD', 'side': 'sell', 'filled_qty': '0.5',
             'filled_avg_price': '100', 'filled_at': _iso(T0 + dt.timedelta(minutes=3))}
    r = _one({'BTC/USD': [_rec(T0)]}, _cache([], orders=[order]))[0]
    assert r['verdict'] == 'MATCHED' and r['match_source'] == 'order'


def test_raw_w19_dump_with_activities_is_accepted():
    raw = {'activities': [_fill(T0), {'activity_type': 'CFEE', 'qty': '-1', 'symbol': 'BTCUSD'}]}
    assert len(pa.fills_from_cache(raw)) == 1


# ------------------------------------------------------------------ report / repair

def _mixed_memory():
    data = {'BTC/USD': [], 'ETH/USD': [], 'AAPL': []}
    for i in range(3):          # matched broker_stops
        data['BTC/USD'].append(_rec(T0 + dt.timedelta(days=i), pnl=-1.0))
    for i in range(4):          # phantoms (no fills)
        data['ETH/USD'].append(_rec(T0 + dt.timedelta(days=i), pnl=5.0))
    data['ETH/USD'].append(_rec(T0 + dt.timedelta(days=9), pnl=3.0, estimated=True))  # already flagged
    data['ETH/USD'].append(_rec(T0 + dt.timedelta(days=10), reason='desync', pnl=2.0))
    data['AAPL'].append(_rec(T0, reason='trailing'))
    fills = [_fill(T0 + dt.timedelta(days=i, minutes=1)) for i in range(3)]
    fills.append(_fill(T0 + dt.timedelta(minutes=2), sym='AAPL'))
    return data, _cache(fills)


def test_report_repair_lists_only_unflagged_in_scope_phantoms_and_never_mutates():
    data, cache = _mixed_memory()
    before = json.dumps(data, sort_keys=True)
    rep = pa.build_report(data, cache, source_sha256='abc')
    assert json.dumps(data, sort_keys=True) == before
    s = rep['in_scope']
    assert (s['n'], s['MATCHED'], s['PHANTOM'], s['AMBIGUOUS']) == (8, 3, 5, 0)
    assert s['phantom_estimated_set'] == 1
    rp = rep['repair']
    assert rp['n'] == 4 and rp['source_sha256'] == 'abc'
    assert all(i.startswith('ETH/USD|') for i in rp['ids'])
    assert all(r['patch'] == {'estimated': True} for r in rp['records'])
    assert 'NOT APPLIED' in rp['action']
    assert rep['by_reason']['desync']['PHANTOM'] == 1          # classified, not repaired
    assert rep['legacy_unflagged_estimated_by_design'] == {'desync': 1}
    assert rep['by_reason']['trailing']['MATCHED'] == 1
    assert rep['n_records_estimated_true'] == 1
    assert set(rep['sensitivity_match_min']) == {'5.0', '15.0', '60.0', '240.0'}


def test_kelly_after_repair_drops_phantoms_from_the_sample():
    data = {'BTC/USD': [_rec(T0 + dt.timedelta(hours=i), pnl=(4.0 if i % 3 else -2.0))
                        for i in range(60)]}
    rep = pa.build_report(data, _cache([]))
    k = rep['kelly']['crypto']
    assert k['n_admissible_now'] == 60 and k['n_phantom_in_sample_now'] == 60
    assert k['kelly_now'] is not None and k['kelly_after_repair'] is None
    assert k['kelly_mult_after_repair'] == 1.0
    assert rep['kelly']['stock']['n_admissible_now'] == 0


def test_kelly_mult_mirrors_base_loop_mapping():
    assert pa.kelly_mult(None) == 1.0
    assert pa.kelly_mult(0.125) == pytest.approx(1.0)
    assert pa.kelly_mult(0.05) == pytest.approx(0.5)
    assert pa.kelly_mult(0.25) == pytest.approx(1.5)
    assert pa.kelly_mult(0.1995) == 1.5


def test_kelly_cap_literal_matches_strategy_config():
    sc = pytest.importorskip('strategy_config')
    assert pa.KELLY_CAP == sc.KELLY_CAP


def _kelly_cases():
    base = T0
    win_loss = [(f'X{i % 3}/USD' if i % 2 else f'S{i % 4}', 3.0 if i % 3 else -1.5)
                for i in range(260)]
    d1 = {}
    for i, (sym, pnl) in enumerate(win_loss):
        d1.setdefault(sym, []).append(_rec(base + dt.timedelta(minutes=i), pnl=pnl,
                                           estimated=(i % 7 == 0)))
    d2 = {'A/USD': [_rec(base, pnl=1.0)] * 10}                       # < min_trades
    d3 = {'A/USD': [_rec(base + dt.timedelta(minutes=i), pnl=1.0) for i in range(60)]}  # no losses
    d4 = {'A/USD': [_rec(base + dt.timedelta(minutes=i), pnl=(2.0 if i % 2 else -3.0))
                    for i in range(80)],
          'B': [_rec(base + dt.timedelta(minutes=i), pnl=(0.5 if i % 4 else -0.2), estimated=True)
                for i in range(80)]}
    return [d1, d2, d3, d4]


@pytest.mark.parametrize('case', range(4))
@pytest.mark.parametrize('book', ['crypto', 'stock', None])
def test_kelly_mirror_equals_trading_utils(tmp_path, monkeypatch, case, book):
    pytest.importorskip('dotenv')
    tu = pytest.importorskip('trading_utils')
    data = _kelly_cases()[case]
    p = tmp_path / 'tm.json'
    p.write_text(json.dumps(data))
    monkeypatch.setattr(tu, '_TRADE_MEMORY_FILE', p)
    want = tu.compute_kelly_fraction(asset_type=book)
    got, _ = pa.kelly_mirror(data, book)
    assert (got is None) == (want is None)
    if want is not None:
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


# ------------------------------------------------------------------ CLI / safety

def test_cli_replay_offline_writes_only_named_outputs(tmp_path, monkeypatch, capsys):
    def _blocked(*a, **k):
        raise AssertionError('network used')
    monkeypatch.setattr(socket, 'create_connection', _blocked)
    monkeypatch.setattr(socket.socket, 'connect', _blocked)
    data, cache = _mixed_memory()
    tm = tmp_path / 'trade_memory.json'
    tm.write_text(json.dumps(data))
    raw = tm.read_bytes()
    st = os.stat(tm)
    cp = tmp_path / 'cache.json'
    cp.write_text(json.dumps(cache))
    out, js = tmp_path / 'repair.json', tmp_path / 'report.json'
    assert pa.main(['--replay', str(cp), '--trade-memory', str(tm), '--out', str(out),
                    '--json', str(js)]) == 0
    printed = capsys.readouterr().out
    assert 'PHANTOM=5' in printed and 'NOT APPLIED' in printed
    assert tm.read_bytes() == raw and os.stat(tm).st_mtime_ns == st.st_mtime_ns
    rep = json.loads(out.read_text())
    assert rep['n'] == 4 and rep['source_sha256'] == hashlib.sha256(raw).hexdigest()
    assert json.loads(js.read_text())['in_scope']['PHANTOM'] == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'cache.json', 'repair.json', 'report.json', 'trade_memory.json']


def test_cli_requires_a_source():
    with pytest.raises(SystemExit):
        pa.main([])


def test_write_sink_refuses_live_state_files(tmp_path):
    for name in ('trade_memory.json', 'position_state.json'):
        with pytest.raises(SystemExit):
            pa._write_file(str(tmp_path / name), '{}')
        assert not (tmp_path / name).exists()


FORBIDDEN_CALLS = {'submit_order', 'cancel_order', 'cancel_all_orders', 'replace_order',
                   'close_position', 'close_all_positions', 'post', 'put', 'patch',
                   'delete', 'remove', 'unlink', 'rename', 'rmtree', 'truncate',
                   'record_trade', '_save', 'dump_trade_memory'}


def assert_read_only_source(path):
    """AST pin: no broker write verb, only GET requests, and every write-mode
    open() lives in the single guarded sink ``_write_file``."""
    tree = ast.parse(open(path).read())
    src = open(path).read()
    for bad in ("'POST'", '"POST"', "'DELETE'", '"DELETE"', "'PATCH'", '"PATCH"',
                "'PUT'", '"PUT"', 'requests.', 'alpaca_trade_api', 'from alpaca'):
        assert bad not in src, bad
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node

    def enclosing_fn(n):
        while n in parents:
            n = parents[n]
            if isinstance(n, ast.FunctionDef):
                return n.name
        return None

    n_open_w = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        assert name not in FORBIDDEN_CALLS, f'forbidden call {name} at line {node.lineno}'
        # str.replace is fine; os.replace (an atomic file swap) is not
        assert not (name == 'replace' and getattr(f.value, 'id', None) in ('os', 'shutil')), \
            f'os.replace at line {node.lineno}'
        for kw in node.keywords:
            if kw.arg == 'method':
                assert isinstance(kw.value, ast.Constant) and kw.value.value == 'GET'
        if name == 'Request':
            assert any(kw.arg == 'method' for kw in node.keywords), 'Request without method=GET'
        if name == 'open':
            mode = node.args[1] if len(node.args) > 1 else next(
                (kw.value for kw in node.keywords if kw.arg == 'mode'), None)
            if mode is None:
                continue
            if isinstance(mode, ast.Constant) and set(str(mode.value)) <= set('rbt'):
                continue
            assert enclosing_fn(node) == '_write_file', \
                f'write-mode open() outside _write_file at line {node.lineno}'
            n_open_w += 1
    return n_open_w


@pytest.mark.parametrize('script', ['trade_memory_phantom_audit.py', 'paper_quirk_census.py'])
def test_source_is_read_only(script):
    """Neither CLI can submit/cancel/replace an order or write the live state
    files: only GET requests, one guarded write sink, no atomic-swap calls."""
    path = os.path.join(ROOT, 'scripts', script)
    assert assert_read_only_source(path) == 1
    src = open(path).read()
    assert "PROTECTED_NAMES = ('trade_memory.json', 'position_state.json')" in src
    assert "if 'paper' not in base:" in src
