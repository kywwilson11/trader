"""scripts/paper_quirk_census.py — X12 paper-quirk census (measurement-only, read-only).

Synthetic bundles through ``detect`` / ``--replay`` and saved rows through
``evaluate`` / ``--evaluate``; no network (sockets blocked for the CLI tests).
The read-only AST pin for this script lives in
tests/test_trade_memory_phantom_audit.py::test_source_is_read_only (both scripts).
"""
import json
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import paper_quirk_census as pc  # noqa: E402

OLD, CUR = 'old-asset-id', 'cur-asset-id'


def _pos(sym='BTCUSD', qty='1.0', avail='0', avg='50000', cb='50000', px='60000', aid=OLD):
    return {'symbol': sym, 'asset_id': aid, 'qty': qty, 'qty_available': avail,
            'avg_entry_price': avg, 'cost_basis': cb, 'current_price': px,
            'market_value': str(float(qty) * float(px))}


def _stop(sym='BTC/USD', qty='1.0', aid=CUR, oid='s1', status='new', side='sell', typ='stop_limit'):
    return {'id': oid, 'symbol': sym, 'asset_id': aid, 'type': typ, 'side': side,
            'qty': qty, 'filled_qty': '0', 'stop_price': '55000', 'limit_price': '54000',
            'status': status, 'submitted_at': '2026-09-27T08:09:31Z'}


def _bundle(positions=None, open_orders=None, recent=None, equity='60093.63', cash='93.63',
            now='2026-09-27T12:00:00+00:00', activities=None, state=None, assets=None):
    return {'now': now, 'account': {'equity': equity, 'cash': cash, 'last_equity': equity},
            'positions': [_pos()] if positions is None else positions,
            'open_orders': [_stop()] if open_orders is None else open_orders,
            'recent_orders': recent or [], 'assets': assets or {'BTCUSD': CUR},
            'activities': activities or [], 'position_state': state or []}


def _fill(sym, side, qty, t='2026-09-27T12:05:00Z'):
    return {'activity_type': 'FILL', 'symbol': sym, 'side': side, 'qty': str(qty),
            'price': '60000', 'transaction_time': t}


# ------------------------------------------------------------------ detectors

def test_basis_lost_on_string_zero_and_null_and_b_new_semantics():
    row = pc.detect(_bundle(positions=[_pos(avg='0', cb='0'), _pos('ETHUSD', avg=None, cb=None),
                                       _pos('SOLUSD')]))
    assert [x['sym'] for x in row['detectors']['B']] == ['BTCUSD', 'ETHUSD']
    assert row['detectors']['B'][0]['avg_entry_price'] == '0'        # raw string kept
    assert row['detectors']['B_new'] == []                           # no --last: standing
    nxt = pc.detect(_bundle(positions=[_pos(avg='0', cb='0'), _pos('ETHUSD', avg=None, cb=None),
                                       _pos('SOLUSD', avg='0', cb='0')]), prev_row=row)
    assert nxt['detectors']['B_new'] == ['SOLUSD']
    assert 'B' in nxt['fired']


def test_vanish_vs_last_snapshot_sell_explained_carried_and_reappeared():
    prev = pc.detect(_bundle(positions=[_pos(), _pos('ETHUSD', qty='2')]))
    row = pc.detect(_bundle(positions=[], open_orders=[]), prev_row=prev)
    assert {(x['sym'], x['source']) for x in row['detectors']['V']} == {
        ('BTCUSD', 'last_snapshot'), ('ETHUSD', 'last_snapshot')}
    sold = pc.detect(_bundle(positions=[_pos()], activities=[_fill('ETH/USD', 'sell', 2.0)]),
                     prev_row=prev)
    assert sold['detectors']['V'] == []                              # our own exit
    carried = pc.detect(_bundle(positions=[_pos()], now='2026-09-27T12:15:00+00:00'),
                        prev_row=row)
    assert [(x['sym'], x['source']) for x in carried['detectors']['V']] == [('ETHUSD', 'carried')]
    assert carried['detectors']['reappeared'] == ['BTCUSD']


def test_vanish_vs_position_state_tracking():
    st = [{'hwm': {'BTC/USD': 1.0, 'XRP/USD': 2.0}, 'trailing': {}}]
    row = pc.detect(_bundle(state=st))
    assert [(x['sym'], x['source']) for x in row['detectors']['V']] == [('XRPUSD', 'position_state')]
    assert row['tracked_symbols'] == ['BTCUSD', 'XRPUSD']


def test_qty_drift_net_of_fills_and_in_kind_fees_with_tolerance_and_usd():
    prev = pc.detect(_bundle(now='2026-09-27T12:00:00+00:00'))
    drift = pc.detect(_bundle(positions=[_pos(qty='0.9')], now='2026-09-27T12:15:00+00:00'),
                      prev_row=prev)
    q = drift['detectors']['Q'][0]
    assert q['sym'] == 'BTCUSD' and q['residual'] == pytest.approx(-0.1)
    assert q['usd'] == pytest.approx(6000.0)
    acts = [_fill('BTC/USD', 'sell', 0.099),
            {'activity_type': 'CFEE', 'symbol': 'BTCUSD', 'qty': '-0.001',
             'created_at': '2026-09-27T12:10:00Z'},
            _fill('BTC/USD', 'sell', 5.0, t='2026-09-27T11:00:00Z')]   # before prev: ignored
    ok = pc.detect(_bundle(positions=[_pos(qty='0.9')], activities=acts,
                           now='2026-09-27T12:15:00+00:00'), prev_row=prev)
    assert ok['detectors']['Q'] == []
    tiny = pc.detect(_bundle(positions=[_pos(qty='1.0000009')]), prev_row=prev)
    assert tiny['detectors']['Q'] == []                              # < 1e-6 * qty
    over = pc.detect(_bundle(positions=[_pos(qty='1.0000011')]), prev_row=prev)
    assert len(over['detectors']['Q']) == 1
    assert pc.detect(_bundle(positions=[_pos(qty='0.5')]))['detectors']['Q'] == []  # no --last


def test_equity_approximately_cash_while_positions_exist():
    assert pc.detect(_bundle(equity='93.63', cash='93.63'))['detectors']['E']['fired']
    ok = pc.detect(_bundle())['detectors']['E']
    assert not ok['fired'] and ok['gap_frac'] == pytest.approx(1.0)
    empty = pc.detect(_bundle(positions=[], open_orders=[], equity='93.63', cash='93.63'))
    assert not empty['detectors']['E']['fired']


def test_asset_id_split_table_against_resting_stop_and_current_asset():
    row = pc.detect(_bundle())
    t = row['detectors']['A_table'][0]
    assert t['position_asset_id'] == OLD and t['resting_stop_asset_ids'] == [CUR]
    assert t['current_asset_id'] == CUR and t['position_on_current_id'] is False
    assert t['split'] is True and t['stop_on_other_id'] is True
    assert t['resting_stop_qty'] == pytest.approx(1.0)
    assert [x['sym'] for x in row['detectors']['A']] == ['BTCUSD']
    same = pc.detect(_bundle(open_orders=[_stop(aid=OLD)], assets={'BTCUSD': OLD}))
    assert same['detectors']['A'] == [] and same['detectors']['A_table'][0]['split'] is False
    none = pc.detect(_bundle(open_orders=[]))
    assert none['detectors']['A'] == [] and none['detectors']['A_table'][0]['split'] is None
    via_recent = pc.detect(_bundle(open_orders=[], recent=[_stop(aid=CUR, status='canceled')]))
    assert via_recent['detectors']['A_table'][0]['split'] is True
    assert via_recent['detectors']['A_table'][0]['resting_stop_asset_ids'] == []


def test_reservation_anomalies():
    ok = pc.detect(_bundle())                                        # full-qty stop reserves it
    assert ok['detectors']['R'] == []
    phantom = pc.detect(_bundle(open_orders=[]))
    assert [x['kind'] for x in phantom['detectors']['R']] == ['reserve_without_order']
    over = pc.detect(_bundle(open_orders=[_stop(), _stop(oid='s2', qty='0.5')]))
    assert [x['kind'] for x in over['detectors']['R']] == ['over_reserve']
    free = pc.detect(_bundle(positions=[_pos(avail='1.0')], open_orders=[]))
    assert free['detectors']['R'] == []
    closed = pc.detect(_bundle(open_orders=[_stop(status='canceled')]))
    assert [x['kind'] for x in closed['detectors']['R']] == ['reserve_without_order']


# ------------------------------------------------------------------ alert rule

def _row(ts, **det):
    d = {'V': [], 'B': [], 'Q': [], 'A': [], 'R': [], 'E': {'fired': False}}
    for k, v in det.items():
        d[k] = v
    return {'ts': ts, 'detectors': d}


def _v(s):
    return [{'sym': s}]


def test_rule_consecutive_vqe_and_single_q_usd():
    rows = [_row('2026-09-27T00:00:00+00:00', V=_v('BTCUSD')),
            _row('2026-09-27T00:15:00+00:00'),
            _row('2026-09-27T00:30:00+00:00', V=_v('BTCUSD'))]
    assert pc.evaluate(rows)['alerts'] == []                         # not adjacent
    rows = [_row('2026-09-27T00:00:00+00:00', V=_v('BTCUSD')),
            _row('2026-09-27T00:15:00+00:00', V=_v('ETHUSD'))]
    assert pc.evaluate(rows)['alerts'] == []                         # different symbols
    rows = [_row('2026-09-27T00:15:00+00:00', V=_v('BTCUSD'), E={'fired': True}),
            _row('2026-09-27T00:00:00+00:00', V=_v('BTCUSD'), E={'fired': True})]  # unsorted
    rules = sorted(a['rule'] for a in pc.evaluate(rows)['alerts'])
    assert rules == ['E_consecutive', 'V_consecutive']
    q49 = [{'sym': 'BTCUSD', 'usd': 49.99}]
    q50 = [{'sym': 'BTCUSD', 'usd': 50.0}]
    assert pc.evaluate([_row('2026-09-27T00:00:00+00:00', Q=q49)])['alerts'] == []
    al = pc.evaluate([_row('2026-09-27T00:00:00+00:00', Q=q50)])['alerts']
    assert [a['rule'] for a in al] == ['Q_single_usd']
    al = pc.evaluate([_row('2026-09-27T00:00:00+00:00', Q=q49),
                      _row('2026-09-27T00:15:00+00:00', Q=q49)])['alerts']
    assert [a['rule'] for a in al] == ['Q_consecutive']


def test_rule_b_standing_is_log_only_new_loss_that_persists_alerts_a_never_alerts():
    b = [{'sym': 'BTCUSD'}]
    standing = [_row(f'2026-09-27T00:{m:02d}:00+00:00', B=b, A=_v('BTCUSD')) for m in (0, 15, 30)]
    res = pc.evaluate(standing)
    assert res['alerts'] == [] and res['rows_fired']['A'] == 3
    assert res['log_only']['A'] == {'2026-09-27': ['BTCUSD']}
    new = [_row('2026-09-27T00:00:00+00:00'), _row('2026-09-27T00:15:00+00:00', B=b),
           _row('2026-09-27T00:30:00+00:00', B=b)]
    assert [a['rule'] for a in pc.evaluate(new)['alerts']] == ['B_new_consecutive']
    blip = [_row('2026-09-27T00:00:00+00:00'), _row('2026-09-27T00:15:00+00:00', B=b),
            _row('2026-09-27T00:30:00+00:00')]
    assert pc.evaluate(blip)['alerts'] == []


def test_x12_thirty_day_verdict():
    def span(**det):
        return [_row('2026-09-01T00:00:00+00:00', **det), _row('2026-10-02T00:00:00+00:00')]
    assert pc.evaluate([_row('2026-09-01T00:00:00+00:00')])['x12_verdict'] == 'insufficient_span'
    assert pc.evaluate(span())['x12_verdict'] == 'close_x12'
    assert pc.evaluate(span(A=_v('BTCUSD')))['x12_verdict'] == 'close_x12'
    assert pc.evaluate(span(V=_v('BTCUSD')))['x12_verdict'] == 'owner_guard_item'
    assert pc.evaluate(span(E={'fired': True}))['x12_verdict'] == 'owner_guard_item'
    assert pc.evaluate(span(Q=[{'sym': 'X', 'usd': 1.0}]))['x12_verdict'] == 'review'


# ------------------------------------------------------------------ CLI / safety

@pytest.fixture
def no_net(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError('network used')
    monkeypatch.setattr(socket, 'create_connection', _blocked)
    monkeypatch.setattr(socket.socket, 'connect', _blocked)


def test_cli_replay_last_out_and_evaluate(tmp_path, no_net, capsys):
    st = tmp_path / 'position_state.json'
    st.write_text(json.dumps({'hwm': {'BTC/USD': 1.0}}))
    st_bytes = st.read_bytes()
    b1 = _bundle(now='2026-09-27T12:00:00+00:00')
    b1.pop('position_state')
    b2 = _bundle(positions=[_pos(qty='0.9')], now='2026-09-27T12:15:00+00:00')
    b2.pop('position_state')
    for i, b in enumerate((b1, b2), 1):
        (tmp_path / f'b{i}.json').write_text(json.dumps(b))
    out = tmp_path / 'census' / 'rows.jsonl'
    out.parent.mkdir()
    assert pc.main(['--replay', str(tmp_path / 'b1.json'), '--position-state', str(st),
                    '--out', str(out)]) == 0
    assert 'A asset-id table' in capsys.readouterr().out
    assert pc.main(['--replay', str(tmp_path / 'b2.json'), '--position-state', str(st),
                    '--last', str(out), '--out', str(out), '--json', '-']) == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row['prev_ts'] == '2026-09-27T12:00:00+00:00' and 'Q' in row['fired']
    assert len(out.read_text().splitlines()) == 2
    assert st.read_bytes() == st_bytes
    assert pc.main(['--evaluate', str(out.parent), '--json', '-']) == 0
    res = json.loads(capsys.readouterr().out)
    assert res['n_rows'] == 2 and [a['rule'] for a in res['alerts']] == ['Q_single_usd']


def test_cli_alert_rules_and_source_required(capsys):
    assert pc.main(['--alert-rules']) == 0
    txt = capsys.readouterr().out
    assert 'two CONSECUTIVE' in txt and '$50' in txt and 'LOG-ONLY' in txt
    with pytest.raises(SystemExit):
        pc.main([])


def test_write_sink_refuses_live_state_files(tmp_path):
    for name in ('trade_memory.json', 'position_state.json', 'stock__position_state.json'):
        with pytest.raises(SystemExit):
            pc._write_file(str(tmp_path / name), '{}', append=True)
        assert not (tmp_path / name).exists()
