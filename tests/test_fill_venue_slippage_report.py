"""scripts/fill_venue_slippage_report.py — X2 venue-of-fill / X4 markout CLI (measurement-only).

Synthetic bundles only; no network (the socket is blocked for the replay tests).
"""
import datetime as dt
import json
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import fill_venue_slippage_report as fv  # noqa: E402

T0 = dt.datetime(2026, 3, 2, 12, 0, 0)


def _iso(t, digits=9):
    return t.strftime('%Y-%m-%dT%H:%M:%S.') + f'{t.microsecond:06d}{"0" * (digits - 6)}Z'


def _order(oid, i, side='buy', typ='market', cid=None, sym='DOGE/USD'):
    return {'id': oid, 'symbol': sym, 'side': side, 'type': typ,
            'client_order_id': cid or f'{i:08x}-aaaa-bbbb-cccc-000000000000',
            'submitted_at': _iso(T0 + dt.timedelta(hours=i), 6)}


def _bundle(n, *, fill_bps_vs_us_mid=20.0, hs_us_bps=20.0, hs_us1_bps=0.5,
            tactic_cid=None, typ='market', side='buy', move30_bps=0.0, sym='DOGE/USD'):
    """n orders; each fills at arrival us mid*(1+side*fill_bps); us and us-1 mids equal."""
    orders, fills, quotes, bars = {}, [], {}, {}
    ser = bars.setdefault(f'us-1|{sym}', {})
    s = 1 if side == 'buy' else -1
    for i in range(n):
        oid = f'o{i}'
        mid = 0.1
        orders[oid] = _order(oid, i, side, typ, tactic_cid(i) if tactic_cid else None, sym)
        vw = mid * (1 + s * fill_bps_vs_us_mid / 1e4)
        tf = T0 + dt.timedelta(hours=i, seconds=1)
        fills.append({'order_id': oid, 'symbol': sym, 'qty': '100', 'price': str(vw),
                      'transaction_time': _iso(tf)})
        for loc, hs in (('us', hs_us_bps), ('us-1', hs_us1_bps)):
            quotes[f'{oid}|{loc}'] = {'bp': mid * (1 - hs / 1e4), 'ap': mid * (1 + hs / 1e4),
                                      't': _iso(T0 + dt.timedelta(hours=i) - dt.timedelta(seconds=5))}
        for m in range(0, 32):
            px = mid * (1 + (move30_bps / 1e4 if m >= 30 else 0.0))
            ser[(tf.replace(second=0, microsecond=0) + dt.timedelta(minutes=m)).strftime('%Y-%m-%dT%H:%M')] = px
    return {'orders': orders, 'fills': fills, 'arrival_quotes': quotes, 'bars': bars}


def test_parse_ts_handles_alpaca_fraction_widths():
    assert fv.parse_ts('2026-04-26T16:05:43.783273706Z') == dt.datetime(2026, 4, 26, 16, 5, 43, 783273)
    assert fv.parse_ts('2026-03-17T07:52:03.75432Z') == dt.datetime(2026, 3, 17, 7, 52, 3, 754320)
    assert fv.parse_ts('2026-03-17T07:52:03Z') == dt.datetime(2026, 3, 17, 7, 52, 3)
    assert fv.parse_ts('2026-03-17T07:52:03.1+00:00') == dt.datetime(2026, 3, 17, 7, 52, 3, 100000)


def test_classify_tactic_uses_live_client_order_id_tags():
    assert fv.classify_tactic({'client_order_id': 'maker-0123456789abcdef0123'}) == 'maker'
    assert fv.classify_tactic({'client_order_id': 'mktfb-abc', 'type': 'market'}) == 'taker_fallback'
    assert fv.classify_tactic({'client_order_id': 'cstop-abc'}) == 'cstop'
    assert fv.classify_tactic({'client_order_id': 'e117132b-0e9c', 'type': 'market'}) == 'legacy_market'
    assert fv.classify_tactic({'client_order_id': 'e117132b-0e9c', 'type': 'limit'}) == 'legacy_limit'


def test_fee_constants_match_fees_module():
    fees = pytest.importorskip('fees')
    assert fv.MAKER_FEE_BPS == fees.CRYPTO_MAKER_BPS
    assert fv.TAKER_FEE_BPS == fees.CRYPTO_TAKER_BPS


def test_aggregate_fills_vwap_and_crypto_only():
    g = fv.aggregate_fills([
        {'order_id': 'a', 'symbol': 'LINK/USD', 'qty': '1', 'price': '10', 'transaction_time': '2026-01-01T00:00:01Z'},
        {'order_id': 'a', 'symbol': 'LINK/USD', 'qty': '3', 'price': '12', 'transaction_time': '2026-01-01T00:00:02Z'},
        {'order_id': 'b', 'symbol': 'TSLA', 'qty': '1', 'price': '1', 'transaction_time': '2026-01-01T00:00:01Z'}])
    assert set(g) == {'a'}
    assert g['a']['vwap'] == pytest.approx(11.5)
    assert g['a']['n'] == 2 and g['a']['last'] == '2026-01-01T00:00:02Z'


def test_x2_us_spread_wins_when_takers_pay_the_us_half_spread():
    rep = fv.build_report(_bundle(40, fill_bps_vs_us_mid=20.0, hs_us_bps=20.0, hs_us1_bps=0.5), n_boot=200)
    assert rep['x2']['verdict'] == 'us'
    assert rep['x2']['mae_half_spread_us'] == pytest.approx(0.0, abs=1e-6)
    assert rep['x2']['mae_half_spread_us1'] == pytest.approx(19.5, abs=1e-6)
    assert rep['x2']['per_symbol']['DOGE/USD']['n'] == 40


def test_x2_us1_wins_and_no_change_and_insufficient_n():
    assert fv.build_report(_bundle(40, fill_bps_vs_us_mid=0.5, hs_us_bps=20.0, hs_us1_bps=0.5),
                           n_boot=200)['x2']['verdict'] == 'us-1'
    # 10 vs 10 bps error: neither wins by the 25 % margin
    assert fv.build_report(_bundle(40, fill_bps_vs_us_mid=10.0, hs_us_bps=20.0, hs_us1_bps=0.0),
                           n_boot=200)['x2']['verdict'] == 'no_change'
    assert fv.build_report(_bundle(29), n_boot=200)['x2']['verdict'] == 'insufficient_n'


def test_x2_ignores_limit_and_maker_fills():
    b = _bundle(40, typ='limit')
    assert fv.build_report(b, n_boot=200)['x2']['n_taker'] == 0


def test_markout_signs_buy_and_sell():
    rows = fv.build_rows(_bundle(1, fill_bps_vs_us_mid=0.0, move30_bps=10.0))
    assert rows[0]['markout_30'] == pytest.approx(10.0, abs=1e-6)
    assert rows[0]['net_markout_30'] == pytest.approx(10.0 - fv.TAKER_FEE_BPS, abs=1e-6)
    assert rows[0]['markout_1'] == pytest.approx(0.0, abs=1e-6)
    rows = fv.build_rows(_bundle(1, side='sell', fill_bps_vs_us_mid=0.0, move30_bps=10.0))
    assert rows[0]['markout_30'] == pytest.approx(-10.0, abs=1e-6)   # price rose after a sell
    assert rows[0]['slip_us'] == pytest.approx(0.0, abs=1e-6)


def test_price_at_respects_max_lag():
    keys = ['2026-03-02T12:00']
    assert fv.price_at(keys, {'2026-03-02T12:00': 1.0}, dt.datetime(2026, 3, 2, 12, 9)) == 1.0
    assert fv.price_at(keys, {'2026-03-02T12:00': 1.0}, dt.datetime(2026, 3, 2, 12, 11)) is None
    assert fv.price_at(keys, {'2026-03-02T12:00': 1.0}, dt.datetime(2026, 3, 2, 11, 59)) is None


def _maker_vs_taker(maker_move, taker_move, n_m=60, n_t=30):
    m = _bundle(n_m, typ='limit', fill_bps_vs_us_mid=-20.0, move30_bps=maker_move,
                tactic_cid=lambda i: f'maker-{i:020x}')
    t = _bundle(n_t, fill_bps_vs_us_mid=20.0, move30_bps=taker_move)
    # shift taker orders so ids/timestamps do not collide
    t_orders = {f't{k}': dict(v, id=f't{k}') for k, v in t['orders'].items()}
    t_fills = [dict(f, order_id=f"t{f['order_id']}") for f in t['fills']]
    t_quotes = {f't{k}': v for k, v in t['arrival_quotes'].items()}
    return {'orders': {**m['orders'], **t_orders}, 'fills': m['fills'] + t_fills,
            'arrival_quotes': {**m['arrival_quotes'], **t_quotes},
            'bars': {k: {**m['bars'].get(k, {}), **t['bars'].get(k, {})} for k in m['bars']}}


def test_x4_toxic_needs_ci_below_zero_and_min_counts():
    # identical timestamps => same bar series; vary by construction through fill price only:
    # maker fills 20 bps BELOW mid (markout +20 - 15 fee) vs taker 20 bps above (-20 - 25) -> maker better
    rep = fv.build_report(_maker_vs_taker(0.0, 0.0), n_boot=300)
    assert rep['x4']['n_maker'] == 60 and rep['x4']['n_taker'] == 30
    assert rep['x4']['verdict'] == 'not_toxic'
    assert rep['x4']['diff_mean'] == pytest.approx((20 - 15) - (-20 - 25), rel=1e-3)
    few = fv.build_report(_maker_vs_taker(0.0, 0.0, n_m=49), n_boot=300)
    assert few['x4']['verdict'] == 'insufficient_n'


def test_x4_toxic_when_maker_net_markout_is_worse():
    rows = [{'side': 'buy', 'tactic': 'maker', 'type': 'limit', 'net_markout_30': -30.0 + (i % 5)} for i in range(60)]
    rows += [{'side': 'buy', 'tactic': 'taker_fallback', 'type': 'market', 'net_markout_30': -10.0 + (i % 5)} for i in range(30)]
    out = fv.x4_verdict(rows, n_boot=300)
    assert out['verdict'] == 'toxic' and out['ci95'][1] < 0


def test_cli_replay_offline_no_network(tmp_path, monkeypatch, capsys):
    def _blocked(*a, **k):
        raise AssertionError('network touched in --replay')
    monkeypatch.setattr(socket, 'create_connection', _blocked)
    monkeypatch.setattr(socket.socket, 'connect', _blocked)
    p = tmp_path / 'b.json'
    p.write_text(json.dumps(_bundle(35)))
    out = tmp_path / 'r.json'
    assert fv.main(['--replay', str(p), '--json', str(out), '--n-boot', '100']) == 0
    assert '[X2] taker fills n=35' in capsys.readouterr().out
    rep = json.loads(out.read_text())
    assert rep['x2']['verdict'] == 'us' and rep['n_orders'] == 35
    assert fv.main(['--replay', str(p), '--json', '-', '--n-boot', '100']) == 0
    assert json.loads(capsys.readouterr().out)['x4']['verdict'] == 'insufficient_n'
    assert sorted(os.listdir(tmp_path)) == ['b.json', 'r.json']


def test_cli_requires_a_source():
    with pytest.raises(SystemExit):
        fv.main([])
