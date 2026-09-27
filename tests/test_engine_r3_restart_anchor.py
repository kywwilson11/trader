"""ENGINE round-3 W11 — strategy_config.RESTART_STOP_ANCHOR_DESIRED (O2 + J11
+ J2: ONE anchor rule for the server stop re-placed after a restart).

Startup cancels this bot's working orders, rebuilds positions, then
``_replace_protective_stops`` re-places a server stop per position. Legacy
(flag OFF, default): ``max(entry, hwm) * (1 - hard stop distance)`` — the
HARD distance anchored at the HWM — while the software truth
``base_loop._desired_stop_for`` is max(entry-anchored hard stop,
HWM-anchored TRAIL when trailing is active). Consequences (ENGINE r1/r2):
  O2  crypto zero-basis book: 6 % startup stop, then cycle 1 cancels and
      re-places all six at the 5 % trail -> 18 order writes instead of 6;
  J11 entry 100 / hwm 101 / ATR 1 (trail NOT active): resting stop 98.475,
      software hard stop 97.5 -> a fill is classified 'trail' (wrong journal
      kind, wrong lockout under STOP_CLASSIFY_V2), and the level differs
      from the policy the backtest validated.
ON: both books place exactly ``_desired_stop_for(pos)[0]`` (stocks rounded
to cents like the legacy path); crypto's ``_resting_stop_px`` records that
level, so cycle 1's ``_maybe_update_resting_stop`` sees no improvement.
J2 decision (stocks, pinned below): ON does NOT place a native
trailing_stop for a restored ``trailing_activated=True`` and leaves the flag
as restored — a native trail re-anchors at the submit-time price with the
entry denominator (stock_loop._manage_stops) and so cannot equal
``_desired_stop_for``; the plain stop at the software trail level can.

OFF pins: literal goldens captured from the PRE-EDIT crypto_loop.py /
stock_loop.py (scratch copies, ENGINE W11), plus full-tape A/B runs against
the verbatim legacy bodies kept below.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import crypto_loop                                 # noqa: E402
import stock_loop                                  # noqa: E402
import strategy_config                             # noqa: E402
from types_mod import Position                     # noqa: E402
from fake_alpaca_broker import FakeClock           # noqa: E402
from test_engine_r1_inherited_positions import (   # noqa: E402,F401
    harness, drive as r1_drive, FLAT_PREDS)
from test_engine_r2_replay_harness import (        # noqa: E402,F401
    replay, CP, SNAPSHOT, UNIVERSE, _load_tape, _crypto_cash_snapshot,
    _preds_b)

FLAG = 'RESTART_STOP_ANCHOR_DESIRED'


def _set_flag(monkeypatch, on):
    # raising=False: the mutation check runs this file against the pre-edit
    # modules, where the constant does not exist yet.
    monkeypatch.setattr(strategy_config, FLAG, on, raising=False)


def _proj(events):
    """Order-event projection used by the literal goldens."""
    return [[e['kind'], e.get('symbol'), e.get('type'), e.get('stop_price'),
             e.get('limit_price'), e['step']]
            for e in events if e['kind'] in ('submit', 'cancel', 'fill',
                                             'reject', 'expire')]


def _writes(events):
    return [e for e in events if e['kind'] in ('submit', 'cancel')]


# --- verbatim legacy bodies (pre-ENGINE-R3 W11) for the A/B OFF pins -------

def _legacy_crypto_replace(self):
    for symbol, pos in self.positions.items():
        anchor = max(pos.entry_price, pos.high_water_mark)
        stop_price = anchor * (1 - self._stop_distance_for(pos))
        self._place_resting_stop(symbol, pos, stop_price)


def _legacy_stock_replace(self):
    for symbol, info in self.positions.items():
        try:
            entry_atr = info.entry_atr
            if entry_atr is not None and info.entry_price > 0:
                raw = (entry_atr * self.ATR_STOP_MULTIPLIER) / info.entry_price
                stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw))
            else:
                stop_dist = self.STOP_LOSS_PCT
            anchor = max(info.entry_price, info.high_water_mark)
            stop_price = round(anchor * (1 - stop_dist), 2)
            qty = int(float(info.qty))
            if qty <= 0:
                continue
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side='sell',
                type='stop', stop_price=stop_price,
                time_in_force='day',
                client_order_id=stock_loop.make_client_order_id('restop'),
            )
            info.stop_order_id = order.id
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

BTC_E = CP['BTC/USD']          # stub ATR = 1 % of this (replay fixture)
BTC_Q = 0.02


def _j11_snapshot():
    snap = json.loads(json.dumps(SNAPSHOT))
    eq = 30000.0 + BTC_Q * BTC_E
    snap['account'].update(cash='30000', non_marginable_buying_power='30000',
                           buying_power='30000', equity=repr(eq),
                           portfolio_value=repr(eq), last_equity=repr(eq))
    snap['positions'] = [{'symbol': 'BTCUSD', 'qty': repr(BTC_Q),
                          'avg_entry_price': repr(BTC_E),
                          'cost_basis': repr(BTC_Q * BTC_E),
                          'current_price': repr(BTC_E),
                          'asset_class': 'crypto'}]
    return snap


def _run_crypto_j11(replay, btc_mults, n):
    """BTC held with a REAL basis E, persisted hwm 1.01*E (trail NOT armed:
    trail_activate_pct 1.5 %), ATR = 1 % of E -> the J11 numbers x E/100."""
    tape = {s: [CP[s]] * len(btc_mults) for s in CP}
    tape['BTC/USD'] = [BTC_E * m for m in btc_mults]
    loop, broker = replay.build_crypto(snapshot=_j11_snapshot(), tape=tape,
                                       clock=FakeClock())
    replay.state_path.write_text(json.dumps({'hwm': {'BTC/USD': BTC_E * 1.01}}))
    marks = replay.drive(loop, broker, n)
    return loop, broker, marks


STOCK_ENTRY, STOCK_HWM, STOCK_OPEN = 50.0, 51.5, 51.0   # stub ATR 0.5 = 1 %


def _run_stock_keeper(replay, n, trailing=True, hwm=STOCK_HWM):
    """AAA held from entry 50 (ATR 0.5), persisted hwm and trailing flag;
    synthetic RTH tape opens at 51 (-4 % gap at 10:00 -> 48.96)."""
    loop, broker, tape = replay.build_stock(
        lambda now: {'AAA': 0.0}, names={'AAA': STOCK_OPEN})
    broker.positions['AAA'] = {
        'symbol': 'AAA', 'qty': 100.0, 'avg_entry_price': STOCK_ENTRY,
        'cost_basis': 100 * STOCK_ENTRY, 'asset_class': 'us_equity',
        'snapshot_price': STOCK_OPEN}
    replay.state_path.write_text(json.dumps(
        {'hwm': {'AAA': hwm}, 'trailing': {'AAA': trailing}}))
    marks = replay.drive(loop, broker, n, seconds=300.0)
    return loop, broker, marks


# Literal goldens: _proj(...) of the PRE-EDIT modules (flag absent), ENGINE
# W11 capture run 2026-09-27 (scratch w11/golden.json).
GOLDEN = {
    'r1_zero_basis': [
        ['submit', 'BTC/USD', 'stop_limit', 79302.724, 77716.6695, 0],
        ['submit', 'ETH/USD', 'stop_limit', 2531.3636, 2480.7363, 0],
        ['submit', 'XRP/USD', 'stop_limit', 1.4246, 1.3961, 0],
        ['submit', 'SOL/USD', 'stop_limit', 113.317, 111.0507, 0],
        ['submit', 'DOGE/USD', 'stop_limit', 0.090508, 0.088698, 0],
        ['submit', 'LINK/USD', 'stop_limit', 13.2446, 12.9797, 0],
        ['cancel', 'BTC/USD', 'stop_limit', 79302.724, 77716.6695, 1],
        ['submit', 'BTC/USD', 'stop_limit', 80146.37, 78543.4426, 1],
        ['cancel', 'ETH/USD', 'stop_limit', 2531.3636, 2480.7363, 1],
        ['submit', 'ETH/USD', 'stop_limit', 2558.293, 2507.1271, 1],
        ['cancel', 'XRP/USD', 'stop_limit', 1.4246, 1.3961, 1],
        ['submit', 'XRP/USD', 'stop_limit', 1.4398, 1.411, 1],
        ['cancel', 'SOL/USD', 'stop_limit', 113.317, 111.0507, 1],
        ['submit', 'SOL/USD', 'stop_limit', 114.5225, 112.232, 1],
        ['cancel', 'DOGE/USD', 'stop_limit', 0.090508, 0.088698, 1],
        ['submit', 'DOGE/USD', 'stop_limit', 0.091471, 0.089642, 1],
        ['cancel', 'LINK/USD', 'stop_limit', 13.2446, 12.9797, 1],
        ['submit', 'LINK/USD', 'stop_limit', 13.3855, 13.1178, 1],
    ],
    # identical to r1 (same zero-basis book, flat tape): 6 + 6 cancels + 6
    'r2_o2': [
        ['submit', 'BTC/USD', 'stop_limit', 79302.724, 77716.6695, 0],
        ['submit', 'ETH/USD', 'stop_limit', 2531.3636, 2480.7363, 0],
        ['submit', 'XRP/USD', 'stop_limit', 1.4246, 1.3961, 0],
        ['submit', 'SOL/USD', 'stop_limit', 113.317, 111.0507, 0],
        ['submit', 'DOGE/USD', 'stop_limit', 0.090508, 0.088698, 0],
        ['submit', 'LINK/USD', 'stop_limit', 13.2446, 12.9797, 0],
        ['cancel', 'BTC/USD', 'stop_limit', 79302.724, 77716.6695, 1],
        ['submit', 'BTC/USD', 'stop_limit', 80146.37, 78543.4426, 1],
        ['cancel', 'ETH/USD', 'stop_limit', 2531.3636, 2480.7363, 1],
        ['submit', 'ETH/USD', 'stop_limit', 2558.293, 2507.1271, 1],
        ['cancel', 'XRP/USD', 'stop_limit', 1.4246, 1.3961, 1],
        ['submit', 'XRP/USD', 'stop_limit', 1.4398, 1.411, 1],
        ['cancel', 'SOL/USD', 'stop_limit', 113.317, 111.0507, 1],
        ['submit', 'SOL/USD', 'stop_limit', 114.5225, 112.232, 1],
        ['cancel', 'DOGE/USD', 'stop_limit', 0.090508, 0.088698, 1],
        ['submit', 'DOGE/USD', 'stop_limit', 0.091471, 0.089642, 1],
        ['cancel', 'LINK/USD', 'stop_limit', 13.2446, 12.9797, 1],
        ['submit', 'LINK/USD', 'stop_limit', 13.3855, 13.1178, 1],
    ],
    'crypto_j11': [
        ['submit', 'BTC/USD', 'stop_limit', 83078.0399, 81416.4791, 0],
        ['fill', 'BTC/USD', 'stop_limit', 83078.0399, 81416.4791, 2],
    ],
    'stock_keeper': [
        ['submit', 'AAA', 'stop', 50.47, None, 0],
        ['fill', 'AAA', 'stop', 50.47, None, 9],
    ],
}


# ---------------------------------------------------------------------------
# 0. The constant
# ---------------------------------------------------------------------------

def test_constant_default_off():
    assert getattr(strategy_config, FLAG) is False


# ---------------------------------------------------------------------------
# 1. OFF = byte-identical (literal goldens from the pre-edit modules)
# ---------------------------------------------------------------------------

def test_off_r1_zero_basis_restart_golden(harness, monkeypatch):
    _set_flag(monkeypatch, False)
    loop, broker, rec = harness([1.0, 1.0, 1.0], FLAT_PREDS)
    r1_drive(loop, broker, rec, 2)
    assert _proj(broker.events) == GOLDEN['r1_zero_basis']


def test_off_r2_o2_restart_golden(replay, monkeypatch):
    _set_flag(monkeypatch, False)
    loop, broker = replay.build_crypto(tape={s: [CP[s]] * 3 for s in CP},
                                       clock=FakeClock())
    replay.drive(loop, broker, 2)
    assert _proj(broker.events) == GOLDEN['r2_o2']
    assert len(_writes(broker.events)) == 18


def test_off_crypto_j11_golden(replay, monkeypatch):
    _set_flag(monkeypatch, False)
    loop, broker, marks = _run_crypto_j11(replay, [1.0, 1.0, 0.97], 2)
    assert _proj(broker.events) == GOLDEN['crypto_j11']
    sells = replay.rec.rows('sell')
    assert [r['server_stop_kind'] for r in sells] == ['trail']   # legacy label


def test_off_stock_keeper_golden(replay, monkeypatch):
    _set_flag(monkeypatch, False)
    loop, broker, marks = _run_stock_keeper(replay, 10)
    assert _proj(broker.events) == GOLDEN['stock_keeper']


@pytest.mark.parametrize('scenario', ['inherited_book', 'cash_entries'])
def test_off_full_tape_equals_verbatim_legacy(replay, monkeypatch, scenario):
    """Full recorded tape, both r2 scenarios: flag OFF == the verbatim
    pre-edit _replace_protective_stops (normalized broker events)."""
    _set_flag(monkeypatch, False)
    out = []
    for legacy in (False, True):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()
        tape = _load_tape()
        if scenario == 'inherited_book':
            loop, broker = replay.build_crypto(tape=tape, replay_quote_age=True)
        else:
            loop, broker = replay.build_crypto(
                snapshot=_crypto_cash_snapshot(), tape=tape,
                preds_fn=_preds_b, replay_quote_age=True)
        if legacy:
            loop._replace_protective_stops = \
                _legacy_crypto_replace.__get__(loop)
        replay.drive(loop, broker, len(tape) - 1)
        out.append(broker.normalized_events())
    assert out[0] == out[1] and len(out[0]) >= 6


def test_off_stock_equals_verbatim_legacy(replay, monkeypatch):
    _set_flag(monkeypatch, False)
    out = []
    for legacy in (False, True):
        replay.rec.__init__()
        loop, broker, tape = replay.build_stock(
            lambda now: {'AAA': 0.0}, names={'AAA': STOCK_OPEN})
        broker.positions['AAA'] = {
            'symbol': 'AAA', 'qty': 100.0, 'avg_entry_price': STOCK_ENTRY,
            'cost_basis': 100 * STOCK_ENTRY, 'asset_class': 'us_equity',
            'snapshot_price': STOCK_OPEN}
        replay.state_path.write_text(json.dumps(
            {'hwm': {'AAA': 50.4}, 'trailing': {'AAA': False}}))
        if legacy:
            loop._replace_protective_stops = \
                _legacy_stock_replace.__get__(loop)
        replay.drive(loop, broker, 10, seconds=300.0)
        out.append(broker.normalized_events())
    assert out[0] == out[1] and len(out[0]) >= 2


# ---------------------------------------------------------------------------
# 2. ON — crypto
# ---------------------------------------------------------------------------

def test_on_zero_basis_restart_six_orders_no_churn(replay, monkeypatch):
    """O2: the real zero-basis snapshot, flat tape -> exactly 6 order writes
    in startup + cycle 1 (0 cancels), each at the 5 % trail level
    _desired_stop_for returns; nothing in cycle 2 either."""
    _set_flag(monkeypatch, True)
    loop, broker = replay.build_crypto(tape={s: [CP[s]] * 3 for s in CP},
                                       clock=FakeClock())
    marks = replay.drive(loop, broker, 2)
    start = broker.events[:marks[0]]
    c1 = broker.events[marks[0]:marks[1]]
    assert len(_writes(start)) == 6 and _writes(c1) == []
    assert not broker.events_of('cancel')
    assert broker.events[marks[1]:marks[2]] == []
    loop.macro_regime = None
    for e in _writes(start):
        s = e['symbol']
        pos = loop.positions[s]
        want = loop._desired_stop_for(pos)[0]
        assert want == pytest.approx(CP[s] * (1 - strategy_config.CRYPTO_POLICY[
            'trail_fallback_pct']))
        assert e['type'] == 'stop_limit'
        assert e['stop_price'] == crypto_loop.CryptoLoop._round_px(want)
        assert loop._resting_stop_px[s] == want
        assert pos.stop_order_id == e['id']
    assert all(len(broker.open_orders(s)) == 1 for s in UNIVERSE)


def test_on_crypto_j11_rests_at_software_hard_stop_and_classifies_hard(
        replay, monkeypatch):
    """J11: entry E / hwm 1.01E / ATR 0.01E, trail not armed -> the resting
    stop is the software hard stop 0.975E (97.5 per 100), not the legacy
    0.98475E (98.475); a fill there is 'hard'."""
    _set_flag(monkeypatch, True)
    loop, broker, marks = _run_crypto_j11(replay, [1.0, 1.0, 0.97], 2)
    start = broker.events[:marks[0]]
    (st,) = [e for e in start if e['kind'] == 'submit']
    hard = BTC_E * (1 - 0.025)
    assert st['stop_price'] == crypto_loop.CryptoLoop._round_px(hard)
    assert st['stop_price'] != crypto_loop.CryptoLoop._round_px(
        BTC_E * 1.01 * 0.975)
    # cycle 1 (flat): no cancel/replace — the level already equals desired
    assert _writes(broker.events[marks[0]:marks[1]]) == []
    sells = replay.rec.rows('sell')
    assert len(sells) == 1 and sells[0]['exit_reason'] == 'server_stop'
    assert sells[0]['server_stop_kind'] == 'hard'
    assert sells[0]['stop_px'] == pytest.approx(hard, rel=1e-9)


def test_on_crypto_classify_levels_unit(replay, monkeypatch):
    """_classify_server_stop on a fill AT the placed level: ON 'hard',
    legacy 'trail' (same position, trail not armed)."""
    _set_flag(monkeypatch, True)
    loop, broker = replay.build_crypto(tape={s: [CP[s]] for s in CP},
                                       clock=FakeClock())
    loop.macro_regime = None
    pos = Position(qty=0.001, entry_price=100.0, high_water_mark=101.0,
                   entry_atr=1.0)
    loop.positions = {'BTC/USD': pos}
    loop._replace_protective_stops()
    placed = loop._resting_stop_px['BTC/USD']
    assert placed == pytest.approx(97.5)
    from types import SimpleNamespace as NS
    assert loop._classify_server_stop('BTC/USD', pos, NS(stop_price=placed)) \
        == ('hard', placed)
    assert loop._classify_server_stop('BTC/USD', pos, NS(stop_price=98.475)) \
        == ('trail', 98.475)


GRID_R = (1.0, 1.01, 1.015, 1.03, 1.05, 1.10)
GRID_A = (0.005, 0.01, 0.02, 0.04)


@pytest.mark.parametrize('on', [False, True])
def test_crypto_restart_level_over_grid(replay, monkeypatch, on):
    """Every (hwm/entry, ATR/entry) cell: ON places exactly
    _desired_stop_for(pos)[0]; OFF places the legacy formula."""
    _set_flag(monkeypatch, on)
    loop, broker = replay.build_crypto(tape={s: [CP[s]] for s in CP},
                                       clock=FakeClock())
    loop.macro_regime = None
    for r in GRID_R:
        for a in GRID_A:
            pos = Position(qty=0.001, entry_price=100.0,
                           high_water_mark=100.0 * r, entry_atr=100.0 * a)
            loop.positions = {'BTC/USD': pos}
            loop._replace_protective_stops()
            got = loop._resting_stop_px['BTC/USD']
            if on:
                assert got == loop._desired_stop_for(pos)[0], (r, a)
            else:
                assert got == 100.0 * r * (1 - loop._stop_distance_for(pos))


# ---------------------------------------------------------------------------
# 3. ON — stocks (J2 decision pinned)
# ---------------------------------------------------------------------------

def test_on_stock_keeper_trailing_restored_plain_stop_at_desired(
        replay, monkeypatch):
    """entry 50 / hwm 51.5 / ATR 0.5, trailing_activated restored True.
    ON: ONE plain 'stop' at round(_desired_stop_for, 2) = 50.50 (the software
    trail hwm - 2 ATR), not the legacy 50.47; J2 decision: no native
    trailing_stop is ever placed, the flag stays as restored (True), and the
    gap fill is classified 'trail' — correct, since the stop IS the trail
    level (above the hard stop 49.00)."""
    _set_flag(monkeypatch, True)
    loop, broker, marks = _run_stock_keeper(replay, 10)
    subs = [e for e in broker.events if e['kind'] == 'submit']
    assert [(e['symbol'], e['type']) for e in subs] == [('AAA', 'stop')]
    assert subs[0]['stop_price'] == 50.50
    assert not broker.events_of('cancel')
    pos = Position(qty=100, entry_price=STOCK_ENTRY, high_water_mark=STOCK_HWM,
                   entry_atr=0.5, trailing_activated=True)
    loop.macro_regime = None
    assert round(loop._desired_stop_for(pos)[0], 2) == 50.50
    sells = replay.rec.rows('sell')
    assert len(sells) == 1 and sells[0]['exit_reason'] == 'server_stop'
    assert sells[0]['server_stop_kind'] == 'trail'
    assert sells[0]['stop_px'] == 50.50


def test_on_stock_trailing_flag_kept_and_never_upgraded(replay, monkeypatch):
    """J2 pin, before any fill: 5 cycles (09:20-09:40 ET, open from 09:30,
    price 51 >= the 1 % activation) — the restored flag stays True and the
    upgrade path never cancels the plain stop for a native trailing_stop."""
    _set_flag(monkeypatch, True)
    loop, broker, marks = _run_stock_keeper(replay, 5)
    pos = loop.positions['AAA']
    assert pos.trailing_activated is True
    assert [e['type'] for e in broker.events if e['kind'] == 'submit'] == ['stop']
    assert broker.open_orders('AAA')[0]['id'] == pos.stop_order_id


def test_on_stock_j11_hard_stop_and_classify(replay, monkeypatch):
    """Stock J11 analogue: entry 50 / hwm 50.4 (+0.8 %, below the 1 %
    activation) / ATR 0.5, trailing False. ON 49.00 = the hard stop ->
    'hard'; legacy 49.39 -> 'trail'."""
    from types import SimpleNamespace as NS
    for on, want, kind in ((True, 49.00, 'hard'), (False, 49.39, 'trail')):
        _set_flag(monkeypatch, on)
        replay.rec.__init__()
        loop, broker, tape = replay.build_stock(
            lambda now: {'AAA': 0.0}, names={'AAA': STOCK_OPEN})
        broker.positions['AAA'] = {
            'symbol': 'AAA', 'qty': 100.0, 'avg_entry_price': STOCK_ENTRY,
            'cost_basis': 100 * STOCK_ENTRY, 'asset_class': 'us_equity',
            'snapshot_price': STOCK_OPEN}
        loop.macro_regime = None
        pos = Position(qty=100, entry_price=STOCK_ENTRY, high_water_mark=50.4,
                       entry_atr=0.5, trailing_activated=False)
        loop.positions = {'AAA': pos}
        loop._replace_protective_stops()
        (sub,) = [e for e in broker.events if e['kind'] == 'submit']
        assert sub['type'] == 'stop' and sub['stop_price'] == want
        assert loop._classify_server_stop('AAA', pos, NS(stop_price=want))[0] \
            == kind


def test_flag_read_at_call_time(replay, monkeypatch):
    """getattr on strategy_config at call time: flipping the constant between
    two calls on the SAME loop changes the level."""
    loop, broker = replay.build_crypto(tape={s: [CP[s]] for s in CP},
                                       clock=FakeClock())
    loop.macro_regime = None
    lv = []
    for on in (False, True):
        _set_flag(monkeypatch, on)
        pos = Position(qty=0.001, entry_price=100.0, high_water_mark=101.0,
                       entry_atr=1.0)
        loop.positions = {'BTC/USD': pos}
        loop._replace_protective_stops()
        lv.append(loop._resting_stop_px['BTC/USD'])
    assert lv == [pytest.approx(98.475), pytest.approx(97.5)]
