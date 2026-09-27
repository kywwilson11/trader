"""ENGINE round-1 W1: what the crypto loop DOES with the inherited paper book.

Drives the REAL ``CryptoLoop.run()`` (startup order-cancel ->
_reconstruct_positions -> _replace_protective_stops -> cycles) against
tests/fake_alpaca_broker.py loaded with the real 2026-09-27 00:20 account
snapshot: six crypto positions, every one with avg_entry_price=0 /
cost_basis=0 (Alpaca paper quirk), cash $93.63, zero open orders.

Everything network-touching is stubbed at its real seam (predictions are a
fixed dict; macro regime neutral; LLM disabled; no journal/state file in the
repo is touched — all redirected to tmp_path or captured). Each assertion
states the policy intent it checks, citing strategy_config.CRYPTO_POLICY:
stop_fallback_pct 0.06, trail_fallback_pct 0.05, trail_activate_pct 0.015.
Tests marked CURRENT-BEHAVIOUR document an owner item (see the ENGINE W1
report) rather than bless it.
"""

import datetime as _dt
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import base_loop                                   # noqa: E402
import crypto_loop                                 # noqa: E402
import order_utils                                 # noqa: E402
from types_mod import MacroRegime, Position        # noqa: E402
from strategy_config import CRYPTO_POLICY, MAX_BOOK_RISK_PCT  # noqa: E402
from fake_alpaca_broker import FakeAlpacaBroker    # noqa: E402

P = CRYPTO_POLICY

# --- the real account snapshot (generals/engine/account_snapshot.json,
#     read-only GET at 2026-09-27 00:20; fields the loop reads) ------------
SNAPSHOT = {
    'account': {
        'status': 'ACTIVE', 'crypto_status': 'ACTIVE', 'currency': 'USD',
        'buying_power': '374.52', 'regt_buying_power': '187.26',
        'effective_buying_power': '374.52',
        'non_marginable_buying_power': '93.63', 'cash': '93.63',
        'portfolio_value': '121930.35', 'equity': '121930.35',
        'last_equity': '123095.9849438596608',
        'long_market_value': '121836.72', 'multiplier': '1',
        'shorting_enabled': False, 'trading_blocked': False,
    },
    'positions': [
        {'symbol': 'BTCUSD', 'qty': '0.249291739', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '84364.6', 'asset_class': 'crypto'},
        {'symbol': 'DOGEUSD', 'qty': '114913.12732127', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '0.0962855', 'asset_class': 'crypto'},
        {'symbol': 'ETHUSD', 'qty': '7.904614454', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '2692.94', 'asset_class': 'crypto'},
        {'symbol': 'LINKUSD', 'qty': '1576.529633666', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '14.09', 'asset_class': 'crypto'},
        {'symbol': 'SOLUSD', 'qty': '212.082911929', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '120.55', 'asset_class': 'crypto'},
        {'symbol': 'XRPUSD', 'qty': '13641.540903233', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '1.51554', 'asset_class': 'crypto'},
    ],
    'open_orders': [],
}
CP = {p['symbol'][:-3] + '/USD': float(p['current_price'])
      for p in SNAPSHOT['positions']}
QTY = {p['symbol'][:-3] + '/USD': float(p['qty'])
       for p in SNAPSHOT['positions']}
UNIVERSE = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD', 'LINK/USD']
SPREAD_BPS = 10.0
ATR_FRAC = 0.01          # stub hourly ATR = 1% of price (sizing input only)
THRESHOLD = 0.15


def _rp(p):
    return crypto_loop.CryptoLoop._round_px(p)


class Rec:
    def __init__(self):
        self.journal, self.trades, self.risk, self.notes = [], [], [], []
        self.errors = []

    def rows(self, action):
        return [r for r in self.journal if r.get('action') == action]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """build(mults, preds) -> (loop, broker, rec); run via drive(loop, n)."""
    rec = Rec()
    for k in ('TRADER_ORDER_STREAM', 'TRADER_USE_ALPACA_PY'):
        monkeypatch.delenv(k, raising=False)
    import time as _time, random as _random
    monkeypatch.setattr(_time, 'sleep', lambda s: None)
    monkeypatch.setattr(_random, 'uniform', lambda a, b: 0.0)

    import notify, macro_calendar, trade_journal, portfolio, market_data
    import risk_budget, funding, volatility, meta_label, shadow
    monkeypatch.setattr(notify, 'flatten_requested', lambda: False)
    monkeypatch.setattr(notify, 'halt_active', lambda: False)
    monkeypatch.setattr(notify, 'set_halt', lambda *a, **k: None)
    monkeypatch.setattr(notify, 'notify', lambda *a, **k: rec.notes.append(a))
    monkeypatch.setattr(notify, 'ping_heartbeat', lambda *a, **k: None)
    monkeypatch.setattr(macro_calendar, 'macro_standdown', lambda: (False, ''))
    monkeypatch.setattr(macro_calendar, 'calendar_exhausted', lambda: False)
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', tmp_path)
    monkeypatch.setattr(trade_journal, 'log_decision',
                        lambda row: rec.journal.append(dict(row)))
    monkeypatch.setattr(base_loop, 'log_decision',
                        lambda row: rec.journal.append(dict(row)))
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: rec.trades.append((a, k)))
    monkeypatch.setattr(base_loop, 'compute_kelly_fraction', lambda **k: None)
    monkeypatch.setattr(base_loop, 'get_macro_regime', lambda api, at: MacroRegime(
        stress_level=0.0, vix=18.0, cape=None, regime_label='neutral'))
    monkeypatch.setattr(base_loop, 'get_gpu_temp', lambda: None)
    monkeypatch.setattr(base_loop, 'sentiment_gate', lambda s, a: (1.0, []))
    monkeypatch.setattr(base_loop, 'load_llm_config', lambda: {'enabled': False})
    monkeypatch.setattr(base_loop, '_flatten_flag_path',
                        lambda book: str(tmp_path / f'flatten_{book}.flag'))
    monkeypatch.setattr(market_data, 'get_live_atr',
                        lambda api, s, asset_type=None: CP.get(s, 100.0) * ATR_FRAC)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca', lambda *a, **k: None)
    monkeypatch.setattr(crypto_loop, 'fetch_bars_alpaca', lambda *a, **k: None)
    monkeypatch.setattr(portfolio, 'get_correlation_matrix_cached',
                        lambda *a, **k: {})
    monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                        lambda *a, **k: 1.0)
    monkeypatch.setattr(risk_budget, 'record_book_risk_and_report',
                        lambda book, risks, rho, **k: rec.risk.append(list(risks)))
    monkeypatch.setattr(funding, 'funding_tilt', lambda s: 1.0)
    monkeypatch.setattr(volatility, 'get_crypto_rv_mult',
                        lambda *a, **k: (1.0, 'normal', None))
    monkeypatch.setattr(volatility, 'update_crypto_rv_state', lambda *a, **k: None)
    monkeypatch.setattr(meta_label, 'meta_probability_live', lambda *a, **k: None)
    monkeypatch.setattr(shadow, 'maybe_log_shadow', lambda *a, **k: None)

    def build(mults, preds, extra_tape=None):
        tape = {s: [CP[s] * m for m in mults] for s in CP}
        tape.update(extra_tape or {})
        broker = FakeAlpacaBroker(SNAPSHOT, tape=tape, spread_bps=SPREAD_BPS)
        monkeypatch.setattr(base_loop, 'get_api', lambda: broker)
        monkeypatch.setattr(base_loop.BaseTradingLoop, '_load_hard_stop_lockout',
                            lambda self: None)
        loop = crypto_loop.CryptoLoop()
        loop._lockout_file = str(tmp_path / 'hard_stop_lockout.json')
        loop._position_state_file = lambda: str(tmp_path / 'position_state.json')
        loop._hot_reload_check = lambda: None

        def _load_models():
            loop.model = object()
            loop.config = {'trade_threshold': THRESHOLD, 'forward_bars': 24}
            loop.trade_threshold = THRESHOLD
        loop._load_models = _load_models
        loop._get_predictions = lambda bench: (dict(preds), {})
        return loop, broker, rec

    yield build


def drive(loop, broker, rec, n_cycles):
    """Real run(): startup, then n cycles; the tape advances one step
    BEFORE each cycle (startup = step 0, cycle k = step k). Returns the
    broker event index at the end of startup and of each cycle."""
    marks = []
    real = crypto_loop.CryptoLoop._run_one_cycle

    def one():
        if not marks:
            marks.append(len(broker.events))     # end of startup
        broker.tick()
        try:
            real(loop)
        except Exception as e:                   # 'nothing raises' evidence
            rec.errors.append(repr(e))
            raise KeyboardInterrupt
        marks.append(len(broker.events))
        if loop.cycle >= n_cycles:
            raise KeyboardInterrupt
    loop._run_one_cycle = one
    # pytest re-sets PYTEST_CURRENT_TEST per phase; drop it for the call so
    # order_utils._journal_entry_fills reaches the CAPTURED journal (the
    # suppression exists to protect the live journal, which is stubbed).
    os.environ.pop('PYTEST_CURRENT_TEST', None)
    with pytest.raises(KeyboardInterrupt):
        loop.run()
    assert rec.errors == []
    return marks


def _stops(events):
    return {e['symbol']: e for e in events
            if e['kind'] == 'submit' and e['type'] == 'stop_limit'}


FLAT_PREDS = {s: 0.0 for s in UNIVERSE}
BULL_BTC = dict(FLAT_PREDS, **{'BTC/USD': 2.0})


# ---------------------------------------------------------------------------
# 1. Startup: reconstruction + resting stops at the 6% fallback
# ---------------------------------------------------------------------------

def test_startup_reconstructs_zero_basis_and_rests_6pct_fallback_stops(harness):
    loop, broker, rec = harness([1.0, 1.0, 1.0], FLAT_PREDS)
    marks = drive(loop, broker, rec, 2)
    startup = broker.events[:marks[0]]
    # Startup cancel is universe-scoped and finds nothing (0 open orders).
    assert not [e for e in startup if e['kind'] in ('cancel', 'cancel_all')]
    # All six inherited positions are tracked, qty exact, entry_price 0.0
    # (order_utils.reconstruct_positions passes avg_entry_price through).
    assert set(loop.positions) == set(UNIVERSE)
    for s, pos in loop.positions.items():
        assert pos.qty == pytest.approx(QTY[s])
        assert pos.entry_price == 0.0
        # TP needs entry_price > 0 (_reconstruct_positions) -> none.
        assert pos.take_profit_price is None
    # Startup peak = first real equity read (no saved state): the drawdown
    # ladder starts at 0 dd, not against the $100k seed (c26 D15).
    assert loop._peak_equity == pytest.approx(121930.35, abs=0.01)
    # _replace_protective_stops: anchor=max(entry 0, hwm=current price);
    # _stop_distance_for falls back to stop_fallback_pct because entry<=0
    # (the ATR branch needs entry_price > 0) -> stop 6% below the CURRENT
    # price, limit RESTING_STOP_LIMIT_GAP (2%) under the trigger, full qty.
    st = _stops(startup)
    assert set(st) == set(UNIVERSE) and len(startup) == 6
    for s, e in st.items():
        assert P['stop_fallback_pct'] == 0.06
        assert e['stop_price'] == _rp(CP[s] * (1 - 0.06))
        assert e['limit_price'] == _rp(CP[s] * 0.94 * 0.98)
        assert e['qty'] == pytest.approx(QTY[s])
        assert e['side'] == 'sell' and e['type'] == 'stop_limit'


# ---------------------------------------------------------------------------
# 2. _desired_stop_for with entry_price=0 == pure 5% trail from the HWM
# ---------------------------------------------------------------------------

def test_desired_stop_for_zero_basis_is_pure_5pct_trail(harness):
    loop, broker, rec = harness([1.0], FLAT_PREDS)
    loop.macro_regime = None
    for s in UNIVERSE:
        hwm = CP[s]
        pos = Position(qty=QTY[s], entry_price=0.0, high_water_mark=hwm,
                       entry_atr=hwm * ATR_FRAC)
        stop, stop_d, trail_d, active = loop._desired_stop_for(pos)
        # entry 0 -> ATR branch skipped -> fallback distances.
        assert (stop_d, trail_d) == (P['stop_fallback_pct'],
                                     P['trail_fallback_pct'])
        # trail_activate_pct 0.015 is meant to arm the trail only after a
        # +1.5% gain; hwm >= 0*(1.015) holds for ANY price -> always armed.
        assert active is True
        # The hard stop entry*(1-0.06) == 0 can never bind (price <= 0);
        # the effective protection is the 5% trail off the running HWM.
        assert stop == pytest.approx(hwm * (1 - P['trail_fallback_pct']))
    # Counterfactual (policy intent for a real entry at today's price):
    # ATR stop 2.5x1% = 2.5% hard stop, trail NOT armed at hwm == entry.
    pos = Position(qty=1.0, entry_price=100.0, high_water_mark=100.0,
                   entry_atr=1.0)
    stop, stop_d, _, active = loop._desired_stop_for(pos)
    assert active is False and stop == pytest.approx(100.0 * (1 - 0.025))


# ---------------------------------------------------------------------------
# 3. Cycle 1 cancel-and-replaces every 6% startup stop with the 5% trail
# ---------------------------------------------------------------------------

def test_cycle1_churns_startup_stop_to_5pct_trail_then_settles(harness):
    loop, broker, rec = harness([1.0, 1.0, 1.0], FLAT_PREDS)
    marks = drive(loop, broker, rec, 2)
    c1 = broker.events[marks[0]:marks[1]]
    c2 = broker.events[marks[1]:marks[2]]
    # desired 0.95*hwm vs resting 0.94*hwm: 0.95/0.94 = 1.0106 >= the
    # RESTING_STOP_MIN_IMPROVE 1.01 churn floor -> cancel + re-place, once.
    assert 0.95 / 0.94 >= crypto_loop.CryptoLoop.RESTING_STOP_MIN_IMPROVE
    assert len([e for e in c1 if e['kind'] == 'cancel']) == 6
    st = _stops(c1)
    assert set(st) == set(UNIVERSE)
    for s, e in st.items():
        assert e['stop_price'] == _rp(CP[s] * (1 - P['trail_fallback_pct']))
        assert loop._resting_stop_px[s] == pytest.approx(CP[s] * 0.95)
    # Flat tape: no further churn, no exits, exactly one live stop per name.
    assert c2 == []
    assert all(len(broker.open_orders(s)) == 1 for s in UNIVERSE)
    assert rec.trades == [] and not rec.rows('sell')


# ---------------------------------------------------------------------------
# 4. Book stop-risk reads 0 for $122k of inherited exposure
# ---------------------------------------------------------------------------

def test_book_stop_risks_zero_basis_reads_zero_and_enb_budget_full(harness):
    from portfolio import book_risk_budget
    loop, broker, rec = harness([1.0, 1.0], FLAT_PREDS)
    drive(loop, broker, rec, 1)
    # CURRENT-BEHAVIOUR (owner item): max(0, entry 0 - stop) = 0 per name,
    # so the GATE-1 registry (and the ENB cap) sees a risk-free book.
    assert rec.risk == [[0.0] * 6]
    assert book_risk_budget(loop._book_stop_risks(), 0.5,
                            MAX_BOOK_RISK_PCT) == pytest.approx(MAX_BOOK_RISK_PCT)
    # Counterfactual: the same book anchored at the mark (5% trail risk per
    # name) exhausts the 2.5% MAX_BOOK_RISK_PCT budget -> 0 (entry blocked).
    eq = loop._equity
    risks = [QTY[s] * CP[s] * P['trail_fallback_pct'] / eq for s in UNIVERSE]
    assert sum(risks) == pytest.approx(0.05, abs=0.001)
    assert book_risk_budget(risks, 0.5, MAX_BOOK_RISK_PCT) == 0.0


# ---------------------------------------------------------------------------
# 5. Bullish pred on a HELD zero-basis name: the per-symbol cap must bind
# ---------------------------------------------------------------------------

def test_addon_on_zero_basis_position_blocked_by_symbol_cap(harness):
    loop, broker, rec = harness([1.0, 1.0, 1.0], BULL_BTC)
    drive(loop, broker, rec, 2)
    # MAX_NOTIONAL_PER_SYMBOL ($3000) caps a name's exposure INCLUDING an
    # add-on; BTC already holds 0.2493 x $84,365 = $21,031. With entry=0 the
    # cap valued it at qty*0 = $0 and admitted a buy every cycle.
    assert QTY['BTC/USD'] * CP['BTC/USD'] > loop.MAX_NOTIONAL_PER_SYMBOL
    buys = [e for e in broker.events if e.get('side') == 'buy']
    assert buys == []                                  # no buy ever submitted
    # Every universe name is a held zero-basis position, so every one now
    # stops at the position_cap gate before becoming a candidate: no
    # entry_window row (n_candidates == 0), no submit, no rejection spam.
    assert rec.rows('entry_window') == []
    assert not broker.events_of('reject')
    assert 'BTC/USD' not in loop.last_trade_time       # nothing traded
    # Sizing cap (defence in depth): _compute_position_size's room is also
    # valued at the mark for a zero-basis position -> 0 room -> size 0.
    q = order_utils.get_crypto_quote(broker, 'BTC/USD')
    assert loop._compute_position_size('BTC/USD', 2.0, q) == 0


def test_symbol_cap_unchanged_for_real_entry_price(harness):
    """entry_price > 0: the cap still values the position at COST (qty *
    entry) exactly as before — the fix only changes the entry<=0 case."""
    loop, broker, rec = harness([1.0], FLAT_PREDS)
    loop.positions = {'BTC/USD': Position(qty=0.01, entry_price=50_000.0,
                                          high_water_mark=84_364.6)}
    loop._equity = 121930.35
    loop._peak_equity = 121930.35
    q = order_utils.get_crypto_quote(broker, 'BTC/USD')
    # cost basis $500 -> room $2,500 (mark $844 would leave $2,156).
    size = loop._compute_position_size('BTC/USD', 2.0, q)
    assert 0 < size <= loop.MAX_NOTIONAL_PER_SYMBOL - 500


# ---------------------------------------------------------------------------
# 6. A cash-starved NEW entry: rejected every cycle, nothing stamped
# ---------------------------------------------------------------------------

def test_cash_starved_new_entry_rejected_every_cycle_no_cooldown(harness):
    """CURRENT-BEHAVIOUR (owner item): with $93.63 of cash any sized entry
    (>= MIN_ORDER_NOTIONAL $100) is refused by the broker; the maker ladder
    tries the bid-join rung then the fallback limit, both rejected, returns
    'unfilled' — no buy row, no cooldown, no trade-budget count, so the
    same two rejected submits repeat every 30 s cycle while the pred holds."""
    loop, broker, rec = harness(
        [1.0, 1.0, 1.0], dict(FLAT_PREDS, **{'AVAX/USD': 2.0}),
        extra_tape={'AVAX/USD': [20.0, 20.0, 20.0]})
    CP['AVAX/USD'] = 20.0
    try:
        loop.get_symbol_universe = lambda: UNIVERSE + ['AVAX/USD']
        drive(loop, broker, rec, 2)
    finally:
        CP.pop('AVAX/USD', None)
    rej = broker.events_of('reject', 'AVAX/USD')
    assert len(rej) == 4                           # 2 per cycle x 2 cycles
    assert all('insufficient balance for USD' in e['reason'] for e in rej)
    assert not broker.events_of('fill', 'AVAX/USD')
    assert 'AVAX/USD' not in loop.last_trade_time  # no cooldown stamped
    assert loop._daily_trades.get('AVAX/USD', 0) == 0
    assert not [r for r in rec.rows('buy')]
    fills = [r for r in rec.rows('entry_fills') if r['symbol'] == 'AVAX/USD']
    assert [r['tactic'] for r in fills] == ['unfilled', 'unfilled']
    # entry_window still lists the rejected name as 'admitted' (it cleared
    # every gate; the journal has no field for a broker refusal).
    assert [w['admitted'] for w in rec.rows('entry_window')] == \
        [['AVAX/USD'], ['AVAX/USD']]
    # The six inherited stops are untouched by the failed entries.
    assert all(len(broker.open_orders(s)) == 1 for s in UNIVERSE)


# ---------------------------------------------------------------------------
# 7. -7% gap: resting stops fill server-side, then the breaker trips
# ---------------------------------------------------------------------------

def test_minus7_gap_resting_stops_fill_then_breaker_flattens_nothing(harness):
    loop, broker, rec = harness([1.0, 0.93, 0.93], FLAT_PREDS)
    marks = drive(loop, broker, rec, 2)
    # The gap tick (before cycle 1) crosses the STARTUP 6% stop (0.94) and
    # stays above its 2%-lower limit (0.9212): all six fill at the bid.
    fills = broker.events_of('fill')
    assert len(fills) == 6 and all(e['type'] == 'stop_limit' for e in fills)
    assert broker.positions == {}
    # Breaker: (last_equity - equity)/last_equity >= CIRCUIT_BREAKER_PCT 5%
    # -> trip (account-wide daily baseline 123,095.98).
    eq = float(broker.get_account().equity)
    assert (123095.98 - eq) / 123095.98 > loop.CIRCUIT_BREAKER_PCT
    trip = rec.rows('circuit_breaker_trip')
    assert len(trip) == 1 and trip[0]['n_positions_at_trip'] == 6
    assert loop._halted_until is not None and loop._buys_allowed is False
    assert loop.positions == {}
    # CURRENT-BEHAVIOUR (owner item): the breaker runs BEFORE _manage_stops,
    # finds no broker position, reports no failure, and journals all six
    # exits as estimated 'circuit_breaker' trades at the quote mid; the six
    # REAL server-stop fills get no sell row and no lockout.
    reasons = [k.get('exit_reason') for a, k in rec.trades]
    assert reasons == ['circuit_breaker'] * 6
    assert all(k.get('estimated') is True for a, k in rec.trades)
    assert not rec.rows('sell')
    # No flatten order was needed and nothing was re-bought.
    assert not [e for e in broker.events
                if e['kind'] == 'submit' and e['type'] == 'market']
    assert not [e for e in broker.events if e.get('side') == 'buy']


# ---------------------------------------------------------------------------
# 8. +3% then -6% from the peak: trail ratchets, server stops fill
# ---------------------------------------------------------------------------

def test_plus3_then_minus6_trail_ratchets_and_server_stops_exit(harness):
    peak, low = 1.03, 1.03 * 0.94
    loop, broker, rec = harness([1.0, peak, low], FLAT_PREDS)
    marks = drive(loop, broker, rec, 2)
    c1 = broker.events[marks[0]:marks[1]]
    # Cycle 1 at +3%: HWM ratchets; desired 0.95*1.03 vs resting 0.94 ->
    # cancel + re-place at the 5% trail of the NEW high.
    st = _stops(c1)
    for s in UNIVERSE:
        assert st[s]['stop_price'] == _rp(CP[s] * peak * 0.95)
        assert loop.positions.get(s) is None or \
            loop.positions[s].high_water_mark == pytest.approx(CP[s] * peak)
    # Tick to -6% from the peak (bid 0.9677 <= trigger 0.9785, >= limit
    # 0.9589): every resting stop fills server-side.
    fills = broker.events_of('fill')
    assert len(fills) == 6 and all(e['type'] == 'stop_limit' for e in fills)
    # Breaker: realised drawdown ~4.1% < 5% -> NOT tripped.
    eq = float(broker.get_account().equity)
    assert (123095.98 - eq) / 123095.98 < loop.CIRCUIT_BREAKER_PCT
    assert not rec.rows('circuit_breaker_trip')
    # Cycle 2 _manage_stops sees the fills: one server_stop sell row each,
    # classified 'trail' (stop px >= entry 0), pnl 0.0 (entry unknown), and
    # the 24h lockout applied (STOP_CLASSIFY_V2 OFF => unconditional).
    sells = rec.rows('sell')
    assert len(sells) == 6
    assert all(r['exit_reason'] == 'server_stop'
               and r['server_stop_kind'] == 'trail'
               and r['pnl_pct'] == 0.0 and r['estimated'] is False
               for r in sells)
    assert loop.positions == {}
    assert set(loop.hard_stop_lockout) == set(UNIVERSE)
    assert not [e for e in broker.events if e.get('side') == 'buy']


# ---------------------------------------------------------------------------
# 9. A NEW fill reported with avg_entry_price=0 (paper zero-basis quirk)
# ---------------------------------------------------------------------------

def test_new_fill_with_zero_broker_basis_uses_order_fill_price(harness):
    """Policy: a fresh entry is protected by a resting stop at
    entry*(1-stop_dist) (crypto_loop._after_entry_protection) and journaled
    at its real fill price. When the broker's position reports
    avg_entry_price=0 for a filled buy, the order's filled_avg_price is the
    only honest basis; without it the stop was computed as 0*(1-d) = $0
    (rejected by the broker) and the buy row journaled fill_price 0."""
    loop, broker, rec = harness(
        [1.0, 1.0], dict(FLAT_PREDS, **{'AVAX/USD': 2.0}),
        extra_tape={'AVAX/USD': [20.0, 20.0]})
    broker.cash = 10_000.0
    broker.report_zero_basis = True
    CP['AVAX/USD'] = 20.0
    try:
        loop.get_symbol_universe = lambda: UNIVERSE + ['AVAX/USD']
        drive(loop, broker, rec, 1)
    finally:
        CP.pop('AVAX/USD', None)
    buy_fills = [e for e in broker.events_of('fill', 'AVAX/USD')
                 if e['side'] == 'buy']
    assert len(buy_fills) == 1
    fill_px = buy_fills[0]['fill_price']
    assert broker.positions['AVAXUSD']['avg_entry_price'] == 0.0  # the quirk
    pos = loop.positions['AVAX/USD']
    assert pos.entry_price == pytest.approx(fill_px)
    # stop_dist = clamp(entry_ATR * atr_stop_mult 2.5 / entry) ~ 2.5%
    # (stub ATR = 1% of the $20 tape price); TP = 2:1 RR.
    d = min(P['stop_ceil_pct'], max(P['stop_floor_pct'],
                                    20.0 * ATR_FRAC * P['atr_stop_mult'] / fill_px))
    stops = [e for e in broker.events_of('submit', 'AVAX/USD')
             if e['type'] == 'stop_limit']
    assert len(stops) == 1 and not broker.events_of('reject', 'AVAX/USD')
    assert stops[0]['stop_price'] == _rp(fill_px * (1 - d))
    assert pos.take_profit_price == pytest.approx(
        fill_px * (1 + min(P['tp_ceil_pct'], d * P['tp_rr'])))
    buy = rec.rows('buy')
    assert len(buy) == 1 and buy[0]['fill_price'] == pytest.approx(fill_px)
    assert abs(buy[0]['slippage_bps']) < 100      # not the -10000 bps of $0


def test_reconstruct_warns_on_unknown_basis(harness, caplog):
    """Startup must say LOUDLY that inherited positions have no cost basis
    (log-only; the stop/TP/risk consequences are owner items)."""
    import logging
    loop, broker, rec = harness([1.0, 1.0], FLAT_PREDS)
    with caplog.at_level(logging.WARNING, logger='base_loop'):
        drive(loop, broker, rec, 1)
    msgs = [r.getMessage() for r in caplog.records
            if 'cost basis unknown' in r.getMessage()]
    assert len(msgs) == 6
    assert any('BTC/USD' in m for m in msgs)
