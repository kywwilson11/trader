"""ENGINE R5 W16: invariants the LIVE paper bots obeyed on 2026-09-27.

Stdlib-only (plus the import-free ``strategy_config`` and a source-text read
of ``crypto_loop.py``): parses the small, secrets-free evidence excerpt in
``tests/fixtures/live_bots_2026-09-27_excerpt.txt`` (bots_stdout.log of pid
166905 + a read-only broker order listing + position_state.json) and pins
the resting-stop contract the fake-broker harness
(tests/test_engine_r1_inherited_positions.py) predicted, as it actually
happened on Alpaca paper:

  I1  every [RESTING-STOP] limit == round(stop * (1 - RESTING_STOP_LIMIT_GAP))
      at the _round_px precision (6 dp below $1, else 4 dp);
  I2  startup stops sit at (1 - stop_fallback_pct) x HWM0 and the cycle-1
      re-place at (1 - trail_fallback_pct) x HWM1 with HWM1 >= HWM0, and the
      re-place clears RESTING_STOP_MIN_IMPROVE (the O2 churn trigger);
  I3  broker side: per symbol, the startup stop was CANCELED before its
      replacement was SUBMITTED, exactly one stop per symbol is live, qty is
      the full position qty, and every broker price equals the logged price;
  I4  position_state.json HWM >= every HWM implied by a logged stop.

Runs under pytest, or directly: ``python tests/test_engine_r5_live_invariants.py``.
"""
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, 'fixtures', 'live_bots_2026-09-27_excerpt.txt')
UNIVERSE = ('BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD', 'LINK/USD')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import strategy_config  # noqa: E402  (no imports of its own)

_RS = re.compile(r'^(\S+ \S+) \[crypto_loop\] INFO: \[RESTING-STOP\] (\S+): '
                 r'GTC stop_limit @ \$([0-9.]+) \(limit \$([0-9.]+)\)$')
_ORD = re.compile(r'^(\S+) (\S+) sell stop_limit gtc stop ([0-9.]+) lim ([0-9.]+) '
                  r'qty ([0-9.]+) (\w+) canceled_at (\S+) cstop-$')


def _class_const(name):
    src = open(os.path.join(ROOT, 'crypto_loop.py')).read()
    m = re.search(r'^\s+%s = ([0-9.]+)' % name, src, re.M)
    assert m, name
    return float(m.group(1))


GAP = _class_const('RESTING_STOP_LIMIT_GAP')
MIN_IMPROVE = _class_const('RESTING_STOP_MIN_IMPROVE')
STOP_FB = strategy_config.CRYPTO_POLICY['stop_fallback_pct']
TRAIL_FB = strategy_config.CRYPTO_POLICY['trail_fallback_pct']


def _sections():
    out, cur = {}, None
    for line in open(FIXTURE):
        line = line.rstrip('\n')
        if line.startswith('### SECTION '):
            cur = line.split()[-1]
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def _dp(p):
    return 6 if p < 1 else 4          # crypto_loop.CryptoLoop._round_px


def _ts(s):
    """Alpaca ISO-8601 'Z' with up to 9 fractional digits -> aware datetime."""
    s = s.rstrip('Z')
    if '.' in s:
        head, frac = s.split('.')
        s = head + '.' + (frac + '000000')[:6]
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)


def _resting_stops():
    rows = []
    for line in _sections()['stdout']:
        m = _RS.match(line)
        if m:
            rows.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
    return rows


def _by_symbol():
    per = {}
    for _, sym, stop, lim in _resting_stops():
        per.setdefault(sym, []).append((stop, lim))
    return per


def _orders():
    rows = []
    for line in _sections()['broker_orders']:
        m = _ORD.match(line)
        assert m, line
        rows.append({'submitted': _ts(m.group(1)), 'symbol': m.group(2),
                     'stop': float(m.group(3)), 'limit': float(m.group(4)),
                     'qty': float(m.group(5)), 'status': m.group(6),
                     'canceled': None if m.group(7) == 'None' else _ts(m.group(7))})
    return rows


def test_fixture_is_small_and_secret_free():
    assert os.path.getsize(FIXTURE) <= 60 * 1024
    text = open(FIXTURE).read()
    for pat in ('APCA', 'SECRET', 'API_KEY', 'Bearer', 'sk-'):
        assert pat not in text


def test_i1_limit_is_gap_under_stop_at_round_px_precision():
    rows = _resting_stops()
    assert len(rows) == 12                       # 6 startup + 6 cycle-1
    for _, sym, stop, lim in rows:
        # both logged values are _round_px'd from the same unrounded stop s:
        # |lim - (1-GAP)*round(s)| <= 0.5u + (1-GAP)*0.5u < 1u
        u = 10.0 ** -_dp(lim)
        assert abs(lim - stop * (1 - GAP)) < u + 1e-12, (sym, stop, lim)
        assert round(stop, _dp(stop)) == stop and round(lim, _dp(lim)) == lim


def test_i2_startup_fallback_then_cycle1_trail_hwm_monotone_and_churn():
    per = _by_symbol()
    assert set(per) == set(UNIVERSE)
    for sym, seq in per.items():
        assert len(seq) == 2, sym                # one startup, one cycle-1 re-place
        (s0, _), (s1, _) = seq
        u = 10.0 ** -_dp(s0)
        h0 = s0 / (1 - STOP_FB)                  # flag-OFF restart anchor (RESTART_STOP_ANCHOR_DESIRED False)
        h1 = s1 / (1 - TRAIL_FB)                 # zero-basis desired stop = pure trail from HWM
        assert h1 >= h0 - 2 * u / (1 - STOP_FB), (sym, h0, h1)
        # the re-place happened because desired >= resting * MIN_IMPROVE
        assert s1 >= s0 * MIN_IMPROVE - u, (sym, s0, s1)
    assert strategy_config.RESTART_STOP_ANCHOR_DESIRED is False


def test_i3_broker_cancel_precedes_replace_one_live_stop_prices_match_log():
    per = _by_symbol()
    orders = _orders()
    assert len(orders) == 12
    for sym in UNIVERSE:
        mine = sorted((o for o in orders if o['symbol'] == sym),
                      key=lambda o: o['submitted'])
        assert [o['status'] for o in mine] == ['canceled', 'new'], sym
        old, new = mine
        assert old['canceled'] is not None and old['canceled'] < new['submitted'], sym
        assert old['qty'] == new['qty'] > 0
        (s0, l0), (s1, l1) = per[sym]
        assert (old['stop'], old['limit']) == (s0, l0), sym
        assert (new['stop'], new['limit']) == (s1, l1), sym
    live = [o for o in orders if o['status'] == 'new']
    assert sorted(o['symbol'] for o in live) == sorted(UNIVERSE)


def test_i4_persisted_hwm_never_below_logged_stop_implied_hwm():
    state = json.loads('\n'.join(_sections()['position_state']))
    per = _by_symbol()
    for sym, seq in per.items():
        (s0, _), (s1, _) = seq
        u = 10.0 ** -_dp(s1)
        implied = max(s0 / (1 - STOP_FB), s1 / (1 - TRAIL_FB))
        assert state['hwm'][sym] >= implied - 2 * u / (1 - TRAIL_FB), sym
        # no HWM ratchet re-place was due: HWM still < 1% above the cycle-1 anchor
        assert state['hwm'][sym] * (1 - TRAIL_FB) < s1 * MIN_IMPROVE, sym


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('PASS', name)
    print('%d passed' % n)
