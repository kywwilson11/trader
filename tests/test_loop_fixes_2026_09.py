"""2026-09 Jetson campaign — live-loop fixes (hunt G1 F1/F2/F4, G2-3, G2-4).

Mac-safe: every loop method is pulled out of the REAL source by AST and
exec'd against stub globals (extract-and-exec), so no torch/joblib/dotenv
import is needed. run_bots is imported directly (it is lightweight: the
loop modules are imported lazily inside main()).

  F1   stock_loop._prepare_overnight_keepers iterated the set it discards
       from -> RuntimeError aborted the EOD flatten.
  F2   base_loop._load_models let a corrupt artifact escape run(); now fails
       CLOSED and hands retry to _hot_reload_check's backoff.
  F4   run_bots.main returned 0 after a loop crashed.
  G2-4 stock_loop's inline "position gone" check missed the legacy SDK's
       'position does not exist' text; now uses order_utils._is_not_found.
  G2-3 hard-stop lockout / external-close 24h window used naive local
       wall-clock subtraction (off by 1 h across DST).
"""

import ast
import datetime as _dt
import json
import logging
import os
import signal
import sys
import textwrap
import threading
import time as _time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from order_utils import _is_not_found  # noqa: E402  (Mac-importable)

_LOG = logging.getLogger('test_loop_fixes_2026_09')


# ---------------------------------------------------------------------------
# extract-and-exec helpers (as in the G1 hunt scaffolding)
# ---------------------------------------------------------------------------

def _extract(fname, cls, meth, glb):
    path = REPO / fname
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == meth:
                    code = textwrap.dedent(ast.get_source_segment(src, f))
                    ns = dict(glb)
                    exec(code, ns)
                    return ns[meth]
    raise KeyError((cls, meth))


def _chicago_naive(y, mo, d, h, mi):
    """Naive LOCAL datetime (with correct .fold) for a Chicago wall time,
    built the way the loops build theirs (fromtimestamp)."""
    from zoneinfo import ZoneInfo
    aware = _dt.datetime(y, mo, d, h, mi, tzinfo=ZoneInfo('America/Chicago'))
    return _dt.datetime.fromtimestamp(aware.timestamp())


@pytest.fixture(autouse=True)
def _no_real_notify(monkeypatch):
    """Failure paths in the extracted methods do `from notify import
    notify`; keep every test hermetic (no webhook/Telegram egress)."""
    stub = types.ModuleType('notify')
    stub.notify = lambda *a, **k: None
    stub.ping_heartbeat = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'notify', stub)


def _local_zone_is_chicago():
    """True when naive local conversions in THIS process follow
    America/Chicago (CDT in July, CST in January)."""
    summer = _dt.datetime(2026, 7, 1, 17, 0, tzinfo=_dt.timezone.utc).timestamp()
    winter = _dt.datetime(2026, 1, 15, 18, 0, tzinfo=_dt.timezone.utc).timestamp()
    return (_dt.datetime.fromtimestamp(summer).hour == 12
            and _dt.datetime.fromtimestamp(winter).hour == 12)


_SUBPROC_MARK = 'TRADER_TEST_LOOP_FIXES_DST_CHILD'


@pytest.fixture
def chicago_tz():
    """Run the test with the process local zone = America/Chicago (the
    Jetson's zone). Uses time.tzset when the interpreter has it (CPython on
    macOS/most Linux); the Jetson's conda python lacks tzset, so there the
    G2-3 cases run in a child interpreter started with TZ set — see
    test_dst_cases_run_in_chicago_child."""
    if _local_zone_is_chicago():
        yield
        return
    if not hasattr(_time, 'tzset'):
        pytest.skip('no time.tzset: covered by test_dst_cases_run_in_chicago_child')
    old = os.environ.get('TZ')
    os.environ['TZ'] = 'America/Chicago'
    _time.tzset()
    try:
        if not _local_zone_is_chicago():
            pytest.skip('America/Chicago zone data unavailable')
        yield
    finally:
        if old is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = old
        _time.tzset()


def test_dst_cases_run_in_chicago_child():
    """When this interpreter cannot switch its local zone in-process, run
    the g23 DST cases in a child interpreter with TZ=America/Chicago."""
    if os.environ.get(_SUBPROC_MARK):
        pytest.skip('already the child')
    if _local_zone_is_chicago() or hasattr(_time, 'tzset'):
        pytest.skip('g23 cases run in-process')
    import subprocess
    env = dict(os.environ, TZ='America/Chicago', PY_COLORS='0')
    env[_SUBPROC_MARK] = '1'
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', str(Path(__file__).resolve()),
         '-k', 'g23', '-q', '-p', 'no:cacheprovider', '-rs'],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    tail = (r.stdout + r.stderr)[-3000:]
    assert r.returncode == 0, tail
    assert 'skipped' not in tail and ' passed' in tail, tail


def _fake_datetime_module(now_value):
    class _FakeDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now_value if tz is None else now_value.astimezone(tz)
    return SimpleNamespace(datetime=_FakeDT, timedelta=_dt.timedelta,
                           timezone=_dt.timezone)


# ---------------------------------------------------------------------------
# F1 — overnight keepers: set mutated during iteration
# ---------------------------------------------------------------------------

class _Pos:
    def __init__(self):
        self.entry_atr = None
        self.entry_price = 100.0
        self.qty = 10
        self.stop_order_id = None
        self.trailing_activated = True


class _KeeperApi:
    """GTC stop submission fails for symbols in ``fail_stop``; every other
    order (the flatten sells) succeeds."""

    def __init__(self, fail_stop):
        self.fail_stop = set(fail_stop)
        self.stops = []
        self.sells = []

    def submit_order(self, **kw):
        if kw.get('type') == 'stop':
            if kw['symbol'] in self.fail_stop:
                raise Exception('insufficient qty available for order')
            self.stops.append(kw['symbol'])
            return SimpleNamespace(id='stop-' + kw['symbol'])
        self.sells.append(kw['symbol'])
        return SimpleNamespace(id='sell-' + kw['symbol'])

    def get_position(self, sym):
        return SimpleNamespace(qty='10')


def _keeper_glb(sold):
    return {
        'logger': _LOG,
        'cancel_orders_for_symbol': lambda api, s, timeout=8: True,
        'make_client_order_id': lambda p: p + '-x',
        '_is_not_found': _is_not_found,
        'get_all_positions': lambda api: None,
        'get_stock_quote': lambda api, s: None,
        'place_stock_limit_order': lambda *a, **k: None,
        'time': SimpleNamespace(sleep=lambda s: None),
        'manage_order_lifecycle': lambda api, oid, **k: (
            sold.append(oid),
            SimpleNamespace(status='filled', filled_avg_price='99'))[1],
    }


def _keeper_self(api, symbols):
    return SimpleNamespace(
        positions={k: _Pos() for k in symbols}, api=api,
        ATR_STOP_MULTIPLIER=2.0, ATR_STOP_FLOOR_PCT=0.02,
        ATR_STOP_CEIL_PCT=0.08, STOP_LOSS_PCT=0.03)


@pytest.mark.parametrize('keepers,fail', [
    (['AAA'], {'AAA'}),
    (['AAA', 'BBB'], {'AAA', 'BBB'}),
    (['AAA', 'BBB'], {'AAA'}),
])
def test_f1_failed_keeper_is_discarded_without_runtimeerror(keepers, fail):
    prep = _extract('stock_loop.py', 'StockLoop',
                    '_prepare_overnight_keepers', _keeper_glb([]))
    api = _KeeperApi(fail)
    s = _keeper_self(api, keepers)
    ks = set(keepers)
    prep(s, ks)                                  # must not raise
    assert ks == set(keepers) - fail             # failed ones routed out
    assert sorted(api.stops) == sorted(set(keepers) - fail)


def test_f1_success_path_unchanged():
    prep = _extract('stock_loop.py', 'StockLoop',
                    '_prepare_overnight_keepers', _keeper_glb([]))
    api = _KeeperApi(set())
    s = _keeper_self(api, ['AAA', 'BBB'])
    ks = {'AAA', 'BBB'}
    prep(s, ks)
    assert ks == {'AAA', 'BBB'}
    for sym in ('AAA', 'BBB'):
        assert s.positions[sym].stop_order_id == 'stop-' + sym
        assert s.positions[sym].trailing_activated is False


def test_f1_flatten_sells_non_keeper_and_routes_failed_keeper():
    sold = []
    glb = _keeper_glb(sold)
    prep = _extract('stock_loop.py', 'StockLoop',
                    '_prepare_overnight_keepers', glb)
    flat = _extract('stock_loop.py', 'StockLoop', 'flatten_before_close', glb)

    s = SimpleNamespace()
    s.flattened_today = False
    s.positions = {'AAA': _Pos(), 'CCC': _Pos()}
    s.api = _KeeperApi({'AAA'})                 # keeper AAA's GTC stop fails
    s.llm_scores = {}
    s.ATR_STOP_MULTIPLIER = 2.0
    s.ATR_STOP_FLOOR_PCT = 0.02
    s.ATR_STOP_CEIL_PCT = 0.08
    s.STOP_LOSS_PCT = 0.03
    s.ORDER_TIMEOUT = 1
    s._in_flatten_window = lambda: True
    s._select_overnight_keepers = lambda: {'AAA'}
    s._prepare_overnight_keepers = lambda k: prep(s, k)
    s.get_symbol_universe = lambda: ['AAA', 'CCC']
    exits = []
    s._record_confirmed_exit = lambda sym, *a, **k: exits.append(sym)
    s._journal_external_close = lambda *a, **k: None

    flat(s)                                      # no RuntimeError
    assert 'CCC' in s.api.sells                  # non-keeper still sold
    assert 'AAA' in s.api.sells                  # failed keeper flattened
    assert s.positions == {}
    assert s.flattened_today is True
    assert sorted(exits) == ['AAA', 'CCC']


# ---------------------------------------------------------------------------
# F2 — _load_models fails closed; hot-reload backoff retries
# ---------------------------------------------------------------------------

def _f2_setup(monkeypatch, state, calls, clock):
    def load_models(dev, prefix=''):
        calls.append(clock['t'])
        if state['broken']:
            raise RuntimeError('Error(s) in loading state_dict: size mismatch '
                               'for lstm.weight_ih_l0')
        return 'MODEL', {'trade_threshold': 0.2}, 'SCALER', ['f']

    tu = types.ModuleType('trading_utils')
    tu.model_reload_key = lambda p: 111.0
    monkeypatch.setitem(sys.modules, 'trading_utils', tu)
    pn = types.ModuleType('predict_now')
    pn._lgb_models = {}
    pn._q10_models = {}
    monkeypatch.setitem(sys.modules, 'predict_now', pn)
    os_stub = types.ModuleType('order_stream')
    os_stub.start_order_stream = lambda: None
    monkeypatch.setitem(sys.modules, 'order_stream', os_stub)

    fake_time = SimpleNamespace(time=lambda: clock['t'],
                                sleep=lambda s: None)
    cancels = []
    glb = {'logger': _LOG, 'choose_inference_device': lambda: 'cpu',
           'load_models': load_models, 'time': fake_time,
           'cancel_all_open_orders':
               lambda api, symbols=None: cancels.append(tuple(symbols or ())),
           'datetime': _dt}
    return glb, cancels


def _f2_self(**kw):
    s = SimpleNamespace(MODEL_PREFIX='stock', model=None, config={},
                        scaler_X=None, feature_cols=None, trade_threshold=0.15,
                        _failed_reload=(None, 0.0), model_mtime=0)
    s.__dict__.update(kw)
    return s


def test_f2_corrupt_artifact_fails_closed_then_backoff_reloads(monkeypatch, caplog):
    state = {'broken': True}
    calls = []
    clock = {'t': 1000.0}
    glb, _ = _f2_setup(monkeypatch, state, calls, clock)
    load = _extract('base_loop.py', 'BaseTradingLoop', '_load_models', glb)
    hot = _extract('base_loop.py', 'BaseTradingLoop', '_hot_reload_check', glb)

    s = _f2_self()
    with caplog.at_level(logging.ERROR, logger=_LOG.name):
        load(s)                                  # must not raise
    assert s.model is None and s.scaler_X is None and s.feature_cols is None
    assert s.config == {}
    assert s._failed_reload == (111.0, 1000.0)
    assert s.model_mtime is None
    assert any('size mismatch' in r.getMessage() for r in caplog.records)
    assert len(calls) == 1

    clock['t'] += 30                             # inside the 300 s backoff
    hot(s)
    assert len(calls) == 1 and s.model is None

    clock['t'] += 300                            # backoff elapsed, repaired
    state['broken'] = False
    hot(s)
    assert len(calls) == 2
    assert s.model == 'MODEL'
    assert s.trade_threshold == 0.2
    assert s._failed_reload == (None, 0.0)       # cleared on success
    assert s.model_mtime == 111.0


def test_f2_file_not_found_path_unchanged(monkeypatch):
    clock = {'t': 1000.0}
    glb, _ = _f2_setup(monkeypatch, {'broken': False}, [], clock)

    def fnf(dev, prefix=''):
        raise FileNotFoundError('model_v2.pth')
    glb['load_models'] = fnf
    load = _extract('base_loop.py', 'BaseTradingLoop', '_load_models', glb)
    s = _f2_self()
    load(s)
    assert s.model is None
    assert s._failed_reload == (None, 0.0)
    assert s.model_mtime == 111.0


def test_f2_no_exception_escapes_run_startup(monkeypatch):
    """The REAL run() skeleton: a corrupt artifact no longer escapes; the
    exits/stops startup (scoped cancel, reconstruct) still runs."""
    clock = {'t': 1000.0}
    glb, cancels = _f2_setup(monkeypatch, {'broken': True}, [], clock)
    load = _extract('base_loop.py', 'BaseTradingLoop', '_load_models', glb)
    run = _extract('base_loop.py', 'BaseTradingLoop', 'run', glb)

    steps = []
    s = _f2_self()
    s.api = object()
    s._load_models = lambda: load(s)
    s.get_symbol_universe = lambda: ['AAA']
    s._reconstruct_positions = lambda: steps.append('reconstruct')
    s._print_startup = lambda: steps.append('startup')

    def one_cycle():
        steps.append('cycle')
        raise KeyboardInterrupt                  # exit the while-True
    s._run_one_cycle = one_cycle

    with pytest.raises(KeyboardInterrupt):
        run(s)
    assert steps == ['reconstruct', 'startup', 'cycle']
    assert cancels == [('AAA',)]
    assert s.model is None and s._failed_reload[0] == 111.0


# ---------------------------------------------------------------------------
# F4 — run_bots exit status
# ---------------------------------------------------------------------------

def _run_bots_main(monkeypatch, run_impl):
    import run_bots as rb

    class _FakeLoop:
        def run(self):
            run_impl()

    fake = types.ModuleType('crypto_loop')
    fake.CryptoLoop = _FakeLoop
    monkeypatch.setitem(sys.modules, 'crypto_loop', fake)
    monkeypatch.setattr(rb, '_OPS_ENABLED', False)
    monkeypatch.setattr(sys, 'argv', ['run_bots.py', '--crypto-only'])

    def _sleep(_s):
        # stagger / poll sleeps: wait for the loop thread to finish instead
        # of burning real seconds
        rb._shutdown.wait(2.0)
        _time.sleep(0.05)
    monkeypatch.setattr(rb, 'time', SimpleNamespace(sleep=_sleep))

    msgs = []

    class _Rec:
        def info(self, fmt, *a, **k):
            msgs.append(fmt % a if a else fmt)
        exception = error = warning = info
    monkeypatch.setattr(rb, 'logger', _Rec())

    rb._shutdown.clear()
    rb._crashed.clear()
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        rc = rb.main()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        rb._shutdown.clear()
        rb._crashed.clear()
    return rc, msgs


def test_f4_crashed_loop_returns_nonzero(monkeypatch):
    def boom():
        raise RuntimeError('size mismatch')
    rc, msgs = _run_bots_main(monkeypatch, boom)
    assert rc == 1
    running = [m for m in msgs if 'loop(s) running' in m]
    assert running and running[0].startswith('[BOTS] 0 loop(s)')


def test_f4_clean_exit_returns_zero(monkeypatch):
    rc, _ = _run_bots_main(monkeypatch, lambda: None)
    assert rc == 0


# ---------------------------------------------------------------------------
# G2-4 — shared not-found classifier at both stock_loop sites
# ---------------------------------------------------------------------------

class _NotFoundApi:
    """Legacy SDK behaviour, measured live: an unheld position raises
    APIError whose text is 'position does not exist' (no '404' in it)."""

    def get_position(self, sym):
        raise Exception('position does not exist')


def test_g24_legacy_text_is_not_found():
    assert _is_not_found(Exception('position does not exist'))


def test_g24_execute_sells_treats_legacy_text_as_gone(monkeypatch):
    tu = types.ModuleType('trading_utils')
    tu.cooldown_ok = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, 'trading_utils', tu)
    sells = _extract('stock_loop.py', 'StockLoop', '_execute_sells',
                     {'logger': _LOG, '_is_not_found': _is_not_found})
    journaled = []
    s = SimpleNamespace(positions={'AAPL': _Pos()}, api=_NotFoundApi())
    s._journal_external_close = lambda sym, info: journaled.append(sym)
    sells(s, {})
    assert journaled == ['AAPL']
    assert s.positions == {}


def test_g24_execute_sells_transient_error_keeps_tracking(monkeypatch):
    tu = types.ModuleType('trading_utils')
    tu.cooldown_ok = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, 'trading_utils', tu)
    sells = _extract('stock_loop.py', 'StockLoop', '_execute_sells',
                     {'logger': _LOG, '_is_not_found': _is_not_found})

    class _Api:
        def get_position(self, sym):
            raise Exception('429 rate limit exceeded')
    journaled = []
    s = SimpleNamespace(positions={'AAPL': _Pos()}, api=_Api())
    s._journal_external_close = lambda sym, info: journaled.append(sym)
    sells(s, {})
    assert journaled == [] and 'AAPL' in s.positions


def test_g24_flatten_treats_legacy_text_as_gone():
    sold = []
    glb = _keeper_glb(sold)
    flat = _extract('stock_loop.py', 'StockLoop', 'flatten_before_close', glb)
    journaled = []
    s = SimpleNamespace(flattened_today=False, positions={'AAPL': _Pos()},
                        api=_NotFoundApi(), llm_scores={}, ORDER_TIMEOUT=1)
    s._in_flatten_window = lambda: True
    s._select_overnight_keepers = lambda: set()
    s.get_symbol_universe = lambda: ['AAPL']
    s._journal_external_close = lambda sym, info: journaled.append(sym)
    s._record_confirmed_exit = lambda *a, **k: None
    flat(s)
    assert journaled == ['AAPL']
    assert s.positions == {}
    assert s.flattened_today is True            # not "will retry" forever
    assert sold == []


def test_g24_inline_copies_removed():
    src = (REPO / 'stock_loop.py').read_text()
    assert "'no position' in err_str" not in src


# ---------------------------------------------------------------------------
# G2-3 — DST-safe elapsed time (hard-stop lockout, external-close window)
# ---------------------------------------------------------------------------

def _locked_fn(now_value):
    return _extract('base_loop.py', 'BaseTradingLoop', '_is_hard_stop_locked',
                    {'datetime': _fake_datetime_module(now_value)})


def _lock_self(stamp, hours=24):
    saved = []
    s = SimpleNamespace(hard_stop_lockout={'BTC/USD': stamp},
                        HARD_STOP_LOCKOUT_HOURS=hours)
    s._save_hard_stop_lockout = lambda: saved.append(True)
    return s, saved


def test_g23_lockout_spring_forward_still_locked(chicago_tz):
    # 2027-03-14 02:00 CST -> 03:00 CDT. 23.5 h of real time elapse, but
    # the naive wall-clock difference is 24.5 h.
    stamp = _chicago_naive(2027, 3, 13, 12, 0)
    now = _dt.datetime.fromtimestamp(stamp.timestamp() + 23.5 * 3600)
    assert (now - stamp).total_seconds() == 24.5 * 3600   # the old bug
    s, saved = _lock_self(stamp)
    assert _locked_fn(now)(s, 'BTC/USD') is True
    assert 'BTC/USD' in s.hard_stop_lockout and not saved


def test_g23_lockout_fall_back_expires_on_real_time(chicago_tz):
    # 2026-11-01 02:00 CDT -> 01:00 CST. 24.5 h real, naive 23.5 h.
    stamp = _chicago_naive(2026, 10, 31, 12, 0)
    now = _dt.datetime.fromtimestamp(stamp.timestamp() + 24.5 * 3600)
    assert (now - stamp).total_seconds() == 23.5 * 3600   # the old bug
    s, saved = _lock_self(stamp)
    assert _locked_fn(now)(s, 'BTC/USD') is False
    assert 'BTC/USD' not in s.hard_stop_lockout and saved


def test_g23_lockout_repeated_hour_honours_fold(chicago_tz):
    # 01:30 CDT (fold=0) -> 01:10 CST (fold=1) is 40 real minutes.
    stamp = _chicago_naive(2026, 11, 1, 1, 30)
    now = _dt.datetime.fromtimestamp(stamp.timestamp() + 40 * 60)
    assert now.fold == 1 and (now.hour, now.minute) == (1, 10)
    s, _ = _lock_self(stamp, hours=0.5)
    assert _locked_fn(now)(s, 'BTC/USD') is False


def test_g23_lockout_duration_unchanged_off_dst(chicago_tz):
    stamp = _chicago_naive(2026, 7, 1, 12, 0)
    s, _ = _lock_self(stamp)
    just_before = _dt.datetime.fromtimestamp(stamp.timestamp() + 24 * 3600 - 1)
    assert _locked_fn(just_before)(s, 'BTC/USD') is True
    at = _dt.datetime.fromtimestamp(stamp.timestamp() + 24 * 3600)
    assert _locked_fn(at)(s, 'BTC/USD') is False


def test_g23_lockout_persisted_expiry_is_real_24h(chicago_tz, tmp_path):
    save = _extract('base_loop.py', 'BaseTradingLoop', '_save_hard_stop_lockout',
                    {'datetime': _dt, 'json': json, 'os': os, 'logger': _LOG})
    stamp = _chicago_naive(2027, 3, 13, 12, 0)             # spans spring-forward
    lf = str(tmp_path / 'hard_stop_lockout.json')
    s = SimpleNamespace(hard_stop_lockout={'BTC/USD': stamp},
                        HARD_STOP_LOCKOUT_HOURS=24, _lockout_file=lf,
                        MODEL_PREFIX='')
    save(s)
    data = json.loads(Path(lf).read_text())
    assert data['BTC/USD'] == pytest.approx(stamp.timestamp() + 24 * 3600)


def _recover_self(now_value):
    fn = _extract('stock_loop.py', 'StockLoop', '_recover_external_exit',
                  {'datetime': _fake_datetime_module(now_value)})
    recorded = []
    s = SimpleNamespace(llm_scores={}, last_trade_time={}, _tp_order_ids={})
    s._record_confirmed_exit = lambda sym, info, order, q, **k: recorded.append(
        (sym, k.get('exit_reason')))
    return fn, s, recorded


def test_g23_external_close_24h_window_is_real_time(chicago_tz):
    # Fill at 2027-03-13 10:00 CST; now = +23.5 h real (naive diff 24.5 h):
    # inside the 24 h window, so the confirmed row must be written.
    fill_utc = _dt.datetime(2027, 3, 13, 16, 0, tzinfo=_dt.timezone.utc)
    now = _dt.datetime.fromtimestamp(fill_utc.timestamp() + 23.5 * 3600)
    fn, s, recorded = _recover_self(now)

    order = SimpleNamespace(side='sell', status='filled',
                            filled_at=fill_utc.isoformat().replace('+00:00', 'Z'),
                            filled_avg_price='101.5')
    s.api = SimpleNamespace(list_orders=lambda **k: [order])
    info = SimpleNamespace(stop_order_id=None)
    assert fn(s, 'AAPL', info) is True
    assert recorded == [('AAPL', 'external_close')]


def test_g23_external_close_rejects_older_than_24h_real(chicago_tz):
    # Fall-back: 24.5 h real but naive 23.5 h -> must be REJECTED.
    fill_utc = _dt.datetime(2026, 10, 31, 17, 0, tzinfo=_dt.timezone.utc)
    now = _dt.datetime.fromtimestamp(fill_utc.timestamp() + 24.5 * 3600)
    fn, s, recorded = _recover_self(now)
    order = SimpleNamespace(side='sell', status='filled', filled_at=fill_utc,
                            filled_avg_price='101.5')
    s.api = SimpleNamespace(list_orders=lambda **k: [order])
    assert fn(s, 'AAPL', SimpleNamespace(stop_order_id=None)) is False
    assert recorded == []
