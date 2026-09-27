"""ENGINE R5 (W15) — crash-loop backoff + give-up latch for run_pipeline's bot
monitor (`_check_restart_bots`), behind TRADER_BOT_RESTART_BACKOFF (default OFF).

Evidence: research/campaign_2026-09_jetson/research_engine.md "R4 scout — E6".
April 2026: the stock bot crashed 1,778 times at startup with ONE signature and
was restarted on every 60 s monitor pass, forever.

Fixture schema — tests/fixtures/april_crashloop_2026.json (built from the real
stock_bot_output.log by streaming it through run_pipeline.crash_signature; the
timestamps equal the W14 census `stock_crashes.json` one for one):
  signatures          list[str]  distinct crash signatures (crash_signature output)
  crashes             list[[ts, sig_idx]]  every observed crash, oldest first;
                      ts = 'YYYY-MM-DD HH:MM:SS' local time of the last log line
                      before the traceback; sig_idx indexes `signatures`
  episode_ends        list[ts]   the weekly-retrain stops that ended each crash episode
  episode_alive_since list[ts]   the bot start preceding each episode's first crash
  replay_model        {min_lifetime_s: 21, detect_s: 30} — W14 hazard-replay model:
                      a bot alive since t crashes at the first observed crash
                      >= t + min_lifetime_s and is detected detect_s later
  expected            the scout's pre-registered acceptance numbers
  example_tail        one real April traceback, verbatim

Stdlib only (no base_loop / torch import): run_pipeline imports stdlib +
adaptive_config (~12 MB RSS).
"""
import datetime as dt
import bisect
import importlib
import inspect
import io
import json
import sys
import types
from pathlib import Path

import pytest

rp = importlib.import_module('run_pipeline')

_FX_PATH = Path(__file__).resolve().parent / 'fixtures' / 'april_crashloop_2026.json'


@pytest.fixture(scope='module')
def fx():
    return json.loads(_FX_PATH.read_text())


def _epoch(ts):
    return dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------

def test_crash_signature_real_april_tail(fx):
    sig = rp.crash_signature(fx['example_tail'])
    assert sig == fx['signatures'][0]
    assert sig.startswith('fundamentals.py:258 | ValueError')  # innermost, not stock_loop.py:574


@pytest.mark.parametrize('bad', ['', None, b'Traceback', 123, 'garbage\nno trace',
                                 'Traceback (most recent call last)\n',
                                 'Traceback (most recent call last)\nValueError: x'])
def test_crash_signature_garbage_is_unknown(bad):
    assert rp.crash_signature(bad) == 'unknown'


def test_crash_signature_last_traceback_wins_and_masks_digits():
    tail = ('Traceback (most recent call last)\n'
            '  File "/x/a.py", line 1, in f\n'
            'KeyError: 1\n'
            'noise\n'
            'Traceback (most recent call last)\n'
            '  File "/x/b.py", line 10, in g\n'
            '    h()\n'
            '  File "/x/c.py", line 20, in h\n'
            '    raise ConnectionError(\'port 8443\')\n'
            '    ^^^^^\n'
            'ConnectionError: port 8443 refused\n'
            '2026-04-17 08:32:16 [base_loop] INFO: Loading prediction models...\n')
    assert rp.crash_signature(tail) == 'c.py:20 | ConnectionError: port # refused'


def test_next_restart_delay_ladder():
    assert [rp.next_restart_delay(k) for k in range(7)] == [60, 120, 240, 480, 960, 960, 960]
    assert rp.next_restart_delay(0) == 60 and rp.next_restart_delay(-3) == 60
    assert rp.next_restart_delay(10 ** 6) == 960
    assert rp.next_restart_delay(2, base=10, cap=30) == 30


def test_should_give_up_counts_same_signature_inside_window():
    h = [(t, 'A') for t in (0, 100, 200, 300)]
    assert not rp.should_give_up(h, 300)
    assert rp.should_give_up(h + [(400, 'A')], 400)
    assert not rp.should_give_up(h + [(400, 'B')], 400)          # latest sig decides
    assert not rp.should_give_up([(0, 'A')] + h[1:] + [(3700, 'A')], 3700)  # t=0 out of window
    assert rp.should_give_up([(0, 'A'), (1, 'B'), (2, 'A'), (3, 'B'), (4, 'A'),
                              (5, 'B'), (6, 'A'), (7, 'A')], 7)  # 5 A's, not consecutive
    assert not rp.should_give_up([], 0)


def test_consecutive_same_is_windowed():
    h = [(0, 'A'), (100, 'A'), (200, 'B'), (300, 'A'), (400, 'A')]
    assert rp.consecutive_same(h, 400) == 1
    assert rp.consecutive_same([(0, 'A'), (5000, 'A')], 5000) == 0  # quiet hour resets
    assert rp.consecutive_same([], 0) == 0


def test_distinct_signatures_always_restart_at_base():
    """Mixed-signature case: no crash repeats its predecessor -> every delay is
    base (60 s = the next monitor pass, today's behaviour)."""
    hist, delays = [], []
    for k in range(200):
        t = k * 45.0
        hist.append((t, ('A', 'B', 'C')[k % 3]))
        delays.append(rp.next_restart_delay(rp.consecutive_same(hist, t)))
    assert set(delays) == {60}


# ---------------------------------------------------------------------------
# April replay (W14 hazard model) — the pre-registered acceptance numbers
# ---------------------------------------------------------------------------

def _replay(fx, backoff):
    C = [_epoch(t) for t, _ in fx['crashes']]
    S = [fx['signatures'][k] for _, k in fx['crashes']]
    ends = [_epoch(t) for t in fx['episode_ends']]
    alive0 = [_epoch(t) for t in fx['episode_alive_since']]
    lmin = fx['replay_model']['min_lifetime_s']
    detect = fx['replay_model']['detect_s']
    restarts = giveups = 0
    for e, end in enumerate(ends):
        lo = ends[e - 1] if e else float('-inf')
        idx = [k for k, c in enumerate(C) if lo < c < end]
        ep_C, ep_S = [C[k] for k in idx], [S[k] for k in idx]
        t_alive, hist = alive0[e], []
        while True:
            j = bisect.bisect_left(ep_C, t_alive + lmin)
            if j >= len(ep_C):
                break
            now = ep_C[j] + detect
            wait = 0
            if backoff:
                hist.append((now, ep_S[j]))
                if rp.should_give_up(hist, now):
                    giveups += 1
                    break
                wait = rp.next_restart_delay(rp.consecutive_same(hist, now)) - rp.BACKOFF_BASE_SEC
            if now + wait >= end:
                break
            restarts += 1
            t_alive = now + wait
    return restarts, giveups


def test_april_replay_reproduces_acceptance_numbers(fx):
    exp = fx['expected']
    assert len(fx['crashes']) == 1778 and len(fx['signatures']) == 1
    cur, cur_g = _replay(fx, backoff=False)
    on, on_g = _replay(fx, backoff=True)
    assert (cur, cur_g) == (exp['current_restarts'], 0)
    assert on == exp['giveup_restarts'] and on_g == exp['giveup_alerts']
    assert round(100 * (1 - on / cur), 1) == exp['avoided_pct']


# ---------------------------------------------------------------------------
# wiring — fake procs, fake clock, the real _check_restart_bots
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class _Proc:
    """Fake Popen: alive until `crash_at` on `clock`, then exited with rc."""
    _next_pid = 4242

    def __init__(self, clock, crash_at=float('inf'), rc=1):
        self.clock, self.crash_at, self.rc = clock, crash_at, rc
        self.pid = _Proc._next_pid
        _Proc._next_pid += 1
        self.returncode = None
        self.signalled = []

    def poll(self):
        if self.returncode is None and self.clock() >= self.crash_at:
            self.returncode = self.rc
        return self.returncode

    def terminate(self):
        self.signalled.append('term')

    def kill(self):
        self.signalled.append('kill')

    def wait(self, timeout=None):
        return self.returncode


class _FH(io.StringIO):
    pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Patched run_pipeline: fake clock/_start_bot/notify; fresh backoff state."""
    clock = _Clock(1_000_000.0)
    started, notes = [], []
    ns = types.SimpleNamespace(clock=clock, started=started, notes=notes,
                               crash_after=None, tail='')

    def fake_start_bot(cmd, log_path):
        at = ns.crash_after(clock.t) if ns.crash_after else float('inf')
        p = _Proc(clock, at)
        started.append((cmd, log_path, p))
        return p, _FH()

    fake_notify = types.ModuleType('notify')
    fake_notify.notify = lambda msg, level='warning', dedupe_key=None: notes.append(
        (msg, level, dedupe_key))
    monkeypatch.setitem(sys.modules, 'notify', fake_notify)
    monkeypatch.setattr(rp, '_start_bot', fake_start_bot)
    monkeypatch.setattr(rp, 'mark_progress', lambda: None)
    monkeypatch.setattr(rp, 'CRYPTO_BOT_LOG', str(tmp_path / 'crypto_bot_output.log'))
    monkeypatch.setattr(rp, 'STOCK_BOT_LOG', str(tmp_path / 'stock_bot_output.log'))
    # raising=False: the OFF-path test also runs against the pre-edit module
    monkeypatch.setattr(rp, '_backoff_clock', clock, raising=False)
    monkeypatch.setattr(rp, '_read_crash_tail', lambda name, proc: ns.tail, raising=False)
    monkeypatch.setattr(rp, '_manually_stopped', set())
    monkeypatch.setattr(rp, '_backoff_state', {}, raising=False)
    monkeypatch.setattr(rp, '_restart_giveup', set(), raising=False)
    monkeypatch.setattr(rp, '_all_handles', [])
    monkeypatch.setattr(rp, '_COMBINED_BOTS', False)
    monkeypatch.setattr(rp, '_BOT_SCOPE', (False, True))
    return ns


def _pass(env, bots, log, dt_sec=60.0):
    env.clock.t += dt_sec
    rp._check_restart_bots(bots, log)


def test_flag_default_off_and_house_parse():
    src = inspect.getsource(rp)
    assert ("BOT_RESTART_BACKOFF = os.getenv('TRADER_BOT_RESTART_BACKOFF', '0')"
            ".strip().lower() in ('1', 'true', 'yes')") in src
    import os
    if 'TRADER_BOT_RESTART_BACKOFF' not in os.environ:
        assert rp.BOT_RESTART_BACKOFF is False


# The pre-edit function body (captured from the original run_pipeline.py):
# with the flag OFF every one of these statements must still execute, in order.
_PRE_EDIT_RESTART_BLOCK = '''            # Close the old log file handle before opening a new one
            try:
                bot_fh.close()
            except Exception:
                pass
            _untrack_handle(bot_fh)
            if name == 'Bots':  # combined-mode process'''


def _strip_on_blocks(src):
    """Drop every `if BOT_RESTART_BACKOFF...:` statement and its body."""
    out, skip_indent = [], None
    for ln in src.splitlines(keepends=True):
        ind = len(ln) - len(ln.lstrip())
        if skip_indent is not None:
            if ln.strip() and ind <= skip_indent:
                skip_indent = None
            else:
                continue
        if ln.strip().startswith('if BOT_RESTART_BACKOFF'):
            skip_indent = ind
            continue
        out.append(ln)
    return ''.join(out)


# sha256 of the PRE-EDIT sources (inspect.getsource of the original
# run_pipeline.py, captured from a scratch copy before the edit).
_PRE_EDIT_SHA = {
    '_check_restart_bots': '25a2fa529956fd7320bab8ee90fbf3662137394fd1627fb3fd88e4a4d901cd05',
    '_restart_bots': '6d9c28bae96eec84a7819b3b4a30c4b69bf0b2c6fbeeff59435f08b5387260a3',
    '_start_bots_now': 'f49d9bbb1380300cb2f0f6675c67a1de117f99c7ec8a0cb8bcccc379856b3d93',
}


@pytest.mark.parametrize('fn', sorted(_PRE_EDIT_SHA))
def test_off_path_source_pin(fn):
    """With every flag-guarded block removed, the three touched functions are
    byte-identical to the pre-edit code (flag OFF => original statements)."""
    import hashlib
    src = inspect.getsource(getattr(rp, fn))
    assert hashlib.sha256(_strip_on_blocks(src).encode()).hexdigest() == _PRE_EDIT_SHA[fn]
    if fn == '_check_restart_bots':
        assert _PRE_EDIT_RESTART_BLOCK in src


def test_off_path_restarts_every_pass_like_today(env, monkeypatch):
    """Pre-edit behaviour (verified against the scratch copy of the original):
    a crashed bot is restarted on the very next pass, every pass, forever."""
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', False, raising=False)
    env.crash_after = lambda t: t + 1  # every restart crashes 1 s later
    old = _Proc(env.clock, crash_at=0)
    old_fh = _FH()
    bots = [('Stock', old, old_fh)]
    log = _FH()
    for _ in range(12):
        _pass(env, bots, log)
    assert len(env.started) == 12
    cmd, path, first = env.started[0]
    assert cmd == [rp.PYTHON, '-u', 'stock_loop.py'] and path == rp.STOCK_BOT_LOG
    assert old_fh.closed
    assert log.getvalue().splitlines()[0] == f"Stock bot crashed (exit 1), restarted as PID {first.pid}"
    assert all(lvl == 'warning' and key == 'bot-crash-Stock' for _, lvl, key in env.notes)
    assert len(env.notes) == 12
    assert 'giving up' not in log.getvalue()


def test_on_backoff_ladder_then_single_giveup(env, monkeypatch, fx):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    env.tail = fx['example_tail']
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    restart_times = []
    for _ in range(200):
        n = len(env.started)
        _pass(env, bots, log)
        if len(env.started) > n:
            restart_times.append(env.clock.t)
    # 5th identical crash inside 60 min -> no 5th restart
    assert len(env.started) == 4
    assert bots == []
    assert rp._restart_giveup == {'Stock'}
    crit = [n for n in env.notes if n[1] == 'critical']
    assert len(crit) == 1 and crit[0][2] == 'bot-giveup-Stock'
    assert (f"[BOTS] giving up on Stock after 5 identical crashes ({fx['signatures'][0]})"
            in log.getvalue())
    # detection-pass restart, then +60/+180/+420 s after detection
    gaps = [b - a for a, b in zip(restart_times, restart_times[1:])]
    assert gaps == [120.0, 240.0, 480.0]  # 60 s alive+detect pass + deferred wait


def test_distinct_signature_crash_restarts_on_detection_pass(env, monkeypatch):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    tails = iter(f'Traceback (most recent call last)\n  File "/x/m{k}.py", line {k}, in f\n'
                 f'E{k}Error: boom\n' for k in range(50))
    monkeypatch.setattr(rp, '_read_crash_tail', lambda name, proc: next(tails))
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    for k in range(20):
        _pass(env, bots, log)          # every pass detects a new, distinct crash
        assert len(env.started) == k + 1  # ... and restarts on that same pass
    assert rp._restart_giveup == set()
    assert 'deferred' not in log.getvalue()


def test_giveup_latch_never_touches_a_running_bot(env, monkeypatch, fx):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    monkeypatch.setattr(rp, '_BOT_SCOPE', (True, True))
    env.tail = fx['example_tail']
    env.crash_after = lambda t: t + 30
    crypto = _Proc(env.clock)  # never crashes
    crypto_fh = _FH()
    bots = [('Crypto', crypto, crypto_fh),
            ('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    for _ in range(200):
        _pass(env, bots, log)
    assert rp._restart_giveup == {'Stock'}
    assert bots == [('Crypto', crypto, crypto_fh)]
    assert crypto.signalled == [] and not crypto_fh.closed
    assert all(cmd[-1] == 'stock_loop.py' for cmd, _, _ in env.started)


def test_manual_start_unlatches_and_resets_history(env, monkeypatch, fx):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    env.tail = fx['example_tail']
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    for _ in range(200):
        _pass(env, bots, log)
    assert rp._restart_giveup == {'Stock'} and bots == []
    result = rp._start_bots_now(bots, log, False, True)   # operator start_bot
    assert result == ('accepted', '')
    assert rp._restart_giveup == set()
    assert rp._backoff_state.get('Stock', {}).get('hist', []) == []
    n = len(env.started)
    _pass(env, bots, log)   # crashes again (6th identical inside 60 min) ...
    assert len(env.started) == n + 1  # ... but history was cleared: restart at base
    assert rp._restart_giveup == set()


def test_weekly_restart_bots_unlatches(env, monkeypatch, fx):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    env.tail = fx['example_tail']
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    for _ in range(200):
        _pass(env, bots, log)
    assert rp._restart_giveup == {'Stock'}
    rp._restart_bots(bots, log, False, True)
    assert rp._restart_giveup == set() and rp._backoff_state == {}
    assert [b[0] for b in bots] == ['Stock']


def test_manual_start_during_deferred_wait_cannot_duplicate(env, monkeypatch, fx):
    """A dead entry parked in a backoff wait is dropped by the operator start,
    so the deferred restart cannot later run beside the new process."""
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)
    env.tail = fx['example_tail']
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    while 'deferred' not in log.getvalue():
        _pass(env, bots, log)
    assert bots[0][1].poll() is not None      # parked, dead
    env.crash_after = None                    # the fixed bot now stays up
    rp._start_bots_now(bots, log, False, True)
    for _ in range(30):
        _pass(env, bots, log)
    assert len(bots) == 1 and bots[0][1].poll() is None


def test_on_path_internal_error_fails_open(env, monkeypatch):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', True)

    def boom(*a, **k):
        raise RuntimeError('x')
    monkeypatch.setattr(rp, 'should_give_up', boom)
    env.crash_after = lambda t: t + 30
    bots = [('Stock', _Proc(env.clock, crash_at=env.clock.t + 30), _FH())]
    log = _FH()
    for _ in range(10):
        _pass(env, bots, log)
    assert len(env.started) == 10   # = today's behaviour, never raises


def test_read_crash_tail_ignores_output_before_process_start(tmp_path, monkeypatch):
    log = tmp_path / 'stock_bot_output.log'
    old = ('Traceback (most recent call last)\n  File "/x/old.py", line 1, in f\n'
           'KeyError: old\n')
    log.write_text(old)
    monkeypatch.setattr(rp, 'STOCK_BOT_LOG', str(log))
    monkeypatch.setattr(rp, '_backoff_state', {})
    clock = _Clock()
    p = _Proc(clock)
    rp._backoff_note_proc('Stock', p)          # offset = size before this run
    with open(log, 'a') as f:
        f.write('killed without a traceback\n')
    assert rp.crash_signature(rp._read_crash_tail('Stock', p)) == 'unknown'
    assert rp.crash_signature(rp._read_crash_tail('Stock', _Proc(clock))) == \
        'old.py:1 | KeyError: old'           # unknown start -> whole tail


def test_april_timeline_through_real_monitor(env, monkeypatch, fx):
    """Drive the real _check_restart_bots over the three April episodes on a
    60 s pass grid (weekly _restart_bots between episodes): 12 restarts and
    3 critical alerts ON, vs >= 99 % more restarts OFF."""
    C = [_epoch(t) for t, _ in fx['crashes']]
    ends = [_epoch(t) for t in fx['episode_ends']]
    alive0 = [_epoch(t) for t in fx['episode_alive_since']]
    lmin = fx['replay_model']['min_lifetime_s']
    env.tail = fx['example_tail']

    def run(flag):
        monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', flag)
        monkeypatch.setattr(rp, '_backoff_state', {})
        monkeypatch.setattr(rp, '_restart_giveup', set())
        env.started.clear()
        env.notes.clear()
        restarts = 0
        for e, end in enumerate(ends):
            lo = ends[e - 1] if e else float('-inf')
            ep_C = [c for c in C if lo < c < end]

            def crash_after(t, ep_C=ep_C, end=end):
                j = bisect.bisect_left(ep_C, t + lmin)
                return ep_C[j] if j < len(ep_C) else float('inf')
            env.crash_after = crash_after
            env.clock.t = alive0[e]
            bots = []
            rp._restart_bots(bots, _FH(), False, True)
            n0 = len(env.started)
            log = _FH()
            while bots:
                env.clock.t += 60.0
                if env.clock.t >= end:
                    break
                rp._check_restart_bots(bots, log)
            restarts += len(env.started) - n0
        crit = sum(1 for n in env.notes if n[1] == 'critical')
        return restarts, crit

    on, on_crit = run(True)
    off, off_crit = run(False)
    assert (on, on_crit) == (12, 3)
    assert off_crit == 0 and 1 - on / off >= 0.99


# ---------------------------------------------------------------------------
# F1 (class A, flag-independent): a start inside the crash -> monitor-pass
# window must not leave the dead entry for the next pass to ALSO restart.
# ---------------------------------------------------------------------------

def _live(bots, name):
    return [p for n, p, _ in bots if n == name and p.poll() is None]


@pytest.mark.parametrize('flag', [False, True])
@pytest.mark.parametrize('mode', ['split', 'combined', 'weekly_restart'])
def test_start_inside_crash_window_leaves_one_live_process(env, monkeypatch, mode, flag):
    monkeypatch.setattr(rp, 'BOT_RESTART_BACKOFF', flag, raising=False)
    combined = mode == 'combined'
    monkeypatch.setattr(rp, '_COMBINED_BOTS', combined)
    name = 'Bots' if combined else 'Stock'
    dead_fh = _FH()
    bots = [(name, _Proc(env.clock, crash_at=env.clock.t + 10), dead_fh)]
    log = _FH()
    env.clock.t += 20                        # crashed; monitor has not run yet
    if mode == 'weekly_restart':
        rp._restart_bots(bots, log, False, True)
    else:
        assert rp._start_bots_now(bots, log, False, True) == ('accepted', '')
    for _ in range(5):
        _pass(env, bots, log)                # the next monitor passes
    assert len(_live(bots, name)) == 1
    assert len(bots) == 1 and len(env.started) == 1
    assert dead_fh.closed
