"""G3 ops & orchestration fixes (2026-09-26 Jetson campaign) — Mac-safe.

run_pipeline and log_config are stdlib-only at import (run_pipeline pulls in
adaptive_config, also pure); every file a test writes is redirected into
tmp_path; bots are FAKE processes (no real bot, pipeline, training or order);
the only real subprocesses are tiny `sys.executable -c` children and bash.

Pins:
  G3-1  combined mode: a post-suspend bot start launches run_bots.py for
        _BOT_SCOPE (never a split crypto_loop.py) — the GUI's two-click
        sequence leaves exactly ONE process trading crypto; split mode
        unchanged; main()'s two post-suspend blocks use _start_pending_bots.
  G3-2  suspend_and_start_bot with no training phase running is handled as
        start_bot and never latches _suspend_requested (the next phase then
        runs to completion instead of aborting with -99).
  G3-3  run_phase: an undecodable byte or a failing log write returns the
        child's exit code instead of deadlocking in proc.wait().
  G3-4  write_status: per-thread tmp path — a forced heartbeat/main
        interleave never publishes a torn pipeline_status.json.
  G3-5  log_config: 4 processes rotating one trader.log -> 0 logging errors,
        full-size backups; same path; handler follows a foreign rotation.
  G3-6  the five formerly TTY-only messages go through _announce
        (print(..., flush=True) + trader.log), the signal-handler sites don't.
  G3-7  backup_state.sh backs up the study DBs without the sqlite3 CLI.
"""
import ast
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
from logging.handlers import RotatingFileHandler

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(ROOT, 'run_pipeline.py')
BACKUP = os.path.join(ROOT, 'scripts', 'backup_state.sh')

rp = pytest.importorskip('run_pipeline')
import log_config  # noqa: E402  (stdlib-only)


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _func_src(name):
    src = _read(PIPELINE)
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f'{name} not found in run_pipeline.py')


# ---------------------------------------------------------------- fakes

class FakeProc:
    n = 1000

    def __init__(self, cmd):
        FakeProc.n += 1
        self.pid = FakeProc.n
        self.cmd = cmd
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    """run_pipeline with every file in tmp_path and a fake _start_bot."""
    monkeypatch.setattr(rp, 'STATUS_FILE', str(tmp_path / 'pipeline_status.json'))
    monkeypatch.setattr(rp, 'COMMAND_RESULT_FILE', str(tmp_path / 'command_result.json'))
    monkeypatch.setattr(rp, 'PIPELINE_COMMAND', str(tmp_path / 'pipeline_command.json'))
    monkeypatch.setattr(rp, 'CRYPTO_BOT_LOG', str(tmp_path / 'crypto_bot_output.log'))
    monkeypatch.setattr(rp, 'STOCK_BOT_LOG', str(tmp_path / 'stock_bot_output.log'))
    monkeypatch.setattr(rp, '_manually_stopped', set())
    monkeypatch.setattr(rp, '_COMBINED_BOTS', False)
    monkeypatch.setattr(rp, '_BOT_SCOPE', (True, True))
    monkeypatch.setattr(rp, '_suspend_requested', False)
    monkeypatch.setattr(rp, '_last_status_write', 0)
    monkeypatch.setattr(rp, '_all_handles', [])
    monkeypatch.setattr(rp, '_start_bot',
                        lambda cmd, log_path: (FakeProc(cmd), io.StringIO()))
    return rp


def _alive(bots):
    return [(n, ' '.join(p.cmd[2:])) for n, p, _ in bots if p.poll() is None]


def _crypto_traders(bots):
    return [c for _, c in _alive(bots)
            if 'crypto_loop' in c or ('run_bots' in c and '--stock-only' not in c)]


def _ack(tmp_path):
    with open(tmp_path / 'command_result.json') as f:
        return json.load(f)


def _status(phase):
    return {'phase': phase, 'phase_label': '', 'phase_idx': 0, 'started_at': '',
            'total_phases': 1, 'bots_running': False,
            'crypto_bot_running': False, 'stock_bot_running': False,
            '_pending_bot_start': None}


# ================================================================= G3-1

def test_g3_1_combined_two_click_sequence_one_crypto_process(pipe, tmp_path):
    """The hunter's sequence: suspend_and_start_bot{crypto} mid-phase ->
    post-suspend start -> GUI start_bot{stock} -> exactly one crypto trader."""
    rp = pipe
    rp._COMBINED_BOTS = True
    bots, log = [], io.StringIO()
    rp._start_pending_bots(bots, log, {'crypto': True, 'stock': False})
    assert _alive(bots) == [('Bots', 'run_bots.py')]
    rp._handle_command({'command': 'start_bot', 'crypto': False, 'stock': True},
                       bots, log, _status('trading'))
    assert len(_crypto_traders(bots)) == 1
    assert not any('crypto_loop' in c or 'stock_loop' in c for _, c in _alive(bots))
    assert _ack(tmp_path)['result'] == 'rejected'
    # "Stop Crypto" in combined mode now stops the only crypto trader.
    rp._handle_command({'command': 'stop_bot', 'crypto': True, 'stock': False},
                       bots, log, _status('trading'))
    assert _crypto_traders(bots) == []


@pytest.mark.parametrize('scope,flag', [((True, False), '--crypto-only'),
                                        ((False, True), '--stock-only'),
                                        ((True, True), None)])
def test_g3_1_combined_pending_launches_bot_scope(pipe, scope, flag):
    rp = pipe
    rp._COMBINED_BOTS = True
    rp._BOT_SCOPE = scope
    bots = []
    rp._start_pending_bots(bots, io.StringIO(), {'crypto': False, 'stock': True})
    assert len(bots) == 1 and bots[0][0] == 'Bots'
    cmd = bots[0][1].cmd
    assert cmd[2] == 'run_bots.py'
    assert (flag in cmd) if flag else len(cmd) == 3


def test_g3_1_combined_pending_noop_when_combined_alive(pipe):
    rp = pipe
    rp._COMBINED_BOTS = True
    bots = [('Bots', FakeProc(['py', '-u', 'run_bots.py']), io.StringIO())]
    rp._start_pending_bots(bots, io.StringIO(), {'crypto': True, 'stock': True})
    assert len(bots) == 1


@pytest.mark.parametrize('pending,expect', [
    ({'crypto': True, 'stock': False}, [('Crypto', 'crypto_loop.py')]),
    ({'crypto': False, 'stock': True}, [('Stock', 'stock_loop.py')]),
    ({'crypto': True, 'stock': True},
     [('Crypto', 'crypto_loop.py'), ('Stock', 'stock_loop.py')]),
    ({}, []), (None, []),
])
def test_g3_1_split_mode_unchanged(pipe, pending, expect):
    bots = []
    pipe._start_pending_bots(bots, io.StringIO(), pending)
    assert _alive(bots) == expect


def test_g3_1_start_single_bot_refuses_in_combined_mode(pipe):
    rp = pipe
    rp._COMBINED_BOTS = True
    bots, log = [], io.StringIO()
    rp._start_single_bot(bots, 'Crypto', log)
    assert bots == []
    assert 'Refusing per-book Crypto start' in log.getvalue()


def test_g3_1_main_post_suspend_blocks_use_helper():
    main = _func_src('main')
    assert main.count('_start_pending_bots(bots, log_fh, pending)') == 2
    assert '_start_single_bot' not in main


# ================================================================= G3-2

@pytest.mark.parametrize('phase', rp._IDLE_PHASES)
def test_g3_2_idle_suspend_is_start_bot_no_latch(pipe, tmp_path, phase):
    rp = pipe
    bots, status = [], _status(phase)
    rp._handle_command({'command': 'suspend_and_start_bot', 'crypto': True,
                        'stock': False}, bots, io.StringIO(), status)
    assert rp._suspend_requested is False
    assert status['_pending_bot_start'] is None
    assert _alive(bots) == [('Crypto', 'crypto_loop.py')]
    assert status['crypto_bot_running'] is True
    ack = _ack(tmp_path)
    assert (ack['command'], ack['result']) == ('suspend_and_start_bot', 'accepted')


def test_g3_2_idle_suspend_combined_launches_run_bots(pipe, tmp_path):
    rp = pipe
    rp._COMBINED_BOTS = True
    bots = []
    rp._handle_command({'command': 'suspend_and_start_bot', 'crypto': True,
                        'stock': False}, bots, io.StringIO(), _status('trading'))
    assert _alive(bots) == [('Bots', 'run_bots.py')]
    assert rp._suspend_requested is False
    # a second one while the combined process lives: rejected, no duplicate
    rp._handle_command({'command': 'suspend_and_start_bot', 'crypto': True,
                        'stock': False}, bots, io.StringIO(), _status('trading'))
    assert len(bots) == 1
    assert _ack(tmp_path)['result'] == 'rejected'


def test_g3_2_training_phase_still_latches(pipe):
    rp = pipe
    bots, status = [], _status('crypto_search')
    rp._handle_command({'command': 'suspend_and_start_bot', 'crypto': True,
                        'stock': False}, bots, io.StringIO(), status)
    assert rp._suspend_requested is True
    assert status['_pending_bot_start'] == {'crypto': True, 'stock': False}
    assert bots == []


def test_g3_2_next_phase_not_aborted(pipe):
    """Hunter's R2: after the wait-loop command, the next phase runs to rc 0."""
    rp = pipe
    status = _status('trading')
    rp._handle_command({'command': 'suspend_and_start_bot', 'crypto': True,
                        'stock': False}, [], io.StringIO(), status)
    phase = {'idx': 0, 'id': 'crypto_harvest', 'label': 'Harvest',
             'cmd': [sys.executable, '-c',
                     'for i in range(5): print("line", i, flush=True)\n'
                     'print("DONE", flush=True)']}
    plog = io.StringIO()
    assert rp.run_phase(phase, plog, status) == 0
    assert 'DONE' in plog.getvalue()
    assert '[SUSPEND]' not in plog.getvalue()


# ================================================================= G3-3

_DRIVER = textwrap.dedent('''
    import io, json, os, sys
    from pathlib import Path
    root, tmp, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, root)
    import log_config
    log_config._LOG_DIR = Path(tmp) / 'logs'
    log_config._LOG_FILE = Path(tmp) / 'logs' / 'trader.log'
    import run_pipeline as rp
    rp.STATUS_FILE = os.path.join(tmp, 'pipeline_status.json')
    rp.PIPELINE_COMMAND = os.path.join(tmp, 'pipeline_command.json')
    child = ("import sys\\n"
             "sys.stdout.buffer.write(b'trial 1 \\\\xff\\\\n'); sys.stdout.flush()\\n"
             "for i in range(4000): print('x'*60, i, flush=True)\\n"
             "print('CHILD DONE', flush=True)\\n"
             "sys.exit(int(sys.argv[1]))\\n")
    rc_want = 7 if mode == 'failing_log_rc7' else 0

    class FailingLog(io.StringIO):
        n = 0
        def write(self, s):
            FailingLog.n += 1
            if FailingLog.n > 5:
                raise OSError(28, 'No space left on device')
            return super().write(s)

    log = FailingLog() if mode.startswith('failing_log') else io.StringIO()
    phase = {'idx': 0, 'id': 'crypto_meta', 'label': 'x',
             'cmd': [sys.executable, '-c', child, str(rc_want)]}
    rc = rp.run_phase(phase, log, {'total_phases': 1})
    v = log.getvalue()
    print('RESULT ' + json.dumps({'rc': rc, 'replaced': '\\ufffd' in v,
                                  'done': 'CHILD DONE' in v}))
''')


@pytest.mark.parametrize('mode,rc', [('undecodable', 0), ('failing_log', 0),
                                     ('failing_log_rc7', 7)])
def test_g3_3_run_phase_never_deadlocks(tmp_path, mode, rc):
    drv = tmp_path / 'driver.py'
    drv.write_text(_DRIVER)
    try:
        r = subprocess.run([sys.executable, str(drv), ROOT, str(tmp_path), mode],
                           capture_output=True, text=True, timeout=90,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    except subprocess.TimeoutExpired:
        pytest.fail(f'run_phase deadlocked ({mode})')
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(line[-1][len('RESULT '):])
    assert res['rc'] == rc
    if mode == 'undecodable':
        assert res['replaced'] and res['done']  # decoded, logged to the end
    else:
        assert 'Error reading phase output' in r.stdout  # _announce reached stdout


def test_g3_3_popen_decodes_with_replace():
    src = _func_src('run_phase')
    assert "errors='replace'" in src
    assert '_drain_phase_output(proc)' in src


def test_g3_3_drain_kills_child_when_unreadable():
    killed = []

    class Raw:
        def read1(self, n):
            raise OSError('boom')

    class P:
        stdout = type('S', (), {'buffer': Raw()})()

        def kill(self):
            killed.append(1)
    rp._drain_phase_output(P())
    assert killed == [1]


# ================================================================= G3-4

class _JsonShim:
    """json stand-in whose dump pauses one named thread after buffering."""
    load, loads, JSONDecodeError = json.load, json.loads, json.JSONDecodeError

    def __init__(self, pause_thread):
        self.pause_thread = pause_thread
        self.buffered = threading.Event()
        self.release = threading.Event()

    def dump(self, obj, f, **kw):
        json.dump(obj, f, **kw)
        if threading.current_thread().name == self.pause_thread:
            self.buffered.set()
            self.release.wait(5)


def _g3_4_status():
    return {'started_at': '2026-09-26T00:00:00', 'phase': 'crypto_search',
            'trial_current': 9, 'phase_results': {}}


def test_g3_4_heartbeat_paused_main_writes_between(pipe, tmp_path, monkeypatch):
    """Hunter's r3 interleave: heartbeat buffers its JSON and pauses, the main
    thread (no lock) writes a LONGER status and publishes, heartbeat resumes."""
    rp = pipe
    shim = _JsonShim('heartbeat')
    monkeypatch.setattr(rp, 'json', shim)
    status = _g3_4_status()

    def hb():
        with rp._heartbeat_lock:
            rp.write_status(status, force=True)
    t = threading.Thread(target=hb, name='heartbeat')
    t.start()
    assert shim.buffered.wait(5)
    rp.write_status(dict(status, phase_results={'x': 'y' * 300}), force=True)
    shim.release.set()
    t.join(5)
    with open(rp.STATUS_FILE) as f:
        json.load(f)  # raises on a torn file
    assert [p for p in os.listdir(tmp_path) if '.tmp.' in p] == []


def test_g3_4_main_paused_heartbeat_writes_between(pipe, tmp_path, monkeypatch):
    """Mirror image: the main thread buffers and pauses while a heartbeat
    thread writes a LONGER status and publishes."""
    rp = pipe
    status = _g3_4_status()
    other = threading.Thread(
        target=lambda: rp.write_status(dict(status, phase_results={'x': 'y' * 300}),
                                       force=True), name='heartbeat')

    class Shim(_JsonShim):
        def dump(self, obj, f, **kw):
            json.dump(obj, f, **kw)
            if threading.current_thread() is threading.main_thread():
                other.start()
                other.join(5)
    monkeypatch.setattr(rp, 'json', Shim('unused'))
    rp.write_status(status, force=True)
    with open(rp.STATUS_FILE) as f:
        assert json.load(f)['phase'] == 'crypto_search'
    assert [p for p in os.listdir(tmp_path) if '.tmp.' in p] == []


def test_g3_4_tmp_is_per_thread():
    src = _func_src('write_status')
    assert "f'.tmp.{os.getpid()}.{threading.get_ident()}'" in src


def test_g3_4_partial_tmp_removed_on_runtime_error(pipe, tmp_path, monkeypatch):
    rp = pipe

    class Boom:
        def dump(self, obj, f, **kw):
            f.write('{"partial"')
            raise RuntimeError('dictionary changed size during iteration')
    monkeypatch.setattr(rp, 'json', Boom())
    with pytest.raises(RuntimeError):
        rp.write_status({'started_at': ''}, force=True)
    assert [p for p in os.listdir(tmp_path) if '.tmp.' in p] == []


# ================================================================= G3-5

_WRITER = textwrap.dedent('''
    import logging, sys
    sys.path.insert(0, sys.argv[1])
    from log_config import SharedRotatingFileHandler
    path, tag, maxb, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    h = SharedRotatingFileHandler(path, maxBytes=maxb, backupCount=5, encoding='utf-8')
    h.setFormatter(logging.Formatter('%(message)s'))
    lg = logging.getLogger('x'); lg.addHandler(h); lg.setLevel(logging.INFO)
    for i in range(n):
        lg.info('%s %06d %s', tag, i, 'y' * 60)
''')


def test_g3_5_four_writers_no_errors_full_backups(tmp_path):
    maxb, n = 20_000, 3000
    w = tmp_path / 'writer.py'
    w.write_text(_WRITER)
    path = tmp_path / 'trader.log'
    procs = [subprocess.Popen([sys.executable, str(w), ROOT, str(path), f'P{k}',
                               str(maxb), str(n)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in range(4)]
    errs = []
    for p in procs:
        _, e = p.communicate(timeout=120)
        errs.append(e)
        assert p.returncode == 0, e
    assert not any('Logging error' in e for e in errs), errs
    backups = [tmp_path / f'trader.log.{i}' for i in range(1, 6)]
    assert all(b.exists() for b in backups)
    for b in backups:
        size = b.stat().st_size
        assert size >= 0.9 * maxb, (b.name, size)
        for line in b.read_text().splitlines():
            assert re.fullmatch(r'P[0-3] \d{6} y{60}', line), line


def test_g3_5_follows_foreign_rotation(tmp_path):
    """Two handlers on one file (= two processes): after A rotates, B writes
    into the NEW trader.log, not the renamed backup; B's rollover then sees
    the fresh file and does not rotate a second time."""
    path = str(tmp_path / 'trader.log')
    fmt = logging.Formatter('%(message)s')
    a = log_config.SharedRotatingFileHandler(path, maxBytes=200, backupCount=3)
    b = log_config.SharedRotatingFileHandler(path, maxBytes=200, backupCount=3)
    for h in (a, b):
        h.setFormatter(fmt)

    def rec(msg):
        return logging.LogRecord('x', logging.INFO, __file__, 1, msg, None, None)
    try:
        a.handle(rec('A' * 150))
        a.handle(rec('A' * 150))            # A rotates: .1 = first line
        b.handle(rec('B-after-rotation'))
        assert 'B-after-rotation' in open(path).read()
        assert 'B-after-rotation' not in open(path + '.1').read()
        # B asked to roll over while A already did: inode check -> just reopen
        a.handle(rec('A' * 150))            # A rotates again
        b.doRollover()                      # stale view -> no second rotation
        assert not os.path.exists(path + '.3')
    finally:
        a.close()
        b.close()


@pytest.fixture
def fresh_lc(monkeypatch, tmp_path):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    monkeypatch.setattr(log_config, '_configured', False)
    monkeypatch.setattr(log_config, '_file_handler', None)
    monkeypatch.setattr(log_config, '_LOG_DIR', tmp_path / 'logs')
    monkeypatch.setattr(log_config, '_LOG_FILE', tmp_path / 'logs' / 'trader.log')
    yield log_config
    for h in list(root.handlers):
        if h not in before:
            h.close()
    root.handlers[:] = before
    root.setLevel(level)
    lg = logging.getLogger('g3_file_only')
    lg.handlers[:] = []
    lg.propagate = True


def test_g3_5_get_logger_uses_shared_handler_same_path(fresh_lc):
    lc = fresh_lc
    before = {id(h) for h in logging.getLogger().handlers}
    lc.get_logger('g3_probe')
    added = [h for h in logging.getLogger().handlers if id(h) not in before]
    fhs = [h for h in added if isinstance(h, RotatingFileHandler)]
    assert len(fhs) == 1 and type(fhs[0]) is lc.SharedRotatingFileHandler
    assert fhs[0].baseFilename == str(lc._LOG_FILE)
    assert (fhs[0].maxBytes, fhs[0].backupCount) == (lc._MAX_BYTES, lc._BACKUP_COUNT)


def test_g3_5_file_logger_does_not_propagate(fresh_lc):
    lc = fresh_lc
    lg = lc.get_file_logger('g3_file_only')
    assert lg.propagate is False
    assert lg.handlers == [lc._file_handler]
    lc.get_file_logger('g3_file_only')
    assert len(lg.handlers) == 1  # idempotent
    lg.warning('g3 file-only probe')
    lc._file_handler.flush()
    assert 'g3 file-only probe' in lc._LOG_FILE.read_text(encoding='utf-8')


# ================================================================= G3-6

_G3_6_MESSAGES = ['GPU still hot', 'Error reading phase output',
                  'Crypto training data is', 'Stock training data is',
                  'D03: shadow mode is ON']


def test_g3_6_five_sites_use_announce():
    src = _read(PIPELINE)
    for needle in _G3_6_MESSAGES:
        calls = [m.group(0) for m in re.finditer(
            r'(_announce|_print)\((?:[^()]|\([^()]*\))*?' + re.escape(needle), src)]
        assert calls and all(c.startswith('_announce(') for c in calls), needle


def test_g3_6_announce_prints_flush_and_logs():
    body = _func_src('_announce')
    assert '_print(msg, always=True, flush=True)' in body
    assert 'get_file_logger' in body
    assert '_STDOUT_IS_TTY' not in body
    pr = _func_src('_print')
    assert 'if _STDOUT_IS_TTY or always:' in pr


def test_g3_6_signal_handler_sites_untouched():
    for name in ('_signal_handler',):
        body = _func_src(name)
        assert '_announce(' not in body


def test_g3_6_announce_reaches_non_tty_stdout(capsys, monkeypatch):
    got = []
    lg = logging.getLogger('g3_announce_capture')
    lg.propagate = False
    lg.setLevel(logging.DEBUG)  # as get_file_logger does

    class H(logging.Handler):
        def emit(self, r):
            got.append((r.levelname, r.getMessage()))
    h = H()
    lg.addHandler(h)
    monkeypatch.setattr(log_config, 'get_file_logger', lambda name: lg)
    monkeypatch.setattr(rp, '_STDOUT_IS_TTY', False)
    try:
        rp._announce('[GATE] probe')
        rp._announce('harvest skipped probe', level='info')
    finally:
        lg.removeHandler(h)
    out = capsys.readouterr().out
    assert '[GATE] probe' in out and 'harvest skipped probe' in out
    assert got == [('WARNING', '[GATE] probe'), ('INFO', 'harvest skipped probe')]


def test_g3_6_announce_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError('broken pipe')
    monkeypatch.setattr(log_config, 'get_file_logger', boom)
    monkeypatch.setattr('builtins.print', boom)
    rp._announce('x')  # must not raise


# ================================================================= G3-7

def _db_step(tmp_path):
    src = _read(BACKUP)
    block = src.split('# 1. SQLite')[1].split('# 2. JSON')[0]
    stage = tmp_path / 'stage'
    stage.mkdir()
    work = tmp_path / 'work'
    work.mkdir()
    for name in ('v2_study.db', 'stock_v2_study.db'):
        con = sqlite3.connect(work / name)
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('CREATE TABLE t (i INTEGER)')
        con.executemany('INSERT INTO t VALUES (?)', [(i,) for i in range(100)])
        con.commit()
        con.close()
    script = tmp_path / 'step1.sh'
    script.write_text(f'set -euo pipefail\nSTAGE="{stage}"\ncd "{work}"\n#' + block
                      + '\necho STEP1_END\n')
    return script, stage


def _rows(path):
    con = sqlite3.connect(path)
    try:
        return con.execute('SELECT COUNT(*) FROM t').fetchone()[0]
    finally:
        con.close()


BASH = shutil.which('bash') or '/bin/bash'


def test_g3_7_backs_up_without_sqlite_cli(tmp_path):
    script, stage = _db_step(tmp_path)
    empty = tmp_path / 'emptybin'
    empty.mkdir()                               # PATH has no sqlite3 at all
    r = subprocess.run([BASH, str(script)], capture_output=True, text=True,
                       env={'PATH': str(empty), 'TRADER_PYBIN': sys.executable})
    assert r.returncode == 0, r.stderr
    assert 'WARN' not in r.stdout and 'STEP1_END' in r.stdout
    for name in ('v2_study.db', 'stock_v2_study.db'):
        assert _rows(stage / name) == 100


def test_g3_7_prefers_cli_when_present(tmp_path):
    script, stage = _db_step(tmp_path)
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    marker = tmp_path / 'cli_calls'
    fake = fakebin / 'sqlite3'
    fake.write_text(f'#!{BASH}\necho "$@" >> "{marker}"\n')
    fake.chmod(0o755)
    r = subprocess.run([BASH, str(script)], capture_output=True, text=True,
                       env={'PATH': str(fakebin), 'TRADER_PYBIN': '/nonexistent/python'})
    assert r.returncode == 0, r.stderr
    calls = marker.read_text().splitlines()
    assert len(calls) == 2 and all('.backup' in c for c in calls)


def test_g3_7_failure_warns_and_continues(tmp_path):
    script, _ = _db_step(tmp_path)
    empty = tmp_path / 'emptybin'
    empty.mkdir()
    r = subprocess.run([BASH, str(script)], capture_output=True, text=True,
                       env={'PATH': str(empty), 'TRADER_PYBIN': '/nonexistent/python'})
    assert r.returncode == 0
    assert r.stdout.count('WARN: sqlite backup failed') == 2
    assert 'STEP1_END' in r.stdout


def test_g3_7_bash_syntax():
    r = subprocess.run([BASH, '-n', BACKUP], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
