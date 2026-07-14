"""Review batch b19 — monitor_drift.py, gpu_lock.py, hw_monitor.py.

Pins the robustness/observability fixes:
  monitor_drift: poison history lines drop only themselves (TypeError /
    non-numeric / non-dict), poison manifests -> None, CUSUM + prune run
    even when PSI is not checkable, cross-process history lock, loud
    trade_memory import failure, docstring wiring matches run_pipeline.
  gpu_lock: interrupted waiter no longer clears the real holder's info,
    status tolerates incomplete info files, SH (not EX) free-probe,
    atomic info write with warn-on-failure, dead _lock_fd global removed.
  hw_monitor: zone-scan done-flag set after the scan (combined-bots
    thread race), one-shot warning when the temp sensor goes dark,
    cosmetic/doc pins.
"""

import datetime as dt
import inspect
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gpu_lock
import hw_monitor
import monitor_drift


# ---------------------------------------------------------------------------
# monitor_drift
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every monitor_drift file at a temp dir."""
    monkeypatch.setattr(monitor_drift, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(monitor_drift, '_STATE_FILE',
                        tmp_path / 'drift_state.json')
    return tmp_path


def _aware_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def test_naive_ts_line_drops_only_itself(sandbox):
    """A naive-ts line (TypeError on aware>=naive) must not kill the check."""
    naive = dt.datetime.now().isoformat()  # no tzinfo
    with open(monitor_drift.history_file(''), 'w') as f:
        f.write(json.dumps({'ts': naive, 'preds': {'A': 9.9}}) + '\n')
        f.write(json.dumps({'ts': _aware_now(), 'preds': {'B': 0.5}}) + '\n')
    vals = monitor_drift.load_recent_predictions('')
    assert list(vals) == pytest.approx([0.5])

    monitor_drift.prune_history('')  # must not raise
    text = monitor_drift.history_file('').read_text()
    assert '9.9' not in text and '0.5' in text


def test_poison_pred_values_drop_only_their_line(sandbox):
    """Non-numeric / non-dict preds crash asarray without per-value float()."""
    ts = _aware_now()
    with open(monitor_drift.history_file(''), 'w') as f:
        f.write(json.dumps({'ts': ts, 'preds': {'A': 0.1}}) + '\n')
        f.write(json.dumps({'ts': ts, 'preds': {'B': 'garbage',
                                                'C': 0.7}}) + '\n')
        f.write(json.dumps({'ts': ts, 'preds': 'not-a-dict'}) + '\n')
        f.write(json.dumps([1, 2, 3]) + '\n')       # valid JSON, not a dict
        f.write(json.dumps({'ts': ts, 'preds': {'D': 0.9}}) + '\n')
    vals = sorted(monitor_drift.load_recent_predictions(''))
    assert vals == pytest.approx([0.1, 0.9])

    monitor_drift.prune_history('')  # must not raise on any poison line
    kept = monitor_drift.history_file('').read_text()
    assert '0.1' in kept and '0.9' in kept


def test_load_ref_deciles_poison_manifest(sandbox):
    """Non-dict holdout (AttributeError) / unsized deciles (TypeError)."""
    p = sandbox / 'model_v2.manifest.json'
    p.write_text(json.dumps({'holdout': 'oops'}))
    assert monitor_drift.load_ref_deciles('') is None
    p.write_text(json.dumps({'holdout': {'pred_deciles': 5}}))
    assert monitor_drift.load_ref_deciles('') is None
    p.write_text(json.dumps([1, 2]))  # non-dict manifest
    assert monitor_drift.load_ref_deciles('') is None


def test_run_check_cusum_and_prune_when_psi_not_checkable(sandbox,
                                                          monkeypatch):
    """CUSUM + prune must run even when PSI has no deciles / too few preds
    (weekend stock windows, legacy manifests) — they sat after an early
    return before."""
    with open(sandbox / 'model_v2.manifest.json', 'w') as f:
        json.dump({'holdout': {'hit_rate': 0.55}}, f)  # no pred_deciles
    trades = [{'ts': f'2026-06-01T00:{i:02d}:00+00:00', 'action': 'sell',
               'exit': 1, 'pnl_pct': -1.0, 'estimated': False}
              for i in range(20)]
    import trade_memory
    monkeypatch.setattr(trade_memory, '_load', lambda: {'BTC/USD': trades})

    old_ts = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(days=30)).isoformat()
    with open(monitor_drift.history_file(''), 'w') as f:
        f.write(json.dumps({'ts': old_ts, 'preds': {'BTC/USD': 1.0}}) + '\n')

    assert monitor_drift.run_check('', 'crypto') is None  # PSI unchanged
    st = monitor_drift._load_state().get('crypto', {})
    assert st.get('cusum_last_ts')  # CUSUM processed outcomes anyway
    assert monitor_drift.history_file('').read_text() == ''  # prune ran


def test_run_check_checkable_path_keeps_cusum_state(sandbox, monkeypatch):
    """Reordered CUSUM (before the PSI state save) must not be clobbered."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = list(np.percentile(ref, np.arange(0, 101, 10)))
    with open(sandbox / 'model_v2.manifest.json', 'w') as f:
        json.dump({'holdout': {'pred_deciles': edges, 'hit_rate': 0.55}}, f)
    monitor_drift.log_predictions(
        '', {f'S{i}': v for i, v in enumerate(rng.normal(0, 1, 200))})
    trades = [{'ts': '2026-06-01T00:00:00+00:00', 'action': 'sell', 'exit': 1,
               'pnl_pct': 1.0, 'estimated': False}]
    import trade_memory
    monkeypatch.setattr(trade_memory, '_load', lambda: {'BTC/USD': trades})

    r = monitor_drift.run_check('', 'crypto')
    assert r is not None and r['level'] == 'ok'
    st = monitor_drift._load_state()['crypto']
    assert 'cusum' in st and 'cusum_last_ts' in st   # CUSUM keys survive
    assert st['last_psi'] == r['psi']                # PSI keys written too


def test_live_outcomes_logs_import_failure(monkeypatch, capsys):
    """Import failure must be loud — silence looks like 'no trades'."""
    monkeypatch.setitem(sys.modules, 'trade_memory', None)
    assert monitor_drift._live_outcomes('crypto', None) == []
    assert 'trade_memory unavailable' in capsys.readouterr().out


def test_history_lock_sidecar_created(sandbox):
    monitor_drift.log_predictions('', {'BTC/USD': 0.5})
    assert (sandbox / 'pred_history.jsonl.lock').exists()
    assert list(monitor_drift.load_recent_predictions('')) == [0.5]


def test_history_lock_degrades_without_fcntl(sandbox, monkeypatch):
    """Best-effort: no fcntl (non-POSIX) must not drop writes or prunes."""
    monkeypatch.setattr(monitor_drift, 'fcntl', None)
    monitor_drift.log_predictions('', {'BTC/USD': 0.5})
    assert len(monitor_drift.load_recent_predictions('')) == 1
    monitor_drift.prune_history('')
    assert len(monitor_drift.load_recent_predictions('')) == 1


def test_log_predictions_still_never_raises(sandbox, monkeypatch):
    """The lock wrapper must not break the never-raise contract."""
    monkeypatch.setattr(monitor_drift, 'history_file',
                        lambda p: Path('/nonexistent-dir/x.jsonl'))
    monitor_drift.log_predictions('', {'BTC/USD': 0.5})  # swallowed


def test_module_docstring_wiring_current():
    doc = monitor_drift.__doc__
    assert '--if-drift' not in doc          # flag never existed
    assert '_maybe_run_drift_check' in doc  # actual in-process daily runner
    assert '_check_drift_trigger' in doc    # actual flag consumer


# ---------------------------------------------------------------------------
# gpu_lock
# ---------------------------------------------------------------------------

@pytest.fixture
def lock_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_lock, '_LOCK_FILE', tmp_path / '.gpu.lock')
    monkeypatch.setattr(gpu_lock, '_INFO_FILE', tmp_path / '.info.json')
    return tmp_path


class _FlockStub:
    """flock that simulates a held lock, then ^C while parked waiting."""
    LOCK_EX = gpu_lock.fcntl.LOCK_EX
    LOCK_SH = gpu_lock.fcntl.LOCK_SH
    LOCK_NB = gpu_lock.fcntl.LOCK_NB
    LOCK_UN = gpu_lock.fcntl.LOCK_UN

    @staticmethod
    def flock(fd, op):
        if op == _FlockStub.LOCK_EX | _FlockStub.LOCK_NB:
            raise BlockingIOError          # someone holds the lock
        if op == _FlockStub.LOCK_EX:
            raise KeyboardInterrupt        # user ^C during the blocking wait
        # LOCK_SH / LOCK_UN: no-op


def test_interrupted_waiter_preserves_holder_info(lock_sandbox, monkeypatch,
                                                  capsys):
    """A waiter killed mid-flock must not delete the real holder's info
    file or print a false 'Released' (finally ran full cleanup before)."""
    holder = {'owner': 'real_holder', 'pid': 1234, 'acquired_at': 'x'}
    (lock_sandbox / '.info.json').write_text(json.dumps(holder))
    monkeypatch.setattr(gpu_lock, 'fcntl', _FlockStub)

    with pytest.raises(KeyboardInterrupt):
        with gpu_lock.acquire_for_training('waiter'):
            pass

    assert gpu_lock.get_lock_info() == holder      # info survives
    out = capsys.readouterr().out
    assert "Released by 'waiter'" not in out
    assert 'waiting' in out


def test_status_tolerates_incomplete_info(lock_sandbox):
    """Partial/foreign info files must not KeyError out of a status call."""
    with gpu_lock.acquire_for_training('partial_test'):
        (lock_sandbox / '.info.json').write_text(
            json.dumps({'owner': 'partial_test'}))
        status = gpu_lock.gpu_lock_status()
    assert 'locked' in status and 'partial_test' in status and '?' in status


def test_probe_does_not_collide_with_another_probe(lock_sandbox):
    """SH probes coexist: a concurrent probe must not read as 'busy'
    (the old EX probe made two probes report each other as holders)."""
    import fcntl as real_fcntl
    (lock_sandbox / '.gpu.lock').touch()
    fd = open(lock_sandbox / '.gpu.lock', 'r')
    try:
        real_fcntl.flock(fd, real_fcntl.LOCK_SH)   # a probe mid-flight
        assert gpu_lock.is_gpu_free() is True
    finally:
        real_fcntl.flock(fd, real_fcntl.LOCK_UN)
        fd.close()


def test_probe_still_detects_exclusive_holder(lock_sandbox):
    with gpu_lock.acquire_for_training('holder'):
        assert gpu_lock.is_gpu_free() is False
    assert gpu_lock.is_gpu_free() is True


def test_write_info_warns_on_failure(lock_sandbox, monkeypatch, capsys):
    monkeypatch.setattr(gpu_lock, '_INFO_FILE',
                        lock_sandbox / 'nodir' / 'info.json')
    gpu_lock._write_info('x')  # must not raise
    assert '[GPU-LOCK] warn' in capsys.readouterr().out


def test_write_info_atomic_and_clear_info_logs():
    assert 'os.replace' in inspect.getsource(gpu_lock._write_info)
    assert 'warn' in inspect.getsource(gpu_lock._clear_info)


def test_lock_fd_global_removed():
    assert not hasattr(gpu_lock, '_lock_fd')
    assert '_lock_fd' not in Path(gpu_lock.__file__).read_text()


def test_probe_uses_shared_lock():
    assert 'LOCK_SH' in inspect.getsource(gpu_lock.is_gpu_free)


def test_gpu_lock_docstrings_updated():
    assert "device = 'cuda'" not in gpu_lock.__doc__  # bots are CPU-always
    assert 'non-reentrant' in gpu_lock.__doc__
    assert 'RuntimeError' not in gpu_lock.acquire_for_training.__doc__
    assert 'GUI' not in inspect.getsource(gpu_lock.gpu_lock_status)


# ---------------------------------------------------------------------------
# hw_monitor
# ---------------------------------------------------------------------------

@pytest.fixture
def hw_reset():
    """Reset hw_monitor module caches around each test (mirrors
    tests/test_hw_monitor.py plus the new one-shot warn flag)."""
    def _reset():
        hw_monitor._zone_scan_done = False
        hw_monitor._zone_path_cache = None
        hw_monitor._temp_cache = (0.0, None)
        hw_monitor._warned_unavailable = False
    _reset()
    yield
    _reset()


def test_zone_scan_done_flag_set_after_scan():
    """The done-flag must be set AFTER the scan populates the cache, so a
    second combined-bots thread entering mid-scan redoes the idempotent
    scan instead of reading a spurious None."""
    src = inspect.getsource(hw_monitor._find_gpu_thermal_zone)
    assert src.count('_zone_scan_done = True') == 1
    assert (src.index('_zone_scan_done = True')
            > src.index('for name in sorted'))


def test_zone_scan_still_memoized(hw_reset, monkeypatch):
    calls = {'n': 0}

    def fake_listdir(d):
        calls['n'] += 1
        raise OSError('no sysfs')

    monkeypatch.setattr(hw_monitor.os, 'listdir', fake_listdir)
    monkeypatch.setattr(hw_monitor.os.path, 'exists', lambda p: False)
    assert hw_monitor._find_gpu_thermal_zone() is None
    assert hw_monitor._find_gpu_thermal_zone() is None
    assert calls['n'] == 1  # second call served from the cache


def test_one_shot_warning_when_sensor_unavailable(hw_reset, monkeypatch,
                                                  caplog):
    """First None warns once (thermal protection silently off otherwise);
    it must NOT log every 30s cycle."""
    monkeypatch.setattr(hw_monitor, '_find_gpu_thermal_zone', lambda: None)
    with caplog.at_level(logging.WARNING, logger='hw_monitor'):
        assert hw_monitor.get_gpu_temp() is None
        assert hw_monitor.get_gpu_temp() is None
    warns = [r for r in caplog.records
             if 'GPU temperature unavailable' in r.getMessage()]
    assert len(warns) == 1


def test_one_shot_warning_on_read_error(hw_reset, monkeypatch, caplog):
    monkeypatch.setattr(hw_monitor, '_find_gpu_thermal_zone',
                        lambda: '/nonexistent/thermal/temp')
    with caplog.at_level(logging.WARNING, logger='hw_monitor'):
        assert hw_monitor.get_gpu_temp() is None
        assert hw_monitor.get_gpu_temp() is None
    warns = [r for r in caplog.records
             if 'GPU temperature unavailable' in r.getMessage()]
    assert len(warns) == 1


def test_hw_monitor_cosmetic_pins():
    src = Path(hw_monitor.__file__).read_text()
    # FileNotFoundError dropped from the tuple (subclass of OSError)
    assert 'except (ValueError, OSError)' in src
    assert 'except (FileNotFoundError, ValueError, OSError)' not in src
    # f-prefix removed from the placeholder-free print
    assert 'print("[HW] Cannot read GPU temp' in src
    # __main__: a legitimate 0.0C reading is not 'unavailable'
    assert 'if temp is not None' in src
    # docstrings document the failure returns / constraints
    assert '(None, None)' in hw_monitor.get_ram_usage.__doc__
    assert 'Diagnostic only' in hw_monitor.is_gpu_available.__doc__
    assert '_TEMP_CACHE_TTL' in hw_monitor.wait_for_cool_gpu.__doc__
