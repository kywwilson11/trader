"""Shared fixtures for the trader test suite."""

import json
import os

# ---------------------------------------------------------------------------
# Test-session log directory (2026-09 INTEL W18). MUST stay the first thing
# after `import os`, ahead of every repo import (llm_client is imported further
# down this very file, and 22 repo modules call log_config.get_logger at module
# scope). log_config._log_paths() (log_config.py:102-135) reads TRADER_LOG_DIR
# at the first _setup(), so every test-process record then goes to a
# per-session temp dir instead of the PRODUCTION <repo>/logs/trader.log that
# the live bots and Jetson forensics use. A non-empty value set by the caller
# wins (setdefault semantics; an EMPTY value counts as unset, because log_config
# treats '' as "use the production default"). mkdtemp runs only when needed.
# The directory is left in place (it is tmp); its path is printed as one
# `test logs:` line in the repo-root hygiene section below. Production is
# unchanged: nothing outside tests sets the variable.
# ---------------------------------------------------------------------------
import tempfile

_LOG_DIR_ENV = 'TRADER_LOG_DIR'
if os.environ.get(_LOG_DIR_ENV):
    _TEST_LOG_DIR_SOURCE = 'caller'
else:
    os.environ[_LOG_DIR_ENV] = tempfile.mkdtemp(prefix='trader-test-logs-')
    _TEST_LOG_DIR_SOURCE = 'conftest'
_TEST_LOG_DIR = os.environ[_LOG_DIR_ENV]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root and scripts/ are on sys.path so imports work
_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / 'scripts'))

# Exclude the headline test suite from pytest (uses its own custom runner)
collect_ignore = [str(Path(__file__).resolve().parent / 'test_sentiment_headlines.py')]


@pytest.fixture
def sample_ohlcv_df():
    """120-row OHLCV DataFrame with realistic random walk prices."""
    np.random.seed(42)
    n = 120
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 50000, size=n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


@pytest.fixture
def tmp_json_file(tmp_path):
    """Factory fixture: write a dict to a temp JSON file and return the path."""
    def _make(data, filename="test.json"):
        p = tmp_path / filename
        p.write_text(json.dumps(data))
        return p
    return _make



# ---------------------------------------------------------------------------
# Suite-wide LLM cost-ledger sandbox (2026-09 INTEL W13).
#
# llm_client persists the shared spend ledger through ONE module constant,
# llm_client._COST_FILE (llm_client.py:264); the flock sidecar (<_COST_FILE>.lock,
# :286), the atomic-write temp (<_COST_FILE>.tmp, :863) and the rollover history
# (llm_cost_history.jsonl, _cost_history_path :881) are all derived from it at
# CALL time. Six read-shaped getters reach _maybe_reset_quota, so any test that
# touches them used to open <repo>/llm_cost.json.lock for write and, on a new
# PT day, rewrite <repo>/llm_cost.json. This autouse fixture points the constant
# at a fresh, EXISTING per-test directory and resets the two in-memory ledger
# globals, all through monkeypatch (restored after every test).
#
#  - The ledger code itself is untouched: writes really happen (the directory
#    exists, so no fail-soft OSError path is taken) and can be read back via
#    llm_client._COST_FILE or the fixture's return value.
#  - A test (or a module/requested fixture) that sets _COST_FILE itself runs
#    AFTER this autouse fixture on the same monkeypatch, so its value wins.
#  - The directory is NOT the test's tmp_path, so tmp_path listings are unchanged.
#  - The real module object is captured at conftest import (llm_client imports
#    only stdlib + llm_config: no file, network or subprocess activity at
#    import), so a test that swaps a stub into sys.modules['llm_client'] cannot
#    divert the sandbox. If llm_client cannot be imported, the fixture is a
#    no-op (returns None) and collection is unaffected.
#  - In-process only: a test that spawns a separate Python process which
#    imports llm_client is not covered.
#  - Also (INTEL W18/W19) swaps in a FRESH llm_client._call_meta_tls
#    (threading.local, llm_client.py:265) per test, so get_last_call_meta()
#    starts at None in every thread and a stubbed test never reads transport
#    meta left by an earlier real-client test; restored by monkeypatch.
# ---------------------------------------------------------------------------
try:
    import llm_client as _LLM_CLIENT
except Exception:  # missing dep / broken module: degrade to a no-op
    _LLM_CLIENT = None


@pytest.fixture(autouse=True)
def _llm_cost_ledger_sandbox(monkeypatch, tmp_path_factory):
    """Per-test sandbox dir holding llm_client's ledger files, or None."""
    if _LLM_CLIENT is None:
        return None
    import tempfile
    base = tmp_path_factory.getbasetemp() / 'llm_cost_ledger'
    base.mkdir(exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix='t', dir=str(base)))
    monkeypatch.setattr(_LLM_CLIENT, '_COST_FILE', str(sandbox / 'llm_cost.json'))
    monkeypatch.setattr(_LLM_CLIENT, '_cost_reset_date', '')
    monkeypatch.setattr(_LLM_CLIENT, '_daily_cost', 0.0)
    if hasattr(_LLM_CLIENT, '_call_meta_tls'):
        import threading
        monkeypatch.setattr(_LLM_CLIENT, '_call_meta_tls', threading.local())
    return sandbox

# ---------------------------------------------------------------------------
# Repo-root hygiene instrument (2026-09 INTEL W1).
#
# Snapshots the regular files directly in <repo> and directly in <repo>/tests
# (no recursion) at session start, re-snapshots at session finish, and prints a
# `repo-root hygiene` terminal section listing every file CREATED (`NEW `) or
# MODIFIED (`MOD `, mtime or size changed) during the run — or `clean`.
# Deleted files are ignored. Subdirectories (logs/, models/, journals/, ...)
# are deliberately out of scope.
#
#   TRADER_TESTS_HYGIENE_JSON=<path>  also write the result as JSON there
#                                     (point it OUTSIDE the repo; nothing is
#                                     written to disk when unset).
#   TRADER_TESTS_STRICT_CLEAN=1       opt-in enforcement: a non-empty list
#                                     turns an otherwise-passing session's exit
#                                     status into 1 (pytest.ExitCode.TESTS_FAILED).
#                                     Default: report-only, exit status untouched.
#
# The helpers are pure and importable so tests/test_intel_testarch_2026_09.py
# can drive them against a throwaway mini-project.
# ---------------------------------------------------------------------------
_HYGIENE_JSON_ENV = 'TRADER_TESTS_HYGIENE_JSON'
_HYGIENE_STRICT_ENV = 'TRADER_TESTS_STRICT_CLEAN'
_HYGIENE_SKIP_NAMES = frozenset({'__pycache__', '.pytest_cache'})
# Paths (relative, posix) a test is PROVEN to legitimately need to write. A
# path may be added only with a file:line citation of the writing code path.
# Allowlisted paths are still printed (as `ALW `) — they only stop tripping
# strict mode. Intentionally empty: the report shows the truth.
_HYGIENE_ALLOWLIST = frozenset()


def _hygiene_dirs(root):
    """The watched directories: the root itself and root/tests (top level only)."""
    root = Path(root)
    return [root, root / 'tests']


def _hygiene_snapshot(dirs, base):
    """{relative posix path: (mtime_ns, size)} for regular files directly in dirs.

    No recursion; symlinks, directories, __pycache__/.pytest_cache and *.pyc are
    skipped; unreadable/missing directories contribute nothing.
    """
    snap = {}
    for d in dirs:
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for entry in it:
                name = entry.name
                if name in _HYGIENE_SKIP_NAMES or name.endswith('.pyc'):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                rel = Path(os.path.relpath(entry.path, base)).as_posix()
                snap[rel] = (st.st_mtime_ns, st.st_size)
    return snap


def _hygiene_diff(before, after):
    """Sorted [('NEW'|'MOD', path)] — created or changed files; deletions ignored."""
    out = []
    for path in sorted(after):
        if path not in before:
            out.append(('NEW', path))
        elif after[path] != before[path]:
            out.append(('MOD', path))
    return out


def _hygiene_sessionstart(session, root):
    config = session.config
    if hasattr(config, 'workerinput'):  # xdist worker: the controller reports
        return
    config._trader_hygiene_root = Path(root)
    config._trader_hygiene_before = _hygiene_snapshot(_hygiene_dirs(root), root)


def _hygiene_sessionfinish(session, root):
    config = session.config
    before = getattr(config, '_trader_hygiene_before', None)
    if before is None:
        return
    after = _hygiene_snapshot(_hygiene_dirs(root), root)
    entries = [('ALW' if p in _HYGIENE_ALLOWLIST else k, p)
               for k, p in _hygiene_diff(before, after)]
    dirty = [e for e in entries if e[0] != 'ALW']
    strict = os.environ.get(_HYGIENE_STRICT_ENV, '') == '1'
    forced = False
    if strict and dirty and session.exitstatus in (
            pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        forced = True
    result = {
        'root': str(root),
        'watched': [Path(os.path.relpath(d, root)).as_posix()
                    for d in _hygiene_dirs(root)],
        'clean': not dirty,
        'strict': strict,
        'exitstatus_forced': forced,
        'entries': [{'kind': k, 'path': p} for k, p in entries],
    }
    config._trader_hygiene_result = result
    out_path = os.environ.get(_HYGIENE_JSON_ENV)
    if out_path:
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(result, indent=2) + '\n')
        except OSError as exc:  # measurement must never break the run
            result['json_error'] = repr(exc)


def _hygiene_terminal_summary(terminalreporter, config):
    result = getattr(config, '_trader_hygiene_result', None)
    if result is None:
        return
    tr = terminalreporter
    tr.write_sep('=', 'repo-root hygiene')
    if not result['entries']:
        tr.write_line('clean')
    for e in result['entries']:
        tr.write_line('%s %s' % (e['kind'], e['path']))
    if result['strict'] and not result['clean']:
        tr.write_line('strict (%s=1): repo root not clean%s' % (
            _HYGIENE_STRICT_ENV,
            ' -> session exit status set to 1 (TESTS_FAILED)'
            if result['exitstatus_forced'] else ' (exit status already non-zero)'))
    if 'json_error' in result:
        tr.write_line('could not write %s: %s' % (_HYGIENE_JSON_ENV, result['json_error']))
    # INTEL W18 lines come LAST: the section parsers (ab_check.sh's awk block,
    # test_intel_testarch's _section) stop at the first line outside the
    # clean/NEW/MOD/ALW/strict vocabulary, so these never disturb them.
    log_dir = getattr(config, '_trader_test_log_dir', None)
    if log_dir is not None:
        tr.write_line('test logs: %s (%s, set by %s)' % (
            log_dir, _LOG_DIR_ENV, getattr(config, '_trader_test_log_dir_source', '?')))
    verdict = getattr(config, '_trader_prodlog_verdict', None)
    if verdict is not None:
        tr.write_line('production log untouched: %s' % verdict)


# ---------------------------------------------------------------------------
# Production-log guard (2026-09 INTEL W18). Records <root>/logs/trader.log's
# (inode, mtime_ns, size) at session start and judges it at session finish:
#   yes                 unchanged
#   no (...)            changed with no live foreign writer holding it open,
#                       or THIS process has a logging handler on it
#   unattributable (...) changed while other processes (the live bots on the
#                       Jetson) held it open and this process has no handler
#                       on it -- the change cannot be pinned on the session
# Report-only (one line at the end of the repo-root hygiene section); the exit
# status is never touched. Holders are found via /proc/<pid>/fd (Linux);
# elsewhere the list is empty, so any change reads `no`.
# ---------------------------------------------------------------------------
def _prodlog_path(root):
    return Path(root) / 'logs' / 'trader.log'


def _prodlog_stat(path):
    """(st_ino, st_mtime_ns, st_size) of path, or None when it is absent."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _prodlog_holders(path, exclude_pid=None):
    """Sorted pids (other than exclude_pid) holding path open, via /proc."""
    try:
        target = os.path.realpath(path)
        pids = [p for p in os.listdir('/proc') if p.isdigit()]
    except OSError:
        return []
    out = set()
    for p in pids:
        pid = int(p)
        if pid == exclude_pid:
            continue
        fd_dir = '/proc/%s/fd' % p
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink('%s/%s' % (fd_dir, fd)) == target:
                    out.add(pid)
                    break
            except OSError:
                continue
    return sorted(out)


def _prodlog_own_handlers(path):
    """baseFilenames of logging.FileHandlers in THIS process pointing at path."""
    import logging
    target = os.path.realpath(path)
    loggers = [logging.getLogger()] + [
        lg for lg in list(logging.Logger.manager.loggerDict.values())
        if isinstance(lg, logging.Logger)]
    hits = set()
    for lg in loggers:
        for h in list(getattr(lg, 'handlers', ())):
            base = getattr(h, 'baseFilename', None)
            if base and os.path.realpath(base) == target:
                hits.add(base)
    return sorted(hits)


def _prodlog_verdict(before, after, holders, own):
    """Pure: the `production log untouched:` value (see the block comment)."""
    if own:
        return 'no (this process has a logging handler on it: %s)' % ', '.join(own)
    if before == after:
        return 'yes'
    if holders:
        return ('unattributable (changed while live writer pid(s) %s held it open; '
                'this process has no handler on it)' % ','.join(map(str, holders)))
    return 'no (%s -> %s)' % (before, after)


def _logdir_sessionstart(session, root):
    config = session.config
    config._trader_test_log_dir = _TEST_LOG_DIR
    config._trader_test_log_dir_source = _TEST_LOG_DIR_SOURCE
    config._trader_prodlog_path = _prodlog_path(root)
    config._trader_prodlog_before = _prodlog_stat(config._trader_prodlog_path)


def _logdir_sessionfinish(session, root):
    config = session.config
    path = getattr(config, '_trader_prodlog_path', None)
    if path is None:
        return
    try:
        config._trader_prodlog_verdict = _prodlog_verdict(
            config._trader_prodlog_before, _prodlog_stat(path),
            _prodlog_holders(path, exclude_pid=os.getpid()),
            _prodlog_own_handlers(path))
    except Exception as exc:  # measurement must never break the run
        config._trader_prodlog_verdict = 'unknown (%r)' % (exc,)


def pytest_sessionstart(session):
    _hygiene_sessionstart(session, Path(_ROOT))
    _logdir_sessionstart(session, Path(_ROOT))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    _hygiene_sessionfinish(session, Path(_ROOT))
    _logdir_sessionfinish(session, Path(_ROOT))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    _hygiene_terminal_summary(terminalreporter, config)
