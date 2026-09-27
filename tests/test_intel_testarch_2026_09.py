"""INTEL W1 (2026-09 Jetson campaign) — the repo-root hygiene instrument.

tests/conftest.py snapshots the regular files directly in <repo>/ and <repo>/tests/
at session start/finish and prints a `repo-root hygiene` terminal section
(`clean`, or one `NEW <path>` / `MOD <path>` line per created/modified file);
TRADER_TESTS_HYGIENE_JSON mirrors it to JSON and TRADER_TESTS_STRICT_CLEAN=1
turns a dirty root into exit status 1. scripts/ab_check.sh echoes the section
and honours strict mode only when the caller sets it.

Pure helpers are imported from the REAL conftest by path; the hook behaviour is
exercised end-to-end in a throwaway mini-project whose conftest delegates to
those same helpers, so nothing here is a re-implementation. pytest + stdlib
only (dev-Mac safe); the subprocess runs never touch the real repo root.
"""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REAL_CONFTEST = REPO / 'tests' / 'conftest.py'
AB_CHECK = REPO / 'scripts' / 'ab_check.sh'

_HYGIENE_ENV = ('TRADER_TESTS_HYGIENE_JSON', 'TRADER_TESTS_STRICT_CLEAN')


def _load_real_conftest():
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            '_intel_w1_real_conftest', str(REAL_CONFTEST))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


@pytest.fixture(scope='module')
def rc():
    return _load_real_conftest()


# ---------------------------------------------------------------------------
# (iv) pure helpers
# ---------------------------------------------------------------------------

def test_diff_new_modified_unchanged_deleted(rc):
    before = {'a.txt': (1, 10), 'b.txt': (1, 10), 'gone.txt': (1, 1),
              'tests/t.py': (5, 5)}
    after = {'a.txt': (1, 10),            # unchanged
             'b.txt': (2, 10),            # mtime changed
             'tests/t.py': (5, 6),        # size changed
             'new.json': (3, 2)}          # created; gone.txt deleted -> ignored
    assert rc._hygiene_diff(before, after) == [
        ('MOD', 'b.txt'), ('NEW', 'new.json'), ('MOD', 'tests/t.py')]


def test_diff_identical_is_empty(rc):
    snap = {'x': (1, 1), 'tests/y': (2, 2)}
    assert rc._hygiene_diff(snap, dict(snap)) == []
    assert rc._hygiene_diff({}, {}) == []


def test_snapshot_top_level_only_and_skips(rc, tmp_path):
    (tmp_path / 'top.txt').write_text('x')
    (tmp_path / 'mod.pyc').write_bytes(b'\0')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'inner.json').write_text('{}')
    (tmp_path / 'tests' / '__pycache__').mkdir()
    (tmp_path / 'tests' / '__pycache__' / 'c.pyc').write_bytes(b'\0')
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs' / 'trader.log').write_text('deep')   # not recursed
    (tmp_path / '.pytest_cache').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'top.txt')
    snap = rc._hygiene_snapshot(rc._hygiene_dirs(tmp_path), tmp_path)
    assert sorted(snap) == ['tests/inner.json', 'top.txt']
    st = (tmp_path / 'top.txt').stat()
    assert snap['top.txt'] == (st.st_mtime_ns, st.st_size)


def test_snapshot_missing_dir_is_empty(rc, tmp_path):
    assert rc._hygiene_snapshot([tmp_path / 'nope'], tmp_path) == {}


def test_allowlist_is_empty(rc):
    # Nothing is hidden by default; an entry needs a proven file:line writer.
    assert rc._HYGIENE_ALLOWLIST == frozenset()


# ---------------------------------------------------------------------------
# (i)-(iii) end-to-end through a subprocess pytest in a mini-project
# ---------------------------------------------------------------------------

_MINI_CONFTEST = '''
import importlib.util, sys
from pathlib import Path

_saved = list(sys.path)
_spec = importlib.util.spec_from_file_location('_real_trader_conftest', {real!r})
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)
sys.path[:] = _saved
_ROOT = Path(__file__).resolve().parent.parent


def pytest_sessionstart(session):
    _rc._hygiene_sessionstart(session, _ROOT)


import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    _rc._hygiene_sessionfinish(session, _ROOT)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    _rc._hygiene_terminal_summary(terminalreporter, config)
'''


def _mini_project(tmp_path, test_body):
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    (proj / 'pytest.ini').write_text('[pytest]\n')
    (proj / 'existing.txt').write_text('v1\n')
    (proj / 'tests' / 'conftest.py').write_text(
        _MINI_CONFTEST.format(real=str(REAL_CONFTEST)))
    (proj / 'tests' / 'test_mini.py').write_text(
        'from pathlib import Path\n'
        'ROOT = Path(__file__).resolve().parent.parent\n\n'
        'def test_it():\n' + textwrap.indent(textwrap.dedent(test_body), '    '))
    return proj


def _run_mini(proj, **env_extra):
    env = {k: v for k, v in os.environ.items()
           if k not in _HYGIENE_ENV and k not in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS')}
    env['PY_COLORS'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)


_SECTION_PREFIXES = ('NEW ', 'MOD ', 'ALW ', 'strict ', 'could not write ')


def _section(stdout):
    lines = stdout.splitlines()
    idx = [i for i, ln in enumerate(lines) if 'repo-root hygiene' in ln]
    assert idx, stdout
    out = []
    for ln in lines[idx[0] + 1:]:
        if not ln.startswith(_SECTION_PREFIXES) and ln != 'clean':
            break
        out.append(ln)
    return out


def test_new_file_reported_and_json_written(tmp_path):
    proj = _mini_project(tmp_path, "(ROOT / 'polluter.json').write_text('{}')\n")
    js = tmp_path / 'out' / 'hygiene.json'
    proc = _run_mini(proj, TRADER_TESTS_HYGIENE_JSON=str(js))
    assert proc.returncode == 0, proc.stdout + proc.stderr   # report-only default
    assert _section(proc.stdout) == ['NEW polluter.json']
    data = json.loads(js.read_text())
    assert data['clean'] is False and data['strict'] is False
    assert data['entries'] == [{'kind': 'NEW', 'path': 'polluter.json'}]
    assert data['exitstatus_forced'] is False


def test_modification_reported_as_mod(tmp_path):
    proj = _mini_project(tmp_path, """\
        p = ROOT / 'existing.txt'
        p.write_text(p.read_text() + 'v2\\n')
        (ROOT / 'tests' / 'side.db').write_text('x')
        """)
    proc = _run_mini(proj)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _section(proc.stdout) == ['MOD existing.txt', 'NEW tests/side.db']


def test_clean_session_says_clean_and_writes_nothing(tmp_path):
    proj = _mini_project(tmp_path, "assert True\n")
    before = sorted(p.name for p in proj.iterdir())
    proc = _run_mini(proj, TRADER_TESTS_STRICT_CLEAN='1')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _section(proc.stdout) == ['clean']
    assert sorted(p.name for p in proj.iterdir()) == before


def test_strict_env_fails_dirty_session(tmp_path):
    proj = _mini_project(tmp_path, "(ROOT / 'polluter.json').write_text('{}')\n")
    proc = _run_mini(proj, TRADER_TESTS_STRICT_CLEAN='1')
    assert proc.returncode == int(pytest.ExitCode.TESTS_FAILED), proc.stdout + proc.stderr
    sec = _section(proc.stdout)
    assert sec[0] == 'NEW polluter.json'
    assert any('TRADER_TESTS_STRICT_CLEAN=1' in ln and 'exit status set to 1' in ln
               for ln in sec)
    assert '1 passed' in proc.stdout   # the test itself passed; hygiene failed the run


# ---------------------------------------------------------------------------
# scripts/ab_check.sh surfaces the block; strict only when the caller asks
# ---------------------------------------------------------------------------

def _ab_run(tmp_path, block, **env_extra):
    stub = tmp_path / 'fake_pytest.sh'
    stub.write_text('#!/bin/sh\ncat <<"EOF"\n'
                    + block
                    + '== 1600 passed in 1.0s ==\nEOF\nexit 0\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {k: v for k, v in os.environ.items() if k not in _HYGIENE_ENV}
    env.update({'AB_CHECK_PYTEST': str(stub), 'AB_CHECK_MIN_PASSED': '10'})
    env.update(env_extra)
    return subprocess.run(['sh', str(AB_CHECK)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=60)


_DIRTY = '====== repo-root hygiene ======\nNEW llm_cost.json\nMOD tests/x.py\n'
_CLEAN = '====== repo-root hygiene ======\nclean\n'


def test_ab_check_echoes_block_report_only_by_default(tmp_path):
    proc = _ab_run(tmp_path, _DIRTY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'NEW llm_cost.json' in proc.stdout and 'MOD tests/x.py' in proc.stdout
    assert 'ab_check: PASS' in proc.stdout


def test_ab_check_strict_fails_dirty_root(tmp_path):
    proc = _ab_run(tmp_path, _DIRTY, TRADER_TESTS_STRICT_CLEAN='1')
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'ab_check: FAIL' in proc.stdout
    assert 'ab_check: PASS' not in proc.stdout


def test_ab_check_strict_passes_clean_root(tmp_path):
    proc = _ab_run(tmp_path, _CLEAN, TRADER_TESTS_STRICT_CLEAN='1')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'clean' in proc.stdout and 'ab_check: PASS' in proc.stdout
