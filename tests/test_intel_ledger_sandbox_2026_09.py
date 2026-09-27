"""INTEL W13 (2026-09-27): the suite-wide LLM cost-ledger sandbox.

tests/conftest.py's autouse `_llm_cost_ledger_sandbox` points
llm_client._COST_FILE (llm_client.py:264; the .lock/.tmp/history paths derive
from it at call time) at a fresh, existing per-test directory and resets the
in-memory ledger globals, all via monkeypatch.

These tests prove the sandbox does NOT mask ledger behaviour: the real ledger
API writes (lock, atomic replace, rollover + history append) land in the
sandbox and read back; nothing under the production module's directory is
opened or changed; state does not leak between tests; a test's own
_COST_FILE wins; and conftest still imports (fixture = no-op) when llm_client
cannot be imported.

Safety: every write below is preceded by `_guard()`, which fails the test
BEFORE any I/O if _COST_FILE still points at the production root — so a
regressed/missing sandbox fails here instead of touching the real ledger.
"""
import builtins
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

llm_client = pytest.importorskip("llm_client")

REAL_CONFTEST = Path(__file__).resolve().parent / 'conftest.py'
# The exact expression llm_client.py:264 uses for the production default.
PROD_DIR = os.path.dirname(os.path.abspath(llm_client.__file__))
ROOT_LEDGER = [os.path.join(PROD_DIR, n) for n in (
    'llm_cost.json', 'llm_cost.json.lock', 'llm_cost.json.tmp',
    'llm_cost_history.jsonl')]
_MODEL = 'claude-haiku-4-5'
_USAGE = {'promptTokenCount': 100, 'candidatesTokenCount': 20,
          'thoughtsTokenCount': 0}


def _today():
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _guard():
    cf = os.path.abspath(llm_client._COST_FILE)
    assert os.path.dirname(cf) != PROD_DIR, (
        'ledger sandbox inactive: _COST_FILE is the production path %s' % cf)
    return Path(cf).parent


def _stat(paths):
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
            out[p] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            out[p] = None
    return out


def _exercise_ledger(monkeypatch):
    """Drive the REAL ledger API through every write path, recording every
    (path, mode) passed to open()/os.replace meanwhile ('replace' for both
    ends of an os.replace). Returns (opened, spend)."""
    sandbox = _guard()
    opened = []
    real_open, real_replace = builtins.open, os.replace

    def spy_open(file, *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)):
            mode = a[0] if a else k.get('mode', 'r')
            opened.append((os.path.abspath(os.fsdecode(file)), mode))
        return real_open(file, *a, **k)

    def spy_replace(src, dst, *a, **k):
        opened.extend((os.path.abspath(os.fsdecode(p)), 'replace')
                      for p in (src, dst))
        return real_replace(src, dst, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(builtins, 'open', spy_open)
        m.setattr(os, 'replace', spy_replace)
        llm_client._maybe_reset_quota()                    # lock + rollover
        llm_client._record_cost(_MODEL, 0, 0, dict(_USAGE))  # lock+tmp+replace
        spend = llm_client._daily_cost
        # Stale day on disk -> next write rolls over and appends history.
        (sandbox / 'llm_cost.json').write_text(
            json.dumps({'date': '2000-01-01', 'cost': 0.25}))
        llm_client._cost_reset_date = '2000-01-01'  # restored by conftest
        llm_client._record_cost(_MODEL, 0, 0, dict(_USAGE))
    return opened, spend


# --- (i) writes land in the sandbox and read back --------------------------

def test_sandbox_is_active_existing_and_outside_prod(_llm_cost_ledger_sandbox):
    sb = _llm_cost_ledger_sandbox
    assert sb is not None and sb.is_dir() and not any(sb.iterdir())
    assert llm_client._COST_FILE == str(sb / 'llm_cost.json')
    assert sys.modules['llm_client'] is llm_client  # the real module is patched
    assert _guard() == sb
    assert (llm_client._daily_cost, llm_client._cost_reset_date) == (0.0, '')
    assert llm_client._cost_history_path() == str(sb / 'llm_cost_history.jsonl')


def test_real_ledger_writes_land_in_sandbox_and_read_back(monkeypatch):
    sandbox = _guard()
    opened, spend = _exercise_ledger(monkeypatch)
    assert spend > 0
    today = _today()
    # every ledger path the code opened/replaced is inside the sandbox
    # (llm_config.json is also READ, by pricing — not a ledger file)
    ledger = {p for p, _ in opened if os.path.basename(p).startswith('llm_cost')}
    assert ledger and all(os.path.dirname(p) == str(sandbox) for p in ledger)
    paths = {p for p, _ in opened}
    for name in ('llm_cost.json', 'llm_cost.json.lock', 'llm_cost.json.tmp',
                 'llm_cost_history.jsonl'):
        assert str(sandbox / name) in paths, name
    assert sorted(os.listdir(sandbox)) == [
        'llm_cost.json', 'llm_cost.json.lock', 'llm_cost_history.jsonl']
    data = json.loads((sandbox / 'llm_cost.json').read_text())
    # rollover reset the stale day, then the second call's spend was added
    assert data['date'] == today
    assert data['cost'] == pytest.approx(round(spend, 6))
    [line] = (sandbox / 'llm_cost_history.jsonl').read_text().splitlines()
    rec = json.loads(line)
    assert (rec['date'], rec['cost'], rec['src']) == ('2000-01-01', 0.25,
                                                      'file')
    assert llm_client.get_daily_cost()[0] == pytest.approx(spend)


# --- (ii) the production-root ledger family is untouched ---------------------

def test_prod_root_ledger_files_unchanged(monkeypatch):
    before = _stat(ROOT_LEDGER)
    opened, _ = _exercise_ledger(monkeypatch)
    in_prod = [(p, m) for p, m in opened if os.path.dirname(p) == PROD_DIR]
    assert not [p for p, _ in in_prod
                if os.path.basename(p).startswith('llm_cost')]
    # nothing at all is opened for writing in the production directory
    assert not [pm for pm in in_prod if pm[1] == 'replace'
                or any(c in pm[1] for c in 'wax+')]
    assert _stat(ROOT_LEDGER) == before


# --- (iii) per-test: no state leaks between tests ----------------------------

_SEEN = {}


def test_isolation_first_writes(monkeypatch):
    _guard()
    llm_client._record_cost(_MODEL, 0, 0, dict(_USAGE))
    assert llm_client._daily_cost > 0
    _SEEN['file'] = llm_client._COST_FILE
    _SEEN['cost'] = llm_client._daily_cost


def test_isolation_second_starts_clean():
    sandbox = _guard()
    assert (llm_client._daily_cost, llm_client._cost_reset_date) == (0.0, '')
    assert list(sandbox.iterdir()) == []
    assert llm_client.get_daily_cost()[0] == 0.0
    if not _SEEN:
        pytest.skip('run together with test_isolation_first_writes')
    assert llm_client._COST_FILE != _SEEN['file']
    prev = json.loads(Path(_SEEN['file']).read_text())
    assert prev['cost'] == pytest.approx(round(_SEEN['cost'], 6))


# --- a test's own _COST_FILE wins over the autouse sandbox -------------------

@pytest.fixture
def own_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, '_COST_FILE', str(tmp_path / 'mine.json'))
    return tmp_path


def test_requested_fixture_override_wins(own_ledger, _llm_cost_ledger_sandbox):
    assert llm_client._COST_FILE == str(own_ledger / 'mine.json')
    llm_client._record_cost(_MODEL, 0, 0, dict(_USAGE))
    assert json.loads((own_ledger / 'mine.json').read_text())['cost'] > 0
    assert list(_llm_cost_ledger_sandbox.iterdir()) == []
    assert 'llm_cost.json' not in os.listdir(own_ledger)  # tmp_path unpolluted


def test_in_body_override_wins(monkeypatch, tmp_path, _llm_cost_ledger_sandbox):
    monkeypatch.setattr(llm_client, '_COST_FILE', str(tmp_path / 'x.json'))
    llm_client._maybe_reset_quota()
    assert json.loads((tmp_path / 'x.json').read_text())['date'] == _today()
    assert list(_llm_cost_ledger_sandbox.iterdir()) == []


# --- (iv) conftest without llm_client; restore after the session ------------

_MINI_CONFTEST = '''
import importlib.util, json, os, sys
_saved = list(sys.path)
_spec = importlib.util.spec_from_file_location('_real_trader_conftest', {real!r})
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)
sys.path[:] = _saved
_llm_cost_ledger_sandbox = _rc._llm_cost_ledger_sandbox


def pytest_sessionfinish(session, exitstatus):
    m = _rc._LLM_CLIENT
    out = None if m is None else [m._COST_FILE, m._daily_cost, m._cost_reset_date]
    with open(os.environ['W13_OUT'], 'w') as f:
        json.dump(out, f)
'''

_MINI_TEST_OK = '''
import json, os
import llm_client

def test_a(_llm_cost_ledger_sandbox):
    assert llm_client._COST_FILE == str(_llm_cost_ledger_sandbox / 'llm_cost.json')
    llm_client._record_cost('claude-haiku-4-5', 0, 0, {{'promptTokenCount': 100,
        'candidatesTokenCount': 20, 'thoughtsTokenCount': 0}})
    assert json.loads(open(llm_client._COST_FILE).read())['cost'] > 0

def test_b(_llm_cost_ledger_sandbox):
    assert llm_client._daily_cost == 0.0 and llm_client._cost_reset_date == ''
    assert os.listdir(_llm_cost_ledger_sandbox) == []
'''

_MINI_TEST_BLOCKED = '''
import pytest

def test_noop(_llm_cost_ledger_sandbox):
    assert _llm_cost_ledger_sandbox is None
    with pytest.raises(ImportError):
        import llm_client  # noqa: F401
'''


def _run_mini(tmp_path, test_src, block=None):
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    (proj / 'pytest.ini').write_text('[pytest]\n')
    (proj / 'tests' / 'conftest.py').write_text(
        _MINI_CONFTEST.format(real=str(REAL_CONFTEST)))
    (proj / 'tests' / 'test_mini.py').write_text(textwrap.dedent(test_src))
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONPATH')
           and not k.startswith('TRADER_TESTS_')}
    env.update(PY_COLORS='0', PYTHONDONTWRITEBYTECODE='1',
               W13_OUT=str(tmp_path / 'out.json'))
    if block:
        site = tmp_path / 'site'
        site.mkdir()
        (site / 'sitecustomize.py').write_text(
            'import sys\nsys.modules[%r] = None\n' % block)
        env['PYTHONPATH'] = str(site)
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         '--basetemp', str(tmp_path / 'bt'), 'tests'],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=180)
    return proc, json.loads((tmp_path / 'out.json').read_text())


@pytest.mark.parametrize('block', ['llm_client', 'llm_config'])
def test_conftest_imports_and_noops_without_llm_client(tmp_path, block):
    proc, out = _run_mini(tmp_path, _MINI_TEST_BLOCKED, block=block)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 passed' in proc.stdout
    assert out is None  # conftest's captured module is None -> no-op


def test_subprocess_session_sandboxes_then_restores(tmp_path):
    proc, out = _run_mini(tmp_path, _MINI_TEST_OK.format())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '2 passed' in proc.stdout
    # after the session every patched global is back to its import value
    assert out == [os.path.join(PROD_DIR, 'llm_cost.json'), 0.0, '']
