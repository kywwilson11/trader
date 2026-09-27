"""INTEL W18 (2026-09-27): pytest logging stays out of the production log.

tests/conftest.py sets TRADER_LOG_DIR to a per-session temp dir (mkdtemp,
prefix 'trader-test-logs-') at MODULE scope, right after `import os` and ahead
of every repo import; log_config._log_paths() (log_config.py:102-135) then
puts every test-process trader.log record there instead of <repo>/logs/
trader.log, which the live bots and Jetson forensics use. A caller-set,
non-empty value wins. conftest also snapshots the production log at session
start and prints `production log untouched: ...` (plus a `test logs: <dir>`
line) at the end of the repo-root hygiene section.

Also pins the W19 add-on to the W13 ledger sandbox: a fresh
llm_client._call_meta_tls per test, so get_last_call_meta() never leaks
across tests.

pytest + stdlib + the stdlib-only log_config/llm_client (dev-Mac safe). The
subprocess checks run a COPY of conftest and log_config in a throwaway
mini-project, so they never touch the real <repo>/logs.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REAL_CONFTEST = REPO / 'tests' / 'conftest.py'
PROD_LOG = REPO / 'logs' / 'trader.log'
ENV = 'TRADER_LOG_DIR'


def _live_conftest(config):
    """The conftest module object pytest actually loaded for this session."""
    for p in config.pluginmanager.get_plugins():
        f = getattr(p, '__file__', None)
        if f and Path(f).resolve() == REAL_CONFTEST.resolve():
            return p
    raise AssertionError('tests/conftest.py is not a loaded plugin')


@pytest.fixture
def rc(request):
    return _live_conftest(request.config)


def _expected_dir():
    raw = os.environ.get(ENV)
    assert raw, '%s is not set in the test process' % ENV
    d = Path(raw)
    return (d if d.is_absolute() else REPO / d).resolve()


def _is_under(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


# ---------------------------------------------------------------------------
# (a) effective log dir
# ---------------------------------------------------------------------------

def test_effective_log_dir_is_session_dir_not_repo_logs(request, rc):
    import log_config
    exp = _expected_dir()
    assert not _is_under(exp, REPO / 'logs')
    assert Path(request.config._trader_test_log_dir) == Path(os.environ[ENV])
    assert request.config._trader_test_log_dir_source == rc._TEST_LOG_DIR_SOURCE
    if rc._TEST_LOG_DIR_SOURCE == 'conftest':
        assert exp.name.startswith('trader-test-logs-')
    d, f = log_config._log_paths()
    assert Path(d).resolve() == exp and Path(f).resolve() == exp / 'trader.log'
    log_config.get_logger('intel-test')   # configures now if nothing did yet
    assert Path(log_config._file_handler.baseFilename).resolve() == exp / 'trader.log'


def test_env_is_set_before_any_repo_import_in_conftest_source():
    """Static guard on ordering: the env block precedes every non-stdlib import."""
    src = REAL_CONFTEST.read_text(encoding='utf-8').splitlines()
    set_line = next(i for i, ln in enumerate(src)
                    if 'tempfile.mkdtemp(' in ln and 'trader-test-logs-' in ln)
    imports = [i for i, ln in enumerate(src)
               if ln.startswith(('import ', 'from ')) and ln.split()[1].split('.')[0]
               not in ('json', 'os', 'tempfile')]
    assert imports and set_line < min(imports), (set_line, imports)


# ---------------------------------------------------------------------------
# (b) production log snapshot + verdict
# ---------------------------------------------------------------------------

def test_production_log_untouched_so_far(request, rc):
    cfg = request.config
    assert cfg._trader_prodlog_path == PROD_LOG
    # causal: this process has no logging handler on the production file
    assert rc._prodlog_own_handlers(PROD_LOG) == []
    before, now = cfg._trader_prodlog_before, rc._prodlog_stat(PROD_LOG)
    if now != before:
        # tolerated ONLY while another live process (the Jetson bots) holds it
        holders = rc._prodlog_holders(PROD_LOG, exclude_pid=os.getpid())
        assert holders, ('production log changed %s -> %s with no live foreign '
                         'writer holding it open' % (before, now))


def test_prodlog_verdict_pure(rc):
    a, b = (1, 10, 100), (1, 20, 150)
    assert rc._prodlog_verdict(a, a, [], []) == 'yes'
    assert rc._prodlog_verdict(None, None, [], []) == 'yes'
    assert rc._prodlog_verdict(a, b, [], []).startswith('no (')
    assert rc._prodlog_verdict(None, a, [], []).startswith('no (')
    assert rc._prodlog_verdict(a, b, [42, 7], []).startswith('unattributable (')
    assert '42,7' in rc._prodlog_verdict(a, b, [42, 7], [])
    own = rc._prodlog_verdict(a, a, [], ['/x/logs/trader.log'])
    assert own.startswith('no (this process has a logging handler')


def test_prodlog_stat_and_own_handlers(rc, tmp_path):
    import logging
    p = tmp_path / 'logs' / 'trader.log'
    assert rc._prodlog_stat(p) is None
    p.parent.mkdir()
    p.write_text('x')
    st = p.stat()
    assert rc._prodlog_stat(p) == (st.st_ino, st.st_mtime_ns, st.st_size)
    lg = logging.getLogger('intel-w18-own-%s' % uuid.uuid4().hex)
    h = logging.FileHandler(str(p))
    lg.addHandler(h)
    try:
        assert rc._prodlog_own_handlers(p) == [str(p)]
        if os.path.isdir('/proc'):
            assert rc._prodlog_holders(p) == [os.getpid()]
            assert rc._prodlog_holders(p, exclude_pid=os.getpid()) == []
    finally:
        lg.removeHandler(h)
        h.close()
    assert rc._prodlog_own_handlers(p) == []


# ---------------------------------------------------------------------------
# (c) a real record lands in the session dir, not production
# ---------------------------------------------------------------------------

def _read_new(path, before):
    """Bytes appended since `before` ((ino, mtime, size) or None); whole file
    if it was rotated or created meanwhile."""
    try:
        st = os.stat(path)
    except OSError:
        return b''
    with open(path, 'rb') as fh:
        if before is not None and before[0] == st.st_ino and st.st_size >= before[2]:
            fh.seek(before[2])
        return fh.read()


def test_log_record_lands_in_session_dir_not_production(rc):
    import log_config
    exp = _expected_dir()
    marker = 'intel-w18-%s' % uuid.uuid4().hex
    prod_before = rc._prodlog_stat(PROD_LOG)
    log_config.get_logger('intel-test').info(marker)
    fh = log_config._file_handler
    fh.flush()
    base = Path(fh.baseFilename)
    assert base.resolve() == exp / 'trader.log'
    text = base.read_text(encoding='utf-8', errors='replace')
    rolled = base.with_name('trader.log.1')
    if marker not in text and rolled.exists():
        text += rolled.read_text(encoding='utf-8', errors='replace')
    assert marker in text
    assert marker.encode() not in _read_new(PROD_LOG, prod_before)
    rolled_prod = PROD_LOG.with_name('trader.log.1')
    prod_now = rc._prodlog_stat(PROD_LOG)
    if prod_before is not None and (prod_now is None or prod_now[0] != prod_before[0]):
        assert marker.encode() not in _read_new(rolled_prod, None)   # rotated meanwhile


# ---------------------------------------------------------------------------
# (d) subprocess mini-project: conftest sets the env before the FIRST repo
#     import (a stub llm_client that calls get_logger at module scope, which
#     conftest itself imports), and reports the verdict line.
# ---------------------------------------------------------------------------

_STUB_LLM_CLIENT = '''
import os
ENV_AT_IMPORT = os.environ.get('TRADER_LOG_DIR')
from log_config import get_logger
logger = get_logger(__name__)          # the 22-module import-time pattern
logger.info('w18-stub-import')
_COST_FILE = 'unused.json'
_cost_reset_date = ''
_daily_cost = 0.0
'''

_MINI_TEST = '''
import json, os
from pathlib import Path
import llm_client, log_config
ROOT = Path(__file__).resolve().parent.parent


def test_probe(request):
    cfg = request.config
    out = {
        'env_at_stub_import': llm_client.ENV_AT_IMPORT,
        'env_now': os.environ.get('TRADER_LOG_DIR'),
        'handler': log_config._file_handler.baseFilename,
        'source': getattr(cfg, '_trader_test_log_dir_source', None),
        'cfg_dir': getattr(cfg, '_trader_test_log_dir', None),
        'proj_logs_exists': (ROOT / 'logs').exists(),
    }
    Path(os.environ['W18_OUT']).write_text(json.dumps(out))
    if os.environ.get('W18_POLLUTE'):
        (ROOT / 'logs').mkdir(exist_ok=True)
        (ROOT / 'logs' / 'trader.log').write_text('polluted\\n')
'''


def _run_mini(tmp_path, **env_extra):
    proj = tmp_path / 'proj'
    (proj / 'tests').mkdir(parents=True)
    (proj / 'pytest.ini').write_text('[pytest]\n')
    shutil.copy2(REPO / 'log_config.py', proj / 'log_config.py')
    (proj / 'llm_client.py').write_text(_STUB_LLM_CLIENT)
    shutil.copy2(REAL_CONFTEST, proj / 'tests' / 'conftest.py')
    (proj / 'tests' / 'test_mini.py').write_text(textwrap.dedent(_MINI_TEST))
    tmpdir = tmp_path / 'tmp'
    tmpdir.mkdir()
    env = {k: v for k, v in os.environ.items()
           if k not in (ENV, 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONPATH')
           and not k.startswith('TRADER_TESTS_')}
    env.update(PY_COLORS='0', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(tmpdir),
               W18_OUT=str(tmp_path / 'out.json'))
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         '--basetemp', str(tmp_path / 'bt'), 'tests'],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=180)
    out_path = tmp_path / 'out.json'
    out = json.loads(out_path.read_text()) if out_path.exists() else None
    return proj, tmpdir, proc, out


def test_subprocess_conftest_sets_env_before_first_repo_import(tmp_path):
    proj, tmpdir, proc, out = _run_mini(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out is not None, proc.stdout + proc.stderr
    env_dir = out['env_at_stub_import']
    assert env_dir is not None                        # set BEFORE the stub imported
    assert Path(env_dir).parent == tmpdir             # mkdtemp honoured TMPDIR
    assert Path(env_dir).name.startswith('trader-test-logs-')
    assert out['env_now'] == env_dir == out['cfg_dir']
    assert out['source'] == 'conftest'
    assert out['handler'] == str(Path(env_dir) / 'trader.log')
    assert out['proj_logs_exists'] is False           # the "production" dir untouched
    assert 'w18-stub-import' in (Path(env_dir) / 'trader.log').read_text()
    assert not (proj / 'logs').exists()
    assert 'test logs: %s (TRADER_LOG_DIR, set by conftest)' % env_dir in proc.stdout
    assert 'production log untouched: yes' in proc.stdout


def test_subprocess_caller_value_wins_and_pollution_reported(tmp_path):
    mine = tmp_path / 'mine'
    proj, tmpdir, proc, out = _run_mini(tmp_path, TRADER_LOG_DIR=str(mine),
                                        W18_POLLUTE='1')
    assert proc.returncode == 0, proc.stdout + proc.stderr   # report-only
    assert out is not None, proc.stdout + proc.stderr
    assert out['env_at_stub_import'] == str(mine) == out['cfg_dir']
    assert out['source'] == 'caller'
    assert out['handler'] == str(mine / 'trader.log')
    assert list(tmpdir.iterdir()) == []               # no mkdtemp when caller set it
    assert 'test logs: %s (TRADER_LOG_DIR, set by caller)' % mine in proc.stdout
    assert 'production log untouched: no (' in proc.stdout


# ---------------------------------------------------------------------------
# W19 add-on: the ledger sandbox also resets llm_client's thread-local meta
# ---------------------------------------------------------------------------

def test_call_meta_isolation_first_leaves_meta(rc):
    lc = rc._LLM_CLIENT
    if lc is None or not hasattr(lc, '_call_meta_tls'):
        pytest.skip('llm_client (with W19 call meta) not importable')
    lc._call_meta_tls.last = {'provider': 'gemini', 'model': 'leak'}
    assert lc.get_last_call_meta()['model'] == 'leak'


def test_call_meta_isolation_second_starts_clean(rc):
    lc = rc._LLM_CLIENT
    if lc is None or not hasattr(lc, '_call_meta_tls'):
        pytest.skip('llm_client (with W19 call meta) not importable')
    assert lc.get_last_call_meta() is None
