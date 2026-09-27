"""TRADER_LOG_DIR override for log_config (2026-09-27, ENGINE r6 W18).

Finding R5-W16 (class D): every pytest process appended to the PRODUCTION
<repo>/logs/trader.log (33,452 fake lines in one night), so live forensics on
that file were impossible. log_config now resolves the directory through ONE
helper, log_config._log_paths(), read at the first _setup() call:

  * unset / empty      -> <repo>/logs/trader.log, byte-identical to before
  * set (abs or        -> <dir>/trader.log (+ .1-.5, + trader.log.lock);
    repo-relative)        relative values resolve against the repo root, not CWD
  * uncreatable / not  -> default path + ONE stderr line, never raises
    writable
  * module-level _LOG_DIR/_LOG_FILE reassignment (tests) beats the variable
  * set after a handler exists -> no re-pointing (the process keeps its file)

Every case runs in a CHILD interpreter so log_config's module state is fresh;
this process never imports log_config. Cases that need a real write under the
DEFAULT path run against a copy of log_config.py in tmp_path (its default dir is
then tmp_path/repo/logs), so nothing here writes into the real <repo>/logs --
which the LIVE bots are writing to while this runs (read-only checks only).
Stdlib-only.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_LOGS = ROOT / 'logs'
ENV = 'TRADER_LOG_DIR'

_FMT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
_DATE_FMT = '%Y-%m-%d %H:%M:%S'


def _run(code, repo, env_value=None, cwd=None):
    """Run `code` in a fresh interpreter with `repo` first on sys.path."""
    env = dict(os.environ)
    env.pop(ENV, None)
    if env_value is not None:
        env[ENV] = str(env_value)
    prog = f"import sys; sys.path.insert(0, {str(repo)!r})\n" + code
    r = subprocess.run([sys.executable, '-c', prog], env=env, cwd=str(cwd or repo),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r


def _last_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


def _copy_repo(tmp_path):
    """A throwaway 'repo' holding only log_config.py: its default log dir is
    tmp_path/repo/logs, so default-path writes stay sandboxed."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    shutil.copy2(ROOT / 'log_config.py', repo / 'log_config.py')
    return repo


def _repo_logs_listing():
    try:
        return set(os.listdir(REPO_LOGS))
    except FileNotFoundError:
        return set()


def _assert_absent_from_repo_logs(before, *markers):
    """Nothing NEW in <repo>/logs and no marker in its live file(s). A live bot
    rotating mid-test can only add trader.log.N names; those are tolerated."""
    new = {n for n in _repo_logs_listing() - before
           if not re.fullmatch(r'trader\.log\.\d+', n)}
    assert not new, new
    for name in ('trader.log', 'trader.log.1'):
        p = REPO_LOGS / name
        if p.exists():
            data = p.read_bytes()
            for m in markers:
                assert m.encode() not in data, (name, m)


# ------------------------------------------------------------------ unset

_FINGERPRINT = r'''
import io, json, logging, pathlib
logging.FileHandler._open = lambda self: io.StringIO()   # no disk writes
mk = []
_real_mkdir = pathlib.Path.mkdir
pathlib.Path.mkdir = lambda self, *a, **k: mk.append([str(self), a, k])
import log_config
root = logging.getLogger(); before = list(root.handlers)
log_config._setup()
fh = log_config._file_handler
fl = log_config.get_file_logger('w18_fp')
print(json.dumps({
    'base': fh.baseFilename, 'cls': type(fh).__name__,
    'maxBytes': fh.maxBytes, 'backupCount': fh.backupCount,
    'encoding': fh.encoding, 'mode': fh.mode, 'level': fh.level,
    'fmt': fh.formatter._fmt, 'datefmt': fh.formatter.datefmt,
    'n_added': len([h for h in root.handlers if h not in before]),
    'fl_is_fh': fl.handlers == [fh], 'mkdir': mk}))
'''


@pytest.mark.parametrize('env_value', [None, ''])
def test_unset_or_empty_is_byte_identical_default(env_value):
    """Golden values captured from the PRE-edit module (w18 scratch probe:
    orig vs new fingerprints cmp-identical, unset and empty)."""
    before = _repo_logs_listing()
    fp = _last_json(_run(_FINGERPRINT, ROOT, env_value).stdout)
    golden = str(ROOT / 'logs' / 'trader.log')
    assert fp['base'] == golden
    assert fp['base'] + '.lock' == str(ROOT / 'logs' / 'trader.log.lock')
    assert fp['cls'] == 'SharedRotatingFileHandler'
    assert (fp['maxBytes'], fp['backupCount']) == (10 * 1024 * 1024, 5)
    assert (fp['encoding'], fp['mode'], fp['level']) == ('utf-8', 'a', 10)
    assert (fp['fmt'], fp['datefmt']) == (_FMT, _DATE_FMT)
    assert fp['n_added'] == 2 and fp['fl_is_fh'] is True
    assert fp['mkdir'] == [[str(ROOT / 'logs'), [], {'exist_ok': True}]]
    _assert_absent_from_repo_logs(before)


def test_unset_real_write_lands_in_module_dir_logs(tmp_path):
    repo = _copy_repo(tmp_path)
    marker = f'w18-default-{uuid.uuid4().hex}'
    _run(f"import log_config\nlog_config.get_logger('w18').info({marker!r})\n", repo)
    assert marker in (repo / 'logs' / 'trader.log').read_text(encoding='utf-8')


# ------------------------------------------------------------------ override

def test_override_redirects_get_logger_and_file_logger(tmp_path):
    ov = tmp_path / 'ov' / 'nested'          # created by makedirs (parents)
    m1 = f'w18-root-{uuid.uuid4().hex}'
    m2 = f'w18-fileonly-{uuid.uuid4().hex}'
    before = _repo_logs_listing()
    r = _run("import json, log_config\n"
             f"log_config.get_logger('w18').info({m1!r})\n"
             f"log_config.get_file_logger('w18_f').warning({m2!r})\n"
             "print(json.dumps(log_config._file_handler.baseFilename))\n",
             ROOT, ov)
    assert _last_json(r.stdout) == str(ov / 'trader.log')
    text = (ov / 'trader.log').read_text(encoding='utf-8')
    assert m1 in text and m2 in text
    _assert_absent_from_repo_logs(before, m1, m2)


def test_rotation_under_override_uses_sibling_lock(tmp_path):
    ov = tmp_path / 'ov'
    marker = f'w18-rot-{uuid.uuid4().hex}'
    before = _repo_logs_listing()
    _run("import log_config\n"
         "log_config._MAX_BYTES = 400\n"
         "lg = log_config.get_logger('w18_rot')\n"
         f"for i in range(40): lg.debug({marker!r} + ' %03d', i)\n",
         ROOT, ov)
    assert (ov / 'trader.log.1').exists()
    assert (ov / 'trader.log.lock').exists()
    assert marker in (ov / 'trader.log.1').read_text(encoding='utf-8')
    _assert_absent_from_repo_logs(before, marker)


def test_relative_value_resolves_against_repo_root_not_cwd(tmp_path):
    repo = _copy_repo(tmp_path)
    elsewhere = tmp_path / 'cwd'
    elsewhere.mkdir()
    marker = f'w18-rel-{uuid.uuid4().hex}'
    _run(f"import log_config\nlog_config.get_logger('w18').info({marker!r})\n",
         repo, 'rel/logs', cwd=elsewhere)
    assert marker in (repo / 'rel' / 'logs' / 'trader.log').read_text(encoding='utf-8')
    assert not (elsewhere / 'rel').exists()
    assert not (repo / 'logs').exists()


def _assert_fallback(r, repo, marker):
    lines = [ln for ln in r.stderr.splitlines() if ENV in ln]
    assert len(lines) == 1, r.stderr
    assert 'unusable' in lines[0]
    assert marker in (repo / 'logs' / 'trader.log').read_text(encoding='utf-8')


def test_uncreatable_dir_falls_back_with_one_stderr_line(tmp_path):
    repo = _copy_repo(tmp_path)
    blocker = tmp_path / 'afile'
    blocker.write_text('x')                  # a FILE where a dir is needed
    marker = f'w18-fb-{uuid.uuid4().hex}'
    r = _run(f"import log_config\nlog_config.get_logger('w18').info({marker!r})\n",
             repo, blocker / 'sub')
    _assert_fallback(r, repo, marker)


@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0,
                    reason='root ignores directory permissions')
def test_unwritable_dir_falls_back_with_one_stderr_line(tmp_path):
    repo = _copy_repo(tmp_path)
    ro = tmp_path / 'ro'
    ro.mkdir()
    ro.chmod(0o555)
    marker = f'w18-ro-{uuid.uuid4().hex}'
    try:
        r = _run(f"import log_config\nlog_config.get_logger('w18').info({marker!r})\n",
                 repo, ro)
    finally:
        ro.chmod(0o755)
    _assert_fallback(r, repo, marker)
    assert not (ro / 'trader.log').exists()


# ------------------------------------------------------------------ timing / precedence

def test_read_at_first_setup_not_at_import(tmp_path):
    """conftest-style: a value set after `import log_config` but before the
    first get_logger call is honoured."""
    repo = _copy_repo(tmp_path)
    ov = tmp_path / 'late_but_before_setup'
    marker = f'w18-call-{uuid.uuid4().hex}'
    _run("import os, log_config\n"
         f"os.environ[{ENV!r}] = {str(ov)!r}\n"
         f"log_config.get_logger('w18').info({marker!r})\n", repo)
    assert marker in (ov / 'trader.log').read_text(encoding='utf-8')
    assert not (repo / 'logs').exists()


def test_set_after_handler_exists_does_not_repoint(tmp_path):
    repo = _copy_repo(tmp_path)
    late = tmp_path / 'too_late'
    marker = f'w18-late-{uuid.uuid4().hex}'
    _run("import os, log_config\n"
         "log_config.get_logger('w18')\n"
         f"os.environ[{ENV!r}] = {str(late)!r}\n"
         f"log_config.get_logger('w18b').info({marker!r})\n", repo)
    assert marker in (repo / 'logs' / 'trader.log').read_text(encoding='utf-8')
    assert not late.exists()


def test_module_attribute_reassignment_beats_env(tmp_path):
    """tests/test_review_b20.py + test_g3_ops_2026_09.py monkeypatch _LOG_DIR /
    _LOG_FILE; those redirects must keep winning once conftest sets the var."""
    repo = _copy_repo(tmp_path)
    ov = tmp_path / 'ov'
    pdir = tmp_path / 'patched'
    marker = f'w18-patch-{uuid.uuid4().hex}'
    _run("import log_config\nfrom pathlib import Path\n"
         f"log_config._LOG_DIR = Path({str(pdir)!r})\n"
         f"log_config._LOG_FILE = Path({str(pdir / 'trader.log')!r})\n"
         f"log_config.get_logger('w18').info({marker!r})\n", repo, ov)
    assert marker in (pdir / 'trader.log').read_text(encoding='utf-8')
    assert not (ov / 'trader.log').exists()
    assert not (repo / 'logs').exists()


def test_single_source_helper_and_env_name():
    src = (ROOT / 'log_config.py').read_text(encoding='utf-8')
    assert "_LOG_DIR_ENV = 'TRADER_LOG_DIR'" in src
    assert src.count('= _log_paths()') == 1        # the one call, in _setup
    assert 'fh_cls(str(log_file)' in src
    assert "self.baseFilename + '.lock'" in src    # lock derives from the handler
