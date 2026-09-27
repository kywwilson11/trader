"""Jetson ops fixes (2026-09-26 ops audit) — Mac-safe source/AST contracts.

No heavy imports: the setup script and run_pipeline are read as text/AST;
fundamentals is importable on the Mac (stdlib + llm_config only).

Pins:
  (a) the systemd unit emitted by scripts/setup_jetson_system.sh has NO
      CUDA_VISIBLE_DEVICES line (training must see the GPU) and DOES carry
      OOMPolicy=continue, EnvironmentFile=-.env and run_pipeline.ENV's
      LD_PRELOAD / LD_LIBRARY_PATH;
  (b) the unit interpreter is the jetson env (TRADER_PYBIN override), never
      `command -v python3` (which under sudo is /usr/bin/python3);
  (c) run_pipeline's training env drops an inherited EMPTY
      CUDA_VISIBLE_DEVICES, passes a non-empty one through, is what the phase
      Popen uses, and BOT_ENV still hides the GPU; combined-mode status
      reports the books of a live 'Bots' process as running;
  (d) fundamentals formats a non-numeric P/E (and neighbours) without raising.
"""
import ast
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(ROOT, 'scripts', 'setup_jetson_system.sh')
PIPELINE = os.path.join(ROOT, 'run_pipeline.py')


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _unit_text():
    """The trader.service heredoc body, verbatim from the setup script."""
    src = _read(SETUP)
    m = re.search(r"cat > /etc/systemd/system/trader\.service <<UNIT\n(.*?)\nUNIT\n",
                  src, re.S)
    assert m, 'trader.service heredoc not found in setup script'
    return m.group(1)


def _pipeline_tree():
    return ast.parse(_read(PIPELINE))


def _pipeline_func(name, extra_globals=None):
    """Extract one top-level function from run_pipeline.py and exec it alone."""
    tree = _pipeline_tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = dict(extra_globals or {})
            exec(compile(mod, PIPELINE, 'exec'), ns)
            return ns[name]
    raise AssertionError(f'{name} not found in run_pipeline.py')


def _pipeline_env_literals():
    """LD_PRELOAD value + the LD_LIBRARY_PATH string prefix of run_pipeline.ENV."""
    tree = _pipeline_tree()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'ENV'):
            d = node.value
            out = {}
            for k, v in zip(d.keys, d.values):
                if k is None:
                    continue
                if k.value == 'LD_PRELOAD':
                    out['LD_PRELOAD'] = v.value
                elif k.value == 'LD_LIBRARY_PATH':
                    # ('a:' 'b:' + os.environ.get(...)) -> BinOp(Constant, Call)
                    assert isinstance(v, ast.BinOp)
                    out['LD_LIBRARY_PATH'] = v.left.value.rstrip(':')
            return out
    raise AssertionError('ENV assignment not found in run_pipeline.py')


# --------------------------------------------------------------------- (a)

def test_unit_has_no_cuda_visible_devices():
    unit = _unit_text()
    for line in unit.splitlines():
        if line.lstrip().startswith('#'):
            continue
        assert 'CUDA_VISIBLE_DEVICES' not in line, line


def test_unit_oom_policy_and_env_file():
    unit = _unit_text().splitlines()
    assert 'OOMPolicy=continue' in unit
    assert 'EnvironmentFile=-${TRADER_DIR}/.env' in unit


def test_unit_ld_vars_match_run_pipeline_env():
    src = _read(SETUP)
    unit = _unit_text().splitlines()
    assert 'Environment=LD_PRELOAD=${UNIT_LD_PRELOAD}' in unit
    assert 'Environment=LD_LIBRARY_PATH=${UNIT_LD_LIBRARY_PATH}' in unit
    env = _pipeline_env_literals()
    assert f"UNIT_LD_PRELOAD={env['LD_PRELOAD']}\n" in src
    assert f"UNIT_LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}\n" in src


def test_unit_still_installed_not_enabled():
    src = _read(SETUP)
    assert 'systemctl enable' not in re.sub(r'echo "[^"\n]*"', '', src.split(
        '# --- 6. systemd service')[1].split('# --- 7.')[0])
    assert 'installed (NOT enabled' in src


# --------------------------------------------------------------------- (b)

def test_pybin_is_jetson_env_with_override():
    src = _read(SETUP)
    assert ('PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"'
            in src.splitlines())
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith('#')]
    assert not any('command -v python3' in ln for ln in code)
    assert "import pyarrow, dotenv, torch" in src
    # the default must equal run_pipeline.PYTHON (the documented prod path)
    tree = _pipeline_tree()
    python = next(n.value.value for n in tree.body
                  if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], 'id', None) == 'PYTHON')
    assert f'${{TRADER_PYBIN:-{python}}}' in src
    # checked before any system change (step 1 = headless)
    assert src.index('import pyarrow, dotenv, torch') < src.index('# --- 1. Headless')
    assert 'ExecStart=${PYBIN} -u run_pipeline.py' in _unit_text()


def test_pybin_check_fails_loudly_on_bad_interpreter(tmp_path):
    """Run just the step-0 block with a fake interpreter: must exit non-zero."""
    src = _read(SETUP)
    block = src.split('# --- 0. Interpreter')[1].split('# --- 1. Headless')[0]
    script = tmp_path / 'step0.sh'
    script.write_text('set -euo pipefail\n#' + block)
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\nexit 1\n')
    fake.chmod(0o755)
    for pybin in (str(fake), str(tmp_path / 'missing')):
        r = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                           env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                                'TRADER_PYBIN': pybin})
        assert r.returncode != 0
        assert 'FATAL' in r.stderr


def test_bash_syntax():
    r = subprocess.run(['bash', '-n', SETUP], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------- (c)

def test_training_env_drops_empty_cuda_visible_devices():
    f = _pipeline_func('_training_env')
    base = {'CUDA_VISIBLE_DEVICES': '', 'LD_PRELOAD': 'x'}
    out = f(base)
    assert 'CUDA_VISIBLE_DEVICES' not in out
    assert out['LD_PRELOAD'] == 'x'
    assert base['CUDA_VISIBLE_DEVICES'] == ''  # input not mutated


@pytest.mark.parametrize('val', ['0', '0,1', '-1'])
def test_training_env_passes_explicit_value(val):
    f = _pipeline_func('_training_env')
    assert f({'CUDA_VISIBLE_DEVICES': val})['CUDA_VISIBLE_DEVICES'] == val


def test_training_env_absent_key_stays_absent():
    f = _pipeline_func('_training_env')
    assert f({'A': '1'}) == {'A': '1'}


def test_phase_popen_uses_train_env_and_bots_hide_gpu():
    src = _read(PIPELINE)
    assert 'TRAIN_ENV = _training_env(ENV)' in src
    tree = _pipeline_tree()
    run_phase = next(n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == 'run_phase')
    envs = [kw.value.id for call in ast.walk(run_phase)
            if isinstance(call, ast.Call) and getattr(call.func, 'attr', '') == 'Popen'
            for kw in call.keywords if kw.arg == 'env']
    assert envs == ['TRAIN_ENV']
    bot_env = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], 'id', None) == 'BOT_ENV')
    pairs = {k.value: v.value for k, v in zip(bot_env.value.keys, bot_env.value.values)
             if k is not None}
    assert pairs['CUDA_VISIBLE_DEVICES'] == ''
    start_bot = next(n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == '_start_bot')
    assert 'env=BOT_ENV' in ast.unparse(start_bot)


class _P:
    def __init__(self, alive):
        self._alive = alive

    def poll(self):
        return None if self._alive else 1


@pytest.mark.parametrize('scope,alive,expect', [
    ((True, True), True, (True, True, True)),
    ((True, False), True, (True, False, True)),
    ((False, True), True, (False, True, True)),
    ((True, True), False, (False, False, False)),
])
def test_combined_bots_status(scope, alive, expect):
    f = _pipeline_func('_update_per_bot_status', {'_BOT_SCOPE': scope})
    status = {}
    f([('Bots', _P(alive), None)], status)
    assert (status['crypto_bot_running'], status['stock_bot_running'],
            status['bots_running']) == expect


def test_split_bots_status_unchanged():
    f = _pipeline_func('_update_per_bot_status', {'_BOT_SCOPE': (True, True)})
    status = {}
    f([('Crypto', _P(True), None), ('Stock', _P(False), None)], status)
    assert status == {'crypto_bot_running': True, 'stock_bot_running': False,
                      'bots_running': True}


# --------------------------------------------------------------------- (d)

fundamentals = pytest.importorskip('fundamentals')


@pytest.mark.parametrize('pe,expect', [
    (12.345, 'P/E=12.3'),
    ('12.345', 'P/E=12.3'),
    ('', 'P/E=n/a'),
    ('N/A', 'P/E=n/a'),
    ('Infinity', 'P/E=n/a'),
    (float('nan'), 'P/E=n/a'),
])
def test_pe_formatting_never_raises(pe, expect):
    text = fundamentals.format_fundamentals_for_llm('X', {'pe_ratio': pe})
    assert expect in text


def test_pe_none_is_omitted():
    text = fundamentals.format_fundamentals_for_llm('X', {'pe_ratio': None})
    assert 'P/E' not in text
    assert text == 'Fundamentals: limited data'


def test_neighbour_fields_string_safe():
    fund = {'pb_ratio': 'N/A', 'market_cap': 'N/A', 'revenue_growth': '',
            'eps': 'x', 'dividend_yield': 'N/A', 'beta': '',
            'week52_high': 'N/A', 'week52_low': 10.0}
    text = fundamentals.format_fundamentals_for_llm('X', fund)
    for token in ('P/B=n/a', 'MktCap=n/a', 'RevGrowth=n/a', 'EPS=n/a',
                  'DivYield=n/a', 'Beta=n/a'):
        assert token in text
    assert '52wk' not in text


def test_neighbour_fields_numeric_unchanged():
    fund = {'pe_ratio': 25.3, 'pb_ratio': 3.14, 'market_cap': 2.5e12,
            'revenue_growth': 0.123, 'eps': 1.234, 'dividend_yield': 0.0037,
            'beta': 1.1, 'week52_high': 200.0, 'week52_low': 100.0}
    text = fundamentals.format_fundamentals_for_llm('X', fund)
    assert text == ('Fundamentals: P/E=25.3, P/B=3.1, MktCap=$2.5T, '
                    'RevGrowth=12.3%, EPS=1.23, DivYield=0.37%, Beta=1.10, '
                    '52wk=$100.00-$200.00')


@pytest.mark.parametrize('val,expect', [
    (None, None), ('', None), ('N/A', None), (True, None),
    (float('inf'), None), ('3.5', 3.5), (7, 7.0),
])
def test_safe_float(val, expect):
    assert fundamentals._safe_float(val) == expect


# --------------------------------------------------------------------- (e)
# 2026-09-26 G3-7: backup_state.sh's no-sqlite3-CLI fallback interpreter is
# the same jetson python as the unit's (one override var, one default).

BACKUP = os.path.join(ROOT, 'scripts', 'backup_state.sh')


def test_backup_pybin_matches_setup_and_pipeline():
    src = _read(BACKUP)
    line = 'PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"'
    assert line in src.splitlines()
    assert line in _read(SETUP).splitlines()
    tree = _pipeline_tree()
    python = next(n.value.value for n in tree.body
                  if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], 'id', None) == 'PYTHON')
    assert f'${{TRADER_PYBIN:-{python}}}' in src
    # the CLI stays first choice; the stdlib online-backup API is the fallback
    step1 = src.split('# 1. SQLite')[1].split('# 2. JSON')[0]
    assert step1.index('command -v sqlite3') < step1.index('s.backup(d)')
    assert 'WARN: sqlite backup failed for $db' in step1


def test_backup_bash_syntax():
    r = subprocess.run(['bash', '-n', BACKUP], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
