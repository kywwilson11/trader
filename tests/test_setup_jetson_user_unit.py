"""scripts/setup_jetson_system.sh --user / --print-unit (2026-09-27, ENGINE W5).

The prod Jetson has no sudo, so the system-level trader.service (step 6) cannot
be installed there. The script gained:
  --print-unit           print the step-6 system unit, touch nothing
  --user --print-unit    print the systemd --user variant, touch nothing
  --user                 install ~/.config/systemd/user/trader.service,
                         `systemctl --user daemon-reload` + `enable` (never start,
                         never enable-linger — that line is only printed)

Pure bash-rendering tests: they need bash + coreutils/awk/grep, NOT systemd.
Every systemctl/loginctl the script could reach is a fake on PATH, HOME and
XDG_* point into tmp_path — nothing is ever installed.
"""
import os
import re
import shutil
import subprocess

import pytest

if shutil.which('bash') is None:  # pragma: no cover - Mac/Linux always have it
    pytest.skip('bash not available', allow_module_level=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(ROOT, 'scripts', 'setup_jetson_system.sh')

# The system unit exactly as the PRE-EDIT script (sha256 01e2f8ed…, working tree
# of 2026-09-26 20:37) wrote it to /etc/systemd/system/trader.service, expanded
# with the default TRADER_PYBIN and SUDO_USER=kyle; @TRADER_DIR@ = repo root.
GOLDEN_SYSTEM_UNIT = """\
[Unit]
Description=Trader pipeline (bots + weekly retrain)
After=network-online.target chrony.service
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=all
User=kyle
WorkingDirectory=@TRADER_DIR@
ExecStart=/home/kyle/miniforge3/envs/jetson/bin/python -u run_pipeline.py --combined-bots --bot-only
Environment=PYTHONUNBUFFERED=1
Environment=LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6
Environment=LD_LIBRARY_PATH=/home/kyle/miniforge3/envs/jetson/lib:/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/nvidia/cusparselt/lib
# NO CUDA_VISIBLE_DEVICES here: training children MUST see the GPU. The bots
# are already hidden from it by run_pipeline.BOT_ENV (run_pipeline.py:310,
# CUDA_VISIBLE_DEVICES='') and run_bots.py:45 (setdefault ''); run_pipeline
# also drops an inherited empty value for training (_training_env).
# Leading '-': a missing .env is not fatal. Makes TRADER_TELEGRAM_* /
# TRADER_HEALTHCHECK_URL visible to the PARENT (kill switch, crash alerts).
# systemd syntax: KEY=VALUE lines, no 'export' prefix.
EnvironmentFile=-@TRADER_DIR@/.env
Restart=on-failure
RestartSec=30
WatchdogSec=900
# An OOM-killed child must not stop the whole unit (systemd default
# DefaultOOMPolicy=stop): let run_pipeline's phase retry / bot restart act.
OOMPolicy=continue
# OOM: kill the pipeline before the kernel picks a victim at random
OOMScoreAdjust=200
MemoryMax=6G

[Install]
WantedBy=multi-user.target
"""

# Lines the user unit must carry byte-identical to the system unit.
PRESERVED_KEYS = ('Description=', 'Type=', 'NotifyAccess=', 'WorkingDirectory=',
                  'ExecStart=', 'Environment=', 'EnvironmentFile=', 'Restart=',
                  'RestartSec=', 'WatchdogSec=', 'OOMPolicy=', 'OOMScoreAdjust=',
                  'MemoryMax=')


def _fake_bin(tmp_path, systemctl_body='exit 0'):
    """Dir with recording fakes for systemctl/loginctl + a fake jetson python."""
    b = tmp_path / 'bin'
    b.mkdir(exist_ok=True)
    log = tmp_path / 'calls.log'
    (b / 'systemctl').write_text(
        '#!/bin/sh\necho "systemctl $*" >> "%s"\n%s\n' % (log, systemctl_body))
    (b / 'loginctl').write_text('#!/bin/sh\necho "loginctl $*" >> "%s"\nexit 0\n' % log)
    (b / 'python').write_text('#!/bin/sh\nexit 0\n')  # passes the step-0 import gate
    for f in ('systemctl', 'loginctl', 'python'):
        (b / f).chmod(0o755)
    return b, log


def _run(args, tmp_path, fake_bin=None, runtime_dir=True, extra_env=None):
    home = tmp_path / 'home'
    home.mkdir(exist_ok=True)
    env = {
        'PATH': (f'{fake_bin}:' if fake_bin else '') + os.environ.get('PATH', '/usr/bin:/bin'),
        'HOME': str(home),
        'XDG_CONFIG_HOME': str(home / '.config'),
        'SUDO_USER': 'kyle',
    }
    if runtime_dir:
        rd = tmp_path / 'run'
        rd.mkdir(exist_ok=True)
        env['XDG_RUNTIME_DIR'] = str(rd)
    if fake_bin:
        env['TRADER_PYBIN'] = str(fake_bin / 'python')
    env.update(extra_env or {})
    return subprocess.run(['bash', SETUP, *args], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=60)


def _golden():
    return GOLDEN_SYSTEM_UNIT.replace('@TRADER_DIR@', ROOT)


def _unit_dir(tmp_path):
    return tmp_path / 'home' / '.config' / 'systemd' / 'user'


def _calls(log):
    return log.read_text().splitlines() if log.exists() else []


needs_non_root = pytest.mark.skipif(
    hasattr(os, 'geteuid') and os.geteuid() == 0,
    reason='--user refuses to run as root by design')


# ---------------------------------------------------------------- rendering

def test_bash_syntax():
    r = subprocess.run(['bash', '-n', SETUP], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_default_system_unit_rendering_unchanged(tmp_path):
    r = _run(['--print-unit'], tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout == _golden()


def test_print_unit_equals_step6_heredoc(tmp_path):
    """--print-unit renders what step 6 writes: expand the heredoc independently."""
    src = open(SETUP, encoding='utf-8').read()
    body = re.search(r"cat > /etc/systemd/system/trader\.service <<UNIT\n(.*?)\nUNIT\n",
                     src, re.S).group(1) + '\n'
    vals = {'TRADER_USER': 'kyle', 'TRADER_DIR': ROOT}
    for k in ('PYBIN', 'UNIT_LD_PRELOAD', 'UNIT_LD_LIBRARY_PATH'):
        m = re.search(r'^%s="?(.*?)"?$' % k, src, re.M)
        vals[k] = m.group(1)
    vals['PYBIN'] = re.sub(r'^\$\{TRADER_PYBIN:-(.*)\}$', r'\1', vals['PYBIN'])
    assert not re.search(r'`|\$\(|\\', body)  # plain ${VAR} expansion only
    expect = re.sub(r'\$\{(\w+)\}', lambda m: vals[m.group(1)], body)
    r = _run(['--print-unit'], tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout == expect


def test_user_unit_shape(tmp_path):
    sys_u = _run(['--print-unit'], tmp_path).stdout.splitlines()
    r = _run(['--user', '--print-unit'], tmp_path)
    assert r.returncode == 0, r.stderr
    usr = r.stdout.splitlines()
    assert not any(ln.startswith('User=') for ln in usr)
    assert 'WantedBy=default.target' in usr
    assert not any('multi-user.target' in ln or 'network-online' in ln
                   for ln in usr if not ln.startswith('#'))
    for key in PRESERVED_KEYS:
        s = [ln for ln in sys_u if ln.startswith(key)]
        u = [ln for ln in usr if ln.startswith(key)]
        assert s and s == u, key
    # everything else identical: the ONLY removed lines are User=/After=/Wants=/WantedBy=
    removed = [ln for ln in sys_u if ln not in usr]
    assert sorted(ln.split('=')[0] for ln in removed) == ['After', 'User', 'WantedBy', 'Wants']
    added = [ln for ln in usr if ln not in sys_u]
    assert all(ln.startswith('#') or ln == 'WantedBy=default.target' for ln in added)
    # no systemd >= 254 directives (prod box is systemd 249)
    assert not any(ln.startswith(('RestartSteps=', 'RestartMaxDelaySec=')) for ln in usr)


def test_print_unit_touches_nothing(tmp_path):
    fb, log = _fake_bin(tmp_path)
    for args in (['--print-unit'], ['--user', '--print-unit']):
        r = _run(args, tmp_path, fake_bin=fb)
        assert r.returncode == 0, r.stderr
    assert _calls(log) == []
    assert list((tmp_path / 'home').iterdir()) == []


# ---------------------------------------------------------------- --user install

@needs_non_root
def test_user_refuses_when_user_manager_unreachable(tmp_path):
    fb, log = _fake_bin(tmp_path, systemctl_body='exit 1')
    r = _run(['--user'], tmp_path, fake_bin=fb)
    assert r.returncode != 0
    assert 'XDG_RUNTIME_DIR' in r.stderr and 'loginctl enable-linger' in r.stderr
    assert not _unit_dir(tmp_path).exists()
    assert not any('loginctl' in c for c in _calls(log))


@needs_non_root
def test_user_refuses_without_xdg_runtime_dir(tmp_path):
    fb, log = _fake_bin(tmp_path)  # systemctl would succeed
    r = _run(['--user'], tmp_path, fake_bin=fb, runtime_dir=False)
    assert r.returncode != 0
    assert 'XDG_RUNTIME_DIR=<unset>' in r.stderr
    assert not _unit_dir(tmp_path).exists()


@needs_non_root
def test_user_enable_failure_leaves_no_unit(tmp_path):
    fb, log = _fake_bin(tmp_path, systemctl_body=(
        'case "$*" in *" enable "*|*" enable") exit 1;; esac\nexit 0'))
    r = _run(['--user'], tmp_path, fake_bin=fb)
    assert r.returncode != 0
    ud = _unit_dir(tmp_path)
    left = sorted(p.name for p in ud.rglob('*') if p.is_file()) if ud.exists() else []
    assert left == []
    assert 'systemctl --user enable trader.service' in _calls(log)


@needs_non_root
def test_user_install_happy_path(tmp_path):
    fb, log = _fake_bin(tmp_path)
    expect = _run(['--user', '--print-unit'], tmp_path, fake_bin=fb).stdout
    assert f'ExecStart={fb}/python -u run_pipeline.py' in expect  # TRADER_PYBIN honoured
    assert _calls(log) == []  # --print-unit calls nothing
    r = _run(['--user'], tmp_path, fake_bin=fb)
    assert r.returncode == 0, r.stderr
    unit = _unit_dir(tmp_path) / 'trader.service'
    assert unit.read_text() == expect
    assert sorted(p.name for p in _unit_dir(tmp_path).iterdir()) == ['trader.service']
    calls = _calls(log)
    assert 'systemctl --user daemon-reload' in calls
    assert 'systemctl --user enable trader.service' in calls
    assert not any(' start' in c or 'loginctl' in c for c in calls)  # never started/lingered
    assert 'loginctl enable-linger ' in r.stdout
    assert 'systemctl --user start trader' in r.stdout
    assert 'journalctl --user -u trader -f' in r.stdout


@needs_non_root
def test_user_existing_unit_left_untouched(tmp_path):
    fb, log = _fake_bin(tmp_path)
    ud = _unit_dir(tmp_path)
    ud.mkdir(parents=True)
    (ud / 'trader.service').write_text('# hand-edited\n')
    r = _run(['--user'], tmp_path, fake_bin=fb)
    assert r.returncode == 0, r.stderr
    assert (ud / 'trader.service').read_text() == '# hand-edited\n'
    assert not any('enable' in c or 'daemon-reload' in c for c in _calls(log))
