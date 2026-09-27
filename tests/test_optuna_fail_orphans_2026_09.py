"""scripts/optuna_fail_orphans.py — tiny real Optuna studies in tmp_path.

Pins: dry run is byte- and mtime-neutral (read-only sqlite, no optuna
import needed); --apply turns exactly the RUNNING trials into FAIL through
the storage API and leaves COMPLETE trials (and their values) untouched;
--apply refuses while a trainer is alive (DB untouched); study discovery,
--study / --trial / --min-age-min filters; a missing DB is never created.
"""
import hashlib
import os
import sys
from pathlib import Path

import pytest

optuna = pytest.importorskip('optuna')
from optuna.trial import TrialState  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import optuna_fail_orphans as ofo  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_db(tmp_path, studies=(('v2_search', 2, 1),)):
    """studies: (name, n_complete, n_running). Returns the DB path."""
    db = tmp_path / 'v2_study.db'
    url = f'sqlite:///{db}'
    for name, n_done, n_run in studies:
        st = optuna.create_study(study_name=name, storage=url,
                                 direction='maximize')
        st.optimize(lambda t: t.suggest_float('x', 0, 1), n_trials=n_done)
        for _ in range(n_run):
            st.ask({'x': optuna.distributions.FloatDistribution(0, 1)})
    return db


def _digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest(), \
        os.stat(p).st_mtime_ns


def _states(db, name='v2_search'):
    st = optuna.load_study(study_name=name, storage=f'sqlite:///{db}')
    return [(t.number, t.state, t.value) for t in st.trials]


def test_dry_run_lists_running_and_never_touches_db(tmp_path, capsys,
                                                    monkeypatch):
    db = _make_db(tmp_path)
    before = _digest(db)
    called = []
    monkeypatch.setattr(ofo, 'apply_fail', lambda *a, **k: called.append(1))
    assert ofo.main(['--db', str(db)]) == 0
    out = capsys.readouterr().out
    assert 'RUNNING selected=1' in out and 'trial #2' in out
    assert 'mode=dry-run' in out and 'newest RUNNING' in out
    assert _digest(db) == before and not called
    # explicit --dry-run is the same
    assert ofo.main(['--db', str(db), '--dry-run']) == 0
    assert _digest(db) == before


def test_apply_fails_only_running_trials(tmp_path, monkeypatch, capsys):
    db = _make_db(tmp_path, studies=(('v2_search', 3, 2),))
    pre = _states(db)
    monkeypatch.setattr(ofo, 'live_trainers', lambda: [])
    assert ofo.main(['--db', str(db), '--apply']) == 0
    post = _states(db)
    assert [s for _, s, _ in post] == [TrialState.COMPLETE] * 3 + \
        [TrialState.FAIL] * 2
    assert [(n, v) for n, _, v in post[:3]] == [(n, v) for n, _, v in pre[:3]]
    assert 'RUNNING -> FAIL' in capsys.readouterr().out
    # idempotent: second run finds nothing and leaves the DB alone
    before = _digest(db)
    assert ofo.main(['--db', str(db), '--apply']) == 0
    assert _digest(db) == before


def test_apply_refused_while_trainer_alive(tmp_path, monkeypatch, capsys):
    db = _make_db(tmp_path)
    before = _digest(db)
    monkeypatch.setattr(ofo, 'live_trainers', lambda: [4242])
    monkeypatch.setattr(ofo, 'apply_fail',
                        lambda *a, **k: pytest.fail('must not write'))
    assert ofo.main(['--db', str(db), '--apply']) == 3
    assert 'REFUSED' in capsys.readouterr().out
    assert _digest(db) == before


def test_pgrep_unavailable_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError('pgrep')
    monkeypatch.setattr(ofo.subprocess, 'run', boom)
    assert ofo.live_trainers() == [-1]


def test_study_discovery_and_filters(tmp_path, monkeypatch, capsys):
    db = _make_db(tmp_path, studies=(('v2_search', 1, 2), ('other', 1, 1)))
    assert ofo.main(['--db', str(db)]) == 0
    out = capsys.readouterr().out
    assert "studies=['other', 'v2_search']" in out
    assert 'RUNNING selected=3' in out
    assert ofo.main(['--db', str(db), '--study', 'other']) == 0
    assert 'RUNNING selected=1' in capsys.readouterr().out
    assert ofo.main(['--db', str(db), '--study', 'nope']) == 2
    # --trial restricts by NUMBER; --apply only touches that one
    monkeypatch.setattr(ofo, 'live_trainers', lambda: [])
    assert ofo.main(['--db', str(db), '--study', 'v2_search', '--trial', '1',
                     '--apply']) == 0
    st = dict((n, s) for n, s, _ in _states(db))
    assert st[1] == TrialState.FAIL and st[2] == TrialState.RUNNING
    assert _states(db, 'other')[1][1] == TrialState.RUNNING
    # --min-age-min: freshly started trials are younger than a day
    capsys.readouterr()
    assert ofo.main(['--db', str(db), '--min-age-min', '1440']) == 0
    assert 'RUNNING selected=0' in capsys.readouterr().out


def test_apply_backup_written_before_change(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    bak = tmp_path / 'pre.db'
    monkeypatch.setattr(ofo, 'live_trainers', lambda: [])
    assert ofo.main(['--db', str(db), '--apply', '--backup', str(bak)]) == 0
    assert _states(bak)[-1][1] == TrialState.RUNNING
    assert _states(db)[-1][1] == TrialState.FAIL


def test_missing_db_is_not_created(tmp_path):
    db = tmp_path / 'absent.db'
    assert ofo.main(['--db', str(db)]) == 2
    assert ofo.main(['--db', str(db), '--apply']) == 2
    assert not db.exists()


def test_trainer_pattern_matches_real_cmdlines():
    import re
    pat = re.compile(ofo.TRAINER_PATTERN)
    assert pat.search('/home/kyle/miniforge3/envs/jetson/bin/python -u '
                      'scripts/hypersearch_v2.py --trials 40 --shadow')
    assert pat.search('python3 /home/kyle/trader/scripts/hypersearch_v2.py')
    assert not pat.search('python -m pytest tests/test_hypersearch_v2.py')
    assert not pat.search('python scripts/optuna_fail_orphans.py --db '
                          'v2_study.db')
