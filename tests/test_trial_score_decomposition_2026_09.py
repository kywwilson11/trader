"""scripts/trial_score_decomposition.py — measurement-only trial-score classifier.

Pins: (1) the pre-registered classifier returns exactly the hand-built class
for a tiny Optuna study whose user_attrs were constructed per class; (2) the
layer-2 ORACLE / RANDOM construction on a synthetic target (same admit rate by
permutation, oracle gross > threshold, random gross ~ the unconditional mean,
n_possible = ceil(n/fb)); (3) the Sharpe-band inversion is exact; (4) main()
exits 0 and writes nothing outside --out (the study DB is snapshotted, never
opened or modified).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    'trial_score_decomposition', ROOT / 'scripts' / 'trial_score_decomposition.py')
tsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsd)

COST = 0.60
BPY = 8760.0


def _l3(n, g, s, pr=0.05, rows=30000, hold=24.0):
    k = len(n)
    return {'n_trades': list(n), 'gross_ret_mean': list(g),
            'gross_ret_std': [s] * k,
            'net_ret_mean': [x - COST for x in g], 'cost_drag': [COST] * k,
            'hit_rate': [0.5] * k, 'mean_hold_bars': [hold] * k,
            'threshold_pass_rate': [pr] * k, 'n_rows': [rows] * k}


# (class, fold_sharpes, regime_min, threshold, layer-3 attrs)
CASES = {
    'b': ([-1.0, -1.2, -0.9], -0.2, 0.80, _l3([400, 400, 400], [0.30, 0.25, 0.35], 0.5)),
    'a': ([-1.5, -1.4, -1.6], -0.3, 0.90, _l3([100, 100, 100], [0.01, -0.02, 0.03], 3.0)),
    'd': ([-0.4, -1.0, -1.1], -0.1, 0.95, _l3([5, 60, 60], [0.1, 0.1, 0.1], 2.0)),
    'c': ([1.0, 1.2, -1.5], -0.1, 0.85, _l3([300, 300, 300], [0.9, 0.95, 0.4], 1.0)),
    '+': ([0.9, 1.1, 1.0], -0.2, 0.99, _l3([300, 300, 300], [0.95, 1.0, 0.9], 0.5)),
}


def _score(folds, rmin, form='legacy'):
    pre = float(np.mean(folds)) - 0.5 * float(np.std(folds))
    if rmin < -0.5:
        return pre * 0.7 if form == 'legacy' else pre - 0.3 * abs(pre)
    return pre


@pytest.fixture
def study_db(tmp_path):
    optuna = pytest.importorskip('optuna')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = tmp_path / 'tiny_study.db'
    st = optuna.create_study(study_name='v2_search', direction='maximize',
                             storage=f'sqlite:///{db}')
    dists = {'forward_bars': optuna.distributions.CategoricalDistribution([12, 24]),
             'seq_len': optuna.distributions.IntDistribution(8, 40, step=2),
             'trade_threshold': optuna.distributions.FloatDistribution(0.05, 2.0)}
    order = []
    for cls, (folds, rmin, thr, l3) in CASES.items():
        attrs = {'cfg': {'forward_bars': 24, 'seq_len': 20,
                         'trade_threshold': thr, 'target_kind': 'raw'},
                 'fold_sharpes': folds, 'avg_sharpe': float(np.mean(folds)),
                 'std_sharpe': float(np.std(folds)),
                 'regime_sharpes': {'bull': 0.1, 'bear': rmin, 'sideways': 0.0,
                                    'min': rmin}}
        attrs.update(l3)
        st.add_trial(optuna.trial.create_trial(
            params={'forward_bars': 24, 'seq_len': 20, 'trade_threshold': thr},
            distributions=dists, value=_score(folds, rmin), user_attrs=attrs))
        order.append(cls)
    # two L1-only trials: a legacy-penalised one that jumps an unpenalised one
    for folds, rmin in (([-2.0, -2.1, -1.9], -3.0), ([-1.6, -1.7, -1.5], -0.1)):
        st.add_trial(optuna.trial.create_trial(
            params={'forward_bars': 12, 'seq_len': 8, 'trade_threshold': 0.3},
            distributions=dists, value=_score(folds, rmin),
            user_attrs={'cfg': {'forward_bars': 12, 'seq_len': 8,
                                'trade_threshold': 0.3, 'target_kind': 'raw'},
                        'fold_sharpes': folds,
                        'regime_sharpes': {'bull': 0.0, 'bear': rmin,
                                           'sideways': 0.0, 'min': rmin}}))
        order.append('l1')
    return db, order


def test_classifier_returns_the_constructed_class(study_db):
    db, order = study_db
    name, trials = tsd.load_trials(db, 'auto')
    assert name == 'v2_search' and len(trials) == len(order)
    rows = tsd.layer1(trials, COST, 1.2)
    for r, cls in zip(rows, order):
        tsd.classify(r, BPY, COST)
        if cls == 'l1':
            continue
        assert r['source'] == 'L3'
        assert r['primary'] == cls, (cls, r['primary'], r['flags'], r.get('pooled'))
    by = {c: r for r, c in zip(rows, order)}
    assert by['a']['economic'] == 'a0'
    assert by['b']['economic'] == 'b'
    assert by['d']['flags']['d_floor'] and not by['d']['flags']['d_cost']
    assert by['c']['flags']['c_sign']
    assert by['+']['economic'] == '+'


def test_layer1_penalty_form_backsolve_and_rank_flip(study_db):
    db, order = study_db
    _, trials = tsd.load_trials(db, 'auto')
    rows = tsd.layer1(trials, COST, 1.2)
    pen, unpen = rows[-2], rows[-1]
    assert pen['penalty_flag'] and pen['penalty_form'] == 'legacy*0.7'
    assert math.isclose(pen['backsolved_legacy'], pen['pre_penalty'], rel_tol=1e-9)
    assert not unpen['penalty_flag'] and unpen['penalty_form'] == 'none'
    # legacy *0.7 lifts -2.02 to -1.41, above the unpenalised -1.64
    assert pen['pre_penalty'] < unpen['pre_penalty'] and pen['score'] > unpen['score']
    assert pen['flags']['c_rank'] and not unpen['flags']['c_rank']
    assert pen['flags']['d_cost'] and pen['flags']['d_admit']
    tsd.classify(pen, BPY, COST)
    assert pen['primary'] == 'd' and pen['source'] == 'L1'


def test_repaired_penalty_form_detected():
    tr = {'number': 0, 'state': 'COMPLETE', 'params': {},
          'value': _score([-1.0, -1.2, -0.8], -2.0, form='repaired'),
          'attrs': {'fold_sharpes': [-1.0, -1.2, -0.8],
                    'regime_sharpes': {'bear': -2.0, 'min': -2.0},
                    'cfg': {'trade_threshold': 1.0}}}
    (row,) = tsd.layer1([tr], COST, 1.2)
    assert row['penalty_form'] == 'repaired-0.3|s|'


def test_layer2_oracle_random_construction_on_synthetic_target():
    rng = np.random.default_rng(7)
    n, fb, thr = 6000, 12, 0.5
    y = rng.normal(0.05, 2.0, n)
    f = tsd.layer2_fold(y, -1.0, thr, fb, COST, True, BPY, 30,
                        np.random.default_rng(1))
    assert f['n_possible'] == math.ceil(n / fb)
    assert math.isclose(f['pass_rate_long'], float((y > thr).mean()))
    # oracle: every entry is a row with y > thr => gross mean > thr
    assert f['oracle']['gross_mean'] > thr
    # random = permutation => identical admit COUNT (same marginal)
    perm = np.random.default_rng(3).permutation(y)
    assert int((perm > thr).sum()) == int((y > thr).sum())
    # random gross per trade ~ the unconditional mean (no selection)
    se = y.std() / math.sqrt(f['random']['n_trades'])
    assert abs(f['random']['gross_mean'] - y.mean()) < 4 * se
    assert f['oracle']['sharpe'] > f['random']['sharpe_mean']
    lo, hi = f['gross_band']
    assert lo < hi   # S < 0 => gross consistent with S rises with n
    assert f['proxy_class'] in ('a', 'b', 'a~', '?')


def test_sharpe_band_inverts_exactly():
    rng = np.random.default_rng(11)
    gross = rng.normal(0.2, 1.5, 250)
    n_rows, fb = 20000, 24
    net = gross - COST
    S = tsd.sharpe_from_trades(net, n_rows, fb, BPY)
    g = tsd._gband(S, net.std(), len(net), n_rows, fb, BPY, COST)
    assert math.isclose(g, gross.mean(), rel_tol=1e-9, abs_tol=1e-12)
    nstar = tsd.zero_skill_trades(S, gross.mean(), net.std(), n_rows, fb, BPY, COST)
    assert math.isclose(nstar, len(net), rel_tol=1e-9)


def test_fold_trade_stats_gross_net_identity_and_hold():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 2, 3000)
    p = y + rng.normal(0, 2, 3000)
    st = tsd.fold_trade_stats(p, y, 0.8, 24, COST, long_only=True)
    assert math.isclose(st['gross_ret_mean'] - st['net_ret_mean'], COST, abs_tol=1e-12)
    assert math.isclose(st['cost_drag'], COST, abs_tol=1e-12)
    assert st['n_rows'] == 3000
    assert math.isclose(st['threshold_pass_rate'], float((p > 0.8).mean()))
    assert 23.0 < st['mean_hold_bars'] <= 24.0


def test_parse_log_maps_bracket_index_to_trial_number(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text(
        "[ADAPTIVE] crypto: mode=initial, trials=40, forward_bars=[12, 18, 24]\n"
        "  [TIMEOUT] Trial 9 at epoch 8 fold 2\n"
        "[I 2026] Trial 9 finished with value: -1.1541173081919316 and parameters: {}\n"
        "[ 10] score=-1.154 (mean=-1.22 std=0.85) folds=[-2.10/-1.50/-0.07] | fb=32\n")
    info = tsd.parse_log(log)
    assert info['forward_bars'] == [12, 18, 24]
    t = info['trials'][9]
    assert t['folds_print'] == [-2.10, -1.50, -0.07]
    assert t['timeout'] == {'epoch': 8, 'fold': 2}
    assert math.isclose(t['value'], -1.1541173081919316)


class _RepoWriteAudit:
    """Record every write-mode open / rename / replace / mkdir / sqlite connect
    that THIS process makes under `root` while installed. An in-process audit
    hook (not a whole-tree mtime snapshot) so that other processes writing
    runtime files in the shared repo root (bots, other test runs, llm_cost.json,
    v2_study.db) cannot make the assertion flaky. Audit hooks cannot be
    removed, so the recorder is armed/disarmed with a flag."""

    def __init__(self, root):
        self.root = os.path.realpath(str(root)) + os.sep
        self.armed = False
        self.writes = []
        sys.addaudithook(self._hook)

    def _under_root(self, path):
        try:
            if isinstance(path, (bytes, bytearray)):
                path = path.decode('utf-8', 'replace')
            path = os.fspath(path)
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            return os.path.realpath(path).startswith(self.root)
        except Exception:
            return False

    def _hook(self, event, args):
        if not self.armed:
            return
        try:
            if event == 'open':
                path, mode = args[0], args[1] or ''
                if isinstance(path, int) or path is None:
                    return
                if isinstance(path, (bytes, bytearray)):
                    path = path.decode('utf-8', 'replace')
                if any(c in str(mode) for c in ('w', 'a', 'x', '+')) and self._under_root(path):
                    self.writes.append((event, os.fspath(path)))
            elif event in ('os.rename', 'os.replace', 'os.mkdir', 'os.remove', 'os.unlink'):
                for a in args[:2]:
                    if a is not None and not isinstance(a, int) and self._under_root(a):
                        self.writes.append((event, os.fspath(a)))
            elif event == 'sqlite3.connect':
                db = args[0] if args else ''
                if isinstance(db, (bytes, bytearray)):
                    db = db.decode('utf-8', 'replace')
                db = os.fspath(db) if not isinstance(db, str) else db
                if not db or db == ':memory:':
                    return
                if db and not db.startswith('file:') and self._under_root(db):
                    self.writes.append((event, db))
                elif db.startswith('file:') and 'mode=ro' not in db and self._under_root(db.split('?')[0][5:]):
                    self.writes.append((event, db))
        except Exception:
            pass


_AUDIT = _RepoWriteAudit(ROOT)


def test_main_exit0_no_repo_writes_db_untouched(study_db, tmp_path, capsys):
    db, _ = study_db
    db_bytes, db_mtime = db.read_bytes(), db.stat().st_mtime_ns
    out = tmp_path / 'out'
    _AUDIT.writes.clear()
    _AUDIT.armed = True
    try:
        rc = tsd.main(['--db', str(db), '--out', str(out), '--no-layer2'])
    finally:
        _AUDIT.armed = False
    assert rc == 0
    assert (out / 'trial_decomposition_crypto.json').exists()
    assert (out / 'tiny_study_snapshot.db').exists()
    assert db.read_bytes() == db_bytes and db.stat().st_mtime_ns == db_mtime
    # the tool itself must not write anywhere under the repo root
    assert _AUDIT.writes == [], _AUDIT.writes
    txt = capsys.readouterr().out
    assert 'PRIMARY histogram' in txt and 'PRE-REGISTERED' in txt
    # missing DB and garbage input: still exit 0
    assert tsd.main(['--db', str(tmp_path / 'nope.db'), '--out', str(out)]) == 0
    bad = tmp_path / 'bad.db'
    bad.write_bytes(b'not a sqlite file')
    assert tsd.main(['--db', str(bad), '--out', str(out), '--no-layer2']) == 0
