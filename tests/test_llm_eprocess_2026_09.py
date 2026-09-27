"""INTEL W10 (2026-09): llm_eprocess — anytime-valid LLM-spend ledger.

Measurement-only module; these tests pin (a) e-process validity on synthetic
nulls, (b) Scout C's 90-vs-180-day power direction, (c) the signed-sheet
refusal, (d) the params round-trip, (e) d_t on a hand-computed 3-day journal.
Pure numpy; ~10 s on the Jetson.
"""
import copy
import datetime as dt
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import llm_eprocess as E

ROOT = Path(E.__file__).resolve().parent
ALPHA = 0.025


@pytest.fixture(scope='module')
def cfg():
    return E.config_from_sheet(E.load_sheet(E.DEFAULT_PARAMS_PATH))


# --- params sheet ---------------------------------------------------------
def test_committed_sheet_equals_default_and_roundtrips(tmp_path):
    sheet = E.load_sheet(E.DEFAULT_PARAMS_PATH)
    assert sheet == E.DEFAULT_SHEET
    for k in ('signed_by', 'signed_at', 'registration_id', 'registration_sha'):
        assert sheet[k] is None                      # unsigned: this module never signs
    for k, v in sheet.items():
        if isinstance(v, dict):
            assert 'value' in v and v.get('reason'), k
    p = tmp_path / 'p.json'
    p.write_text(json.dumps(sheet, indent=2))
    again = E.load_sheet(p)
    assert again == sheet
    assert E.config_from_sheet(again) == E.config_from_sheet(sheet)
    assert E.registration_sha(again) == E.registration_sha(sheet)
    c = E.config_from_sheet(sheet)
    assert (c['threshold'], c['horizon'], c['burn_days'], c['delta']) == (40.0, 180, 10, 1.0)
    assert c['exclude_through_utc'] == '2026-05-07'


def test_config_rejects_inconsistent_threshold_and_default():
    bad = copy.deepcopy(E.DEFAULT_SHEET)
    bad['alpha']['value']['threshold'] = 20.0
    with pytest.raises(ValueError):
        E.config_from_sheet(bad)
    bad = copy.deepcopy(E.DEFAULT_SHEET)
    bad['inconclusive_default']['value'] = 'KILL_HARM'
    with pytest.raises(ValueError):
        E.config_from_sheet(bad)


# --- refusal ----------------------------------------------------------------
def _signed(sheet):
    s = copy.deepcopy(sheet)
    s.update(signed_by='owner', signed_at='2026-10-01T00:00:00Z',
             registration_id='test-reg')
    s['registration_sha'] = E.registration_sha(s)
    return s


def test_live_refuses_unsigned_exit3(tmp_path, capsys):
    out = tmp_path / 'o.json'
    rc = E.main(['--journals', str(tmp_path), '--days', '5',
                 '--params', str(E.DEFAULT_PARAMS_PATH), '--json', str(out)])
    assert rc == 3
    assert 'REFUSED (NOT_SIGNED)' in capsys.readouterr().err
    assert json.loads(out.read_text())['verdict'] == 'NOT_SIGNED'


def test_live_refuses_tampered_sheet_exit3(tmp_path, capsys):
    s = _signed(E.DEFAULT_SHEET)
    assert E.signature_status(s) == (True, 'signed')
    s['delta_bp_per_day']['value'] = 0.5             # edit after signing
    ok, why = E.signature_status(s)
    assert not ok and 'tampered' in why
    p = tmp_path / 'tampered.json'
    p.write_text(json.dumps(s))
    rc = E.main(['--journals', str(tmp_path), '--days', '5', '--params', str(p)])
    assert rc == 3
    assert 'tampered' in capsys.readouterr().err
    p.write_text('{not json')
    assert E.main(['--journals', str(tmp_path), '--days', '5', '--params', str(p)]) == 3


def test_print_registration_sha_writes_nothing(tmp_path, capsys):
    assert E.main(['--print-registration-sha', '--params',
                   str(E.DEFAULT_PARAMS_PATH)]) == 0
    assert capsys.readouterr().out.strip() == E.registration_sha(E.DEFAULT_SHEET)


def test_usage_error_exit2(capsys):
    assert E.main([]) == 2


# --- e-process validity (synthetic nulls) -----------------------------------
R_NULL = 2000
SE_NULL = math.sqrt(ALPHA * (1 - ALPHA) / R_NULL)     # 0.35 pp


def test_nulls_false_fire_within_alpha(cfg):
    res = E.selftest(cfg, reps=R_NULL, seed=11)['by_mu']
    bound = ALPHA + 2 * SE_NULL                        # 3.2 %
    assert res['keep_null']['fire_by_horizon']['KEEP'] <= bound
    assert res['harm_null']['fire_by_horizon']['KILL_HARM'] <= bound
    assert res['futility_null']['fire_by_horizon']['KILL_FUTILITY'] <= bound


def test_null_false_fire_dependent_mds(cfg):
    """Weak-null validity needs only E[d_t | past] at the null, not
    independence: volatility-clustered martingale-difference noise (scale
    doubles after a large day). NB an AR(1) d with zero UNCONDITIONAL mean is
    NOT in the weak null (its conditional mean is 0.3*d_{t-1}); the W10
    report records its over-rejection separately."""
    rng = np.random.default_rng(5)
    T = cfg['burn_days'] + cfg['horizon']
    n, lots = E.synth_noise(rng, R_NULL, T)
    med = np.median(np.abs(n))
    d = np.empty_like(n)
    d[:, 0] = n[:, 0]
    for t in range(1, T):
        d[:, t] = n[:, t] * np.where(np.abs(d[:, t - 1]) > med, 1.5, 0.75)
    bound = ALPHA + 2 * SE_NULL
    res = E.run_ledger_batch(d, d - 0.1, lots, cfg)
    assert np.mean(res['first']['KILL_HARM'] > 0) <= bound
    keep = E.run_ledger_batch(d + 0.1, d, lots, cfg)
    assert np.mean(keep['first']['KEEP'] > 0) <= bound
    fut = E.run_ledger_batch(d + cfg['delta'], d + cfg['delta'] - 0.1, lots, cfg)
    assert np.mean(fut['first']['KILL_FUTILITY'] > 0) <= bound


def test_agrapa_bernoulli_null_and_formula():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, size=(4000, 300)).astype(float)
    K = E.agrapa_eprocess(x, 0.5, True)
    assert np.mean(K.max(axis=1) >= 40) <= ALPHA + 2 * math.sqrt(ALPHA * (1 - ALPHA) / 4000)
    # first two steps by hand: lam_1 = 0 (prior mean 1/2 == m0), then
    # mu_1 = (1/2 + x1)/2, var_1 = (1/4 + (x1 - mu_1)^2)/2
    xs = np.array([1.0, 1.0])
    K2 = E.agrapa_eprocess(xs, 0.4, True)
    lam1 = min((0.5 - 0.4) / (0.25 + 0.01), 0.5 / 0.4)
    mu1 = (0.5 + 1.0) / 2
    var1 = (0.25 + (1.0 - mu1) ** 2) / 2
    lam2 = min((mu1 - 0.4) / (var1 + (mu1 - 0.4) ** 2), 0.5 / 0.4)
    assert K2[0] == pytest.approx(1 + lam1 * 0.6)
    assert K2[1] == pytest.approx((1 + lam1 * 0.6) * (1 + lam2 * 0.6))
    # downward mirror is nonnegative and capped at c/(1-m0)
    Kd = E.agrapa_eprocess(np.zeros(50), 0.9, False)
    assert np.all(Kd > 0) and np.all(np.diff(Kd) >= 0)


# --- power: horizon 180 vs 90 (Scout C: 91 % vs 43 % KEEP at +1 bp/day) ----
def test_power_plus1_horizon_180_vs_90(cfg):
    res = E.selftest(cfg, reps=1000, seed=21, mus=[('plus1', 1.0)])['by_mu']['plus1']
    p180 = res['fire_by_horizon']['KEEP']
    p90 = res['fire_by_interim']['KEEP']              # sequential rules do not depend on H before H
    assert abs(p180 - 0.91) <= 0.06, p180
    assert abs(p90 - 0.43) <= 0.08, p90
    assert p180 - p90 >= 0.30
    h90 = E.selftest(cfg, reps=1000, seed=22, mus=[('plus1', 1.0)],
                     horizon=90)['by_mu']['plus1']['fire_by_horizon']['KEEP']
    assert abs(h90 - 0.43) <= 0.08, h90              # an actual horizon-90 run agrees


# --- decision rules ---------------------------------------------------------
def test_clip_rule_voids_keep(cfg):
    c = dict(cfg)
    nb = c['burn_days']
    d = np.full(nb + 60, 5.0)
    d[:nb] = 0.0                                       # B -> floor 2.5*delta
    d[nb::10] = -100.0                                 # 10 % lower clips
    res = E.run_ledger_batch(d, d, np.full(d.shape, 100.0), c)
    assert res['E_keep'].max() >= c['threshold']
    assert res['first']['KEEP'][0] == 0               # voided: favouring clip rate 10 % > 5 %
    d2 = np.where(d == -100.0, 5.0, d)
    assert E.run_ledger_batch(d2, d2, np.full(d.shape, 100.0), c)['verdict'][0] == 'KEEP'


def test_horizon_fallback_and_inconclusive_default(cfg):
    c = dict(cfg, burn_days=3, horizon=40, threshold=1e12)   # sequential disabled
    rng = np.random.default_rng(0)
    d = rng.normal(0, 6, 43)
    d[:3] = [-6.0, 0.0, 6.0]                                  # MAD 6 -> B ~ 44 bp, no clipping
    d[3:] += 0.5 - d[3:].mean()                               # inside (0, delta): no test rejects
    lots = np.full((1, 43), 100.0)
    r = E.run_ledger_batch(d, d - 0.1, lots, c)
    assert (r['verdict'][0], r['basis'][0]) == ('CONTINUE', 'horizon_inconclusive_no_default')
    r = E.run_ledger_batch(d, d - 0.1, lots, dict(c, inconclusive_default='KEEP'))
    assert (r['verdict'][0], r['basis'][0]) == ('KEEP', 'inconclusive_default')
    up = d + 5.0
    r = E.run_ledger_batch(up, up - 0.1, lots, c)
    assert r['verdict'][0] == 'KEEP' and r['basis'][0].startswith('fallback_')
    short = E.run_ledger_batch(d[:20], d[:20], lots[:, :20], c)
    assert short['verdict'][0] == 'CONTINUE' and short['basis'][0] == 'running'


def test_t_sf_matches_scipy():
    st = pytest.importorskip('scipy.stats')
    for t in (-3.1, -0.4, 0.0, 0.7, 1.96, 4.2):
        for df in (3, 7, 29, 169):
            assert E.t_sf(t, df) == pytest.approx(float(st.t.sf(t, df)), abs=1e-10)


def test_hedged_cs_covers_mean():
    rng = np.random.default_rng(9)
    L, U = E.hedged_cs(rng.beta(2, 2, 400))
    assert np.all(np.diff(L) >= 0) and np.all(np.diff(U) <= 0)
    assert L[-1] < 0.5 < U[-1] and U[-1] - L[-1] < 0.25


# --- d_t on a synthetic 3-day journal fixture (hand-computed) ---------------
def _write_day(dirp, day, rows, gz=False):
    txt = ''.join(json.dumps(r) + '\n' for r in rows)
    if gz:
        with gzip.open(dirp / f'{day}.jsonl.gz', 'wt') as f:
            f.write(txt)
    else:
        (dirp / f'{day}.jsonl').write_text(txt)


def test_daily_tilt_three_day_fixture(tmp_path):
    jd = tmp_path / 'journals'
    jd.mkdir()
    _write_day(jd, '2026-04-20', [   # legacy window: must be excluded
        {'ts': '2026-04-20T12:00:00+00:00', 'action': 'buy', 'symbol': 'BTC/USD',
         'llm_multiplier': 2.0, 'final_notional': 9e9, 'fill_price': 1.0}])
    _write_day(jd, '2026-10-01', [
        {'ts': '2026-10-01T10:00:00+00:00', 'action': 'buy', 'symbol': 'BTC/USD',
         'llm_multiplier': 1.25, 'final_notional': 1000.0, 'fill_price': 100.0},
        {'ts': '2026-10-01T11:00:00+00:00', 'action': 'llm_analysis', 'cost_usd': 0.01},
        {'ts': '2026-10-01T12:00:00+00:00', 'action': 'skip', 'symbol': 'ETH/USD'}])
    _write_day(jd, '2026-10-02', [
        {'ts': '2026-10-02T15:00:00+00:00', 'action': 'buy', 'symbol': 'AAPL',
         'llm_multiplier': 0.75, 'final_notional': 2000.0, 'fill_price': 200.0},
        {'ts': '2026-10-02T16:00:00+00:00', 'action': 'llm_analysis', 'cost_usd': 0.02}])
    _write_day(jd, '2026-10-03', [
        {'ts': '2026-10-03T09:00:00+00:00', 'action': 'sell', 'symbol': 'BTC/USD',
         'fill_price': 110.0}], gz=True)
    rows, hashes = E.load_journal_rows(jd, dt.date(2026, 4, 1), dt.date(2026, 10, 3),
                                       exclude_through='2026-05-07')
    assert len(rows) == 5 and '2026-10-03.jsonl' in hashes and '2026-04-20.jsonl' in hashes
    lots = E.build_lots(rows, lambda asset, row: 0.2)          # 0.2 % round trip
    assert [(l['symbol'], l['asset']) for l in lots] == [('BTC/USD', 'crypto'), ('AAPL', 'stock')]
    days = [dt.date(2026, 10, 1), dt.date(2026, 10, 2), dt.date(2026, 10, 3)]
    marks = {'BTC/USD': {days[0]: 104.0, days[1]: 102.0, days[2]: 111.0},
             'AAPL': {days[1]: 190.0, days[2]: 195.0}}
    eq = {d: 1e5 for d in days}
    s = E.daily_tilt_series(lots, days, marks, eq, E.daily_costs(rows),
                            c_hi_usd=1.0, m_bar=1.0)
    tau_b = 10 * (1 - 1 / 1.25)                          # +2
    tau_a = 10 * (1 - 1 / 0.75)                          # -10/3
    phi = 0.001                                           # half of 0.2 %
    g1 = tau_b * (104 - 100) - phi * tau_b * 100
    g2 = tau_b * (102 - 104) + tau_a * (190 - 200) - phi * abs(tau_a) * 200
    g3 = tau_b * (110 - 102) - phi * tau_b * 110 + tau_a * (195 - 190)
    assert s['G'] == pytest.approx([g1, g2, g3])
    assert s['d_lo'] == pytest.approx([1e4 * (g1 - .01) / 1e5, 1e4 * (g2 - .02) / 1e5,
                                       1e4 * g3 / 1e5])
    assert s['d_lo'] == pytest.approx([0.779, 2.8646667, -0.0886667], abs=1e-6)
    assert s['d_hi'] == pytest.approx(1e4 * (np.array([g1, g2, g3]) - 1.0) / 1e5)
    assert list(s['lots_cum']) == [1, 2, 2]
    # budget-neutral primary: burn-in mean multiplier
    assert E.burn_in_m_bar(lots, days[0], 10) == pytest.approx(1.0)
    s2 = E.daily_tilt_series(lots, days, marks, eq, {}, 1.0, m_bar=1.25)
    assert s2['G'][0] == pytest.approx(0.0)              # tau_BTC = 0 at m_bar = m


def test_ledger_report_json_shape(cfg):
    days = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(25)]
    rng = np.random.default_rng(2)
    d = rng.normal(0, 2, 25)
    ser = {'d_lo': d, 'd_hi': d - 0.1, 'lots_cum': np.arange(1, 26) * 3.0}
    body = E.ledger_report(cfg, days, ser, m_bar=1.1)
    assert body['verdict'] in E.VERDICTS
    assert len(body['days']) == 25 and 'E_keep' not in body['days'][9]
    last = body['days'][-1]
    assert last['cs_day'] == 15 and {'E_keep', 'E_harm', 'E_futility',
                                     'cs_lo_bp', 'cs_hi_bp'} <= set(last)
    assert last['cs_lo_bp'] <= last['cs_hi_bp']
    json.dumps(body)


# --- CLI + measurement-only --------------------------------------------------
def test_selftest_cli_writes_only_requested_json(tmp_path, capsys):
    out = tmp_path / 'st.json'
    assert E.main(['--selftest', '--reps', '50', '--json', str(out)]) == 0
    res = json.loads(out.read_text())
    assert set(res['by_mu']) == {'keep_null', 'harm_null', 'futility_null',
                                 'plus1', 'minus1'}
    assert res['signed'] is False
    assert 'scenario' in capsys.readouterr().out


def test_nothing_in_repo_reads_the_verdict():
    pat = re.compile(r'^\s*(import|from)\s+llm_eprocess\b|llm_eprocess_report', re.M)
    hits = [p for p in list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
            if p.name != 'llm_eprocess.py' and pat.search(p.read_text(errors='replace'))]
    assert hits == []
    src = (ROOT / 'llm_eprocess.py').read_text()
    assert not re.search(r'^\s*(import|from)\s+(llm_client|torch)\b', src, re.M)
