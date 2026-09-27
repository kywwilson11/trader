"""INTEL 2026-09 W6 — llm_eval verdict honesty (Scout B E1) + --out.

  1. Report-only size-audit p-values in the b2 block (b2_dk_fixedb_p,
     b2_ewc_p beside the existing b2_im_p): present, None-safe, never raise,
     and the pre-existing report dict is byte-identical minus the new keys
     (golden captured from the pre-edit llm_eval.py on the Jetson,
     2026-09-27). The verdict carrier (DK/t_{G-1} p_value) is unchanged.
  2. scripts/har_size_audit.py runs end-to-end, reuses llm_eval's
     estimators by import, writes JSON, exits 0.
  3. llm_eval --out DIR: default report path unchanged (repo root, what
     gui.py reads), --out redirects both reports, DIR is created.

Mac-runnable: stdlib + numpy/scipy only, no Alpaca.
"""
import importlib.util
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

import llm_eval

REPO = Path(__file__).resolve().parent.parent
_LLM_EVAL_SRC = Path(llm_eval.__file__).read_text()
NEW_ENC_KEYS = {'b2_dk_fixedb_p', 'dk_fixedb_bandwidth', 'dk_fixedb_b',
                'b2_ewc_p', 'ewc_nu'}

# compute_incremental_report(_panel(), forward_bars=6) from the PRE-edit
# llm_eval.py (json.dumps(sort_keys=True)).
_GOLDEN_PRE = json.loads(r'''{"echo_gap": 0.0082, "echo_gap_abs": 0.0082, "effective_n_hint": 26.5, "encompassing": {"b1_pred": 0.38899, "b2_im_mean": 0.05258, "b2_im_p": 0.1802, "b2_s": 0.05708, "dof": 159, "estimator": "driscoll_kraay", "g_clusters": 160, "im_blocks_used": 8, "p_value": 0.2374, "se_hac": 0.04813, "t": 1.186}, "grid": {"agree_bear": {"avg_fwd_ret_pct": -0.408, "n": 113}, "agree_bull": {"avg_fwd_ret_pct": 0.309, "n": 134}, "llm_bear_ml_bull": {"avg_fwd_ret_pct": 0.19, "n": 117}, "llm_bull_ml_bear": {"avg_fwd_ret_pct": -0.296, "n": 116}}, "hac_lag_hours": 5, "hac_lag_rows": 5, "legacy_b2": {"b2_s": 0.05708, "dof": 477, "estimator": "newey_west_rows", "hac_lag_rows": 5, "note": "deprecated rows-HAC \u2014 printed alongside for one release (c26 D09)", "p_value": 0.2548, "se_hac": 0.05007, "t": 1.14}, "min_n": 60, "n": 480, "n_clusters": 160, "n_distinct_t0": 160, "n_dropped": {"nonfinite": 0, "pred_none": 0, "realized_none": 0, "s_none": 0}, "n_input": 480, "n_s_exactly_half": 0, "partial_spearman_s_given_pred": 0.0528, "pred_degenerate": false, "pseudo_replication": true, "raw_spearman_s_vs_return": 0.061, "rows_per_t0": 3.0, "s_degenerate": false, "span_hours": 159.0, "time_ordered": true, "verdict": "no measurable incremental value beyond the ML pred at this sample \u2014 candidate to disable and save the spend"}''')


def _panel(T=160, K=3, seed=11):
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal((T, K))
    s = 0.5 + 0.2 * rng.standard_normal((T, K))
    y = 0.3 * pred + 0.4 * (s - 0.5) + rng.standard_normal((T, K))
    t0 = 1.7e9 + 3600.0 * np.repeat(np.arange(T), K)
    return list(zip(np.round(s.ravel(), 6).tolist(),
                    np.round(y.ravel(), 6).tolist(),
                    np.round(pred.ravel(), 6).tolist(), t0.tolist()))


# --------------------------------------------------------------------------- #
# 1. size-audit fields
# --------------------------------------------------------------------------- #

def test_existing_report_byte_identical_minus_new_keys():
    rep = llm_eval.compute_incremental_report(_panel(), forward_bars=6)
    rep = json.loads(json.dumps(rep))
    for k in NEW_ENC_KEYS:
        rep['encompassing'].pop(k)
    assert json.dumps(rep, sort_keys=True) == json.dumps(_GOLDEN_PRE, sort_keys=True)


def test_new_fields_present_named_and_in_range():
    enc = llm_eval.compute_incremental_report(_panel(), forward_bars=6)['encompassing']
    assert NEW_ENC_KEYS <= set(enc)
    assert 'b2_im_p' in enc                                  # existing IM field
    G = 160
    assert enc['ewc_nu'] == math.floor(0.4 * G ** (2 / 3))   # 11
    assert enc['dk_fixedb_bandwidth'] == max(round(1.3 * math.sqrt(G)), 2 * 6)
    assert enc['dk_fixedb_b'] == round(enc['dk_fixedb_bandwidth'] / G, 4)
    for k in ('b2_dk_fixedb_p', 'b2_ewc_p', 'b2_im_p'):
        assert 0.0 <= enc[k] <= 1.0


def test_verdict_carrier_ignores_new_fields(monkeypatch):
    base = llm_eval.compute_incremental_report(_panel(), forward_bars=6)
    monkeypatch.setattr(llm_eval, '_dk_fixedb_pvalue',
                        lambda *a, **k: {'b2_dk_fixedb_p': 0.0})
    monkeypatch.setattr(llm_eval, '_ewc_pvalue', lambda *a, **k: {'b2_ewc_p': 0.0})
    forced = llm_eval.compute_incremental_report(_panel(), forward_bars=6)
    assert forced['verdict'] == base['verdict']
    assert forced['encompassing']['p_value'] == base['encompassing']['p_value']


def test_kv_pvalue_exact_at_5pct_knot_and_monotone():
    f = llm_eval._kv_fixedb_bartlett_pvalue
    for b in (0.05, 0.2, 0.4, 0.8, 1.0):
        cv = 1.9600 + 2.9694 * b + 0.4160 * b * b - 0.5324 * b ** 3
        assert abs(f(cv, b) - 0.05) < 1e-3
        assert f(cv * 1.01, b) < 0.05 < f(cv * 0.99, b)
        ps = [f(t, b) for t in np.linspace(0, 12, 60)]
        assert all(p2 <= p1 for p1, p2 in zip(ps, ps[1:]))
    assert abs(f(1.96, 1e-9) - 0.05) < 1e-3                  # b -> 0: normal
    assert f(2.0, 0.0) is None and f(2.0, 1.2) is None       # outside KV fit
    assert f(None, 0.2) is None and f(float('nan'), 0.2) is None


def test_small_T_none_safe_and_never_raise():
    # b = M/T > 1 at tiny T -> fixed-b p None; EWC nu < 1 at G=2 -> None.
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(8), rng.standard_normal(8), rng.standard_normal(8)])
    resid = rng.standard_normal(8)
    cid = np.repeat(np.arange(2), 4)
    fb = llm_eval._dk_fixedb_pvalue(X, resid, cid, 0.1, 24)
    assert fb['b2_dk_fixedb_p'] is None and fb['dk_fixedb_b'] > 1
    assert llm_eval._ewc_pvalue(X, resid, cid, 0.1) == {'b2_ewc_p': None, 'ewc_nu': None}
    # garbage never raises
    assert llm_eval._dk_fixedb_pvalue(None, None, None, None, 24)['b2_dk_fixedb_p'] is None
    assert llm_eval._ewc_pvalue(X, resid[:3], cid, 0.1)['b2_ewc_p'] is None
    # a tiny timestamped panel through the full report path does not raise
    rows = [(0.4 + 0.02 * i, 0.1 * ((-1) ** i), 0.01 * i, 1.7e9 + 3600.0 * (i // 3))
            for i in range(12)]
    enc = llm_eval.compute_incremental_report(rows, forward_bars=24)['encompassing']
    assert enc['b2_dk_fixedb_p'] is None


def test_size_audit_line_format():
    line = llm_eval._size_audit_line({'p_value': 0.0412, 'b2_dk_fixedb_p': 0.12,
                                      'b2_ewc_p': None, 'b2_im_p': 0.3})
    assert line == ('size-audit: DK p=0.041, fixed-b p=0.120, EWC p=n/a, '
                    'IM p=0.300 (verdict still DK; see har_size_audit)')
    assert 'print(_size_audit_line(enc))' in inspect.getsource(llm_eval.run_eval)


# --------------------------------------------------------------------------- #
# 2. scripts/har_size_audit.py
# --------------------------------------------------------------------------- #

def _load_audit():
    spec = importlib.util.spec_from_file_location(
        'har_size_audit', REPO / 'scripts' / 'har_size_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_har_size_audit_runs_and_reuses_llm_eval(tmp_path, capsys):
    mod = _load_audit()
    assert mod.L is llm_eval
    src = (REPO / 'scripts' / 'har_size_audit.py').read_text()
    for fn in ('_driscoll_kraay_se', '_im_block_pvalue', '_ewc_pvalue',
               '_dk_fixedb_pvalue'):
        assert f'def {fn}' not in src                        # reused, not copied
    out = tmp_path / 'a.json'
    rc = mod.main(['--K', '2', '--fb', '3', '--n-eff', '20', '--reps', '3',
                   '--seed', '1', '--json', str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    cell = data['cells'][0]
    assert cell['T_clusters'] == 60 and cell['reps'] == 3
    assert set(cell['size']) == {'dk_prod', 'dk_fixedb', 'ewc', 'im'}
    assert set(data['fit_at_n_eff_20']) == set(cell['size'])
    assert '[3%, 7%]' in data['rule']
    assert 'size@n_eff=20' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3. --out
# --------------------------------------------------------------------------- #

def test_default_report_path_unchanged():
    root = Path(llm_eval.__file__).resolve().parent
    for name in ('llm_eval_report.json', 'llm_advisor_report.json'):
        assert str(llm_eval._report_path(llm_eval.BASE_DIR / name)) == str(root / name)
        # every write site keeps the BASE_DIR literal as its default
        assert _LLM_EVAL_SRC.count(f"_report_path(BASE_DIR / '{name}', out_dir)") >= 3
    # gui.py reads exactly these root paths
    gui_src = (REPO / 'gui.py').read_text()
    assert 'BASE_DIR / "llm_eval_report.json"' in gui_src
    assert 'BASE_DIR / "llm_advisor_report.json"' in gui_src


def test_out_redirects_and_creates_dir(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    jdir = tmp_path / 'journals'
    jdir.mkdir()
    monkeypatch.setattr(llm_eval, 'BASE_DIR', root)
    monkeypatch.setattr(llm_eval, 'JOURNAL_DIR', jdir)
    out = tmp_path / 'x' / 'y'
    assert llm_eval._report_path(root / 'r.json', out) == out / 'r.json' and out.is_dir()
    out2 = tmp_path / 'o2'
    llm_eval.main(['--days', '0', '--out', str(out2)])
    llm_eval.main(['--days', '0', '--advisor', '--out', str(out2)])
    assert json.loads((out2 / 'llm_eval_report.json').read_text())['verdict'] == 'no_data'
    assert json.loads((out2 / 'llm_advisor_report.json').read_text())['verdict'] == 'no_data'
    assert not (root / 'llm_eval_report.json').exists()
    assert not (root / 'llm_advisor_report.json').exists()
    llm_eval.main(['--days', '0'])                           # default -> root
    assert (root / 'llm_eval_report.json').exists()


def test_run_eval_full_report_honours_out_and_prints_size_audit(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'root'
    root.mkdir()
    monkeypatch.setattr(llm_eval, 'BASE_DIR', root)
    monkeypatch.setattr(llm_eval, 'JOURNAL_DIR', tmp_path)
    monkeypatch.setattr(llm_eval, '_read_daily_cost', lambda: None)   # no repo-root lock touch
    rng = np.random.default_rng(3)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    syms = ['AAA/USD', 'BBB/USD', 'CCC/USD']
    today = datetime.now().date().isoformat()
    with open(tmp_path / f'{today}.jsonl', 'w') as f:
        for i in range(40):
            ts = datetime.fromtimestamp(base + 3600.0 * i, tz=timezone.utc)
            f.write(json.dumps({
                'action': 'llm_analysis', 'ts': ts.isoformat(), 'asset_type': 'crypto',
                'forward_bars': 6,
                'scores': {s: {'s': float(rng.uniform(0.2, 0.8)),
                               'pred': float(rng.normal(0, 0.5))} for s in syms}}) + '\n')
    grid = base - 3 * 86400 + np.arange(24 * 14) * 3600.0
    paths = {s: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(grid)))) for s in syms}
    monkeypatch.setattr(llm_eval, '_bars_lookup',
                        lambda api, symbol, asset_type, start, end: (grid, paths[symbol]))
    out = tmp_path / 'out'
    rep = llm_eval.run_eval(days=0, api=object(), out_dir=out)
    assert rep['incremental']['encompassing']['estimator'] == 'driscoll_kraay'
    written = json.loads((out / 'llm_eval_report.json').read_text())
    assert NEW_ENC_KEYS <= set(written['incremental']['encompassing'])
    assert not (root / 'llm_eval_report.json').exists()
    assert 'size-audit: DK p=' in capsys.readouterr().out
