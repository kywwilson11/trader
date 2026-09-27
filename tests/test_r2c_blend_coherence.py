"""Packet R2C-02 — blend/certificate coherence (H1, H2, M2, M4-guard, L4, L9).

Mac-runnable: pure numpy + source pins. NEVER imports scripts/hypersearch_v2
(torch) or lightgbm — the save-block wiring is pinned by sparing
source-structure asserts; the pure decision logic lives in blend_fit and is
tested directly (including the H1 failure path with stub boosters).

[A] blend_fit.DEFAULT_LSTM_WEIGHT / effective_lstm_weight (H1 cert==deploy)
[B] H1 failure-path flow simulation with stub boosters
[C] _policy_sharpe strict '>' (L9 — deployed-path parity)
[D] fit_blend_weight_v2 kish_divisor (L4 — default None byte-identical)
[E] blend_fit.reselect_trade_threshold (H2 grid kernel)
[F] strategy_config default-OFF pins (BLEND_FIT_ON_REFIT / _THRESHOLD_RESELECT)
[G] sparing source-structure asserts on scripts/hypersearch_v2.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blend_fit import (DEFAULT_LSTM_WEIGHT, _policy_sharpe,
                       effective_lstm_weight, fit_blend_weight_v2,
                       reselect_trade_threshold)

HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()


# ---------------------------------------------------------------------------
# [A] DEFAULT_LSTM_WEIGHT / effective_lstm_weight
# ---------------------------------------------------------------------------

class TestEffectiveLstmWeight:
    def test_default_matches_every_serving_side_literal(self):
        # The constant is the certificate-visible home of the serving-side
        # fallback: model_lgb.ensemble_predict's keyword default and the
        # predict_now/backtest cfg.get literals must all agree with it
        # (B12.2: a future 0.6 -> 0.5 owner decision changes them in
        # lockstep, starting from DEFAULT_LSTM_WEIGHT).
        ml = (REPO / 'model_lgb.py').read_text()
        m = re.search(r'lstm_weight:\s*float\s*=\s*([0-9.]+)', ml)
        assert m and float(m.group(1)) == DEFAULT_LSTM_WEIGHT
        for fname in ('predict_now.py', 'backtest.py'):
            src = (REPO / fname).read_text()
            lits = re.findall(r"get\('lstm_weight',\s*([0-9.]+)\)", src)
            assert lits, f'{fname} lost its lstm_weight fallback literal'
            for lit in lits:
                assert float(lit) == DEFAULT_LSTM_WEIGHT, (
                    f'{fname} fallback {lit} != DEFAULT_LSTM_WEIGHT')

    def test_no_boosters_passthrough(self):
        # Legacy raw-LSTM path: nothing ships, nothing is invented.
        assert effective_lstm_weight(None, False) is None
        assert effective_lstm_weight(0.42, False) == 0.42

    def test_boosters_shipping_resolves_none_to_default(self):
        assert effective_lstm_weight(None, True) == DEFAULT_LSTM_WEIGHT

    def test_boosters_shipping_keeps_fitted_weight(self):
        assert effective_lstm_weight(0.42, True) == 0.42

    def test_default_override(self):
        assert effective_lstm_weight(None, True, default=0.5) == 0.5

    def test_returns_float_type(self):
        w = effective_lstm_weight(np.float64(0.37), True)
        assert isinstance(w, float) and w == pytest.approx(0.37)


# ---------------------------------------------------------------------------
# [B] H1 failure-path cert==deploy flow (stub boosters)
# ---------------------------------------------------------------------------

class _StubBooster:
    """Stands in for a trained lightgbm.Booster in the lgb_pack tuple."""


def _save_block_weight_flow(v3, lgb_pack, fitted_weight):
    """Mirror of the hypersearch_v2 save-block H1 logic (the source pins in
    [G] keep this replica honest): returns (cert_w, deploy_w) where cert_w
    is the lstm_weight kwarg evaluate_on_holdout receives and deploy_w is
    what live serving resolves from the shipped config + booster files."""
    lstm_weight = fitted_weight
    if v3:
        ship = bool(lgb_pack and lgb_pack[0] is not None)
        lstm_weight = effective_lstm_weight(lstm_weight, ship)
    cert_w = lstm_weight  # -> evaluate_on_holdout(lstm_weight=...)
    config = {}
    if v3 and lstm_weight is not None:
        config['lstm_weight'] = round(float(lstm_weight), 4)
    boosters_on_disk = bool(v3 and lgb_pack and lgb_pack[0] is not None)
    deploy_w = (config.get('lstm_weight', DEFAULT_LSTM_WEIGHT)
                if boosters_on_disk else None)  # predict_now semantics
    return cert_w, deploy_w


class TestH1FailurePathCertEqualsDeploy:
    def test_blend_fit_failure_with_stub_boosters(self):
        # THE H1 defect: fit failed (weight None) but boosters ship ->
        # pre-fix the certificate scored raw LSTM (cert_w None) while live
        # blended at 0.6. Post-fix both sides see the same default.
        lgb_pack = (_StubBooster(), None, None, None)
        cert_w, deploy_w = _save_block_weight_flow(True, lgb_pack, None)
        assert cert_w == deploy_w == DEFAULT_LSTM_WEIGHT

    def test_fit_success_unchanged(self):
        lgb_pack = (_StubBooster(), _StubBooster(), -0.8, 1000)
        cert_w, deploy_w = _save_block_weight_flow(True, lgb_pack, 0.55)
        assert cert_w == deploy_w == 0.55

    def test_no_lgb_pack_stays_raw_lstm(self):
        # LGB training failed entirely: no boosters ship, raw-LSTM
        # certificate is honest (nothing blends live).
        for pack in (None, (None, None, None, None)):
            cert_w, deploy_w = _save_block_weight_flow(True, pack, None)
            assert cert_w is None and deploy_w is None

    def test_legacy_v3_off_untouched(self):
        cert_w, deploy_w = _save_block_weight_flow(False, None, None)
        assert cert_w is None and deploy_w is None


# ---------------------------------------------------------------------------
# [C] _policy_sharpe strict '>' (L9)
# ---------------------------------------------------------------------------

class TestPolicySharpeStrictGreater:
    def test_exact_threshold_rows_not_taken(self):
        # Every deployed path (objective_utils.simulate_trades_core,
        # backtest, predict_now) uses strict '>'; pred == threshold must
        # not enter. All rows exactly at threshold -> zero takes -> 0.0.
        pred = np.full(50, 0.5)
        y = np.ones(50)
        assert _policy_sharpe(pred, y, 0.5) == 0.0

    def test_rows_above_threshold_still_taken(self):
        rng = np.random.default_rng(0)
        pred = np.concatenate([np.full(30, 0.5), np.full(30, 0.6)])
        y = np.concatenate([-np.ones(30), rng.normal(1.0, 0.1, 30)])
        s = _policy_sharpe(pred, y, 0.5)
        assert s > 0  # only the 0.6 (winning) rows enter

    def test_matches_simulate_trades_core_take_set(self):
        from objective_utils import simulate_trades_core
        rng = np.random.default_rng(1)
        pred = rng.choice([0.3, 0.5, 0.7], size=200)
        y = rng.normal(size=200)
        # fb=1, no cost: entry set == {pred > thr} rows exactly.
        _, entries = simulate_trades_core(pred, y, 0.5, 1, 0.0,
                                          long_only=True)
        assert np.array_equal(entries, np.flatnonzero(pred > 0.5))


# ---------------------------------------------------------------------------
# [D] fit_blend_weight_v2 kish_divisor (L4)
# ---------------------------------------------------------------------------

def _panel(seed=12, n=3000):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 0.7 * a + 0.3 * b + rng.normal(0, 1.5, n)
    return a, b, y


class TestKishDivisor:
    def test_none_is_byte_identical_to_legacy(self):
        a, b, y = _panel()
        base = fit_blend_weight_v2(a, b, y, forward_bars=24)
        kish_none = fit_blend_weight_v2(a, b, y, forward_bars=24,
                                        kish_divisor=None)
        assert base == kish_none
        # Hand-computed legacy SE for the same inputs (the exact old math)
        d = a - b
        denom = float(d @ d)
        w_raw = float(((y - b) @ d) / denom)
        eps = y - (w_raw * a + (1.0 - w_raw) * b)
        sigma2 = float(eps @ eps) / (len(a) - 1)
        assert base['se'] == pytest.approx(
            float(np.sqrt(sigma2 / denom * 24)), rel=1e-12)
        assert base['n_eff'] == pytest.approx(len(a) / 24)

    def test_divisor_scales_se_and_n_eff(self):
        a, b, y = _panel()
        f1 = fit_blend_weight_v2(a, b, y, forward_bars=1)
        f4 = fit_blend_weight_v2(a, b, y, forward_bars=1, kish_divisor=4.0)
        assert f4['w_raw'] == f1['w_raw']  # estimator untouched
        assert f4['se'] == pytest.approx(f1['se'] * 2.0, rel=1e-12)
        assert f4['n_eff'] == pytest.approx(f1['n_eff'] / 4.0)

    def test_significance_can_flip_off(self):
        # Borderline case significant on the temporal correction alone
        # must lose significance under a large cross-sectional deff —
        # the exact L4 failure mode ('significant' firing too liberally).
        a, b, y = _panel(seed=12)
        f1 = fit_blend_weight_v2(a, b, y, forward_bars=1)
        assert f1['significant'] is True
        fk = fit_blend_weight_v2(a, b, y, forward_bars=1,
                                 kish_divisor=48.0)
        assert fk['significant'] is False
        assert fk['w'] == 0.5  # insignificant -> exact simple average

    def test_divisor_below_one_clamps_to_legacy(self):
        a, b, y = _panel()
        base = fit_blend_weight_v2(a, b, y, forward_bars=24)
        half = fit_blend_weight_v2(a, b, y, forward_bars=24,
                                   kish_divisor=0.5)
        assert half == base  # kish can only WIDEN the SE, never shrink it

    def test_degenerate_dict_shape_unchanged(self):
        # test_c26_T1 pins the thin-input dict with full ==; the kish arg
        # must not grow the schema.
        fit = fit_blend_weight_v2([1, 2, 3], [3, 2, 1], [0, 1, 0],
                                  kish_divisor=3.0)
        assert fit == {'w': 0.5, 'w_raw': None, 'se': None,
                       'significant': False, 'n': 3, 'n_eff': None}


# ---------------------------------------------------------------------------
# [E] reselect_trade_threshold (H2)
# ---------------------------------------------------------------------------

class TestReselectTradeThreshold:
    def test_recovers_grid_optimum(self):
        thr, sc = reselect_trade_threshold(
            None, None, [0.1, 1.0], 0.9,
            lambda p, y, t: -abs(t - 0.37))
        assert thr == pytest.approx(0.37)
        assert sc == pytest.approx(0.0)

    def test_grid_matches_optuna_step_and_endpoints(self):
        seen = []
        reselect_trade_threshold(None, None, [0.96, 2.0], 1.0,
                                 lambda p, y, t: seen.append(t) or 0.0)
        assert len(seen) == 105  # (2.0 - 0.96)/0.01 + 1
        assert seen[0] == pytest.approx(0.96)
        assert seen[-1] == pytest.approx(2.0)
        steps = np.diff(seen)
        assert np.allclose(steps, 0.01)

    def test_stock_range_grid_survives_float_rounding(self):
        # The stock book's v3 range [0.18, 0.57]: (0.57-0.18)/0.01 is
        # 39.000000000000004 in floats — the grid must still land exactly
        # 40 points with the true upper endpoint included.
        seen = []
        reselect_trade_threshold(None, None, [0.18, 0.57], 0.2,
                                 lambda p, y, t: seen.append(t) or 0.0)
        assert len(seen) == 40
        assert seen[0] == pytest.approx(0.18)
        assert seen[-1] == pytest.approx(0.57)
        assert np.allclose(np.diff(seen), 0.01)

    def test_ties_break_toward_old_threshold(self):
        # Constant score -> every threshold ties -> least policy change.
        thr, _ = reselect_trade_threshold(None, None, [0.1, 1.0], 0.42,
                                          lambda p, y, t: 1.0)
        assert thr == pytest.approx(0.42)
        # Old threshold off-grid/outside the range -> nearest grid point.
        thr, _ = reselect_trade_threshold(None, None, [0.96, 2.0], 0.20,
                                          lambda p, y, t: 1.0)
        assert thr == pytest.approx(0.96)

    def test_preds_flow_to_scorer(self):
        preds = np.array([0.5, 1.5])
        y = np.array([1.0, -1.0])
        got = {}

        def score(p, yy, t):
            got['p'], got['y'] = p, yy
            return float((p > t).sum())

        thr, sc = reselect_trade_threshold(preds, y, [0.96, 2.0], 1.5,
                                           score)
        assert got['p'] is preds and got['y'] is y
        # Max score 1 (one pred above threshold) ties over [0.96, 1.49];
        # the tie breaks toward old=1.5 -> 1.49, the least policy change.
        assert thr == pytest.approx(1.49)
        assert sc == 1.0


# ---------------------------------------------------------------------------
# [F] strategy_config default-OFF pins
# ---------------------------------------------------------------------------

def test_sub_flags_default_off():
    import strategy_config
    assert strategy_config.BLEND_FIT_ON_REFIT is False
    assert strategy_config.BLEND_THRESHOLD_RESELECT is False
    # The parent flag the whole save path hangs off stays OFF too.
    assert strategy_config.HYPERSEARCH_V3 is True  # 2026-09-27: flipped at the gotcha-#2 event (founder)


# ---------------------------------------------------------------------------
# [G] sparing source-structure asserts on scripts/hypersearch_v2.py
# ---------------------------------------------------------------------------

class TestHypersearchSourcePins:
    def test_h1_effective_weight_feeds_cert_and_config(self):
        # The parity resolution runs BEFORE the holdout gate, so the SAME
        # lstm_weight reaches evaluate_on_holdout AND the config write.
        i_eff = HS.rindex('effective_lstm_weight(lstm_weight, _ship)')
        i_gate = HS.index('holdout_report = evaluate_on_holdout')
        i_cfg = HS.index("config['lstm_weight'] = round(float(lstm_weight)")
        assert i_eff < i_gate < i_cfg

    def test_h1_uses_the_blend_fit_constant(self):
        assert 'DEFAULT_LSTM_WEIGHT' in HS
        assert 'effective_lstm_weight' in HS

    def test_h1_parity_block_inside_v3_scope(self):
        # Flag-OFF purity: the parity resolution lives INSIDE `if _v3:` —
        # HYPERSEARCH_V3=False never touches lstm_weight (stays None ->
        # legacy raw-LSTM certificate). Pinned two ways: position (between
        # the `if _v3:` open and the winner's-curse block that is back at
        # the outer scope) and indentation (16 = inside the parity `try:`
        # nested in the 12-space `if _v3:` body).
        i_assign = HS.index('\n                lstm_weight = '
                            'effective_lstm_weight(lstm_weight, _ship)')
        i_v3_open = HS.index('if _v3:')
        i_outer = HS.index("Winner's-curse instrumentation")
        assert i_v3_open < i_assign < i_outer

    def test_h1_ship_condition_matches_artifact_write(self):
        # The parity block's "will boosters ship?" predicate must be the
        # SAME condition the atomic save uses to write the booster files —
        # if these ever diverge, cert==deploy silently breaks again.
        assert '_ship = bool(lgb_pack and lgb_pack[0] is not None)' in HS
        assert 'if _v3 and lgb_pack and lgb_pack[0] is not None:' in HS

    def test_h1_gate_receives_resolved_variable(self):
        # evaluate_on_holdout is fed the same `lstm_weight` local the
        # config write reads (no shadow variable), and its blend branch
        # still keys on that kwarg being non-None.
        assert 'lstm_weight=lstm_weight,' in HS
        assert ('if lgb_booster is not None and '
                'lstm_weight is not None:') in HS

    def test_m4_guard_in_lgb_fold_build(self):
        seg = HS.split('def train_lgb_ensemble', 1)[1].split('\ndef ', 1)[0]
        assert 'purge_val_labels=_purge' in seg
        assert '_hypersearch_v3() and not _purge' in seg
        # The legacy expression must be GONE from this function (the guard
        # replaced it) ...
        assert 'purge_val_labels=_objective_v3()' not in seg
        # ... but must survive in the trial objective (trial scores are
        # OBJECTIVE_V3's domain, untouched by this packet).
        assert 'purge_val_labels=_objective_v3()' in HS

    def test_m4_guard_in_blend_fit_rows(self):
        # The blend-fit row purge sits between the oof_rows read and the
        # y_fit construction.
        i_rows = HS.index("best_state_holder['oof_rows'][-1]")
        i_purge = HS.index('blend-fit rows whose label windows')
        i_yfit = HS.index('y_fit = returns[rows]')
        assert i_rows < i_purge < i_yfit

    def test_m4_purge_rule_matches_fold_builder(self):
        # The blend-fit purge must apply the EXACT rule
        # get_walk_forward_folds uses (label_times <= holdout boundary) —
        # a '<' / '>' drift here re-opens M4 one row at a time.
        assert '_hb = get_holdout_boundary(all_times)' in HS
        assert '_keep = all_label_times[rows] <= _hb' in HS
        # ... and the fold builder's own purge rule survives verbatim.
        assert ('val_mask = val_mask & (all_label_times[valid]\n'
                '                                   '
                '<= get_holdout_boundary(all_times))') in HS

    def test_m2_refit_fit_flag_gated(self):
        assert 'BLEND_FIT_ON_REFIT' in HS
        assert 'lstm_oof_refit' in HS
        # Deployment source selection: refit only under the flag.
        assert '_fit_on_refit and fit_refit is not None' in HS

    def test_m2_refit_inference_uses_shipping_stack(self):
        # M2's whole point: the refit-side fit inputs are the SHIPPED
        # state under the SHIP scaler (the one _scaled_fit transform the
        # LGB leg already used) — never the fold checkpoint.
        assert '_rmdl.load_state_dict(ship_state)' in HS
        assert '_scaled_fit = ship_scaler.transform(' in HS

    def test_h2_reselect_flag_gated_and_before_cert(self):
        assert 'BLEND_THRESHOLD_RESELECT' in HS
        i_resel = HS.index('reselect_trade_threshold')
        i_gate = HS.index('holdout_report = evaluate_on_holdout')
        assert i_resel < i_gate
        # The best_cfg mutation is guarded by the deploy flag read just
        # above it.
        i_mut = HS.index("best_cfg['trade_threshold'] = float(_new_thr)")
        i_flag = HS.rindex('BLEND_THRESHOLD_RESELECT', 0, i_mut)
        assert 0 < i_mut - i_flag < 1500
        assert 'if _tt_deploy:' in HS

    def test_pre_gate_ordering_preserved(self):
        # T1's invariant survives the packet: LGB legs train (save=False)
        # BEFORE the holdout gate.
        assert HS.index('save=False)') \
            < HS.index('holdout_report = evaluate_on_holdout')

    def test_legacy_lgb_call_still_flag_guarded(self):
        k = HS.index('if not _v3:')
        assert 'train_lgb_ensemble(save_prefix, best_scaler' in HS[k:k + 200]

    def test_pinned_trade_threshold_literal_verbatim(self):
        # The D25/H2 work must not disturb the legacy fallback literal
        # (test_grp_training regex-pins it).
        assert "_space.get('trade_threshold', [0.05, 1.0])" in HS

    def test_blend_diag_write_guarded(self):
        # The failure path now carries lstm_weight without blend_diag; the
        # config write must not stamp a None diag.
        i_cfg = HS.index("config['lstm_weight'] = round(float(lstm_weight)")
        window = HS[i_cfg:i_cfg + 300]
        assert 'if blend_diag is not None:' in window
