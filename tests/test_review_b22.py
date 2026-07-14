"""Review batch b22 — scripts/reliability_report.py, scripts/rank_gradient_report.py.

Covers:
  - reliability_report: exact metric ties no longer print 'no gain' labels next
    to a 'safe to flip' verdict — labels are tri-state (better / tie (no worse)
    / worse) on the same lower-or-equal criterion family the gate uses, ties
    print a 'do NOT flip' verdict, and (near-)identical p arrays warn loudly;
    --bins < 2 rejected upfront (bins=0 degenerated ECE to a vacuous 0.0 tie);
    missing payload keys / length mismatches exit 2 with an actionable one-line
    error instead of a raw traceback; docstring states the implemented
    lower-or-equal flip criterion, not the old strict-lower one.
  - rank_gradient_report: --preds works on its own documented frame (the
    'symbol' column no longer KeyErrors on panel_from_frame's 'Ticker'
    default; 'Ticker' dumps still tolerated); --buckets on a full
    decision_report.json (buckets nested under 'conviction') prints the
    per-bucket evidence lines, not just ratio+verdict; exit status is
    verdict-driven (0 CONFIRMED, 1 no-go/insufficient) so scripted use cannot
    mistake a FAIL for success; docstring carries the 'dump step NOT yet
    authored' warning and the percent units contract, and --cost-pct's help
    states the units.

All tests are Mac-safe: stdlib + numpy/pandas only (conftest puts scripts/ on
sys.path; both scripts import only calibration / rank_gradient /
portfolio_backtest, which are pure numpy, plus pandas in the --preds branch).
"""

import json
import sys
from pathlib import Path

import pytest

import rank_gradient_report as rgr
import reliability_report as rel

REPO = Path(__file__).resolve().parent.parent
RGR_SRC = (REPO / 'scripts' / 'rank_gradient_report.py').read_text(encoding='utf-8')


def _run_rel(monkeypatch, capsys, tmp_path, payload, *extra):
    f = tmp_path / 'calib.json'
    f.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, 'argv',
                        ['reliability_report.py', '--in', str(f)] + list(extra))
    rc = rel.main()
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _run_rgr(monkeypatch, capsys, tmp_path, flag, payload, *extra):
    f = tmp_path / 'input.json'
    f.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, 'argv',
                        ['rank_gradient_report.py', flag, str(f)] + list(extra))
    rc = rgr.main()
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


# ---------------------------------------------------------------------------
# reliability_report: tie semantics (labels + verdict must agree)
# ---------------------------------------------------------------------------

class TestReliabilityTieSemantics:
    def test_identical_arrays_do_not_green_light(self, monkeypatch, capsys,
                                                 tmp_path):
        # Same array dumped into both keys (the zero-evidence Jetson accident):
        # the old output printed 'no gain' twice yet VERDICT green-lit the flip.
        p = [0.4, 0.6, 0.55, 0.3, 0.7, 0.65]
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': p, 'p_purged': p,
                                 'y': [0, 1, 1, 0, 1, 0]})
        assert rc == 0
        assert 'safe to flip' not in out
        assert 'do NOT flip' in out
        assert out.count('tie (no worse)') == 2
        assert 'no gain' not in out
        assert 'identical' in err  # loud warning on the degenerate dump

    def test_equal_metrics_without_identical_arrays_tie(self, monkeypatch,
                                                        capsys, tmp_path):
        # Same (p, y) multiset, different element order: Brier and ECE tie
        # EXACTLY while the arrays differ — the metric-tie branch must fire
        # on its own, without the identical-array warning.
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': [0.6, 0.8, 0.2, 0.4],
                                 'p_purged': [0.8, 0.6, 0.4, 0.2],
                                 'y': [1, 1, 0, 0]})
        assert rc == 0
        assert 'do NOT flip' in out
        assert 'safe to flip' not in out
        assert out.count('tie (no worse)') == 2
        assert 'identical' not in err

    def test_strict_improvement_still_green_lights(self, monkeypatch, capsys,
                                                   tmp_path):
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': [0.45] * 3 + [0.55] * 3,
                                 'p_purged': [0.05] * 3 + [0.95] * 3,
                                 'y': [0, 0, 0, 1, 1, 1]})
        assert rc == 0
        assert out.count('(better)') == 2
        assert 'safe to flip' in out
        assert 'do NOT flip' not in out

    def test_regression_labeled_worse_not_no_gain(self, monkeypatch, capsys,
                                                  tmp_path):
        # Swapped arrays: purged strictly worse on both metrics. The old
        # two-state label understated this as 'no gain'.
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': [0.05] * 3 + [0.95] * 3,
                                 'p_purged': [0.45] * 3 + [0.55] * 3,
                                 'y': [0, 0, 0, 1, 1, 1]})
        assert rc == 0
        assert out.count('(worse)') == 2
        assert 'keep legacy' in out
        assert 'safe to flip' not in out

    def test_docstring_states_lower_or_equal_criterion(self):
        # The implemented gate (calibration.compare_calibrations) is <=; the
        # docstring claimed strictly 'lower Brier AND ECE'.
        doc = ' '.join(rel.__doc__.split())  # collapse line wraps
        assert 'lower-or-equal Brier AND ECE' in doc
        assert '(lower Brier AND ECE)' not in doc
        assert 'at least as well-calibrated' in doc


# ---------------------------------------------------------------------------
# reliability_report: --bins validation
# ---------------------------------------------------------------------------

class TestReliabilityBinsValidation:
    PAYLOAD = {'p_legacy': [0.4, 0.6], 'p_purged': [0.3, 0.7], 'y': [0, 1]}

    @pytest.mark.parametrize('bins', ['0', '1', '-3'])
    def test_bins_lt_2_rejected(self, monkeypatch, capsys, tmp_path, bins):
        # --bins 0 degenerated ECE to 0.0 for BOTH calibrators (a vacuous tie
        # that satisfied the ECE half of the flip criterion) and rendered an
        # empty reliability section.
        f = tmp_path / 'calib.json'
        f.write_text(json.dumps(self.PAYLOAD))
        monkeypatch.setattr(sys, 'argv', ['reliability_report.py', '--in',
                                          str(f), '--bins', bins])
        with pytest.raises(SystemExit) as ei:
            rel.main()
        assert ei.value.code == 2
        assert '--bins must be >= 2' in capsys.readouterr().err

    def test_bins_2_accepted(self, monkeypatch, capsys, tmp_path):
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path, self.PAYLOAD,
                                '--bins', '2')
        assert rc == 0
        assert 'VERDICT' in out


# ---------------------------------------------------------------------------
# reliability_report: payload validation (loud, actionable, exit 2)
# ---------------------------------------------------------------------------

class TestReliabilityPayloadValidation:
    def test_missing_key_returns_2_with_actionable_error(self, monkeypatch,
                                                         capsys, tmp_path):
        # Previously a raw KeyError traceback.
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': [0.4], 'p_purged': [0.5]})
        assert rc == 2
        assert 'ERROR' in err and "'y'" in err
        assert 'p_legacy' in err  # names what IS there / what is required
        assert 'VERDICT' not in out

    def test_length_mismatch_returns_2(self, monkeypatch, capsys, tmp_path):
        # Previously failed far downstream via numpy broadcast with a cryptic
        # message.
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path,
                                {'p_legacy': [0.4, 0.5, 0.6],
                                 'p_purged': [0.4, 0.5, 0.6],
                                 'y': [0, 1]})
        assert rc == 2
        assert 'length mismatch' in err
        assert 'VERDICT' not in out

    def test_non_dict_payload_returns_2(self, monkeypatch, capsys, tmp_path):
        rc, out, err = _run_rel(monkeypatch, capsys, tmp_path, [0.4, 0.5])
        assert rc == 2
        assert 'ERROR' in err


# ---------------------------------------------------------------------------
# rank_gradient_report: --preds on the documented frame
# ---------------------------------------------------------------------------

def _pred_rows(ticker_key='symbol'):
    # 3 periods x 7 symbols, signal 7..1, fwd_return = signal/10 (percent):
    # rank_1_3 mean 0.6, rank_6_7 mean 0.15 < 0.5*0.6 -> gradient CONFIRMED.
    rows = []
    for t in (1, 2, 3):
        for i, sym in enumerate(('AAA', 'BBB', 'CCC', 'DDD', 'EEE',
                                 'FFF', 'GGG')):
            rows.append({'ts': f'2026-01-0{t}T00:00:00', ticker_key: sym,
                         'signal': float(7 - i),
                         'fwd_return': (7 - i) / 10.0})
    return rows


class TestRankGradientPredsCLI:
    def test_documented_symbol_frame_end_to_end(self, monkeypatch, capsys,
                                                tmp_path):
        # The documented contract is a 'symbol' column (module docstring,
        # ic_by_name's shared-dump note); panel_from_frame's default
        # ticker_col='Ticker' made this KeyError.
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--preds',
                                _pred_rows())
        assert rc == 0
        assert 'rank_1_3: mean_net 0.6  (n=9)' in out
        assert 'rank_4_5: mean_net 0.35  (n=6)' in out
        assert 'rank_6_7: mean_net 0.15  (n=6)' in out
        assert 'CONFIRMED' in out

    def test_ticker_column_dump_tolerated(self, monkeypatch, capsys,
                                          tmp_path):
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--preds',
                                _pred_rows(ticker_key='Ticker'))
        assert rc == 0
        assert 'CONFIRMED' in out

    def test_cost_pct_shifts_bucket_means(self, monkeypatch, capsys,
                                          tmp_path):
        # Sanity that --cost-pct reaches the kernel in the documented units.
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--preds',
                                _pred_rows(), '--cost-pct', '0.1')
        assert 'rank_1_3: mean_net 0.5  (n=9)' in out


# ---------------------------------------------------------------------------
# rank_gradient_report: --buckets evidence lines + verdict-driven exit status
# ---------------------------------------------------------------------------

class TestRankGradientBucketsCLI:
    GOOD = {'rank_1_3': {'mean_net_pct': 0.5, 'n': 30},
            'rank_4_5': {'mean_net_pct': 0.2, 'n': 20},
            'rank_6_7': {'mean_net_pct': 0.05, 'n': 20}}

    def test_decision_report_wrapper_prints_bucket_evidence(self, monkeypatch,
                                                            capsys, tmp_path):
        # decision_report.json nests the buckets under 'conviction'; the
        # verdict unwrapped it but the evidence print loop did not, silently
        # dropping the bucket means/n the gate-reviewer is supposed to eyeball.
        payload = {'generated': 'x', 'days': 30, 'conviction': self.GOOD}
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--buckets',
                                payload)
        assert rc == 0
        assert 'rank_1_3: mean_net 0.5  (n=30)' in out
        assert 'rank_4_5: mean_net 0.2  (n=20)' in out
        assert 'rank_6_7: mean_net 0.05  (n=20)' in out
        assert 'CONFIRMED' in out

    def test_flat_bucket_dict_still_prints(self, monkeypatch, capsys,
                                           tmp_path):
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--buckets',
                                self.GOOD)
        assert rc == 0
        assert 'rank_1_3: mean_net 0.5  (n=30)' in out

    def test_no_gradient_exits_1(self, monkeypatch, capsys, tmp_path):
        # `... && enable-flag` scripting must not treat 'ship NEITHER' as
        # success (old main() returned 0 unconditionally).
        payload = {'rank_1_3': {'mean_net_pct': 0.1, 'n': 30},
                   'rank_6_7': {'mean_net_pct': 0.2, 'n': 20}}
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--buckets',
                                payload)
        assert rc == 1
        assert 'ship NEITHER' in out

    def test_insufficient_coverage_exits_1(self, monkeypatch, capsys,
                                           tmp_path):
        payload = {'rank_1_3': {'mean_net_pct': 0.5, 'n': 30}}
        rc, out, err = _run_rgr(monkeypatch, capsys, tmp_path, '--buckets',
                                payload)
        assert rc == 1
        assert 'insufficient rank coverage' in out


# ---------------------------------------------------------------------------
# rank_gradient_report: docstring / help-text contracts
# ---------------------------------------------------------------------------

class TestRankGradientDocs:
    def test_producer_warning_present(self):
        # Same warning as ic_by_name: the frame producer does not exist yet.
        assert 'NOT yet authored' in rgr.__doc__
        assert 'from backtest.py over the universe' not in rgr.__doc__

    def test_units_contract_stated(self):
        assert 'PERCENT' in rgr.__doc__
        assert 'Target_Return_' in rgr.__doc__
        assert 'cost-pct' in rgr.__doc__

    def test_cost_pct_help_states_units(self):
        assert 'SAME units as fwd_return' in RGR_SRC
        assert 'round_trip_cost_pct' in RGR_SRC

    def test_exit_status_documented(self):
        assert 'Exit status' in rgr.__doc__
