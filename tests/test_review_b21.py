"""Review batch b21 — scripts/harvest_crypto_data.py, scripts/cscv_audit.py,
scripts/ic_by_name.py.

Covers:
  - harvest_crypto_data: total-failure paths exit nonzero (no-data and
    both-writes-failed) so run_pipeline's retry/notify fires; benchmark-fetch
    failure warns that BTC cross-asset features are omitted; the summary no
    longer counts TB_* label columns as features and lists them as targets;
    dead `import time` removed. (Module imports dotenv → source-inspection +
    extracted-block execution on the Mac, per the test_prediction_cache
    pattern.)
  - cscv_audit: docstring recipe guards build_oos_blocks() returning None;
    `--blocks ''` routes to the --blocks branch (no Path(None) TypeError);
    odd/zero --n-groups and indivisible --n-blocks rejected upfront; n/a
    message mentions the divisibility/even conditions; verdict line says
    "at or below the OOS median" (lambda <= 0).
  - ic_by_name: zero-IC names sort between positive and negative (falsy-zero
    bug); t-stat + n_finite + hurdle thresholds printed; --min-t exposed and
    passed to promote_set; misspelled keys fail loud with the available keys;
    --time-key stable-sorts unordered dumps; docstring no longer claims a
    backtest.py dump that does not exist.

All tests are Mac-safe: stdlib + numpy/pandas/scipy only.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

import cscv_audit as ca
import ic_by_name as icb
import validation

REPO = Path(__file__).resolve().parent.parent
HARVEST_SRC = (REPO / 'scripts' / 'harvest_crypto_data.py').read_text(
    encoding='utf-8')


# ---------------------------------------------------------------------------
# harvest_crypto_data (source inspection — module needs dotenv/indicators)
# ---------------------------------------------------------------------------

class TestHarvestFailureSignaling:
    def test_no_data_path_exits_nonzero(self):
        # A bare `return` exited 0 and run_pipeline recorded a successful
        # phase, then trained on stale data with no retry/notification.
        idx = HARVEST_SRC.index('ERROR: No data fetched for any ticker')
        following = HARVEST_SRC[idx:idx + 300]
        assert 'sys.exit(1)' in following
        # The old silent path is gone: no bare `return` right after the ERROR.
        block = following.split('sys.exit(1)')[0]
        assert '\n        return\n' not in block

    def test_save_result_checked_and_exits_nonzero(self):
        assert "if not save_training_data(final_df, 'crypto'):" in HARVEST_SRC
        idx = HARVEST_SRC.index("if not save_training_data")
        assert 'sys.exit(1)' in HARVEST_SRC[idx:idx + 300]
        # No remaining bare (unchecked) call anywhere in the file.
        for line in HARVEST_SRC.splitlines():
            if line.strip().startswith('save_training_data('):
                pytest.fail(f'unchecked save_training_data call: {line!r}')

    def test_benchmark_failure_warns_features_omitted(self):
        assert 'btc_close is None' in HARVEST_SRC
        assert 'OMITTED' in HARVEST_SRC
        # Names the columns so the operator knows what changed in the schema.
        assert 'BTC_Return_1h' in HARVEST_SRC
        assert 'BTC_SMA_Ratio' in HARVEST_SRC
        assert 'BTC_RSI' in HARVEST_SRC

    def test_dead_time_import_removed(self):
        assert not any(l.strip() == 'import time'
                       for l in HARVEST_SRC.splitlines())


class TestHarvestSummaryCounts:
    def _summary_ns(self, columns):
        """Execute the actual summary block from main() on a synthetic frame."""
        lines = HARVEST_SRC.splitlines()
        i = next(k for k, l in enumerate(lines)
                 if l.strip().startswith('target_cols ='))
        j = next(k for k, l in enumerate(lines) if 'Target columns:' in l)
        block = '\n'.join(l[4:] for l in lines[i:j + 1])
        ns = {'final_df': pd.DataFrame(columns=columns)}
        exec(block, ns)  # single namespace so comprehensions resolve names
        return ns

    def test_tb_label_columns_not_counted_as_features(self):
        ns = self._summary_ns([
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI',
            'Daily_Sentiment', 'Ticker',
            'Target_Return_4', 'Target_Return',
            'TB_Ret_4', 'TB_Bars_4', 'TB_Reason_4',
        ])
        # OHLCV stay counted (training keeps them as features); TB_* do not.
        assert ns['feature_count'] == 7
        assert ns['tb_cols'] == ['TB_Ret_4', 'TB_Bars_4', 'TB_Reason_4']

    def test_tb_labels_listed_as_targets(self, capsys):
        self._summary_ns(['Close', 'Target_Return_4', 'TB_Ret_4', 'Ticker'])
        out = capsys.readouterr().out
        assert 'TB_Ret_4' in out.split('Target columns:')[1]


# ---------------------------------------------------------------------------
# cscv_audit
# ---------------------------------------------------------------------------

def _run_cscv(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, 'argv', ['cscv_audit.py'] + argv)
    rc = ca.main()
    return rc, capsys.readouterr().out


class TestCscvDocRecipe:
    def test_recipe_guards_none(self):
        # The old one-liner called .tolist() straight on build_oos_blocks(),
        # which returns None for <8 finite trade returns — an AttributeError
        # that aborts the whole overnight study.optimize() run.
        assert 'None if blocks is None else blocks.tolist()' in ca.__doc__
        assert '8).tolist()' not in ca.__doc__

    def test_recipe_premise_short_trials_yield_none(self):
        assert validation.build_oos_blocks([0.1, 0.2], 8) is None
        assert validation.build_oos_blocks([], 8) is None


class TestCscvArgs:
    def test_blocks_empty_string_stays_in_blocks_branch(self, monkeypatch):
        # '' satisfied argparse's required group but was falsy, so control
        # fell into the --returns branch and crashed on Path(None).
        monkeypatch.setattr(sys, 'argv', ['cscv_audit.py', '--blocks', ''])
        with pytest.raises(OSError):  # file error on --blocks, not TypeError
            ca.main()

    @pytest.mark.parametrize('n_groups', ['7', '0'])
    def test_odd_or_tiny_n_groups_rejected(self, monkeypatch, capsys, n_groups):
        monkeypatch.setattr(sys, 'argv', ['cscv_audit.py', '--blocks', 'x.json',
                                          '--n-groups', n_groups])
        with pytest.raises(SystemExit) as ei:
            ca.main()
        assert ei.value.code == 2
        assert 'even' in capsys.readouterr().err

    def test_returns_indivisible_n_blocks_rejected(self, monkeypatch, capsys):
        # Validated BEFORE the file is read (the path does not exist).
        monkeypatch.setattr(sys, 'argv', ['cscv_audit.py',
                                          '--returns', 'nonexistent.json',
                                          '--n-blocks', '12', '--n-groups', '8'])
        with pytest.raises(SystemExit) as ei:
            ca.main()
        assert ei.value.code == 2
        assert 'divisible' in capsys.readouterr().err


class TestCscvOutput:
    def _trials(self):
        import numpy as np
        rng = np.random.default_rng(7)
        return [list(rng.normal(0.01 * i, 0.02, 32)) for i in range(6)]

    def test_returns_path_filters_short_trials_and_reports(
            self, monkeypatch, capsys, tmp_path):
        trials = self._trials() + [[0.1, 0.2]]  # short trial -> None block row
        f = tmp_path / 'returns.json'
        f.write_text(json.dumps(trials))
        rc, out = _run_cscv(monkeypatch, capsys,
                            ['--returns', str(f), '--n-blocks', '8',
                             '--n-groups', '4'])
        assert rc == 0
        assert 'PBO' in out
        # lambda <= 0 counts splits AT the median too (validation.pbo_cscv).
        assert 'at or below the OOS median' in out
        assert 'was below the OOS median' not in out

    def test_blocks_path_accepts_json_null_rows(self, monkeypatch, capsys,
                                                tmp_path):
        # The corrected recipe stores null for degenerate trials — the audit
        # must filter them, not crash.
        blocks = [None] + [validation.build_oos_blocks(r, 8).tolist()
                           for r in self._trials()]
        f = tmp_path / 'blocks.json'
        f.write_text(json.dumps(blocks))
        rc, out = _run_cscv(monkeypatch, capsys,
                            ['--blocks', str(f), '--n-groups', '4'])
        assert rc == 0
        assert 'PBO' in out and 'n/a' not in out

    def test_na_message_mentions_divisibility(self, monkeypatch, capsys,
                                              tmp_path):
        # Width 12 with n_groups 8 fails only the divisibility condition the
        # old message never mentioned (--blocks widths are unknowable upfront).
        blocks = [validation.build_oos_blocks(r, 12).tolist()
                  for r in self._trials()]
        f = tmp_path / 'blocks.json'
        f.write_text(json.dumps(blocks))
        rc, out = _run_cscv(monkeypatch, capsys,
                            ['--blocks', str(f), '--n-groups', '8'])
        assert rc == 0
        assert 'n/a' in out
        assert 'divisible' in out


# ---------------------------------------------------------------------------
# ic_by_name
# ---------------------------------------------------------------------------

def _rows(name, pred, fwd, ts=None):
    out = []
    for i, (p, f) in enumerate(zip(pred, fwd)):
        r = {'symbol': name, 'pred': p, 'fwd_return': f}
        if ts is not None:
            r['ts'] = ts[i]
        out.append(r)
    return out


def _run_icb(monkeypatch, capsys, tmp_path, rows, *extra):
    f = tmp_path / 'rows.json'
    f.write_text(json.dumps(rows))
    monkeypatch.setattr(sys, 'argv', ['ic_by_name.py', '--in', str(f)] +
                        list(extra))
    rc = icb.main()
    return rc, capsys.readouterr().out


class TestIcSortOrder:
    def test_zero_ic_sorts_between_positive_and_negative(
            self, monkeypatch, capsys, tmp_path):
        # `-(ic or -9)` gave IC==0.0 sort key 9, placing it BELOW negatives.
        rows = (_rows('POSX', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) +
                _rows('ZERO', [1, 2, 3, 4, 5], [2, 5, 3, 1, 4]) +  # rho == 0
                _rows('NEGX', [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) +
                _rows('NONE', [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]))   # ic None
        rc, out = _run_icb(monkeypatch, capsys, tmp_path, rows)
        assert rc == 0
        names = [l.split()[1] for l in out.splitlines() if ' IC=' in l]
        assert names == ['POSX', 'ZERO', 'NEGX', 'NONE']
        assert 'IC=0.0 ' in out  # ZERO really is a computed zero, not None


class TestIcHurdleVisibility:
    def test_hidden_t_hurdle_printed_and_binding(self, monkeypatch, capsys,
                                                 tmp_path):
        # ic=0.8, consistency=1.0 passes both PRINTED hurdles of the old
        # output, yet t = 0.8*sqrt(4) = 1.6 < 2.0 rejects it — the reason
        # must now be visible in the row and the header.
        rows = _rows('WEAK', [1, 2, 3, 4, 5], [2, 1, 3, 5, 4])
        rc, out = _run_icb(monkeypatch, capsys, tmp_path, rows)
        line = next(l for l in out.splitlines() if 'WEAK' in l)
        assert 'hold' in line
        assert 't=1.6' in line
        assert 'ALL required' in out and '>= 2.0' in out

    def test_min_t_flag_reaches_promote_set(self, monkeypatch, capsys,
                                            tmp_path):
        rows = _rows('WEAK', [1, 2, 3, 4, 5], [2, 1, 3, 5, 4])
        rc, out = _run_icb(monkeypatch, capsys, tmp_path, rows,
                           '--min-t', '0')
        line = next(l for l in out.splitlines() if 'WEAK' in l)
        assert 'PROMOTE' in line
        assert "PROMOTE SET (1): ['WEAK']" in out

    def test_min_t_default_matches_promote_set(self):
        import inspect
        from ic_diagnostic import promote_set
        assert (inspect.signature(promote_set).parameters['min_t'].default
                == 2.0)

    def test_n_finite_printed_not_just_raw_n(self, monkeypatch, capsys,
                                             tmp_path):
        # 6 finite pairs + 4 no-coverage rows: the significance sample is 6,
        # and the row must show it (the old print showed only n=10).
        pred = [1, 2, 3, 4, 5, 6] + [7, 8, 9, 10]
        fwd = [1, 2, 3, 4, 5, 6] + [None] * 4
        rows = _rows('PAD', pred, fwd)
        rc, out = _run_icb(monkeypatch, capsys, tmp_path, rows)
        line = next(l for l in out.splitlines() if 'PAD' in l)
        assert 'n=10' in line
        assert 'n_finite=6' in line
        assert 't=2.24' in line  # 1.0*sqrt(5), computed on n_finite


class TestIcKeyValidation:
    def test_misspelled_pred_key_fails_loud(self, monkeypatch, tmp_path):
        rows = _rows('AAA', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        f = tmp_path / 'rows.json'
        f.write_text(json.dumps(rows))
        monkeypatch.setattr(sys, 'argv', ['ic_by_name.py', '--in', str(f),
                                          '--pred-key', 'signl'])
        with pytest.raises(SystemExit) as ei:
            icb.main()
        msg = str(ei.value)
        assert 'signl' in msg and 'available' in msg and 'symbol' in msg

    def test_misspelled_name_key_exits_cleanly(self, monkeypatch, tmp_path):
        # Previously a raw KeyError from deep inside ic_diagnostic.
        rows = _rows('AAA', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        f = tmp_path / 'rows.json'
        f.write_text(json.dumps(rows))
        monkeypatch.setattr(sys, 'argv', ['ic_by_name.py', '--in', str(f),
                                          '--name-key', 'symbl'])
        with pytest.raises(SystemExit) as ei:
            icb.main()
        assert 'symbl' in str(ei.value)

    def test_signal_key_compat_with_rank_gradient_frame(self, monkeypatch,
                                                        capsys, tmp_path):
        # rank_gradient_report's frame names the column 'signal'.
        rows = [{'symbol': 'SIG', 'signal': float(i),
                 'fwd_return': float(i)} for i in range(1, 6)]
        rc, out = _run_icb(monkeypatch, capsys, tmp_path, rows,
                           '--pred-key', 'signal')
        assert rc == 0
        line = next(l for l in out.splitlines() if 'SIG' in l and ' IC=' in l)
        assert 'IC=1.0' in line


class TestIcTimeKey:
    def _regime_rows(self):
        # First half fwd=pred (IC +1 per quarter), second half fwd=-pred
        # (IC -1): ordered consistency is exactly 0.5, overall IC exactly 0.
        pred = [float(i % 5 + 1) for i in range(20)]
        fwd = [p if i < 10 else -p for i, p in enumerate(pred)]
        return _rows('REG', pred, fwd, ts=list(range(20)))

    def _reg_line(self, out):
        return next(l for l in out.splitlines() if 'REG' in l and ' IC=' in l)

    def test_time_key_restores_bar_order(self, monkeypatch, capsys, tmp_path):
        ordered = self._regime_rows()
        shuffled = ordered[:]
        # Seed picked so the scrambled quarters give consistency 0.25 != 0.5.
        random.Random(2).shuffle(shuffled)
        _, out_ordered = _run_icb(monkeypatch, capsys, tmp_path, ordered)
        _, out_sorted = _run_icb(monkeypatch, capsys, tmp_path, shuffled,
                                 '--time-key', 'ts')
        assert self._reg_line(out_sorted) == self._reg_line(out_ordered)
        assert 'consistency=0.5' in self._reg_line(out_ordered)

    def test_unordered_rows_without_flag_differ(self, monkeypatch, capsys,
                                                tmp_path):
        # Negative control: the same shuffle WITHOUT --time-key scores the
        # sub-periods on scrambled rows (the failure mode the docstring warns
        # about); overall IC is permutation-invariant, consistency is not.
        ordered = self._regime_rows()
        shuffled = ordered[:]
        # Seed picked so the scrambled quarters give consistency 0.25 != 0.5.
        random.Random(2).shuffle(shuffled)
        _, out_ordered = _run_icb(monkeypatch, capsys, tmp_path, ordered)
        _, out_shuffled = _run_icb(monkeypatch, capsys, tmp_path, shuffled)
        assert self._reg_line(out_shuffled) != self._reg_line(out_ordered)


class TestIcDocstring:
    def test_no_longer_claims_backtest_dump_exists(self):
        # Modernized (c26 final review F14): backtest.py now SHIPS the
        # producer (B02 stage0_preds.json, default ON) — the docstring must
        # say so instead of the stale "NOT yet authored" claim.
        assert 'NOT yet authored' not in icb.__doc__
        assert 'stage0_preds.json' in icb.__doc__

    def test_states_ordering_and_overlap_preconditions(self):
        doc = icb.__doc__
        assert 'bar-ordered' in doc
        assert 'non-overlapping' in doc
        assert '--pred-key signal' in doc
        assert '--time-key' in doc
