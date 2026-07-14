"""Review batch b17 — volatility.py, gap_audit.py, execution_report.py fixes.

Covers: fit_garch missing-'arch' WARNING-once (instead of debug-per-call);
per-bar (not annualized) sigma docstring contracts; gap_audit yfinance
MultiIndex flattening + single-ticker guard + visible Student-t fit failures;
execution_report JOURNAL_DIR sourced from trade_journal, corrupt-line
counting, stamped artifact written even on empty windows, fees.py-matching
maker-share predicate with a small-n caveat, and corrected fee-comment math.
Also pins the three BARS_PER_YEAR copies equal (consolidation into
strategy_config was deferred — out of b17 file ownership)."""

import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

import execution_report
import fees
import gap_audit
import trade_journal
import volatility


# --- volatility: missing 'arch' is visible, once ---

class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def debug(self, *args, **kwargs):
        pass


class TestFitGarchArchMissing:
    def test_warns_once_and_returns_none(self, monkeypatch):
        # None in sys.modules forces ImportError even where arch IS installed,
        # so this exercises the same path on Mac, CI and Jetson.
        monkeypatch.setitem(sys.modules, 'arch', None)
        monkeypatch.setattr(volatility, '_arch_warned', False)
        rec = _RecordingLogger()
        monkeypatch.setattr(volatility, 'logger', rec)
        returns = np.random.RandomState(0).randn(200)
        assert volatility.fit_garch(returns) is None
        assert volatility.fit_garch(returns) is None
        arch_warns = [w for w in rec.warnings if 'arch' in w]
        assert len(arch_warns) == 1                # once, not per call
        assert volatility._arch_warned is True

    def test_short_series_still_quiet_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'arch', None)
        monkeypatch.setattr(volatility, '_arch_warned', False)
        rec = _RecordingLogger()
        monkeypatch.setattr(volatility, 'logger', rec)
        assert volatility.fit_garch(np.random.RandomState(1).randn(50)) is None
        assert rec.warnings == []                  # length gate fires first


class TestSigmaUnitsDocs:
    def test_forecast_volatility_doc_is_per_bar(self):
        doc = volatility.forecast_volatility.__doc__
        assert 'PER-BAR' in doc
        assert '0.25 = 25%' not in doc             # stale annualized example

    def test_module_doc_no_phantom_atr_fallback(self):
        doc = volatility.__doc__
        assert 'Falls back to ATR' not in doc
        assert 'None' in doc                       # returns None; callers fall back

    def test_get_sigma_doc_flags_annualization(self):
        assert 'BARS_PER_YEAR' in volatility.get_sigma.__doc__


def test_bars_per_year_copies_in_sync():
    """Consolidating BARS_PER_YEAR into strategy_config was deferred (files
    out of b17 ownership); until then, drift between promotion-gate
    annualization and live vol targeting must fail loudly here."""
    pat = re.compile(r"BARS_PER_YEAR\s*=\s*(\{[^}]*\})")
    copies = {}
    for rel in ('volatility.py', 'backtest.py', 'scripts/hypersearch_v2.py'):
        m = pat.search((REPO / rel).read_text())
        assert m, f"BARS_PER_YEAR assignment not found in {rel}"
        copies[rel] = ast.literal_eval(m.group(1))
    assert copies['volatility.py'] == copies['backtest.py'] \
        == copies['scripts/hypersearch_v2.py'] == {'crypto': 8760, 'stock': 1638}


# --- gap_audit: yfinance MultiIndex + single-ticker guard ---

def _fake_yf_frame(multi=True):
    idx = pd.date_range('2024-01-02', periods=30, freq='B')
    cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = pd.DataFrame({c: np.linspace(100.0, 130.0, 30) for c in cols},
                      index=idx)
    if multi:
        df.columns = pd.MultiIndex.from_product(
            [cols, ['AAPL']], names=['Price', 'Ticker'])
    return df


class TestFetchDaily:
    def test_flattens_multiindex_columns(self, monkeypatch):
        import yfinance as yf
        monkeypatch.setattr(yf, 'download',
                            lambda *a, **k: _fake_yf_frame(multi=True))
        df = gap_audit.fetch_daily('AAPL')
        assert not isinstance(df.columns, pd.MultiIndex)
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close']
        # Downstream consumer gets clean 1-D series, not (n,1) frames.
        overnight, intraday = gap_audit.overnight_intraday_returns(df)
        assert overnight.ndim == 1 and len(overnight) == 29

    def test_flat_columns_pass_through(self, monkeypatch):
        import yfinance as yf
        monkeypatch.setattr(yf, 'download',
                            lambda *a, **k: _fake_yf_frame(multi=False))
        df = gap_audit.fetch_daily('AAPL')
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close']

    @pytest.mark.parametrize('bad', ['AAPL MSFT', 'AAPL,MSFT',
                                     ['AAPL', 'MSFT'], '', '   ', None])
    def test_rejects_non_single_ticker(self, bad):
        with pytest.raises(ValueError):
            gap_audit.fetch_daily(bad)             # raises before any network


class TestGapStatsTFit:
    def test_fit_failure_visible_and_sentinel_kept(self, monkeypatch, capsys):
        import scipy.stats
        def boom(*a, **k):
            raise ValueError('no convergence')
        monkeypatch.setattr(scipy.stats.t, 'fit', boom)
        r = np.random.RandomState(2).normal(0, 0.01, 500)
        s = gap_audit.gap_stats(r)
        assert s['t_df'] is None                   # sentinel unchanged
        assert s['std'] is not None                # rest still computed
        out = capsys.readouterr().out
        assert 'Student-t fit failed' in out and 'ValueError' in out

    def test_fit_success_path_unchanged(self):
        r = np.random.RandomState(0).standard_t(4, size=3000) * 0.01
        s = gap_audit.gap_stats(r)
        assert s['t_df'] is not None and 2.0 < s['t_df'] < 10.0


# --- execution_report ---

def test_journal_dir_comes_from_trade_journal():
    assert execution_report.JOURNAL_DIR is trade_journal.JOURNAL_DIR


def test_dead_sys_path_insert_removed():
    src = (REPO / 'execution_report.py').read_text()
    assert 'sys.path.insert' not in src
    assert '\nimport sys\n' not in src


def test_fee_comment_math_corrected():
    src = (REPO / 'execution_report.py').read_text()
    assert '30bps + 25*maker_share' not in src     # 55 bps at share=1: nonsense
    assert '50 - 10*maker_share' in src            # entry 25-10*s + taker exit 25


@pytest.fixture
def tmp_journal(monkeypatch, tmp_path):
    jdir = tmp_path / 'journals'
    jdir.mkdir()
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', jdir)
    monkeypatch.setattr(execution_report, 'BASE_DIR', tmp_path)
    return jdir, tmp_path


def _write_journal(jdir, entries, extra_lines=()):
    path = jdir / f"{datetime.now().date().isoformat()}.jsonl"
    with open(path, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')
        for ln in extra_lines:
            f.write(ln + '\n')
    return path


def test_load_counts_corrupt_lines_not_blanks(tmp_journal):
    jdir, _ = tmp_journal
    _write_journal(
        jdir,
        [{'symbol': 'BTC/USD', 'action': 'buy', 'slippage_bps': 3.0},
         {'symbol': 'AAPL', 'action': 'sell', 'slippage_bps': 1.0,
          'exit_reason': 'stop'}],
        extra_lines=['{"truncated": tru', '', '   '])
    rows, n_skipped = execution_report._load(0)
    assert len(rows) == 2
    assert n_skipped == 1          # blank lines are expected, not corruption


def test_run_report_notes_unparseable_lines(tmp_journal, capsys):
    jdir, _ = tmp_journal
    _write_journal(jdir, [{'symbol': 'BTC/USD', 'action': 'buy',
                           'slippage_bps': 1.0}],
                   extra_lines=['not json at all'])
    execution_report.run_report(days=0)
    assert '1 unparseable journal line' in capsys.readouterr().out


def test_run_report_empty_window_writes_stamped_artifact(tmp_journal, capsys):
    _, base = tmp_journal
    rep = execution_report.run_report(days=3)
    assert rep['window_days'] == 3
    datetime.fromisoformat(rep['generated_at'])    # parseable timestamp
    on_disk = json.loads((base / 'execution_report.json').read_text())
    assert on_disk == rep                          # stale file impossible
    assert 'No fills with slippage data yet' in capsys.readouterr().out


def test_maker_share_uses_live_gate_predicate_and_caveats_small_n(
        tmp_journal, capsys):
    jdir, base = tmp_journal
    _write_journal(jdir, [
        {'symbol': 'BTC/USD', 'action': 'buy', 'entry_tactic': 'maker_join',
         'slippage_bps': 1.0},
        {'symbol': 'ETH/USD', 'action': 'buy', 'entry_tactic': 'taker',
         'slippage_bps': 2.0},
        # Legacy slash-less crypto symbol: fees.realized_crypto_maker_share
        # ignores it, so the report must too or the cross-check diverges
        # (the old broad predicate would have made the share 1/3).
        {'symbol': 'BTCUSD', 'action': 'buy', 'entry_tactic': 'taker',
         'slippage_bps': 2.0},
        {'symbol': 'AAPL', 'action': 'buy', 'entry_tactic': 'taker',
         'slippage_bps': 0.5},
    ])
    rep = execution_report.run_report(days=0)
    assert rep['crypto_maker_share'] == 0.5        # 1 maker of 2 slash entries
    assert rep['window_days'] == 0 and 'generated_at' in rep
    out = capsys.readouterr().out
    assert f'below live-gate min n={fees.MAKER_SHARE_MIN_ENTRIES}' in out
    on_disk = json.loads((base / 'execution_report.json').read_text())
    assert on_disk['crypto_maker_share'] == 0.5


def test_maker_share_no_caveat_at_live_gate_n(tmp_journal, capsys):
    jdir, _ = tmp_journal
    n = fees.MAKER_SHARE_MIN_ENTRIES
    _write_journal(jdir, [
        {'symbol': 'BTC/USD', 'action': 'buy', 'slippage_bps': 1.0,
         'entry_tactic': 'maker_join' if i % 2 else 'taker'}
        for i in range(n)])
    execution_report.run_report(days=0)
    assert 'below live-gate min' not in capsys.readouterr().out
