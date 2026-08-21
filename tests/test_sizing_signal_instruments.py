"""2026-07 decision-audit instruments: sizing decomposition + signal-exit
counterfactual.

base_loop imports torch, so its wiring is pinned at source level (same
pattern as test_prediction_cache); signal_exit_audit is exercised
functionally with synthetic bars and a monkeypatched fetcher.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# --- sizing decomposition wiring (source contract) ---

def test_sizing_detail_captures_every_layer():
    src = (REPO / "base_loop.py").read_text()
    start = src.index("def _compute_position_size")
    fn = src[start:src.index("\n    def ", start)]
    for key in ("'stop_dist'", "'base'", "'kelly_mult'", "'vol_mult'",
                "'signal_conf'", "'vix_tilt'", "'dd_mult'", "'macro_mult'",
                "'corr_mult'", "'hmm_mult'", "'disagree_mult'",
                "'sentiment_mult'", "'llm_mult'", "'meta_mult'",
                "'book_vol_mult'", "'tilt_raw'", "'tilt'",
                "'sized_pre_caps'", "'book_risk_scale'", "'leverage_div'",
                "'degraded_inputs'"):
        assert key in fn, f"sizing detail missing {key}"
    # stash is flag-gated measurement, not behavior
    assert "CONVICTION_JOURNAL_ENABLED" in fn
    assert "_last_sizing_detail" in fn


def test_buy_journal_row_carries_sizing():
    src = (REPO / "base_loop.py").read_text()
    rec = src[src.index('buy_rec = {"symbol": symbol, "action": "buy"'):]
    rec = rec[:rec.index("log_decision(buy_rec)")]
    assert "_last_sizing_detail" in rec
    assert "buy_rec['sizing']" in rec


def test_sizing_arithmetic_unchanged():
    # the decomposition must be write-only: every detail line is either an
    # assignment into detail{} or a hoisted local multiplied exactly once
    src = (REPO / "base_loop.py").read_text()
    start = src.index("def _compute_position_size")
    fn = src[start:src.index("\n    def ", start)]
    # the final size formula is untouched
    assert "sized = base * kelly_mult * vol_mult * tilt" in fn
    # clamp unchanged
    assert "tilt = max(0.1, min(TILT_MAX, tilt))" in fn


# --- signal-exit counterfactual (functional) ---

def _bars(rip=True, n=400):
    idx = pd.date_range('2026-06-01', periods=n, freq='h', tz='UTC')
    if rip:
        close = 100.0 * (1 + 0.002) ** np.arange(n)       # steady ramp up
    else:
        close = 100.0 * (1 - 0.002) ** np.arange(n)       # steady bleed
    return pd.DataFrame({
        'Open': close, 'High': close * 1.003, 'Low': close * 0.997,
        'Close': close, 'Volume': np.full(n, 10.0)}, index=idx)


def _rows(ts):
    return [{'action': 'sell', 'exit_reason': 'signal_sell',
             'symbol': 'BTC/USD', 'ts': str(ts), 'pnl_pct': 1.0},
            {'action': 'sell', 'exit_reason': 'hard_stop',   # ignored
             'symbol': 'BTC/USD', 'ts': str(ts), 'pnl_pct': -2.0},
            {'action': 'buy', 'symbol': 'BTC/USD', 'ts': str(ts)}]


def test_signal_exit_audit_prices_forgone_upside(monkeypatch):
    import market_data
    import decision_report as dr
    bars = _bars(rip=True)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, sym, **k: bars)
    out = dr.signal_exit_audit(_rows(bars.index[50]), api=object())
    assert out['n_signal_sells'] == 1          # hard_stop row excluded
    assert out['priced'] == 1
    # dumped into a steady rip: the stop stack would have out-earned the flip
    assert out['counterfactual_mean_net_pct'] > 0
    assert out['given_up_total_pct'] > 0
    assert out['realized_mean_pnl_pct'] == 1.0


def test_signal_exit_audit_credits_good_flips(monkeypatch):
    import market_data
    import decision_report as dr
    bars = _bars(rip=False)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, sym, **k: bars)
    out = dr.signal_exit_audit(_rows(bars.index[50]), api=object())
    assert out['priced'] == 1
    # selling into a bleed saved money: counterfactual is negative
    assert out['counterfactual_mean_net_pct'] < 0


def test_signal_exit_audit_empty_and_unresolved(monkeypatch):
    import market_data
    import decision_report as dr
    assert dr.signal_exit_audit([], api=object()) == {'n_signal_sells': 0}
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda api, sym, **k: None)
    out = dr.signal_exit_audit(_rows(pd.Timestamp('2026-06-03', tz='UTC')),
                               api=object())
    assert out['priced'] == 0 and out['_unresolved'] == 1
