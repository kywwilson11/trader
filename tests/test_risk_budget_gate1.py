"""Wave-8 #7: cross-book account stop-risk GATE-1 measurement.

The per-book ENB cap runs independently in each loop process, so the stock book
(COIN/MSTR/MARA) and crypto book (spot BTC/ETH) can each run to MAX_BOOK_RISK_PCT
behind the SAME factor — ~5% combined vs the ~3% account cap. These tests cover
the shared registry + the combined-risk report that JOURNALS (does not clamp)
this stacking, so the live cap can be justified on real data before it ships.
"""
import json

import pytest

from risk_budget import (
    read_registry,
    write_book_risk,
    account_risk_gate1_report,
    record_book_risk_and_report,
    ACCOUNT_RISK_CAP,
)


def test_registry_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "reg.json")
    assert read_registry(p) == {}
    reg = write_book_risk("crypto", 0.024, 0.7, path=p, now=1000.0)
    assert reg["crypto"]["risk"] == 0.024
    on_disk = json.loads((tmp_path / "reg.json").read_text())
    assert on_disk["crypto"]["rho"] == 0.7 and on_disk["crypto"]["ts"] == 1000.0


def test_lockstep_books_stack_to_the_sum_and_breach_cap():
    # Both books at 0.025 diversified risk; lockstep (rho_cross=1) -> account
    # risk == the per-book SUM (0.05), which BREACHES the 0.03 account cap.
    reg = {"stock": {"risk": 0.025, "rho": 0.6, "ts": 100.0},
           "crypto": {"risk": 0.025, "rho": 0.8, "ts": 100.0}}
    rep = account_risk_gate1_report(reg, rho_cross=1.0, now=100.0)
    assert rep["account_stop_risk"] == pytest.approx(0.05, abs=1e-6)
    assert rep["book_sum"] == pytest.approx(0.05)
    assert rep["concentration"] == pytest.approx(1.0)
    assert rep["over_cap"] is True
    assert rep["headroom"] == pytest.approx(ACCOUNT_RISK_CAP - 0.05)


def test_hedged_books_net_below_the_sum():
    reg = {"stock": {"risk": 0.02, "rho": 0.6, "ts": 0.0},
           "crypto": {"risk": 0.02, "rho": 0.6, "ts": 0.0}}
    hedged = account_risk_gate1_report(reg, rho_cross=-1.0, now=0.0)
    lockstep = account_risk_gate1_report(reg, rho_cross=1.0, now=0.0)
    assert hedged["account_stop_risk"] < lockstep["account_stop_risk"]
    assert hedged["account_stop_risk"] == pytest.approx(0.0, abs=1e-9)  # equal & opposite


def test_stale_book_is_dropped():
    reg = {"stock": {"risk": 0.025, "rho": 0.6, "ts": 0.0},      # 2000s old -> stale
           "crypto": {"risk": 0.02, "rho": 0.8, "ts": 2000.0}}
    rep = account_risk_gate1_report(reg, rho_cross=1.0, stale_after_s=900.0, now=2000.0)
    assert rep["stale_books"] == ["stock"]
    assert rep["stock_risk"] == 0.0                              # dropped
    assert rep["account_stop_risk"] == pytest.approx(0.02)       # crypto only


def test_single_book_and_empty_registry_fail_open():
    one = account_risk_gate1_report({"crypto": {"risk": 0.018, "rho": 0.7, "ts": 5.0}},
                                    rho_cross=1.0, now=5.0)
    assert one["account_stop_risk"] == pytest.approx(0.018)
    assert one["over_cap"] is False
    assert one["concentration"] == pytest.approx(1.0)   # one book -> account == sum
    empty = account_risk_gate1_report({}, now=0.0)
    assert empty["account_stop_risk"] == 0.0 and empty["over_cap"] is False
    assert empty["concentration"] is None
    # corrupt entries are ignored, not crashed
    bad = account_risk_gate1_report({"stock": "oops", "crypto": {"risk": None}}, now=0.0)
    assert bad["account_stop_risk"] == 0.0


def test_record_writes_diversified_risk_and_reports(tmp_path):
    p = str(tmp_path / "reg.json")
    # Crypto book reports first (two 0.02 positions, fairly correlated).
    record_book_risk_and_report("crypto", [0.02, 0.02], 0.8, rho_cross=1.0,
                                path=p, now=10.0)
    # Stock book reports; now both are in the registry and combine.
    rep = record_book_risk_and_report("stock", [0.02, 0.02], 0.7, rho_cross=1.0,
                                      path=p, now=11.0)
    reg = read_registry(p)
    assert set(reg) == {"crypto", "stock"}
    assert rep["stock_risk"] > 0 and rep["crypto_risk"] > 0
    # diversified each-book risk < the naive 0.04 sum (rho<1 within book)
    assert reg["stock"]["risk"] < 0.04
    assert rep["book_sum"] == pytest.approx(rep["stock_risk"] + rep["crypto_risk"])
