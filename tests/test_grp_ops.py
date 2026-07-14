"""Tests for the 2026-07 GRP-ops behavior-neutral fixes: monitor_drift's
load_holdout_hit_rate AttributeError escape, hw_monitor's get_ram_usage
OSError contract, execution_report's fees-derived entry-fee arithmetic,
and stock_config's TRADABLE_POOL doc-honesty comment."""
import builtins
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_load_holdout_hit_rate_nondict_manifest_returns_none(
        monkeypatch, tmp_path):
    import monitor_drift as md
    monkeypatch.setattr(md, 'BASE_DIR', tmp_path)
    manifest_path = tmp_path / 'model_v2.manifest.json'

    manifest_path.write_text(json.dumps(['not', 'a', 'dict']))
    assert md.load_holdout_hit_rate('') is None

    manifest_path.write_text(json.dumps({'holdout': ['x']}))
    assert md.load_holdout_hit_rate('') is None

    manifest_path.write_text(json.dumps({'holdout': {'hit_rate': 0.55}}))
    assert md.load_holdout_hit_rate('') == 0.55


def test_get_ram_usage_returns_none_pair_on_oserror(monkeypatch):
    import hw_monitor as hw

    def boom(*a, **k):
        raise PermissionError('EACCES')

    monkeypatch.setattr(builtins, 'open', boom)
    assert hw.get_ram_usage() == (None, None)


def test_execution_report_entry_fee_derives_from_fees(
        capsys, monkeypatch, tmp_path):
    import execution_report as er
    import fees

    monkeypatch.setattr(er, 'BASE_DIR', tmp_path)

    rows = []
    for i in range(40):
        rows.append({
            'symbol': 'BTC/USD',
            'action': 'buy',
            'slippage_bps': 1.0,
            'entry_tactic': 'maker' if i < 20 else 'taker',
        })
    monkeypatch.setattr(er, '_load', lambda days: (rows, 0))

    er.run_report(1)
    captured = capsys.readouterr().out

    share = 0.5
    expected_fee = (fees.CRYPTO_TAKER_BPS
                    - (fees.CRYPTO_TAKER_BPS - fees.CRYPTO_MAKER_BPS) * share)
    assert (f"{expected_fee:.1f} bps vs {fees.CRYPTO_TAKER_BPS:.0f} taker"
            in captured)


def test_stock_config_tradable_pool_marked_not_wired():
    src = (Path(__file__).resolve().parent.parent
           / 'stock_config.py').read_text()
    assert 'NOT YET WIRED' in src
    assert 'scores universe + TRAINING_CANDIDATE_POOL every hour' not in src

    import stock_config
    assert stock_config.TRADABLE_POOL_ENABLED is False
