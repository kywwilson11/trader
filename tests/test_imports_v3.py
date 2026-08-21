"""P0 verification-stack tests (campaign 2026-08, B16 core).

Four groups:
1. Skip-policy unit tests for tests/test_imports.py::import_or_skip_missing_heavy
   (missing HEAVY dep -> SKIP; anything else -> FAIL).
2. scripts/ab_check.sh gate-harness tests via a stubbed AB_CHECK_PYTEST —
   direct regressions for the D36 false-PASS (empty output), the sanity floor,
   the timeout wall, and the flaky/persistent NEW-name triage.
3. LightGBM train/save/load/predict round-trip smoke (D37) — runs only where
   lightgbm exists (CI Jetson-parity leg + Jetson; skips on the dev Mac).
4. alpaca-py import smoke (D37) — the two module families alpaca_compat.py
   lazily imports; skips where alpaca-py is absent.
"""

import importlib
import importlib.util
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TESTS_DIR))

from test_imports import HEAVY_DEPS, import_or_skip_missing_heavy  # noqa: E402


# ---------------------------------------------------------------------------
# Group 1 — skip policy: heavy-dep-missing skips, everything else fails
# ---------------------------------------------------------------------------

def test_heavy_dep_missing_skips(tmp_path, monkeypatch):
    """ModuleNotFoundError rooted at a HEAVY dep -> pytest.skip, not FAIL."""
    if importlib.util.find_spec("torch"):
        pytest.skip("torch importable here (full-deps env) — scenario N/A")
    assert "torch" in HEAVY_DEPS
    (tmp_path / "fake_mod_heavy_p0.py").write_text("import torch\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        with pytest.raises(pytest.skip.Exception):
            import_or_skip_missing_heavy("fake_mod_heavy_p0")
    finally:
        sys.modules.pop("fake_mod_heavy_p0", None)


def test_nonheavy_missing_fails(tmp_path, monkeypatch):
    """A non-heavy missing module (typo / repo-internal) must FAIL, not skip."""
    (tmp_path / "fake_mod_typo_p0.py").write_text(
        "import definitely_not_a_real_dep_p0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        with pytest.raises(ModuleNotFoundError) as excinfo:
            import_or_skip_missing_heavy("fake_mod_typo_p0")
        assert not isinstance(excinfo.value, pytest.skip.Exception)
    finally:
        sys.modules.pop("fake_mod_typo_p0", None)


def test_broken_import_fails(tmp_path, monkeypatch):
    """A plain ImportError (bad from-import) must FAIL, not skip."""
    (tmp_path / "fake_mod_broken_p0.py").write_text(
        "from json import no_such_name_p0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        with pytest.raises(ImportError) as excinfo:
            import_or_skip_missing_heavy("fake_mod_broken_p0")
        assert not isinstance(excinfo.value, pytest.skip.Exception)
    finally:
        sys.modules.pop("fake_mod_broken_p0", None)


# ---------------------------------------------------------------------------
# Group 2 — ab_check.sh gate harness (stubbed pytest via AB_CHECK_PYTEST)
# ---------------------------------------------------------------------------

def _write_stub(tmp_path, body):
    stub = tmp_path / "fake_pytest.sh"
    stub.write_text("#!/bin/sh\n" + body)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_ab(stub, env_extra):
    env = dict(os.environ)
    env["AB_CHECK_PYTEST"] = str(stub)
    env.update(env_extra)
    return subprocess.run(
        ["sh", str(_REPO_ROOT / "scripts" / "ab_check.sh")],
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=60)


def test_empty_output_is_fail(tmp_path):
    """D36 exact scenario: pytest produces no output -> FAIL, never PASS."""
    stub = _write_stub(tmp_path, "exit 2\n")
    proc = _run_ab(stub, {"AB_CHECK_MIN_PASSED": "10"})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "summary line missing" in out or "sanity floor" in out
    assert "ab_check: PASS" not in proc.stdout


def test_low_passed_count_is_fail(tmp_path):
    """A partial run below the default sanity floor must FAIL."""
    stub = _write_stub(tmp_path, 'echo "== 3 passed in 0.1s =="\nexit 0\n')
    proc = _run_ab(stub, {})  # default MIN_PASSED=1500 stays in force
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "sanity floor" in out


def test_clean_run_passes(tmp_path):
    """A clean full run (no FAILED/ERROR names) still exits 0."""
    stub = _write_stub(tmp_path, 'echo "== 1600 passed in 1.0s =="\nexit 0\n')
    proc = _run_ab(stub, {"AB_CHECK_MIN_PASSED": "10"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NEW failures: none" in proc.stdout
    # Downstream tooling greps these exact line formats — pin them.
    assert "DISAPPEARED failures" in proc.stdout
    assert "ab_check: PASS" in proc.stdout


def test_new_name_fails_and_is_labeled_flaky(tmp_path):
    """A NEW name that passes the isolated rerun is labeled flaky — still FAIL."""
    stub = _write_stub(tmp_path, (
        'if [ "$1" = "tests/" ]; then\n'
        '  echo "FAILED tests/fake_p0.py::test_new"\n'
        '  echo "== 1 failed, 1600 passed in 1.0s =="\n'
        '  exit 1\n'
        'else\n'
        '  echo "== 1 passed in 0.1s =="\n'
        '  exit 0\n'
        'fi\n'))
    proc = _run_ab(stub, {"AB_CHECK_MIN_PASSED": "10"})
    assert proc.returncode == 1
    assert "FAILED tests/fake_p0.py::test_new" in proc.stdout
    assert "[flaky" in proc.stdout
    assert "ab_check: FAIL" in proc.stdout


def test_new_name_fails_and_is_labeled_persistent(tmp_path):
    """A NEW name that fails the isolated rerun too is labeled persistent."""
    stub = _write_stub(tmp_path, (
        'echo "FAILED tests/fake_p0.py::test_new"\n'
        'if [ "$1" = "tests/" ]; then\n'
        '  echo "== 1 failed, 1600 passed in 1.0s =="\n'
        'else\n'
        '  echo "== 1 failed in 0.1s =="\n'
        'fi\n'
        'exit 1\n'))
    proc = _run_ab(stub, {"AB_CHECK_MIN_PASSED": "10"})
    assert proc.returncode == 1
    assert "[persistent]" in proc.stdout


def test_timeout_is_fail(tmp_path):
    """A hung pytest run is killed by the watchdog and reported as FAIL."""
    stub = _write_stub(tmp_path, "sleep 60\n")
    t0 = time.monotonic()
    proc = _run_ab(stub, {"AB_CHECK_TIMEOUT_S": "2",
                          "AB_CHECK_MIN_PASSED": "10"})
    wall = time.monotonic() - t0
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "TIMED OUT" in out
    assert wall < 30


# ---------------------------------------------------------------------------
# Group 3 — LightGBM round-trip smoke (D37; CI parity leg + Jetson only)
# ---------------------------------------------------------------------------
# NOTE: importorskip deliberately lives INSIDE the test (module-level would
# skip this whole file on the dev Mac, silencing groups 1/2).

def test_lgb_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    import model_lgb

    # Round-trip through the repo's own atomic-save path, sandboxed.
    monkeypatch.setattr(model_lgb, "_MODEL_DIR", str(tmp_path))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    y = 0.1 * X[:, 0] + 0.01 * rng.normal(size=50)
    model = model_lgb.train_lgb(
        X, y, params={"num_leaves": 7, "min_data_in_leaf": 5})
    model_lgb.save_lgb_model(model, prefix="citest")
    loaded = model_lgb.load_lgb_model(prefix="citest")
    assert loaded is not None

    p_mem = model_lgb.predict_lgb(model, X[0])
    p_disk = model_lgb.predict_lgb(loaded, X[0])
    assert np.isfinite(p_disk)
    assert p_disk == pytest.approx(p_mem)

    assert model_lgb.ensemble_predict(1.0, None) == 1.0
    assert model_lgb.ensemble_predict(1.0, 0.0, lstm_weight=0.6) == \
        pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Group 4 — alpaca-py import smoke (D37; parity leg + Jetson only)
# ---------------------------------------------------------------------------

def test_alpaca_py_imports():
    pytest.importorskip("alpaca")
    # The two module families alpaca_compat.py lazily imports.
    import alpaca.data.historical  # noqa: F401
    import alpaca.trading.client  # noqa: F401
