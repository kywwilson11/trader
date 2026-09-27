"""SIG-R3-HGATES: adaptive_state['cum_holdout_gates'] finally has a writer.

adaptive_config seeds / back-fills `cum_holdout_gates` (adaptive_config.py
_default_state + load_adaptive_state) but nothing ever incremented it, so any
"deflate by the number of HOLDOUT evaluations" read silently got 0. Documented
intent (research/campaign_2026-08/02_research.md, B03.2): "+1 per winner
actually scored on the holdout" — i.e. per EVALUATION (pass or fail); a None
report scored nothing and is not counted.

The patch (scripts/hypersearch_v2.py only — adaptive_config.py untouched):
`_record_holdout_gate(adaptive_state, asset_type)` increments the in-memory
state AND the on-disk state, called right after evaluate_on_holdout when the
report is not None. The pins below prove the count survives every downstream
persistence branch of main(): update_after_search (saves the in-memory
object), the initial-params save + record_trials, and the losing-run
record_trials (reloads from disk).

Module under test: SIG_R3_HS (default: the repo's scripts/hypersearch_v2.py).
"""
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('optuna')
pytest.importorskip('sklearn')

REPO = Path(os.environ.get('SIG_R3_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R3_HS', str(REPO / 'scripts' / 'hypersearch_v2.py'))

import adaptive_config as ac  # noqa: E402


@pytest.fixture(scope='module')
def hs():
    spec = importlib.util.spec_from_file_location('hs_sig_r3_hgates', HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stub_state_dir(tmp_path, monkeypatch):
    """Adaptive state lives in tmp_path (never the repo root)."""
    monkeypatch.setattr(ac, 'BASE_DIR', tmp_path)
    st = ac._default_state('hgtest')
    st['cum_holdout_gates'] = 3
    st['cum_trials'] = 40
    ac.save_adaptive_state(st)
    return tmp_path


def _disk(tmp_path):
    return json.loads((tmp_path / 'adaptive_state_hgtest.json').read_text())


def test_record_holdout_gate_increments_memory_and_disk(hs, stub_state_dir):
    mem = ac.load_adaptive_state('hgtest')
    hs._record_holdout_gate(mem, 'hgtest')
    assert mem['cum_holdout_gates'] == 4
    assert _disk(stub_state_dir)['cum_holdout_gates'] == 4
    assert _disk(stub_state_dir)['cum_trials'] == 40      # nothing else moved


def test_count_survives_update_after_search_branch(hs, stub_state_dir):
    mem = ac.load_adaptive_state('hgtest')
    hs._record_holdout_gate(mem, 'hgtest')
    ac.update_after_search(mem, 0.5, {'seq_len': 20, 'hidden_dim': 128},
                           study_db_path=None, new_trials_completed=5)
    d = _disk(stub_state_dir)
    assert d['cum_holdout_gates'] == 4 and d['cum_trials'] == 45


def test_count_survives_losing_run_record_trials_branch(hs, stub_state_dir):
    mem = ac.load_adaptive_state('hgtest')
    mem['best_params'] = {'seq_len': 20}    # => main() skips the in-memory save
    hs._record_holdout_gate(mem, 'hgtest')
    ac.record_trials('hgtest', 7, event='search_no_update')  # reloads disk
    d = _disk(stub_state_dir)
    assert d['cum_holdout_gates'] == 4 and d['cum_trials'] == 47


def test_count_survives_initial_params_save_then_record_trials(hs,
                                                               stub_state_dir):
    mem = ac.load_adaptive_state('hgtest')
    hs._record_holdout_gate(mem, 'hgtest')
    mem['best_params'] = {'seq_len': 20}
    ac.save_adaptive_state(mem)             # main()'s initial-params save
    ac.record_trials('hgtest', 2, event='search_no_update')
    assert _disk(stub_state_dir)['cum_holdout_gates'] == 4


def test_fail_soft_on_corrupt_state(hs, stub_state_dir):
    (stub_state_dir / 'adaptive_state_hgtest.json').write_text('{corrupt')
    mem = {'asset_type': 'hgtest', 'cum_holdout_gates': 1}
    hs._record_holdout_gate(mem, 'hgtest')  # must not raise
    assert mem['cum_holdout_gates'] == 2


def test_main_calls_it_once_per_scored_holdout():
    """Call-site pin (main() cannot run without a store): the statement right
    after `holdout_report = evaluate_on_holdout(...)` is
    `if holdout_report is not None: _record_holdout_gate(adaptive_state,
    asset_type)`, and it is the only call."""
    src = Path(HS_PATH).read_text()
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == '_record_holdout_gate']
    assert len(calls) == 1
    found = False
    for node in ast.walk(main):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for i, st in enumerate(body[:-1]):
            if (isinstance(st, ast.Assign)
                    and getattr(st.targets[0], 'id', None) == 'holdout_report'
                    and isinstance(st.value, ast.Call)
                    and getattr(st.value.func, 'id', None) == 'evaluate_on_holdout'):
                nxt = body[i + 1]
                assert isinstance(nxt, ast.If)
                assert ast.unparse(nxt.test) == 'holdout_report is not None'
                assert ast.unparse(nxt.body[0]) == \
                    '_record_holdout_gate(adaptive_state, asset_type)'
                found = True
    assert found
