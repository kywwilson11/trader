"""Source-contract pins for the training group (2026-07 design pass):
scripts/hypersearch_v2.py, scripts/harvest_crypto_data.py,
scripts/wave6_stage0.py.

hypersearch_v2 / harvest_crypto_data cannot be imported on the dev Mac
(torch / dotenv), so hypersearch-side invariants are pinned against the
SOURCE (same pattern as test_review_b17.test_bars_per_year_copies_in_sync).
Consolidating the duplicated literals into adaptive_config is an
objective-file edit (CLAUDE.md gotcha #2) and was deliberately NOT done —
these pins make silent drift fail the suite instead.
"""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from adaptive_config import DEFAULT_SEARCH_SPACE

HS = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
HARVEST = (REPO / 'scripts' / 'harvest_crypto_data.py').read_text()


def test_hypersearch_forward_bars_literal_matches_default():
    m = re.search(r'^FORWARD_BARS\s*=\s*(\[[^\]]*\])', HS, re.M)
    assert m, 'FORWARD_BARS literal not found in hypersearch_v2.py'
    assert ast.literal_eval(m.group(1)) == DEFAULT_SEARCH_SPACE['forward_bars']


def test_objective_fallback_literals_match_default_search_space():
    """The _space.get() fallbacks in the objective duplicate
    DEFAULT_SEARCH_SPACE (module_review fixes_deferred[23]); pin them
    equal so drift fails loudly without editing the objective."""
    pairs = re.findall(r"_space\.get\('(\w+)',\s*(\[[^\]]*\])\)", HS)
    found = {name: ast.literal_eval(lit) for name, lit in pairs}
    # forward_bars falls back to the FORWARD_BARS global (pinned above)
    expected = {k: v for k, v in DEFAULT_SEARCH_SPACE.items()
                if k != 'forward_bars'}
    assert found == expected


def test_wave6_reference_horizons_match_default():
    import wave6_stage0
    assert wave6_stage0.FORWARD_BARS == DEFAULT_SEARCH_SPACE['forward_bars']


def test_adaptive_update_runs_after_study_reads():
    """update_after_search may DELETE the study DB on categorical
    expansion; importance/PBO/Monte-Carlo must read the study FIRST."""
    call = HS.index('update_after_search(adaptive_state')
    assert HS.index('get_param_importances') < call
    assert HS.index('pbo_from_fold_scores') < call
    assert HS.index('Monte Carlo robustness') < call


def test_missing_horizon_substitution_warns():
    assert '[WARN] Target_Return columns missing' in HS


def test_holdout_dead_params_removed():
    assert 'trial_values' not in HS


def test_no_dead_dual_reference():
    assert 'hypersearch_dual' not in HS


def test_harvest_frees_previous_panel_before_concat():
    assert HARVEST.index('del existing') \
        < HARVEST.index('final_df = pd.concat(all_data)')
