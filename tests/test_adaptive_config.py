"""Tests for adaptive_config.py — edge detection, expansion, mode decisions."""

import json
import os
import tempfile
from unittest import mock

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_config import (
    DEFAULT_SEARCH_SPACE,
    HARD_LIMITS,
    detect_edges,
    expand_search_space,
    decide_mode,
    get_trial_count,
    get_search_space_for_trial,
    update_after_search,
    load_adaptive_state,
    save_adaptive_state,
    get_max_forward_bars,
    get_forward_bars_list,
)


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    def test_categorical_at_high_edge(self):
        """forward_bars=48 in [12,18,24,32,48] -> high edge."""
        best = {'forward_bars': 48}
        space = {'forward_bars': [12, 18, 24, 32, 48]}
        edges = detect_edges(best, space)
        assert ('forward_bars', 'high') in edges

    def test_categorical_at_low_edge(self):
        """seq_len=12 in [12,18,24,32] -> low edge."""
        best = {'seq_len': 12}
        space = {'seq_len': [12, 18, 24, 32]}
        edges = detect_edges(best, space)
        assert ('seq_len', 'low') in edges

    def test_categorical_in_middle(self):
        """hidden_dim=128 in [64,96,128,192,256] -> no edge."""
        best = {'hidden_dim': 128}
        space = {'hidden_dim': [64, 96, 128, 192, 256]}
        edges = detect_edges(best, space)
        assert len(edges) == 0

    def test_float_near_high_boundary(self):
        """dropout=0.39 with range [0.10, 0.40] -> high edge (within 10%)."""
        best = {'dropout': 0.39}
        space = {'dropout': [0.10, 0.40]}
        edges = detect_edges(best, space)
        assert ('dropout', 'high') in edges

    def test_float_near_low_boundary(self):
        """dropout=0.12 with range [0.10, 0.40] -> low edge."""
        best = {'dropout': 0.12}
        space = {'dropout': [0.10, 0.40]}
        edges = detect_edges(best, space)
        assert ('dropout', 'low') in edges

    def test_float_in_middle(self):
        """dropout=0.25 with range [0.10, 0.40] -> no edge."""
        best = {'dropout': 0.25}
        space = {'dropout': [0.10, 0.40]}
        edges = detect_edges(best, space)
        assert len(edges) == 0

    def test_int_at_max(self):
        """num_layers=2 with [1,2] -> high edge."""
        best = {'num_layers': 2}
        space = {'num_layers': [1, 2]}
        edges = detect_edges(best, space)
        assert ('num_layers', 'high') in edges

    def test_int_at_min(self):
        """num_layers=1 with [1,2] -> low edge."""
        best = {'num_layers': 1}
        space = {'num_layers': [1, 2]}
        edges = detect_edges(best, space)
        assert ('num_layers', 'low') in edges

    def test_multiple_edges(self):
        """Multiple params at edges detected simultaneously."""
        best = {'forward_bars': 48, 'seq_len': 12, 'dropout': 0.25}
        space = {
            'forward_bars': [12, 18, 24, 32, 48],
            'seq_len': [12, 18, 24, 32],
            'dropout': [0.10, 0.40],
        }
        edges = detect_edges(best, space)
        assert ('forward_bars', 'high') in edges
        assert ('seq_len', 'low') in edges
        assert len(edges) == 2  # dropout is in middle

    def test_unknown_param_ignored(self):
        """Params not in search space are ignored."""
        best = {'unknown_param': 999}
        space = {'forward_bars': [12, 18, 24, 32, 48]}
        edges = detect_edges(best, space)
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# Search Space Expansion
# ---------------------------------------------------------------------------

class TestExpansion:
    def test_expand_categorical_high(self):
        """[12,18,24,32,48] + high edge -> adds 64."""
        space = {'forward_bars': [12, 18, 24, 32, 48]}
        edges = [('forward_bars', 'high')]
        new_space, logs = expand_search_space(space, edges)
        assert 64 in new_space['forward_bars']
        assert new_space['forward_bars'] == sorted(new_space['forward_bars'])
        assert len(logs) > 0

    def test_expand_categorical_low(self):
        """[12,18,24,32] + low edge -> adds 8."""
        space = {'seq_len': [12, 18, 24, 32]}
        edges = [('seq_len', 'low')]
        new_space, logs = expand_search_space(space, edges)
        assert 8 in new_space['seq_len']
        assert len(logs) > 0

    def test_expand_float_high(self):
        """dropout [0.10, 0.40] + high edge -> [0.10, 0.50]."""
        space = {'dropout': [0.10, 0.40]}
        edges = [('dropout', 'high')]
        new_space, logs = expand_search_space(space, edges)
        assert new_space['dropout'][1] == 0.50
        assert new_space['dropout'][0] == 0.10  # low unchanged

    def test_expand_float_low(self):
        """dropout [0.10, 0.40] + low edge -> [0.05, 0.40]."""
        space = {'dropout': [0.10, 0.40]}
        edges = [('dropout', 'low')]
        new_space, logs = expand_search_space(space, edges)
        assert new_space['dropout'][0] == 0.05
        assert new_space['dropout'][1] == 0.40  # high unchanged

    def test_hard_limits_respected_categorical(self):
        """hidden_dim never exceeds 384 even after expansion."""
        space = {'hidden_dim': [64, 96, 128, 192, 256, 384]}
        edges = [('hidden_dim', 'high')]
        new_space, logs = expand_search_space(space, edges)
        assert max(new_space['hidden_dim']) <= 384

    def test_hard_limits_respected_float(self):
        """dropout never goes below 0.05."""
        space = {'dropout': [0.05, 0.40]}
        edges = [('dropout', 'low')]
        new_space, logs = expand_search_space(space, edges)
        # Can't expand below 0.05 (already at hard limit)
        assert new_space['dropout'][0] >= 0.05

    def test_no_expansion_when_no_edges(self):
        """No edges -> space unchanged."""
        import copy
        space = copy.deepcopy(DEFAULT_SEARCH_SPACE)
        new_space, logs = expand_search_space(space, [])
        assert new_space == space
        assert len(logs) == 0

    def test_no_duplicate_values_after_expansion(self):
        """Expanding an already-expanded space doesn't add duplicates."""
        space = {'forward_bars': [8, 12, 18, 24, 32, 48, 64]}
        edges = [('forward_bars', 'high')]
        new_space, logs = expand_search_space(space, edges)
        assert len(new_space['forward_bars']) == len(set(new_space['forward_bars']))

    def test_expand_multiple_params(self):
        """Multiple edges expand multiple params."""
        space = {
            'forward_bars': [12, 18, 24, 32, 48],
            'seq_len': [12, 18, 24, 32],
        }
        edges = [('forward_bars', 'high'), ('seq_len', 'low')]
        new_space, logs = expand_search_space(space, edges)
        assert 64 in new_space['forward_bars']
        assert 8 in new_space['seq_len']
        assert len(logs) >= 2  # forward_bars adds 64+96, seq_len adds 8


# ---------------------------------------------------------------------------
# Mode Decision
# ---------------------------------------------------------------------------

class TestModeDecision:
    def test_refine_when_improving(self):
        """Score improving -> refine."""
        state = {
            'mode': 'refine',
            'best_score': 2.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'cycles_without_improvement': 0,
        }
        assert decide_mode(state, 2.5) == 'refine'

    def test_explore_on_edge(self):
        """Edge detected -> explore."""
        state = {
            'mode': 'refine',
            'best_score': 2.0,
            'best_params': {'forward_bars': 48},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'cycles_without_improvement': 0,
        }
        assert decide_mode(state, 2.0) == 'explore'

    def test_explore_on_stagnation(self):
        """3 cycles without improvement -> explore."""
        state = {
            'mode': 'refine',
            'best_score': 2.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'cycles_without_improvement': 3,
        }
        assert decide_mode(state, 2.0) == 'explore'

    def test_back_to_refine_after_explore(self):
        """After explore cycle -> refine."""
        state = {
            'mode': 'explore',
            'best_score': 2.0,
            'best_params': {'forward_bars': 48},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'cycles_without_improvement': 5,
        }
        # Even with edges and stagnation, explore -> refine
        assert decide_mode(state, 2.0) == 'refine'

    def test_refine_with_no_params(self):
        """No best params -> refine (no edges to detect)."""
        state = {
            'mode': 'refine',
            'best_score': 0.0,
            'best_params': {},
            'search_space': DEFAULT_SEARCH_SPACE,
            'cycles_without_improvement': 0,
        }
        assert decide_mode(state, 0.0) == 'refine'


# ---------------------------------------------------------------------------
# Trial Count
# ---------------------------------------------------------------------------

class TestTrialCount:
    def test_initial_count(self):
        assert get_trial_count('initial', is_initial=True) == 200

    def test_refine_count(self):
        assert get_trial_count('refine') == 70

    def test_explore_count(self):
        assert get_trial_count('explore') == 120

    def test_unknown_mode_defaults_to_refine(self):
        assert get_trial_count('unknown_mode') == 70


# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """Save state then load it, should match."""
        state = {
            'asset_type': 'test',
            'best_score': 3.14,
            'best_params': {'forward_bars': 24, 'hidden_dim': 192},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 1,
            'expansion_history': [],
            'last_updated': '',
        }
        state_path = tmp_path / 'adaptive_state_test.json'
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            save_adaptive_state(state)
            loaded = load_adaptive_state('test')

        assert loaded['best_score'] == 3.14
        assert loaded['best_params']['forward_bars'] == 24
        assert loaded['mode'] == 'refine'
        assert loaded['cycles_without_improvement'] == 1

    def test_default_state_creation(self, tmp_path):
        """No file -> sensible defaults."""
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            state = load_adaptive_state('newtype')

        assert state['asset_type'] == 'newtype'
        assert state['best_score'] == 0.0
        assert state['mode'] == 'refine'
        assert state['search_space'] == DEFAULT_SEARCH_SPACE

    def test_max_forward_bars(self, tmp_path):
        """get_max_forward_bars returns correct max from search space."""
        state = {
            'asset_type': 'test',
            'best_score': 0,
            'best_params': {},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48, 64]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            save_adaptive_state(state)
            result = get_max_forward_bars('test')
        assert result == 64

    def test_forward_bars_list(self, tmp_path):
        """get_forward_bars_list returns sorted list."""
        state = {
            'asset_type': 'test',
            'best_score': 0,
            'best_params': {},
            'search_space': {'forward_bars': [48, 12, 64, 24]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            save_adaptive_state(state)
            result = get_forward_bars_list('test')
        assert result == [12, 24, 48, 64]


# ---------------------------------------------------------------------------
# update_after_search integration
# ---------------------------------------------------------------------------

class TestUpdateAfterSearch:
    def test_improvement_resets_stagnation(self, tmp_path):
        """Significant improvement resets cycles_without_improvement."""
        state = {
            'asset_type': 'test',
            'best_score': 2.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 2,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            result = update_after_search(state, 2.5, {'forward_bars': 32})
        assert result['cycles_without_improvement'] == 0
        assert result['best_score'] == 2.5

    def test_no_improvement_increments_stagnation(self, tmp_path):
        """No significant improvement increments stagnation counter."""
        state = {
            'asset_type': 'test',
            'best_score': 2.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 1,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            result = update_after_search(state, 2.05, {'forward_bars': 24})
        assert result['cycles_without_improvement'] == 2

    def test_edge_triggers_expansion(self, tmp_path):
        """Best at edge triggers search space expansion."""
        state = {
            'asset_type': 'test',
            'best_score': 2.0,
            'best_params': {'forward_bars': 48},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            result = update_after_search(state, 2.5, {'forward_bars': 48})
        assert 64 in result['search_space']['forward_bars']
        assert len(result['expansion_history']) == 1

    def test_state_persisted_to_disk(self, tmp_path):
        """update_after_search saves state to disk."""
        state = {
            'asset_type': 'test',
            'best_score': 1.0,
            'best_params': {'forward_bars': 24},
            'search_space': {'forward_bars': [12, 18, 24, 32, 48]},
            'mode': 'refine',
            'cycles_without_improvement': 0,
            'expansion_history': [],
            'last_updated': '',
        }
        with mock.patch('adaptive_config.BASE_DIR', tmp_path):
            update_after_search(state, 2.0, {'forward_bars': 24})
            # Verify file was written
            path = tmp_path / 'adaptive_state_test.json'
            assert path.exists()
            with open(path) as f:
                saved = json.load(f)
            assert saved['best_score'] == 2.0


# ---------------------------------------------------------------------------
# get_search_space_for_trial
# ---------------------------------------------------------------------------

class TestGetSearchSpaceForTrial:
    def test_returns_copy(self):
        """Modifying returned space doesn't affect state."""
        state = {
            'search_space': {'forward_bars': [12, 18, 24]},
        }
        space = get_search_space_for_trial(state)
        space['forward_bars'].append(999)
        assert 999 not in state['search_space']['forward_bars']

    def test_defaults_when_missing(self):
        """Missing search_space -> returns defaults."""
        state = {}
        space = get_search_space_for_trial(state)
        assert space == DEFAULT_SEARCH_SPACE
