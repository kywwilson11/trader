"""Adaptive search space management for hyperparameter optimization.

Detects when best parameters hit search space boundaries ("edges"),
automatically expands the search space, and cycles between explore/refine
modes to avoid boundary stagnation.

State is persisted per asset type in adaptive_state_{asset_type}.json.
"""

import json
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Default search spaces (matching hypersearch_v2.py defaults)
DEFAULT_SEARCH_SPACE = {
    'forward_bars': [12, 18, 24, 32, 48],
    'seq_len': [8, 40],           # suggest_int range, step=2
    'hidden_dim': [64, 384],      # suggest_int range, step=32
    'batch_size': [512, 1024, 2048],
    'num_layers': [1, 2],
    'n_heads': [2, 4],
    'dropout': [0.10, 0.40],
    'learning_rate': [5e-4, 3e-3],
    'weight_decay': [1e-5, 5e-4],
    'huber_delta': [0.5, 2.0],
    'trade_threshold': [0.05, 1.0],
}

# Parameters where values are discrete choices (categorical/int list)
CATEGORICAL_PARAMS = {'forward_bars', 'batch_size', 'n_heads'}

# Parameters where values are ranges [min, max] (int or float)
RANGE_PARAMS = {'seq_len', 'hidden_dim', 'num_layers',
                'dropout', 'learning_rate', 'weight_decay', 'huber_delta',
                'trade_threshold'}

# Expansion pools: new boundary values when edges are detected
# For categorical params: additional discrete values to add
# For range params: new boundary value to extend to
EXPANSION_POOLS = {
    'forward_bars': {'low': [8], 'high': [64, 96]},
    'seq_len': {'low': [4], 'high': [48]},
    'hidden_dim': {'low': [32], 'high': [512]},
    'batch_size': {'low': [256], 'high': [4096]},
    'num_layers': {'low': [], 'high': [3]},
    'n_heads': {'low': [1], 'high': [8]},
    'dropout': {'low': [0.05], 'high': [0.50]},
    'learning_rate': {'low': [2e-4], 'high': [5e-3]},
    'weight_decay': {'low': [5e-6], 'high': [1e-3]},
    'huber_delta': {'low': [0.3], 'high': [3.0]},
    'trade_threshold': {'low': [0.03], 'high': [1.5]},
}

# Hard limits: absolute boundaries that must never be exceeded
HARD_LIMITS = {
    'forward_bars': {'min': 8, 'max': 96},
    'seq_len': {'min': 4, 'max': 64},
    'hidden_dim': {'min': 32, 'max': 512},
    'batch_size': {'min': 256, 'max': 4096},
    'num_layers': {'min': 1, 'max': 4},
    'n_heads': {'min': 1, 'max': 8},
    'dropout': {'min': 0.05, 'max': 0.60},
    'learning_rate': {'min': 1e-4, 'max': 1e-2},
    'weight_decay': {'min': 1e-6, 'max': 5e-3},
    'huber_delta': {'min': 0.1, 'max': 5.0},
    'trade_threshold': {'min': 0.01, 'max': 2.0},
}

# Trial counts by mode
TRIAL_COUNTS = {
    'initial': 200,
    'refine': 70,
    'explore': 120,
    'post_explore': 80,
}

# Stagnation threshold: cycles without >5% improvement before exploring
STAGNATION_CYCLES = 3
IMPROVEMENT_THRESHOLD = 0.05  # 5%

# Edge detection: float params within this fraction of boundary = "at edge"
FLOAT_EDGE_FRACTION = 0.10


def _state_path(asset_type: str) -> Path:
    return BASE_DIR / f'adaptive_state_{asset_type}.json'


def _default_state(asset_type: str) -> dict:
    """Create a default adaptive state for a new asset type."""
    import copy
    return {
        'asset_type': asset_type,
        'best_score': 0.0,
        'best_params': {},
        'search_space': copy.deepcopy(DEFAULT_SEARCH_SPACE),
        'mode': 'refine',
        'cycles_without_improvement': 0,
        'expansion_history': [],
        'last_updated': datetime.now().isoformat(),
    }


def _migrate_search_space(space: dict) -> dict:
    """Migrate old categorical lists to ranges for seq_len/hidden_dim."""
    for param in ('seq_len', 'hidden_dim'):
        if param in space and len(space[param]) > 2:
            # Old format: [8, 12, 18, 24, 32] → new format: [8, 32]
            vals = sorted(space[param])
            space[param] = [vals[0], vals[-1]]
    return space


def load_adaptive_state(asset_type: str) -> dict:
    """Load adaptive state from disk, or create defaults if not found."""
    path = _state_path(asset_type)
    if path.exists():
        with open(path) as f:
            state = json.load(f)
        # Ensure all expected keys exist (forward compat)
        defaults = _default_state(asset_type)
        for key in defaults:
            if key not in state:
                state[key] = defaults[key]
        # Migrate old categorical lists to ranges
        if 'search_space' in state:
            state['search_space'] = _migrate_search_space(state['search_space'])
        return state
    return _default_state(asset_type)


def save_adaptive_state(state: dict) -> None:
    """Atomically save adaptive state to disk."""
    state['last_updated'] = datetime.now().isoformat()
    path = _state_path(state['asset_type'])
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, str(path))


def detect_edges(best_params: dict, search_space: dict) -> list:
    """Check if best params are at or near search space boundaries.

    Returns list of (param_name, "low"|"high") tuples indicating which
    parameters are at which boundary.
    """
    edges = []
    for param, value in best_params.items():
        if param not in search_space:
            continue
        space = search_space[param]

        if param in CATEGORICAL_PARAMS:
            # Categorical: check if value is first or last in sorted list
            sorted_vals = sorted(space)
            if len(sorted_vals) < 2:
                continue
            if value == sorted_vals[0]:
                edges.append((param, 'low'))
            elif value == sorted_vals[-1]:
                edges.append((param, 'high'))
        elif param in RANGE_PARAMS:
            # Range [min, max]: check if value is within 10% of boundary
            lo, hi = space[0], space[-1]
            range_size = hi - lo
            if range_size <= 0:
                continue
            if (value - lo) / range_size <= FLOAT_EDGE_FRACTION:
                edges.append((param, 'low'))
            elif (hi - value) / range_size <= FLOAT_EDGE_FRACTION:
                edges.append((param, 'high'))

    return edges


def expand_search_space(search_space: dict, edges: list) -> tuple:
    """Expand search space for parameters at edges.

    Returns (new_search_space, log_entries, categoricals_changed) where
    log_entries describe what was expanded and categoricals_changed is True
    if any categorical distribution was modified (requires Optuna study DB reset).
    """
    import copy
    new_space = copy.deepcopy(search_space)
    log_entries = []
    categoricals_changed = False

    for param, direction in edges:
        if param not in EXPANSION_POOLS:
            continue

        pool = EXPANSION_POOLS[param]
        limits = HARD_LIMITS.get(param, {})
        expansion_values = pool.get(direction, [])

        if not expansion_values:
            continue

        if param in CATEGORICAL_PARAMS:
            current = set(new_space[param])
            for val in expansion_values:
                # Respect hard limits
                if 'min' in limits and val < limits['min']:
                    continue
                if 'max' in limits and val > limits['max']:
                    continue
                if val not in current:
                    current.add(val)
                    categoricals_changed = True
                    log_entries.append(
                        f"{param}: added {val} ({direction} expansion)")
            new_space[param] = sorted(current)
        elif param in RANGE_PARAMS:
            lo, hi = new_space[param][0], new_space[param][-1]
            new_val = expansion_values[0]
            if direction == 'low':
                new_lo = max(new_val, limits.get('min', new_val))
                if new_lo < lo:
                    log_entries.append(
                        f"{param}: low bound {lo} -> {new_lo}")
                    lo = new_lo
            elif direction == 'high':
                new_hi = min(new_val, limits.get('max', new_val))
                if new_hi > hi:
                    log_entries.append(
                        f"{param}: high bound {hi} -> {new_hi}")
                    hi = new_hi
            new_space[param] = [lo, hi]

    return new_space, log_entries, categoricals_changed


def decide_mode(state: dict, new_best_score: float) -> str:
    """Decide whether to explore or refine.

    Returns "explore" if:
      - Any edge detected in best params
      - Stagnation: N+ cycles without meaningful improvement
    Returns "refine" otherwise.
    """
    # If we just explored, go back to refine
    if state.get('mode') == 'explore':
        return 'refine'

    # Check for edges
    if state.get('best_params') and state.get('search_space'):
        edges = detect_edges(state['best_params'], state['search_space'])
        if edges:
            return 'explore'

    # Check for stagnation
    cycles = state.get('cycles_without_improvement', 0)
    if cycles >= STAGNATION_CYCLES:
        return 'explore'

    return 'refine'


def get_trial_count(mode: str, is_initial: bool = False) -> int:
    """Get the number of trials for a given mode."""
    if is_initial:
        return TRIAL_COUNTS['initial']
    return TRIAL_COUNTS.get(mode, TRIAL_COUNTS['refine'])


def get_search_space_for_trial(state: dict) -> dict:
    """Return the current search space bounds from state.

    This is what hypersearch_v2.py uses to configure Optuna trial suggestions.
    """
    import copy
    return copy.deepcopy(state.get('search_space', DEFAULT_SEARCH_SPACE))


def update_after_search(state: dict, new_best_score: float,
                        new_best_params: dict,
                        study_db_path: str = None) -> dict:
    """Update adaptive state after a hypersearch completes.

    Handles:
      - Score tracking and improvement detection
      - Edge detection and search space expansion
      - Mode transitions
      - Deleting stale Optuna study DB when categorical distributions change
    """
    old_score = state.get('best_score', 0.0)

    # Track improvement
    if old_score > 0 and new_best_score > old_score * (1 + IMPROVEMENT_THRESHOLD):
        state['cycles_without_improvement'] = 0
    else:
        state['cycles_without_improvement'] = state.get(
            'cycles_without_improvement', 0) + 1

    # Update best if improved
    if new_best_score > old_score:
        state['best_score'] = new_best_score
        state['best_params'] = new_best_params

    # Detect edges and expand if needed
    edges = detect_edges(state['best_params'], state['search_space'])
    if edges:
        new_space, log_entries, categoricals_changed = expand_search_space(
            state['search_space'], edges)
        state['search_space'] = new_space
        if log_entries:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'edges': [(p, d) for p, d in edges],
                'expansions': log_entries,
            }
            state['expansion_history'].append(entry)

        # Optuna doesn't allow changing categorical distributions in an
        # existing study.  Delete the study DB so the next run starts fresh.
        if categoricals_changed and study_db_path and os.path.exists(study_db_path):
            os.remove(study_db_path)
            print(f"[ADAPTIVE] Deleted {study_db_path} "
                  f"(categorical search space expanded — incompatible with old study)")

    # Decide next mode
    state['mode'] = decide_mode(state, new_best_score)

    save_adaptive_state(state)
    return state


def get_max_forward_bars(asset_type: str) -> int:
    """Get the maximum forward_bars value from adaptive state.

    Used by harvest scripts to know which Target_Return_N columns to generate.
    """
    state = load_adaptive_state(asset_type)
    fb_space = state['search_space'].get('forward_bars',
                                          DEFAULT_SEARCH_SPACE['forward_bars'])
    return max(fb_space)


def get_forward_bars_list(asset_type: str) -> list:
    """Get the full forward_bars list from adaptive state.

    Used by harvest scripts to generate all needed Target_Return_N columns.
    """
    state = load_adaptive_state(asset_type)
    fb_space = state['search_space'].get('forward_bars',
                                          DEFAULT_SEARCH_SPACE['forward_bars'])
    return sorted(fb_space)
