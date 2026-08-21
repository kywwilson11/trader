"""HMM-based regime detection for adaptive trading parameters.

Fits a Gaussian HMM to return series and classifies the current market
into bull/bear/neutral regimes — Hamilton (1989)-style Markov
regime-switching, implemented as a Gaussian HMM.
"""

import time
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)

# Cache: {symbol: (model, labels, timestamp)}
_hmm_cache: dict[str, tuple[object, dict, float]] = {}
_REFIT_INTERVAL = 86400  # 1 day

# Transition smoothing: require N consecutive bars in new regime before switching
_REGIME_PERSISTENCE = 3
_last_regime: dict[str, tuple[str, int]] = {}  # {symbol: (label, consecutive_count)}

# Failure visibility: a permanently broken detector (missing hmmlearn,
# incompatible model, degenerate fits) must not hide at DEBUG level.
_import_warned = False           # hmmlearn absence is permanent per process: warn once
_fail_counts: dict[str, int] = {}  # {kind: count} for rate-limited fit/predict warnings
_WARN_EVERY = 25


def _warn_every_nth(kind: str, msg: str) -> None:
    """WARNING on the 1st and every Nth failure, DEBUG otherwise (spam guard)."""
    n = _fail_counts.get(kind, 0) + 1
    _fail_counts[kind] = n
    if n == 1 or n % _WARN_EVERY == 0:
        logger.warning("%s (failure #%d this process)", msg, n)
    else:
        logger.debug("%s", msg)


def fit_hmm(returns: np.ndarray, n_states: int = 3):
    """Fit a Gaussian HMM to a return series.

    Args:
        returns: 1D array of percentage returns
        n_states: Number of hidden states (default 3: bear/neutral/bull)

    Returns:
        (fitted_model, state_labels) or (None, None) on failure.
        state_labels maps state_id -> {'label': str, 'mean': float, 'vol': float}
    """
    if len(returns) < 200:
        logger.debug("HMM: insufficient data (%d points, need 200)", len(returns))
        return None, None

    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        # Undeclared/broken dependency disables the whole regime layer
        # (every sizing call quietly falls back to neutral) — surface it.
        global _import_warned
        if not _import_warned:
            _import_warned = True
            logger.warning("hmmlearn unavailable — HMM regime layer disabled "
                           "(sizing stays neutral): %s", e)
        return None, None

    try:
        X = returns.reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type='full',
                            n_iter=100, random_state=42)
        model.fit(X)

        # Label states by mean return
        means = model.means_.flatten()
        vols = np.sqrt(model.covars_.flatten())
        sorted_states = np.argsort(means)

        labels_list = ['bear', 'neutral', 'bull'] if n_states == 3 else \
                      ['bear', 'bull'] if n_states == 2 else \
                      [f'state_{i}' for i in range(n_states)]

        state_labels = {}
        for rank, state_id in enumerate(sorted_states):
            state_labels[state_id] = {
                'label': labels_list[rank],
                'mean': float(means[state_id]),
                'vol': float(vols[state_id]),
            }

        return model, state_labels

    except Exception as e:
        _warn_every_nth('fit', f"HMM fit failed: {e}")
        return None, None


def get_current_regime(model, state_labels: dict,
                       recent_returns: np.ndarray) -> dict:
    """Classify the current regime using a fitted HMM.

    Args:
        model: Fitted GaussianHMM
        state_labels: State ID -> label mapping from fit_hmm()
        recent_returns: Recent return series (at least 10 points)

    Returns:
        dict with 'label', 'probabilities', 'sizing_mult'.
        'label' comes from the Viterbi MAP state path (model.predict);
        'probabilities' are the marginal posteriors for the last bar
        (model.predict_proba) — the two can disagree, so the probs are
        diagnostics, NOT the label's own distribution.
    """
    if model is None or len(recent_returns) < 10:
        return _default_regime()

    try:
        X = recent_returns.reshape(-1, 1)
        state_seq = model.predict(X)
        current_state = state_seq[-1]

        # State probabilities for current observation
        probs = model.predict_proba(X)[-1]

        info = state_labels.get(current_state, {'label': 'unknown', 'mean': 0, 'vol': 0})
        label = info['label']

        # Regime-based sizing adjustment (sizing only — base_loop consumes
        # 'sizing_mult' and 'label'; entry thresholds/stops are NOT wired)
        if label == 'bull':
            sizing_mult = 1.2
        elif label == 'bear':
            sizing_mult = 0.3
        else:  # neutral or high-vol
            # Check if this is a high-vol state
            if info['vol'] > np.median([s['vol'] for s in state_labels.values()]) * 1.5:
                sizing_mult = 0.5
                label = 'high_vol'
            else:
                sizing_mult = 1.0

        prob_dict = {}
        for sid, p in enumerate(probs):
            if sid in state_labels:
                prob_dict[state_labels[sid]['label']] = round(float(p), 3)

        return {
            'label': label,
            'probabilities': prob_dict,
            'sizing_mult': sizing_mult,
        }

    except Exception as e:
        _warn_every_nth('predict', f"HMM regime prediction failed: {e}")
        return _default_regime()


def _default_regime():
    return {
        'label': 'unknown',
        'probabilities': {},
        'sizing_mult': 1.0,
    }


def get_cached_regime(symbol: str, returns: np.ndarray) -> dict:
    """Get regime for a symbol, fitting/caching HMM as needed.

    Args:
        symbol: Trading symbol
        returns: Full return series for fitting (percentage returns)

    Returns:
        Regime dict with label, probabilities, sizing_mult
        (see get_current_regime).
    """
    now = time.time()

    if symbol in _hmm_cache:
        model, labels, ts = _hmm_cache[symbol]
        if now - ts < _REFIT_INTERVAL:
            regime = get_current_regime(model, labels, returns[-50:])
            return _smooth_regime(symbol, regime)

    # Fit new model
    model, labels = fit_hmm(returns)
    if model is not None:
        _hmm_cache[symbol] = (model, labels, now)
        regime = get_current_regime(model, labels, returns[-50:])
        # Log the RAW fitted label/probs before smoothing: on the first call
        # for a symbol _smooth_regime always forces 'unknown', which would
        # discard the fresh fit from its only INFO-level trace.
        raw_label, raw_probs = regime['label'], regime['probabilities']
        regime = _smooth_regime(symbol, regime)
        logger.info("[REGIME] %s: fitted %s (probs=%s), smoothed -> %s",
                    symbol, raw_label, raw_probs, regime['label'])
        return regime

    return _default_regime()


# KNOWN-INVERTED (c26 S3 / 02_research B06) — deliberately NOT fixed here:
# this "smoothing" (a) allows a switch after ONE observation of the NEW label
# whenever the OLD label had persisted >= _REGIME_PERSISTENCE, and (b) returns
# the NEUTRAL default instead of HOLDING the previous regime when persistence
# is short — the exact opposite of "require N consecutive observations of the
# new label before switching" (the stated intent). The layer is
# kill-recommended (research/KILL_LIST.md "HMM regime layer", owner decision
# pending) and its sizing multiplier is EXCLUDED from composition under
# strategy_config.DERISK_STACK_V2; current behavior is pinned by
# tests/test_review_b07.py::TestRegimeSmoothing, so correcting the state
# machine is an owner decision tied to the layer's fate — do not "fix" it
# in passing.
def _smooth_regime(symbol: str, regime: dict) -> dict:
    """Require N consecutive bars in new regime before switching (reduces whipsaw)."""
    label = regime['label']
    if symbol in _last_regime:
        prev_label, count = _last_regime[symbol]
        if label == prev_label:
            _last_regime[symbol] = (label, count + 1)
        else:
            _last_regime[symbol] = (label, 1)
            if count >= _REGIME_PERSISTENCE:
                # Was in previous regime long enough; allow switch
                pass
            else:
                # Not enough persistence — keep previous regime's adjustments
                return _default_regime()
    else:
        _last_regime[symbol] = (label, 1)
        return _default_regime()  # First observation, use neutral
    return regime
