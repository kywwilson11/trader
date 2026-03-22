"""HMM-based regime detection for adaptive trading parameters.

Fits a Gaussian HMM to return series and classifies the current market
into bull/bear/neutral regimes. Based on Sargent/Sims (2011 Nobel) regime
switching models.
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
            label = labels_list[rank] if rank < len(labels_list) else f'state_{state_id}'
            state_labels[state_id] = {
                'label': label,
                'mean': float(means[state_id]),
                'vol': float(vols[state_id]) if state_id < len(vols) else 0.0,
            }

        return model, state_labels

    except Exception as e:
        logger.debug("HMM fit failed: %s", e)
        return None, None


def get_current_regime(model, state_labels: dict,
                       recent_returns: np.ndarray) -> dict:
    """Classify the current regime using a fitted HMM.

    Args:
        model: Fitted GaussianHMM
        state_labels: State ID -> label mapping from fit_hmm()
        recent_returns: Recent return series (at least 10 points)

    Returns:
        dict with 'label', 'probabilities', 'sizing_mult', 'threshold_mult', 'stop_mult'
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

        # Regime-based parameter adjustments
        if label == 'bull':
            sizing_mult = 1.2
            threshold_mult = 0.8   # lower threshold = more trades
            stop_mult = 1.0
        elif label == 'bear':
            sizing_mult = 0.3
            threshold_mult = 1.5   # higher threshold = fewer trades
            stop_mult = 0.8        # tighter stops
        else:  # neutral or high-vol
            # Check if this is a high-vol state
            if info['vol'] > np.median([s['vol'] for s in state_labels.values()]) * 1.5:
                sizing_mult = 0.5
                threshold_mult = 1.2
                stop_mult = 1.3    # wider stops for high vol
                label = 'high_vol'
            else:
                sizing_mult = 1.0
                threshold_mult = 1.0
                stop_mult = 1.0

        prob_dict = {}
        for sid, p in enumerate(probs):
            if sid in state_labels:
                prob_dict[state_labels[sid]['label']] = round(float(p), 3)

        return {
            'label': label,
            'probabilities': prob_dict,
            'sizing_mult': sizing_mult,
            'threshold_mult': threshold_mult,
            'stop_mult': stop_mult,
        }

    except Exception as e:
        logger.debug("HMM regime prediction failed: %s", e)
        return _default_regime()


def _default_regime():
    return {
        'label': 'unknown',
        'probabilities': {},
        'sizing_mult': 1.0,
        'threshold_mult': 1.0,
        'stop_mult': 1.0,
    }


def get_cached_regime(symbol: str, returns: np.ndarray) -> dict:
    """Get regime for a symbol, fitting/caching HMM as needed.

    Args:
        symbol: Trading symbol
        returns: Full return series for fitting (percentage returns)

    Returns:
        Regime dict with label, sizing_mult, threshold_mult, stop_mult.
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
        regime = _smooth_regime(symbol, regime)
        logger.info("[REGIME] %s: %s (probs=%s)", symbol, regime['label'],
                    regime['probabilities'])
        return regime

    return _default_regime()


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
