"""LightGBM ensemble model for stacking with LSTM predictions.

Flattens sequential features into a single row for tree-based learning.
CPU-only, fast on ARM64, sub-millisecond inference.
"""

import os
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def flatten_sequence(sequence: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Flatten a (seq_len, n_features) sequence into a single 1D vector.

    Creates features like RSI_t, RSI_t-1, RSI_t-2, ... for each feature.

    Args:
        sequence: Shape (seq_len, n_features) array
        feature_names: List of feature names

    Returns:
        (flat_array, flat_feature_names)
    """
    seq_len, n_features = sequence.shape
    flat = sequence.flatten()  # row-major: all t=0, then all t=1, etc.

    flat_names = []
    for t in range(seq_len):
        lag = seq_len - 1 - t  # t=0 is oldest, so lag = seq_len-1
        for feat in feature_names:
            if lag == 0:
                flat_names.append(feat)
            else:
                flat_names.append(f"{feat}_lag{lag}")

    return flat, flat_names


def train_lgb(X_flat: np.ndarray, y: np.ndarray,
              X_val_flat: np.ndarray = None, y_val: np.ndarray = None,
              params: dict = None,
              sample_weight: np.ndarray = None,
              sample_weight_val: np.ndarray = None) -> object:
    """Train a LightGBM model on flattened features.

    Args:
        X_flat: Training features, shape (n_samples, n_flat_features)
        y: Target returns, shape (n_samples,)
        X_val_flat: Validation features (optional)
        y_val: Validation targets (optional)
        params: LightGBM parameters (optional override)
        sample_weight: per-row training weights (optional). LightGBM scales each
            row's gradient AND hessian by its weight, so a mean-1 average-
            uniqueness vector (sample_weights.fold_train_weights) de-biases
            overlapping-label over-counting while keeping the tuned leaf
            regularizer calibrated (total mass ~N). None == uniform (unchanged).
        sample_weight_val: per-row weights for the validation set (optional);
            pass the SAME mean-1 vector kind so early-stopping sees weighted loss.

    Returns:
        Trained lightgbm.Booster
    """
    import lightgbm as lgb

    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': 4,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_flat, label=y, weight=sample_weight)
    valid_sets = [train_data]
    if X_val_flat is not None and y_val is not None:
        val_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data,
                               weight=sample_weight_val)
        valid_sets.append(val_data)

    callbacks = [lgb.log_evaluation(period=50)]
    if X_val_flat is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=20))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    logger.info("[LGB] Trained with %d features, %d rounds",
                X_flat.shape[1], model.current_iteration())
    return model


def save_lgb_model(model, prefix: str = ''):
    """Save LightGBM model to disk atomically (tmp + os.replace).

    A bot's lazy load_lgb_model — triggered by a manifest hot-reload —
    must never read a half-written booster file (2026-07 review P1).
    """
    pfx = f'{prefix}_' if prefix else ''
    path = os.path.join(_MODEL_DIR, f'{pfx}lgb_model.txt')
    tmp = path + '.tmp'
    model.save_model(tmp)
    os.replace(tmp, path)
    logger.info("[LGB] Saved to %s", path)


def load_lgb_model(prefix: str = ''):
    """Load LightGBM model from disk.

    Returns:
        lightgbm.Booster or None if file doesn't exist.
    """
    import lightgbm as lgb
    pfx = f'{prefix}_' if prefix else ''
    path = os.path.join(_MODEL_DIR, f'{pfx}lgb_model.txt')

    if not os.path.exists(path):
        return None

    model = lgb.Booster(model_file=path)
    logger.info("[LGB] Loaded from %s (%d trees)", path, model.num_trees())
    return model


def predict_lgb(model, flat_features: np.ndarray) -> float:
    """Get prediction from LightGBM model.

    Args:
        model: Trained lightgbm.Booster
        flat_features: 1D array of flattened features

    Returns:
        Predicted return (float)
    """
    pred = model.predict(flat_features.reshape(1, -1))
    return float(pred[0])


def ensemble_predict(lstm_pred: float, lgb_pred: float | None,
                     lstm_weight: float = 0.6) -> float:
    """Combine LSTM and LightGBM predictions.

    Args:
        lstm_pred: LSTM predicted return
        lgb_pred: LightGBM predicted return (None if model unavailable)
        lstm_weight: Weight for LSTM (default 0.6, LGB gets 0.4)

    Returns:
        Combined prediction
    """
    if lgb_pred is None:
        return lstm_pred

    lgb_weight = 1.0 - lstm_weight
    return lstm_pred * lstm_weight + lgb_pred * lgb_weight
