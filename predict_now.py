"""ML prediction engine — RegressionLSTM loading, JIT tracing, live predictions.

Loads a single RegressionLSTM model from disk, optionally JIT-traces it
for faster inference, and provides get_live_prediction() which fetches recent
bars, computes features, and returns a predicted return percentage.

Bar-fetching and ATR logic live in market_data.py.
"""

# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import joblib
import os
import torch

# Tiny-LSTM CPU inference is fastest with 1-2 threads; the default 6-thread
# spawn just steals cores from training during the weekly retrain window.
if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '2')))

from model_v2 import RegressionLSTM
from indicators import compute_features, compute_stock_features
from market_data import (
    fetch_bars_alpaca, fetch_bars_yfinance,
    fetch_stock_bars_alpaca,
)

# Lazy-load with retry: try once per cycle, stop spamming after first failure log
_get_live_sentiment = None
_sentiment_import_failed = False

# LightGBM ensemble + q10 tail models, loaded lazily PER PREFIX — the
# challenger shadow stack must never pair its LSTM with the champion's
# boosters (single-slot globals did exactly that)
_lgb_models: dict[str, object | None] = {}
_q10_models: dict[str, tuple[object, float] | None] = {}

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _prefixed_paths(prefix):
    """Return dict of file paths for a given model prefix (e.g. 'stock')."""
    p = f'{prefix}_' if prefix else ''
    return {
        'model': f'{p}model_v2.pth',
        'config': f'{p}config_v2.pkl',
        'scaler': f'{p}scaler_v2.pkl',
        'features': f'{p}feature_cols_v2.pkl',
    }


# --- MODEL LOADING ---

def load_model(inference_device=None, prefix=''):
    """Load a RegressionLSTM model from disk.

    Args:
        inference_device: Override device for inference (e.g. 'cpu' for GPU fallback)
        prefix: File prefix (e.g. 'stock' -> stock_model_v2.pth)

    Returns:
        (model, scaler_X, config, seq_len, feature_cols)
    """
    dev = torch.device(inference_device) if inference_device else device
    paths = _prefixed_paths(prefix)

    config = joblib.load(paths['config'])
    scaler_X = joblib.load(paths['scaler'])
    feature_cols = joblib.load(paths['features'])

    model = RegressionLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        n_heads=config.get('n_heads', 4),
    ).to(dev)
    model.load_state_dict(torch.load(paths['model'], map_location=dev, weights_only=True))
    model.eval()

    try:
        dummy = torch.randn(1, config['seq_len'], config['input_dim']).to(dev)
        model = torch.jit.trace(model, dummy, check_trace=False)
        print(f"  [JIT] Model traced successfully")
    except Exception as e:
        print(f"  [JIT] Trace failed: {e}, using eager mode")

    return model, scaler_X, config, config['seq_len'], feature_cols


def load_models(inference_device=None, prefix=''):
    """Load a regression model, printing summary info.

    Args:
        inference_device: Override device for inference
        prefix: File prefix (e.g. 'stock' -> stock_model_v2.pth)

    Returns:
        (model, config, scaler_X, feature_cols)

    Raises:
        FileNotFoundError: If model files don't exist
    """
    pfx_label = f"{prefix} " if prefix else ""
    model, scaler_X, config, seq_len, feature_cols = load_model(inference_device, prefix)
    th = config.get('trade_threshold', 0.15)
    fb = config.get('forward_bars', 4)
    print(f"{pfx_label}Model loaded: "
          f"seq={seq_len}, threshold={th:.2f}, fb={fb}, "
          f"heads={config.get('n_heads', 4)}")
    return model, config, scaler_X, feature_cols


# --- LIVE PREDICTION ---

def get_live_prediction(symbol, model, scaler_X, config, feature_cols,
                        api=None, inference_device=None,
                        asset_type='crypto', spy_close=None, btc_close=None,
                        return_snapshot=False):
    """Get predicted return for a symbol.

    Args:
        symbol: Ticker symbol (Alpaca format 'BTC/USD' if api provided, else yfinance 'BTC-USD')
        model: RegressionLSTM model (or JIT-traced variant)
        scaler_X: Feature scaler
        config: Model config dict (must contain 'seq_len', 'trade_threshold')
        feature_cols: List of feature column names
        api: Alpaca API object — if provided, uses Alpaca bars instead of yfinance
        inference_device: Override device for inference (e.g. 'cpu')
        asset_type: 'crypto' or 'stock' — determines feature computation and data source
        spy_close: SPY close Series for stock relative strength (optional)
        btc_close: BTC/USD close Series for crypto cross-asset features (optional)
        return_snapshot: If True, return (predicted_return, indicator_snapshot_dict)

    Returns:
        float predicted_return, or None on error
        If return_snapshot=True: (predicted_return, snapshot_dict) or (None, None)
    """
    dev = torch.device(inference_device) if inference_device else device
    seq_len = config['seq_len']

    print(f"\n--- ANALYZING {symbol} ---")

    # --- Fetch bars ---
    if asset_type == 'stock':
        if api is not None:
            df = fetch_stock_bars_alpaca(api, symbol)
        else:
            df = fetch_bars_yfinance(symbol)
    else:
        if api is not None:
            alpaca_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            df = fetch_bars_alpaca(api, alpaca_sym)
        else:
            yf_sym = symbol.replace('/', '-') if '/' in symbol else symbol
            df = fetch_bars_yfinance(yf_sym)

    _none_ret = (None, None) if return_snapshot else None

    if df is None or df.empty:
        print("Error: No data found for symbol.")
        return _none_ret

    # --- Compute technical features ---
    if asset_type == 'stock':
        df = compute_stock_features(df, spy_close=spy_close)
    else:
        df = compute_features(df, btc_close=btc_close)
    df = df.dropna()

    if len(df) < seq_len:
        print(f"  Not enough data for sequence (need {seq_len}, have {len(df)})")
        return _none_ret

    # Inject live sentiment if the model was trained with it
    if 'Daily_Sentiment' in feature_cols and 'Daily_Sentiment' not in df.columns:
        global _get_live_sentiment, _sentiment_import_failed
        # Retry import each cycle until it succeeds; only log failure once
        if _get_live_sentiment is None:
            try:
                from sentiment_history import get_live_daily_sentiment
                _get_live_sentiment = get_live_daily_sentiment
                if _sentiment_import_failed:
                    print("  [SENTIMENT] sentiment_history recovered")
            except Exception as e:
                if not _sentiment_import_failed:
                    print(f"  [SENTIMENT] sentiment_history unavailable: {e}")
                    _sentiment_import_failed = True
        if _get_live_sentiment is not None:
            try:
                df['Daily_Sentiment'] = _get_live_sentiment(symbol, asset_type)
            except Exception:
                df['Daily_Sentiment'] = 0.0
        else:
            df['Daily_Sentiment'] = 0.0

    # Inject live funding features if the model was trained with them
    # (crypto only; same constant-across-window approach as sentiment)
    if asset_type == 'crypto' and 'Funding_Rate_Ann' in feature_cols \
            and 'Funding_Rate_Ann' not in df.columns:
        ff = None
        try:
            from funding import live_funding_features
            alp_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            ff = live_funding_features(alp_sym)
        except Exception:
            pass
        for col in ('Funding_Rate_Ann', 'Funding_Z', 'Funding_Chg_24h'):
            df[col] = (ff or {}).get(col, 0.0)

    # Inject live open-interest features (crypto only; OKX vs local
    # rolling history — 0.0 until ~1d/7d of history accumulates)
    if asset_type == 'crypto' and 'OI_Chg_24h' in feature_cols \
            and 'OI_Chg_24h' not in df.columns:
        of = None
        try:
            from oi_archive import live_oi_features
            alp_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            of = live_oi_features(alp_sym)
        except Exception:
            pass
        for col in ('OI_Chg_24h', 'OI_Z'):
            df[col] = (of or {}).get(col, 0.0)

    # Inject live top-trader long/short positioning z (crypto only;
    # OKX serves the full 30d window in one call — no cold start)
    if asset_type == 'crypto' and 'TT_LS_Z' in feature_cols \
            and 'TT_LS_Z' not in df.columns:
        lf = None
        try:
            from oi_archive import live_ls_features
            alp_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            lf = live_ls_features(alp_sym)
        except Exception:
            pass
        df['TT_LS_Z'] = (lf or {}).get('TT_LS_Z', 0.0)

    try:
        current_features = df[feature_cols].values
    except KeyError as e:
        print(f"  Feature mismatch: {e}")
        return _none_ret

    # --- Scale and build input tensor ---
    current_features_scaled = scaler_X.transform(current_features)
    sequence = current_features_scaled[-seq_len:]
    sequence = sequence.reshape(1, seq_len, -1)
    tensor_input = torch.tensor(sequence, dtype=torch.float32).to(dev)

    # --- Run inference ---
    with torch.inference_mode():
        output = model(tensor_input)

    lstm_pred = float(output.cpu().item())

    # LightGBM ensemble: combine LSTM and LGB predictions
    pfx = config.get('prefix', '')
    if pfx not in _lgb_models:
        try:
            from model_lgb import load_lgb_model
            _lgb_models[pfx] = load_lgb_model(prefix=pfx)
        except Exception:
            _lgb_models[pfx] = None
    lgb_model = _lgb_models[pfx]

    predicted_return = lstm_pred
    flat = None
    if lgb_model is not None:
        try:
            from model_lgb import flatten_sequence, predict_lgb, ensemble_predict
            flat, _ = flatten_sequence(sequence.reshape(seq_len, -1), feature_cols)
            lgb_pred = predict_lgb(lgb_model, flat)
            predicted_return = ensemble_predict(lstm_pred, lgb_pred)
        except Exception:
            pass  # Fall back to LSTM-only

    # q10 tail prediction (left-tail risk of THIS state; entry veto input)
    if pfx not in _q10_models:
        try:
            import json as _json
            import lightgbm as _lgb
            p = f'{pfx}_' if pfx else ''
            booster = _lgb.Booster(model_file=f'{p}lgb_q10.txt')
            with open(f'{p}lgb_q10_meta.json') as f:
                floor = float(_json.load(f)['floor'])
            _q10_models[pfx] = (booster, floor)
        except Exception:
            _q10_models[pfx] = None
    q10_pred = None
    _q10_floor = None
    if _q10_models[pfx] is not None:
        try:
            q10_booster, _q10_floor = _q10_models[pfx]
            if flat is None:
                from model_lgb import flatten_sequence
                flat, _ = flatten_sequence(sequence.reshape(seq_len, -1),
                                           feature_cols)
            q10_pred = float(q10_booster.predict(flat.reshape(1, -1))[0])
        except Exception:
            q10_pred = None

    trade_threshold = config.get('trade_threshold', 0.15)

    price = df['Close'].iloc[-1]
    print(f"Current Price:   ${price:.2f}")
    print(f"Predicted Return: {predicted_return:+.4f}% (threshold={trade_threshold:.2f})")

    if predicted_return > trade_threshold:
        print("Recommendation:  [BUY]")
    elif predicted_return < -trade_threshold:
        print("Recommendation:  [SELL/AVOID]")
    else:
        print("Recommendation:  [HOLD/WEAK]")

    if return_snapshot:
        # Build snapshot of latest indicator values (all available, not just model features)
        last_row = df.iloc[-1]
        # Use previous completed bar for volume (current bar is incomplete)
        prev_row = df.iloc[-2] if len(df) >= 2 else last_row
        _SNAPSHOT_COLS = [
            'Close', 'RSI', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR',
            'SMA_20', 'Price_SMA20_Ratio', 'BBL_20_2.0', 'BBU_20_2.0',
            'BBP_20_2.0', 'BBB_20_2.0',
            'ROC', 'Return_4h', 'Return_12h', 'Volatility_12h',
            'Daily_Sentiment', 'Hurst',  # Hurst was missing — the loop's
            # mean-reversion filter read snapshot['Hurst'] and always got
            # None, so that gate never fired; meta features need it too
            # Cross-asset (may not exist for all asset types)
            'BTC_Return_1h', 'BTC_SMA_Ratio', 'BTC_RSI',
            'RS_vs_SPY', 'Price_VWAP_Ratio', 'ATR_Pct',
        ]
        snapshot = {}
        for col in _SNAPSHOT_COLS:
            if col in last_row.index:
                val = last_row[col]
                if val is not None and val == val:
                    snapshot[col] = float(val)
        # Tail-risk model outputs for the loop's q10 entry veto
        if q10_pred is not None and _q10_floor is not None:
            snapshot['Q10'] = q10_pred
            snapshot['Q10_Floor'] = _q10_floor
        # Volume from last completed bar — only include if real data exists
        # (Alpaca crypto bars often report zero volume)
        prev_vol = prev_row.get('Volume', 0) if 'Volume' in prev_row.index else 0
        if prev_vol and prev_vol > 0 and 'Volume_Ratio' in prev_row.index:
            val = prev_row['Volume_Ratio']
            if val is not None and val == val:
                snapshot['Volume_Ratio'] = float(val)
        return predicted_return, snapshot

    return predicted_return


if __name__ == "__main__":
    print(f"Using device: {device}")

    try:
        model, config, scaler_X, feature_cols = load_models()
    except FileNotFoundError:
        print("Error: Model files not found. Run hypersearch_v2.py first.")
        exit(1)

    symbols = [
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
        'LINK-USD',
    ]
    for sym in symbols:
        get_live_prediction(sym, model, scaler_X, config, feature_cols)
