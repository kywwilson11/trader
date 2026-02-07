"""
Continuous improvement orchestrator — daily retrain + promotion gate.

Runs as a long-lived process:
1. Harvest fresh data
2. Run bear hypersearch (250 trials)
3. Run bull hypersearch (250 trials)
4. Compare new models against current deployed models
5. Promote only if new model beats current (validation accuracy)
6. Version models in models/ directory, keep last 5

Usage:
    python evolve.py              # Full continuous loop
    python evolve.py --dry-run    # One cycle, no actual training
    python evolve.py --once       # One full cycle then exit
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import gc
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from model import CryptoLSTM
from hw_monitor import wait_for_cool_gpu, get_gpu_temp, get_ram_usage

MODELS_DIR = 'models'
SCORES_FILE = os.path.join(MODELS_DIR, 'scores.json')
MAX_VERSIONS = 5
DAILY_INTERVAL_HOURS = 24

# Paths for active models (crypto_loop.py reads these)
ACTIVE_BEAR = 'bear_model.pth'
ACTIVE_BEAR_CFG = 'bear_config.pkl'
ACTIVE_BULL = 'bull_model.pth'
ACTIVE_BULL_CFG = 'bull_config.pkl'
SCALER_PATH = 'scaler_X.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'


def parse_args():
    parser = argparse.ArgumentParser(description='Continuous model improvement')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate one cycle without training')
    parser.add_argument('--once', action='store_true',
                        help='Run one full cycle then exit')
    parser.add_argument('--trials', type=int, default=250,
                        help='Trials per hypersearch (default: 250)')
    return parser.parse_args()


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return {'bear': {}, 'bull': {}, 'current_bear': None, 'current_bull': None}


def save_scores(scores):
    with open(SCORES_FILE, 'w') as f:
        json.dump(scores, f, indent=2)


def get_next_version(scores, target):
    """Get the next version number for a target (bear/bull)."""
    versions = scores.get(target, {})
    if not versions:
        return 1
    return max(int(v) for v in versions.keys()) + 1


def harvest_data():
    """Run harvest_data.py to get fresh training data."""
    print("\n[EVOLVE] Harvesting fresh data...")
    result = subprocess.run(
        [sys.executable, '-u', 'harvest_data.py'],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[EVOLVE] WARNING: harvest_data.py exited with code {result.returncode}")
        return False
    return True


def run_hypersearch(target, trials):
    """Run hypersearch_dual.py for a target (bear/bull)."""
    print(f"\n[EVOLVE] Running {target} hypersearch ({trials} trials)...")
    result = subprocess.run(
        [sys.executable, '-u', 'hypersearch_dual.py', '--target', target, '--trials', str(trials)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[EVOLVE] WARNING: {target} hypersearch exited with code {result.returncode}")
        return False
    return True


def evaluate_model(model_path, config_path, target):
    """Evaluate a model on the validation set. Returns target class accuracy.
    target: 'bear' (class 0) or 'bull' (class 2)
    """
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        return 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = joblib.load(config_path)
    scaler_X = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    model = CryptoLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=config.get('num_classes', 3),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load data for validation
    df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
    exclude_cols = ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    tickers = df['Ticker'].unique()
    all_scaled_list, all_returns_list = [], []
    ticker_boundaries = {}
    offset = 0
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        scaled = scaler_X.transform(tdf[feat_cols].values).astype(np.float32)
        returns = tdf['Target_Return'].values.astype(np.float32)
        all_scaled_list.append(scaled)
        all_returns_list.append(returns)
        ticker_boundaries[ticker] = (offset, offset + len(scaled))
        offset += len(scaled)

    all_scaled = np.vstack(all_scaled_list)
    all_returns = np.concatenate(all_returns_list)

    bull_threshold = config.get('bull_threshold', 0.15)
    bear_threshold = -bull_threshold
    seq_len = config['seq_len']

    classes = np.ones(len(all_returns), dtype=np.int64)
    classes[all_returns > bull_threshold] = 2
    classes[all_returns < bear_threshold] = 0

    valid_indices = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        for i in range(start + seq_len, end):
            valid_indices.append(i)

    # Use last 20% as validation
    split = int(len(valid_indices) * 0.8)
    val_idx = valid_indices[split:]

    # Import SequenceDataset from hypersearch_dual
    from hypersearch_dual import SequenceDataset
    val_ds = SequenceDataset(val_idx, all_scaled, classes, seq_len)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    target_class = 0 if target == 'bear' else 2
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            _, p = torch.max(model(X_b), 1)
            all_preds.extend(p.cpu().numpy())
            all_labels.extend(y_b.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mask = all_labels == target_class
    if mask.sum() == 0:
        return 0.0

    accuracy = float((all_preds[mask] == target_class).mean())

    del model, val_ds, val_loader, all_scaled, all_returns, df
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy


def promote_model(target, version, score, scores):
    """Copy a new model to the active path and update scores."""
    src_model = f'{target}_model.pth'
    src_config = f'{target}_config.pkl'

    if not os.path.exists(src_model):
        print(f"[EVOLVE] No {target} model to promote")
        return False

    # Save versioned copy
    dst_model = os.path.join(MODELS_DIR, f'{target}_v{version}.pth')
    dst_config = os.path.join(MODELS_DIR, f'{target}_v{version}_config.pkl')
    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_config, dst_config)

    # Update scores
    scores[target][str(version)] = {
        'score': score,
        'timestamp': datetime.now().isoformat(),
    }
    scores[f'current_{target}'] = version
    save_scores(scores)

    print(f"[EVOLVE] Promoted {target}_v{version} (score={score:.4f})")
    return True


def cleanup_old_versions(scores, target):
    """Keep only the last MAX_VERSIONS versions."""
    versions = scores.get(target, {})
    if len(versions) <= MAX_VERSIONS:
        return

    sorted_versions = sorted(versions.keys(), key=int)
    to_remove = sorted_versions[:-MAX_VERSIONS]

    for v in to_remove:
        model_path = os.path.join(MODELS_DIR, f'{target}_v{v}.pth')
        config_path = os.path.join(MODELS_DIR, f'{target}_v{v}_config.pkl')
        for p in [model_path, config_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"[EVOLVE] Cleaned up {p}")
        del versions[v]

    save_scores(scores)


def run_cycle(trials, dry_run=False):
    """Run one full evolution cycle."""
    print(f"\n{'='*70}")
    print(f"[EVOLVE] CYCLE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    ensure_models_dir()
    scores = load_scores()

    # 1. Check GPU temp
    temp = wait_for_cool_gpu(70)
    if temp is not None:
        print(f"[EVOLVE] GPU temp: {temp:.0f}C — OK to proceed")

    used, total = get_ram_usage()
    if used is not None:
        print(f"[EVOLVE] RAM: {used:.0f}/{total:.0f} MB")

    if dry_run:
        print("[EVOLVE] DRY RUN — skipping data harvest and training")
        # Still evaluate current models
        for target in ['bear', 'bull']:
            model_path = f'{target}_model.pth'
            config_path = f'{target}_config.pkl'
            if os.path.exists(model_path):
                score = evaluate_model(model_path, config_path, target)
                print(f"[EVOLVE] Current {target} model score: {score:.4f}")
            else:
                print(f"[EVOLVE] No {target} model found")
        return

    # 2. Harvest fresh data
    harvest_data()

    # 3. Run bear hypersearch
    # Get current bear score for comparison
    current_bear_score = 0.0
    if os.path.exists(ACTIVE_BEAR):
        current_bear_score = evaluate_model(ACTIVE_BEAR, ACTIVE_BEAR_CFG, 'bear')
        print(f"[EVOLVE] Current bear score: {current_bear_score:.4f}")

    temp = wait_for_cool_gpu(70)
    gc.collect()
    torch.cuda.empty_cache()

    run_hypersearch('bear', trials)

    # Evaluate new bear model
    new_bear_score = evaluate_model(ACTIVE_BEAR, ACTIVE_BEAR_CFG, 'bear')
    print(f"[EVOLVE] New bear score: {new_bear_score:.4f} (was: {current_bear_score:.4f})")

    bear_version = get_next_version(scores, 'bear')
    if new_bear_score > current_bear_score:
        promote_model('bear', bear_version, new_bear_score, scores)
    else:
        # Still save the version but don't make it active
        print(f"[EVOLVE] New bear model not better, keeping current")
        # Save versioned copy anyway for analysis
        if os.path.exists('bear_model.pth'):
            dst = os.path.join(MODELS_DIR, f'bear_v{bear_version}.pth')
            shutil.copy2('bear_model.pth', dst)
            scores['bear'][str(bear_version)] = {
                'score': new_bear_score,
                'timestamp': datetime.now().isoformat(),
                'promoted': False,
            }
            save_scores(scores)

    cleanup_old_versions(scores, 'bear')

    # 4. Cool down between searches
    gc.collect()
    torch.cuda.empty_cache()
    wait_for_cool_gpu(65)

    # 5. Run bull hypersearch
    current_bull_score = 0.0
    if os.path.exists(ACTIVE_BULL):
        current_bull_score = evaluate_model(ACTIVE_BULL, ACTIVE_BULL_CFG, 'bull')
        print(f"[EVOLVE] Current bull score: {current_bull_score:.4f}")

    run_hypersearch('bull', trials)

    new_bull_score = evaluate_model(ACTIVE_BULL, ACTIVE_BULL_CFG, 'bull')
    print(f"[EVOLVE] New bull score: {new_bull_score:.4f} (was: {current_bull_score:.4f})")

    bull_version = get_next_version(scores, 'bull')
    if new_bull_score > current_bull_score:
        promote_model('bull', bull_version, new_bull_score, scores)
    else:
        print(f"[EVOLVE] New bull model not better, keeping current")
        if os.path.exists('bull_model.pth'):
            dst = os.path.join(MODELS_DIR, f'bull_v{bull_version}.pth')
            shutil.copy2('bull_model.pth', dst)
            scores['bull'][str(bull_version)] = {
                'score': new_bull_score,
                'timestamp': datetime.now().isoformat(),
                'promoted': False,
            }
            save_scores(scores)

    cleanup_old_versions(scores, 'bull')

    print(f"\n[EVOLVE] CYCLE COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[EVOLVE] Bear: v{scores.get('current_bear', '?')} | Bull: v{scores.get('current_bull', '?')}")


def main():
    args = parse_args()

    if args.dry_run:
        run_cycle(args.trials, dry_run=True)
        return

    if args.once:
        run_cycle(args.trials)
        return

    # Continuous loop
    while True:
        try:
            run_cycle(args.trials)
        except Exception as e:
            print(f"\n[EVOLVE] ERROR in cycle: {e}")
            import traceback
            traceback.print_exc()

        next_run = datetime.now() + timedelta(hours=DAILY_INTERVAL_HOURS)
        print(f"\n[EVOLVE] Next cycle at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[EVOLVE] Sleeping {DAILY_INTERVAL_HOURS}h...")
        time.sleep(DAILY_INTERVAL_HOURS * 3600)


if __name__ == '__main__':
    main()
