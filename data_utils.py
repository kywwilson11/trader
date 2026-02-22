"""Centralized data I/O — Parquet-first loading with CSV fallback.

Provides atomic writes, incremental append, migration, and validation
for training data used by harvest scripts and hypersearch.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent

# Mapping from prefix to file stem
_FILE_STEMS = {
    'crypto': 'training_data',
    'stock': 'stock_training_data',
}


def _stem(prefix: str) -> str:
    return _FILE_STEMS.get(prefix, f'{prefix}_training_data')


def get_data_path(prefix: str) -> Path:
    """Return the best available data file path (.parquet preferred, .csv fallback)."""
    stem = _stem(prefix)
    parquet = _BASE_DIR / f'{stem}.parquet'
    if parquet.exists():
        return parquet
    return _BASE_DIR / f'{stem}.csv'


def load_training_data(prefix: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load training data, trying Parquet first then CSV.

    Args:
        prefix: 'crypto' or 'stock'
        columns: optional column subset (Parquet only — CSV loads all then filters)

    Returns:
        DataFrame with DatetimeIndex, or empty DataFrame if no data found.
    """
    stem = _stem(prefix)
    parquet_path = _BASE_DIR / f'{stem}.parquet'
    csv_path = _BASE_DIR / f'{stem}.csv'

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path, columns=columns)
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Datetime' in df.columns:
                    df = df.set_index('Datetime')
                    df.index = pd.to_datetime(df.index)
            print(f"[DATA] Loaded {len(df)} rows from {parquet_path.name}")
            return df
        except Exception as e:
            print(f"[DATA] Parquet load failed ({e}), trying CSV fallback")

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if columns:
                available = [c for c in columns if c in df.columns]
                df = df[available]
            print(f"[DATA] Loaded {len(df)} rows from {csv_path.name}")
            return df
        except Exception as e:
            print(f"[DATA] CSV load failed: {e}")

    return pd.DataFrame()


def save_training_data(df: pd.DataFrame, prefix: str):
    """Atomically save training data as Parquet (snappy) + CSV for backward compat."""
    stem = _stem(prefix)

    # Parquet (atomic write via temp file + rename)
    parquet_path = _BASE_DIR / f'{stem}.parquet'
    fd, tmp = tempfile.mkstemp(suffix='.parquet', dir=_BASE_DIR)
    os.close(fd)
    try:
        df.to_parquet(tmp, compression='snappy')
        os.replace(tmp, parquet_path)
        pq_size = parquet_path.stat().st_size / (1024 * 1024)
        print(f"[DATA] Saved {len(df)} rows to {parquet_path.name} ({pq_size:.1f} MB)")
    except Exception as e:
        print(f"[DATA] Parquet save failed: {e}")
        if os.path.exists(tmp):
            os.unlink(tmp)

    # CSV (backward compat)
    csv_path = _BASE_DIR / f'{stem}.csv'
    fd, tmp = tempfile.mkstemp(suffix='.csv', dir=_BASE_DIR)
    os.close(fd)
    try:
        df.to_csv(tmp)
        os.replace(tmp, csv_path)
    except Exception as e:
        print(f"[DATA] CSV save failed: {e}")
        if os.path.exists(tmp):
            os.unlink(tmp)


def get_latest_timestamp(prefix: str, ticker: str) -> pd.Timestamp | None:
    """Find the latest bar timestamp for a given ticker in existing data.

    Returns None if no data exists or ticker not found.
    """
    df = load_training_data(prefix, columns=['Ticker'])
    if df.empty or 'Ticker' not in df.columns:
        return None
    ticker_rows = df[df['Ticker'] == ticker]
    if ticker_rows.empty:
        return None
    return ticker_rows.index.max()


def append_ticker_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new OHLCV bars into existing data, dedup on index, sort chronologically."""
    if existing_df.empty:
        return new_df.sort_index()
    if new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined


def migrate_csv_to_parquet(prefix: str) -> bool:
    """One-time migration: read CSV, write Parquet. Returns True on success."""
    stem = _stem(prefix)
    csv_path = _BASE_DIR / f'{stem}.csv'
    parquet_path = _BASE_DIR / f'{stem}.parquet'

    if not csv_path.exists():
        print(f"[DATA] No CSV found for {prefix}")
        return False
    if parquet_path.exists():
        print(f"[DATA] Parquet already exists for {prefix}, skipping migration")
        return True

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.to_parquet(parquet_path, compression='snappy')
        csv_size = csv_path.stat().st_size / (1024 * 1024)
        pq_size = parquet_path.stat().st_size / (1024 * 1024)
        print(f"[DATA] Migrated {prefix}: {csv_size:.1f} MB CSV -> {pq_size:.1f} MB Parquet "
              f"({len(df)} rows)")
        return True
    except Exception as e:
        print(f"[DATA] Migration failed for {prefix}: {e}")
        return False


def validate_training_data(df: pd.DataFrame, asset_type: str) -> dict:
    """Validate training data quality. Returns summary dict and prints report.

    Checks:
      - Missing/stale tickers
      - Timestamp gaps (>2h for crypto, >8h for stocks accounting for overnight)
      - NaN/inf values
      - Date range coverage
    """
    report = {
        'rows': len(df),
        'tickers': [],
        'date_range': None,
        'gaps': [],
        'nan_columns': [],
        'inf_columns': [],
        'issues': 0,
    }

    if df.empty:
        print("[DATA-VALIDATE] Empty dataset!")
        report['issues'] = 1
        return report

    # Date range
    report['date_range'] = (str(df.index.min()), str(df.index.max()))

    # Per-ticker stats
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique().tolist()
        report['tickers'] = tickers

        gap_threshold_h = 3 if asset_type == 'crypto' else 16  # crypto=24/7, stocks have overnight
        for ticker in tickers:
            tdf = df[df['Ticker'] == ticker].sort_index()
            if len(tdf) < 2:
                continue
            diffs = tdf.index.to_series().diff().dropna()
            big_gaps = diffs[diffs > pd.Timedelta(hours=gap_threshold_h)]
            if len(big_gaps) > 0:
                for gap_ts, gap_dur in big_gaps.items():
                    report['gaps'].append({
                        'ticker': ticker,
                        'at': str(gap_ts),
                        'duration_h': round(gap_dur.total_seconds() / 3600, 1),
                    })

    # NaN check
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        report['nan_columns'] = [
            {'column': col, 'count': int(cnt), 'pct': round(cnt / len(df) * 100, 1)}
            for col, cnt in nan_cols.items()
        ]

    # Inf check
    numeric = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    import numpy as np
    inf_mask = np.isinf(numeric)
    inf_counts = inf_mask.sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        report['inf_columns'] = [
            {'column': col, 'count': int(cnt)}
            for col, cnt in inf_cols.items()
        ]

    report['issues'] = len(report['gaps']) + len(report['nan_columns']) + len(report['inf_columns'])

    # Print report
    print(f"\n[DATA-VALIDATE] {asset_type} training data report:")
    print(f"  Rows: {report['rows']:,}")
    print(f"  Tickers: {', '.join(report['tickers'])}")
    print(f"  Date range: {report['date_range'][0]} to {report['date_range'][1]}")

    if report['gaps']:
        # Only show first 5 gaps per ticker
        shown = {}
        for g in report['gaps']:
            t = g['ticker']
            shown[t] = shown.get(t, 0) + 1
            if shown[t] <= 3:
                print(f"  GAP: {t} at {g['at']} ({g['duration_h']}h)")
        total_gaps = len(report['gaps'])
        if total_gaps > sum(min(3, v) for v in shown.values()):
            print(f"  ... {total_gaps} total gaps across {len(shown)} tickers")

    if report['nan_columns']:
        for nc in report['nan_columns'][:5]:
            print(f"  NaN: {nc['column']} ({nc['count']} rows, {nc['pct']}%)")

    if report['inf_columns']:
        for ic in report['inf_columns'][:5]:
            print(f"  Inf: {ic['column']} ({ic['count']} rows)")

    if report['issues'] == 0:
        print("  No issues found")

    return report
