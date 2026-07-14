"""Centralized data I/O — Parquet-first loading with CSV fallback.

Provides atomic writes, incremental append, migration, and validation
for training data used by harvest scripts and hypersearch.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np

_BASE_DIR = Path(__file__).resolve().parent

# Mapping from prefix to file stem
_FILE_STEMS = {
    'crypto': 'training_data',
    'stock': 'stock_training_data',
}

# Normal saves write the CSV seconds-to-minutes after the parquet; a CSV newer
# than the parquet by more than this means a parquet save failed and the
# parquet on disk is a frozen stale copy.
_STALE_PARQUET_SLACK_S = 3600


def _stem(prefix: str) -> str:
    return _FILE_STEMS.get(prefix, f'{prefix}_training_data')


def _csv_is_fresher(parquet_path: Path, csv_path: Path) -> bool:
    """True when the CSV is so much newer than the parquet that the parquet
    must be a stale leftover from a failed save."""
    try:
        if not csv_path.exists():
            return False
        gap_s = csv_path.stat().st_mtime - parquet_path.stat().st_mtime
    except OSError:
        return False
    if gap_s > _STALE_PARQUET_SLACK_S:
        print(f"[DATA] {parquet_path.name} is {gap_s / 3600:.1f}h older than "
              f"{csv_path.name} — treating parquet as stale, using CSV")
        return True
    return False


def get_data_path(prefix: str) -> Path:
    """Return the best available data file path (.parquet preferred, .csv fallback)."""
    stem = _stem(prefix)
    parquet = _BASE_DIR / f'{stem}.parquet'
    csv = _BASE_DIR / f'{stem}.csv'
    if parquet.exists() and not _csv_is_fresher(parquet, csv):
        return parquet
    return csv


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

    def _read_csv():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]
        print(f"[DATA] Loaded {len(df)} rows from {csv_path.name}")
        return df

    # A much-newer CSV means the parquet is a frozen leftover from a failed
    # save — serve the fresh CSV instead of silently training on stale data.
    if parquet_path.exists() and _csv_is_fresher(parquet_path, csv_path):
        try:
            return _read_csv()
        except Exception as e:
            print(f"[DATA] CSV load failed ({e}), trying stale Parquet")

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
            return _read_csv()
        except Exception as e:
            print(f"[DATA] CSV load failed: {e}")

    return pd.DataFrame()


def _atomic_to_disk(df: pd.DataFrame, final_path: Path) -> bool:
    """Write df to final_path via a deterministic sibling .tmp + os.replace.

    Deterministic tmp names let the next run overwrite a crash-orphaned tmp
    instead of accumulating full-dataset-sized mkstemp files; mkstemp's 0600
    mode also narrowed the data files' permissions on every save.
    """
    tmp = final_path.parent / (final_path.name + '.tmp')
    try:
        if final_path.suffix == '.parquet':
            df.to_parquet(tmp, compression='snappy')
        else:
            df.to_csv(tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, final_path)
        return True
    except Exception as e:
        print(f"[DATA] save failed for {final_path.name}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False


def save_training_data(df: pd.DataFrame, prefix: str) -> bool:
    """Atomically save training data as Parquet (snappy) + CSV for backward compat.

    Returns True when at least one format persisted the new frame; False when
    BOTH writes failed (the files on disk still hold the OLD data).
    """
    stem = _stem(prefix)
    parquet_path = _BASE_DIR / f'{stem}.parquet'
    csv_path = _BASE_DIR / f'{stem}.csv'

    pq_ok = _atomic_to_disk(df, parquet_path)
    if pq_ok:
        pq_size = parquet_path.stat().st_size / (1024 * 1024)
        print(f"[DATA] Saved {len(df)} rows to {parquet_path.name} ({pq_size:.1f} MB)")

    csv_ok = _atomic_to_disk(df, csv_path)

    if csv_ok and not pq_ok and parquet_path.exists():
        # Loaders prefer parquet on existence, so a leftover parquet would
        # silently shadow the fresh CSV for every downstream consumer.
        try:
            parquet_path.unlink()
            print(f"[DATA] parquet save failed — removed stale {parquet_path.name}, "
                  "loaders will use the fresh CSV")
        except OSError as e:
            print(f"[DATA] WARNING: could not remove stale {parquet_path.name} ({e})")
    if not pq_ok and not csv_ok:
        print(f"[DATA] SAVE FAILED for '{prefix}' — both writes failed, "
              "data on disk is STALE")
    return pq_ok or csv_ok


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
        # Atomic — an interrupt mid-write must not leave a corrupt .parquet
        # where existence-only checks (get_data_path) would pick it up.
        if not _atomic_to_disk(df, parquet_path):
            print(f"[DATA] Migration failed for {prefix}")
            return False
        csv_size = csv_path.stat().st_size / (1024 * 1024)
        pq_size = parquet_path.stat().st_size / (1024 * 1024)
        print(f"[DATA] Migrated {prefix}: {csv_size:.1f} MB CSV -> {pq_size:.1f} MB Parquet "
              f"({len(df)} rows)")
        return True
    except Exception as e:
        print(f"[DATA] Migration failed for {prefix}: {e}")
        return False


def _stock_gap_spans_trading_days(gap_end: pd.Timestamp, gap_dur: pd.Timedelta) -> bool:
    """True when a stock bar gap swallows >=2 full weekdays — likely data loss.

    Weekends (56-66h), overnights (8-18h depending on bar source) and
    single-day holidays are calendar closures, not data loss — flagging them
    buried real gaps under thousands of spurious report entries.
    """
    gap_start = gap_end - gap_dur
    first_full_day = (gap_start + pd.Timedelta(days=1)).date()
    return np.busday_count(first_full_day, gap_end.date()) >= 2


def validate_training_data(df: pd.DataFrame, asset_type: str) -> dict:
    """Validate training data quality. Returns summary dict and prints report.

    Checks:
      - Missing/stale tickers
      - Timestamp gaps (>3h for crypto; stocks: >16h AND spanning >=2 full
        weekdays, so weekends/overnights/single-day holidays are not flagged)
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
            for gap_ts, gap_dur in big_gaps.items():
                if asset_type != 'crypto' and not _stock_gap_spans_trading_days(
                        gap_ts, gap_dur):
                    continue  # calendar closure, not data loss
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
        # Only show first 3 gaps per ticker
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
