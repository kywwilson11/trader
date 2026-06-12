"""Harvest stock training data — Alpaca + yfinance hourly OHLCV.

Supports incremental harvesting (only fetches new bars since last run) and
saves as Parquet + CSV. Falls back through multiple data sources.

Data sources (in priority order):
  - Alpaca: 2016 – present (via get_bars auto-pagination)
  - yfinance: Most recent 730 days (max for hourly)
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time

import pandas as pd
from dotenv import load_dotenv

from indicators import compute_stock_features
from stock_config import load_stock_universe
from adaptive_config import get_forward_bars_list
from data_sources import fetch_with_fallback
from data_utils import (load_training_data, save_training_data,
                         append_ticker_data, validate_training_data)

load_dotenv()

from stock_config import TRAINING_CANDIDATE_POOL, AS_OF_TOP_K, CANDIDATE_START

_UNIVERSE = [t for t in load_stock_universe() if '/' not in t]
# Universe + sector-diverse candidates (training-only; bots don't trade
# the extras). Candidates fetch from CANDIDATE_START to bound memory.
STOCK_TICKERS = sorted(set(_UNIVERSE) | set(TRAINING_CANDIDATE_POOL))
_CANDIDATE_ONLY = set(TRAINING_CANDIDATE_POOL) - set(_UNIVERSE)

BENCHMARK = 'SPY'

ALPACA_START = '2016-01-01'

# Multi-horizon forward returns (bars ahead) — read from adaptive state
FORWARD_BARS = get_forward_bars_list('stock')


def _get_alpaca_api():
    """Build Alpaca REST client, or None if credentials missing."""
    try:
        # Increase SDK internal retry backoff (default 3s is too aggressive)
        os.environ.setdefault('APCA_RETRY_WAIT', '10')
        os.environ.setdefault('APCA_RETRY_MAX', '5')
        if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_API_SECRET'):
            return None
        # Shared constructor: legacy SDK with automatic alpaca-py fallback
        from trading_utils import get_api
        return get_api()
    except Exception as e:
        print(f"WARNING: Could not create Alpaca API client: {e}")
        return None


def _get_incremental_start(existing_df, ticker):
    """Find start date for incremental fetch (48h overlap for safety)."""
    if existing_df.empty or 'Ticker' not in existing_df.columns:
        return ALPACA_START

    ticker_rows = existing_df[existing_df['Ticker'] == ticker]
    if ticker_rows.empty:
        return ALPACA_START

    latest = ticker_rows.index.max()
    # Go back 48h for overlap to catch any gaps
    start = latest - pd.Timedelta(hours=48)
    return str(start.date())


def fetch_spy_close(api=None):
    """Fetch SPY hourly close from all sources for benchmark relative strength."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = fetch_with_fallback(BENCHMARK, ALPACA_START, api=api, asset_type='stock')
    if df is None or df.empty:
        return None
    return df['Close']


def prepare_stock_data(ticker, spy_close=None, api=None, existing_ohlcv=None,
                        start_date=None):
    """Fetch bars, merge with existing, compute features, add targets."""
    print(f"Processing {ticker}...")

    # Fetch new bars (incremental or full)
    fetch_start = start_date or ALPACA_START
    new_ohlcv = fetch_with_fallback(ticker, fetch_start, api=api, asset_type='stock')

    # Merge with existing OHLCV if incremental
    if existing_ohlcv is not None and not existing_ohlcv.empty:
        if new_ohlcv is not None and not new_ohlcv.empty:
            ohlcv = append_ticker_data(existing_ohlcv, new_ohlcv)
            new_bars = len(ohlcv) - len(existing_ohlcv)
            print(f"  [INCREMENTAL] {ticker}: {new_bars} new bars "
                  f"(total {len(ohlcv)})")
        else:
            ohlcv = existing_ohlcv
            print(f"  [INCREMENTAL] {ticker}: no new bars, using existing {len(ohlcv)}")
    elif new_ohlcv is not None:
        ohlcv = new_ohlcv
    else:
        return None

    if ohlcv.empty:
        return None

    # Recompute ALL features on full history (indicators need lookback windows)
    df = compute_stock_features(ohlcv, spy_close=spy_close, symbol=ticker)

    # Multi-horizon targets: return over N bars as a percentage
    for fb in FORWARD_BARS:
        future_close = df['Close'].shift(-fb)
        df[f'Target_Return_{fb}'] = (future_close - df['Close']) / df['Close'] * 100

    # Triple-barrier targets matched to the LIVE exit stack (ATR stop /
    # trailing / TP / EOD flatten — the same policy_exits kernel the
    # backtester runs). For stocks the EOD barrier fixes the structural
    # label mismatch: raw Target_Return_12..48 spans 1.8-7.4 trading days
    # while live stock holds are capped at ~6.5h by the 15:50 flatten.
    from policy_exits import compute_tb_labels
    for col, vals in compute_tb_labels(df, FORWARD_BARS, 'stock').items():
        df[col] = vals

    # Backward compat: Target_Return = shortest horizon
    df['Target_Return'] = df[f'Target_Return_{FORWARD_BARS[0]}']

    df = df.dropna()
    df = _asof_tradability_mask(df, ticker)
    return df


def _asof_membership_mask(df, top_k=AS_OF_TOP_K):
    """Keep rows whose name ranked top-K by 30d dollar volume AS OF that
    day. With this, a 2024 listing contributes no 2021 rows, and a name
    only contributes history from periods a mechanical liquidity rule
    would have selected it — membership look-ahead removed."""
    import pandas as pd
    if '_DV30' not in df.columns or 'Ticker' not in df.columns:
        return df
    key = pd.DataFrame({'day': df.index.normalize(),
                        'tick': df['Ticker'].values,
                        'dv': df['_DV30'].values})
    rep = key.groupby(['day', 'tick'])['dv'].max()
    ranks = rep.groupby(level=0).rank(ascending=False, method='min')
    keep = ranks[ranks <= top_k]
    flag = key.merge(keep.rename('rank').reset_index(),
                     on=['day', 'tick'], how='left')['rank'].notna()
    kept = df[flag.values]
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[AS-OF] membership mask: dropped {dropped}/{len(df)} rows "
              f"outside the as-of top-{top_k} by dollar volume")
    return kept


# As-of tradability floors. Several CURRENT universe names traded as
# illiquid sub-$2 stocks in 2021-23 (POET, QBTS, RDW...). Training on
# those rows injects look-ahead — "this name later became liquid enough
# to make today's list" — and teaches microstructure (cent-wide books,
# halts, 20% gaps) the bot will never trade at today's notionals.
# Together with the candidate pool + as-of membership mask this removes
# the listing/liquidity AND membership look-ahead; the residual is names
# that faded before today (no free historical-membership data exists).
MIN_DOLLAR_VOLUME = 5_000_000   # 30d median daily $ volume
MIN_PRICE = 3.0                  # institutional-floor convention


def _asof_tradability_mask(df, ticker):
    """Drop rows from periods when the name wasn't realistically tradable.

    Also stamps _DV30 (trailing 30d median daily dollar volume) used by
    the cross-sectional as-of membership mask after the harvest concat.
    """
    try:
        from panel_ranks import dv30  # ONE dv implementation, shared with
        dv_aligned = dv30(df)         # the live panel mask (parity)
        df = df.copy()
        df['_DV30'] = dv_aligned.values
        dv_ok = dv_aligned >= MIN_DOLLAR_VOLUME
        px_ok = df['Close'] >= MIN_PRICE
        mask = (dv_ok & px_ok).values
        dropped = int((~mask).sum())
        if dropped:
            print(f"  [AS-OF] {ticker}: dropped {dropped}/{len(df)} rows "
                  f"below tradability floors (illiquid/penny phase)")
        return df[mask]
    except Exception as e:
        print(f"  [AS-OF] {ticker}: mask skipped ({e})")
        return df


def main():
    api = _get_alpaca_api()
    if api is None:
        print("WARNING: No Alpaca API credentials — using yfinance only (limited to ~730 days)")
    else:
        print("Alpaca API connected — fetching historical data from 2016")

    # Load existing data for incremental harvesting
    existing = load_training_data('stock')
    is_incremental = not existing.empty
    if is_incremental:
        print(f"Existing data: {len(existing)} rows — incremental mode")
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_ohlcv = [c for c in ohlcv_cols if c in existing.columns]
    else:
        print("No existing data — full harvest")
        available_ohlcv = []

    spy_close = fetch_spy_close(api=api)

    all_data = []
    for t in STOCK_TICKERS:
        # For incremental: extract this ticker's existing OHLCV
        existing_ohlcv = None
        start = CANDIDATE_START if t in _CANDIDATE_ONLY else ALPACA_START
        if is_incremental and available_ohlcv and 'Ticker' in existing.columns:
            ticker_data = existing[existing['Ticker'] == t]
            if not ticker_data.empty:
                existing_ohlcv = ticker_data[available_ohlcv]
                start = _get_incremental_start(existing, t)
                print(f"  [INCREMENTAL] {t}: fetching from {start}")

        stock_df = prepare_stock_data(t, spy_close, api=api,
                                       existing_ohlcv=existing_ohlcv,
                                       start_date=start)
        if stock_df is not None:
            stock_df['Ticker'] = t
            all_data.append(stock_df)

    # Combine and save
    if not all_data:
        print("ERROR: No data fetched for any ticker. Check API credentials and network.")
        return
    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()

    # Cross-sectional as-of membership (uses _DV30 stamped per ticker)
    final_df = _asof_membership_mask(final_df)

    # Cross-sectional rank features over the surviving members (wave-3
    # flagship: selection is a RELATIVE decision — give the models each
    # name's rank within the panel THIS hour, not just its own history).
    # DV30 (the dollar-volume turnover proxy) is exposed to the rank
    # layer then dropped: only its RANK is a feature, never the level.
    final_df['DV30'] = final_df['_DV30']
    from panel_ranks import add_panel_ranks, neutral_fill_cs
    final_df = add_panel_ranks(final_df)
    final_df = neutral_fill_cs(final_df)
    final_df = final_df.drop(columns=['_DV30', 'DV30'], errors='ignore')

    # Add historical sentiment — LAGGED one day for point-in-time integrity.
    # The daily score for day D aggregates ALL of day D's articles
    # (including ones published after each bar), so giving day-D bars the
    # day-D score leaked intraday-future news into training. Day-D bars now
    # see day D-1's COMPLETED score — exactly what live inference can know.
    try:
        import datetime as _dt
        from sentiment_history import fetch_stock_sentiment_history
        start_date = str((final_df.index.min() - pd.Timedelta(days=1)).date())
        end_date = str(final_df.index.max().date())
        sentiment = fetch_stock_sentiment_history(
            STOCK_TICKERS, start_date, end_date, cached_only=True)
        final_df['Daily_Sentiment'] = [
            sentiment.get((ticker, str(date - _dt.timedelta(days=1))), 0.0)
            for ticker, date in zip(final_df['Ticker'], final_df.index.date)
        ]
        filled = sum(1 for v in final_df['Daily_Sentiment'] if v != 0.0)
        print(f"Daily_Sentiment (lagged 1d): {filled}/{len(final_df)} bars have sentiment")
    except Exception as e:
        print(f"WARNING: Could not load stock sentiment history: {e}")
        final_df['Daily_Sentiment'] = 0.0

    # Save as Parquet + CSV
    save_training_data(final_df, 'stock')

    # Summary
    print(f"\nDone! Saved {len(final_df)} rows of stock training data")
    print(f"Stocks harvested: {len(all_data)}/{len(STOCK_TICKERS)}")
    target_cols = [c for c in final_df.columns if c.startswith('Target_Return')]
    exclude = set(target_cols) | {'Ticker', 'Date', 'Datetime'}
    feature_count = len([c for c in final_df.columns if c not in exclude])
    print(f"Feature columns: {feature_count}")
    print(f"Target columns: {target_cols}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

    # Validation report
    validate_training_data(final_df, 'stock')


if __name__ == '__main__':
    main()
