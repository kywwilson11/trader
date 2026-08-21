"""Harvest crypto training data — Alpaca + yfinance + CryptoCompare hourly OHLCV.

Supports incremental harvesting (only fetches new bars since last run) and
saves as Parquet + CSV. Falls back through multiple data sources.

Data sources (in priority order):
  - Alpaca: Jan 2021 – present (via get_crypto_bars auto-pagination)
  - yfinance: Most recent 730 days (max for hourly)
  - CryptoCompare: DEAD in production since the CoinDesk migration —
    keyless requests return HTTP 401 (verified 2026-07-02); the leg is
    kept in data_sources pending an API key or removal, so effective
    redundancy is Alpaca (2021+) + yfinance (last 730d)

RUNBOOK — TRADER_RAW_SIDECAR activation (D39/D08): set the env var on the
Jetson with NO raw_ohlcv.parquet present. Incremental state then comes from
the raw sidecar ONLY, so the absent file forces ONE full refetch from
ALPACA_START that rebuilds the warmup head the feature-store incremental
path loses every run. Enable TRADER_YF_WINDOW_SLICE in the SAME event —
the CRYPTO store is the one D08 actually corrupts (crypto merges yfinance
bars into the same grid, so every incremental harvest lets ~730d of Yahoo
composite prices overwrite Alpaca venue rows via the store-level
keep='last'). Both flags are model-facing (training-store contents change)
— gotcha #2 applies: delete v2_study.db + stock_v2_study.db and reset the
adaptive best_score.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import pandas as pd
from dotenv import load_dotenv

from indicators import compute_features
from adaptive_config import get_forward_bars_list
from data_sources import fetch_with_fallback
from data_utils import (load_training_data, save_training_data,
                         append_ticker_data, validate_training_data,
                         raw_sidecar_enabled, load_raw_ohlcv, save_raw_ohlcv,
                         merge_raw_ohlcv, find_interior_gaps)
from market_data import fetch_historical_bars

load_dotenv()

# Top 6 cryptos by market cap, matching crypto_loop.py
# yfinance format — converted to Alpaca format (/ instead of -) for Alpaca calls
CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
    'LINK-USD',
]

BENCHMARK = 'BTC-USD'

ALPACA_START = '2021-01-01'

# Multi-horizon forward returns (bars ahead) — read from adaptive state
FORWARD_BARS = get_forward_bars_list('crypto')


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


def fetch_btc_close(api=None):
    """Fetch BTC-USD hourly close from all sources for cross-asset features."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = fetch_with_fallback(BENCHMARK, ALPACA_START, api=api, asset_type='crypto')
    if df is None or df.empty:
        return None
    return df['Close']


def prepare_data(ticker, btc_close=None, api=None, existing_ohlcv=None,
                 start_date=None, src_totals=None, raw_out=None):
    """Fetch bars, merge with existing, compute features, add targets."""
    print(f"Processing {ticker}...")

    # Fetch new bars (incremental or full)
    fetch_start = start_date or ALPACA_START
    new_ohlcv = fetch_with_fallback(ticker, fetch_start, api=api, asset_type='crypto')

    # D08 provenance accounting (newly fetched bars, pre-merge)
    if (src_totals is not None and new_ohlcv is not None
            and 'Src' in new_ohlcv.columns):
        for s, n in new_ohlcv['Src'].value_counts().items():
            src_totals[s] = src_totals.get(s, 0) + int(n)

    # Merge with existing OHLCV if incremental. (No overlap-divergence
    # guard here: corporate actions are stock-only, and crypto cross-venue
    # closes would false-positive.)
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

    # D39 sidecar capture (raw bars, provenance kept), then strip Src so
    # it never reaches the feature computation or the saved feature store
    # (flag OFF this drop keeps the store byte-identical).
    if raw_out is not None:
        raw_out[ticker] = ohlcv
    ohlcv = ohlcv.drop(columns=['Src'], errors='ignore')

    if ohlcv.empty:
        return None

    # Recompute ALL features on full history (indicators need lookback windows)
    df = compute_features(ohlcv, btc_close=btc_close)

    # Perp funding positioning features (Binance archive, point-in-time
    # ffill — each 8h print is known at its funding timestamp). Crypto
    # only; absent archive -> features omitted and the preset filter just
    # excludes them, keeping old datasets trainable.
    try:
        from funding_archive import funding_features_for_index
        alpaca_sym = ticker.replace('-', '/')
        ffeats = funding_features_for_index(alpaca_sym, df.index)
        if ffeats is not None:
            for col, vals in ffeats.items():
                df[col] = vals
    except Exception as e:
        print(f"  [FUNDING] {ticker}: feature merge skipped ({e})")

    # Open-interest + top-trader positioning (Binance metrics archive,
    # hourly point-in-time)
    try:
        from oi_archive import (oi_features_for_index,
                                ls_features_for_index,
                                taker_features_for_index)
        alpaca_sym = ticker.replace('-', '/')
        for feats in (oi_features_for_index(alpaca_sym, df.index),
                      ls_features_for_index(alpaca_sym, df.index),
                      taker_features_for_index(alpaca_sym, df.index)):
            if feats is not None:
                for col, vals in feats.items():
                    df[col] = vals
    except Exception as e:
        print(f"  [OI] {ticker}: feature merge skipped ({e})")

    # B05.1 crypto spread stamp (quote-first tier map) — DARK behind
    # TRADER_CRYPTO_SPREAD_STAMP. KILL_LIST:90-adjacent: owner ruling plus
    # the gotcha-#2 re-harvest/retrain event required before activation
    # (backtest/meta_label auto-consume Eff_Spread_Pct on column presence).
    try:
        from liquidity import stamp_crypto_spreads
        df = stamp_crypto_spreads(df, ticker.replace('-', '/'))
    except Exception as e:
        print(f"  [SPREAD] {ticker}: crypto stamp skipped ({e})")

    # B21 cost-regime meta features (Option B) — DARK behind
    # TRADER_COST_REGIME_FEATURES; model-facing, rides the same bundled
    # retrain event (gotcha #2). One memoized FRED fetch per process.
    try:
        from cost_regime import stamp_cost_regime_features
        df = stamp_cost_regime_features(df, 'crypto')
    except Exception as e:
        print(f"  [COST-REGIME] {ticker}: skipped ({e})")

    # Multi-horizon targets: return over N bars as a percentage
    for fb in FORWARD_BARS:
        future_close = df['Close'].shift(-fb)
        df[f'Target_Return_{fb}'] = (future_close - df['Close']) / df['Close'] * 100

    # Triple-barrier targets matched to the LIVE exit stack (ATR stop /
    # trailing / TP, vertical barrier at fb bars — same policy_exits
    # kernel the backtester runs), so the model can learn the return the
    # policy actually realizes instead of a hold-exactly-fb-bars fiction.
    from policy_exits import compute_tb_labels
    for col, vals in compute_tb_labels(df, FORWARD_BARS, 'crypto').items():
        df[col] = vals

    # Backward compat: Target_Return = shortest horizon
    df['Target_Return'] = df[f'Target_Return_{FORWARD_BARS[0]}']

    df = _fill_archive_features(df)
    df = df.dropna()
    return df


# Archive-derived feature columns (Binance funding/metrics merges)
ARCHIVE_FEATURES = ['Funding_Rate_Ann', 'Funding_Z', 'Funding_Chg_24h',
                    'OI_Chg_24h', 'OI_Z', 'TT_LS_Z', 'Taker_Imb_24h']


def _fill_archive_features(df):
    """Neutral-fill archive features so dropna() can't eat their rows.

    These columns are NaN before their archive's first print (OI starts
    2023; the first OI sync covers ~1y until back-fill finishes) and
    during z-score warmups. Letting dropna() drop those rows would
    silently discard YEARS of training data. 0.0 is the exact value
    live serving injects when history is missing, so train and serve
    agree that "no data" reads as "neutral signal".
    """
    for col in ARCHIVE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def main():
    api = _get_alpaca_api()
    if api is None:
        print("WARNING: No Alpaca API credentials — using yfinance + CryptoCompare only")
    else:
        print("Alpaca API connected — fetching historical data from 2021")

    # Sync the Binance funding-rate archive (idempotent: skips months
    # already stored; one bigger burst on the first run only)
    try:
        from funding_archive import sync as sync_funding
        sync_funding()
    except Exception as e:
        print(f"WARNING: funding archive sync failed ({e}) — "
              f"funding features will be omitted this harvest")

    # Sync the Binance OI metrics archive (daily files; newest-first,
    # capped per run — older history back-fills across harvests)
    try:
        from oi_archive import sync as sync_oi
        sync_oi()
    except Exception as e:
        print(f"WARNING: OI archive sync failed ({e}) — "
              f"OI features will be omitted this harvest")

    # Load existing data for incremental harvesting.
    # Under TRADER_RAW_SIDECAR (D39), incremental state comes from the RAW
    # store ONLY: an empty/absent sidecar forces a full refetch from
    # ALPACA_START — the head-rebuild runbook mechanism, no CLI flag needed.
    use_sidecar = raw_sidecar_enabled()
    raw = load_raw_ohlcv('crypto') if use_sidecar else pd.DataFrame()
    if use_sidecar:
        existing = pd.DataFrame()
        available_ohlcv = []
        if raw.empty:
            print("[SIDECAR] raw OHLCV store EMPTY — full refetch")
        else:
            print(f"[SIDECAR] raw OHLCV store: {len(raw)} rows")
        is_incremental = False
    else:
        existing = load_training_data('crypto')
        is_incremental = not existing.empty
        if is_incremental:
            print(f"Existing data: {len(existing)} rows — incremental mode")
            # Extract per-ticker OHLCV from existing (before features were computed)
            # We need raw OHLCV for merge, but existing has features. Extract what we need.
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_ohlcv = [c for c in ohlcv_cols if c in existing.columns]
        else:
            print("No existing data — full harvest")
            available_ohlcv = []

    btc_close = fetch_btc_close(api=api)
    if btc_close is None:
        print("WARNING: benchmark fetch failed — BTC cross-asset features "
              "(BTC_Return_1h/BTC_SMA_Ratio/BTC_RSI) will be OMITTED from "
              "this dataset")

    all_data = []
    src_totals = {}
    raw_out = {}
    for t in CRYPTO_TICKERS:
        # For incremental: extract this ticker's existing OHLCV
        existing_ohlcv = None
        start = ALPACA_START
        if use_sidecar and not raw.empty and 'Ticker' in raw.columns:
            t_raw = raw[raw['Ticker'] == t]
            if not t_raw.empty:
                cols = [c for c in ['Open', 'High', 'Low', 'Close',
                                    'Volume', 'Src'] if c in t_raw.columns]
                existing_ohlcv = t_raw[cols]
                start = str((t_raw.index.max()
                             - pd.Timedelta(hours=48)).date())
                print(f"  [SIDECAR] {t}: fetching from {start}")
                # Interior-gap repair (F1-gated, bounded): refetch just
                # the holes instead of a full refetch.
                if api is not None:
                    for g0, g1 in find_interior_gaps(t_raw.index, 'crypto',
                                                     max_windows=5):
                        patch = fetch_historical_bars(
                            api, t.replace('-', '/'), str(g0.date()),
                            asset_type='crypto',
                            end_date=str((g1 + pd.Timedelta(days=1)).date()))
                        if patch is not None and not patch.empty:
                            patch['Src'] = 'alpaca'
                            existing_ohlcv = append_ticker_data(
                                existing_ohlcv, patch)
                            print(f"  [GAP-REPAIR] {t}: refilled {g0}..{g1} "
                                  f"(+{len(patch)} bars)")
        elif is_incremental and available_ohlcv and 'Ticker' in existing.columns:
            ticker_data = existing[existing['Ticker'] == t]
            if not ticker_data.empty:
                existing_ohlcv = ticker_data[available_ohlcv]
                start = _get_incremental_start(existing, t)
                print(f"  [INCREMENTAL] {t}: fetching from {start}")

        crypto_df = prepare_data(t, btc_close=btc_close, api=api,
                                  existing_ohlcv=existing_ohlcv,
                                  start_date=start,
                                  src_totals=src_totals,
                                  raw_out=(raw_out if use_sidecar else None))
        if crypto_df is not None:
            crypto_df['Ticker'] = t
            all_data.append(crypto_df)

    # Persist the raw sidecar (D39), then free it before the concat.
    if use_sidecar and raw_out:
        for t, frame in raw_out.items():
            raw = merge_raw_ohlcv(raw, frame, t)
        save_raw_ohlcv(raw, 'crypto')
        del raw, raw_out

    # Free the previous full feature panel before pd.concat allocates the
    # new one — peak-RSS matters on the 8GB Jetson (harvest can run beside
    # the bots). `existing` is not referenced past this point.
    del existing

    # Combine and save — sort chronologically for time-series split in training
    if not all_data:
        print("ERROR: No data fetched for any ticker. Check API credentials and network.")
        sys.exit(1)  # nonzero so run_pipeline's retry/notify machinery fires
    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()

    # Add historical sentiment (Fear & Greed Index for crypto)
    try:
        from sentiment_history import fetch_crypto_sentiment_history
        start_date = str(final_df.index.min().date())
        end_date = str(final_df.index.max().date())
        sentiment = fetch_crypto_sentiment_history(start_date, end_date)
        final_df['Daily_Sentiment'] = pd.Series(
            final_df.index.date.astype(str), index=final_df.index
        ).map(sentiment).fillna(0.0).values
        filled = (final_df['Daily_Sentiment'] != 0).sum()
        print(f"Daily_Sentiment: {filled}/{len(final_df)} bars have sentiment")
    except Exception as e:
        print(f"WARNING: Could not fetch crypto sentiment history: {e}")
        final_df['Daily_Sentiment'] = 0.0

    # Save as Parquet + CSV
    if not save_training_data(final_df, 'crypto'):
        print("ERROR: Could not save training data (both Parquet and CSV "
              "writes failed) — data on disk is STALE")
        sys.exit(1)

    # Summary
    print(f"\nDone! Saved {len(final_df)} rows of training data")
    print(f"Cryptos harvested: {len(all_data)}/{len(CRYPTO_TICKERS)}")
    if src_totals:
        total = sum(src_totals.values())
        comp = ', '.join(f"{s} {n} ({n / total:.0%})"
                         for s, n in sorted(src_totals.items(),
                                            key=lambda kv: -kv[1]))
        print(f"Source composition (newly fetched bars): {comp}")
    # Mirror training's exclude list (hypersearch_v2): TB_* are labels, not
    # features; OHLCV stay counted because training keeps them as features.
    target_cols = [c for c in final_df.columns if c.startswith('Target_Return')]
    tb_cols = [c for c in final_df.columns if c.startswith('TB_')]
    exclude = set(target_cols) | set(tb_cols) | {'Ticker', 'Date', 'Datetime'}
    feature_count = len([c for c in final_df.columns if c not in exclude])
    print(f"Feature columns: {feature_count}")
    print(f"Target columns: {target_cols + tb_cols}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

    # Validation report
    validate_training_data(final_df, 'crypto')


if __name__ == '__main__':
    main()
