"""Harvest stock training data — Alpaca + yfinance hourly OHLCV.

Supports incremental harvesting (only fetches new bars since last run) and
saves as Parquet + CSV. Falls back through multiple data sources.

Data sources (in priority order):
  - Alpaca: 2016 – present (via get_bars auto-pagination)
  - yfinance: Most recent 730 days (max for hourly)

RUNBOOK — TRADER_RAW_SIDECAR activation (D39/D08): set the env var on the
Jetson with NO stock_raw_ohlcv.parquet present. Incremental state then comes
from the raw sidecar ONLY, so the absent file forces ONE full refetch from
ALPACA_START that rebuilds the ~112-bars-per-run warmup head the
feature-store incremental path loses. Enable TRADER_YF_WINDOW_SLICE in the
SAME event (kills the yfinance-730d overwrite of Alpaca rows). Both are
model-facing (training-store contents change) — gotcha #2 applies: delete
v2_study.db + stock_v2_study.db and reset the adaptive best_score.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import pandas as pd
from dotenv import load_dotenv

from indicators import compute_stock_features
from stock_config import load_stock_universe
from adaptive_config import get_forward_bars_list
from data_sources import fetch_with_fallback
from data_utils import (load_training_data, save_training_data,
                         append_ticker_data, validate_training_data,
                         raw_sidecar_enabled, load_raw_ohlcv, save_raw_ohlcv,
                         merge_raw_ohlcv, find_interior_gaps,
                         overlap_close_divergence, OVERLAP_DIVERGENCE_MAX)
from market_data import fetch_historical_bars

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


# B05.1 minute-EDGE fetch window (bars are heavy: ~390/day/name). Bounded so
# the Jetson never pulls the full 2016+ minute history.
MINUTE_EDGE_DAYS = int(os.getenv('TRADER_MINUTE_EDGE_DAYS', '120'))


def _minute_edge_overlay(api, ticker, df):
    """Trailing MINUTE_EDGE_DAYS of 1-min bars -> per-day EDGE stamp
    (liquidity.edge_spread_daily_from_minute). Only called when
    liquidity.STOCK_MINUTE_EDGE; None on any failure (hourly stamp kept).
    Jetson-only in practice (needs the Alpaca API + minute history).
    NOTE (B05 prerequisite, unresolved): confirm Basic-plan minute bars are
    SIP-sourced, not IEX-only, before trusting the levels."""
    try:
        if api is None:
            return None
        start = df.index.max() - pd.Timedelta(days=MINUTE_EDGE_DAYS)
        bars = api.get_bars(ticker, '1Min', start=start.isoformat(),
                            adjustment='all')
        rows, ts = [], []
        for b in bars:
            rows.append({'Open': float(b.o), 'High': float(b.h),
                         'Low': float(b.l), 'Close': float(b.c)})
            ts.append(b.t)
        if not rows:
            return None
        mdf = pd.DataFrame(rows, index=pd.DatetimeIndex(ts))
        from liquidity import edge_spread_daily_from_minute
        return edge_spread_daily_from_minute(mdf, df.index, symbol=ticker)
    except Exception as e:
        print(f"  [SPREAD-MIN] {ticker}: minute EDGE skipped ({e})")
        return None


def fetch_spy_close(api=None):
    """Fetch SPY hourly close from all sources for benchmark relative strength."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = fetch_with_fallback(BENCHMARK, ALPACA_START, api=api, asset_type='stock')
    if df is None or df.empty:
        return None
    return df['Close']


def prepare_stock_data(ticker, spy_close=None, api=None, existing_ohlcv=None,
                        start_date=None, src_totals=None, raw_out=None):
    """Fetch bars, merge with existing, compute features, add targets."""
    print(f"Processing {ticker}...")

    # Fetch new bars (incremental or full)
    fetch_start = start_date or ALPACA_START
    new_ohlcv = fetch_with_fallback(ticker, fetch_start, api=api, asset_type='stock')

    # D08 provenance accounting (newly fetched bars, pre-merge)
    if (src_totals is not None and new_ohlcv is not None
            and 'Src' in new_ohlcv.columns):
        for s, n in new_ohlcv['Src'].value_counts().items():
            src_totals[s] = src_totals.get(s, 0) + int(n)

    # Merge with existing OHLCV if incremental
    if existing_ohlcv is not None and not existing_ohlcv.empty:
        if new_ohlcv is not None and not new_ohlcv.empty:
            # B15 merge guard: refuse an incremental merge whose 48h
            # overlap closes diverge >1% — split/adjustment drift would
            # otherwise splice two adjustment regimes into one series.
            max_div, n_overlap = overlap_close_divergence(existing_ohlcv,
                                                          new_ohlcv)
            if n_overlap and max_div > OVERLAP_DIVERGENCE_MAX:
                print(f"  [MERGE-GUARD] {ticker}: overlapping closes diverge "
                      f"{max_div:.1%} over {n_overlap} bars (>1%) — "
                      f"split/adjustment drift.\n"
                      f"  [MERGE-GUARD] REFUSING incremental merge; keeping "
                      f"existing rows. Full refetch required (delete this "
                      f"ticker's rows, or rebuild via TRADER_RAW_SIDECAR "
                      f"with the sidecar absent).")
                ohlcv = existing_ohlcv
            else:
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
    df = compute_stock_features(ohlcv, spy_close=spy_close, symbol=ticker)

    # Per-name effective spread (Ardia-Guidotti-Kroencke EDGE), PERCENT of
    # price, from a strictly TRAILING window — point-in-time like _DV30. This
    # replaces the flat offline spread haircut so the meta-label / backtest /
    # hypersearch cost matches the real-spread LIVE gate (wave 6). Never NaN
    # (floored), so it survives the dropna below.
    try:
        from liquidity import (edge_spread_series, SPREAD_FLOOR_PCT,
                               SPREAD_CAP_PCT, STOCK_MINUTE_EDGE)
        sp = edge_spread_series(df, symbol=ticker)
        if STOCK_MINUTE_EDGE:
            # B05.1 minute-bar EDGE (DARK): overlay the per-day minute
            # estimate where covered; hourly stamp retained elsewhere.
            msp = _minute_edge_overlay(api, ticker, df)
            if msp is not None and msp.notna().any():
                n_cov = int(msp.notna().sum())
                sp = sp.where(msp.isna(), msp)
                print(f"  [SPREAD-MIN] {ticker}: minute EDGE overlaid on "
                      f"{n_cov}/{len(sp)} bars")
        df['Eff_Spread_Pct'] = sp.values
        print(f"  [SPREAD] {ticker}: median {sp.median():.3f}% "
              f"floor-hit {float((sp == SPREAD_FLOOR_PCT).mean()):.0%} "
              f"cap-hit {float((sp == SPREAD_CAP_PCT).mean()):.0%}")
    except Exception as e:
        print(f"  [SPREAD] {ticker}: EDGE stamp skipped ({e}) — "
              f"flat fallback will be used downstream")

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

    # FINRA daily shorting-flow features (wave 4; informed sell-side
    # pressure, day-D file maps to day-D+1 bars — point-in-time)
    try:
        from short_flow import svr_features_for_index
        sf = svr_features_for_index(ticker, df.index)
        if sf is not None:
            for col, vals in sf.items():
                df[col] = vals
    except Exception as e:
        print(f"  [SHORT-FLOW] {ticker}: merge skipped ({e})")

    # B21 cost-regime meta features (Option B) — DARK behind
    # TRADER_COST_REGIME_FEATURES; model-facing, rides the same bundled
    # retrain event (gotcha #2). One memoized FRED fetch per process.
    try:
        from cost_regime import stamp_cost_regime_features
        df = stamp_cost_regime_features(df, 'stock')
    except Exception as e:
        print(f"  [COST-REGIME] {ticker}: skipped ({e})")

    # Backward compat: Target_Return = shortest horizon
    df['Target_Return'] = df[f'Target_Return_{FORWARD_BARS[0]}']

    df = _fill_warmup_features(df)
    df = df.dropna()
    df = _asof_tradability_mask(df, ticker)
    return df


# Daily-window warmup fill — SHARED with the live path (predict_now), single
# source of truth in indicators.py: the harvest keeps warmup rows with the
# same neutral 0.0/0.5 values the live path serves on its short frames.
# Diverging fills here would silently break train/serve parity.
from indicators import (
    WARMUP_FEATURES_ZERO, WARMUP_FEATURES_HALF,
    fill_warmup_features as _fill_warmup_features,
)


def _asof_membership_mask(df, top_k=AS_OF_TOP_K):
    """Keep rows whose name ranked top-K by 30d dollar volume AS OF that
    day. With this, a 2024 listing contributes no 2021 rows, and a name
    only contributes history from periods a mechanical liquidity rule
    would have selected it — membership look-ahead removed."""
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


def _summary_feature_split(columns):
    """Mirror training's exclude list (hypersearch_v2): TB_* are labels,
    not features; OHLCV stay counted because training keeps them as
    features. Same shape as the crypto twin's summary block."""
    target_cols = [c for c in columns if c.startswith('Target_Return')]
    tb_cols = [c for c in columns if c.startswith('TB_')]
    exclude = set(target_cols) | set(tb_cols) | {'Ticker', 'Date', 'Datetime'}
    feature_count = len([c for c in columns if c not in exclude])
    return feature_count, target_cols, tb_cols


def main():
    api = _get_alpaca_api()
    if api is None:
        print("WARNING: No Alpaca API credentials — using yfinance only (limited to ~730 days)")
    else:
        print("Alpaca API connected — fetching historical data from 2016")

    # Load existing data for incremental harvesting.
    # Under TRADER_RAW_SIDECAR (D39), incremental state comes from the RAW
    # store ONLY: an empty/absent sidecar forces a full refetch from
    # ALPACA_START/CANDIDATE_START — the head-rebuild runbook mechanism,
    # no CLI flag needed.
    use_sidecar = raw_sidecar_enabled()
    raw = load_raw_ohlcv('stock') if use_sidecar else pd.DataFrame()
    if use_sidecar:
        existing = pd.DataFrame()
        if raw.empty:
            print("[SIDECAR] raw OHLCV store EMPTY — full refetch")
        else:
            print(f"[SIDECAR] raw OHLCV store: {len(raw)} rows")
    else:
        existing = load_training_data('stock')
    is_incremental = not existing.empty
    if is_incremental:
        print(f"Existing data: {len(existing)} rows — incremental mode")
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_ohlcv = [c for c in ohlcv_cols if c in existing.columns]
    else:
        if not use_sidecar:
            print("No existing data — full harvest")
        available_ohlcv = []

    # Sync the FINRA daily short-volume archive (newest-first, capped;
    # back-fills across harvests like the OI archive)
    try:
        from short_flow import sync as sync_short_flow
        sync_short_flow()
    except Exception as e:
        print(f"WARNING: short-flow sync failed ({e}) — "
              f"SVR features omitted this harvest")

    spy_close = fetch_spy_close(api=api)

    all_data = []
    src_totals = {}
    raw_out = {}
    for t in STOCK_TICKERS:
        # For incremental: extract this ticker's existing OHLCV
        existing_ohlcv = None
        start = CANDIDATE_START if t in _CANDIDATE_ONLY else ALPACA_START
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
                    for g0, g1 in find_interior_gaps(t_raw.index, 'stock',
                                                     max_windows=5):
                        patch = fetch_historical_bars(
                            api, t, str(g0.date()), asset_type='stock',
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

        stock_df = prepare_stock_data(t, spy_close, api=api,
                                       existing_ohlcv=existing_ohlcv,
                                       start_date=start,
                                       src_totals=src_totals,
                                       raw_out=(raw_out if use_sidecar
                                                else None))
        if stock_df is not None:
            stock_df['Ticker'] = t
            all_data.append(stock_df)

    # Persist the raw sidecar (D39), then free it before the concat —
    # peak-RSS matters on the 8GB Jetson.
    if use_sidecar and raw_out:
        for t, frame in raw_out.items():
            raw = merge_raw_ohlcv(raw, frame, t)
        save_raw_ohlcv(raw, 'stock')
        del raw, raw_out

    # Combine and save
    if not all_data:
        print("ERROR: No data fetched for any ticker. Check API credentials and network.")
        sys.exit(1)  # nonzero so run_pipeline's retry/notify machinery fires
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
    if not save_training_data(final_df, 'stock'):
        print("ERROR: Could not save training data (both Parquet and CSV "
              "writes failed) — data on disk is STALE")
        sys.exit(1)

    # Summary
    print(f"\nDone! Saved {len(final_df)} rows of stock training data")
    print(f"Stocks harvested: {len(all_data)}/{len(STOCK_TICKERS)}")
    if src_totals:
        total = sum(src_totals.values())
        comp = ', '.join(f"{s} {n} ({n / total:.0%})"
                         for s, n in sorted(src_totals.items(),
                                            key=lambda kv: -kv[1]))
        print(f"Source composition (newly fetched bars): {comp}")
    feature_count, target_cols, tb_cols = _summary_feature_split(final_df.columns)
    print(f"Feature columns: {feature_count}")
    print(f"Target columns: {target_cols + tb_cols}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

    # Validation report
    validate_training_data(final_df, 'stock')


if __name__ == '__main__':
    main()
