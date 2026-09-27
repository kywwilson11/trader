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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from indicators import compute_stock_features
from stock_config import load_stock_universe
from adaptive_config import get_forward_bars_list
from data_sources import fetch_with_fallback
from data_utils import (load_training_data, save_training_data,
                         append_ticker_data, validate_training_data,
                         raw_sidecar_enabled, load_raw_ohlcv, save_raw_ohlcv,
                         merge_raw_ohlcv, find_interior_gaps, _ensure_utc_index,
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
try:
    MINUTE_EDGE_DAYS = int(os.getenv('TRADER_MINUTE_EDGE_DAYS', '120') or '120')
except ValueError:
    MINUTE_EDGE_DAYS = 120


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


# --- R4 (L7 full fix, 2026-09-26): stamp TB labels AFTER every row filter ---
# TB_Bars_{fb} is a POSITIONAL offset in the frame compute_tb_labels walks
# (policy_exits.py caveat), and backtest.simulate_ticker / meta_label replay
# the exit kernel over the STORED rows. So the labels are stamped on exactly
# the rows the store keeps (feature dropna + as-of tradability + as-of
# membership already applied), continued — past the LAST stored row only —
# into every real bar after it (the rows the Target_Return NaN tail drops),
# which reproduces the old series-end behaviour. Interior bars a filter
# removed are therefore invisible to the label walk exactly as they are to
# the backtester (label == backtest, docs/MAP.md §6 invariant 1) and every
# span is a valid row offset for sample_weights uniqueness. Rows whose walk
# never crossed a removed bar keep byte-identical labels.

class TBSpanError(RuntimeError):
    """A post-stamp filter removed INTERIOR rows (TB_Bars_* spans invalid)
    or the stamp precondition failed. Fatal: main() exits 3 before any
    training-store write."""


TB_SPAN_EXIT_CODE = 3
_TB_PRICE_COLS = ('Open', 'High', 'Low', 'Close', 'ATR')

# ticker -> slim OHLC+ATR frame of that ticker's real bars from its first
# stored row on; filled by prepare_stock_data, consumed (and cleared) by
# main()'s post-membership re-stamp. Module-level so prepare_stock_data's
# signature (stubbed by tests) is unchanged.
_TB_WALK_BARS = {}


def _tb_price_frame(df):
    """Slim copy of exactly the columns compute_tb_labels reads."""
    return df[[c for c in _TB_PRICE_COLS if c in df.columns]].copy()


def _stamp_tb_labels(stored, bars, asset_type='stock'):
    """Stamp TB_Ret/Bars/Reason_{fb} onto `stored` — ONE ticker's final
    rows (sorted, unique index) — by walking the exit kernel over those
    rows followed by the bars in `bars` strictly AFTER the last stored
    row. Returns a new frame; rows whose window runs off the continuation
    get NaN (the caller drops them: suffix-only by construction)."""
    if len(stored) == 0:
        return stored
    for name, ix in (('stored', stored.index), ('bars', bars.index)):
        if not (ix.is_unique and ix.is_monotonic_increasing):
            raise TBSpanError(f"TB stamp precondition: {name} index is not "
                              f"sorted+unique (policy_exits caveat)")
    cols = [c for c in _TB_PRICE_COLS
            if c in stored.columns and c in bars.columns]
    tail = bars.loc[bars.index > stored.index[-1], cols]
    df = pd.concat([stored[cols], tail]) if len(tail) else stored[cols]
    from policy_exits import compute_tb_labels
    labels = compute_tb_labels(df, FORWARD_BARS, asset_type)
    n = len(stored)
    out = stored.assign(**{col: np.asarray(vals)[:n]
                           for col, vals in labels.items()})
    # Legacy column order: TB_* right after the last Target_Return_{fb}
    # (where the pre-R4 stamp put them) — the store schema is unchanged.
    rest = [c for c in out.columns if not c.startswith('TB_')]
    tb = [c for c in out.columns if c.startswith('TB_')]
    tr = [i for i, c in enumerate(rest) if c.startswith('Target_Return_')]
    if tb and tr:
        k = tr[-1] + 1
        out = out[rest[:k] + tb + rest[k:]]
    return out


def _tb_cols(columns):
    return [c for c in columns if c.startswith('TB_')]


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

    # Fail loud on a mis-typed bar index (R5: a mixed-tz merge once
    # degraded it to object Index and crashed deep inside indicators).
    _ensure_utc_index(ohlcv, ticker)

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

    # Triple-barrier targets (TB_*) are stamped at the END of this
    # function, AFTER the dropna + tradability mask (R4 / L7 full fix —
    # see _stamp_tb_labels).

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
    # Every real bar (walk-continuation source for the TB stamp), taken
    # BEFORE any row filter.
    walk_bars = _tb_price_frame(df)
    # (1) ALL per-ticker row filters first. TB_* are not stamped yet, so
    # this dropna keeps exactly the rows the old stamp-then-dropna kept
    # (a TB NaN only ever sat on a Target_Return NaN tail row).
    df = df.dropna()
    df = _asof_tradability_mask(df, ticker)
    # (2) Triple-barrier targets matched to the LIVE exit stack (ATR stop
    # / trailing / TP / EOD flatten — the same policy_exits kernel the
    # backtester runs), stamped on the FILTERED rows (R4 / L7 full fix).
    # For stocks the EOD barrier fixes the structural label mismatch: raw
    # Target_Return_12..48 spans 1.8-7.4 trading days while live stock
    # holds are capped at ~6.5h by the 15:50 flatten.
    df = _stamp_tb_labels(df, walk_bars, 'stock')
    _tb_stamp_index = df.index
    tb_cols = _tb_cols(df.columns)
    if tb_cols:
        df = df.dropna(subset=tb_cols)   # suffix-only by construction
    if not _warn_tb_span_violation(_tb_stamp_index, df.index, ticker,
                                   'post-stamp TB-NaN drop'):
        raise TBSpanError(f"{ticker}: interior rows removed after the TB "
                          f"stamp")
    if len(df):
        _TB_WALK_BARS[ticker] = walk_bars.loc[walk_bars.index >= df.index[0]]
    return df


# Daily-window warmup fill — SHARED with the live path (predict_now), single
# source of truth in indicators.py: the harvest keeps warmup rows with the
# same neutral 0.0/0.5 values the live path serves on its short frames.
# Diverging fills here would silently break train/serve parity.
from indicators import (
    WARMUP_FEATURES_ZERO, WARMUP_FEATURES_HALF,
    fill_warmup_features as _fill_warmup_features,
)


# --- R2C-06 (L7) TB-span/filter-ordering guard --------------------------
# policy_exits' caveat: a TB_Bars_* span is invalid as a row offset after
# ANY interior row removal. History: R2C-06 stamped BEFORE the dropna +
# as-of masks and only warned; the 2026-09-26 Jetson re-harvest fired it
# on every stock name (RS_vs_SPY is NaN wherever SPY's 12-bar ROC is
# exactly 0 -> dropna removed ~30 interior bars/name) and the crypto twin
# carried the same defect silently (Volume_Ratio NaN on zero-volume
# stretches). R4 (06 plan deferred item 8) now stamps AFTER every filter
# (_stamp_tb_labels + _restamp_after_membership), so the only post-stamp
# removal left is the suffix TB-NaN drop. These helpers still only
# REPORT (return the ok flag); their callers turn a False into a fatal
# TBSpanError / exit 3 with no store write — a fired guard is now a bug.

def _removals_prefix_suffix_only(pre_index, post_index):
    """(ok, n_interior_gaps): ok iff post_index is ONE contiguous run of
    pre_index — i.e. rows were removed only from the front and/or back.
    Rows in post but not in pre count as violations too. Empty post is
    vacuously ok (no surviving row carries a stale span)."""
    if len(post_index) == 0:
        return True, 0
    pos = pre_index.get_indexer(post_index)
    if (pos < 0).any():
        return False, int((pos < 0).sum())
    gaps = int(((pos[1:] - pos[:-1]) != 1).sum())
    return gaps == 0, gaps


def _warn_tb_span_violation(pre_index, post_index, ticker, stage):
    """Loud [TB-GUARD] warning when a filter removed INTERIOR rows after
    TB stamping. Returns the ok flag so callers/tests can assert on it.

    Fail-soft on the check itself (e.g. get_indexer refuses a non-unique
    per-ticker index): a broken CHECK must not kill the harvest, and it is
    not evidence of a violation — it prints its own distinct line and
    returns True (no L7 trigger)."""
    try:
        ok, n_bad = _removals_prefix_suffix_only(pre_index, post_index)
    except Exception as e:  # measurement check must never kill a harvest
        print(f"  [TB-GUARD] {ticker}: {stage} span check itself failed "
              f"({type(e).__name__}: {e}) — check skipped, NOT a "
              f"violation.")
        return True
    if not ok:
        print(f"  [TB-GUARD] {ticker}: {stage} removed INTERIOR rows "
              f"({n_bad} discontinuities) after TB stamping — TB_Bars_* "
              f"positional spans would be INVALID for this name "
              f"(policy_exits.py caveat). R4 stamps after every row "
              f"filter, so this is a harvest BUG — the caller aborts "
              f"with no store write (report to owner).")
    return ok


def _tb_membership_guard(pre_tickers, post_tickers):
    """Per-ticker prefix/suffix check across the cross-sectional
    membership mask. pre_tickers/post_tickers: one-column 'Ticker'
    frames (index + ticker) captured before/after the mask. A ticker
    removed entirely is fine (no surviving rows carry spans)."""
    ok_all = True
    for t in post_tickers['Ticker'].unique():
        pre_idx = pre_tickers.index[pre_tickers['Ticker'] == t]
        post_idx = post_tickers.index[post_tickers['Ticker'] == t]
        ok_all &= _warn_tb_span_violation(pre_idx, post_idx, t,
                                          'as-of membership mask')
    return ok_all


def _restamp_after_membership(final_df, pre_tickers, walk_bars):
    """Re-stamp TB_* for every ticker the cross-sectional membership mask
    removed rows from (R4 / L7 full fix), on its surviving member rows +
    its real bars after the last member row (walk_bars[ticker], see
    _TB_WALK_BARS). Tickers the mask left untouched keep their
    prepare-time stamp (identical by construction). Prefix-only removal
    re-stamps to identical values; suffix/interior removal is where the
    stored-row walk changes. Rows left with a NaN TB label (window ran off
    the continuation — not expected) are dropped; the caller's final
    guard verifies that stayed suffix-only. Raises TBSpanError when an
    interior-hit ticker has no walk bars to re-stamp from."""
    tb_cols = _tb_cols(final_df.columns)
    if not tb_cols or final_df.empty:
        return final_df
    tick = final_df['Ticker'].to_numpy()
    pre_tick = pre_tickers['Ticker'].to_numpy()
    arrays = None
    n_interior = n_other = 0
    for t in pd.unique(tick):
        m = tick == t
        post_idx = final_df.index[m]
        pre_idx = pre_tickers.index[pre_tick == t]
        if len(post_idx) == len(pre_idx):
            continue           # the mask only removes rows: untouched
        ok, n_bad = _removals_prefix_suffix_only(pre_idx, post_idx)
        bars = walk_bars.get(t)
        if bars is None:
            if not ok:
                raise TBSpanError(f"{t}: membership removed interior rows "
                                  f"and no walk bars exist to re-stamp")
            continue
        if arrays is None:
            arrays = {c: final_df[c].to_numpy(dtype=np.float64, copy=True)
                      for c in tb_cols}
        cols = [c for c in _TB_PRICE_COLS if c in final_df.columns]
        stored = pd.DataFrame({c: final_df[c].to_numpy()[m] for c in cols},
                              index=post_idx)
        stamped = _stamp_tb_labels(stored, bars, 'stock')
        for c in tb_cols:
            if c in stamped.columns:
                arrays[c][m] = stamped[c].to_numpy(dtype=np.float64)
        if ok:
            n_other += 1
        else:
            n_interior += 1
            print(f"  [TB-RESTAMP] {t}: membership mask removed interior "
                  f"rows ({n_bad} discontinuities) — TB_* re-stamped on "
                  f"the member rows")
    if arrays is None:
        return final_df
    for c in tb_cols:
        final_df[c] = arrays[c]
    nan_rows = final_df[tb_cols].isna().any(axis=1).to_numpy()
    if nan_rows.any():
        print(f"  [TB-RESTAMP] dropping {int(nan_rows.sum())} row(s) whose "
              f"re-stamped window ran off the data")
        final_df = final_df[~nan_rows]
    print(f"[TB-RESTAMP] membership re-stamp: {n_interior} ticker(s) with "
          f"interior removals, {n_other} with prefix/suffix-only removals")
    return final_df


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
    _TB_WALK_BARS.clear()
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

        try:
            stock_df = prepare_stock_data(t, spy_close, api=api,
                                           existing_ohlcv=existing_ohlcv,
                                           start_date=start,
                                           src_totals=src_totals,
                                           raw_out=(raw_out if use_sidecar
                                                    else None))
        except TBSpanError as e:
            print(f"FATAL [TB-GUARD] {e} — aborting, NO training store "
                  f"written")
            sys.exit(TB_SPAN_EXIT_CODE)
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
    # R2C-06 (L7): capture index+Ticker only (cheap) so the TB-span guard
    # can verify the mask removed prefix/suffix rows only, per ticker.
    # R4: the mask can remove interior DAYS for names near the top-K cut,
    # so every ticker it touched is re-stamped on its member rows; the
    # post-re-stamp index is then the reference the final guard (just
    # before the save) checks nothing downstream removed interior rows.
    _pre_member = final_df[['Ticker']].copy()
    final_df = _asof_membership_mask(final_df)
    try:
        final_df = _restamp_after_membership(final_df, _pre_member,
                                             _TB_WALK_BARS)
    except TBSpanError as e:
        print(f"FATAL [TB-GUARD] {e} — aborting, NO training store written")
        sys.exit(TB_SPAN_EXIT_CODE)
    _TB_WALK_BARS.clear()
    _pre_member = final_df[['Ticker']].copy()

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
        from sentiment_history import (fetch_stock_sentiment_history,
                                       stock_sentiment_lookup_dates)
        start_date = str((final_df.index.min() - pd.Timedelta(days=2)).date())
        end_date = str(final_df.index.max().date())
        sentiment = fetch_stock_sentiment_history(
            STOCK_TICKERS, start_date, end_date, cached_only=True)
        # Key = ((t - 6h).date() - 1 day): legacy articles are Chicago-dated,
        # so a bar before 06:00 UTC (the 00:00 UTC extended-hours bar) must
        # fall back one more day to stay strictly in the past (H audit).
        final_df['Daily_Sentiment'] = [
            sentiment.get((ticker, key), 0.0)
            for ticker, key in zip(final_df['Ticker'],
                                   stock_sentiment_lookup_dates(final_df.index))
        ]
        filled = sum(1 for v in final_df['Daily_Sentiment'] if v != 0.0)
        print(f"Daily_Sentiment (lagged 1d): {filled}/{len(final_df)} bars have sentiment")
    except Exception as e:
        print(f"WARNING: Could not load stock sentiment history: {e}")
        final_df['Daily_Sentiment'] = 0.0

    # R4 final TB-span guard (FAIL LOUD): per ticker, the rows about to be
    # saved must be one contiguous run of the rows the TB labels were last
    # stamped on — otherwise TB_Bars_* spans are invalid row offsets.
    if not _tb_membership_guard(_pre_member, final_df[['Ticker']]):
        print("FATAL [TB-GUARD] rows were removed from the interior of a "
              "ticker after the TB stamp — aborting, NO training store "
              "written")
        sys.exit(TB_SPAN_EXIT_CODE)
    del _pre_member

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
