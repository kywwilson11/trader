"""Forward-looking volatility forecasts (HAR-RV primary, EGARCH/GARCH fallback)
for volatility-targeted position sizing.

Adapts to changing market conditions, unlike ATR which is purely
backward-looking. Every forecaster here returns the 1-step PER-BAR (hourly)
sigma as a decimal, or None on failure (short series, convergence, missing
deps) — there is no ATR fallback in this module; callers keep their ATR-based
stops and base sizing when they get None. Annualize via
sqrt(BARS_PER_YEAR[asset_type]) before comparing against annual vol numbers
(options_overlay.iv_from_har expects an ANNUAL rv sigma, not this module's
raw per-bar output).

Based on Engle (2003 Nobel) ARCH/GARCH and Corsi (2009) HAR-RV.
"""

import os
import time
from log_config import get_logger
import numpy as np

logger = get_logger(__name__)

# Cache fitted models per symbol (refit hourly, not every 30s cycle)
_model_cache: dict[str, tuple[object, float]] = {}
_REFIT_INTERVAL = 3600  # seconds
_arch_warned = False    # missing-'arch' warning fires once, not per fit call


def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1):
    """Fit a GARCH(p,q) model to a return series.

    Args:
        returns: Array of percentage returns (e.g. daily or hourly)
        p: GARCH lag order (default 1)
        q: ARCH lag order (default 1)

    Returns:
        Fitted arch model result, or None on failure.
    """
    if len(returns) < 100:
        logger.debug("GARCH: insufficient data (%d points, need 100)", len(returns))
        return None

    try:
        from arch import arch_model
    except ImportError as e:
        # A missing dep silently disabling GARCH forever must be visible:
        # WARNING once (not debug-per-call), then quiet.
        global _arch_warned
        if not _arch_warned:
            logger.warning("GARCH disabled — 'arch' package not installed "
                           "(%s); get_sigma will rely on HAR-RV only", e)
            _arch_warned = True
        return None

    try:
        # Scale returns to percentage if they look like decimals
        if np.abs(returns).mean() < 0.01:
            returns = returns * 100

        # Try EGARCH first (log-variance spec; symmetric here — o=0, no leverage/asymmetry term fit)
        try:
            am = arch_model(returns, vol='EGARCH', p=p, q=q, mean='Zero',
                            rescale=False)
            result = am.fit(disp='off', show_warning=False)
            return result
        except Exception:
            pass
        # Fallback to standard GARCH
        am = arch_model(returns, vol='Garch', p=p, q=q, mean='Zero',
                        rescale=False)
        result = am.fit(disp='off', show_warning=False)
        return result
    except Exception as e:
        logger.debug("GARCH fit failed: %s", e)
        return None


def forecast_volatility(model_result) -> float | None:
    """Forecast 1-step-ahead conditional volatility (sigma).

    Returns:
        PER-BAR (hourly) sigma as a decimal (e.g. 0.008 = 0.8% per bar),
        or None. NOT annualized — multiply by sqrt(BARS_PER_YEAR) first
        where an annual figure is needed (e.g. options_overlay.iv_from_har).
    """
    if model_result is None:
        return None
    try:
        forecasts = model_result.forecast(horizon=1)
        variance = forecasts.variance.values[-1, 0]
        if variance <= 0:
            return None
        # Model was fit on percentage returns, so sigma is in percentage points
        sigma_pct = np.sqrt(variance)
        return sigma_pct / 100.0  # Convert to decimal
    except Exception as e:
        logger.debug("GARCH forecast failed: %s", e)
        return None


# DEAD in live paths: stops are ATR-based (base_loop) and this floor/ceil
# does not track strategy_config stop policy. Kept only because
# base_loop.py:40 still imports the name — delete both together.
def get_garch_stop(entry_price: float, sigma: float, multiplier: float = 2.0,
                   floor_pct: float = 0.03, ceil_pct: float = 0.10) -> float:
    """Compute stop-loss price using GARCH volatility.

    Args:
        entry_price: Entry price
        sigma: GARCH sigma (decimal, e.g. 0.02 = 2%)
        multiplier: Number of sigmas for stop distance
        floor_pct: Minimum stop distance as fraction of price
        ceil_pct: Maximum stop distance as fraction of price

    Returns:
        Stop price (below entry for long positions).
    """
    stop_dist = max(floor_pct, min(ceil_pct, sigma * multiplier))
    return entry_price * (1 - stop_dist)


def get_cached_sigma(symbol: str, returns: np.ndarray) -> float | None:
    """Get GARCH sigma for a symbol, using cache to avoid refitting every cycle.

    Args:
        symbol: Trading symbol
        returns: Recent return series (percentage)

    Returns:
        Sigma (decimal) or None if fitting fails.
    """
    now = time.time()

    if symbol in _model_cache:
        cached_result, cached_time = _model_cache[symbol]
        if now - cached_time < _REFIT_INTERVAL:
            sigma = forecast_volatility(cached_result)
            if sigma is not None:
                return sigma

    # Fit new model
    result = fit_garch(returns)
    if result is not None:
        _model_cache[symbol] = (result, now)
        sigma = forecast_volatility(result)
        if sigma is not None:
            logger.debug("GARCH %s: sigma=%.4f", symbol, sigma)
            return sigma

    return None


BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
BARS_PER_DAY = {'crypto': 24.0, 'stock': 6.5}


# --- HAR-RV on realized range (Corsi 2009; HARQ insanity filter from
# Bollerslev-Patton-Quaedvlieg 2016). Wave-4, red-team conf 5/5:
# realized measures OBSERVE variance from intraday data instead of
# inferring it by MLE from daily closes — a 3-regressor OLS then beats
# GARCH(1,1) on RV forecasting near-universally (Hansen-Lunde's "nothing
# beats GARCH" verdict only holds for close-to-close data). ---

_HAR_MIN_DAYS = 60
_HAR_WINDOW = 250
_har_cache: dict[str, tuple[object, float]] = {}   # symbol -> (day, sigma)
_har_gap_logged: set = set()   # symbols already warned about thin HAR history


def daily_realized_range(bars) -> 'pd.Series':
    """Daily Parkinson realized variance from intraday High/Low bars:
    RRV_d = sum_i ln(H_i/L_i)^2 / (4 ln 2), in squared-decimal units."""
    import pandas as pd
    hl = np.log(bars['High'] / bars['Low']) ** 2 / (4.0 * np.log(2.0))
    hl = hl.replace([np.inf, -np.inf], np.nan).dropna()
    return hl.groupby(hl.index.normalize()).sum()


def _har_sigma_from_rrv(rrv, asset_type: str = 'stock', shrink: bool = False,
                        c_scale: float = 1.0) -> float | None:
    """HAR-RV fit + forecast on an already-built daily RRV Series. The
    legacy path (har_forecast_sigma) calls this with the defaults —
    byte-identical math. Two OPT-IN branches, exercised only by the
    TRADER_HAR_DAILY_FEED feed path (B11 binding: keep log-HAR; no WLS,
    no HARQ):
      shrink=True — ridge-to-prior on the OLS betas, lam = n/(n+120):
        slopes shrink toward Corsi consensus (0.40, 0.30, 0.25), the
        intercept toward the value that keeps the sample mean; effect
        vanishes by n~250.
      c_scale — Hansen-Lunde level correction applied AFTER the BPQ
        clamp: sigma_daily = sqrt(c_scale * rrv_hat_clamped). Per-bar
        division by sqrt(BARS_PER_DAY) unchanged (keeps the sqrt(6.5) /
        compute_vol_adjusted_size sqrt(1638) consistency).
    """
    import pandas as pd
    try:
        rrv = rrv[rrv > 0].tail(_HAR_WINDOW)
        if len(rrv) < _HAR_MIN_DAYS:
            return None
        m5 = rrv.rolling(5, min_periods=5).mean()
        m22 = rrv.rolling(22, min_periods=22).mean()
        df = pd.DataFrame({'y': np.log(rrv).shift(-1), 'x1': np.log(rrv),
                           'x2': np.log(m5), 'x3': np.log(m22)}).dropna()
        # At exactly 60 complete days m22 leaves 38 regression rows and
        # 38 < (60 - 22) is False — the guard passes, no off-by-one.
        if len(df) < _HAR_MIN_DAYS - 22:
            return None
        X = np.column_stack([np.ones(len(df)), df['x1'], df['x2'], df['x3']])
        beta, *_ = np.linalg.lstsq(X, df['y'].values, rcond=None)
        if shrink:
            n = len(df)
            lam = n / (n + 120.0)
            prior_intercept = (float(df['y'].mean())
                               - 0.40 * float(df['x1'].mean())
                               - 0.30 * float(df['x2'].mean())
                               - 0.25 * float(df['x3'].mean()))
            beta_prior = np.array([prior_intercept, 0.40, 0.30, 0.25])
            beta = lam * beta + (1.0 - lam) * beta_prior
        resid = df['y'].values - X @ beta
        sig2_resid = float(np.var(resid, ddof=4)) if len(df) > 8 else 0.0
        x_now = np.array([1.0, np.log(rrv.iloc[-1]),
                          np.log(m5.iloc[-1]), np.log(m22.iloc[-1])])
        rrv_hat = float(np.exp(x_now @ beta + 0.5 * sig2_resid))
        rrv_hat = min(max(rrv_hat, float(rrv.min())), float(rrv.max()))
        sigma_daily = np.sqrt(c_scale * rrv_hat)
        return float(sigma_daily / np.sqrt(BARS_PER_DAY.get(asset_type, 6.5)))
    except Exception as e:
        logger.debug("HAR forecast failed: %s", e)
        return None


def har_forecast_sigma(bars, asset_type: str = 'stock') -> float | None:
    """Next-day HAR-RV sigma forecast as a PER-BAR decimal (the same
    units forecast_volatility returns), or None when history is thin.

    log RRV_{d+1} ~ c + log RRV_d + log mean(RRV,5d) + log mean(RRV,22d)
    fit by rolling OLS over <=250 days; log-normal bias correction;
    forecast clamped to the estimation window's [min, max] (the BPQ
    'insanity filter' — one bad range print must not triple the stop).
    """
    try:
        rrv = daily_realized_range(bars)
        return _har_sigma_from_rrv(rrv, asset_type)
    except Exception as e:
        logger.debug("HAR forecast failed: %s", e)
        return None


def har_daily_feed_enabled() -> bool:
    """TRADER_HAR_DAILY_FEED (D30), delegated to market_data's call-time
    reader. Fallback False keeps volatility standalone-importable (no
    import cycle: market_data does not import volatility)."""
    try:
        from market_data import har_daily_feed_enabled as _mde
        return _mde()
    except Exception:
        return False


# --- D30 feed-path state: per-symbol crypto complete-day RRV store ---
# Live crypto frames are ~10 days deep — never the >=60 complete days HAR
# needs — so complete-day RRVs accumulate in a persisted JSON
# ({symbol: {ISO-date: rrv}}, window capped at _HAR_WINDOW), reusing the
# Wave B-3 completed-day merge semantics. BTC seeds one-time from
# crypto_rv_history.json (same estimator, same completed-day semantics —
# values identical by construction; read-only on the B-3 file). Stocks
# need no store: their RRVs come from market_data's daily-bars cache.
_HAR_RRV_FILE = os.path.join(os.path.dirname(__file__),
                             'har_rrv_history.json')
_har_rrv_store = {'loaded': False, 'symbols': {}}


def _har_rrv_load() -> None:
    """Load (and BTC-seed) the per-symbol HAR RRV store. Corrupt/missing
    -> fresh start (warn on corruption). Never raises."""
    import json
    syms = {}
    try:
        with open(_HAR_RRV_FILE, 'r') as f:
            data = json.load(f)
        for sym, days in (data or {}).items():
            d = {}
            for k, v in (days or {}).items():
                v = float(v)
                if np.isfinite(v) and v > 0:
                    d[str(k)] = v
            syms[str(sym)] = d
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("[HAR-RRV] history file corrupt (%s) — starting "
                       "empty", e)
        syms = {}
    try:
        btc = syms.get(_CRYPTO_RV_SOURCE, {})
        if len(btc) < _HAR_MIN_DAYS and os.path.exists(_CRYPTO_RV_FILE):
            seeded = dict(_rv_load()['rrv'])
            seeded.update(btc)   # stored values win over the seed
            if len(seeded) > _HAR_WINDOW:
                for k in sorted(seeded)[:-_HAR_WINDOW]:
                    del seeded[k]
            if len(seeded) > len(btc):
                logger.info("[HAR-RRV] seeded %s with %d complete days from "
                            "crypto_rv_history.json", _CRYPTO_RV_SOURCE,
                            len(seeded) - len(btc))
            syms[_CRYPTO_RV_SOURCE] = seeded
    except Exception as e:
        logger.warning("[HAR-RRV] BTC seed failed: %s", e)
    _har_rrv_store['symbols'] = syms
    _har_rrv_store['loaded'] = True


def _har_rrv_save() -> None:
    """Atomic persist (tmp -> os.replace, _rv_save pattern). Never raises."""
    import json
    try:
        tmp = _HAR_RRV_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(_har_rrv_store['symbols'], f)
        os.replace(tmp, _HAR_RRV_FILE)
    except Exception as e:
        logger.warning("[HAR-RRV] history save failed: %s", e)


def _merge_complete_day_rrvs(history: dict, bars, window_days: int,
                             min_bars: int | None = None) -> bool:
    """Merge COMPLETE-day Parkinson RRVs from an hourly frame into
    `history` ({ISO-date: rrv}): every date except the frame's LAST
    calendar day (forming), trimmed to the trailing `window_days` by date.
    Dict overwrite keeps the self-correcting refresh (a later fuller frame
    fixes an earlier thin day). With min_bars=N, days with fewer than N
    bars are additionally skipped — protects the HAR store's HEAD day
    after downtime, where a partial first day would leave the 250-bar
    frame and freeze undercounted. min_bars=None is byte-identical to the
    Wave B-3 inline logic (update_crypto_rv_state). Returns True when
    `history` changed."""
    changed = False
    rrv = daily_realized_range(bars)
    last_day = bars.index[-1].normalize()
    counts = (bars['High'].groupby(bars.index.normalize()).count()
              if min_bars is not None else None)
    for day, val in rrv.items():
        if day != last_day and np.isfinite(val) and val > 0:
            if counts is not None and int(counts.get(day, 0)) < min_bars:
                continue
            key = day.date().isoformat()
            fval = float(val)
            if history.get(key) != fval:
                history[key] = fval
                changed = True
    if len(history) > window_days:
        for k in sorted(history)[:-window_days]:
            del history[k]
        changed = True
    return changed


def _har_feed_sigma(symbol: str, bars, asset_type: str) -> float | None:
    """D30 feed path (TRADER_HAR_DAILY_FEED only): HAR on a COMPLETE-day
    RRV series — crypto from the persisted per-symbol store, stocks from
    market_data's daily-bars cache — with prior shrinkage and (stocks) the
    Hansen-Lunde c level correction. Because the series holds complete
    days only, the partial-day regressor bug is structurally fixed here
    (x_now's rrv.iloc[-1] = last complete day) and the cache is keyed on
    the last COMPLETE day, not the forming one. Returns None to fall
    through to the unchanged legacy path."""
    import pandas as pd
    c_scale = 1.0
    if asset_type == 'crypto':
        if not _har_rrv_store['loaded']:
            _har_rrv_load()
        hist = _har_rrv_store['symbols'].setdefault(symbol, {})
        if _merge_complete_day_rrvs(hist, bars, _HAR_WINDOW, min_bars=20):
            _har_rrv_save()
        if len(hist) < _HAR_MIN_DAYS:
            return None
        days = sorted(hist)
        rrv = pd.Series([hist[d] for d in days], index=pd.DatetimeIndex(days))
        rrv = rrv[rrv > 0]
    else:
        from market_data import load_daily_bars
        daily = load_daily_bars(symbol)
        if daily is None or len(daily) == 0:
            return None
        # Whole-day Parkinson from cached complete daily bars — an
        # internally consistent series (differs from the hourly-sum
        # estimator by construction; corrected jointly with the overnight
        # omission by c below).
        rrv = np.log(daily['High'] / daily['Low']) ** 2 / (4.0 * np.log(2.0))
        rrv = rrv.replace([np.inf, -np.inf], np.nan).dropna()
        rrv = rrv[rrv > 0].tail(_HAR_WINDOW)
        if len(rrv) >= _HAR_MIN_DAYS:
            # Hansen-Lunde (2005) c via Martens scaling (B11 BINDING,
            # stocks only; crypto c = 1.0 identically): over the same
            # trailing window, c = sum((r_cc - mean)^2) / sum(RRV_d),
            # clamped to [1.0, 2.5] (expected 1.25-1.5). Recomputed at
            # each fit — cheap at <=320 rows, and rrv only changes at the
            # daily cache roll anyway.
            r_cc = np.log(daily['Close']).diff().reindex(rrv.index).dropna()
            denom = float(rrv.reindex(r_cc.index).sum())
            if len(r_cc) >= _HAR_MIN_DAYS and denom > 0:
                num = float(((r_cc - r_cc.mean()) ** 2).sum())
                c_scale = min(max(num / denom, 1.0), 2.5)
    if len(rrv) < _HAR_MIN_DAYS:
        return None
    last_complete = rrv.index[-1]
    hit = _har_cache.get(symbol)
    if hit is not None and hit[0] == last_complete:
        return hit[1]
    sigma = _har_sigma_from_rrv(rrv, asset_type, shrink=True, c_scale=c_scale)
    if sigma is not None and sigma > 0:
        _har_cache[symbol] = (last_complete, sigma)
        logger.debug("HAR-feed %s: sigma=%.4f (c=%.2f, n=%dd)",
                     symbol, sigma, c_scale, len(rrv))
        return sigma
    return None


# ALWAYS-ON diagnostic (D30 Jetson before/after evidence, no flag): count
# get_sigma's three outcome classes, log + reset once per hour.
_sigma_srcs = {'har': 0, 'garch': 0, 'none': 0}
_sigma_srcs_logged = time.monotonic()


def _sigma_src(kind: str) -> None:
    global _sigma_srcs_logged
    try:
        _sigma_srcs[kind] += 1
        now = time.monotonic()
        if now - _sigma_srcs_logged >= 3600.0:
            logger.info("[VOL] sigma sources last hour: har=%d garch=%d "
                        "none=%d", _sigma_srcs['har'], _sigma_srcs['garch'],
                        _sigma_srcs['none'])
            for k in _sigma_srcs:
                _sigma_srcs[k] = 0
            _sigma_srcs_logged = now
    except Exception:
        pass


def get_sigma(symbol: str, returns: np.ndarray, bars=None,
              asset_type: str = 'stock') -> float | None:
    """Per-bar sigma: HAR-RV from intraday ranges when enough bars
    exist, EGARCH/GARCH on returns otherwise. One day of cache per
    symbol (daily RRV only changes at the day roll).

    Output is PER-BAR (hourly), NOT annual — annualize via
    sqrt(BARS_PER_YEAR[asset_type]) before use as an annual rv sigma
    (e.g. options_overlay.iv_from_har)."""
    try:
        from strategy_config import HAR_VOL_ENABLED
    except ImportError:
        HAR_VOL_ENABLED = True
    if HAR_VOL_ENABLED and bars is not None and len(bars) > 0:
        # D30 feed path (default OFF): complete-day RRV series deep enough
        # for HAR. Any failure falls through to the UNCHANGED legacy code.
        if har_daily_feed_enabled():
            try:
                sigma = _har_feed_sigma(symbol, bars, asset_type)
                if sigma is not None:
                    _sigma_src('har')
                    return sigma
            except Exception as e:
                logger.debug("[VOL] HAR daily feed failed (%s) — legacy "
                             "path", e)
        day = bars.index[-1].date() if hasattr(bars.index[-1], 'date') else None
        hit = _har_cache.get(symbol)
        if hit is not None and hit[0] == day:
            _sigma_src('har')
            return hit[1]
        sigma = har_forecast_sigma(bars, asset_type)
        if sigma is not None and sigma > 0:
            _har_cache[symbol] = (day, sigma)
            logger.debug("HAR %s: sigma=%.4f", symbol, sigma)
            _sigma_src('har')
            return sigma
        if symbol not in _har_gap_logged:
            _har_gap_logged.add(symbol)
            logger.info("[VOL] %s: HAR-RV unavailable (need >=%d daily obs) — "
                        "falling back to GARCH/cached sigma", symbol, _HAR_MIN_DAYS)
    g = get_cached_sigma(symbol, returns)
    _sigma_src('garch' if g is not None else 'none')
    return g


# --- BTC trailing-RV regime state (c26 S3 / 02_research B06) ---
# VIX is an equity fear gauge; the crypto book de-risks on BTC's OWN daily
# Parkinson realized-range percentile vs a TRAILING window (BTC RV declines
# structurally — never expanding). Live bars are ~10 days deep, so complete-day
# RRVs accumulate in a persisted JSON; below CRYPTO_RV_MIN_HISTORY_DAYS the
# state is 'unknown' and the multiplier fails OPEN to 1.0. The "today" value is
# the trailing-24-hourly-bar realized range (same estimator on a rolling day —
# avoids the partial-day artifact that would otherwise fake a calm print every
# UTC midnight). State + history survive restarts in one atomic file.
# Jetson seeding (optional, one-time): call update_crypto_rv_state('BTC/USD',
# <1Y hourly BTC frame from the harvest parquet>) to backfill the window.

_CRYPTO_RV_SOURCE = 'BTC/USD'
_CRYPTO_RV_FILE = os.path.join(os.path.dirname(__file__),
                               'crypto_rv_history.json')
_CRYPTO_RV_WINDOW_DAYS = 365
_CRYPTO_RV_STALE_SEC = 86400
_crypto_rv = {'state': 'unknown', 'exit_count': 0, 'last_bar_ts': None,
              'updated_mono': None, 'pctile': None, 'history': None}


def _reset_crypto_rv_state():
    """Test seam: clear the in-memory state (does NOT delete the file)."""
    _crypto_rv.update({'state': 'unknown', 'exit_count': 0,
                       'last_bar_ts': None, 'updated_mono': None,
                       'pctile': None, 'history': None})


def _rv_consts() -> dict:
    """B06 state-machine constants; strategy_config with hardcoded fallbacks."""
    try:
        from strategy_config import (
            CRYPTO_RV_ENTER_HIGH_PCT, CRYPTO_RV_ENTER_CRISIS_PCT,
            CRYPTO_RV_EXIT_HIGH_PCT, CRYPTO_RV_EXIT_CRISIS_PCT,
            CRYPTO_RV_EXIT_HOLD_EVALS, CRYPTO_RV_MIN_HISTORY_DAYS,
            CRYPTO_RV_HIGH_MULT, CRYPTO_RV_CRISIS_MULT)
        return {'enter_high': CRYPTO_RV_ENTER_HIGH_PCT,
                'enter_crisis': CRYPTO_RV_ENTER_CRISIS_PCT,
                'exit_high': CRYPTO_RV_EXIT_HIGH_PCT,
                'exit_crisis': CRYPTO_RV_EXIT_CRISIS_PCT,
                'exit_hold': CRYPTO_RV_EXIT_HOLD_EVALS,
                'min_days': CRYPTO_RV_MIN_HISTORY_DAYS,
                'high_mult': CRYPTO_RV_HIGH_MULT,
                'crisis_mult': CRYPTO_RV_CRISIS_MULT}
    except Exception:
        return {'enter_high': 80.0, 'enter_crisis': 95.0, 'exit_high': 65.0,
                'exit_crisis': 90.0, 'exit_hold': 12, 'min_days': 90,
                'high_mult': 0.5, 'crisis_mult': 0.3}


def _rv_load() -> dict:
    """Load the persisted RRV history + state. Corrupt/missing -> fresh start
    (warn on corruption). Never raises."""
    import json
    fresh = {'rrv': {}, 'state': 'unknown', 'exit_count': 0,
             'last_bar_ts': None}
    try:
        with open(_CRYPTO_RV_FILE, 'r') as f:
            data = json.load(f)
        rrv = {}
        for k, v in (data.get('rrv') or {}).items():
            v = float(v)
            if np.isfinite(v) and v > 0:
                rrv[str(k)] = v
        return {'rrv': rrv,
                'state': str(data.get('state', 'unknown')),
                'exit_count': int(data.get('exit_count', 0)),
                'last_bar_ts': data.get('last_bar_ts')}
    except FileNotFoundError:
        return fresh
    except Exception as e:
        logger.warning("[CRYPTO-RV] history file corrupt (%s) — starting "
                       "empty", e)
        return fresh


def _rv_save() -> None:
    """Atomic persist (tmp -> os.replace, hard_stop_lockout pattern).
    Never raises."""
    import json
    try:
        payload = {'rrv': _crypto_rv['history'] or {},
                   'state': _crypto_rv['state'],
                   'exit_count': _crypto_rv['exit_count'],
                   'last_bar_ts': _crypto_rv['last_bar_ts']}
        tmp = _CRYPTO_RV_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, _CRYPTO_RV_FILE)
    except Exception as e:
        logger.warning("[CRYPTO-RV] history save failed: %s", e)


def update_crypto_rv_state(symbol: str, bars) -> None:
    """Accumulate BTC daily Parkinson RRVs + advance the B06 regime state.

    No-op unless `symbol` is the source ('BTC/USD') and bars is an hourly
    High/Low frame on a DatetimeIndex with >= 25 rows. Enters (high/crisis)
    are immediate; exits require a NEW final bar timestamp (and, for
    high -> normal, CRYPTO_RV_EXIT_HOLD_EVALS consecutive calm new-bar
    evaluations). Never raises; failures leave the state unchanged.
    """
    try:
        if symbol != _CRYPTO_RV_SOURCE or bars is None:
            return
        if len(bars) < 25 or 'High' not in bars or 'Low' not in bars:
            return
        if not hasattr(bars.index, 'normalize'):
            return  # needs a DatetimeIndex
        c = _rv_consts()
        if _crypto_rv['history'] is None:
            saved = _rv_load()
            _crypto_rv['history'] = saved['rrv']
            _crypto_rv['state'] = saved['state']
            _crypto_rv['exit_count'] = saved['exit_count']
            _crypto_rv['last_bar_ts'] = saved['last_bar_ts']
        history = _crypto_rv['history']
        # (1) merge COMPLETE days (every date except the frame's last
        # calendar day), then trim to the trailing window by date —
        # min_bars=None keeps this byte-identical to the original inline
        # Wave B-3 logic.
        _merge_complete_day_rrvs(history, bars, _CRYPTO_RV_WINDOW_DAYS)
        # (2) "today" = trailing-24-bar realized range (rolling day; same
        # estimator as daily_realized_range, partial-day artifact avoided).
        hl = np.log(np.asarray(bars['High'].iloc[-24:], dtype=float)
                    / np.asarray(bars['Low'].iloc[-24:], dtype=float)) ** 2
        hl = hl[np.isfinite(hl)]
        current = float(hl.sum() / (4.0 * np.log(2.0)))
        if not np.isfinite(current):
            logger.warning("[CRYPTO-RV] non-finite current RRV — "
                           "state unchanged")
            return
        # (3) not enough complete days yet -> 'unknown' (fail-OPEN 1.0)
        if len(history) < c['min_days']:
            _crypto_rv['state'] = 'unknown'
            _crypto_rv['pctile'] = None
            _crypto_rv['last_bar_ts'] = bars.index[-1].isoformat()
            _crypto_rv['updated_mono'] = time.monotonic()
            _rv_save()
            return
        # (4) trailing percentile of "today" vs the complete-day history
        vals = np.fromiter(history.values(), dtype=float)
        pctile = float(100.0 * np.mean(vals < current))
        # (5) state machine: enter immediately, exit slowly (new bars only)
        new_bar = bars.index[-1].isoformat() != _crypto_rv['last_bar_ts']
        prev = _crypto_rv['state']
        state = prev
        if pctile > c['enter_crisis']:
            state = 'crisis'
            _crypto_rv['exit_count'] = 0
        elif pctile > c['enter_high'] and state != 'crisis':
            state = 'high'
            _crypto_rv['exit_count'] = 0
        elif new_bar:
            if state == 'crisis' and pctile < c['exit_crisis']:
                state = 'high'
                _crypto_rv['exit_count'] = 0
            elif state == 'high':
                if pctile < c['exit_high']:
                    _crypto_rv['exit_count'] += 1
                    if _crypto_rv['exit_count'] >= c['exit_hold']:
                        state = 'normal'
                        _crypto_rv['exit_count'] = 0
                else:
                    _crypto_rv['exit_count'] = 0
        if state == 'unknown':
            state = 'normal'   # sufficient history, nothing elevated
        if state != prev:
            logger.info("[CRYPTO-RV] %s -> %s (pctile %.1f, n=%dd)",
                        prev, state, pctile, len(history))
        # (6) stamp + persist
        _crypto_rv['state'] = state
        _crypto_rv['pctile'] = pctile
        _crypto_rv['last_bar_ts'] = bars.index[-1].isoformat()
        _crypto_rv['updated_mono'] = time.monotonic()
        _rv_save()
    except Exception as e:
        logger.warning("[CRYPTO-RV] state update failed: %s", e)


def get_crypto_rv_mult() -> tuple:
    """(mult, state, pctile) — pure in-memory read, no I/O, never raises.

    Fails OPEN to 1.0 when the state is stale (no update within
    _CRYPTO_RV_STALE_SEC), 'unknown', or 'normal'. Applied to sizing only
    under strategy_config.DERISK_STACK_V2 (base_loop journals it as shadow
    while the flag is OFF)."""
    try:
        c = _rv_consts()
        state = _crypto_rv['state']
        pctile = _crypto_rv['pctile']
        um = _crypto_rv['updated_mono']
        if um is None or (time.monotonic() - um) > _CRYPTO_RV_STALE_SEC:
            return (1.0, 'stale', pctile)
        if state == 'high':
            return (c['high_mult'], 'high', pctile)
        if state == 'crisis':
            return (c['crisis_mult'], 'crisis', pctile)
        return (1.0, state, pctile)
    except Exception:
        return (1.0, 'error', None)


def compute_vol_adjusted_size(base_notional: float, sigma: float,
                              asset_type: str = 'crypto') -> float:
    """Volatility-targeted position sizing (Moreira & Muir 2017).

    Adjusts notional so each position targets approximately the same
    dollar volatility. Higher vol assets get smaller positions.

    The target is derived from the ANNUALIZED portfolio vol target in
    strategy_config, converted to per-bar units. The old code compared a
    2% PER-HOURLY-BAR target (≈ 187%/yr crypto!) against the 1-step
    hourly sigma — typically <1% — so the "vol target" silently doubled
    nearly every position via the 2.0x clamp.

    Args:
        base_notional: Base dollar amount per trade
        sigma: GARCH 1-bar-ahead sigma (decimal, hourly)
        asset_type: 'crypto' or 'stock'

    Returns:
        Adjusted notional (clamped to 0.5x - 1.5x base).
    """
    if sigma <= 0:
        return base_notional
    # PORTFOLIO_VOL_TARGET scope (c26 S3 / B06 item g): under
    # strategy_config.DERISK_STACK_V2 the book-level scalar
    # (portfolio.get_book_vol_scalar_cached, inside the regime-family MIN)
    # is the ONE owner of this target and base_loop composes this
    # per-position ratio at 1.0 — the ATR risk base already scales
    # per-position vol via stop distance. Legacy (flag OFF) multiplies
    # BOTH (the documented double count, worst case 0.25x).
    from strategy_config import PORTFOLIO_VOL_TARGET
    annual_target = PORTFOLIO_VOL_TARGET.get(asset_type, 0.25)
    target_per_bar = annual_target / np.sqrt(BARS_PER_YEAR.get(asset_type, 8760))
    ratio = target_per_bar / sigma
    ratio = max(0.5, min(1.5, ratio))
    return base_notional * ratio
