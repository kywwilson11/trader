"""Hourly bars per year — the one place the annualization calendar is decided.

Zero heavy deps (stdlib only; strategy_config is imported lazily and is itself
a pure-constants module). Consumers pass their OWN legacy dict so the flag-OFF
path evaluates exactly the expression they used before
(`legacy.get(asset_type, default)`) — byte-identical, including under tests
that monkeypatch a module's BARS_PER_YEAR.

Flag: strategy_config.BARS_PER_YEAR_MEASURED (default OFF; read with getattr
so this module is safe before that constant exists). ON switches the STOCK
book to the measured extended-hours calendar; crypto is unchanged.

MEASURED_BARS_PER_YEAR['stock'] = 3827 — scripts/bars_per_year_census.py on
stock_training_data.parquet (rebuilt 2026-09-26, 90 names, 2016-01-12 ..
2026-09-22, 1,536,356 rows): Alpaca hourly bars span the extended session
(04:00-20:00 ET open times, market_data.py _SIP_SESSION_*), 50.7% of rows
overlap RTH. Trailing-365-day, row-weighted (pooled over ticker-days) mean =
15.185 bars/ticker-day x 252 = 3826.6 -> 3827. Pooled because
hypersearch_v2.compute_sharpe converts the SUM of rows over all tickers into
ticker-years; trailing year because the walk-forward folds and the holdout sit
at the recent end and extended-hours depth has grown every year (whole-store
pooled 3443; per-year 2760 in 2016 -> 3864 in 2026). Legacy 1638 = 252 x 6.5
counts RTH HOURS; even an RTH-only hourly store has 7 bars/day (the 09:00 bar
straddles the open) = 1764. Crypto: measured 8751 whole-store / 8766 trailing
year vs legacy 8760 (365 x 24) — kept at 8760 (sqrt ratio 1.000).
Re-run the census after any harvest-window change before trusting 3827.
"""

LEGACY_BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
MEASURED_BARS_PER_YEAR = {'crypto': 8760, 'stock': 3827}


def measured_enabled() -> bool:
    """strategy_config.BARS_PER_YEAR_MEASURED, read at CALL time."""
    try:
        import strategy_config
    except Exception:
        return False
    return bool(getattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False))


def bars_per_year(asset_type, legacy=None, default=8760):
    """Bars per year for `asset_type`.

    OFF -> `legacy.get(asset_type, default)` with `legacy` the caller's own
    table (LEGACY_BARS_PER_YEAR when None) — the pre-flag expression.
    ON  -> MEASURED_BARS_PER_YEAR for known books, else the legacy lookup.
    The returned type follows the table (int here; callers cast as before).
    """
    table = LEGACY_BARS_PER_YEAR if legacy is None else legacy
    if measured_enabled() and asset_type in MEASURED_BARS_PER_YEAR:
        return MEASURED_BARS_PER_YEAR[asset_type]
    return table.get(asset_type, default)


# --- bars per DAY (ENGINE-R2: volatility.py's HAR daily -> per-bar sigma) ---
# Per-day values are DERIVED from the per-year tables (bars_per_year / trading
# days) so a per-day consumer and a per-year consumer can never disagree:
# legacy 6.5 x 252 = 1638 and 24 x 365 = 8760 exactly; measured stock
# 3827 / 252 = 15.1865 (the census's 15.185 bars/ticker-day, re-derived from
# the rounded 3827 so the identity bpy == bpd x days holds exactly), crypto
# 8760 / 365 = 24.0 (unchanged). The numbers keep ONE home: the tables above.
LEGACY_BARS_PER_DAY = {'crypto': 24.0, 'stock': 6.5}
TRADING_DAYS_PER_YEAR = {'crypto': 365, 'stock': 252}


def bars_per_day(asset_type, legacy=None, default=6.5):
    """Bars per trading day for `asset_type`.

    OFF -> `legacy.get(asset_type, default)` with `legacy` the caller's own
    table (LEGACY_BARS_PER_DAY when None) — the pre-flag expression.
    ON  -> MEASURED_BARS_PER_YEAR / TRADING_DAYS_PER_YEAR for known books
    (float), else the legacy lookup.
    """
    table = LEGACY_BARS_PER_DAY if legacy is None else legacy
    if (measured_enabled() and asset_type in MEASURED_BARS_PER_YEAR
            and asset_type in TRADING_DAYS_PER_YEAR):
        return (MEASURED_BARS_PER_YEAR[asset_type]
                / TRADING_DAYS_PER_YEAR[asset_type])
    return table.get(asset_type, default)
