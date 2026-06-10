"""Historical perp funding rates from Binance public archives (training data).

data.binance.vision serves monthly fundingRate zips per perp symbol back to
2020 — free, no key, and (unlike Binance's live REST API) NOT geo-blocked
from US IPs. Each 8h funding print is timestamped at its funding time, so
the series is point-in-time exact: a bar at time t may see every funding
print at or before t.

Produces funding_archive.parquet (long format: symbol, ts, rate) consumed
by the crypto harvest (training features) and by funding.py (live z-score
baseline so train/serve distributions match from day one).

Usage:
    python funding_archive.py            # sync all mapped symbols
    python funding_archive.py --start 2020-01
"""

import argparse
import datetime as dt
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

ARCHIVE_FILE = BASE_DIR / 'funding_archive.parquet'

# Alpaca spot symbol -> Binance USDT-margined perp symbol
BINANCE_SYMBOLS = {
    'BTC/USD': 'BTCUSDT', 'ETH/USD': 'ETHUSDT', 'XRP/USD': 'XRPUSDT',
    'SOL/USD': 'SOLUSDT', 'DOGE/USD': 'DOGEUSDT', 'LINK/USD': 'LINKUSDT',
    'AVAX/USD': 'AVAXUSDT', 'DOT/USD': 'DOTUSDT', 'LTC/USD': 'LTCUSDT',
    'BCH/USD': 'BCHUSDT',
}

_URL = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        "{sym}/{sym}-fundingRate-{month}.zip")


def _months(start: str) -> list[str]:
    """Complete months from start (YYYY-MM) through last month."""
    y, m = map(int, start.split('-'))
    today = dt.date.today()
    out = []
    while (y, m) < (today.year, today.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _parse_zip(data: bytes):
    """Rows of (ts_utc, rate) from one monthly archive zip."""
    import pandas as pd
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                df = pd.read_csv(f)
            # Column names vary across vintages; identify by content:
            # the epoch-ms column is huge ints, the rate is a small float
            time_col = rate_col = None
            for c in df.columns:
                series = pd.to_numeric(df[c], errors='coerce')
                if series.isna().all():
                    continue
                if series.abs().median() > 1e12:
                    time_col = c
                elif series.abs().median() < 0.05:
                    rate_col = c
            if time_col is None or rate_col is None:
                continue
            ts = pd.to_datetime(pd.to_numeric(df[time_col]), unit='ms', utc=True)
            rate = pd.to_numeric(df[rate_col], errors='coerce')
            rows.append(pd.DataFrame({'ts': ts, 'rate': rate}).dropna())
    import pandas as pd
    return pd.concat(rows) if rows else None


def load_archive():
    import pandas as pd
    if ARCHIVE_FILE.exists():
        try:
            return pd.read_parquet(ARCHIVE_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=['symbol', 'ts', 'rate'])


def get_funding_series(alpaca_symbol: str):
    """Funding rate Series (UTC ts index) for one Alpaca symbol, or None."""
    arc = load_archive()
    if arc.empty:
        return None
    sub = arc[arc['symbol'] == alpaca_symbol]
    if sub.empty:
        return None
    s = sub.set_index('ts')['rate'].sort_index()
    return s[~s.index.duplicated(keep='last')]


def sync(symbols=None, start: str = '2020-01') -> bool:
    """Download missing months for each symbol into the parquet store.

    Idempotent: months already present are skipped (one network burst on
    first run, a single small zip per symbol per month thereafter). 404s
    (symbol not yet listed that month) are skipped silently.
    """
    import pandas as pd
    symbols = symbols or list(BINANCE_SYMBOLS)
    arc = load_archive()
    have = set()
    if not arc.empty:
        have = set(zip(arc['symbol'], arc['ts'].dt.strftime('%Y-%m')))

    new_frames = []
    for alp in symbols:
        bsym = BINANCE_SYMBOLS.get(alp)
        if not bsym:
            continue
        for month in _months(start):
            if (alp, month) in have:
                continue
            url = _URL.format(sym=bsym, month=month)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'trader/1.0'})
                data = urllib.request.urlopen(req, timeout=30).read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # not listed yet that month
                print(f"[FUNDING-ARCHIVE] {bsym} {month}: HTTP {e.code}")
                continue
            except Exception as e:
                print(f"[FUNDING-ARCHIVE] {bsym} {month}: {e}")
                continue
            parsed = _parse_zip(data)
            if parsed is not None and not parsed.empty:
                parsed['symbol'] = alp
                new_frames.append(parsed)

    if not new_frames:
        print(f"[FUNDING-ARCHIVE] up to date ({len(arc)} rows)")
        return not arc.empty

    combined = pd.concat([arc] + new_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['symbol', 'ts']).sort_values('ts')
    tmp = str(ARCHIVE_FILE) + '.tmp'
    combined.to_parquet(tmp)
    os.replace(tmp, ARCHIVE_FILE)
    print(f"[FUNDING-ARCHIVE] synced: {len(combined)} rows "
          f"({len(new_frames)} new month-files)")
    return True


def funding_features_for_index(alpaca_symbol: str, index):
    """Point-in-time funding features aligned to an hourly bar index.

    Returns dict of column -> array (or None when no archive data):
      Funding_Rate_Ann  annualized current 8h funding (rate * 3 * 365)
      Funding_Z         z-score vs the trailing 90 funding prints
      Funding_Chg_24h   annualized 24h change in the funding rate

    All three are STATIONARY (rates/z-scores) and known at print time —
    ffill onto the bar grid introduces no look-ahead.
    """
    import numpy as np
    import pandas as pd
    s = get_funding_series(alpaca_symbol)
    if s is None or len(s) < 40:
        return None
    ann = s * 3 * 365
    roll = s.rolling(90, min_periods=30)
    mu, sd = roll.mean(), roll.std()
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    # Flat stretches (funding pinned at Binance's 0.01% default for weeks
    # is common) give sd=0 -> 0/0 NaN; semantically that's z=0, and NaN
    # here would silently drop those bars at the harvest dropna()
    z = z.mask((sd == 0) & mu.notna(), 0.0)
    chg = (s - s.shift(3)) * 3 * 365  # 3 prints = 24h

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    out = {}
    for name, series in (('Funding_Rate_Ann', ann), ('Funding_Z', z),
                         ('Funding_Chg_24h', chg)):
        out[name] = series.reindex(idx, method='ffill').values
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Sync Binance funding archives')
    ap.add_argument('--start', default='2020-01')
    args = ap.parse_args()
    sync(start=args.start)
