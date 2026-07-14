"""Spot-perp BASIS archive — the higher-frequency carry primitive (wave-7).

Funding rate is the 8-hour moving average of the perp premium; the BASIS
(perp mark minus spot index, i.e. Binance's premium index) is the same signal
 before that smoothing, sampled hourly. The system already trades crypto with
funding as a feature, but no basis/premium/mark-price code exists — so the
less-lagged carry/crash primitive is missing.

This is a faithful clone of funding_archive.py (same _months / parquet / sync /
PIT-ffill shape) pointed at Binance's free, zero-auth premiumIndexKlines
monthly archive (hourly closes = the premium fraction, ~720 rows/month, +/-9bps
typical). Offline-buildable and unit-tested on synthetic zips; live serving
(OKX mark/index -> premium with the Binance trailing window for train/serve
parity) is a thin follow-on.

Features mirror funding's: Basis_Bps, Basis_Z (trailing-30d, sd==0 -> 0 mask),
Basis_Chg_24h, plus Basis_minus_Funding (the lead/lag residual — the only
genuinely NOVEL column; the others are largely redundant with funding and
should carry a redundancy-haircut expectation).
"""

import argparse
import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_FILE = BASE_DIR / 'basis_archive.parquet'

# Reuse the funding archive's Alpaca<->Binance symbol map (same perps).
try:
    from funding_archive import BINANCE_SYMBOLS, _months
except Exception:  # pragma: no cover - funding_archive always present
    BINANCE_SYMBOLS = {}

    def _months(start):
        return []

_URL = ("https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"
        "{sym}/1h/{sym}-1h-{month}.zip")

# Trailing window for the z-score: 30 days of HOURLY prints.
_Z_WINDOW = 720
_Z_MINP = 240


def _parse_zip(data: bytes):
    """Rows of (ts_utc, premium) from one monthly premiumIndexKlines zip.

    Klines are headerless (older) or headered (Binance added headers in 2025)
    12-column rows: open_time(ms), open, high, low, close, ... We read
    positionally and drop any header row via numeric coercion: open_time is
    col 0 (epoch ms), the premium fraction is the close in col 4.
    """
    import pandas as pd
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                df = pd.read_csv(f, header=None)
            if df.shape[1] < 5:
                continue
            open_ms = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            premium = pd.to_numeric(df.iloc[:, 4], errors='coerce')
            ok = open_ms.notna() & premium.notna()      # drops a header row
            if not ok.any():
                continue
            ts = pd.to_datetime(open_ms[ok], unit='ms', utc=True)
            # reset_index (NOT .values) preserves the UTC tz that .values strips
            rows.append(pd.DataFrame({
                'ts': ts.reset_index(drop=True),
                'premium': premium[ok].reset_index(drop=True)}))
    return pd.concat(rows) if rows else None


def load_archive():
    import pandas as pd
    if ARCHIVE_FILE.exists():
        try:
            return pd.read_parquet(ARCHIVE_FILE)
        except Exception as e:
            print(f"[BASIS-ARCHIVE] corrupt archive {ARCHIVE_FILE}: {e} "
                  f"— treating as empty")
    return pd.DataFrame(columns=['symbol', 'ts', 'premium'])


def get_basis_series(alpaca_symbol: str):
    """Premium (basis) Series, UTC ts index, for one Alpaca symbol, or None."""
    arc = load_archive()
    if arc.empty:
        return None
    sub = arc[arc['symbol'] == alpaca_symbol]
    if sub.empty:
        return None
    s = sub.set_index('ts')['premium'].sort_index()
    return s[~s.index.duplicated(keep='last')]


def sync(symbols=None, start: str = '2020-01') -> bool:
    """Download missing months into the parquet store (idempotent; 404-skip)."""
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
                    continue
                print(f"[BASIS-ARCHIVE] {bsym} {month}: HTTP {e.code}")
                continue
            except Exception as e:
                print(f"[BASIS-ARCHIVE] {bsym} {month}: {e}")
                continue
            # A parse failure must not abort the sync: that would discard
            # every month already fetched this run, and the bad month would
            # re-download and re-crash every future run (never in 'have').
            try:
                parsed = _parse_zip(data)
            except Exception as e:
                print(f"[BASIS-ARCHIVE] {bsym} {month}: parse failed: {e}")
                continue
            if parsed is not None and not parsed.empty:
                parsed['symbol'] = alp
                new_frames.append(parsed)

    if not new_frames:
        print(f"[BASIS-ARCHIVE] up to date ({len(arc)} rows)")
        return not arc.empty
    combined = pd.concat([arc] + new_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['symbol', 'ts']).sort_values('ts')
    tmp = str(ARCHIVE_FILE) + '.tmp'
    combined.to_parquet(tmp)
    os.replace(tmp, ARCHIVE_FILE)
    print(f"[BASIS-ARCHIVE] synced: {len(combined)} rows "
          f"({len(new_frames)} new month-files)")
    return True


def basis_features_for_index(alpaca_symbol: str, index):
    """Point-in-time basis features aligned to an hourly bar index, or None.

    Basis_Bps          current premium in basis points (premium * 1e4)
    Basis_Z            z-score vs the trailing 30d of hourly prints
    Basis_Chg_24h      24h change in the premium, in bps
    Basis_minus_Funding  basis bps minus the funding-implied per-hour premium
                         bps (the lead/lag residual — the novel column)

    All stationary and known at print time; ffill onto the bar grid is PIT.
    """
    import numpy as np
    import pandas as pd
    s = get_basis_series(alpaca_symbol)
    if s is None or len(s) < 48:
        return None
    bps = s * 1e4
    roll = s.rolling(_Z_WINDOW, min_periods=_Z_MINP)
    mu, sd = roll.mean(), roll.std()
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    z = z.mask((sd == 0) & mu.notna(), 0.0)             # flat-premium stretches
    chg = (s - s.shift(24)) * 1e4

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    out = {}
    for name, series in (('Basis_Bps', bps), ('Basis_Z', z),
                         ('Basis_Chg_24h', chg)):
        out[name] = series.reindex(idx, method='ffill').values

    # Basis minus the funding-implied premium (funding is the 8h MA of basis,
    # so this residual is the lead/lag info funding alone cannot carry).
    try:
        from funding_archive import get_funding_series
        f = get_funding_series(alpaca_symbol)
        if f is not None and len(f):
            fund_hourly_bps = (f / 8.0) * 1e4   # 8h funding -> per-hour premium bps
            fb = fund_hourly_bps.reindex(idx, method='ffill')
            out['Basis_minus_Funding'] = (pd.Series(out['Basis_Bps'], index=idx)
                                          - fb).values
    except Exception:
        pass
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Sync Binance spot-perp basis archives')
    ap.add_argument('--start', default='2020-01')
    args = ap.parse_args()
    sync(start=args.start)
