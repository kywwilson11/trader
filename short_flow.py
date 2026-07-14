"""FINRA daily short-volume flow — per-name positioning features.

Short sellers are informed, anomaly-exploiting traders, and their FLOW
predicts negative returns even though the data is public — the market
underreacts (day-1 reaction of a few bps vs alpha persisting for
months; Boehmer-Jones-Zhang lineage). Unlike the biweekly short
INTEREST snapshot everyone watches, FINRA publishes per-name short
VOLUME daily, same evening, free:

    https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Features (long-window — the daily prints are noisy):
    SVR_21  21d sum(ShortVolume) / sum(TotalVolume)
    SVR_Z   z of SVR_21 vs the name's own trailing 252d

For a LONG-only book this is a sell-side-pressure feature/penalty:
persistently elevated shorting flow marks names informed traders are
leaning on. Point-in-time: file for day D is published ~18:30 ET on D;
training maps day-D values onto day D+1 bars (shift 1) and live uses
the latest completed day — identical information sets.
"""

import datetime as dt
import io
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from log_config import get_logger

logger = get_logger(__name__)

ARCHIVE_FILE = BASE_DIR / 'short_flow.parquet'
_URL = 'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt'

START_DAYS_BACK = 600           # ~280 trading days of SVR_21 z-window + slack
MAX_FILES_PER_SYNC = 150        # back-fills across a few harvests
SVR_WINDOW = 21
Z_WINDOW = 252
Z_MIN = 60
SVR_STALE_DAYS = 7              # warn if the newest print is older than this (weekly sync)
_svr_stale_warned: set[str] = set()


def _panel_symbols() -> set[str]:
    from stock_config import load_stock_universe, TRAINING_CANDIDATE_POOL
    uni = {s for s in load_stock_universe() if '/' not in s}
    return uni | set(TRAINING_CANDIDATE_POOL)


def _parse_file(text: str, keep: set[str]):
    import pandas as pd
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split('|')
        if len(parts) < 5:
            continue
        sym = parts[1].strip().upper()
        if sym not in keep:
            continue
        try:
            rows.append((parts[0].strip(), sym,
                         float(parts[2]), float(parts[4])))
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['date', 'symbol', 'short_vol',
                                     'total_vol'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # Same symbol appears once per market venue tape — aggregate
    return df.groupby(['date', 'symbol'], as_index=False).sum()


def load_archive():
    import pandas as pd
    if ARCHIVE_FILE.exists():
        try:
            return pd.read_parquet(ARCHIVE_FILE)
        except Exception as e:
            logger.warning("[SHORT-FLOW] archive read failed (%s) — serving "
                           "no SVR features until the next sync rebuilds it", e)
    return pd.DataFrame(columns=['date', 'symbol', 'short_vol', 'total_vol'])


def sync(days_back: int = START_DAYS_BACK,
         max_files: int = MAX_FILES_PER_SYNC) -> bool:
    """Download missing daily files, NEWEST FIRST, capped per run.
    Non-trading days 404 and are skipped. Idempotent."""
    import pandas as pd
    keep = _panel_symbols()
    arc = load_archive()
    have = set()
    if not arc.empty:
        have = set(arc['date'].dt.strftime('%Y%m%d'))

    today = dt.date.today()
    attempts = 0    # request budget: counts 404s/errors too (max_files cap)
    fetched = 0     # files actually downloaded (HTTP 200)
    frames = []
    for back in range(1, days_back + 1):
        if attempts >= max_files:
            break
        day = today - dt.timedelta(days=back)
        if day.weekday() >= 5:
            continue
        ds = day.strftime('%Y%m%d')
        if ds in have:
            continue
        try:
            req = urllib.request.Request(_URL.format(d=ds),
                                         headers={'User-Agent': 'trader/1.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode()
            attempts += 1
            fetched += 1
        except urllib.error.HTTPError as e:
            if e.code != 404:           # 404 = holiday, expected
                print(f"[SHORT-FLOW] {ds}: HTTP {e.code}")
            attempts += 1
            continue
        except Exception as e:
            print(f"[SHORT-FLOW] {ds}: {e}")
            attempts += 1
            continue
        parsed = _parse_file(text, keep)
        if parsed is not None:
            frames.append(parsed)

    if not frames:
        print(f"[SHORT-FLOW] up to date ({len(arc)} rows)")
        return not arc.empty
    combined = pd.concat([arc] + frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'symbol'])
    combined = combined.sort_values('date')
    tmp = str(ARCHIVE_FILE) + '.tmp'
    combined.to_parquet(tmp)
    os.replace(tmp, ARCHIVE_FILE)
    print(f"[SHORT-FLOW] fetched {fetched} day-files, ingested {len(frames)} "
          f"({len(combined)} rows)")
    return True


def svr_series(symbol: str):
    """(SVR_21, SVR_Z) daily Series for one symbol, or None.

    Both are indexed by the PRINT date — callers shift(1) for training
    bars; live uses .iloc[-1] (latest completed day)."""
    import pandas as pd
    arc = load_archive()
    if arc.empty:
        return None
    sub = arc[arc['symbol'] == symbol.upper()]
    if len(sub) < SVR_WINDOW + 5:
        return None
    sub = sub.set_index('date').sort_index()
    sv = sub['short_vol'].rolling(SVR_WINDOW, min_periods=SVR_WINDOW).sum()
    tv = sub['total_vol'].rolling(SVR_WINDOW, min_periods=SVR_WINDOW).sum()
    svr = (sv / tv.replace(0, float('nan'))).dropna()
    if svr.empty:
        return None
    roll = svr.rolling(Z_WINDOW, min_periods=Z_MIN)
    mu, sd = roll.mean(), roll.std()
    z = ((svr - mu) / sd).where(sd > 1e-12, 0.0)
    return svr, z


def svr_features_for_index(symbol: str, index):
    """{'SVR_21', 'SVR_Z'} aligned to an intraday bar index (shift-1
    day-mapped, point-in-time), or None when the archive lacks the name."""
    import pandas as pd
    out = svr_series(symbol)
    if out is None:
        return None
    svr, z = out
    idx = pd.DatetimeIndex(index)
    dates = pd.Series(idx.normalize().tz_localize(None)
                      if idx.tz is not None else idx.normalize(), index=idx)
    return {'SVR_21': dates.map(svr.shift(1)).values,
            'SVR_Z': dates.map(z.shift(1)).values}


def live_svr_features(symbol: str) -> dict | None:
    """Latest completed-day values for live injection."""
    out = svr_series(symbol)
    if out is None:
        return None
    svr, z = out
    # Observability only (returned values unchanged): sync runs at the weekly
    # stock harvest, so the newest print can be ~a week old while training uses
    # a strict 1-day lag. Stamp the age; warn once per symbol beyond the guard.
    try:
        age_days = (dt.date.today() - svr.index[-1].date()).days
        key = symbol.upper()
        if age_days > SVR_STALE_DAYS and key not in _svr_stale_warned:
            _svr_stale_warned.add(key)
            logger.warning('[SHORT-FLOW] %s: SVR print is %dd old '
                           '(train uses a 1-day lag)', key, age_days)
    except Exception:
        pass
    return {'SVR_21': float(svr.iloc[-1]), 'SVR_Z': float(z.iloc[-1])}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Sync FINRA daily short volume')
    ap.add_argument('--days-back', type=int, default=START_DAYS_BACK)
    ap.add_argument('--max-files', type=int, default=MAX_FILES_PER_SYNC)
    args = ap.parse_args()
    sync(days_back=args.days_back, max_files=args.max_files)
