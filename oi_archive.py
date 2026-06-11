"""Open-interest features from Binance public archives + live OKX serving.

Evidence: OI dynamics separate trend quality at this system's 12-48h
horizon — rising price WITH rising OI is new money (trend confirmation);
rising price with FALLING OI is short covering (fragile); OI flushes
mark deleveraging events. Funding (already a feature) prices crowded
positioning; OI measures its SIZE.

Training side: data.binance.vision serves DAILY metrics zips per perp
(5-min snapshots, since ~2021-12, free, not geo-blocked). Daily files
are numerous, so sync() walks newest-first with a per-run cap — the
most recent (most training-relevant) history lands first and older
days back-fill across subsequent harvests. Rows are resampled to
hourly before storage.

Live side: Binance live REST is geo-blocked (451) from US IPs, and OI
LEVELS are venue-specific — so live features come from OKX's own OI
against a LOCAL rolling history (same venue-consistent semantics:
both training and serving measure relative OI dynamics, not absolute
levels). Features are 0.0 until enough local history accumulates
(~1 day for the 24h change, ~7 days for the z-score).

Exposed features (stationary): OI_Chg_24h (% change), OI_Z (z vs
trailing 30 days). The parquet also stores top-trader and taker
long/short ratios so future features need no re-download.
"""

import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

ARCHIVE_FILE = BASE_DIR / 'oi_archive.parquet'
_LIVE_HISTORY_FILE = BASE_DIR / 'oi_history.json'

OI_START = '2023-01-01'
MAX_FILES_PER_SYNC = 2000       # bound one sync run (~10-15 min worst case)
_LIVE_CACHE_TTL = 900
_LIVE_THIN_SEC = 3300           # keep ~hourly samples in the local history
_LIVE_MAX_SAMPLES = 24 * 35     # ~35 days

from funding_archive import BINANCE_SYMBOLS  # same Alpaca->Binance map

# OKX perp instruments (same map funding.py uses)
OKX_INSTRUMENTS = {
    'BTC/USD': 'BTC-USDT-SWAP', 'ETH/USD': 'ETH-USDT-SWAP',
    'XRP/USD': 'XRP-USDT-SWAP', 'SOL/USD': 'SOL-USDT-SWAP',
    'DOGE/USD': 'DOGE-USDT-SWAP', 'LINK/USD': 'LINK-USDT-SWAP',
    'AVAX/USD': 'AVAX-USDT-SWAP', 'DOT/USD': 'DOT-USDT-SWAP',
    'LTC/USD': 'LTC-USDT-SWAP', 'BCH/USD': 'BCH-USDT-SWAP',
}

_URL = ("https://data.binance.vision/data/futures/um/daily/metrics/"
        "{sym}/{sym}-metrics-{day}.zip")

_KEEP_COLS = {
    'sum_open_interest': 'oi',
    'sum_open_interest_value': 'oi_value',
    'sum_toptrader_long_short_ratio': 'tt_ls_ratio',
    'sum_taker_long_short_vol_ratio': 'taker_ratio',
}


def _days(start: str) -> list[str]:
    """ISO dates from start through yesterday (UTC), oldest first."""
    d = dt.date.fromisoformat(start)
    end = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    out = []
    while d <= end:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _parse_zip(data: bytes):
    """One daily metrics zip -> hourly-resampled DataFrame (ts + cols)."""
    import pandas as pd
    frames = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                df = pd.read_csv(f)
            if 'create_time' not in df.columns:
                continue
            ts = pd.to_datetime(df['create_time'], utc=True, errors='coerce')
            cols = {}
            for src, dst in _KEEP_COLS.items():
                if src in df.columns:
                    cols[dst] = pd.to_numeric(df[src], errors='coerce')
            if not cols:
                continue
            out = pd.DataFrame(cols)
            out['ts'] = ts
            out = out.dropna(subset=['ts']).set_index('ts').sort_index()
            # Right-labeled buckets: every snapshot inside a bucket is
            # <= its label, so ffill onto bar grids can never leak a
            # future snapshot regardless of bar-stamp convention
            hourly = (out.resample('1h', label='right', closed='right')
                      .last())
            # Flow needs AVERAGING, not a last-5-min snapshot: taker
            # buy/sell ratio is per-window flow, so the hourly value is
            # the mean of the hour's 5-min readings
            if 'taker_ratio' in out.columns:
                hourly['taker_mean'] = (
                    out['taker_ratio']
                    .resample('1h', label='right', closed='right').mean())
            frames.append(hourly.dropna(how='all'))
    if not frames:
        return None
    import pandas as pd
    return pd.concat(frames).reset_index()


def load_archive():
    import pandas as pd
    if ARCHIVE_FILE.exists():
        try:
            return pd.read_parquet(ARCHIVE_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=['symbol', 'ts', 'oi', 'oi_value',
                                 'tt_ls_ratio', 'taker_ratio', 'taker_mean',
                                 'src_day'])


def sync(symbols=None, start: str = OI_START,
         max_files: int = MAX_FILES_PER_SYNC) -> bool:
    """Download missing daily metrics files, NEWEST FIRST, capped per run.

    Recent history (what the row-capped trainer mostly sees) lands in the
    first sync; older days back-fill on later harvests. Idempotent.
    """
    import pandas as pd
    symbols = symbols or list(BINANCE_SYMBOLS)
    arc = load_archive()
    have: set = set()
    if not arc.empty and 'src_day' in arc.columns:
        # Keyed on SOURCE file day, not row dates: right-labeled hourly
        # buckets put each file's final row on the NEXT calendar day,
        # which would otherwise mark never-fetched days as present
        have = set(zip(arc['symbol'], arc['src_day']))

    days_newest_first = list(reversed(_days(start)))
    new_frames = []
    fetched = 0
    for day in days_newest_first:
        if fetched >= max_files:
            break
        for alp in symbols:
            if fetched >= max_files:
                break
            bsym = BINANCE_SYMBOLS.get(alp)
            if not bsym or (alp, day) in have:
                continue
            url = _URL.format(sym=bsym, day=day)
            try:
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'trader/1.0'})
                data = urllib.request.urlopen(req, timeout=30).read()
                fetched += 1
            except urllib.error.HTTPError as e:
                fetched += 1
                if e.code != 404:
                    print(f"[OI-ARCHIVE] {bsym} {day}: HTTP {e.code}")
                continue
            except Exception as e:
                print(f"[OI-ARCHIVE] {bsym} {day}: {e}")
                fetched += 1
                continue
            parsed = _parse_zip(data)
            if parsed is not None and not parsed.empty:
                parsed['symbol'] = alp
                parsed['src_day'] = day
                new_frames.append(parsed)

    if not new_frames:
        print(f"[OI-ARCHIVE] up to date ({len(arc)} rows)")
        return not arc.empty

    combined = pd.concat([arc] + new_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['symbol', 'ts']).sort_values('ts')
    tmp = str(ARCHIVE_FILE) + '.tmp'
    combined.to_parquet(tmp)
    os.replace(tmp, ARCHIVE_FILE)
    remaining = sum(1 for day in days_newest_first for alp in symbols
                    if (alp, day) not in have) - fetched
    print(f"[OI-ARCHIVE] synced {fetched} day-files ({len(combined)} rows"
          f"{f'; ~{max(remaining, 0)} older files back-fill next run' if remaining > 0 else ''})")
    return True


def get_oi_series(alpaca_symbol: str):
    """Hourly oi_value Series (UTC index) for one symbol, or None."""
    arc = load_archive()
    if arc.empty:
        return None
    sub = arc[arc['symbol'] == alpaca_symbol]
    if sub.empty:
        return None
    s = sub.set_index('ts')['oi_value'].sort_index()
    return s[~s.index.duplicated(keep='last')].dropna()


def oi_features_for_index(alpaca_symbol: str, index):
    """Point-in-time OI features aligned to an hourly bar index.

      OI_Chg_24h  % change in OI notional over 24h (new money vs unwind)
      OI_Z        z-score of OI notional vs trailing 30 days

    Stationary; hourly snapshots are known at snapshot time, so ffill
    onto the bar grid introduces no look-ahead. None when the archive
    has under ~8 days for the symbol.
    """
    import numpy as np
    import pandas as pd
    s = get_oi_series(alpaca_symbol)
    if s is None or len(s) < 200:
        return None
    chg = s.pct_change(24) * 100
    roll = s.rolling(720, min_periods=168)
    mu, sd = roll.mean(), roll.std()
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    z = z.mask((sd == 0) & mu.notna(), 0.0)

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return {
        'OI_Chg_24h': chg.reindex(idx, method='ffill').values,
        'OI_Z': z.reindex(idx, method='ffill').values,
    }


def ls_features_for_index(alpaca_symbol: str, index):
    """Top-trader long/short positioning feature aligned to hourly bars.

      TT_LS_Z  z-score of Binance top-trader long/short POSITION ratio
               vs its trailing 30 days

    Complements funding (the PRICE of crowding) and OI (its SIZE) with
    its DIRECTION among the largest accounts. Stationary by z-scoring;
    same point-in-time semantics as the OI features (right-labeled
    hourly snapshots, ffill). None when under ~8 days of archive.
    """
    import numpy as np
    import pandas as pd
    arc = load_archive()
    if arc.empty or 'tt_ls_ratio' not in arc.columns:
        return None
    sub = arc[arc['symbol'] == alpaca_symbol]
    if sub.empty:
        return None
    s = sub.set_index('ts')['tt_ls_ratio'].sort_index()
    s = s[~s.index.duplicated(keep='last')].dropna()
    if len(s) < 200:
        return None
    roll = s.rolling(720, min_periods=168)
    mu, sd = roll.mean(), roll.std()
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    z = z.mask((sd == 0) & mu.notna(), 0.0)
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return {'TT_LS_Z': z.reindex(idx, method='ffill').values}


def taker_features_for_index(alpaca_symbol: str, index):
    """Aggressive-flow imbalance feature aligned to hourly bars.

      Taker_Imb_24h  log of the 24h mean taker buy/sell volume ratio
                     (>0 net aggressive buying, <0 net selling)

    Completes the triad with FLOW: funding prices crowding, OI sizes it,
    TT_LS signs the positioning, this measures who is hitting the tape.
    Uses the hourly-MEAN taker column (taker_mean); archives synced
    before that column existed return None until re-synced days land.
    """
    import numpy as np
    import pandas as pd
    arc = load_archive()
    if arc.empty or 'taker_mean' not in arc.columns:
        return None
    sub = arc[arc['symbol'] == alpaca_symbol]
    if sub.empty:
        return None
    s = sub.set_index('ts')['taker_mean'].sort_index()
    s = s[~s.index.duplicated(keep='last')].dropna()
    if len(s) < 48:
        return None
    m24 = s.rolling(24, min_periods=12).mean()
    feat = np.log(m24.where(m24 > 0))
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return {'Taker_Imb_24h': feat.reindex(idx, method='ffill').values}


# --- Live serving (OKX; venue-consistent local history) ---

_live_cache: dict[str, tuple[float, float]] = {}


def _fetch_okx_oi(symbol: str) -> float | None:
    inst = OKX_INSTRUMENTS.get(symbol)
    if inst is None:
        return None
    now = time.monotonic()
    hit = _live_cache.get(symbol)
    if hit and (now - hit[0]) < _LIVE_CACHE_TTL:
        return hit[1]
    try:
        url = f"https://www.okx.com/api/v5/public/open-interest?instId={inst}"
        req = urllib.request.Request(url, headers={'User-Agent': 'trader/1.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        oi = float(data['data'][0]['oiCcy'])  # coin units (venue-local)
    except Exception:
        return None
    _live_cache[symbol] = (now, oi)
    return oi


def _load_live_history() -> dict:
    try:
        with open(_LIVE_HISTORY_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def live_oi_features(symbol: str) -> dict | None:
    """Live OI_Chg_24h / OI_Z from OKX against a local rolling history.

    OI LEVELS are venue-specific, so the live z compares OKX to its own
    trailing window (same relative-dynamics semantics the model trained
    on). Features read 0.0 until local history accumulates.
    """
    oi = _fetch_okx_oi(symbol)
    if oi is None:
        return None
    now = time.time()
    hist_all = _load_live_history()
    hist = hist_all.get(symbol, [])

    chg = 0.0
    target = now - 86400
    candidates = [(abs(ts - target), v) for ts, v in hist
                  if abs(ts - target) <= 3 * 3600]
    if candidates:
        _, ref = min(candidates)
        if ref > 0:
            chg = (oi - ref) / ref * 100

    z = 0.0
    vals = [v for _, v in hist]
    if len(vals) >= 168:
        import statistics
        mu = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        if sd > 1e-12:
            z = (oi - mu) / sd

    # Thin to ~hourly samples; persist
    if not hist or (now - hist[-1][0]) >= _LIVE_THIN_SEC:
        hist.append([now, oi])
        if len(hist) > _LIVE_MAX_SAMPLES:
            del hist[:len(hist) - _LIVE_MAX_SAMPLES]
        hist_all[symbol] = hist
        try:
            tmp = str(_LIVE_HISTORY_FILE) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(hist_all, f)
            os.replace(tmp, _LIVE_HISTORY_FILE)
        except OSError:
            pass
    return {'OI_Chg_24h': chg, 'OI_Z': z}


_ls_cache: dict[str, tuple[float, dict]] = {}


def live_ls_features(symbol: str) -> dict | None:
    """Live TT_LS_Z from OKX's long/short-ratio history (one call covers
    the full 30d z window — no local accumulation or cold start).

    Definition note: the Binance archive feature is the TOP-TRADER
    POSITION ratio; OKX serves the all-account ratio. Both measure
    long-crowding direction, and as a z vs the SAME venue's trailing
    window the dynamics are serving-consistent (the venue-local pattern
    the OI features use).
    """
    base = symbol.split('/')[0]
    now = time.monotonic()
    hit = _ls_cache.get(symbol)
    if hit and (now - hit[0]) < _LIVE_CACHE_TTL:
        return hit[1]
    try:
        begin_ms = int((time.time() - 31 * 86400) * 1000)
        url = ("https://www.okx.com/api/v5/rubik/stat/contracts/"
               f"long-short-account-ratio?ccy={base}&period=1H"
               f"&begin={begin_ms}")
        req = urllib.request.Request(url, headers={'User-Agent': 'trader/1.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        vals = [float(r[1]) for r in data.get('data', [])]  # newest first
    except Exception:
        return None
    if len(vals) < 168:
        return None
    import statistics
    cur = vals[0]
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    z = (cur - mu) / sd if sd > 1e-12 else 0.0
    out = {'TT_LS_Z': z}
    _ls_cache[symbol] = (now, out)
    return out


_taker_cache: dict[str, tuple[float, dict]] = {}


def live_taker_features(symbol: str) -> dict | None:
    """Live Taker_Imb_24h from OKX taker-volume history (one call,
    no cold start). Response rows are [ts, sellVol, buyVol] newest
    first (OKX rubik docs); hourly ratio = buy/sell, feature = log of
    the newest-24h mean — venue-local dynamics, same as OI/TT_LS."""
    import math
    base = symbol.split('/')[0]
    now = time.monotonic()
    hit = _taker_cache.get(symbol)
    if hit and (now - hit[0]) < _LIVE_CACHE_TTL:
        return hit[1]
    try:
        url = ("https://www.okx.com/api/v5/rubik/stat/taker-volume"
               f"?ccy={base}&instType=CONTRACTS&period=1H")
        req = urllib.request.Request(url, headers={'User-Agent': 'trader/1.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        rows = data.get('data', [])[:24]  # newest 24 hours
        ratios = []
        for r in rows:
            sell, buy = float(r[1]), float(r[2])
            if sell > 0 and buy > 0:
                ratios.append(buy / sell)
    except Exception:
        return None
    if len(ratios) < 12:
        return None
    out = {'Taker_Imb_24h': math.log(sum(ratios) / len(ratios))}
    _taker_cache[symbol] = (now, out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Sync Binance OI archives')
    ap.add_argument('--start', default=OI_START)
    ap.add_argument('--max-files', type=int, default=MAX_FILES_PER_SYNC)
    args = ap.parse_args()
    sync(start=args.start, max_files=args.max_files)
