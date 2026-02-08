"""Fundamental data layer — yfinance + Financial Modeling Prep + SEC EDGAR.

Provides P/E, market cap, insider activity, and LLM-summarized SEC filings.
All data aggressively cached in-memory with TTL (same pattern as sentiment.py).
"""

import time
import datetime
import json
import urllib.request
import urllib.error

from llm_config import load_llm_config

# --- Cache: key -> (timestamp, result) ---
_cache = {}
YFINANCE_TTL = 4 * 3600      # 4 hours
FMP_TTL = 24 * 3600           # 24 hours
SEC_TTL = 7 * 24 * 3600       # 7 days


def _cache_get(key, ttl):
    """Return cached value if fresh, else None."""
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None


def _cache_set(key, val):
    _cache[key] = (time.time(), val)


# --- yfinance fundamentals ---

def get_fundamentals(symbol: str, asset_type: str = "stock") -> dict:
    """Fetch fundamental data for a symbol. Returns dict with available metrics."""
    cache_key = f"fund_{symbol}"
    cached = _cache_get(cache_key, YFINANCE_TTL)
    if cached is not None:
        return cached

    result = {
        "pe_ratio": None,
        "pb_ratio": None,
        "market_cap": None,
        "revenue_growth": None,
        "eps": None,
        "dividend_yield": None,
        "week52_high": None,
        "week52_low": None,
        "sector": None,
        "beta": None,
        "avg_volume": None,
    }

    try:
        import yfinance as yf

        # For crypto, strip /USD -> -USD for yfinance
        yf_symbol = symbol.replace("/", "-") if asset_type == "crypto" else symbol
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}

        result["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        result["pb_ratio"] = info.get("priceToBook")
        result["market_cap"] = info.get("marketCap")
        result["revenue_growth"] = info.get("revenueGrowth")
        result["eps"] = info.get("trailingEps")
        result["dividend_yield"] = info.get("dividendYield")
        result["week52_high"] = info.get("fiftyTwoWeekHigh")
        result["week52_low"] = info.get("fiftyTwoWeekLow")
        result["sector"] = info.get("sector")
        result["beta"] = info.get("beta")
        result["avg_volume"] = info.get("averageVolume")

    except Exception as e:
        print(f"[FUNDAMENTALS] yfinance error for {symbol}: {e}")

    # Enrich stocks with FMP data
    if asset_type == "stock":
        fmp_data = _fetch_fmp_metrics(symbol)
        if fmp_data:
            for k, v in fmp_data.items():
                if v is not None and result.get(k) is None:
                    result[k] = v

    _cache_set(cache_key, result)
    return result


def _fetch_fmp_metrics(symbol: str) -> dict | None:
    """Fetch key metrics from Financial Modeling Prep (free tier)."""
    cache_key = f"fmp_{symbol}"
    cached = _cache_get(cache_key, FMP_TTL)
    if cached is not None:
        return cached

    config = load_llm_config()
    api_key = config.get("fmp_api_key", "")
    if not api_key:
        return None

    result = {}
    try:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?limit=1&apikey={api_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "trader-bot/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data and isinstance(data, list) and len(data) > 0:
            m = data[0]
            result["pe_ratio"] = m.get("peRatio")
            result["pb_ratio"] = m.get("pbRatio")
            result["revenue_growth"] = m.get("revenuePerShare")
            result["eps"] = m.get("netIncomePerShare")
            result["dividend_yield"] = m.get("dividendYield")
    except Exception as e:
        print(f"[FUNDAMENTALS] FMP metrics error for {symbol}: {e}")

    _cache_set(cache_key, result)
    return result


# --- Insider activity (FMP) ---

def get_insider_activity(symbol: str) -> dict:
    """Fetch recent insider trading activity from FMP."""
    cache_key = f"insider_{symbol}"
    cached = _cache_get(cache_key, FMP_TTL)
    if cached is not None:
        return cached

    result = {"net_shares": 0, "recent_buys": 0, "recent_sells": 0, "summary": "N/A"}

    config = load_llm_config()
    api_key = config.get("fmp_api_key", "")
    if not api_key:
        _cache_set(cache_key, result)
        return result

    try:
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&limit=10&apikey={api_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "trader-bot/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        if data and isinstance(data, list):
            buys = sum(1 for t in data if t.get("transactionType", "").lower() in ("p-purchase", "purchase", "buy"))
            sells = sum(1 for t in data if t.get("transactionType", "").lower() in ("s-sale", "sale", "sell"))
            net = sum(
                (t.get("securitiesTransacted", 0) if "purchase" in t.get("transactionType", "").lower() or "buy" in t.get("transactionType", "").lower()
                 else -t.get("securitiesTransacted", 0))
                for t in data
            )
            result = {
                "net_shares": int(net),
                "recent_buys": buys,
                "recent_sells": sells,
                "summary": f"{buys} buys, {sells} sells (net {'+' if net >= 0 else ''}{int(net)} shares)",
            }
    except Exception as e:
        print(f"[FUNDAMENTALS] Insider activity error for {symbol}: {e}")

    _cache_set(cache_key, result)
    return result


# --- SEC EDGAR filings ---

def get_sec_filings(symbol: str) -> list[dict]:
    """Fetch recent SEC filings (10-K, 10-Q, 8-K) from EDGAR."""
    cache_key = f"sec_{symbol}"
    cached = _cache_get(cache_key, SEC_TTL)
    if cached is not None:
        return cached

    result = []
    try:
        today = datetime.date.today()
        start = today - datetime.timedelta(days=365)
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22"
            f"&dateRange=custom&startdt={start}&enddt={today}"
            f"&forms=10-K,10-Q,8-K"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "trader-bot/1.0 (kywwilson@gmail.com)"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:5]:
            src = hit.get("_source", {})
            result.append({
                "form_type": src.get("form_type", ""),
                "filed_date": src.get("file_date", ""),
                "title": src.get("display_names", [""])[0] if src.get("display_names") else "",
            })
    except Exception as e:
        print(f"[FUNDAMENTALS] SEC EDGAR error for {symbol}: {e}")

    _cache_set(cache_key, result)
    return result


def get_filing_summary(symbol: str) -> str:
    """Get LLM summary of most recent 10-K/10-Q filing. Cached 7 days."""
    cache_key = f"filing_sum_{symbol}"
    cached = _cache_get(cache_key, SEC_TTL)
    if cached is not None:
        return cached

    filings = get_sec_filings(symbol)
    if not filings:
        _cache_set(cache_key, "")
        return ""

    # Find most recent 10-K or 10-Q
    target = None
    for f in filings:
        if f["form_type"] in ("10-K", "10-Q"):
            target = f
            break

    if not target:
        _cache_set(cache_key, "")
        return ""

    # Build a prompt from the filing metadata (we don't fetch full text to avoid
    # EDGAR rate limits — just ask LLM to summarize what it knows)
    from llm_client import call_llm

    prompt = (
        f"For {symbol}, the most recent {target['form_type']} was filed on {target['filed_date']}. "
        f"Based on your knowledge of {symbol}'s recent financials and SEC filings, "
        f"summarize the key risks, guidance changes, and notable items in 2-3 sentences."
    )
    summary = call_llm(prompt, system="You are a financial analyst summarizing SEC filings.") or ""

    _cache_set(cache_key, summary)
    return summary


# --- Format for LLM prompt ---

def format_fundamentals_for_llm(symbol: str, fundamentals: dict,
                                 insider: dict | None = None,
                                 filing_summary: str = "") -> str:
    """Format all fundamental data into a text block for LLM consumption."""
    lines = []

    pe = fundamentals.get("pe_ratio")
    if pe is not None:
        lines.append(f"P/E={pe:.1f}")

    pb = fundamentals.get("pb_ratio")
    if pb is not None:
        lines.append(f"P/B={pb:.1f}")

    mcap = fundamentals.get("market_cap")
    if mcap is not None:
        if mcap >= 1e12:
            lines.append(f"MktCap=${mcap/1e12:.1f}T")
        elif mcap >= 1e9:
            lines.append(f"MktCap=${mcap/1e9:.1f}B")
        elif mcap >= 1e6:
            lines.append(f"MktCap=${mcap/1e6:.0f}M")

    rg = fundamentals.get("revenue_growth")
    if rg is not None:
        lines.append(f"RevGrowth={rg:.1%}")

    eps = fundamentals.get("eps")
    if eps is not None:
        lines.append(f"EPS={eps:.2f}")

    dy = fundamentals.get("dividend_yield")
    if dy is not None:
        # yfinance returns either fraction (0.0037) or percentage (0.37) inconsistently
        if dy > 1:
            lines.append(f"DivYield={dy:.1f}%")
        else:
            lines.append(f"DivYield={dy:.2%}")

    sector = fundamentals.get("sector")
    if sector:
        lines.append(f"Sector={sector}")

    beta = fundamentals.get("beta")
    if beta is not None:
        lines.append(f"Beta={beta:.2f}")

    w52h = fundamentals.get("week52_high")
    w52l = fundamentals.get("week52_low")
    if w52h is not None and w52l is not None:
        lines.append(f"52wk=${w52l:.2f}-${w52h:.2f}")

    text = "Fundamentals: " + ", ".join(lines) if lines else "Fundamentals: limited data"

    if insider and insider.get("summary") != "N/A":
        text += f"\nInsider Activity: {insider['summary']}"

    if filing_summary:
        text += f"\nSEC Filing: {filing_summary}"

    return text
