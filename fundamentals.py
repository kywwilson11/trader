"""Fundamental data layer — yfinance + Financial Modeling Prep + SEC EDGAR.

Provides P/E, market cap, insider activity, and LLM-summarized SEC filings.
NOTE: the SEC filing-summary path is currently INOPERATIVE (EFTS query +
field mismatch — see get_sec_filings / get_filing_summary); it fails soft
to "" so no LLM filing summary is actually produced.
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

    # Skip yfinance for crypto — no PE/sector/beta data, and yfinance's SQLite
    # cache triggers driver conflicts in the jetson env.
    if asset_type != "crypto":
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
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
        with urllib.request.urlopen(req, timeout=10) as resp:
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
        with urllib.request.urlopen(req, timeout=10) as resp:
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
    """EFTS full-text search for filings that MENTION the symbol.

    NOT the symbol's own filing history: hits are relevance-ranked
    full-text matches from ANY filer whose documents mention the symbol,
    top 5 kept. Currently INOPERATIVE — the `dateRange=custom` query
    below draws HTTP 500 from EFTS (verified 2026-07-02), so this caches
    [] for 7 days; and even a working response keys the form under
    'file_type' (there is no 'form_type' field), so form_type is always
    "". Repairing it (submissions API, or fixed params + fields) is a
    behavior change deliberately not folded into this doc fix.
    """
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:5]:
            src = hit.get("_source", {})
            ciks = src.get("cik") or src.get("ciks") or []
            result.append({
                "form_type": src.get("form_type", ""),
                "filed_date": src.get("file_date", ""),
                "title": src.get("display_names", [""])[0] if src.get("display_names") else "",
                "doc_id": hit.get("_id", ""),       # "accession:filename"
                "cik": ciks[0] if isinstance(ciks, list) and ciks else ciks,
            })
    except Exception as e:
        print(f"[FUNDAMENTALS] SEC EDGAR error for {symbol}: {e}")

    _cache_set(cache_key, result)
    return result


def _fetch_filing_text(doc_id: str, cik, max_chars: int = 7000) -> str | None:
    """Fetch the actual filing document text from EDGAR.

    doc_id is the EFTS hit id ("0000320193-24-000123:aapl-20240928.htm").
    Prefers the Risk Factors / MD&A region when locatable. EDGAR allows
    10 req/s with a declared User-Agent; we make one request per symbol
    per week (7-day cache), which is far inside the limit.
    """
    try:
        accession, _, filename = doc_id.partition(':')
        if not accession or not filename or cik in (None, ''):
            return None
        acc_nodash = accession.replace('-', '')
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{int(cik)}/{acc_nodash}/{filename}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "trader-bot/1.0 (kywwilson@gmail.com)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        from bs4 import BeautifulSoup
        text = BeautifulSoup(html, 'html.parser').get_text(' ')
        text = ' '.join(text.split())
        if len(text) < 500:
            return None

        # Aim at Risk Factors / MD&A; otherwise take the body after the
        # boilerplate cover pages
        lower = text.lower()
        anchor = max(lower.find('risk factors'),
                     lower.find("management's discussion"))
        start = anchor if anchor > 0 else min(3000, len(text) // 10)
        return text[start:start + max_chars]
    except Exception as e:
        print(f"[FUNDAMENTALS] EDGAR document fetch failed: {e}")
        return None


def get_filing_summary(symbol: str) -> str:
    """LLM summary of the most recent 10-K/10-Q — from the REAL filing text.

    The old version asked the LLM to summarize the filing "based on your
    knowledge" WITHOUT fetching it. Current filings post-date every model's
    training cutoff, so the 'guidance changes' it produced were fabricated
    — and then fed into trade gating as fact for a week per cache entry.
    Design: fetch the document, summarize what it actually says; if the
    fetch fails, state only the verifiable fact (form + date), never
    invented specifics. Cached 7 days.

    RUNTIME REALITY: no summary has ever flowed from here — get_sec_filings
    is inoperative (returns [], and any hits would carry form_type ""), so
    the 10-K/10-Q filter below never matches and "" is cached. The fail-soft
    "" is the correct degraded output (no fabricated prompt section); the
    feature stays dormant pending the EFTS fix.
    """
    cache_key = f"filing_sum_{symbol}"
    cached = _cache_get(cache_key, SEC_TTL)
    if cached is not None:
        return cached

    filings = get_sec_filings(symbol)
    if not filings:
        _cache_set(cache_key, "")
        return ""

    # First 10-K/10-Q among the top-5 relevance-ranked hits — NOT the most
    # recent (EFTS sorts by relevance); with form_type always "" (see
    # get_sec_filings) this currently never matches
    target = None
    for f in filings:
        if f["form_type"] in ("10-K", "10-Q"):
            target = f
            break

    if not target:
        _cache_set(cache_key, "")
        return ""

    filing_text = _fetch_filing_text(target.get("doc_id", ""), target.get("cik"))
    if not filing_text:
        # Verifiable fact only — no from-memory speculation
        summary = (f"A {target['form_type']} was filed on {target['filed_date']} "
                   f"(contents not retrieved — do not speculate on specifics).")
        _cache_set(cache_key, summary)
        return summary

    from llm_client import call_llm
    prompt = (
        f"Below is an excerpt from {symbol}'s {target['form_type']} filed "
        f"{target['filed_date']}. Summarize the key risks, guidance changes, "
        f"and notable items in 2-3 sentences. Use ONLY this text — if it "
        f"doesn't cover something, do not fill the gap from memory.\n\n"
        f"---\n{filing_text}\n---"
    )
    summary = call_llm(prompt, temperature=0.2,
                       system="You are a financial analyst summarizing SEC "
                              "filings strictly from provided text.") or ""
    if summary:
        summary = f"[{target['form_type']} {target['filed_date']}] {summary}"

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
