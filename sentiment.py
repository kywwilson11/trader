"""News sentiment analysis for trading decisions.

Data sources:
- Crypto Fear & Greed Index (free, no API key needed)
- Finnhub news API (free tier: 60 calls/min, needs FINNHUB_API_KEY in .env)

Provides sentiment scoring and trade gating for crypto_loop.py and stock_loop.py.
"""
import json
import os
import time
import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Finnhub client (lazy init) ---
_finnhub_client = None

# --- Cache: key -> (timestamp, result) ---
_cache = {}
CACHE_TTL = 300  # 5 minutes


def _get_finnhub():
    """Lazy-init Finnhub client. Returns None if no API key."""
    global _finnhub_client
    if _finnhub_client is not None:
        return _finnhub_client

    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return None

    try:
        import finnhub
        _finnhub_client = finnhub.Client(api_key=api_key)
        return _finnhub_client
    except ImportError:
        print("[SENTIMENT] finnhub-python not installed")
        return None


# --- Keyword sentiment scoring ---

import math
import re as _re

_POSITIVE = frozenset({
    # Price action
    'surge', 'surges', 'surging', 'rally', 'rallies', 'rallying',
    'gain', 'gains', 'soar', 'soars', 'soaring', 'jump', 'jumps',
    'climbing', 'climbs', 'rises', 'rising', 'rebound', 'rebounds',
    'breakout', 'moon', 'mooning', 'skyrocket',
    # Fundamentals
    'bull', 'bullish', 'profit', 'profits', 'profitable', 'beat',
    'beats', 'record', 'growth', 'boost', 'boosts', 'boosting',
    'strong', 'strength', 'positive', 'optimistic', 'optimism',
    'recovery', 'recovering', 'outperform', 'outperforms',
    'upgrade', 'upgrades', 'upgraded',
    'success', 'successful', 'milestone', 'exceeded', 'exceeds',
    'upbeat', 'robust', 'stellar', 'impressive', 'blowout',
    'smashes', 'crushes', 'crush',
    # Business
    'innovation', 'partnership', 'deal', 'approval', 'approved',
    'launch', 'launches', 'expand', 'expansion', 'adoption',
    'inflow', 'inflows', 'accumulation', 'accumulates', 'accumulating',
    'institutional', 'buy', 'buying', 'accumulate', 'etf',
    # Crypto-specific
    'halving', 'airdrop', 'staking', 'defi',
    # Modifiers
    'tailwind', 'tailwinds', 'upside', 'overweight',
    'sustainable', 'momentum',
    # Analyst / employment
    'raised', 'raises', 'hired', 'hiring',
    'victory', 'wins',
})

_NEGATIVE = frozenset({
    # Price action
    'crash', 'crashes', 'crashing', 'plunge', 'plunges', 'plunging',
    'drop', 'drops', 'dropping', 'decline', 'declines', 'declining',
    'tumble', 'tumbles', 'tumbling', 'sink', 'sinks', 'sinking',
    'slide', 'slides', 'sliding', 'slip', 'slips', 'slipping',
    'selloff', 'sell-off', 'dumping', 'dump', 'dumps', 'plummets',
    'nosedive', 'freefall', 'rout', 'bloodbath', 'carnage', 'tanking',
    'wipes', 'wiped', 'erased', 'erases',
    # Fundamentals
    'bear', 'bearish', 'loss', 'losses', 'miss', 'misses', 'missed',
    'downgrade', 'downgrades', 'downgraded', 'weak', 'weakness',
    'negative', 'pessimistic', 'pessimism', 'ugly', 'terrible',
    'worst', 'disappointing', 'disappointed', 'disappoints',
    'lackluster', 'dismal',
    'underperform', 'underperforms', 'underweight',
    'slashes', 'slashed', 'slash',
    # Business / macro
    'recession', 'bankruptcy', 'bankrupt', 'fraud', 'fraudulent',
    'hack', 'hacked', 'exploit', 'exploited', 'regulation',
    'ban', 'banned', 'warning', 'warns', 'warned', 'crisis',
    'investigation', 'lawsuit', 'sues', 'sued', 'suing',
    'layoff', 'layoffs', 'cut', 'cuts',
    'outflow', 'outflows', 'fine', 'fined', 'subpoena', 'default',
    'inflation', 'tariff', 'tariffs', 'war', 'sanctions', 'shutdown',
    'fear', 'fears', 'risk', 'risks', 'risky', 'concern', 'concerns',
    'uncertainty', 'volatile', 'volatility', 'contagion', 'bubble',
    'overvalued', 'sell', 'selling',
    'bleed', 'bleeds', 'bleeding',
    'freezes', 'freeze', 'frozen',
    'zero', 'worthless',
    'trap',
    # Modifiers
    'headwind', 'headwinds', 'downside', 'downbeat', 'grim',
    'dire', 'ominous', 'trouble', 'troubled', 'struggling',
    'cautious', 'caution',
    'slowing', 'slower', 'slowdown',
})

# Phrases scored as a unit (checked before single-word matching)
_POSITIVE_PHRASES = [
    ('all time high', 1.5), ('all-time high', 1.5),
    ('beat expectations', 1.5), ('beats expectations', 1.5),
    ('strong buy', 1.5), ('price target raised', 1.5),
    ('raises price target', 1.5),
    ('short squeeze', 1.0), ('green light', 1.0),
    ('better than expected', 1.5), ('revenue beat', 1.5),
    ('earnings beat', 1.5), ('guidance raised', 1.5),
    ('rate cut', 1.0), ('rate cuts', 1.0),
    ('cuts interest rates', 1.5), ('cuts rates', 1.5),
    ('cut interest rates', 1.5), ('cut rates', 1.5),
    ('unemployment falls', 1.0), ('unemployment drops', 1.0),
    ('blows past', 1.0), ('crush expectations', 1.5),
    ('pile into', 1.0), ('piling into', 1.0),
]

_NEGATIVE_PHRASES = [
    ('pretty ugly', -1.5), ('not good', -1.0), ('not great', -1.0),
    ('death cross', -1.5), ('going down', -1.0), ('sell off', -1.0),
    ('missed expectations', -1.5), ('misses expectations', -1.5),
    ('price target cut', -1.5), ('price target lowered', -1.5),
    ('price target slashed', -1.5),
    ('guidance lowered', -1.5), ('guidance cut', -1.5),
    ('revenue miss', -1.5), ('earnings miss', -1.5),
    ('worse than expected', -1.5), ('worst since', -1.5),
    ('bear market', -1.5), ('margin call', -1.5),
    ('not saying downside overdone', -1.0),
    ('downside overdone', -0.5),
    ('bull trap', -1.5),
    ('go to zero', -1.5), ('goes to zero', -1.5),
    ('short interest', -1.0),
    ('dries up', -1.0), ('dried up', -1.0),
    ('freezes withdrawals', -1.5), ('frozen withdrawals', -1.5),
    ('no longer', -0.5),
    ('not happening', -1.0), ('not going well', -1.0),
    ('not justified', -1.0),
    ('slashes price target', -2.0), ('slashed price target', -2.0),
    ('price target slashed', -2.0),
    ('slowing momentum', -1.0), ('slowing growth', -1.0),
]

# Negation words — flip sentiment of the next keyword within 3 words
_NEGATORS = frozenset({
    'not', "n't", 'no', 'never', 'neither', 'nor', 'hardly', 'barely',
    "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
    "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
})

# Strip punctuation for clean word matching
_PUNCT = _re.compile(r"[^\w\s'-]")

# Pre-compiled negation prefix pattern (checks if a negator appears before a phrase)
_NEG_PREFIX = _re.compile(
    r'\b(?:' + '|'.join(_re.escape(n) for n in _NEGATORS) + r')\s+',
    _re.IGNORECASE,
)

# Pre-compiled phrase patterns with word boundaries (avoids substring matches)
_POS_PHRASE_RES = [(_re.compile(r'\b' + _re.escape(p) + r'\b', _re.IGNORECASE), w)
                   for p, w in _POSITIVE_PHRASES]
_NEG_PHRASE_RES = [(_re.compile(r'\b' + _re.escape(p) + r'\b', _re.IGNORECASE), w)
                   for p, w in _NEGATIVE_PHRASES]


def _score_text(text):
    """Score a single text string. Returns a float in roughly (-1, 1).

    Uses phrase matching, negation-aware keyword scoring, and tanh smoothing.
    """
    text_lower = text.lower()

    raw_score = 0.0

    # Phase 1: Phrase matching (higher weight, word-boundary, negation-aware)
    for pat, weight in _POS_PHRASE_RES:
        m = pat.search(text_lower)
        if m:
            prefix = text_lower[max(0, m.start() - 15):m.start()]
            if _NEG_PREFIX.search(prefix):
                raw_score -= weight * 0.7  # negated positive → negative
            else:
                raw_score += weight
    for pat, weight in _NEG_PHRASE_RES:
        if pat.search(text_lower):
            raw_score += weight  # weight is already negative

    # Phase 2: Negation-aware single-word matching (bidirectional)
    clean = _PUNCT.sub(' ', text_lower)
    words = clean.split()

    # Build per-word scores, then apply negation in both directions
    word_scores = []  # (index, base_score)
    negator_positions = []

    for i, word in enumerate(words):
        if word in _NEGATORS or word.endswith("n't"):
            negator_positions.append(i)
        elif word in _POSITIVE:
            word_scores.append((i, 1.0))
        elif word in _NEGATIVE:
            word_scores.append((i, -1.0))

    # Apply negation: a negator flips the nearest sentiment word within 3 positions
    used_negators = set()
    for idx, base in word_scores:
        for ni in negator_positions:
            if ni in used_negators:
                continue
            dist = abs(idx - ni)
            # Negator must be within 3 words and not on the same word
            if 0 < dist <= 3:
                # Count non-keyword words between negator and keyword
                lo, hi = min(idx, ni), max(idx, ni)
                filler = sum(1 for j in range(lo + 1, hi)
                             if words[j] not in _POSITIVE and words[j] not in _NEGATIVE
                             and words[j] not in _NEGATORS)
                if filler <= 2:  # allow up to 2 filler words between
                    raw_score -= base * 0.7  # flip: cancel original + add opposite
                    raw_score -= base * 1.0   # (net: -1.7 * base direction)
                    used_negators.add(ni)
                    break
        else:
            # No negation applied — use base score
            raw_score += base

    # Smooth with tanh: maps any raw sum to (-1, 1).
    # Scale by sqrt(word_count) so longer text doesn't saturate to ±1.0.
    # Headlines (~10 words): scale ≈ 0.4 (unchanged behavior)
    # Articles (~200 words): scale ≈ 0.09 (dampened, stays granular)
    n_words = max(len(words), 1)
    scale = 0.4 / math.sqrt(n_words / 10)
    return math.tanh(raw_score * scale)


_HTML_TAG = _re.compile(r'<[^>]+>')
_WHITESPACE_ONLY = _re.compile(r'^[\s\W]*$')


_URL_PATTERN = _re.compile(r'^https?://')

# --- Full article fetching ---
_article_cache = {}  # url -> (timestamp, text or None)
_ARTICLE_CACHE_TTL = 1800  # 30 min — articles don't change

_USER_AGENT = (
    'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
)


def _fetch_article_text(url, timeout=5):
    """Fetch article URL and extract body text using BeautifulSoup.

    Returns extracted text string, or None on failure.
    Results are cached for 30 minutes.
    """
    if not url:
        return None

    now = time.time()
    if url in _article_cache:
        ts, text = _article_cache[url]
        if now - ts < _ARTICLE_CACHE_TTL:
            return text

    try:
        from bs4 import BeautifulSoup

        resp = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': _USER_AGENT},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            _article_cache[url] = (now, None)
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove non-content elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                         'aside', 'iframe', 'form', 'noscript']):
            tag.decompose()

        # Try common article body selectors
        body = None
        for selector in ['article', '[role="main"]', '.article-body',
                         '.post-content', '.entry-content', '.story-body',
                         '.article-content', 'main']:
            body = soup.select_one(selector)
            if body:
                break

        if body is None:
            body = soup.body or soup

        # Extract paragraph text, filtering junk
        paragraphs = body.find_all('p')
        clean_paragraphs = []
        for p in paragraphs:
            t = p.get_text(strip=True)
            # Skip very short paragraphs (captions, links, CTAs)
            if len(t) < 40:
                continue
            # Skip boilerplate patterns
            t_lower = t.lower()
            if any(junk in t_lower for junk in (
                'never miss an important', 'find winning stocks',
                'sign up for', 'subscribe to', 'newsletter',
                'getty images', 'via getty', 'shutterstock',
                'in your inbox', 'related stories',
                'click here', 'read more:', 'read next',
                'simply wall st', 'seeking alpha',
                'download the app', 'join premium',
                'trusted by over', 'million investors',
            )):
                continue
            clean_paragraphs.append(t)
        text = ' '.join(clean_paragraphs)

        # Validate: need at least a sentence worth of content
        if len(text) < 50:
            _article_cache[url] = (now, None)
            return None

        # Cap at ~2000 chars to keep scoring fast
        if len(text) > 2000:
            text = text[:2000]

        _article_cache[url] = (now, text)
        return text

    except Exception:
        _article_cache[url] = (now, None)
        return None


def _validate_text(text):
    """Validate that text is scorable. Returns cleaned text or None if invalid."""
    if not text or not isinstance(text, str):
        return None
    # Strip HTML tags (some Finnhub articles have HTML in summary)
    text = _HTML_TAG.sub(' ', text).strip()
    # Too short to be meaningful
    if len(text) < 10:
        return None
    # Mostly non-word characters (URLs, garbage)
    if _WHITESPACE_ONLY.match(text):
        return None
    # Raw URLs are not scorable text
    if _URL_PATTERN.match(text):
        return None
    # Mostly non-ASCII (likely wrong encoding)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.5:
        return None
    return text


def _deduplicate_articles(articles):
    """Remove duplicate articles by normalized headline. First occurrence wins."""
    seen = set()
    unique = []
    for a in articles:
        key = a.get('headline', '').strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique


def _llm_score_batch(articles):
    """Batch-score articles via the configured LLM. Returns list[float] or None.

    For each article, uses the richest content available:
      full article text > summary > headline (always included).
    Article bodies are fetched in parallel with short timeouts.
    Falls back to None on any failure (caller should use keyword scoring).
    """
    try:
        from llm_client import call_llm
    except ImportError:
        return None

    if not articles:
        return None

    # Fetch full article text in parallel (cached, 5s timeout each)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    urls = [a.get('url', '') for a in articles]
    full_texts = [None] * len(articles)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_article_text, u): i
                   for i, u in enumerate(urls) if u}
        for fut in as_completed(futures, timeout=15):
            try:
                full_texts[futures[fut]] = fut.result()
            except Exception:
                pass

    n = len(articles)
    fetched = sum(1 for t in full_texts if t)
    print(f"[SENTIMENT] Fetched {fetched}/{n} article bodies for LLM")

    lines = []
    for i, a in enumerate(articles):
        h = ' '.join(a.get('headline', '').split())
        s = ' '.join(a.get('summary', '').split())
        body = full_texts[i]

        # Build best-available content: headline is always first
        parts = [f"{i + 1}. {h}"]
        if body:
            # Full text already has summary content; cap at 500 chars
            parts.append(body[:500])
        elif s:
            parts.append(s)
        lines.append(" — ".join(parts))

    prompt = (
        f"Score each article's financial sentiment from -1.0 (very bearish) "
        f"to 1.0 (very bullish). 0.0 = neutral.\n"
        f'Return ONLY a JSON object mapping article number to score, '
        f'e.g. {{"1": 0.3, "2": -0.5, "3": 0.0}}\n\n'
        + "\n".join(lines)
    )

    result = call_llm(
        prompt,
        system="Financial sentiment scorer. Return only a JSON object mapping article number strings to float scores. No explanation.",
        max_tokens=max(256, n * 12),
    )
    if not result:
        return None

    try:
        text = result.strip()
        # Strip markdown code fences if present
        if '```' in text:
            for part in text.split('```')[1:]:
                stripped = part.strip()
                if stripped.startswith('json'):
                    stripped = stripped[4:].strip()
                if stripped.startswith('{'):
                    text = stripped
                    break

        data = json.loads(text)
        if isinstance(data, dict):
            # Look up each article by its 1-based number; default 0.0 if missing
            scores = []
            for i in range(1, n + 1):
                raw = data.get(str(i), data.get(i, 0.0))
                scores.append(max(-1.0, min(1.0, float(raw))))
            matched = sum(1 for i in range(1, n + 1) if str(i) in data or i in data)
            if matched < n * 0.5:
                print(f"[SENTIMENT] LLM only scored {matched}/{n} articles, falling back")
                return None
            if matched < n:
                print(f"[SENTIMENT] LLM scored {matched}/{n} articles (missing default to 0.0)")
            return scores
        print(f"[SENTIMENT] LLM returned {type(data).__name__}, expected dict")
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[SENTIMENT] LLM score parse failed: {e}")

    return None


def score_article_batch(articles):
    """Score articles for display. Returns list of per-article float scores.

    Uses LLM batch scoring when available (one API call for all articles).
    Falls back to keyword scoring with full-text fetch for accuracy.
    """
    if not articles:
        return []

    llm_scores = _llm_score_batch(articles)
    if llm_scores is not None:
        print(f"[SENTIMENT] LLM scored {len(articles)} articles")
        return llm_scores

    # Fallback: keyword scoring with full-text fetch
    print(f"[SENTIMENT] Keyword scoring {len(articles)} articles (LLM unavailable)")
    scores = []
    for a in articles:
        headline = _validate_text(a.get('headline', ''))
        summary = _validate_text(a.get('summary', ''))
        if headline is None and summary is None:
            scores.append(0.0)
            continue
        h = _score_text(headline) if headline else 0.0
        s = _score_text(summary) if summary else 0.0
        full_text = _fetch_article_text(a.get('url', ''))
        if full_text:
            f = _score_text(full_text)
            scores.append(h * 0.25 + s * 0.25 + f * 0.50)
        else:
            scores.append(h * 0.6 + s * 0.4)
    return scores


def _score_articles(articles):
    """Score a list of Finnhub news articles. Deduplicates and aggregates.

    Used by trading loops (get_news_sentiment, get_market_sentiment).
    Tries LLM batch scoring first. Falls back to keyword scoring with
    full-text article fetch for higher accuracy.

    Returns dict:
        sentiment_score: float in (-1, 1), average across articles
        article_count: int
        positive_ratio: float in [0, 1]
        negative_ratio: float in [0, 1]
    """
    articles = _deduplicate_articles(articles)

    if not articles:
        return {
            'sentiment_score': 0.0,
            'article_count': 0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
        }

    # Try LLM batch scoring first (one API call, cost-efficient)
    scores = _llm_score_batch(articles)

    if scores is None:
        # Fallback: keyword scoring with full-text fetch for accuracy
        scores = []
        for article in articles:
            headline = _validate_text(article.get('headline', ''))
            summary = _validate_text(article.get('summary', ''))

            if headline is None and summary is None:
                continue

            h_score = _score_text(headline) if headline else 0.0
            s_score = _score_text(summary) if summary else 0.0

            full_text = _fetch_article_text(article.get('url', ''))
            if full_text:
                f_score = _score_text(full_text)
                combined = h_score * 0.25 + s_score * 0.25 + f_score * 0.50
            else:
                combined = h_score * 0.6 + s_score * 0.4

            scores.append(combined)

    if not scores:
        return {
            'sentiment_score': 0.0,
            'article_count': 0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
        }

    avg = sum(scores) / len(scores)
    pos_count = sum(1 for s in scores if s > 0.05)
    neg_count = sum(1 for s in scores if s < -0.05)
    n = len(scores)

    return {
        'sentiment_score': avg,
        'article_count': n,
        'positive_ratio': pos_count / n,
        'negative_ratio': neg_count / n,
    }


# --- Fear & Greed Index (crypto only, free) ---

def get_fear_greed():
    """Fetch the Crypto Fear & Greed Index (0-100).

    Returns dict with 'value' (int 0-100), 'label' (str), or None on error.
    0-24 = Extreme Fear, 25-49 = Fear, 50 = Neutral, 51-74 = Greed, 75-100 = Extreme Greed
    """
    now = time.time()
    if '__fng__' in _cache:
        ts, result = _cache['__fng__']
        if now - ts < CACHE_TTL:
            return result

    try:
        resp = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
        data = resp.json()['data'][0]
        result = {
            'value': int(data['value']),
            'label': data['value_classification'],
        }
        _cache['__fng__'] = (now, result)
        return result
    except Exception as e:
        print(f"[SENTIMENT] Fear & Greed error: {e}")
        return None


# --- Finnhub news ---

def get_news_sentiment(symbol, asset_type='crypto'):
    """Get news sentiment for a symbol from Finnhub.

    Args:
        symbol: Trading symbol (e.g. 'BTC/USD', 'TSLA')
        asset_type: 'crypto' or 'stock'

    Returns dict with sentiment metrics, or None if unavailable.
    """
    now = time.time()
    cache_key = f'news_{symbol}'
    if cache_key in _cache:
        ts, result = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return result

    client = _get_finnhub()
    if client is None:
        return None

    try:
        today = datetime.date.today()
        week_ago = today - datetime.timedelta(days=7)

        if asset_type == 'crypto':
            # General crypto news, filter for relevant symbol
            articles = client.general_news('crypto', min_id=0)
            base = symbol.replace('/USD', '').replace('-USD', '').lower()
            relevant = [a for a in articles
                        if base in a.get('headline', '').lower()
                        or base in a.get('summary', '').lower()]
            # Fall back to all crypto news if not enough symbol-specific
            if len(relevant) < 3:
                relevant = articles[:20]
        else:
            # Stock: company-specific news
            clean_sym = symbol.replace('/', '')
            articles = client.company_news(
                clean_sym,
                _from=week_ago.strftime('%Y-%m-%d'),
                to=today.strftime('%Y-%m-%d'),
            )
            relevant = articles[:30]

        result = _score_articles(relevant)
        _cache[cache_key] = (now, result)
        return result

    except Exception as e:
        print(f"[SENTIMENT] News error for {symbol}: {e}")
        return None


def get_market_sentiment():
    """Get overall market sentiment from Finnhub general news.

    Returns sentiment dict or None.
    """
    now = time.time()
    if '__market__' in _cache:
        ts, result = _cache['__market__']
        if now - ts < CACHE_TTL:
            return result

    client = _get_finnhub()
    if client is None:
        return None

    try:
        articles = client.general_news('general', min_id=0)
        result = _score_articles(articles[:30])
        _cache['__market__'] = (now, result)
        return result
    except Exception as e:
        print(f"[SENTIMENT] Market sentiment error: {e}")
        return None


# --- Combined sentiment gate for trading decisions ---

def sentiment_gate(symbol, asset_type='crypto'):
    """Compute a trade multiplier based on sentiment.

    Returns tuple: (multiplier: float, reasons: list[str])
        0.15 = severe reduce (catastrophic news, e.g. hack/fraud)
        0.5  = reduce position size (negative sentiment)
        1.0  = normal (neutral or no data)
        1.2  = increase position (positive sentiment)
        1.5  = max increase (strong positive + calm market)

    Design philosophy for profit optimization:
    - The ML models are the primary signal (technical, leading indicator).
      Sentiment is a position-sizing modifier, NOT a trade blocker.
    - Never fully block (min 0.15x): the ML model may be catching a
      bounce/reversal that news hasn't priced in yet.
    - Reward positive confirmation aggressively: when technicals AND news
      agree bullish, size up — momentum compounds.
    - Wide neutral zone: most news is noise. Only act on strong signals.
    - FnG is a real liquidity/spread signal — keep it multiplicative.
    - Symbol news is most actionable — strong weight.
    - Market news is diffuse — very light touch, wide neutral zone.
    """
    multiplier = 1.0
    reasons = []

    # --- Crypto: Fear & Greed Index (liquidity/spread proxy) ---
    if asset_type == 'crypto':
        fng = get_fear_greed()
        if fng is not None:
            val = fng['value']
            if val <= 10:
                multiplier *= 0.35
                reasons.append(f"FnG={val}(extreme_fear)")
            elif val <= 25:
                multiplier *= 0.55
                reasons.append(f"FnG={val}(fear)")
            elif val <= 40:
                multiplier *= 0.8
                reasons.append(f"FnG={val}(cautious)")
            elif val >= 90:
                multiplier *= 0.7
                reasons.append(f"FnG={val}(extreme_greed)")
            elif val >= 75:
                multiplier *= 0.9
                reasons.append(f"FnG={val}(greed)")
            else:
                reasons.append(f"FnG={val}(normal)")

    # --- Symbol-specific news sentiment (strongest signal) ---
    news = get_news_sentiment(symbol, asset_type)
    if news is not None and news['article_count'] > 0:
        score = news['sentiment_score']
        if score <= -0.5:
            multiplier *= 0.15  # Catastrophic (hack, fraud, bankruptcy)
            reasons.append(f"sym_news={score:+.2f}(catastrophic)")
        elif score <= -0.3:
            multiplier *= 0.35  # Heavy negative
            reasons.append(f"sym_news={score:+.2f}(bearish)")
        elif score <= -0.1:
            multiplier *= 0.7   # Mildly negative
            reasons.append(f"sym_news={score:+.2f}(cautious)")
        elif score >= 0.4:
            multiplier *= 1.35  # Strong positive — conviction boost
            reasons.append(f"sym_news={score:+.2f}(strong_bull)")
        elif score >= 0.2:
            multiplier *= 1.2   # Positive confirmation
            reasons.append(f"sym_news={score:+.2f}(bullish)")
        else:
            # -0.1 to 0.2: wide neutral zone — most news is noise
            reasons.append(f"sym_news={score:+.2f}(neutral)")

    # --- Market-wide sentiment (very light touch) ---
    market = get_market_sentiment()
    if market is not None and market['article_count'] > 0:
        mscore = market['sentiment_score']
        if mscore <= -0.4:
            multiplier *= 0.85
            reasons.append(f"market={mscore:+.2f}(bearish)")
        elif mscore >= 0.4:
            multiplier *= 1.1
            reasons.append(f"market={mscore:+.2f}(bullish)")
        else:
            reasons.append(f"market={mscore:+.2f}(neutral)")

    # Clamp: never fully block (ML signal always gets a chance), cap upside
    multiplier = max(0.15, min(1.5, multiplier))

    return multiplier, reasons
