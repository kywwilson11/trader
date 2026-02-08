"""News sentiment analysis for trading decisions.

Data sources:
- Crypto Fear & Greed Index (free, no API key needed)
- Finnhub news API (free tier: 60 calls/min, needs FINNHUB_API_KEY in .env)

Provides sentiment scoring and trade gating for crypto_loop.py and stock_loop.py.
"""
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
    'breakout', 'moon', 'mooning', 'skyrocket', 'spike', 'spikes',
    # Fundamentals
    'bull', 'bullish', 'profit', 'profits', 'profitable', 'beat',
    'beats', 'record', 'growth', 'boost', 'boosts', 'strong',
    'strength', 'positive', 'optimistic', 'optimism', 'recovery',
    'outperform', 'outperforms', 'upgrade', 'upgrades', 'upgraded',
    'success', 'successful', 'milestone', 'exceeded', 'exceeds',
    'upbeat', 'robust', 'stellar', 'impressive', 'blowout',
    # Business
    'innovation', 'partnership', 'deal', 'approval', 'approved',
    'launch', 'launches', 'expand', 'expansion', 'adoption',
    'inflow', 'inflows', 'accumulation', 'institutional',
    'buy', 'buying', 'accumulate', 'etf',
    # Crypto-specific
    'halving', 'airdrop', 'staking', 'defi',
    # Modifiers
    'tailwind', 'tailwinds', 'upside', 'overweight',
    'sustainable', 'momentum',
})

_NEGATIVE = frozenset({
    # Price action
    'crash', 'crashes', 'crashing', 'plunge', 'plunges', 'plunging',
    'drop', 'drops', 'dropping', 'decline', 'declines', 'declining',
    'tumble', 'tumbles', 'tumbling', 'sink', 'sinks', 'sinking',
    'slide', 'slides', 'sliding', 'slip', 'slips', 'slipping',
    'selloff', 'sell-off', 'dumping', 'dump', 'dumps', 'plummets',
    'nosedive', 'freefall', 'rout', 'bloodbath', 'carnage', 'tanking',
    # Fundamentals
    'bear', 'bearish', 'loss', 'losses', 'miss', 'misses', 'missed',
    'downgrade', 'downgrades', 'downgraded', 'weak', 'weakness',
    'negative', 'pessimistic', 'pessimism', 'ugly', 'terrible',
    'worst', 'disappointing', 'disappointed', 'lackluster', 'dismal',
    'underperform', 'underperforms', 'underweight',
    # Business / macro
    'recession', 'bankruptcy', 'bankrupt', 'fraud', 'fraudulent',
    'hack', 'hacked', 'exploit', 'exploited', 'regulation',
    'ban', 'banned', 'warning', 'warns', 'warned', 'crisis',
    'investigation', 'lawsuit', 'layoff', 'layoffs', 'cut', 'cuts',
    'outflow', 'outflows', 'fine', 'fined', 'subpoena', 'default',
    'inflation', 'tariff', 'tariffs', 'war', 'sanctions', 'shutdown',
    'fear', 'fears', 'risk', 'risks', 'risky', 'concern', 'concerns',
    'uncertainty', 'volatile', 'volatility', 'contagion', 'bubble',
    'overvalued', 'sell', 'selling',
    # Modifiers
    'headwind', 'headwinds', 'downside', 'downbeat', 'grim',
    'dire', 'ominous', 'trouble', 'troubled', 'struggling',
})

# Phrases scored as a unit (checked before single-word matching)
_POSITIVE_PHRASES = [
    ('all time high', 1.5), ('all-time high', 1.5), ('ath', 1.0),
    ('beat expectations', 1.5), ('beats expectations', 1.5),
    ('strong buy', 1.5), ('price target raised', 1.5),
    ('short squeeze', 1.0), ('green light', 1.0),
    ('better than expected', 1.5), ('revenue beat', 1.5),
    ('earnings beat', 1.5), ('guidance raised', 1.5),
]

_NEGATIVE_PHRASES = [
    ('pretty ugly', -1.5), ('not good', -1.0), ('not great', -1.0),
    ('death cross', -1.5), ('going down', -1.0), ('sell off', -1.0),
    ('missed expectations', -1.5), ('misses expectations', -1.5),
    ('price target cut', -1.5), ('price target lowered', -1.5),
    ('guidance lowered', -1.5), ('guidance cut', -1.5),
    ('revenue miss', -1.5), ('earnings miss', -1.5),
    ('worse than expected', -1.5), ('worst since', -1.5),
    ('bear market', -1.5), ('margin call', -1.5),
    ('not saying downside overdone', -1.0),
    ('downside overdone', -0.5),  # without negation, slightly negative
]

# Negation words — flip sentiment of the next keyword within 3 words
_NEGATORS = frozenset({
    'not', "n't", 'no', 'never', 'neither', 'nor', 'hardly', 'barely',
    "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
    "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
})

# Strip punctuation for clean word matching
_PUNCT = _re.compile(r"[^\w\s'-]")


def _score_text(text):
    """Score a single text string. Returns a float in roughly (-1, 1).

    Uses phrase matching, negation-aware keyword scoring, and tanh smoothing.
    """
    text_lower = text.lower()

    raw_score = 0.0

    # Phase 1: Phrase matching (higher weight, checked first)
    # Negation-aware: if a negator appears within 4 chars before the phrase, flip it
    _neg_prefix = _re.compile(r'\b(?:' + '|'.join(_re.escape(n) for n in _NEGATORS) + r')\s+')
    for phrase, weight in _POSITIVE_PHRASES:
        idx = text_lower.find(phrase)
        if idx >= 0:
            prefix = text_lower[max(0, idx - 15):idx]
            if _neg_prefix.search(prefix):
                raw_score -= weight * 0.7  # negated positive → negative
            else:
                raw_score += weight
    for phrase, weight in _NEGATIVE_PHRASES:
        if phrase in text_lower:
            raw_score += weight  # weight is already negative

    # Phase 2: Negation-aware single-word matching
    clean = _PUNCT.sub(' ', text_lower)
    words = clean.split()
    negation_window = 0  # countdown: how many words the negation applies to

    for word in words:
        # Check if this word is a negator
        if word in _NEGATORS or word.endswith("n't"):
            negation_window = 3  # affects next 3 words
            continue

        is_pos = word in _POSITIVE
        is_neg = word in _NEGATIVE

        if is_pos or is_neg:
            if negation_window > 0:
                # Flip: positive becomes negative and vice versa (at 0.7x weight)
                if is_pos:
                    raw_score -= 0.7
                else:
                    raw_score += 0.7
                negation_window = 0  # consumed by this keyword
            else:
                if is_pos:
                    raw_score += 1.0
                else:
                    raw_score -= 1.0
        elif negation_window > 0:
            negation_window -= 1

    # Smooth with tanh: maps any raw sum to (-1, 1) with granular values
    # Scale factor 0.4 means: 1 keyword ≈ ±0.38, 2 ≈ ±0.66, 3 ≈ ±0.83
    return math.tanh(raw_score * 0.4)


def _score_articles(articles):
    """Score a list of Finnhub news articles.

    Returns dict:
        sentiment_score: float in (-1, 1), average across articles
        article_count: int
        positive_ratio: float in [0, 1]
        negative_ratio: float in [0, 1]
    """
    if not articles:
        return {
            'sentiment_score': 0.0,
            'article_count': 0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
        }

    scores = []
    for article in articles:
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        # Headline carries more weight than summary
        h_score = _score_text(headline)
        s_score = _score_text(summary) if summary else 0.0
        combined = h_score * 0.6 + s_score * 0.4
        scores.append(combined)

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

    Returns a float:
        0.0  = block trade entirely (extreme negative sentiment)
        0.5  = reduce position size (negative sentiment)
        1.0  = normal (neutral or no data)
        1.25 = slightly increase position (positive sentiment)
        1.5  = increase position (strong positive sentiment)

    Uses Fear & Greed Index for crypto, Finnhub news for both.
    Gracefully returns 1.0 if no sentiment data is available.
    """
    multiplier = 1.0
    reasons = []

    # --- Crypto: Fear & Greed Index ---
    if asset_type == 'crypto':
        fng = get_fear_greed()
        if fng is not None:
            val = fng['value']
            if val <= 15:
                # Extreme fear — contrarian buy signal (slightly bullish)
                multiplier *= 1.15
                reasons.append(f"FnG={val}(extreme_fear/contrarian)")
            elif val <= 30:
                # Fear — moderate caution
                multiplier *= 0.85
                reasons.append(f"FnG={val}(fear)")
            elif val >= 85:
                # Extreme greed — contrarian sell signal (reduce buys)
                multiplier *= 0.5
                reasons.append(f"FnG={val}(extreme_greed/reduce)")
            elif val >= 70:
                # Greed — slight caution
                multiplier *= 0.9
                reasons.append(f"FnG={val}(greed)")
            else:
                reasons.append(f"FnG={val}(neutral)")

    # --- Finnhub news sentiment ---
    news = get_news_sentiment(symbol, asset_type)
    if news is not None and news['article_count'] > 0:
        score = news['sentiment_score']
        if score <= -0.4:
            multiplier *= 0.0  # Block trade
            reasons.append(f"news={score:+.2f}(block)")
        elif score <= -0.2:
            multiplier *= 0.6
            reasons.append(f"news={score:+.2f}(bearish)")
        elif score >= 0.4:
            multiplier *= 1.3
            reasons.append(f"news={score:+.2f}(bullish)")
        elif score >= 0.2:
            multiplier *= 1.1
            reasons.append(f"news={score:+.2f}(positive)")
        else:
            reasons.append(f"news={score:+.2f}(neutral)")

    # --- Market-wide sentiment (light weight) ---
    market = get_market_sentiment()
    if market is not None and market['article_count'] > 0:
        mscore = market['sentiment_score']
        if mscore <= -0.3:
            multiplier *= 0.8
            reasons.append(f"market={mscore:+.2f}(bearish)")
        elif mscore >= 0.3:
            multiplier *= 1.1
            reasons.append(f"market={mscore:+.2f}(bullish)")

    # Clamp to [0, 1.5]
    multiplier = max(0.0, min(1.5, multiplier))

    return multiplier, reasons
