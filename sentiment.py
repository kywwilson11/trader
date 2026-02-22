"""News sentiment analysis for trading decisions.

Data sources:
- Crypto Fear & Greed Index (free, no API key needed)
- Finnhub news API (free tier: 60 calls/min, needs FINNHUB_API_KEY in .env)

Provides sentiment scoring and trade gating for crypto_loop.py and stock_loop.py.
"""
import collections
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

# --- LLM retry queue: (cache_key, articles, queued_at) ---
_llm_retry_queue: collections.deque = collections.deque(maxlen=50)


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
    # Nuanced bearish phrases (avoid false positives from "growth" mentions)
    ("wouldn't touch", -1.5), ("wouldn't buy", -1.5),
    ("wouldn't invest", -1.5),
    ('stay away', -1.0), ('10-foot pole', -1.5),
    ('too much of a premium', -1.0),
    ('too expensive', -1.0), ('overpriced', -1.0),
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
        m = pat.search(text_lower)
        if m:
            prefix = text_lower[max(0, m.start() - 15):m.start()]
            if _NEG_PREFIX.search(prefix):
                raw_score -= weight * 0.7  # negated negative → positive
            else:
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
        key = (a.get('headline') or '').strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique


_LLM_CHUNK_SIZE = 80  # score all articles in one API call to minimize RPM usage


def _parse_llm_json(raw_text):
    """Robustly parse JSON from LLM output. Handles common issues:
    - Markdown code fences (```json ... ```)
    - Single quotes instead of double quotes
    - Trailing text after JSON object
    - Extra whitespace and newlines
    Returns parsed dict or None.
    """
    import ast

    text = raw_text.strip()

    # Reject obviously non-JSON responses
    if len(text) < 5 or '{' not in text:
        print(f"[SENTIMENT] LLM returned non-JSON ({len(text)} chars): {text[:80]}")
        return None

    # Strip markdown code fences
    if '```' in text:
        for part in text.split('```')[1:]:
            stripped = part.strip()
            if stripped.startswith('json'):
                stripped = stripped[4:].strip()
            if stripped.startswith('{'):
                text = stripped
                break

    # Extract just the JSON object if there's surrounding text
    brace_start = text.find('{')
    if brace_start > 0:
        text = text[brace_start:]
    # Find matching closing brace
    depth = 0
    found_close = False
    for i, c in enumerate(text):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                text = text[:i + 1]
                found_close = True
                break

    # Repair truncated JSON (model cut off mid-response)
    if not found_close and depth > 0:
        # Strip last incomplete entry (after last comma) and close
        last_comma = text.rfind(',')
        if last_comma > 0:
            text = text[:last_comma] + '}'
        else:
            text = text.rstrip() + '}'

    # Attempt 1: standard JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: ast.literal_eval (handles single quotes, trailing commas)
    try:
        data = ast.literal_eval(text)
        if isinstance(data, dict):
            return data
    except (ValueError, SyntaxError):
        pass

    # Attempt 3: brute-force fix single quotes
    try:
        fixed = text.replace("'", '"')
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    print(f"[SENTIMENT] LLM JSON parse failed: {text[:120]}")
    return None


# --- Tiered Gemini scoring ---
# Newest articles get the best model, older articles get cheaper models.
# Tiers: (model, cumulative_fraction) — newest first
def _get_scoring_tiers():
    """Build scoring tiers based on smart routing recommendation."""
    from llm_client import get_recommended_model
    model = get_recommended_model('sentiment')
    if "pro" in model:
        # Pro available: 3 tiers (Pro 20%, Flash 40%, Lite rest)
        return [
            ("gemini-2.5-pro", 0.20),
            ("gemini-2.5-flash", 0.60),
            ("gemini-2.5-flash-lite", 1.00),
        ]
    if "flash-lite" in model:
        return [("gemini-2.5-flash-lite", 1.00)]
    # Default: Flash for newest, Flash-Lite for bulk
    return [
        ("gemini-2.5-flash", 0.40),
        ("gemini-2.5-flash-lite", 1.00),
    ]


# Model quality ranking (higher = better)
_MODEL_RANK = {
    "gemini-2.5-pro": 3,
    "gemini-2.5-flash": 2,
    "gemini-2.5-flash-lite": 1,
}


def _build_score_prompt(chunk_articles, full_texts):
    """Build the scoring prompt for a chunk of articles."""
    n = len(chunk_articles)
    lines = []
    for i, a in enumerate(chunk_articles):
        h = ' '.join(a.get('headline', '').split())
        s = ' '.join(a.get('summary', '').split())
        body = full_texts[i] if i < len(full_texts) else None

        parts = [f"{i + 1}. {h}"]
        if body:
            parts.append(body[:500])
        elif s:
            parts.append(s)
        lines.append(" — ".join(parts))

    return (
        f"Score each article's financial sentiment from -1.00 (very bearish) "
        f"to 1.00 (very bullish). 0.00 = neutral.\n"
        f"Use the FULL continuous range — do NOT round to 0.05 or 0.10 increments.\n"
        f'Return ONLY a JSON object mapping article number to score, '
        f'e.g. {{"1": 0.37, "2": -0.63, "3": 0.08, "4": -0.21, "5": 0.54}}\n\n'
        + "\n".join(lines)
    ), n


def _parse_scores(result, n):
    """Parse LLM JSON response into a list of float scores."""
    if not result:
        return None

    data = _parse_llm_json(result)
    if not isinstance(data, dict):
        return None

    scores = []
    matched = 0
    for i in range(1, n + 1):
        key_str = str(i)
        if key_str in data or i in data:
            raw = data.get(key_str, data.get(i))
            try:
                scores.append(max(-1.0, min(1.0, float(raw))))
                matched += 1
            except (TypeError, ValueError):
                scores.append(None)  # malformed value → KW fallback
        else:
            # Missing article — use None sentinel so caller can KW-fallback
            scores.append(None)
    if matched < n * 0.5:
        print(f"[SENTIMENT] LLM chunk only scored {matched}/{n}, failing chunk")
        return None
    if matched < n:
        print(f"[SENTIMENT] LLM scored {matched}/{n}, {n - matched} gaps left for KW fallback")
    return scores


def _llm_score_chunk(chunk_articles, full_texts, model=None):
    """Score a single chunk of articles via LLM. Returns list[float] or None.

    If model is specified, uses that specific Gemini model (for tiered scoring).
    If model is None, uses call_llm() with automatic fallback chain.
    """
    prompt, n = _build_score_prompt(chunk_articles, full_texts)
    system = "Financial sentiment scorer. Return only a JSON object mapping article number strings to float scores. No explanation."
    max_tokens = max(512, n * 40)  # thinking models need more headroom

    if model:
        from llm_client import call_gemini
        result = call_gemini(prompt, system=system, model=model, max_tokens=max_tokens)
    else:
        from llm_client import call_llm
        result = call_llm(prompt, system=system, max_tokens=max_tokens)

    return _parse_scores(result, n)


def _fetch_full_texts(articles):
    """Fetch article bodies in parallel. Returns list of text or None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    urls = [a.get('url', '') for a in articles]
    full_texts = [None] * len(articles)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_article_text, u): i
                   for i, u in enumerate(urls) if u}
        try:
            for fut in as_completed(futures, timeout=15):
                try:
                    full_texts[futures[fut]] = fut.result()
                except Exception:
                    pass
        except TimeoutError:
            pass
    return full_texts


def _llm_score_batch(articles):
    """Batch-score articles using tiered Gemini models.

    Newest articles get the best model (pro), middle get flash, rest get lite.
    Falls back to cheaper models when daily quota is exhausted.
    Sets _scored_by_model on each article.
    Returns list[float] or None if all models fail.
    """
    try:
        from llm_client import call_gemini, get_budget, _429_cooled_down  # noqa: F401
        from llm_config import load_llm_config
    except ImportError:
        return None

    if not articles:
        return None

    # Skip entirely if LLM is in 429 cooldown (don't waste time fetching texts)
    if not _429_cooled_down():
        return None

    full_texts = _fetch_full_texts(articles)
    n = len(articles)
    fetched = sum(1 for t in full_texts if t)
    print(f"[SENTIMENT] Fetched {fetched}/{n} article bodies for LLM")

    all_scores = [None] * n
    all_models = [None] * n

    from llm_client import _trigger_429_cooldown

    prev_end = 0
    for tier_model, cum_frac in _get_scoring_tiers():
        end = min(round(n * cum_frac), n)
        if prev_end >= end:
            prev_end = end
            continue

        # Bail early if cooldown triggered by a previous tier
        if not _429_cooled_down():
            prev_end = end
            continue

        chunk_articles = articles[prev_end:end]
        chunk_texts = full_texts[prev_end:end]

        # Try this tier's assigned model only — no slow fallback chain
        remaining, _ = get_budget(tier_model)
        if remaining > 0:
            scores = _llm_score_chunk(chunk_articles, chunk_texts, model=tier_model)
            if scores is not None:
                for j, s in enumerate(scores):
                    all_scores[prev_end + j] = s
                    if s is not None:
                        all_models[prev_end + j] = tier_model
            # scores=None means parse failure or bad response, NOT a 429.
            # Don't trigger cooldown here — actual 429s are handled inside call_gemini.

        prev_end = end

    # Check if we scored anything
    scored_count = sum(1 for s in all_scores if s is not None)
    if scored_count == 0:
        return None

    # Fill gaps with keyword scoring (articles where all LLM tiers failed)
    gap_count = 0
    for i in range(n):
        if all_scores[i] is None:
            gap_count += 1
            headline = _validate_text(articles[i].get('headline', ''))
            summary = _validate_text(articles[i].get('summary', ''))
            if headline or summary:
                h = _score_text(headline) if headline else 0.0
                s = _score_text(summary) if summary else 0.0
                all_scores[i] = h * 0.6 + s * 0.4
            else:
                all_scores[i] = 0.0
            all_models[i] = 'KW'
    if gap_count > 0:
        print(f"[SENTIMENT] {gap_count} articles fell back to KW scoring (LLM tier gaps)")

    # Tag articles with model info
    model_counts = {}
    for i, a in enumerate(articles):
        m = all_models[i]
        a['_scored_by_model'] = m or 'none'
        if m:
            model_counts[m] = model_counts.get(m, 0) + 1

    tier_summary = ", ".join(f"{m.split('-')[-1]}={c}" for m, c in model_counts.items())
    print(f"[SENTIMENT] Tiered scoring: {tier_summary} ({scored_count}/{n} scored)")

    return all_scores


def score_article_batch(articles):
    """Score articles for display.

    Returns tuple: (scores: list[float], method: str)
        scores: per-article float scores
        method: "LLM" if LLM scored, "KW" if keyword fallback

    Uses tiered Gemini scoring: newest articles get the best model,
    older articles get cheaper models. Falls back to keyword scoring.
    """
    if not articles:
        return [], "KW"

    llm_scores = _llm_score_batch(articles)
    if llm_scores is not None:
        return llm_scores, "LLM"

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
    return scores, "KW"


def try_llm_upgrade(articles):
    """Upgrade articles to better LLM models.

    Checks each article's _scored_by_model and attempts to re-score with a
    better model if daily quota allows. Also upgrades KW-scored articles.
    Returns list of float scores on any upgrade, None if no upgrades possible.
    """
    if not articles:
        return None

    from llm_client import get_budget

    # Find articles that can be upgraded
    upgradeable = []
    upgrade_indices = []
    for i, a in enumerate(articles):
        current_model = a.get('_scored_by_model', '')
        current_rank = _MODEL_RANK.get(current_model, 0)
        # KW articles (rank 0) or lower-tier models are upgradeable
        if current_rank < 3:  # Not yet pro
            upgradeable.append(a)
            upgrade_indices.append(i)

    if not upgradeable:
        return None

    # Find the best model with budget
    best_model = None
    for model in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        remaining, _ = get_budget(model)
        if remaining > 0:
            # Only upgrade if this model is better than what articles already have
            model_rank = _MODEL_RANK[model]
            if any(_MODEL_RANK.get(a.get('_scored_by_model', ''), 0) < model_rank
                   for a in upgradeable):
                best_model = model
                break

    if not best_model:
        return None

    best_rank = _MODEL_RANK[best_model]
    # Only upgrade articles that are actually lower rank
    to_upgrade = [(i, a) for i, a in zip(upgrade_indices, upgradeable)
                  if _MODEL_RANK.get(a.get('_scored_by_model', ''), 0) < best_rank]

    if not to_upgrade:
        return None

    indices, arts = zip(*to_upgrade)
    arts = list(arts)
    full_texts = _fetch_full_texts(arts)
    scores = _llm_score_chunk(arts, full_texts, model=best_model)

    if scores is not None:
        # Build full scores list (None for non-upgraded articles)
        result = [None] * len(articles)
        for idx, score in zip(indices, scores):
            result[idx] = score
            articles[idx]['_scored_by_model'] = best_model
        n_upgraded = len(scores)
        print(f"[SENTIMENT] Upgraded {n_upgraded} articles to {best_model.split('-')[-1]}")
        return result

    return None


def _aggregate_scores(scores):
    """Aggregate a list of per-article scores into a sentiment result dict.

    Returns dict with sentiment_score, article_count, positive_ratio, negative_ratio.
    """
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


def _score_articles(articles):
    """Score a list of Finnhub news articles. Deduplicates and aggregates.

    Used by trading loops (get_news_sentiment, get_market_sentiment).
    Tries LLM batch scoring first. Falls back to keyword scoring with
    full-text article fetch for higher accuracy.

    Returns tuple: (result_dict, used_llm)
        result_dict: sentiment_score, article_count, positive_ratio, negative_ratio
        used_llm: True if LLM scored successfully, False if keyword fallback
    """
    articles = _deduplicate_articles(articles)

    if not articles:
        return _aggregate_scores([]), True  # nothing to retry

    # Try LLM batch scoring first (one API call, cost-efficient)
    scores = _llm_score_batch(articles)
    used_llm = scores is not None

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

    return _aggregate_scores(scores), used_llm


def _try_llm_retry():
    """Drain ONE queued item from _llm_retry_queue.

    Discards stale entries (older than CACHE_TTL) and entries whose cache
    was already updated by a newer call. On LLM success, updates the cache.
    On failure, pushes the item back to the front of the queue.
    """
    if not _llm_retry_queue:
        return

    cache_key, articles, queued_at = _llm_retry_queue.popleft()
    now = time.time()

    # Stale: cache will be refreshed by normal flow anyway
    if now - queued_at > CACHE_TTL:
        return

    # Superseded: a newer call already updated the cache
    if cache_key in _cache:
        cached_ts, _ = _cache[cache_key]
        if cached_ts > queued_at:
            return

    scores = _llm_score_batch(articles)
    if scores is None:
        # LLM still unavailable — push back to front for next attempt
        _llm_retry_queue.appendleft((cache_key, articles, queued_at))
        return

    result = _aggregate_scores(scores)
    _cache[cache_key] = (now, result)
    print(f"[SENTIMENT] LLM retry upgraded {cache_key}")


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


# --- CNN Fear & Greed Index (stocks + VIX) ---

def get_cnn_fear_greed():
    """Fetch CNN Fear & Greed Index (stocks) and VIX.

    Returns dict with 'score' (0-100), 'rating' (str),
    'previous_close', 'previous_1_week', 'vix', 'vix_rating',
    or None on error.  Cached for 5 minutes.
    """
    now = time.time()
    if '__cnn_fng__' in _cache:
        ts, result = _cache['__cnn_fng__']
        if now - ts < CACHE_TTL:
            return result

    try:
        resp = requests.get(
            'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
            headers={
                'User-Agent': _USER_AGENT,
                'Referer': 'https://www.cnn.com/markets/fear-and-greed',
            },
            timeout=8,
        )
        data = resp.json()
        fg = data.get('fear_and_greed', {})
        result = {
            'score': round(fg.get('score', 0), 1),
            'rating': fg.get('rating', '').replace('_', ' ').title(),
            'previous_close': round(fg.get('previous_close', 0), 1),
            'previous_1_week': round(fg.get('previous_1_week', 0), 1),
        }
        # Extract VIX: data[-1].y is actual VIX value, score is CNN's 0-100 rating
        vix_section = data.get('market_volatility_vix', {})
        if isinstance(vix_section, dict):
            vix_ts = vix_section.get('data', [])
            if vix_ts and isinstance(vix_ts[-1], dict):
                result['vix'] = round(float(vix_ts[-1].get('y', 0)), 2)
                result['vix_rating'] = vix_ts[-1].get('rating', '').replace('_', ' ').title()
            else:
                result['vix'] = 0.0
                result['vix_rating'] = ''
        else:
            result['vix'] = 0.0
            result['vix_rating'] = ''
        _cache['__cnn_fng__'] = (now, result)
        return result
    except Exception as e:
        print(f"[SENTIMENT] CNN Fear & Greed error: {e}")
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
    # Per-symbol jitter: spread cache expiries across 2 minutes to avoid
    # all symbols hitting the LLM simultaneously when caches expire together
    symbol_ttl = CACHE_TTL + (hash(cache_key) % 120)
    if cache_key in _cache:
        ts, result = _cache[cache_key]
        if now - ts < symbol_ttl:
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

        result, used_llm = _score_articles(relevant)
        _cache[cache_key] = (now, result)
        if not used_llm:
            _llm_retry_queue.append((cache_key, relevant, now))
        else:
            _try_llm_retry()
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
        result, used_llm = _score_articles(articles[:30])
        _cache['__market__'] = (now, result)
        if not used_llm:
            _llm_retry_queue.append(('__market__', articles[:30], now))
        else:
            _try_llm_retry()
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

    Design philosophy:
    - The ML model is the primary signal. It already uses Daily_Sentiment
      (FnG) as an input feature, so FnG is NOT applied as a position
      reducer — that would double-count sentiment the model already saw.
    - FnG extreme greed (>90) is the one exception: bubble risk is
      asymmetric and worth reducing exposure for.
    - Symbol-specific news is genuinely new info the model can't see.
      Catastrophic events (hacks, fraud) warrant hard reductions.
    - Market news is diffuse — very light touch.
    """
    multiplier = 1.0
    reasons = []

    # --- Crypto: FnG — only reduce on extreme greed (bubble protection) ---
    # The model already uses Daily_Sentiment (derived from FnG) as a feature,
    # so fear-based reductions would double-count what the model already saw.
    # Extreme greed is kept because bubble tops are asymmetric risk.
    if asset_type == 'crypto':
        fng = get_fear_greed()
        if fng is not None:
            val = fng['value']
            if val >= 90:
                multiplier *= 0.7
                reasons.append(f"FnG={val}(extreme_greed)")
            elif val >= 80:
                multiplier *= 0.85
                reasons.append(f"FnG={val}(greed)")
            else:
                reasons.append(f"FnG={val}")

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


def get_recent_headlines(symbol, asset_type='crypto', max_headlines=5):
    """Get recent news headlines for a symbol (no scoring, just text).

    Returns list of headline strings, or empty list.
    Cached alongside get_news_sentiment (same Finnhub call).
    """
    client = _get_finnhub()
    if client is None:
        return []

    try:
        if asset_type == 'crypto':
            articles = client.general_news('crypto', min_id=0)
            base = symbol.replace('/USD', '').replace('-USD', '').lower()
            relevant = [a for a in articles
                        if base in a.get('headline', '').lower()
                        or base in a.get('summary', '').lower()]
            if len(relevant) < 2:
                relevant = articles[:10]
        else:
            import datetime as dt
            today = dt.date.today()
            week_ago = today - dt.timedelta(days=7)
            clean_sym = symbol.replace('/', '')
            articles = client.company_news(
                clean_sym,
                _from=week_ago.strftime('%Y-%m-%d'),
                to=today.strftime('%Y-%m-%d'),
            )
            relevant = articles[:15]

        headlines = []
        for a in relevant[:max_headlines]:
            h = a.get('headline', '').strip()
            if h:
                headlines.append(h)
        return headlines
    except Exception:
        return []
