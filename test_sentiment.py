#!/usr/bin/env python3
"""Comprehensive sentiment scoring test suite — 1000+ headlines.

Tests both _score_text (raw scoring) and _validate_text (article validation).

Run:  python test_sentiment.py
"""
import itertools
from sentiment import _score_text, _validate_text, _score_articles

# ---------------------------------------------------------------------------
# Hand-written edge cases (the originals + new additions)
# ---------------------------------------------------------------------------
# (headline, expected_sign, category)
#   'pos': score > +0.05    'neg': score < -0.05
#   'neutral': |score| <= 0.15    'mixed': any score ok

HAND_WRITTEN = [
    # ===== CLEARLY BULLISH =====
    ("NVDA surges to all time high after blowout earnings beat expectations", "pos", "bullish"),
    ("Bitcoin rallies past $100K as institutional buying accelerates", "pos", "bullish"),
    ("Tesla stock soars 15% on record quarterly deliveries", "pos", "bullish"),
    ("Amazon beats revenue and profit estimates, stock jumps after hours", "pos", "bullish"),
    ("Crypto market cap hits new milestone as ETF inflows surge", "pos", "bullish"),
    ("Apple reports strongest iPhone sales growth in three years", "pos", "bullish"),
    ("PLTR stock rallies after major government contract win", "pos", "bullish"),
    ("Ethereum breakout confirmed as price climbs above key resistance", "pos", "bullish"),
    ("Strong jobs report boosts market optimism", "pos", "bullish"),
    ("Meta shares jump on better than expected ad revenue growth", "pos", "bullish"),
    ("Bitcoin ETF sees record single-day inflow of $1.2 billion", "pos", "bullish"),
    ("AMD stock upgraded to strong buy by Goldman Sachs", "pos", "bullish"),
    ("Solana ecosystem expansion drives SOL to new highs", "pos", "bullish"),
    ("Market rally broadens as S&P 500 hits record", "pos", "bullish"),
    ("COIN stock surges on robust trading volume and revenue beat", "pos", "bullish"),
    ("Fed signals rate cuts, boosting growth stocks", "pos", "bullish"),
    ("Nvidia partnership with major cloud providers sends shares higher", "pos", "bullish"),
    ("Bullish momentum builds as tech sector leads gains", "pos", "bullish"),
    ("Short squeeze sends meme stocks soaring double digits", "pos", "bullish"),
    ("XRP price target raised to $5 by top crypto analyst", "pos", "bullish"),

    # ===== CLEARLY BEARISH =====
    ("Bitcoin crashes below $50K amid global recession fears", "neg", "bearish"),
    ("Jim Cramer On Amazon Stock Slip: Not Saying Downside Overdone, 'I Figure Tomorrow's Pretty Ugly'", "neg", "bearish"),
    ("Tesla stock plunges on disappointing delivery numbers", "neg", "bearish"),
    ("Crypto market bloodbath wipes out $200B in market cap", "neg", "bearish"),
    ("NVDA shares drop 10% after weak guidance and China export ban", "neg", "bearish"),
    ("SEC launches investigation into major crypto exchange fraud", "neg", "bearish"),
    ("Meta layoffs signal broader tech sector weakness", "neg", "bearish"),
    ("Bitcoin selloff accelerates as fear grips the market", "neg", "bearish"),
    ("AMD downgraded to sell as chip demand weakens", "neg", "bearish"),
    ("Inflation data worse than expected, stocks tumble", "neg", "bearish"),
    ("PLTR stock slides on insider selling and slowing growth", "neg", "bearish"),
    ("Crypto exchange hacked for $500M, Bitcoin dumps", "neg", "bearish"),
    ("Recession warning: yield curve inverts again", "neg", "bearish"),
    ("Tariff war escalation sends markets into freefall", "neg", "bearish"),
    ("SOFI stock crashes after earnings miss and guidance cut", "neg", "bearish"),
    ("Major bank warns of housing market crisis ahead", "neg", "bearish"),
    ("Ethereum price tumbles amid regulatory crackdown concerns", "neg", "bearish"),
    ("Analysts warn of headwinds for crypto market amid rising inflation concerns", "neg", "bearish"),
    ("MSTR shares sink as Bitcoin drops below critical support", "neg", "bearish"),
    ("Bear market confirmed as S&P enters correction territory", "neg", "bearish"),

    # ===== NEGATION =====
    ("This rally is not sustainable, warns top analyst", "neg", "negation"),
    ("Earnings did not beat expectations this quarter", "neg", "negation"),
    ("Bitcoin recovery is not happening anytime soon says veteran trader", "neg", "negation"),
    ("The growth story is no longer bullish for NVDA", "neg", "negation"),
    ("Analysts say the gains are not justified by fundamentals", "neg", "negation"),
    ("This is not a breakout, it's a bull trap", "neg", "negation"),
    ("Tesla's expansion plans are not going well according to insiders", "neg", "negation"),
    ("The market hasn't shown any signs of recovery", "neg", "negation"),
    ("Stock market is not going to crash according to Fed chair", "mixed", "negation"),
    ("Bitcoin is not in a bear market despite recent volatility", "mixed", "negation"),
    ("Analysts say fears of recession are not warranted", "mixed", "negation"),
    ("The selloff wasn't as bad as expected", "mixed", "negation"),
    ("NVDA is not declining, it's consolidating before next leg up", "mixed", "negation"),

    # ===== NEUTRAL =====
    ("Apple announces new product event scheduled for March", "neutral", "neutral"),
    ("Microsoft to report earnings after the bell Thursday", "neutral", "neutral"),
    ("Fed meeting minutes to be released at 2 PM ET", "neutral", "neutral"),
    ("Elon Musk tweets about Dogecoin again", "neutral", "neutral"),
    ("Congress debates new cryptocurrency regulation framework", "mixed", "neutral"),
    ("Amazon opens new distribution center in Texas", "neutral", "neutral"),
    ("Bitcoin trading volume unchanged from yesterday", "neutral", "neutral"),
    ("SEC commissioner gives speech on digital assets", "neutral", "neutral"),
    ("NVIDIA announces next-gen GPU architecture at developer conference", "neutral", "neutral"),
    ("Google rebrands crypto custody division", "neutral", "neutral"),

    # ===== MIXED =====
    ("Tesla stock drops despite strong quarterly revenue growth", "mixed", "mixed"),
    ("Bitcoin rises but analysts warn of resistance at $60K", "mixed", "mixed"),
    ("NVDA beats earnings but guidance disappoints", "mixed", "mixed"),
    ("Crypto market recovers slightly after brutal week of losses", "mixed", "mixed"),
    ("Meta stock rises on cost cuts despite declining user growth", "mixed", "mixed"),
    ("AMD shows strong growth but faces increasing competition", "mixed", "mixed"),
    ("Market gains erased by afternoon selloff", "mixed", "mixed"),
    ("Bitcoin ETF approved but adoption slower than expected", "mixed", "mixed"),

    # ===== EARNINGS =====
    ("TSLA smashes Q4 earnings: revenue up 25%, EPS beats by 20 cents", "pos", "earnings"),
    ("AMD reports strong revenue growth, raises full year guidance", "pos", "earnings"),
    ("Netflix subscriber growth blows past estimates", "pos", "earnings"),
    ("COIN earnings crush expectations as crypto trading volumes soar", "pos", "earnings"),
    ("HOOD revenue miss sends shares tumbling after hours", "neg", "earnings"),
    ("SNAP misses on revenue and gives weak guidance, stock plunges", "neg", "earnings"),
    ("PLTR disappoints with slower than expected government revenue", "neg", "earnings"),
    ("Intel posts massive loss, announces thousands of layoffs", "neg", "earnings"),
    ("Worst earnings season since 2020 as majority of companies miss", "neg", "earnings"),
    ("Meta earnings lackluster, ad revenue growth declines", "neg", "earnings"),

    # ===== CRYPTO =====
    ("Bitcoin halving triggers supply shock, price surges past $80K", "pos", "crypto"),
    ("Ethereum staking yields attract institutional money", "pos", "crypto"),
    ("DeFi total value locked hits new all time high", "pos", "crypto"),
    ("XRP wins partial victory in SEC lawsuit", "pos", "crypto"),
    ("Major crypto exchange files for bankruptcy after hack", "neg", "crypto"),
    ("SEC sues another DeFi protocol for selling unregistered securities", "neg", "crypto"),
    ("Stablecoin depegs to $0.90, sparking contagion fears", "neg", "crypto"),
    ("Bitcoin miners selling at record pace amid declining profitability", "neg", "crypto"),
    ("Whale dumps 10,000 BTC on exchange, price slides", "neg", "crypto"),
    ("Crypto lending platform freezes withdrawals", "neg", "crypto"),

    # ===== MACRO =====
    ("Fed cuts interest rates by 50 basis points, stocks rally", "pos", "macro"),
    ("Unemployment falls to 3.4%, economy adds 350K jobs", "pos", "macro"),
    ("GDP growth exceeds expectations at 3.2% annualized", "pos", "macro"),
    ("Trade deal reached between US and China, markets cheer", "pos", "macro"),
    ("Oil prices crash 15% on demand destruction fears", "neg", "macro"),
    ("CPI comes in hot, inflation fears rattle markets", "neg", "macro"),
    ("Global sanctions escalate as geopolitical tensions rise", "neg", "macro"),
    ("Government shutdown looms as debt ceiling talks collapse", "neg", "macro"),
    ("Bank failures spread contagion fears through financial sector", "neg", "macro"),
    ("New tariffs on $200B of imports announced, trade war escalates", "neg", "macro"),

    # ===== ANALYST =====
    ("Goldman Sachs raises price target on NVDA to $200", "pos", "analyst"),
    ("Top analyst issues rare strong buy rating on Bitcoin", "pos", "analyst"),
    ("Morgan Stanley upgrades AMD citing AI tailwinds", "pos", "analyst"),
    ("Barclays downgrades Tesla to underweight on valuation concerns", "neg", "analyst"),
    ("JPMorgan warns of 20% downside risk for S&P 500", "neg", "analyst"),
    ("Analyst slashes PLTR price target citing slowing momentum", "neg", "analyst"),
    ("Citi downgrades crypto sector to sell on regulatory headwinds", "neg", "analyst"),
    ("Cathie Wood says Bitcoin could still go to zero", "neg", "analyst"),

    # ===== SUBTLE =====
    ("MARA stock flat despite Bitcoin rally", "mixed", "subtle"),
    ("Investors remain cautious as market grinds higher", "mixed", "subtle"),
    ("Trading volume dries up as summer doldrums set in", "neg", "subtle"),
    ("Smart money quietly accumulates Bitcoin at these levels", "pos", "subtle"),
    ("Options market signals big move coming for TSLA", "neutral", "subtle"),
    ("Volatility index spikes to highest level since March", "neg", "subtle"),
    ("Bitcoin dominance rises as altcoins bleed out", "mixed", "subtle"),
    ("Fund managers most bearish since 2008 according to BofA survey", "neg", "subtle"),
    ("Hedge funds pile into NVDA calls ahead of earnings", "pos", "subtle"),
    ("Insiders buying SOFI stock at fastest pace in two years", "pos", "subtle"),
    ("Short interest in HOOD spikes to 25% of float", "neg", "subtle"),
    ("Dark pool activity surges in AMD ahead of announcement", "mixed", "subtle"),
]

# ---------------------------------------------------------------------------
# Template-based generation for 1000+ total headlines
# ---------------------------------------------------------------------------

STOCKS = ['TSLA', 'NVDA', 'AMD', 'META', 'PLTR', 'COIN', 'MARA', 'MSTR',
          'SOFI', 'HOOD', 'SNAP', 'ROKU', 'NET', 'CRWD', 'SHOP', 'UBER',
          'ABNB', 'RBLX', 'SMCI', 'ARM', 'AAPL', 'MSFT', 'GOOG', 'AMZN']
CRYPTOS = ['Bitcoin', 'Ethereum', 'XRP', 'Solana', 'Dogecoin',
           'Cardano', 'Avalanche', 'Polkadot', 'Litecoin', 'Chainlink']

POS_VERBS = ['surges', 'rallies', 'soars', 'jumps', 'climbs', 'rebounds']
NEG_VERBS = ['crashes', 'plunges', 'tumbles', 'drops', 'slides', 'sinks']
POS_REASONS = [
    'after blowout earnings', 'on strong revenue growth', 'on bullish analyst upgrade',
    'as institutional buying accelerates', 'after record quarterly results',
    'on better than expected guidance', 'as demand surges',
    'after major partnership announcement', 'as profits beat estimates',
    'on positive momentum and strong volume',
]
NEG_REASONS = [
    'amid recession fears', 'on disappointing earnings', 'after bearish downgrade',
    'as selling pressure mounts', 'on weak guidance and slowing growth',
    'after missing revenue estimates', 'amid regulatory crackdown',
    'on fraud investigation concerns', 'as inflation fears rattle investors',
    'after massive insider selling',
]

TEMPLATES = []

# Template 1: "{STOCK} {pos_verb} {pos_reason}" — bullish
for stock in STOCKS:
    for verb in POS_VERBS:
        for reason in POS_REASONS:
            TEMPLATES.append((f"{stock} {verb} {reason}", "pos", "gen_bullish"))

# Template 2: "{STOCK} {neg_verb} {neg_reason}" — bearish
for stock in STOCKS:
    for verb in NEG_VERBS:
        for reason in NEG_REASONS:
            TEMPLATES.append((f"{stock} {verb} {reason}", "neg", "gen_bearish"))

# Template 3: "{CRYPTO} {pos_verb} {pos_reason}" — bullish crypto
for crypto in CRYPTOS:
    for verb in POS_VERBS:
        for reason in POS_REASONS:
            TEMPLATES.append((f"{crypto} {verb} {reason}", "pos", "gen_crypto_bull"))

# Template 4: "{CRYPTO} {neg_verb} {neg_reason}" — bearish crypto
for crypto in CRYPTOS:
    for verb in NEG_VERBS:
        for reason in NEG_REASONS:
            TEMPLATES.append((f"{crypto} {verb} {reason}", "neg", "gen_crypto_bear"))

# Template 5: Analyst upgrades/downgrades
ANALYSTS = ['Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'Barclays', 'Citi',
            'Wells Fargo', 'Bank of America', 'Deutsche Bank', 'UBS', 'RBC']
for analyst in ANALYSTS:
    for stock in STOCKS[:12]:
        TEMPLATES.append(
            (f"{analyst} upgrades {stock} citing strong growth outlook", "pos", "gen_analyst"))
        TEMPLATES.append(
            (f"{analyst} downgrades {stock} on slowing growth concerns", "neg", "gen_analyst"))

# Template 6: Earnings beat/miss
for stock in STOCKS[:12]:
    TEMPLATES.append(
        (f"{stock} beats earnings estimates, raises guidance", "pos", "gen_earnings"))
    TEMPLATES.append(
        (f"{stock} reports strong revenue growth, profit exceeds expectations", "pos", "gen_earnings"))
    TEMPLATES.append(
        (f"{stock} misses earnings estimates, cuts guidance", "neg", "gen_earnings"))
    TEMPLATES.append(
        (f"{stock} disappoints with weak revenue, stock plunges", "neg", "gen_earnings"))

# Template 7: Negation patterns
NEG_POS_PATTERNS = [
    "{} rally is not sustainable according to analysts",
    "{} growth is not expected to continue next quarter",
    "{} gains are not supported by fundamentals",
    "{} recovery is not happening, says veteran trader",
]
NEG_NEG_PATTERNS = [
    "{} crash is not as severe as feared",
    "{} decline is not likely to continue says analyst",
    "{} losses are not as bad as expected",
]
for stock in STOCKS[:10]:
    for pat in NEG_POS_PATTERNS:
        TEMPLATES.append((pat.format(stock), "neg", "gen_negation"))
    for pat in NEG_NEG_PATTERNS:
        TEMPLATES.append((pat.format(stock), "mixed", "gen_negation"))

# Template 8: Macro headlines
MACRO_POS = [
    "Federal Reserve cuts interest rates, boosting equities",
    "Strong GDP growth drives market optimism",
    "Unemployment hits record low as economy strengthens",
    "Trade deal sends stocks rallying across sectors",
    "Consumer confidence surges to multi-year high",
    "Manufacturing data beats expectations, economy shows resilience",
    "Housing starts surge indicating economic recovery",
    "Retail sales beat forecasts, consumer spending strong",
]
MACRO_NEG = [
    "Inflation surges past expectations, rate hike fears grow",
    "GDP contracts for second straight quarter, recession looms",
    "Trade war escalation sends global markets tumbling",
    "Bank failure sparks contagion fears across financial sector",
    "Government shutdown enters third week as talks collapse",
    "Oil crisis pushes inflation higher, consumer sentiment drops",
    "Debt ceiling standoff rattles bond markets worldwide",
    "Global sanctions trigger supply chain crisis and inflation fears",
]
for h in MACRO_POS:
    TEMPLATES.append((h, "pos", "gen_macro"))
for h in MACRO_NEG:
    TEMPLATES.append((h, "neg", "gen_macro"))

# Template 9: Neutral / informational
NEUTRAL_PATTERNS = [
    "{} to report earnings next Thursday",
    "{} announces executive leadership change",
    "{} schedules annual shareholder meeting",
    "{} files new patent application",
    "{} opens new office in Austin",
]
for stock in STOCKS[:10]:
    for pat in NEUTRAL_PATTERNS:
        TEMPLATES.append((pat.format(stock), "neutral", "gen_neutral"))

# Sample down to ~900 generated to target 1000 total with hand-written
import random
random.seed(42)
if len(TEMPLATES) > 900:
    TEMPLATES = random.sample(TEMPLATES, 900)

ALL_TESTS = HAND_WRITTEN + TEMPLATES

# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

VALIDATION_TESTS = [
    # (input, should_pass)
    ("Normal headline about stocks", True),
    ("", False),
    (None, False),
    ("hi", False),  # too short
    ("   ", False),
    ("<p>Some <b>HTML</b> content here about stocks</p>", True),  # HTML stripped
    ("https://example.com/article?id=123", False),  # URL garbage
    ("\u2603\u2603\u2603\u2603\u2603\u2603\u2603\u2603\u2603\u2603\u2603", False),  # non-ASCII
    (12345, False),  # not a string
    ("A" * 500, True),  # long but valid
]


def run_validation_tests():
    """Test _validate_text with edge cases."""
    passed = 0
    failed = 0
    for text, should_pass in VALIDATION_TESTS:
        result = _validate_text(text)
        ok = (result is not None) == should_pass
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
            display = repr(text)[:50]
            print(f"  FAIL  validate({display}) = {result!r}, expected {'valid' if should_pass else 'None'}")
        else:
            passed += 1
    return passed, failed


def run_article_validation_tests():
    """Test _score_articles with garbage input."""
    cases = [
        # All garbage → should return neutral
        ([{'headline': '', 'summary': ''}], 0.0),
        ([{'headline': None}], 0.0),
        # Mix of valid and garbage
        ([{'headline': 'Bitcoin surges on bullish momentum'},
          {'headline': '', 'summary': ''},
          {'headline': None}], None),  # None = just check it doesn't crash
        # HTML in summary
        ([{'headline': 'Test headline here', 'summary': '<p>Stock <b>rallies</b> strongly today</p>'}], None),
    ]
    passed = 0
    failed = 0
    for articles, expected_score in cases:
        result = _score_articles(articles)
        if expected_score is not None:
            ok = abs(result['sentiment_score'] - expected_score) < 0.01
        else:
            ok = isinstance(result['sentiment_score'], float)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL  _score_articles({articles!r:.60}) = {result['sentiment_score']:.3f}, expected {expected_score}")
    return passed, failed


def main():
    print(f"\nTotal headlines to test: {len(ALL_TESTS)}")
    print(f"  Hand-written: {len(HAND_WRITTEN)}")
    print(f"  Generated:    {len(TEMPLATES)}")

    passed = 0
    failed = 0
    failures = []
    categories = {}

    for headline, expected, cat in ALL_TESTS:
        categories.setdefault(cat, [])
        score = _score_text(headline)

        if expected == "pos":
            ok = score > 0.05
        elif expected == "neg":
            ok = score < -0.05
        elif expected == "neutral":
            ok = abs(score) <= 0.15
        elif expected == "mixed":
            ok = True
        else:
            ok = False

        categories[cat].append((ok, score, expected, headline))
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((score, expected, cat, headline))

    total = passed + failed
    pct = 100 * passed / total if total else 0

    print(f"\n{'='*80}")
    print(f"SENTIMENT SCORING: {passed}/{total} passed ({pct:.1f}%)")
    print(f"{'='*80}\n")

    # Summary by category
    for cat in sorted(categories.keys()):
        results = categories[cat]
        cat_pass = sum(1 for ok, *_ in results if ok)
        cat_total = len(results)
        marker = "PASS" if cat_pass == cat_total else "FAIL"
        print(f"  [{marker}] {cat:20s} — {cat_pass:3d}/{cat_total:3d}")

    # Show failures (capped at 30)
    if failures:
        show = failures[:30]
        print(f"\n{'='*80}")
        print(f"FAILURES ({failed} total, showing first {len(show)}):")
        print(f"{'='*80}")
        for score, expected, cat, headline in show:
            print(f"  {score:+.3f}  (expect {expected:7s})  [{cat:15s}]  {headline[:60]}")

    # --- Validation tests ---
    print(f"\n{'='*80}")
    print(f"VALIDATION TESTS")
    print(f"{'='*80}")
    vp, vf = run_validation_tests()
    ap, af = run_article_validation_tests()
    val_total = vp + vf + ap + af
    val_pass = vp + ap
    print(f"\n  _validate_text:   {vp}/{vp+vf}")
    print(f"  _score_articles:  {ap}/{ap+af}")
    print(f"  Validation total: {val_pass}/{val_total}")

    # --- Final summary ---
    all_pass = passed + val_pass
    all_total = total + val_total
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL: {all_pass}/{all_total} ({100*all_pass/all_total:.1f}%)")
    print(f"  Scoring:    {passed}/{total} ({pct:.1f}%)")
    print(f"  Validation: {val_pass}/{val_total}")
    print(f"{'='*80}")

    return failed == 0 and vf == 0 and af == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
