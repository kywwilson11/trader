#!/usr/bin/env python3
"""Comprehensive sentiment scoring test suite — 120+ headlines.

Run:  python test_sentiment.py
"""
from sentiment import _score_text

# Each entry: (headline, expected_sign, category)
#   expected_sign: 'pos', 'neg', 'neutral', 'mixed'
#   For 'pos': score must be > +0.05
#   For 'neg': score must be < -0.05
#   For 'neutral': |score| <= 0.15
#   For 'mixed': any score is acceptable (inherently ambiguous)

TESTS = [
    # ===== CLEARLY BULLISH (expect positive) =====
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

    # ===== CLEARLY BEARISH (expect negative) =====
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

    # ===== NEGATION — POSITIVE WORD NEGATED (expect negative) =====
    ("This rally is not sustainable, warns top analyst", "neg", "negation"),
    ("Earnings did not beat expectations this quarter", "neg", "negation"),
    ("Bitcoin recovery is not happening anytime soon says veteran trader", "neg", "negation"),
    ("The growth story is no longer bullish for NVDA", "neg", "negation"),
    ("Analysts say the gains are not justified by fundamentals", "neg", "negation"),
    ("This is not a breakout, it's a bull trap", "neg", "negation"),
    ("Tesla's expansion plans are not going well according to insiders", "neg", "negation"),
    ("The market hasn't shown any signs of recovery", "neg", "negation"),

    # ===== NEGATION — NEGATIVE WORD NEGATED (expect positive/mixed) =====
    ("Stock market is not going to crash according to Fed chair", "mixed", "negation"),
    ("Bitcoin is not in a bear market despite recent volatility", "mixed", "negation"),
    ("Analysts say fears of recession are not warranted", "mixed", "negation"),
    ("The selloff wasn't as bad as expected", "mixed", "negation"),
    ("NVDA is not declining, it's consolidating before next leg up", "mixed", "negation"),

    # ===== NEUTRAL — no sentiment signal (expect neutral) =====
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

    # ===== MIXED — genuinely ambiguous (any score ok) =====
    ("Tesla stock drops despite strong quarterly revenue growth", "mixed", "mixed"),
    ("Bitcoin rises but analysts warn of resistance at $60K", "mixed", "mixed"),
    ("NVDA beats earnings but guidance disappoints", "mixed", "mixed"),
    ("Crypto market recovers slightly after brutal week of losses", "mixed", "mixed"),
    ("Meta stock rises on cost cuts despite declining user growth", "mixed", "mixed"),
    ("AMD shows strong growth but faces increasing competition", "mixed", "mixed"),
    ("Market gains erased by afternoon selloff", "mixed", "mixed"),
    ("Bitcoin ETF approved but adoption slower than expected", "mixed", "mixed"),

    # ===== EARNINGS SPECIFIC =====
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

    # ===== CRYPTO SPECIFIC =====
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

    # ===== MACRO / GEOPOLITICAL =====
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

    # ===== ANALYST CALLS =====
    ("Goldman Sachs raises price target on NVDA to $200", "pos", "analyst"),
    ("Top analyst issues rare strong buy rating on Bitcoin", "pos", "analyst"),
    ("Morgan Stanley upgrades AMD citing AI tailwinds", "pos", "analyst"),
    ("Barclays downgrades Tesla to underweight on valuation concerns", "neg", "analyst"),
    ("JPMorgan warns of 20% downside risk for S&P 500", "neg", "analyst"),
    ("Analyst slashes PLTR price target citing slowing momentum", "neg", "analyst"),
    ("Citi downgrades crypto sector to sell on regulatory headwinds", "neg", "analyst"),
    ("Cathie Wood says Bitcoin could still go to zero", "neg", "analyst"),

    # ===== SUBTLE / TRICKY =====
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


def main():
    passed = 0
    failed = 0
    failures = []

    # Group by category for reporting
    categories = {}
    for headline, expected, cat in TESTS:
        categories.setdefault(cat, [])

        score = _score_text(headline)

        if expected == "pos":
            ok = score > 0.05
        elif expected == "neg":
            ok = score < -0.05
        elif expected == "neutral":
            ok = abs(score) <= 0.15
        elif expected == "mixed":
            ok = True  # any score is fine
        else:
            ok = False

        categories[cat].append((ok, score, expected, headline))
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((score, expected, cat, headline))

    # Print results by category
    total = passed + failed
    print(f"\n{'='*80}")
    print(f"SENTIMENT SCORING TEST RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"{'='*80}\n")

    for cat, results in categories.items():
        cat_pass = sum(1 for ok, *_ in results if ok)
        cat_total = len(results)
        marker = "PASS" if cat_pass == cat_total else "FAIL"
        print(f"[{marker}] {cat.upper()} — {cat_pass}/{cat_total}")
        for ok, score, expected, headline in results:
            status = "  OK" if ok else "FAIL"
            print(f"  {status}  {score:+.3f}  (expect {expected:7s})  {headline[:65]}")
        print()

    if failures:
        print(f"\n{'='*80}")
        print(f"FAILURES ({failed}):")
        print(f"{'='*80}")
        for score, expected, cat, headline in failures:
            print(f"  {score:+.3f}  (expect {expected:7s})  [{cat}]  {headline[:70]}")

    print(f"\nTotal: {passed}/{total} ({100*passed/total:.0f}%)")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
