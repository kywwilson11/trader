"""Expanded chapters for the Trader System Manual.

Chapters 4-6 (Entry Gates, Position Sizing, Stops),
Chapters 10-15 (Adaptive, Data, GARCH, Macro, Correlation, HMM),
Chapter 20 (Nobel Prize Theoretical Foundations).
"""


def _clean(text):
    reps = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
        "\u2190": "<-", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x",
        "\u2248": "~=", "\u2260": "!=", "\u00b2": "^2", "\u00b3": "^3",
        "\u03c3": "sigma", "\u03b1": "alpha", "\u03b2": "beta",
        "\u03c9": "omega", "\u2605": "*", "\u2022": "-", "\u00b7": "-",
        "\u2212": "-", "\u00b1": "+/-", "\u221a": "sqrt",
    }
    for old, new in reps.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def s(pdf, text, size=10):
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(1)


def section(pdf, title):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def subsection(pdf, title):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def code(pdf, text):
    pdf.set_font("Courier", "", 7.5)
    pdf.set_fill_color(240, 240, 240)
    for line in _clean(text).split("\n"):
        pdf.cell(0, 3.8, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)


def bullet(pdf, text, size=10):
    pdf.set_font("Helvetica", "", size)
    pdf.cell(5, 5, "-")
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(0.5)


def chapter(pdf, num, title):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, _clean(f"Chapter {num}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)


def table_row(pdf, cells, widths, bold=False):
    style = "B" if bold else ""
    pdf.set_font("Helvetica", style, 7.5)
    for cell, w in zip(cells, widths):
        pdf.cell(w, 4.5, _clean(str(cell)), border=1)
    pdf.ln(4.5)


# =================================================================
# CHAPTER 4 -- ENTRY GATES & EXIT LOGIC (EXPANDED)
# =================================================================
def ch4_expanded(pdf):
    chapter(pdf, 4, "Entry Gates & Exit Logic")

    s(pdf, "Before any buy order is placed, a candidate must pass a gauntlet of 12 independent "
      "gates. Each gate tests a different risk dimension -- market conditions, model confidence, "
      "portfolio state, news, and fundamentals. If ANY single gate rejects, the symbol is "
      "skipped for this cycle. This defense-in-depth approach means a single faulty signal "
      "cannot force a bad trade.")

    section(pdf, "4.1 Gate 1: Cooldown Timer")
    s(pdf, "After closing a position (whether by stop-loss, take-profit, prediction reversal, "
      "or manual close), the symbol enters a cooldown period. CryptoLoop: 60 minutes. "
      "StockLoop: 20 minutes. The cooldown prevents whipsaw -- the pattern where a position "
      "is closed, the price ticks slightly in the predicted direction, and the model "
      "immediately re-enters, only for the original adverse move to continue.")
    s(pdf, "Implementation: last_trade_time[symbol] stores the datetime of the last trade. "
      "cooldown_ok() returns True only if elapsed time exceeds the cooldown period. Crypto "
      "uses a longer cooldown because 24/7 markets produce more noise signals at short "
      "intervals.")

    section(pdf, "4.2 Gate 2: Hard-Stop Lockout")
    s(pdf, "When a position hits its hard stop-loss, the symbol is locked for 24 hours. This "
      "is one of the most important behavioral safeguards in the system. After a hard stop, "
      "human traders experience 'revenge trading' -- the urge to immediately re-enter to "
      "'make back' the loss. Research by Odean (1998) shows this behavior is one of the "
      "primary destroyers of retail trading returns.")
    s(pdf, "The lockout is persisted to hard_stop_lockout.json so it survives bot restarts. "
      "After the 24-hour lockout expires, the re-entry threshold is elevated to 1.5x the "
      "normal threshold, requiring much stronger conviction. This validates that the "
      "original thesis still holds before committing capital again.")

    section(pdf, "4.3 Gate 3: Position Cap")
    s(pdf, "Each symbol has a maximum notional exposure: $3,000 for crypto, $5,000 for stocks. "
      "If the existing position value (qty * entry_price) already exceeds this cap, no "
      "additional buying occurs. This prevents over-concentration in a single name, even "
      "if the model gives an extremely strong signal.")

    section(pdf, "4.4 Gate 4: Prediction Threshold")
    s(pdf, "The model's predicted return must exceed the trade_threshold (a hyperparameter "
      "optimized by Optuna, typically 0.15-0.95%). The threshold represents the minimum edge "
      "the model needs to justify the transaction costs and risk of the trade.")
    s(pdf, "The effective threshold is dynamically adjusted in two cases:")
    bullet(pdf, "After hard-stop lockout: effective = threshold * 1.5 (require stronger signal)")
    bullet(pdf, "When Hurst < 0.45 (mean-reverting regime): effective = max(threshold, "
           "threshold * 1.3). In mean-reverting markets, momentum signals are unreliable, "
           "so a higher bar is required.")

    section(pdf, "4.5 Gate 5: Spread Edge Check")
    s(pdf, "The predicted return must exceed 2x the round-trip spread cost. This is a direct "
      "implementation of the finding from arXiv 2512.12924 (Dec 2025) that spread costs destroy "
      "strategies with less than 1% edge. The should_trade() function computes:")
    code(pdf, """round_trip_cost = spread_pct  (pay ~half spread on buy + ~half on sell)
threshold = round_trip_cost * min_edge (min_edge=2.0)
trade = abs(predicted_return) > threshold

Example: spread=0.15%, pred=+0.25%
  threshold = 0.15% * 2.0 = 0.30%
  0.25% > 0.30%? No -> SKIP (insufficient edge over friction)""")

    section(pdf, "4.6 Gate 6: Winner's Curse Filter")
    s(pdf, "Inspired by Milgrom & Wilson's auction theory (Nobel Prize 2020). The 'winner's "
      "curse' occurs when the highest bidder in an auction overpays -- winning is actually "
      "bad news about the item's true value. In trading, when a stock has already moved "
      "significantly above its average, the buyer is 'winning an auction' that others have "
      "passed on.")
    s(pdf, "When the current price exceeds SMA20 + 2*ATR (the price has extended beyond "
      "normal volatility), the signal is strongest -- but paradoxically, this is when the "
      "buyer is most likely overpaying. The filter requires 1.5x the normal threshold to "
      "enter, demanding much stronger conviction for extended moves.")
    s(pdf, "Milgrom's research showed that optimal bidding strategies should account for "
      "adverse selection -- the fact that winning an auction correlates with overpaying. "
      "Our filter implements this principle: we are more cautious precisely when the model "
      "is most excited.")

    section(pdf, "4.7 Gate 7: Correlation Check")
    s(pdf, "Before buying, the candidate's average absolute correlation with all held positions "
      "is computed. If avg_corr > 0.7, the entry is rejected entirely. This prevents "
      "portfolio concentration where multiple positions move together, amplifying drawdowns "
      "during market stress.")
    s(pdf, "The threshold of 0.7 is chosen because: (1) correlations above 0.7 indicate "
      "roughly 50% shared variance (R^2 = 0.49), meaning positions provide minimal "
      "diversification benefit; (2) during crises, correlations spike toward 1.0 "
      "(Bernanke/Diamond/Dybvig, Nobel 2022), so starting from 0.7 leaves no buffer.")
    s(pdf, "Example: Portfolio holds BTC/USD and ETH/USD. Candidate SOL/USD has correlation "
      "0.85 with BTC and 0.90 with ETH. Average = 0.875 > 0.7 -> REJECTED. Even though "
      "the model predicts SOL will rise, adding it would create a dangerously concentrated "
      "crypto portfolio.")

    section(pdf, "4.8 Gate 8: Macro Regime Halt")
    s(pdf, "When VIX exceeds 35, ALL new stock entries are blocked. VIX > 35 historically "
      "corresponds to major market crises (COVID crash, 2008 GFC, etc.). During these periods, "
      "correlations spike, volatility is extreme, and models trained on normal market conditions "
      "lose their predictive edge.")
    s(pdf, "This gate exists because the LSTM cannot predict 'unknown unknowns' -- systemic "
      "events that were not represented in its training data. The circuit is inspired by "
      "Diamond-Dybvig bank run theory: when the system itself is under stress, the correct "
      "action is to reduce exposure, not to bet on individual signals.")

    section(pdf, "4.9 Gate 9: Safe-Haven Filter")
    s(pdf, "When VIX is between 25-35 (elevated but not crisis), only 'safe-haven' stocks "
      "are allowed: Treasury bonds (TLT), gold (GLD), utilities, consumer staples. These "
      "historically appreciate during risk-off events. This implements the 'flight to quality' "
      "pattern documented in financial crisis research.")

    section(pdf, "4.10 Gate 10: Exposure Cap (Stocks Only)")
    s(pdf, "Total stock portfolio exposure is capped at $50,000. After each fill, "
      "_get_current_exposure() sums the market value of all stock positions. If the sum "
      "exceeds $50k, no more buys are placed. This prevents the portfolio from becoming "
      "over-leveraged during strong bull signals.")

    section(pdf, "4.11 Gate 11: Sentiment Gate")
    s(pdf, "News sentiment is scored using keyword analysis (82 positive + 80 negative terms, "
      "38 positive + 30 negative phrases, negation-aware). The resulting multiplier ranges "
      "from 0.15 (catastrophic news like hack or fraud) to 1.5 (strongly positive).")
    s(pdf, "Design principle: The ML model already uses Daily_Sentiment as a feature, so "
      "Fear & Greed reductions are NOT applied (would double-count). Exception: FnG >= 90 "
      "(extreme greed) triggers a 0.7x bubble protection multiplier. Symbol-specific news "
      "IS new information the model can't see in real-time, so hard gates for catastrophes "
      "are applied. Market-wide sentiment uses a very light touch (0.85x for bearish, "
      "1.1x for bullish) to avoid over-reacting to noise.")

    section(pdf, "4.12 Gate 12: LLM Gate")
    s(pdf, "The LLM analyst (Google Gemini) scores each candidate on a 0-1 scale based on "
      "fundamentals, news, insider activity, and filing analysis. If the score falls below "
      "0.15, the trade is vetoed entirely -- this represents a catastrophic red flag that "
      "the quantitative model cannot detect (e.g., accounting fraud, regulatory action, "
      "pending bankruptcy).")
    s(pdf, "Above 0.15, the LLM score acts as a multiplier: sized *= (0.5 + score). A "
      "score of 0.5 (neutral) gives 1.0x (no change). A score of 0.8 (bullish) gives 1.3x "
      "(30% boost). A score of 0.2 (barely passing) gives 0.7x (30% reduction). This "
      "creates a smooth gradient from 'barely tolerable' to 'strong conviction'.")

    section(pdf, "4.13 Exit Triggers")
    s(pdf, "Exits are triggered by seven independent mechanisms:")
    bullet(pdf, "Prediction reversal: pred < -threshold. The model now believes the position "
           "will lose money.")
    bullet(pdf, "LLM veto: LLM score drops below 0.15 on a held position. New catastrophic "
           "information detected.")
    bullet(pdf, "Hard stop: price <= entry * (1 - stop_dist). ATR-based, see Chapter 6.")
    bullet(pdf, "Trailing stop: After +1% profit, price <= HWM * (1 - trail_dist). Locks "
           "in gains while giving room for further upside.")
    bullet(pdf, "Take-profit: price >= tp_price (3x risk-reward from entry).")
    bullet(pdf, "Circuit breaker: 5% daily equity drawdown -> flatten ALL positions immediately, "
           "sleep 1 hour.")
    bullet(pdf, "Flatten before close: 3:50 PM ET for stocks (avoid overnight gap risk).")


# =================================================================
# CHAPTER 5 -- POSITION SIZING (EXPANDED)
# =================================================================
def ch5_expanded(pdf):
    chapter(pdf, 5, "Position Sizing -- 7-Layer Risk Stack")

    s(pdf, "Position sizing is the most important decision in any trading system -- more "
      "important than entry timing or exit strategy. Ed Thorp (who developed the Kelly "
      "criterion for blackjack before applying it to markets) demonstrated that optimal "
      "sizing is the primary driver of long-term wealth accumulation. Our system applies "
      "7 multiplicative layers, each addressing a different dimension of risk. The layers "
      "are independent and multiplicative: if any layer reduces sizing by 50%, the final "
      "position is at most 50% of what it would otherwise be, regardless of what other "
      "layers say.")

    section(pdf, "5.1 Layer 1: Kelly Criterion + VIX Scaling")

    subsection(pdf, "The Kelly Criterion")
    s(pdf, "John Kelly (1956) at Bell Labs derived a formula for the optimal fraction of "
      "capital to wager on a bet with known odds, maximizing the long-term geometric growth "
      "rate of wealth. The formula is:")
    code(pdf, """f* = (p * b - q) / b

where:
  f* = optimal fraction of capital to bet
  p  = probability of winning
  q  = probability of losing (1 - p)
  b  = payoff ratio (avg_win / avg_loss)""")
    s(pdf, "In our trading context, we estimate p from the win rate of the last 200 trades "
      "(stored in trade_memory.json), and b from the average win size divided by average loss "
      "size. The formula then becomes:")
    code(pdf, """kelly_f = (win_rate * (avg_win/avg_loss) - (1 - win_rate)) / (avg_win/avg_loss)""")

    subsection(pdf, "Why Half-Kelly")
    s(pdf, "We use HALF the Kelly fraction (kelly_f / 2) for three critical reasons:")
    bullet(pdf, "1. Return vs variance tradeoff: Half-Kelly yields 75% of the full-Kelly "
           "expected return with only 50% of the variance. The equity curve is dramatically "
           "smoother.")
    bullet(pdf, "2. Estimation error: Full-Kelly assumes we know the true win rate and payoff "
           "ratio perfectly. We don't -- we estimate them from a finite sample of 200 trades. "
           "Any overestimation of edge leads to catastrophic over-betting. Half-Kelly provides "
           "a safety margin against estimation error.")
    bullet(pdf, "3. Drawdown protection: Full-Kelly regularly produces 50%+ drawdowns from "
           "peak equity. Half-Kelly keeps maximum drawdowns to approximately 25% -- still "
           "painful but survivable.")
    s(pdf, "The half-Kelly fraction is clamped to [0.05, 0.25], meaning we never bet less "
      "than 5% or more than 25% of equity on a single trade. The resulting position size is "
      "further bounded to [NOTIONAL, 3*NOTIONAL] -- floor of $1k-$5k, ceiling of $3k-$15k.")
    s(pdf, "Kelly requires at least 50 historical trades to activate. Before that threshold, "
      "a fixed notional (NOTIONAL_PER_SYMBOL) is used.")

    subsection(pdf, "VIX-Based Kelly Scaling")
    s(pdf, "Per arXiv 2508.16598 (2025, 'Hybrid Kelly-VIX Position Sizing'), the Kelly fraction "
      "should be scaled by the volatility regime. High VIX means the market's uncertainty is "
      "elevated, making our edge estimates less reliable. The scaling is:")
    code(pdf, """VIX > 35:  kelly_scale = 0.3 (crisis -- bet 30% of Kelly)
VIX 25-35: kelly_scale = 0.5 (defensive -- bet 50% of Kelly)
VIX 15-25: kelly_scale = 0.7 (caution -- bet 70% of Kelly)
VIX < 15:  kelly_scale = 1.0 (calm -- full Half-Kelly)""")
    s(pdf, "This means in a VIX=30 environment with half-Kelly of 15%, the effective fraction "
      "becomes 15% * 0.5 = 7.5% of equity -- dramatically reducing exposure when markets "
      "are most uncertain.")

    section(pdf, "5.2 Layer 2: Drawdown-Based Reduction")
    s(pdf, "Tracks peak equity (highest ever equity value) and reduces sizing when underwater. "
      "This implements 'anti-martingale' position management -- sizing DOWN as losses mount "
      "rather than UP (which would be martingale, the fastest way to ruin).")
    code(pdf, """drawdown = (peak_equity - current_equity) / peak_equity

dd >= 20%: base *= 0.25 (75% reduction -- near halt)
dd >= 15%: base *= 0.50 (50% reduction)
dd >= 10%: base *= 0.75 (25% reduction)
dd < 10%:  no change""")
    s(pdf, "The 10%/15%/20% thresholds create a graduated response rather than a binary "
      "circuit breaker. At 10% drawdown, the system is still operating but cautiously. At "
      "20%, it's barely trading -- preserving capital for recovery. The circuit breaker at "
      "5% daily drawdown is a separate mechanism that triggers an immediate full flatten.")

    section(pdf, "5.3 Layer 3: Confidence Scaling")
    s(pdf, "Scales position size proportionally to the strength of the prediction signal. "
      "If the model predicts a 0.30% return with a threshold of 0.15%, the confidence is "
      "2.0 (capped) -- double the normal position. If the prediction barely exceeds threshold "
      "(0.16% vs 0.15%), the confidence is ~1.07 -- barely above normal.")
    code(pdf, """confidence = clamp(pred_return / trade_threshold, 0.5, 2.0)
sized = base * confidence

High confidence (pred = 2x threshold): sized = base * 2.0
Marginal (pred ~ threshold):           sized = base * ~1.0
Low (pred = 0.5x threshold):           sized = base * 0.5""")
    s(pdf, "This is important because not all signals are equal. The model may predict a "
      "0.05% return on one stock and a 0.50% return on another. The latter deserves a much "
      "larger position because the expected payoff justifies the risk.")

    section(pdf, "5.4 Layer 4: GARCH Volatility Targeting")
    s(pdf, "Based on Robert Engle's work (Nobel Prize 2003, see Chapter 12 for full theory). "
      "Each position targets the same daily volatility (2% default), creating risk parity "
      "across the portfolio. A stock with 1% daily vol gets a 2x position; a crypto with "
      "4% daily vol gets a 0.5x position. This means each position contributes roughly "
      "the same amount of risk to the portfolio, regardless of asset class.")
    code(pdf, """sigma = GARCH(1,1) 1-step-ahead forecast (e.g., 0.03 = 3% daily vol)
ratio = target_vol / sigma = 0.02 / 0.03 = 0.67
sized *= clamp(ratio, 0.5, 2.0)

Low vol asset (1%):  ratio=2.0 -> double position (capture more with less risk)
Target vol (2%):     ratio=1.0 -> normal position
High vol asset (4%): ratio=0.5 -> half position (same risk in dollar terms)""")
    s(pdf, "EGARCH is tried first because it captures the empirical asymmetry in volatility: "
      "crashes increase volatility more than rallies of equal magnitude. This means the model "
      "tightens positions faster after negative shocks than it loosens them after positive "
      "shocks -- a conservative asymmetry that protects capital.")

    section(pdf, "5.5 Layer 5: Macro Regime Multiplier")
    s(pdf, "Combines VIX, STLFSI2 (financial stress), CAPE (valuation), and stablecoin "
      "stability into a single multiplicative factor. See Chapter 13 for the complete theory. "
      "The key insight: these indicators capture systemic risks that the LSTM cannot predict "
      "because they represent 'unknown unknowns' -- events outside the model's training "
      "distribution (Bernanke/Diamond/Dybvig, Nobel 2022).")
    code(pdf, """VIX < 15:   1.0x     STLFSI2 > 1.0: additional 0.5x
VIX 15-25:  0.8x     CAPE z > 1.5:  additional 0.7x
VIX 25-35:  0.5x     Stablecoin emergency: 0.0x (halt)
VIX > 35:   0.3x

Compound example: VIX=28 (0.5x) + STLFSI2=1.2 (0.5x) = 0.25x total""")

    section(pdf, "5.6 Layer 6: Correlation Reduction (Markowitz)")
    s(pdf, "Based on Harry Markowitz's portfolio theory (Nobel Prize 1990, see Chapter 14). "
      "Reduces position size in proportion to its correlation with existing holdings:")
    code(pdf, """factor = max(0.5, 1.0 - 0.5 * avg_correlation)

corr=0.0: factor=1.0 (uncorrelated, full size)
corr=0.3: factor=0.85 (mild reduction)
corr=0.5: factor=0.75 (moderate reduction)
corr=0.7: factor=0.65 (heavy reduction, near rejection threshold)""")
    s(pdf, "This is applied AFTER the correlation gate (which rejects at avg_corr > 0.7). "
      "So even for accepted candidates, correlation still reduces their size. A stock that "
      "passes the 0.7 gate with correlation 0.65 still gets a 32.5% size reduction.")

    section(pdf, "5.7 Layer 7: HMM Regime + Ensemble Voting")
    s(pdf, "The HMM (see Chapter 15) classifies the market into bull/bear/neutral/high-vol "
      "regimes based on return distributions. Based on Sargent/Sims regime-switching models "
      "(Nobel Prize 2011).")
    code(pdf, """Bull:     sizing=1.2x (the only layer that INCREASES sizing)
Neutral:  sizing=1.0x
High-vol: sizing=0.5x
Bear:     sizing=0.3x (70% reduction)

Ensemble disagreement: if macro signals and HMM signals disagree,
  an additional 0.8x penalty is applied (20% reduction for uncertainty)

Example: macro says 'caution' (bearish), HMM says 'bull'
  -> These disagree -> sized *= 0.8 regardless of individual multipliers""")
    s(pdf, "The ensemble voting captures an important insight: when independent risk models "
      "disagree, the correct response is to reduce exposure, not to pick a winner. "
      "Disagreement itself is a signal of elevated uncertainty.")


# =================================================================
# CHAPTER 6 -- STOPS (EXPANDED)
# =================================================================
def ch6_expanded(pdf):
    chapter(pdf, 6, "Stop-Loss, Trailing Stop & Take-Profit")

    s(pdf, "Stop management is the mechanical enforcement of loss limits and profit capture. "
      "Without stops, human traders exhibit the 'disposition effect' documented by Kahneman "
      "(Nobel 2002) and Thaler (Nobel 2017): holding losers too long (hoping for recovery) "
      "and selling winners too early (locking in small gains). Research by Odean (1998) found "
      "the disposition effect is 'virtually zero among algorithms' -- but only if we don't "
      "intervene manually.")

    section(pdf, "6.1 ATR-Based Dynamic Stops")
    s(pdf, "Stop distances are computed from Average True Range (ATR), which measures the "
      "typical price range over N bars. Unlike fixed percentage stops, ATR adapts to each "
      "asset's volatility. A stock with ATR of $2 on a $100 stock (2%) gets tighter stops "
      "than a crypto with ATR of $1500 on a $60,000 asset (2.5%).")
    code(pdf, """True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR = SMA(True Range, 14 bars)

Stop distance calculation:
  raw_stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / entry_price
  stop_dist = clamp(raw_stop_dist, FLOOR_PCT, CEIL_PCT)

  Crypto: multiplier=2.5, floor=6%, ceil=15%
  Stock:  multiplier=2.0, floor=5%, ceil=10%

Trailing distance (same formula with TRAIL_MULTIPLIER):
  trail_dist = (entry_atr * ATR_TRAIL_MULTIPLIER) / HWM

Take-profit price:
  tp_dist = stop_dist * TAKE_PROFIT_RR (3.0x risk-reward)
  tp_price = entry * (1 + min(tp_dist, TP_CEIL_PCT))""")
    s(pdf, "Research from research_gap_analysis.md indicates trailing stops at 2-2.5 ATR "
      "are optimal. The 2.0x and 2.5x multipliers align with this. The floor/ceiling bounds "
      "prevent stops from being too tight (whipsawed by normal noise) or too wide (giving "
      "back excessive gains).")

    section(pdf, "6.2 Macro Regime Stop Tightening")
    s(pdf, "During market stress, stops are tightened via macro_regime.stop_mult:")
    code(pdf, """Normal conditions:   stop_mult = 1.0 (no change)
STLFSI2 > 1.0:      stop_mult *= 0.8 (20% tighter)
Stablecoin warning:  stop_mult *= 0.7 (30% tighter)

Example: crypto stop_dist = 8%, STLFSI2 stress active
  effective stop = 8% * 0.8 = 6.4% (narrower)

Rationale: During stress, correlations spike and adverse moves
accelerate. Tighter stops exit faster, limiting cascade risk.""")

    section(pdf, "6.3 Hard Stop (Loss Exit)")
    s(pdf, "The hard stop triggers when price falls below entry * (1 - stop_dist). This is "
      "the 'line in the sand' -- a non-negotiable loss limit. When triggered:")
    bullet(pdf, "Market sell immediately (no limit order, ensures fill)")
    bullet(pdf, "Record trade in trade_memory.json with exit_reason='hard_stop'")
    bullet(pdf, "Add symbol to hard_stop_lockout with 24-hour expiry")
    bullet(pdf, "Persist lockout to disk (survives bot restart)")
    s(pdf, "The 24-hour lockout prevents 'revenge trading' -- the behavioral impulse to "
      "immediately re-enter after a loss. Kahneman's prospect theory shows losses feel 2x "
      "as painful as equivalent gains, creating irrational urgency to 'recover'. The lockout "
      "enforces a cooling period.")

    section(pdf, "6.4 Trailing Stop (Profit Protection)")
    s(pdf, "The trailing stop activates only after the position gains at least "
      "ATR_TRAIL_ACTIVATE_PCT (1% for stocks, 1.5% for crypto). Before that, only the hard "
      "stop is active. Once activated, the stop follows the high-water mark:")
    code(pdf, """Activation: price >= entry * (1 + ACTIVATE_PCT)
Trigger:    price <= HWM * (1 - trail_dist)

Example progression (stock, entry=$100, ATR=2.5%):
  Price hits $101 -> trailing activated (1% gain)
  Price rises to $105 -> HWM=$105, trail=$105*(1-2%)=$102.90
  Price drops to $103 -> still above $102.90, no trigger
  Price rises to $108 -> HWM=$108, trail=$108*(1-2%)=$105.84
  Price drops to $105 -> below $105.84 -> SELL at $105

Result: Captured $5 of the $8 rise, giving back only $3.""")
    s(pdf, "The trailing stop solves the take-profit dilemma: selling too early leaves money "
      "on the table, but holding too long gives back gains. The trail allows unlimited upside "
      "while mechanically locking in profits on pullbacks.")

    section(pdf, "6.5 Take-Profit (3x Risk-Reward)")
    s(pdf, "The take-profit target is set at 3x the stop distance (TAKE_PROFIT_RR=3.0), "
      "capped at TP_CEIL_PCT (25% stocks, 30% crypto). A 3:1 risk-reward ratio means that "
      "even with a 25% win rate, the system breaks even. With our typical 40-55% win rate, "
      "the 3:1 ratio generates positive expectancy.")
    s(pdf, "For stocks, server-side bracket orders are used -- the take-profit is submitted "
      "as a child order when the buy fills. This eliminates the race condition where the bot "
      "misses a rapid move through the TP level.")

    section(pdf, "6.6 Stock Bracket Orders & Trailing Upgrade")
    s(pdf, "Stocks use Alpaca's bracket order type, which creates a parent buy order with "
      "two child orders (stop-loss and take-profit). When the parent fills, children become "
      "active automatically. When either child fills, the other is automatically cancelled. "
      "This prevents 'orphan orders' that could create unintended positions.")
    s(pdf, "After +1% gain, the server-side stop is cancelled and replaced with a trailing "
      "stop order that Alpaca manages. This offloads stop management to the broker, "
      "eliminating latency risk if the bot's connection drops.")

    section(pdf, "6.7 Trade Memory Integration")
    s(pdf, "Every exit (hard stop, trailing, take-profit, prediction sell, LLM veto, flatten) "
      "is recorded in trade_memory.json with: entry/exit price, P&L %, exit reason, LLM score "
      "at entry, and reasoning. This data feeds: (1) Kelly criterion for future position "
      "sizing, (2) LLM prompt injection ('last 10 trades on BTC: 7W/3L, avg +1.2%'), and "
      "(3) the Monte Carlo validation in hypersearch_v2.py.")


# =================================================================
# CHAPTER 10 -- ADAPTIVE (EXPANDED)
# =================================================================
def ch10_expanded(pdf):
    chapter(pdf, 10, "Adaptive Hyperparameter Management")

    section(pdf, "10.1 The Boundary Stagnation Problem")
    s(pdf, "Hyperparameter optimization searches within predefined bounds. If the true optimum "
      "lies outside those bounds, the search will converge to the boundary without finding it. "
      "For example, if seq_len is bounded [8, 40] and the true optimum is at seq_len=4, Optuna "
      "will concentrate samples near 8 (the boundary) and never explore shorter sequences.")
    s(pdf, "This is a well-known problem in Bayesian optimization. The adaptive config system "
      "detects this situation and automatically expands the search space, preventing the search "
      "from 'getting stuck' at boundaries.")

    section(pdf, "10.2 Edge Detection Algorithm")
    s(pdf, "After each search completes, detect_edges() examines whether the best parameters "
      "found are 'at the edge' of the search space:")
    code(pdf, """For categorical params (forward_bars, batch_size, n_heads):
  Check if best value = first or last in sorted list
  Example: forward_bars=[12,18,24,32,48], best=48 -> HIGH EDGE

For range params (seq_len, hidden_dim, dropout, etc.):
  Compute fractional distance from boundary:
    low_frac = (value - min) / (max - min)
    high_frac = (max - value) / (max - min)
  If either fraction <= 0.10 (within 10% of boundary): EDGE

Example: hidden_dim range [64, 384], best=374
  high_frac = (384 - 374) / (384 - 64) = 0.031 <= 0.10 -> HIGH EDGE
  This means the optimum is likely beyond 384.""")

    section(pdf, "10.3 Search Space Expansion")
    s(pdf, "When edges are detected, the system pulls new boundary values from expansion pools. "
      "Each parameter has predefined low and high expansion values:")
    code(pdf, """Expansion pools:
  forward_bars: low=[8], high=[64, 96]
  seq_len: low=[4], high=[48]
  hidden_dim: low=[32], high=[512]
  batch_size: low=[256], high=[4096]
  num_layers: low=[], high=[3]
  n_heads: low=[1], high=[8]
  dropout: low=[0.05], high=[0.50]
  learning_rate: low=[2e-4], high=[5e-3]
  weight_decay: low=[5e-6], high=[1e-3]
  huber_delta: low=[0.3], high=[3.0]
  trade_threshold: low=[0.03], high=[1.5]

Hard limits (never exceeded):
  seq_len: [4, 64], hidden_dim: [32, 512], etc.

IMPORTANT: If categorical params change (adding new values to
forward_bars or batch_size), the Optuna study DB is DELETED because
the TPE sampler's model is incompatible with changed distributions.""")

    section(pdf, "10.4 Mode Transitions")
    s(pdf, "The system alternates between two modes based on search progress:")
    code(pdf, """REFINE mode (70 trials):
  - Used when no edges detected and recent improvement
  - Optuna concentrates search near the current best
  - Fine-tunes local optimum

EXPLORE mode (120 trials):
  - Triggered when: ANY edge detected, OR 3+ cycles without >5% improvement
  - Search space may be expanded
  - More trials to cover the enlarged space
  - After explore completes, toggles back to refine

INITIAL mode (200 trials):
  - First run only
  - Full exploration of default space

Improvement threshold: 5% -- a new best must exceed the previous best
by at least 5% to count as 'improvement'. This prevents counting
noise as progress.""")

    section(pdf, "10.5 State Persistence")
    s(pdf, "All adaptive state is persisted to adaptive_state_{asset_type}.json (e.g., "
      "adaptive_state_crypto.json). This includes: best_score, best_params, current search "
      "space (which may differ from defaults after expansion), mode, cycles_without_improvement, "
      "and a complete expansion_history log with timestamps and details of each expansion. "
      "Atomic writes (write-then-rename) ensure the state file is never corrupted.")


# =================================================================
# CHAPTERS 12-15 (EXPANDED RISK COMPONENTS)
# =================================================================
def ch12_expanded(pdf):
    chapter(pdf, 12, "GARCH Volatility Forecasting")

    section(pdf, "12.1 The Problem with ATR")
    s(pdf, "Average True Range (ATR) equally weights all recent bars -- bar 1 and bar 14 "
      "contribute equally. This means ATR adapts slowly to regime changes. If volatility "
      "suddenly doubles (as in a market crash), ATR takes 14 bars to fully reflect the new "
      "regime. During those 14 bars, stops are too tight and positions are too large.")

    section(pdf, "12.2 GARCH Theory (Robert Engle, Nobel Prize 2003)")
    s(pdf, "Robert Engle developed ARCH (Autoregressive Conditional Heteroskedasticity) in "
      "1982 and its generalization GARCH in collaboration with Tim Bollerslev (1986). The key "
      "insight is that financial volatility is NOT constant -- it clusters. High-volatility "
      "periods tend to be followed by more high-volatility periods, and calm periods follow "
      "calm periods. This is called 'volatility clustering' and is one of the most robust "
      "empirical facts in all of finance.")
    s(pdf, "The GARCH(1,1) model captures this with three parameters:")
    code(pdf, """sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

where:
  sigma_t^2 = conditional variance at time t
  omega     = baseline variance (long-run average, ~constant)
  alpha     = shock coefficient (how much the latest return moves vol)
  beta      = persistence (how much yesterday's vol carries forward)

Constraints: alpha > 0, beta > 0, alpha + beta < 1 (for stationarity)
Typical values: alpha ~ 0.05-0.15, beta ~ 0.80-0.95

Interpretation:
  - alpha + beta close to 1: highly persistent vol (crypto, equities)
  - Large alpha: vol reacts quickly to shocks
  - Large beta: vol changes slowly (long memory)""")
    s(pdf, "GARCH's advantage over ATR: exponential decay of old observations. Recent shocks "
      "matter more than old ones. When volatility spikes, GARCH responds within 1-2 bars "
      "rather than 14. When volatility calms, GARCH reflects it faster too.")

    section(pdf, "12.3 EGARCH Extension")
    s(pdf, "Standard GARCH treats positive and negative shocks symmetrically. In practice, "
      "crashes increase volatility MORE than rallies of equal magnitude. EGARCH (Exponential "
      "GARCH, Nelson 1991) captures this asymmetry with a log-variance specification:")
    code(pdf, """log(sigma_t^2) = omega + alpha * (|z_{t-1}| - E[|z|]) + gamma * z_{t-1}
                                + beta * log(sigma_{t-1}^2)

where z = standardized residual, gamma captures asymmetry.
gamma < 0 means negative returns increase vol more than positive
(the 'leverage effect', named after firm leverage increasing in downturns)""")
    s(pdf, "Our implementation tries EGARCH first and falls back to standard GARCH if EGARCH "
      "fails to converge. This gives us asymmetric vol estimation when data supports it, "
      "and robust standard estimation otherwise.")

    section(pdf, "12.4 Implementation Details (volatility.py)")
    code(pdf, """fit_garch(returns, p=1, q=1):
  - Requires >= 100 data points
  - Scales returns to percentage if < 0.01
  - Try EGARCH: arch_model(vol='EGARCH', mean='Zero')
  - Fallback to GARCH: arch_model(vol='Garch', mean='Zero')
  - mean='Zero': don't model return direction (just variance)

forecast_volatility(model) -> sigma (decimal):
  - 1-step-ahead variance forecast
  - sigma = sqrt(variance) / 100 (percentage -> decimal)

get_cached_sigma(symbol, returns):
  - Cache: 1 hour (_REFIT_INTERVAL=3600s)
  - Refit hourly because vol is persistent (beta ~0.90)
  - Between refits, forecast from cached model

compute_vol_adjusted_size(base, sigma, target=0.02):
  - ratio = target / sigma, clamped [0.5, 2.0]
  - return base * ratio
  - Effect: each position targets same dollar-volatility""")

    section(pdf, "12.5 Three Uses of GARCH in the System")
    bullet(pdf, "1. Position sizing (Layer 4): compute_vol_adjusted_size() scales each "
           "position to target 2% daily vol, creating risk parity across the portfolio.")
    bullet(pdf, "2. Stop-loss placement: get_garch_stop() computes stop distance as "
           "sigma * multiplier, clamped to [3%, 10%]. More responsive than ATR-based stops.")
    bullet(pdf, "3. Risk parity: By normalizing position sizes to volatility, every position "
           "contributes roughly the same amount of portfolio risk, regardless of asset class.")


def ch13_expanded(pdf):
    chapter(pdf, 13, "Macro Regime Detection")

    section(pdf, "13.1 Why Macro Indicators Matter")
    s(pdf, "The LSTM is trained on historical data and excels at detecting patterns in normal "
      "market conditions. But systemic crises -- bank runs, exchange collapses, regulatory "
      "shocks -- are 'tail events' that occur too rarely and too differently each time for "
      "the model to learn from. These are the 'unknown unknowns' that macro indicators detect.")
    s(pdf, "The 2022 Nobel Prize to Bernanke, Diamond, and Dybvig highlighted that financial "
      "crises are not random -- they have detectable precursors in financial stress indicators, "
      "volatility, and systemic risk measures. Our system monitors four such indicators.")

    section(pdf, "13.2 VIX (CBOE Volatility Index)")
    s(pdf, "The VIX measures implied volatility of S&P 500 options over the next 30 days. "
      "It is often called the 'fear gauge' because it rises when options traders expect large "
      "moves (typically downward). Key levels:")
    code(pdf, """VIX < 12:  Extremely low vol (complacent, often precedes correction)
VIX 12-15: Normal, calm markets
VIX 15-20: Slightly elevated, some concern
VIX 20-25: Elevated, meaningful uncertainty
VIX 25-30: High, significant market stress
VIX 30-40: Very high, crisis-level fear
VIX > 40:  Extreme (2008 GFC hit 80, COVID crash hit 82)

Our thresholds:
  VIX > 35: CRISIS -> halt all stock entries, sizing=0.3x
  VIX > 25: DEFENSIVE -> block non-safe-havens, sizing=0.5x
  VIX > 15: CAUTION -> sizing=0.8x""")
    s(pdf, "Source: yfinance (^VIX). Cached 1 hour. When VIX > 20, cache is invalidated "
      "on each check for faster reaction to deteriorating conditions.")

    section(pdf, "13.3 STLFSI2 (St. Louis Financial Stress Index)")
    s(pdf, "The STLFSI2, published weekly by the Federal Reserve Bank of St. Louis, combines "
      "18 financial indicators into a single stress measure. It includes: yield spreads, "
      "volatility indices, and financial market indicators. The index is measured in standard "
      "deviations from the mean, so a value of 0 represents normal conditions.")
    code(pdf, """STLFSI2 interpretation:
  < 0:    Below-average stress (calm markets)
  0-0.5:  Normal stress
  0.5-1.0: Mildly elevated
  > 1.0:  Significantly elevated (our threshold)
  > 2.0:  Severe (approaching crisis)

Our rule: STLFSI2 > 1.0 -> sizing *= 0.5, stop_mult *= 0.8
  50% position reduction + 20% tighter stops""")
    s(pdf, "Source: FRED CSV API (free, no API key). Cached 1 day (weekly data).")

    section(pdf, "13.4 CAPE Ratio (Shiller PE)")
    s(pdf, "Robert Shiller (Nobel 2013) developed the Cyclically Adjusted Price-to-Earnings "
      "ratio (CAPE) as a measure of long-term stock market valuation. It divides the current "
      "S&P 500 price by the average inflation-adjusted earnings over the past 10 years, "
      "smoothing out business cycle effects.")
    code(pdf, """Historical CAPE statistics:
  Mean:  ~25    Median: ~22
  Std:   ~8     Current (2026): ~35+

Our calculation: CAPE_est = SPY_PE * 1.6 (empirical adjustment)
Z-score: (CAPE - 25.0) / 8.0
If z > 1.5 (CAPE > 37): sizing *= 0.7 (30% reduction)

Interpretation: high CAPE means the market is expensive relative to
historical earnings. Returns over the next 10 years are likely lower.
This doesn't predict crashes, but reduces exposure to overvalued markets.""")

    section(pdf, "13.5 Stablecoin Peg Monitoring")
    s(pdf, "Stablecoins (USDT, USDC) are the 'plumbing' of crypto markets. When they depeg "
      "(trade significantly above or below $1.00), it signals systemic stress in the crypto "
      "ecosystem. The 2022 UST/LUNA collapse demonstrated how stablecoin failure cascades "
      "through all crypto markets.")
    code(pdf, """Monitoring: USDT/USD and USDC/USD midpoint prices via Alpaca
  Warning (>0.5% off $1): stop_mult *= 0.7 (tighten stops 30%)
  Emergency (>2% off $1): sizing_mult = 0.0 (HALT ALL CRYPTO TRADING)

5-minute cache TTL (fast reaction needed)

The emergency halt is absolute -- it overrides all other sizing.
A 2% stablecoin depeg indicates potential systemic failure similar
to Diamond-Dybvig bank run dynamics (Nobel 2022).""")


def ch14_expanded(pdf):
    chapter(pdf, 14, "Correlation-Aware Sizing")

    section(pdf, "14.1 The Diversification Problem")
    s(pdf, "Harry Markowitz's seminal 1952 paper 'Portfolio Selection' (Nobel Prize 1990) "
      "demonstrated that portfolio risk depends not just on individual asset risks, but on "
      "how assets move together. A portfolio of 10 perfectly correlated stocks has the same "
      "risk as a single stock -- no diversification benefit at all.")
    s(pdf, "In crypto markets, this is particularly dangerous. BTC, ETH, SOL, and most "
      "altcoins have correlations of 0.7-0.9 during normal times, rising to 0.95+ during "
      "crashes. Without correlation-aware sizing, a 'diversified' crypto portfolio of 6 "
      "coins is really a leveraged bet on BTC with extra steps.")

    section(pdf, "14.2 Ledoit-Wolf Shrinkage Estimation")
    s(pdf, "The sample correlation matrix (computed from 30 days of returns) is noisy when "
      "the number of assets is comparable to the number of observations. Ledoit and Wolf "
      "(2004) proposed shrinking the sample correlation matrix toward a structured target "
      "(typically the identity matrix) to reduce estimation error. The optimal shrinkage "
      "intensity is data-driven.")
    code(pdf, """Process:
  1. Compute sample covariance from 30-day returns
  2. Ledoit-Wolf shrinkage: LedoitWolf().fit(X)
     -> Regularizes toward identity matrix
     -> Optimal shrinkage intensity computed analytically
  3. Convert to correlation: corr = cov / (std * std)
  4. Fallback to numpy corrcoef if sklearn unavailable

Minimum 10 overlapping observations per pair
Cache: 1 hour (correlations are relatively stable intraday)""")

    section(pdf, "14.3 Entry Gate: Portfolio Correlation Check")
    code(pdf, """Before buying candidate:
  1. For each held position: look up |corr(candidate, position)|
  2. avg_corr = mean of absolute correlations
  3. If avg_corr > 0.7: REJECT

Example: Hold BTC, ETH. Candidate: SOL
  corr(SOL, BTC) = 0.85
  corr(SOL, ETH) = 0.90
  avg = 0.875 -> REJECTED (> 0.7)

Example: Hold AAPL, MSFT. Candidate: XOM
  corr(XOM, AAPL) = 0.35
  corr(XOM, MSFT) = 0.40
  avg = 0.375 -> ACCEPTED (< 0.7)""")

    section(pdf, "14.4 Sizing Factor: Continuous Reduction")
    s(pdf, "Even for accepted candidates (avg_corr < 0.7), correlation still reduces sizing:")
    code(pdf, """sizing_factor = max(0.5, 1.0 - 0.5 * avg_corr)

Correlation -> Factor -> Effect:
  0.00 -> 1.00 -> Full size (perfectly uncorrelated)
  0.20 -> 0.90 -> 10% reduction
  0.40 -> 0.80 -> 20% reduction
  0.60 -> 0.70 -> 30% reduction
  0.70 -> 0.65 -> 35% reduction (near rejection)
  1.00 -> 0.50 -> 50% reduction (theoretical maximum)""")

    section(pdf, "14.5 Limitations")
    s(pdf, "Markowitz's framework assumes correlations are stable. In reality, correlations "
      "spike toward 1.0 during crises (Bernanke, Nobel 2022) -- exactly when diversification "
      "is needed most. Our 0.7 threshold provides a buffer: positions that are 0.6 correlated "
      "in normal times may reach 0.85+ during stress, still below the theoretical 1.0. "
      "The Ledoit-Wolf shrinkage partially addresses estimation noise, but cannot solve "
      "the fundamental instability of crisis-time correlations.")


def ch15_expanded(pdf):
    chapter(pdf, 15, "HMM Regime Detection")

    section(pdf, "15.1 Why Regimes Matter")
    s(pdf, "Financial markets switch between distinct 'regimes' -- periods of different "
      "return distributions. A bull market has positive mean returns and moderate volatility. "
      "A bear market has negative mean returns and high volatility. A ranging market has near-zero "
      "mean returns. An LSTM trained on all data sees the average of these regimes, but the "
      "optimal trading strategy differs dramatically between them.")
    s(pdf, "Thomas Sargent and Christopher Sims (Nobel Prize 2011) developed regime-switching "
      "models that formalize this insight. Their work showed that economic time series are "
      "better described by models that allow structural breaks -- abrupt changes in the "
      "data-generating process.")

    section(pdf, "15.2 Hidden Markov Model Theory")
    s(pdf, "An HMM assumes the system transitions between N hidden states according to a "
      "Markov chain (next state depends only on current state, not history). Each state "
      "generates observations from a different probability distribution. The 'hidden' part "
      "means we don't directly observe which state we're in -- we infer it from the "
      "observations (returns).")
    code(pdf, """Our 3-state Gaussian HMM:
  State 0 (Bear):    mean_return < 0, high volatility
  State 1 (Neutral): mean_return ~ 0, moderate volatility
  State 2 (Bull):    mean_return > 0, moderate volatility

Parameters estimated from data:
  - Transition matrix: prob of switching states (e.g., P(bull->bear)=0.02)
  - Emission means: average return per state
  - Emission covariance: volatility per state

Fitting: GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
Requires: >= 200 data points (hourly returns)""")

    section(pdf, "15.3 Regime-Specific Trading Adjustments")
    code(pdf, """Bull regime:
  sizing_mult=1.2x    (the only layer that INCREASES sizing)
  threshold_mult=0.8x (lower threshold -> take more trades)
  stop_mult=1.0x      (normal stops)
  Rationale: favorable momentum, lean into signals

Bear regime:
  sizing_mult=0.3x    (70% reduction -- capital preservation)
  threshold_mult=1.5x (higher threshold -> only strongest signals)
  stop_mult=0.8x      (20% tighter stops)
  Rationale: adverse conditions, protect capital

Neutral/normal:
  sizing_mult=1.0x, threshold_mult=1.0x, stop_mult=1.0x

High-volatility neutral (vol > 1.5x median):
  sizing_mult=0.5x    (reduce for uncertainty)
  threshold_mult=1.2x (somewhat higher bar)
  stop_mult=1.3x      (WIDER stops -- avoid whipsaw in vol)
  Rationale: wide moves both directions, need room""")

    section(pdf, "15.4 Whipsaw Prevention (Smoothing)")
    s(pdf, "HMMs can oscillate between states rapidly if the current returns are near a "
      "state boundary. Without smoothing, position sizes would change every bar, generating "
      "excessive trading costs. The smoothing algorithm requires 3 consecutive bars in a new "
      "regime before switching:")
    code(pdf, """_smooth_regime(symbol, regime):
  If first observation: return neutral default (don't commit)
  If same label as previous: increment counter
  If counter >= 3: accept new regime (persistent enough)
  If label changed:
    Reset counter to 1
    If previous regime lasted >= 3 bars: allow switch
    Else: return neutral default (transition too choppy)

Effect: Regime must persist for ~90 seconds (3 * 30s cycles)
  before affecting position sizing. Filters out momentary blips.""")

    section(pdf, "15.5 Cache Strategy")
    s(pdf, "HMM fitting is expensive (~1-2 seconds on Jetson CPU). The model is refitted "
      "daily (_REFIT_INTERVAL=86400s), not every 30-second cycle. Between refits, the cached "
      "model is used to predict the current state from recent returns. This is valid because "
      "regime transitions are relatively slow (markets don't switch from bull to bear in "
      "one hour) and the HMM's transition matrix captures the persistence of states.")


# =================================================================
# CHAPTER 20 -- NOBEL PRIZE FOUNDATIONS (VASTLY EXPANDED)
# =================================================================
def ch20_expanded(pdf):
    chapter(pdf, 20, "Theoretical Foundations -- Nobel Prize Research")

    s(pdf, "Every significant design decision in this system is grounded in Nobel Prize-winning "
      "economic research. This chapter explains each theory in depth, its relevance to "
      "algorithmic trading, and exactly how we implemented it.")

    # ---- ENGLE ----
    section(pdf, "20.1 Robert Engle (Nobel 2003) -- GARCH Volatility Clustering")

    subsection(pdf, "The Theory")
    s(pdf, "Robert Engle's 1982 paper introduced ARCH (Autoregressive Conditional "
      "Heteroskedasticity), and with Tim Bollerslev's 1986 generalization to GARCH, "
      "revolutionized how financial economists model risk. Before Engle, volatility was "
      "assumed constant -- a 'homoskedastic' world where every day's returns are drawn from "
      "the same distribution.")
    s(pdf, "Engle's insight: volatility is NOT constant. It clusters -- high-vol days tend "
      "to follow high-vol days, low-vol days follow low-vol days. This is visible in any "
      "price chart: calm periods alternate with turbulent periods. GARCH captures this "
      "mathematically with the equation:")
    code(pdf, """sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

sigma_t^2: today's conditional variance (volatility squared)
omega:     long-run average variance (small, positive constant)
alpha:     'reaction' -- how much yesterday's squared return affects today's vol
           Typical: 0.05-0.15. High alpha = vol reacts fast to shocks.
beta:      'persistence' -- how much yesterday's vol carries forward
           Typical: 0.80-0.95. High beta = vol is sticky/persistent.
epsilon:   yesterday's 'innovation' (return minus expected return)

alpha + beta typically ~0.95-0.99 for financial assets
  -> Volatility is highly persistent
  -> Shocks decay slowly (half-life = log(0.5)/log(alpha+beta) ~ 14-69 bars)""")
    s(pdf, "The Nobel committee cited Engle's work as 'indispensable tools for financial "
      "analysts, traders, and regulators.' Before GARCH, risk models systematically "
      "underestimated the probability of large losses during volatile periods.")

    subsection(pdf, "Why GARCH Matters for Trading")
    bullet(pdf, "Forward-looking: ATR weights all recent bars equally. GARCH gives "
           "exponentially more weight to recent observations, adapting faster to regime changes.")
    bullet(pdf, "Volatility forecasting: GARCH produces a 1-step-ahead variance forecast, "
           "allowing us to anticipate tomorrow's volatility, not just measure yesterday's.")
    bullet(pdf, "Asymmetric response: EGARCH extension captures the 'leverage effect' -- "
           "crashes increase vol more than rallies, allowing faster protective reactions.")
    bullet(pdf, "Risk parity: By forecasting each asset's volatility, we can size positions "
           "so that each contributes equal risk to the portfolio.")

    subsection(pdf, "Our Implementation")
    bullet(pdf, "volatility.py: fit_garch() tries EGARCH first, falls back to standard GARCH")
    bullet(pdf, "Refitted hourly (not every 30s) because vol is persistent (beta ~0.90)")
    bullet(pdf, "Position sizing: compute_vol_adjusted_size() targets 2% daily vol per position")
    bullet(pdf, "Stop-loss: get_garch_stop() places stops at sigma * multiplier from entry")
    bullet(pdf, "Memory: < 50MB per model, fits easily on Jetson's 8GB")

    # ---- MARKOWITZ ----
    section(pdf, "20.2 Markowitz, Sharpe & Miller (Nobel 1990) -- Portfolio Theory")

    subsection(pdf, "Mean-Variance Optimization")
    s(pdf, "Harry Markowitz's 1952 paper 'Portfolio Selection' is the foundation of modern "
      "portfolio theory. His insight: investors should not evaluate assets in isolation, but "
      "consider how they interact. A portfolio of two risky assets can be LESS risky than "
      "either asset alone, if their returns are not perfectly correlated.")
    s(pdf, "Markowitz formalized this with mean-variance optimization: given expected returns "
      "and a covariance matrix, find the portfolio weights that minimize variance for a given "
      "expected return (or maximize return for a given variance). The set of optimal portfolios "
      "forms the 'efficient frontier' -- any portfolio not on this frontier is suboptimal.")

    subsection(pdf, "The Sharpe Ratio")
    s(pdf, "William Sharpe (Nobel 1990) simplified Markowitz's framework with the Capital "
      "Asset Pricing Model (CAPM) and the Sharpe ratio: return / volatility. This single "
      "number captures risk-adjusted performance. Our Optuna search maximizes the Sharpe ratio "
      "because: (1) it penalizes strategies that achieve returns through excessive risk, "
      "(2) it's comparable across assets and timeframes, and (3) it's the industry standard "
      "for quantitative strategy evaluation.")
    code(pdf, """Sharpe = (mean_return - risk_free_rate) / std_return
Annualized: Sharpe * sqrt(trading_periods_per_year)

Interpretation:
  Sharpe < 0:   Losing money
  Sharpe 0-0.5: Marginal edge
  Sharpe 0.5-1: Decent strategy
  Sharpe 1-2:   Strong strategy (rare after costs)
  Sharpe > 2:   Probably overfit (realistic with costs: ~0.33)""")

    subsection(pdf, "Our Implementation")
    bullet(pdf, "portfolio.py: correlation-aware position sizing using Ledoit-Wolf shrinkage")
    bullet(pdf, "Rejection gate: avg correlation > 0.7 blocks entry")
    bullet(pdf, "Sizing factor: continuous reduction proportional to correlation")
    bullet(pdf, "Sharpe ratio is the objective function for all Optuna hyperparameter search")

    subsection(pdf, "Limitations We Address")
    s(pdf, "Markowitz optimization is notoriously sensitive to estimation errors in expected "
      "returns and covariances. Small changes in inputs produce wildly different optimal "
      "portfolios. We address this with: (1) Ledoit-Wolf shrinkage for robust covariance "
      "estimation, (2) simple correlation rejection rather than full mean-variance optimization "
      "(more robust), and (3) correlation-based sizing factors rather than exact optimal weights.")

    # ---- FAMA/HANSEN/SHILLER ----
    section(pdf, "20.3 Fama, Hansen & Shiller (Nobel 2013) -- Asset Pricing")

    subsection(pdf, "Eugene Fama: Efficient Markets Hypothesis")
    s(pdf, "Fama's Efficient Market Hypothesis (EMH) states that asset prices reflect all "
      "available information, making it impossible to consistently 'beat the market' through "
      "information-based trading. In its strong form, EMH says even insider information is "
      "reflected in prices.")
    s(pdf, "Implication for our system: Markets ARE efficient in aggregate, but edges exist "
      "in LESS efficient segments. Crypto markets are less efficient than large-cap equities "
      "(fewer institutional participants, higher information asymmetry, 24/7 trading). "
      "Small-cap stocks are less efficient than mega-caps. Our system focuses on exploiting "
      "these pockets of inefficiency.")

    subsection(pdf, "Lars Peter Hansen: Stochastic Discount Factors")
    s(pdf, "Hansen's Generalized Method of Moments (GMM) framework showed that asset returns "
      "are fundamentally tied to economic risk factors through stochastic discount factors. "
      "The key insight for trading: VOLATILITY is a state variable that drives returns. "
      "Risk is not just return variance -- it's time-varying and predictable.")
    s(pdf, "This directly justifies our GARCH implementation: by modeling time-varying "
      "volatility, we're capturing a fundamental driver of risk premiums.")

    subsection(pdf, "Robert Shiller: Irrational Exuberance & CAPE")
    s(pdf, "Shiller showed that stock prices are FAR more volatile than justified by changes "
      "in fundamental value (dividends). This 'excess volatility' is driven by behavioral "
      "factors -- bubbles, panics, and narrative-driven trading. He developed the CAPE ratio "
      "(Cyclically Adjusted PE) as a long-term valuation indicator.")
    s(pdf, "When CAPE is historically high (> 1.5 standard deviations above mean), expected "
      "future returns are lower. Our macro_indicators.py reduces position sizing when "
      "CAPE z-score exceeds 1.5, implementing Shiller's valuation-based risk management.")

    # ---- KAHNEMAN ----
    section(pdf, "20.4 Daniel Kahneman (Nobel 2002) -- Prospect Theory")

    subsection(pdf, "The Theory")
    s(pdf, "Kahneman and Amos Tversky's Prospect Theory (1979) overturned the assumption that "
      "humans are rational economic agents. Their experiments demonstrated systematic biases "
      "in how people evaluate risks and rewards:")
    bullet(pdf, "Loss aversion: Losses feel approximately 2x as painful as equivalent gains "
           "feel good. A $100 loss hurts about twice as much as a $100 gain pleases.")
    bullet(pdf, "Reference dependence: People evaluate outcomes relative to a reference point "
           "(usually their entry price), not in absolute terms. A stock at $105 feels like a "
           "'$5 gain' even if it fell from $120.")
    bullet(pdf, "Certainty effect: People overweight certain outcomes vs. probable ones. "
           "This creates a tendency to lock in small certain gains (selling winners early) "
           "rather than holding for larger uncertain gains.")
    bullet(pdf, "Probability distortion: People overweight small probabilities (lottery "
           "tickets, tail-risk events) and underweight moderate probabilities.")

    subsection(pdf, "The Disposition Effect")
    s(pdf, "The most destructive combination of these biases for traders is the 'disposition "
      "effect': holding losing positions too long (hoping to avoid realizing the loss) while "
      "selling winning positions too early (to lock in the certain gain). Research by Odean "
      "(1998) found retail traders sell winners 50% more frequently than losers.")
    s(pdf, "However, Odean also found the disposition effect is 'virtually zero among "
      "algorithms.' Our mechanical stops and trailing stops eliminate this bias entirely -- "
      "but ONLY if we never override them manually. This is why the system has no manual "
      "override capability by design.")

    subsection(pdf, "Our Implementation")
    bullet(pdf, "Mechanical stops eliminate the disposition effect (no manual override)")
    bullet(pdf, "ATR-based trailing stops allow winners to run (overcome certainty effect)")
    bullet(pdf, "Hard-stop lockout prevents revenge trading (overcome loss aversion)")
    bullet(pdf, "Sentiment analysis exploits others' loss aversion (buy when others panic-sell)")
    bullet(pdf, "FnG >= 90 (extreme greed) triggers bubble protection (others' overconfidence)")

    # ---- THALER ----
    section(pdf, "20.5 Richard Thaler (Nobel 2017) -- Behavioral Economics")

    subsection(pdf, "The Theory")
    s(pdf, "Thaler brought behavioral insights from psychology into mainstream economics. His "
      "key contributions relevant to trading:")
    bullet(pdf, "Mental accounting: People treat money differently based on arbitrary "
           "categories. A trader might treat 'house money' (profits) as less valuable than "
           "original capital, leading to reckless bets with winnings.")
    bullet(pdf, "Endowment effect: People value things they own more than identical things "
           "they don't own. A trader may hold a losing position simply because they 'own' it.")
    bullet(pdf, "Nudge theory: Designing choice architectures that guide people toward better "
           "decisions. Our system IS a nudge architecture -- it automates decisions that "
           "humans systematically make incorrectly.")

    subsection(pdf, "Calendar Anomalies (Exploitable Behavioral Patterns)")
    s(pdf, "Thaler and others documented numerous calendar anomalies that persist because "
      "they're rooted in human behavior:")
    bullet(pdf, "Turn-of-month effect: Returns are higher on the last and first few trading "
           "days of each month (payday flows, institutional rebalancing)")
    bullet(pdf, "January effect: Small-cap stocks outperform in January (tax-loss selling "
           "reversal from December)")
    bullet(pdf, "Day-of-week effect: Mondays historically show lower returns")
    bullet(pdf, "Pre-holiday effect: Markets rise before holidays")
    s(pdf, "Our indicators.py implements Month_sin, Month_cos, Day_sin, Day_cos, and "
      "Turn_of_Month as features, allowing the LSTM to learn these patterns. Many anomalies "
      "have weakened since publication, but some persist in less-efficient markets (crypto).")

    subsection(pdf, "Our Core Behavioral Edge")
    s(pdf, "The most important insight from Kahneman and Thaler for our system: mechanical "
      "execution IS our competitive advantage. Our edge is NOT better predictions (many "
      "models predict similarly). Our edge is that we EXECUTE WITHOUT BIAS. Human traders "
      "panic-sell at bottoms, chase rallies at tops, revenge-trade after losses, and cut "
      "winners short. We don't. This advantage persists precisely because it's rooted in "
      "human psychology that doesn't change.")

    # ---- BERNANKE ----
    section(pdf, "20.6 Bernanke, Diamond & Dybvig (Nobel 2022) -- Financial Crises")

    subsection(pdf, "The Theory")
    s(pdf, "The 2022 Nobel recognized research on financial crises -- the most dangerous "
      "events for any trading system. Ben Bernanke showed how banking system failures amplify "
      "and prolong economic downturns. Diamond and Dybvig developed the formal model of bank "
      "runs, proving they can be RATIONAL -- when depositors believe others will withdraw, "
      "withdrawing is the optimal individual strategy even if the bank is fundamentally sound.")

    subsection(pdf, "Application to Crypto")
    s(pdf, "The Diamond-Dybvig bank run model applies directly to crypto exchanges and "
      "stablecoins. The 2022 Terra/LUNA collapse followed the exact bank-run dynamics: "
      "UST depegged -> panic withdrawals -> algorithmic selling -> deeper depeg -> cascade. "
      "This destroyed $40B in value in days.")
    s(pdf, "Our stablecoin monitoring (macro_indicators.py) implements a depeg circuit breaker: "
      "at 2% deviation from $1.00, ALL crypto trading is halted. This is the algorithmic "
      "equivalent of 'getting in line at the bank before the run completes.'")

    subsection(pdf, "Financial Stress Monitoring")
    s(pdf, "Bernanke demonstrated that financial crises have observable precursors in credit "
      "spreads, interbank rates, and volatility. Our STLFSI2 monitoring captures these "
      "precursors. When the stress index exceeds 1 standard deviation, we reduce sizing by "
      "50% and tighten stops by 20% -- positioning defensively before the crisis fully "
      "materializes.")

    subsection(pdf, "Correlation Spikes")
    s(pdf, "During crises, asset correlations spike toward 1.0 -- 'everything falls together.' "
      "Our correlation gate (max 0.7) provides a buffer: positions that are 0.5-0.6 correlated "
      "in normal times may reach 0.8-0.9 during stress, but we've already limited our exposure. "
      "Without this gate, a 6-coin crypto portfolio with 0.85 correlations is effectively a "
      "single leveraged bet on BTC.")

    # ---- SARGENT/SIMS ----
    section(pdf, "20.7 Sargent & Sims (Nobel 2011) -- Regime-Switching Models")

    subsection(pdf, "The Theory")
    s(pdf, "Thomas Sargent and Christopher Sims developed methods for understanding how "
      "economic policy and structural changes affect the economy. Sims' vector autoregression "
      "(VAR) models allowed economists to identify structural 'shocks' and trace their "
      "propagation through the economy. Sargent's work on rational expectations showed how "
      "agents' beliefs about regime changes affect current behavior.")
    s(pdf, "Their regime-switching framework -- the idea that economic systems transition "
      "between distinct states with different dynamics -- is implemented in our HMM regime "
      "detector. The market's current 'regime' (bull/bear/neutral/high-vol) determines "
      "optimal position sizing and threshold adjustments.")

    subsection(pdf, "Our Implementation")
    s(pdf, "regime_detector.py fits a 3-state Gaussian HMM to return series. States are "
      "labeled by their mean return (ascending: bear, neutral, bull). The HMM's transition "
      "matrix captures how likely it is to switch regimes. Whipsaw prevention requires 3 "
      "consecutive bars in a new regime before adjusting positions.")

    # ---- MILGROM/WILSON ----
    section(pdf, "20.8 Milgrom & Wilson (Nobel 2020) -- Auction Theory")

    subsection(pdf, "The Winner's Curse")
    s(pdf, "Robert Wilson formalized the 'winner's curse': in a common-value auction (where "
      "the item has the same true value to all bidders, but each has imperfect information), "
      "the winning bidder tends to have the most OPTIMISTIC estimate -- and therefore overpays. "
      "Winning is actually BAD news about the true value.")
    s(pdf, "In trading: when our model gives a strong BUY signal and the price has already "
      "moved up significantly, we're 'winning an auction' that other market participants have "
      "passed on. If everyone else is selling at this price, maybe they know something we "
      "don't. This is adverse selection.")

    subsection(pdf, "Our Implementation")
    s(pdf, "Winner's curse filter (base_loop.py): when price > SMA20 + 2*ATR (extended move), "
      "require prediction >= 1.5x threshold. This demands much stronger conviction for "
      "entries where the auction dynamics are most adverse.")
    s(pdf, "Spread-aware limit pricing (order_utils.py): limit orders below the ask hide "
      "our urgency, reducing information leakage to market makers. Market orders are used "
      "only as a fallback after 30s, revealing urgency only when necessary.")

    # ---- AKERLOF/SPENCE/STIGLITZ ----
    section(pdf, "20.9 Akerlof, Spence & Stiglitz (Nobel 2001) -- Information Asymmetry")

    subsection(pdf, "The Theory")
    s(pdf, "Akerlof's 'Market for Lemons' showed how information asymmetry destroys markets. "
      "When sellers know more about product quality than buyers, buyers assume the worst, "
      "driving prices down until only 'lemons' are offered. Stiglitz showed how screening "
      "mechanisms (deductibles in insurance, credit scores in lending) address this.")

    subsection(pdf, "Application to Trading")
    s(pdf, "In financial markets, information asymmetry manifests as widening bid-ask spreads. "
      "When market makers are uncertain about the true value (high information asymmetry), "
      "they widen spreads to protect against informed traders. Our spread edge check "
      "(should_trade: pred > 2x spread) ensures we only trade when our predicted edge "
      "exceeds the cost of adverse selection embedded in the spread.")

    # ---- MERTON/SCHOLES ----
    section(pdf, "20.10 Merton & Scholes (Nobel 1997) -- Volatility as Risk")

    subsection(pdf, "The Theory")
    s(pdf, "Robert Merton and Myron Scholes (with Fischer Black, who died before the prize) "
      "developed option pricing theory. Their central insight: volatility is the KEY risk "
      "factor in financial markets. The Black-Scholes formula prices options as a function of "
      "five variables, with volatility being the only one not directly observable. This led "
      "to the concept of 'implied volatility' -- the market's forward-looking estimate of "
      "future price variation.")

    subsection(pdf, "Application")
    s(pdf, "Their work validates our heavy use of volatility in risk management: VIX (implied "
      "vol), GARCH (historical vol forecasting), ATR (realized vol for stops). Volatility "
      "is not just a risk measure -- it's the primary driver of option premiums, stop-loss "
      "placement, and position sizing. Our entire risk stack is, at its core, a volatility "
      "management system.")
