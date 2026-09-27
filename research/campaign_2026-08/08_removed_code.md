# Campaign 2026-08 — Phase 8: Removed-Code Archive

**Purpose.** Every code block removed by the influence-implementation wave is preserved here
VERBATIM, with its original `file:line` span (working-tree line numbers at the moment of
removal, atop the uncommitted post-R2 state over `c7f846e`), the decision-influence ledger
verdict that removed it (`07_decision_influences.md`), the measured/derived rationale, and
restoration instructions. If a FULL module is ever removed it moves to
`research/code_archive/<name>.py.archived` instead (none this wave).

**Owner ruling (verbatim intent, 2026-08-22):** *"Implement their suggestions. For all code
removed, archive it please (if full modules are removed). Otherwise keep research too about
code removed. Still noteworthy."* This ruling resolved KILL_LIST pending ask #3 (pseudo-CAPE
deletion) and authorized the ledger's unanimous dead-code removals. Evidence-gated verdicts
(DERISK_STACK_V2 flip, LLM veto fate, q10 retirement, entry-window fate, top-N tuning) were
NOT shipped — they stay flag-gated/default-OFF pending their stated Jetson evidence.

**Restoration discipline.** Nothing in this file is a template for casual re-addition. Each
section states what would have to be TRUE (evidence, owner ruling, or both) before the block
could return. Kill-listed items additionally require an explicit owner decision to leave
`research/KILL_LIST.md` — archival here does not soften the kill.

Later implementation packets APPEND their removals as new sections below.

---

## IA-1.1 — Pseudo-CAPE: fetcher, 0.7x sizing haircut, and v2 exclusion-announce machinery

**Removed from:** `macro_indicators.py` (5 blocks) — 2026-08-22, packet IA-1.
**Ledger verdict:** §3.5 pseudo-CAPE 0.7x — **"NO — unanimous, the cleanest verdict in the
audit."** Direct quote: *"Kill-listed by three sources ('fake data driving a real haircut'),
still cutting ~every stock entry 30% when its fabricated z>1.5. Min-aggregation does NOT
launder it (a fake signal inside a min still binds when it is the minimum) — v2's exclusion
plus code deletion (pending ask #3) is the only sound treatment. Needs no measurement: the
input is fictional by construction."*
**Kill-list citations:** `research/KILL_LIST.md` line ~76 — "Pseudo-CAPE (SPY P/E×1.6 proxy)
— fake data driving a real haircut — [rev-07-01, econ-07, nobel-07]" — three independent
sources. Deletion owner-ruled 2026-08-22 (pending ask #3 → RULED; annotated in KILL_LIST).
**Classification:** direct-ship, explicitly owner-ruled. This is the ONE legacy-behavior
change in packet IA-1 that is not provably-no-op: removing the haircut is strictly
size-INCREASING on stock entries whenever the fabricated z-score exceeded 1.5 (it removes an
un-founded 30% haircut; SPY trailing P/E ≈ 25+ makes z = (P/E·1.6 − 25)/8 > 1.5 the common
modern state).

### (a) `fetch_cape` — macro_indicators.py:182-209

```python
# --- Shiller CAPE ---

def fetch_cape() -> float | None:
    """Fetch Shiller CAPE ratio estimate.

    Uses a simple approximation: SPY P/E * 1.6 adjustment factor
    since real-time Shiller CAPE APIs are unreliable.
    """
    cached = _get_cached('cape', _CAPE_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        spy = yf.Ticker('SPY')
        info = spy.info
        pe = info.get('trailingPE')
        if pe is not None:
            # CAPE is roughly 1.5-1.8x trailing PE historically
            cape_est = pe * 1.6
            _set_cached('cape', cape_est)
            logger.info("[MACRO] CAPE estimate: %.1f (PE=%.1f)", cape_est, pe)
            return cape_est
    except Exception as e:
        logger.debug("[MACRO] CAPE fetch error: %s", e)
    logger.warning("[MACRO] CAPE estimate unavailable (no SPY trailingPE) — "
                   "valuation rule blind")
    return None
```

### (b) cache TTL constant — macro_indicators.py:23

```python
_CAPE_CACHE_TTL = 86400     # 1 day
```

### (c) z-score constants + the 0.7x haircut — macro_indicators.py:301-303 and 355-360

```python
# Historical CAPE mean and std (approximate)
_CAPE_MEAN = 25.0
_CAPE_STD = 8.0
```

Inside `get_macro_regime` (fetch at :324, haircut at :355-360; docstring rule line
`CAPE z-score > 1.5 → reduce stock sizing 30%` at :315; `cape=cape,` passed into the
`MacroRegime` constructor at :387):

```python
    cape = fetch_cape() if asset_type == 'stock' else None
```

```python
    # CAPE (stocks only)
    if cape is not None:
        cape_z = (cape - _CAPE_MEAN) / _CAPE_STD
        if cape_z > 1.5:
            sizing_mult *= 0.7
            labels.append('overvalued')
```

### (d) v2 exclusion-announce machinery — macro_indicators.py:48, 75-76, 79, 89-93

Module global (`:48`):

```python
_cape_exclusion_logged = False
```

Inside `regime_family_mults_v2` — the `announce: bool = False` parameter, the docstring
lines *"Pseudo-CAPE is EXCLUDED by design (KILL_LIST; announce=True logs it once,
loudly)."*, the `global _cape_exclusion_logged` statement, and the announce block
(`:89-93`); the sole caller (`base_loop.py:2099-2101`) passed `announce=DERISK_STACK_V2`:

```python
        if announce and asset_type == 'stock' and not _cape_exclusion_logged:
            _cape_exclusion_logged = True
            logger.warning("[DERISK-V2] pseudo-CAPE multiplier EXCLUDED from "
                           "sizing composition (KILL_LIST item still live in "
                           "legacy path; code retained pending owner deletion)")
```

### (e) What was deliberately KEPT

- `types_mod.MacroRegime.cape` field — retained (not in the packet's file set; journal-row
  and fixture compat). `get_macro_regime` now always passes `cape=None`, with a comment.
- Module docstring / header lines referencing CAPE — rewritten to record the deletion.

**Rationale (measured/derived).** The "CAPE" was never CAPE: SPY trailing P/E × 1.6 is a
constant rescaling of a point-in-time trailing P/E — no 10-year real-earnings averaging, no
inflation adjustment; the z-score against a hardcoded mean 25/std 8 is fabricated by
construction. Three independent research sources killed it (rev-07-01 six-agent review,
econ-07, nobel-07). The ledger's synthesis: zero weight, no successor, needs no measurement.
It was the LAST live member of the legacy `macro_mult` composite that had no defensible
successor (VIX tiers → the one v2 map, STLFSI2 → survives, stablecoin → hard gate).

**Restoration instructions.** Do NOT restore this implementation under any evidence — the
input is fictional regardless of outcomes. A legitimate valuation influence would require:
(1) an owner decision removing "Pseudo-CAPE" from `research/KILL_LIST.md`; (2) a REAL
point-in-time Shiller CAPE series (e.g. Shiller's published data, publication-lagged for PIT
discipline); (3) shipping as a default-OFF flag through the measurement path (journaled skips
/ sizing detail) like every other model-facing influence. Mechanically, restoring the old
behavior would mean re-adding blocks (a)-(d) above and re-pointing the callers/tests listed
in the IA-1 packet report.

---

## IA-1.2 — Hurst<0.45 mean-reversion threshold shift (crypto entry funnel)

**Removed from:** `base_loop.py:2486-2497` (`_execute_buys`) — 2026-08-22, packet IA-1.
**Ledger verdict:** §3.3 Hurst threshold shift (×1.3) — **"NO — unanimous."** Direct quote:
*"Dead three ways: levels-not-returns input (never fires), absent from stock path,
journal-invisible. Delete the branch; regime conditioning, if wanted, ships as the corrected
FEATURE through retrain (gotcha #2), not a hand rule."*
**Classification:** direct-ship, provably-no-op — pinned by
`tests/test_ia1_removals.py::TestHurstBranchProvablyDead` (the proof exercises the live
INPUT construction, independent of the removed code, so it establishes the branch was
unreachable before the edit exactly as it documents unreachability after): live `Hurst` is
computed on price LEVELS (`indicator_config.HURST_ON_RETURNS = False` default;
`indicators.py:574-580`), and levels-mode R/S reads far above 0.45 on random walks, trends,
AND strongly mean-reverting price processes alike — the `hurst < 0.45` condition cannot fire
on the live input construction. Corroborated by
`tests/test_live_feature_parity.py::test_hurst_levels_mode_reads_high_on_random_walk` and
the 2026-07 panel finding recorded in `indicator_config.py:22-31`.

### Removed block — base_loop.py:2486-2497 (with surrounding pre-edit context)

```python
            # Prediction gate (higher bar if mean-reverting). The old
            # "recently hard-stopped -> 1.5x" bump was provably
            # unreachable: _is_hard_stop_locked either `continue`s above
            # or DELETES the expired key before returning False, so the
            # membership test below it was always False (2026-07 panel;
            # a REAL post-lockout elevated bar is an owner decision).
            effective_threshold = self.trade_threshold
            # Hurst < 0.45 = mean-reverting; momentum signals less reliable
            hurst = snapshot.get('Hurst')
            if hurst is not None and hurst < 0.45:
                effective_threshold = max(effective_threshold,
                                          self.trade_threshold * 1.3)
```

Downstream consumer (rewired, not removed) — `base_loop.py:2509`:

```python
            if pred_return < effective_threshold:
```

now reads `if pred_return < self.trade_threshold:` (the only value
`effective_threshold` could ever hold on live inputs).

**What was deliberately KEPT.** The `HURST_ON_RETURNS` feature-flag machinery in
`indicator_config.py` and the returns-mode computation in `indicators.py` — that is the
legitimate future form (corrected R/S input, model-facing, ships only with harvest+retrain
per CLAUDE.md gotcha #2). The `Hurst` FEATURE itself keeps flowing to the models unchanged.

**Rationale.** R/S analysis is defined on a process's increments; fed price LEVELS the
partial sums integrate the walk and H reads ~0.8 regardless of regime (2026-07 panel,
verified again by the packet pin tests: rolling-minimum levels-mode Hurst stays ≥ 0.45 for
random-walk, trending, GBM, and OU price constructions down to level lag-1 autocorrelation
~0.65; the measured reachability boundary is level persistence < ~0.5 — a price that loses
half its deviation from its mean EVERY HOUR, unattainable for a traded asset whose hourly
levels are near-integrated at ~0.99+; the boundary itself is pinned by
`test_unreachability_boundary_documented` so the proof cannot overclaim). The branch also
never existed in the stock funnel and produced no journal key — a dead influence that misled
readers of the funnel.

**Restoration instructions.** Do not restore the hand rule. If mean-reversion regime
conditioning is wanted: flip `HURST_ON_RETURNS = True` together with a fresh Jetson
harvest + retrain (delete `v2_study.db`/`stock_v2_study.db`, gotcha #2) so the corrected
feature reaches the MODELS; a renewed explicit threshold rule would additionally need
journaled skip rows (a `hurst_shift` key) and owner sign-off, since as a gate change it is
model-facing.

---

## IA-1.3 — Sentiment `gate <= 0` veto branch (crypto entry funnel)

**Removed from:** `base_loop.py:2558-2570` (`_execute_buys`) — 2026-08-22, packet IA-1.
**Ledger verdict:** §3.3 Sentiment gate (veto + multiplier) — KEEP-COND with the explicit
instruction: *"Veto branch mathematically unreachable (clamp floor 0.15) — delete the dead
limb."* Scope note from the packet: *"the [0.15,1.5] multiplier path stays exactly as-is"* —
`sentiment.sentiment_gate` itself is byte-unchanged except for an invariant comment at the
clamp.
**Classification:** direct-ship, provably-no-op — pinned by
`tests/test_ia1_removals.py::TestSentimentVetoUnreachable` (unreachability proven from the
clamp: `sentiment_gate` returns `max(0.15, min(1.5, m))` ≥ 0.15 > 0 for every input,
including catastrophic-news minimum 0.15×0.85 pre-clamp). The 2026-07 panel had already
verified the branch unreachable (the removed comment said so in the code).

### Removed block — base_loop.py:2558-2570

```python
            # Sentiment gate: multiplier only — sentiment.sentiment_gate
            # clamps to [0.15, 1.5], so this veto branch is currently
            # unreachable (verified 2026-07 panel). Kept as a defensive
            # guard; making sentiment a REAL veto is an owner decision.
            gate, gate_reasons = sentiment_gate(symbol, self.get_asset_type())
            if gate <= 0:
                vc['sentiment_block'] += 1
                self._journal_skip(symbol, 'sentiment_block',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot,
                                   sentiment_gate=gate,
                                   sentiment_reasons=gate_reasons)
                continue
```

The `gate, gate_reasons = sentiment_gate(...)` call itself SURVIVES (the multiplier feeds
`_compute_position_size(sentiment_mult=gate, ...)` and the buy journal row unchanged).

**What was deliberately KEPT / not touched.**
- `sentiment.sentiment_gate` — multiplier math byte-identical; only an invariant comment
  added at the clamp (return strictly positive; do not rebuild a veto without owner ruling).
- `stock_loop.py:964-969` carries the SAME unreachable `gate <= 0` branch — stock_loop is
  outside packet IA-1's file set (AGENT_CONTEXT ownership discipline), so it is REPORTED for
  the packet that owns stock_loop, not removed here.
- `decision_report.py:63` `GATE_REASONS` still lists `'sentiment_block'` — measurement-only
  reader of a key that has always had zero rows; harmless, left for the decision_report owner.
- `trade_journal.py:16` docstring still names `sentiment_block` among skip_reason values —
  doc-only, left for that file's owner (stale once stock_loop's branch is also removed).

**Rationale.** `max(0.15, min(1.5, multiplier))` bounds the return to [0.15, 1.5]; `gate <= 0`
is unsatisfiable, so the branch (its `vc['sentiment_block']` counter and `sentiment_block`
journal-skip row) could never execute — dead code whose presence made the funnel look like it
had a sentiment VETO when it only ever had a sentiment multiplier. Zero behavior change.

**Restoration instructions.** Nothing to restore — the branch never did anything. If the
owner ever decides sentiment SHOULD wield a real veto (an owner decision per the ledger; also
gated on the channel-budget question — ledger §3.3: one channel + at most one advisory read,
decided BEFORE fixing D27), implement it as an explicit threshold on the multiplier (e.g.
`gate <= SENTIMENT_VETO_FLOOR` with the floor > 0.15), default-OFF flag, journaled skip rows,
and llm_eval-style outcome attribution — not by resurrecting this unreachable comparison.

---

*(End of IA-1 sections. Later packets append below this line.)*

## IA-2.1 — Cooldown gating removed from EXIT paths (3 blocks)

**Removed from:** `base_loop.py` (2 blocks) + `stock_loop.py` (1 block) — 2026-08-22, packet IA-2.
**Ledger verdict:** §3.4 "Cooldown + trade budget — MERGE … **Remove cooldown gating from
EXITS** — an entry throttle delaying risk reduction inverts the tool's purpose
(base_loop.py:1625, 1673)." Also §3.7 signal exit KEEP-COND: "Un-gate from cooldown," and
§2 dimension 4: "cooldown also leaks into gating EXITS … an entry-economics argument applied
to holding vetoed risk."
**Classification:** direct-ship safety fix (ledger-prescribed, owner-ruled "implement their
suggestions"). NOT a no-op: a symbol inside its 60-min cooldown whose pred crosses
-threshold (or whose LLM veto persists 2 strikes) now SELLS immediately instead of holding
the risk until the cooldown expires. Entry-side cooldown is UNCHANGED (pinned). Every
bypassed exit journals a `cooldown_bypassed_exit: true` marker on its sell row
(measurement), so the bypass frequency is priceable from day one.

### (a) base_loop.py:1625-1626 — `_execute_sells` (signal exit)

```python
            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue
```

### (b) base_loop.py:1673-1675 — `_execute_llm_veto_sells`

```python
            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                logger.info("%s: LLM VETO (%.2f) but in cooldown", symbol, llm_s)
                continue
```

### (c) stock_loop.py:628-629 — `StockLoop._execute_sells` (signal + rank-drop exit)

```python
            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue
```

**Rationale.** The cooldown exists to throttle entry churn (fee bleed from signal jitter
re-trading one name). Applied to exits it did the opposite of risk management: a position
opened <60 min ago whose signal flipped hard negative, or whose LLM veto persisted two
consecutive analyses, was FORCED to stay open until an entry-economics timer expired. The
2026-07-02 review had already flagged the signal-exit asymmetry; the influence ledger made
the removal prescription explicit. Replacement in-tree: each site now computes
`cooldown_bypassed = not cooldown_ok(...)` and threads
`extra={'cooldown_bypassed_exit': True}` into `_record_confirmed_exit` when it fires.

**Restoration.** Re-add the `if not cooldown_ok(...): continue` guard after the
pred/sell_reason checks in each method. Justified only if Jetson journals show
`cooldown_bypassed_exit` sells are systematically WORSE than the counterfactual hold (i.e.
sub-hour signal flips are noise that round-trips fees) — that would be evidence for a
symmetric 2-reading confirm on the signal exit (the rev-07-02 reconciliation), NOT for
re-gating exits on an entry throttle.

---

## IA-2.2 — Shared (clobbering) hard-stop lockout state file

**Removed from:** `base_loop.py:120-127` (`__init__`) — 2026-08-22, packet IA-2.
**Ledger verdict:** §3.4 hard-stop lockout KEEP-COND: "shared unprefixed file lets books
clobber each other … per-book prefix" (old 2026-07 review P2 queue item, ledger-prescribed).
**Classification:** direct-ship safety fix. Crypto behavior byte-identical (keeps the legacy
unprefixed filename, mirroring `_position_state_file` naming); the stock book moves to
`stock_hard_stop_lockout.json` with a ONE-TIME migration read from the legacy file when the
prefixed file does not exist yet. Saves only ever write the book's own file.

### Replaced block — base_loop.py:120-127

```python
        self.hard_stop_lockout: dict[str, datetime.datetime] = {}
        # KNOWN (2026-07 review P2, deferred): this file is SHARED by both
        # books (unlike _position_state_file) and _save_hard_stop_lockout
        # rewrites it wholesale, so each book's save clobbers the other's
        # persisted lockouts. Per-book prefix fix queued for owner review.
        self._lockout_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'hard_stop_lockout.json')
        self._load_hard_stop_lockout()
```

(Also superseded: the `_save_hard_stop_lockout` comment "both books share the FINAL path
(known deferred P2) but must not share the TEMP path" — final paths are per-book now; the
per-book temp suffix stays because combined-bots mode runs both books in one process.)

**Rationale.** `_save_hard_stop_lockout` rewrites the file wholesale, so whichever book
saved last erased the other book's persisted lockouts across restarts — a 24h risk control
that silently evaporated whenever the other book tripped a stop. Migration read preserves
any live lockouts recorded under the legacy shared file at upgrade time (foreign-book
symbols in a migrated read are harmless: `_is_hard_stop_locked` only consults symbols in the
book's own universe).

**Restoration.** None conceivable — a shared mutable state file wholesale-rewritten by two
writers is a defect, not a design. The legacy `hard_stop_lockout.json` remains on disk as
the crypto book's file (and as the stock book's one-time migration source).

---

## IA-2.3 — Quote-staleness guard under alpaca_compat: NO REMOVAL (claim already fixed in tree)

**Ledger claim (census, §3.1):** "Staleness guard **inert under alpaca_compat** (timestamp
dropped)." **Tree verdict at implementation time: ALREADY FIXED** — `alpaca_compat._shim_quote`
(alpaca_compat.py:76-84) threads `t=getattr(q, 'timestamp', None)` with a docstring recording
the parity restoration (c26-era fix, shipped DIRECT as failure-path-safety parity). No code
was removed or changed by IA-2. Action taken instead: the repaired behavior is now PINNED by
`tests/test_ia2_safety.py::TestQuoteStalenessUnderShim` — a >180s-stale quote in the exact
shim shape is rejected by `order_utils.get_quote`, a fresh one accepted, and an absent
timestamp fails SAFE (accepted, exactly as the legacy no-timestamp path behaves today).

---

## IA-2.4 — Cycle-modulus macro-calendar staleness alarm (replaced by daily-warning + notify-once)

**Removed from:** `base_loop.py:2418-2420` (`_entries_allowed`) — 2026-08-22, packet IA-2.
**Ledger verdict:** §3.2 FOMC/CPI stand-down KEEP: "Needs a staleness alarm for the static
date table." The alarm IS the fix; gate behavior unchanged (fail-open).

### Replaced block — base_loop.py:2418-2420

```python
            if calendar_exhausted() and self.cycle % 2000 == 1:
                logger.warning("[MACRO] static FOMC/CPI table has no future "
                               "events — refresh macro_calendar.py")
```

**Rationale.** `cycle % 2000 == 1` fired roughly every ~33h of crypto cycles (interval-
dependent, never for a bot restarted before cycle 2001 wraps) and never reached the owner's
notification channel. Replacement: one loud `STALE CALENDAR` warning per calendar day plus
one `notify()` per process (Telegram/webhook), state in `_macro_cal_alarm_date` /
`_macro_cal_alarm_notified`; entries stay allowed (fail-open, unchanged).

**Restoration.** None — strictly-better observability for the same condition; the old
modulus line has no property worth restoring.

---

## IA-2.5 — stock_loop sentiment `gate <= 0` veto branch (unreachable twin of IA-1.3)

**Removed from:** `stock_loop.py:969-976` (`StockLoop._execute_buys`) — 2026-08-22, packet
IA-2 (explicitly deferred to the stock_loop-owning packet by IA-1's archive section).
**Ledger verdict:** §3.3 sentiment gate KEEP-COND: "Veto branch mathematically unreachable
(clamp floor 0.15) — delete the dead limb."
**Classification:** direct-ship, provably-no-op. Proof pinned by
`tests/test_ia1_removals.py::TestSentimentVetoUnreachable` — `sentiment.sentiment_gate`
returns `max(0.15, min(1.5, multiplier))` for BOTH asset types (score sweep), so
`gate <= 0` is unsatisfiable and the branch (its `vc['sentiment_block']` counter and skip
row) could never execute.

### Removed block — stock_loop.py:969-976

```python
            # Sentiment gate (veto first; multiplier folds into sizing tilt)
            gate, gate_reasons = sentiment_gate(symbol, 'stock')
            if gate <= 0:
                vc['sentiment_block'] += 1
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "pred_return": pred, "entry_rank": rank,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons})
                continue
```

The `gate, gate_reasons = sentiment_gate(symbol, 'stock')` call SURVIVES (multiplier feeds
`sentiment_mult=gate` into sizing and the buy journal row unchanged).

**Restoration.** Same as IA-1.3: nothing to restore — the branch never did anything. A real
sentiment veto would be an owner decision (channel-budget question first, ledger §3.3),
implemented as an explicit threshold above the 0.15 clamp floor behind a default-OFF flag
with journaled skip rows — not by resurrecting an unreachable comparison.

---

*(End of IA-2 sections. Later packets append below this line.)*

## IA-3 — Price the unpriced gates: NO CODE REMOVED (additive instrumentation only)

**Packet IA-3 (2026-08-22) removed nothing.** Recorded here so the packet trail through this
archive stays complete. The packet implemented the ledger's open-question #4 prescriptions
("the highest-impact gates are the least priced") as purely additive, measurement-only
journaling — every gate keeps its exact admit/veto behavior:

1. **VIX>35 halt** → `vix_halt` skip rows (pred/rank/vix context) at BOTH sites
   (base_loop stock-typed branch + the live stock_loop site); `vc['macro_halt']` key unchanged.
2. **VIX>25 non-safe-haven block** → `vix25_block` skip rows + a once-per-day log naming the
   SAFE_HAVEN-untradable gap (C3, `_log_vix25_gap_once`). Removal/repair of the block itself is
   IA-4's flag — untouched here.
3. **FOMC/CPI stand-down** → `macro_standdown` skip rows WITH window identity
   (`macro_calendar.standdown_window_id`, new journal hook; `macro_standdown` itself
   byte-equivalent, pinned), one row per (window, symbol), threshold-crossing candidates only.
4. **RR_5 high-VIX tiebreak demotion** → one `rr5_demotion` event row per firing (every demoted
   name, pre/post rank, `admission_lost` counterfactual flag) + a priced skip row per name
   pushed out of the top-N ("journal every demotion ... within one release or delete it" —
   this was the release).
5. **Top-N ranks 8–15** → `rank_near_miss` skip rows (rank, pred, `would_be_base_notional`
   pre-tilt proxy from snapshot Close/ATR — no quote fetch/GARCH on the Jetson), once per
   (day, symbol), only on cycles where the entry funnel actually runs.
6. **Circuit-breaker trips** → `circuit_breaker_trip` event row (book, `account_last_equity`
   baseline actually used, drawdown, weekend flag, positions at trip, halted_until) — evidence
   substrate for the per-book-baseline decision (IA-4); never blocks the flatten.

**Cross-file follow-up (CLOSED by the IA-3 hardener, same day):** the five new skip_reason
names (`vix_halt`, `vix25_block`, `macro_standdown`, `rr5_demotion`, `rank_near_miss`) were
added to `decision_report.py` `GATE_REASONS` so the rows are replay-priced (the ledger's open
question #4 names that list as the pricing instrument — leaving the names in
`_unclassified_skip_reasons` would have defeated the packet). The loops' vc counter keys
`macro_halt`/`vix_block` stay unchanged in `UNPRICED_GATES` (entry_window veto_counts
back-compat); membership is pinned by `tests/test_ia3_gate_pricing.py::TestGateReasonsWired`.

---

*(End of IA-3 section. Later packets append below this line.)*

## IA-4 — Flag-gated structural family: NO CODE REMOVED (gated bypasses only)

**Packet IA-4 (2026-08-22) deleted nothing.** Recorded here so the packet trail through this
archive stays complete. The packet implemented the ledger's seven split-verdict /
behavior-loosening prescriptions as default-OFF flags in `strategy_config.py` (flag-OFF
byte-identical, pinned by `tests/test_ia4_flagged.py`; every flag's comment carries its ledger
verdict + flip criterion + deciding instrument):

1. **VIX25_BLOCK_REMOVED** — ON skips the VIX>25 non-safe-haven block at both sites
   (base_loop stock-typed branch + stock_loop live site). The block's CODE remains in place
   behind the flag guard — nothing to archive; restoration = flip the flag back OFF.
2. **CORR_FAMILY_MERGED** (+ `CORR_SANITY_MAX = 0.85`) — ON: the binary >0.7 correlation
   admission gate loosens to the 0.85 sanity bar and the f_corr sizing haircut composes at
   1.0 (legacy AND v2 read it as 1.0); the ENB stop-risk budget is the single correlation
   consumer. All bypassed code remains live behind the flag.
3. **TRADE_BUDGET_BACKSTOP** (+ mult 3) — ON: `_trade_budget_ok`'s cap becomes a runaway
   backstop; the budget machinery itself is untouched.
4. **CRYPTO_VERTICAL_BARRIER** — additive: `base_loop._check_vertical_barrier` (loop-layer
   position-age check, fb-anchored from the model config), entry-ts tracking persisted in
   the position state file, would-fire journaled ALWAYS (`action='vertical_barrier'`,
   `fired` records the flag). `policy_exits.py` UNTOUCHED (pinned: no import/call of the
   kernel anywhere in base_loop). Cross-file note for the decision_report owner:
   `'vertical'` is a NEW exit_reason value on sell rows once the flag flips — exit-reason
   vocabularies (decision_report / journal_stats) may want it enumerated.
5. **SIGNAL_EXIT_CONFIRM_READS** (default 1) — additive arming state + `signal_exit_reading`
   armed/lapsed journal rows at setting 2; at the default the sell row itself remains the
   first-reading record (byte-identical, pinned — including the stock rank-drop exit, which
   is never deferred).
6. **KELLY_SAMPLE_GATE** (+ `KELLY_SAMPLE_MIN_TRADES=50`, `KELLY_SAMPLE_SINCE='2026-08-22'`)
   — additive `trading_utils.uncensored_trade_count` + a neutral-hold on kelly_mult with the
   gate state journaled in the sizing detail. NOTE for the owner: set `KELLY_SAMPLE_SINCE`
   to the actual Jetson deploy date of the D06-fix wave when flipping.
7. **BREAKER_PER_BOOK** — additive `_book_breaker_check` / `_breaker_window_end` /
   `_breaker_note_realized` + an always-on mark stash in `_manage_stops`; OFF the breaker
   calls `order_utils.check_circuit_breaker` account-wide exactly as today (pinned). Trip
   rows under the flag journal `baseline_kind='book_window_v2'`.

**One DIRECT (unflagged) behavior change, ledger-prescribed input fix — no code removed:**
the ENB covered-none rho bypass (ledger §3.3: "Fix D29 equity denominator + the covered-none
rho=0.0 prior bypass"). `portfolio.avg_book_correlation` gained an `uncovered` parameter
(default None = legacy 0.0, pinned by test_portfolio_v3); base_loop's ENB budget site and the
GATE-1 `_record_account_risk` journal site pass `uncovered=0.5`, so a non-empty correlation
matrix covering NONE of the book's pairs now yields the same 0.5 no-data prior an absent
matrix does, instead of rho=0.0 (perfect diversification from zero evidence — the loosest
possible budget). Strictly risk-tightening. The D29 deposit-outlier exclusion half of that
ledger line was verified ALREADY in tree (portfolio `_ewma_vol_diag(exclude_outliers=...)`
wired to DERISK_STACK_V2) — not duplicated.

---

*(End of IA-4 section. Later packets append below this line.)*


---

## 2026-09-26 — Jetson test & improvement campaign removals

Removals made during the 2026-09-26/27 Jetson campaign (hunt → implement), archived verbatim per the delete-nothing rule.

### G5-8 — `volatility.get_garch_stop` + its two tests (removed 2026-09-27)

**Why removed.** Zero production callers. Stops are ATR-based (`base_loop._desired_stop_for` /
`_manage_stops`, `stock_loop._manage_stops`), and this floor/ceil did not track the
`strategy_config` stop policy. The 2026-07 review deferred the removal only because
`base_loop` still imported the name then ("must be done together with a base_loop-owning
change"). That blocker is gone: `base_loop` no longer imports it, and
`tests/test_grp_loops.py::test_dead_imports_pruned` asserts that. The function already
carried a "DEAD" comment.

**Zero-caller proof (2026-09-27, Jetson tree).** `grep -rn get_garch_stop --include=*.py .`
before the removal found these hits only:
- `volatility.py:107`, the definition.
- `tests/test_new_modules.py:96-105`, the two tests below.
- `tests/test_grp_loops.py:22`, which asserts the name is ABSENT from base_loop.
- `archive/jetson_residue_2026-03/manual_expanded.py:658,945`, manual prose (untracked residue).

It has no importer in any root module or `scripts/`, and `docs/graphs/import_graph.json` lists
`base_loop` (which imports `compute_vol_adjusted_size` + `get_sigma`) as the only non-test importer
of `volatility`.

**Re-adding.** It is a pure function with no state. Restore both blocks verbatim if a GARCH-based
stop is ever wanted. It would be model-/policy-facing, so it would need the challenger → shadow
path and a `strategy_config` stop-policy tie-in.

#### Removed block — volatility.py:102-122 (pre-removal numbering; HEAD `438f56a` :107 def)

```python
# DEAD in live paths: stops are ATR-based (base_loop) and this floor/ceil
# does not track strategy_config stop policy. It has NO non-test consumer —
# base_loop no longer imports it (tests/test_grp_loops.py::test_dead_imports_pruned
# asserts the name is absent from base_loop); the only exercise is
# tests/test_new_modules.py.
def get_garch_stop(entry_price: float, sigma: float, multiplier: float = 2.0,
                   floor_pct: float = 0.03, ceil_pct: float = 0.10) -> float:
    """Compute stop-loss price using GARCH volatility.

    Args:
        entry_price: Entry price
        sigma: GARCH sigma (decimal, e.g. 0.02 = 2%)
        multiplier: Number of sigmas for stop distance
        floor_pct: Minimum stop distance as fraction of price
        ceil_pct: Maximum stop distance as fraction of price

    Returns:
        Stop price (below entry for long positions).
    """
    stop_dist = max(floor_pct, min(ceil_pct, sigma * multiplier))
    return entry_price * (1 - stop_dist)
```

#### Retired tests — tests/test_new_modules.py:96-105 (`TestVolatility`)

```python
    def test_get_garch_stop(self):
        from volatility import get_garch_stop
        stop = get_garch_stop(100.0, 0.05, multiplier=2.0)
        assert stop < 100.0
        assert stop > 80.0  # not more than 20% away

    def test_get_garch_stop_floor(self):
        from volatility import get_garch_stop
        # Very low sigma should be floored
        stop = get_garch_stop(100.0, 0.001, multiplier=2.0, floor_pct=0.03)
        assert stop == pytest.approx(97.0, abs=0.01)
```
