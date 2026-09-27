# G2 — execution & broker hunt (order_utils, order_stream, alpaca_compat, trading_utils, execution_policy)

Hunt date: 2026-09-26/27, on the Jetson. This was read-only: no repo file was edited. Every proposed fix was
applied only to **sandbox copies** under `scratchpad/hunt/G2/sandbox*/`, and the affected test files were run
against those copies one file at a time. The only live calls were read-only Alpaca calls (quotes, assets,
positions, account, and GET of a bogus order id). No order was placed.

Artifacts are in `scratchpad/hunt/G2/`:
- repro scripts: `repro_*.py`, with outputs in `*.out`
- proposed diffs: `G2-1.diff`, `G2-1_testfix.diff`, `G2-2.diff`, `G2-3.diff`
- patched copies: `sandbox/` (G2-1 + G2-3) and `sandbox2/` (G2-2)
- unpatched control: `sandbox_orig/`

I checked the already-fixed items first. None of the FIX_* reports from today touches these five files. HARNESS.md already
reports `avg_entry_price=0` flowing through `reconstruct_positions`, so it is not re-reported here.

---

## Findings (ranked by severity)

### G2-1 · order_utils.py:708-799 (manage_order_lifecycle) + :446-485 (place_maker_buy) · class A (race / double order) · HIGH
**Defect.** Both functions send a second order for the same intent while the first order is confirmed still working.
- `manage_order_lifecycle` submits the market fallback without checking that the timeout cancel took effect. The
  cancel can raise (logged "Cancel error" and ignored), or the order can still be `pending_cancel`. In both cases the
  post-cancel fetch returns a live order (`new`, `partially_filled` or `pending_cancel`), and the fallback is placed
  anyway.
- `place_maker_buy` then treats that still-working rung as a zero-fill. It posts the next rung **and** the taker
  fallback on top of it.

**Proof.** `hunt/G2/repro_double_order.py` (fake broker, no network), output in `repro_double_order.out`:
```
[lifecycle cancel=raise] original status=new (still working) + market fallback submitted qty=[1.0] -> ... exposure if limit fills = 2.0
[maker ladder cancel=raise] tactic=taker_fallback submitted=[('limit', 10.0), ('limit', 10.0), ('limit', 9.99000999), ('market', 9.99000999)] still-working buy orders=3 notional working/filled=3998 vs intended 1000
[lifecycle cancel=pending_cancel] ... market fallback submitted qty=[1.0] -> exposure if limit fills = 2.0
[maker ladder cancel=pending_cancel] ... still-working buy orders=3 notional ... =3998 vs intended 1000
```
A $1,000 crypto entry ends up as **~$4,000 of live buy orders**: three working GTC bids plus a filled market order.
This is the same exposure multiplication that D18 fixed for `result is None`. The D18 fix covers only the
*unknown* state. It does not cover the state that is *known to be live*.

**Correct output.** Once the order is confirmed still working, no fallback and no next rung should be sent.
- The lifecycle returns the fetched order, and the caller judges acquisition by `filled_qty`, as it already does.
- The ladder aborts as `maker_unknown`, which is the same handling D18 already adopted.

**Fix** (`G2-1.diff`, +27 lines):
1. Add a module constant `_SETTLED_STATUSES = ('filled','canceled','expired','rejected')`.
2. In `manage_order_lifecycle`, just before `if fallback_to_market and saved_symbol and cancel_on_timeout:`, add this:
   ```python
   if (fallback_to_market and cancel_on_timeout and final_order is not None
           and getattr(final_order, 'status', None) not in _SETTLED_STATUSES):
       logger.error("[LIFECYCLE] %s: cancel not confirmed (status=%s) — skipping fallback", ...)
       return final_order
   ```
3. In `place_maker_buy`, right after the existing `if result is None:` abort, add this:
   ```python
   if getattr(result, 'status', None) not in _SETTLED_STATUSES:
       <log>; if _filled_qty(result) >= _filled_qty(last): last = result
       _journal_entry_fills(symbol, 'maker_unknown', maker_notional, taker_notional)
       return last, 'maker_unknown'
   ```
With the patch, `repro_double_order_patched.out` shows 0 fallbacks, one working order and `maker_unknown`.

**Blast radius.**
- **Callers.**
  - `base_loop._execute_entry_order`: a non-settled final order is judged by `filled_qty`, then `verify_position`, so
    tracking is unchanged.
  - `crypto_loop.place_sell_order`, `stock_loop.place_sell_order` and the `stock_loop` flatten at :384-392: these
    now get the unsettled order back, treat it as a failed sell and retry next cycle. Before, they sent a second sell
    that the broker would reject as `insufficient qty`, because the working limit holds the qty.
  - Stock bracket entries (:1268) are unaffected, because they call with `fallback_to_market=False`.
- **Tests.** I ran all 10 order-path test files against the patched copy.
  - 9 of the 10 files are unchanged: c26_P1 44, c26_X1 25, fault_injection 13, grp_exec 7, ioc_helper 9,
    order_utils 16, order_utils_v3 40, review_b02 27, all passed.
  - `tests/test_c26_T6.py` has **3 failures**. Its fake `_API.cancel_order` (:217) records the cancel but never
    changes status, so the order stays `new`. That fake is exactly the double-order scenario above.
  - Making the fake realistic (`G2-1_testfix.diff`: on cancel, set status to `canceled` unless already filled) gives
    43/43 on **both** the patched and the unpatched module. So the test intent (flag-OFF market fallback, flag-ON
    capped IOC) is preserved.

**Why indisputable.** A replacement order placed while the original is confirmed still working is, by definition, a
second live order for the same intent. The repo already made this rule for the unknown case (D18, "Unknown is NOT
zero-fill").

---

### G2-2 · order_utils.py:222/224 (compute_limit_price), :439 (maker rung), :233 (_round_price_band crypto) · class A (wrong output) · MEDIUM
**Defect.** Crypto limit prices are rounded to 4 dp (and to 2 dp in `_round_price_band` for coins ≥ $1). Alpaca's
`price_increment` is **1e-9 for all six universe coins**, checked live via `GET /v2/assets` (DOGE, XRP, LINK, SOL,
ETH, BTC). On DOGE (~$0.0955), one 4-dp step is ~10.5 bps, which swamps every designed offset:
- The maker rung that logs "joining bid @ $0.0954636" actually posts **0.0955**, which is +3.81 bps in front of the
  bid.
- `compute_limit_price` gives **buy == sell == 0.0956**. The intended ±2.86 bps spread-aware offset is erased.
- The flag-gated capped-IOC price (`ioc_limit_price(..., 'crypto')`) on XRP at ask 1.51 with a 40 bps cap posts
  **1.52 = +66.2 bps**. That breaks the cap the function exists to enforce. With a 20 bps cap it collapses to exactly
  the ask.

**Proof.** `hunt/G2/repro_crypto_rounding.py` uses the live DOGE quote from 23:42 CDT (bid 0.0954636, ask 0.095737).
Output in `repro_crypto_rounding.out`:
```
maker rung 'join the bid': bid=0.0954636 posted=0.0955  (+3.81 bps vs bid)
compute_limit_price: intended buy=0.0956276 sell=0.0955730 (+/-2.86 bps); posted buy=0.0956 sell=0.0956 ... buy == sell: True
XRP capped IOC buy cap=40bps: want <= 1.516040 (40 bps over ask), posted 1.52 (+66.2 bps over ask)
```
With the patch, `repro_crypto_rounding_patched.out` gives: rung at exactly the bid, buy/sell at ±2.86 bps, and IOC at
exactly +20.0 and +40.0 bps.

**Fix** (`G2-2.diff`):
- `_round_price_band` crypto branch becomes `round(px, 9)`.
- The maker rung uses `limit_price=_round_price_band(bid, 'crypto')`.
- `compute_limit_price` rounds to 9 dp instead of 4. Stock callers still re-round to 2 dp in
  `place_stock_limit_order`, and that re-round removes the double-rounding (4 dp then 2 dp) at half-cent edges.
- The sibling outside G2 is `crypto_loop.py:116` (sell limit `round(mid*0.9995, 4)`): it gets the same change from
  its owner.

**Blast radius.**
- **Tests.** All passed except one: c26_P1 44, c26_T6 43, c26_X1 25, execution_policy_v3 33, fault_injection 13,
  grp_exec 7, ioc_helper 9, order_utils_v3 40, review_b02 27.
  `tests/test_order_utils.py::TestComputeLimitPrice::test_rounds_to_four_decimals` pins the 4-dp rounding. Update it
  to 9 dp.
- **Callers.** Crypto entries (maker ladder and base marketable path), `crypto_loop` exits, and the flag-OFF IOC
  helper.
- **Limit.** I verified acceptance of 9-dp prices only through asset metadata, because no order may be placed.
  Confirm with one paper order before shipping.

**Why indisputable.** The code states its own intent: "join the bid", a spread-aware offset, and a slippage *cap*. For
DOGE and XRP the rounding produces a different price, and the venue's own increment shows the rounding is
unnecessary. The parts that are mechanical are the rung price and the IOC cap violation. The `compute_limit_price`
part changes a test-pinned value, so the adjudicator may choose to split it off.

---

### G2-3 · trading_utils.py:153 (cooldown_ok) · class A (wrong timezone arithmetic) · LOW (crypto only, 2 hours a year)
**Defect.** `elapsed = (datetime.datetime.now() - last_trade_time[symbol])` subtracts **naive local** datetimes. The
Jetson is on America/Chicago (CDT/CST), and every writer stamps `datetime.datetime.now()`. So across a DST
transition the elapsed time is off by exactly 60 minutes:
- At spring-forward, a 60-minute crypto cooldown (`CRYPTO_POLICY.cooldown_min=60`) is bypassed after **15 real
  minutes**.
- At fall-back, a cooldown that has already expired stays active for up to another hour.

**Proof.** `hunt/G2/repro_cooldown_dst.py` (run with `TZ=America/Chicago`), output in `repro_cooldown_dst.out`:
```
spring-forward 2027-03-14: true elapsed=15 min, .timestamp() diff=15 min, naive diff used by cooldown_ok=75 min -> cooldown_ok(30m)=True (correct False), cooldown_ok(60m)=True (correct False)
fall-back 2026-11-01: true elapsed=40 min, ... naive diff=-20 min -> cooldown_ok(30m)=False (correct True)
```

**Fix** (`G2-3.diff`, one expression):
```python
elapsed = (datetime.datetime.now().timestamp() - last_trade_time[symbol].timestamp())
```
Why this is exact:
- `datetime.now()` and `datetime.fromtimestamp()` (the restore path at base_loop.py:560) both set `.fold` for the
  repeated hour, and `.timestamp()` honours it. The repro prints fold=1.
- It also works for aware datetimes.

With the patch, `repro_cooldown_dst_patched.out` shows both cases correct.

**Blast radius.**
- `tests/test_trading_utils.py` 9/9 and `tests/test_new_modules.py` 38/38 pass against the patched copy.
- Callers: base_loop:1956/2014/2965 and stock_loop:791/1030. Stocks never trade in the 01:00-03:00 window, so only the
  crypto book is affected.
- Siblings with the same bug, outside G2: `base_loop.py:2762` (`hard_stop_lockout`, 24 h) and `stock_loop.py:929`.

**Why indisputable.** Elapsed time must not be computed from naive wall-clock subtraction in a DST zone. The fix only
restores the true duration.

---

### G2-4 (cross-group: stock_loop.py — root is the diverged copy of order_utils._is_not_found) · stock_loop.py:352, :722 · class A + B · HIGH for the stock book
**Defect.** `stock_loop` hand-copies the "position is gone" classifier as `'not found' in s or '404' in s or 'no
position' in s`. The legacy SDK's real error for an unheld position stringifies to **`'position does not exist'`**.
It carries `status_code=404` as an attribute, but "404" is not in the text. So the copied classifier returns False,
while the shared `order_utils._is_not_found` (which includes `'position does not exist'`) returns True. The effects:
- In `_execute_sells` (:722), a stock position closed at the broker (for example, the bracket TP leg filled) is
  **never** detected as gone. The loop just `continue`s every cycle.
- `_journal_external_close` never runs. This censors the TP wins that the code comment says this path exists to
  capture.
- In the EOD flatten (:352), the same case is logged as a transient "get_position failed — will retry" forever
  instead of being popped.

**Proof.** A live read-only probe, `hunt/G2/probe_notfound_live.py`, output in `probe_notfound_live.out`:
```
get_position('AAPL') -> APIError('position does not exist'), status_code=404; stock_loop:722/352 classifier=False  order_utils._is_not_found=True
get_position('AVAXUSD') -> APIError('position does not exist'), status_code=404; ... classifier=False  order_utils._is_not_found=True
get_order(bogus) -> 'order not found for 0000...'; stock_loop:1415 classifier=True   (that third copy is fine)
```

**Fix.** At :352 and :722, replace the inline string test with `from order_utils import _is_not_found` and
`_is_not_found(e)`. `_is_not_found` is a strict superset of the inline checks, so every string they matched still
matches. :1415 (order-not-found) may stay as is.

**Blast radius.** No test pins these strings (grep of tests/ found none). This is in the owner group of `stock_loop.py`,
not G2, so it is handed over here.

**Why indisputable.** The live broker error provably defeats the copied classifier, while the shared classifier it was
copied from handles it.

---

### G2-5 · trading_utils.py:314-332 (kelly_position_size) · class B (dead code) · LOW (already recorded)
**Defect.** Zero production callers. `grep -rn "kelly_position_size("` over `*.py` outside archive/ finds only its
definition and `tests/test_new_modules.py:275-278`. docs/MODULES.md:522 already lists it as "OBJECTIVE-fix-pending
(dead code, test-pinned presence)".

**Fix.** Move it verbatim to `research/campaign_2026-08/08_removed_code.md` (delete-nothing convention) and drop
`test_kelly_position_size_default`.

**Blast radius.** One test.

**Why indisputable.** No caller, and the repo's own module census already agrees.

---

## Judgment calls (not proposed)
1. **The 180 s quote-staleness rule rejects valid quotes from quiet crypto books.**
   - Live at 23:42 CDT: the DOGE/USD latest quote was 212–225 s old on every poll (`bp 0.0954636 / ap 0.095737`,
     unchanged). `get_quote` returned None, so the held ~$11k DOGE position (114,913 DOGE) had no quote for stop
     evaluation during that window.
   - Alpaca publishes crypto quotes on change, so an old `t` does not mean a frozen feed. Changing this is a threshold
     decision.
2. **Crossed quotes (`ask < bid`) are logged but returned with a negative `spread_pct`.**
   - That lowers `should_trade`'s cost threshold, and a maker "bid-join" at bid ≥ ask becomes a taker fill that is
     journaled as `maker` (it feeds `fees.realized_crypto_maker_share`).
   - `choose_entry_tactic` already treats a crossed quote as missing. Whether `get_quote` should reject it changes gate
     inputs, so it is an owner decision.
3. **Market orders are still canceled at timeout on the stop-exit path** (`base_loop.py:1469`), for `crypto_loop`
   market sells (:133), and for the stock flatten (market order plus a market "fallback", :384-392).
   - The `manage_order_lifecycle` docstring (:616-617) says confirm-only mode is for "emergency flatten, stop exits",
     but only `emergency_flatten` passes `cancel_on_timeout=False`.
   - The D19 design ("never cancel market orders at timeout", 02_research B08) is still deferred_owner.
4. **D18 residual: an *ambiguous* submit exception on a maker rung** (for example, a read timeout after the POST
   landed).
   - It `break`s into the taker fallback with a fresh `make_client_order_id`, so a landed rung plus the fallback can
     double the entry.
   - The fix (query by client_order_id before retrying) is the deferred idempotent-submit design in 02_research B08.
5. **`manage_order_lifecycle` when the post-cancel fetch fails** (`final_order is None`).
   - It still chases using the last in-loop `saved_filled`. Fills between that poll and the cancel are not counted, so
     it can over-buy by that amount.
   - Returning None instead would leave any partial fill untracked until restart. That is a trade-off, so G2-1 was
     scoped to the case where the order is known to be live.
6. **`check_circuit_breaker` returns `(False, 0.0)` when `last_equity <= 0`** (:1198-1202). Its own docstring says
   unknown state should be `None`, which makes callers fail closed. Pinned by
   `test_order_utils_v3.py::test_zero_last_equity_warns_but_returns_pinned_value`.
7. **`verify_position` iterates a `set`, whose order depends on the hash seed.**
   - Under the legacy SDK, `get_position('BTC/USD')` always 404s (measured 0.29 s, versus 0.08 s for `BTCUSD`).
   - So about half of processes pay a wasted round trip on every crypto verify, and the same in the
     `reconstruct_positions` probe fallback.
   - Trying the broker spelling first would fix it. The latency is minor and not memory, so it does not meet class C.
8. **`trading_utils.ORDER_TIMEOUT`** has zero production readers but is deliberately pinned by
   `test_new_modules.py:284` and `test_review_b03.py:85`.
9. **Bar timestamps differ between SDKs.** alpaca-py bars carry UTC-aware `timestamp`, while the legacy SDK carries
   `pd.Timestamp` in America/New_York (see FIX_R5). Both are tz-aware and consumers convert, so no wrong output was
   found. This is noted only for the adapter switch.
10. **alpaca_compat could not be executed here** (alpaca-py is not installed on the Jetson).
    - `_shim_account` does `float()` on fields alpaca-py types as `Optional[str]` (`portfolio_value`, `last_equity`,
      `cash`). A None would make every `get_account()` raise. The breaker would then return `(False, None)` and buys
      would stay suspended.
    - Plausible, but not provable on this box.

## No issues found (coverage)
- **order_utils.py:**
  - `make_client_order_id`, `_symbol_variants`, `_filled_qty`, `install_session_timeout` (verified live: the legacy
    `api._session` is wrapped, `_trader_timeout_wrapped=True`).
  - `get_quote` NaN/≤0/stale checks. The live AAPL after-hours `ap=0` was correctly rejected. The legacy `QuoteV2.t`
    parses to a tz-aware `pd.Timestamp`, so the staleness check is active under the legacy SDK.
  - `_bid_ask`, `_entry_ioc_cap_bps`, `_ioc_entry_fallback` (flag-gated; qty parse is guarded),
    `_journal_entry_fills`, `place_stock_limit_order`, `get_all_positions`.
  - `realized_crypto_maker_share_notional` (flag OFF, cached), `should_trade` (model-facing, not reviewed for
    thresholds).
  - `_list_open_orders` / `_await_orders_clear` / `cancel_*` (limit 500, server hint with an unfiltered fallback).
  - `reconstruct_positions`: the list-first path and the `current_price=None` guard. `avg_entry_price="0"` is already
    in HARNESS.
  - `check_circuit_breaker`: the error path returns `(False, None)` and `base_loop:673` fails closed. Checked live:
    `(False, 0.0088)`.
  - `emergency_flatten`: 2-phase, confirm-only. The `'<list_positions failed>'` sentinel is emitted only at :1242 and
    matched literally at base_loop:751/959/2826, and all three keep every position tracked on it. Consistent.
  - Retry arithmetic: the lifecycle 3-error give-up and the stream-skip-once pacing are correct.
  - Legacy SDK retry: 429/504 retried 3×3 s. With the (10 s, 30 s) session timeout every call is bounded.
- **order_stream.py:** `_next_backoff` (5 s doubling, capped at 300 s, reset after 60 s healthy), `_record` bounded
  eviction, `start_order_stream` start lock and fail-closed on unset base URL, `_on_update` parse guard. `TERMINAL`
  equals the lifecycle's terminal set.
- **alpaca_compat.py:**
  - Every broker attribute the production code reads (census of `.attr` and `getattr(...)` over non-test, non-archive
    `*.py`) exists on the shims.
  - Every `api.<method>` called in production has an adapter method, except the Finnhub-client calls, which are not
    Alpaca.
  - Enum `.value` normalisation for status and side. The `list_orders` `after`/`until` parse. Portfolio-history index
    alignment of equity/timestamp.
- **trading_utils.py:** `get_api` (credential and base-URL warnings), `_install_rest_timeouts`, `model_reload_key`,
  `choose_inference_device`, `predict_symbol`.
  - `compute_kelly_fraction` and `uncensored_trade_count`: the real `trade_memory.json` (123 records, 35 symbols) has
    no None/NaN `pnl_pct` and no non-string `ts`.
- **execution_policy.py:** a pure function, unwired as documented. The NaN/inf/crossed/unknown-class paths are
  correct. test_execution_policy_v3 passes 33/33.
