# ENGINE W21 (R7): #22 phantom-exit audit (dry-run) + X12 paper-quirk census. Read-only, no orders.
Both CLIs use stdlib only; network access is lazy, stdlib-urllib GET-only, and paper-host asserted. `--replay` works offline. The only write path is a sink that refuses trade_memory.json/*position_state.json. trade_memory.json is unchanged (sha256 87d23f21…, mtime 05-03). Raw outputs are in w21/.
## LANDED (new files only)
- scripts/trade_memory_phantom_audit.py: owner #22. It has NO write mode. Each exit row is matched to a broker order on the same symbol and exit side (sell, or buy for `cover`), one order per row, closest first, within ±15 min, price within 5 %, and qty within 2 % when a row carries qty (no row does today).
  - MATCHED / PHANTOM (no same-side fill within ±24 h) / AMBIGUOUS (between the windows, fill taken by a closer row, price off, or before the history start).
  - It also prints the Kelly mirror before and after the repair, plus the proposed repair JSON: row ids `sym|ts|k`, patch estimated=True, and the audited file's sha256. The repair is never applied.
- scripts/paper_quirk_census.py: X12. One snapshot per run gives one jsonl row with detectors V/B(+B_new)/Q/E/A(+#23 table)/R. `--alert-rules` prints the rule and `--evaluate DIR` applies it offline. It never schedules or notifies (so the X12 notify.notify is deliberately dropped).
- tests/test_trade_memory_phantom_audit.py (38): window boundaries (15/15.5/1440/1441 min), side and symbol normalisation, 1:1 matching, partial fills, tolerances, fallback to the order record, repair scope and no mutation, CLI replay (socket blocked, file bytes and mtime unchanged).
  - Kelly mirror == the real `trading_utils.compute_kelly_fraction`: 12 cases, _TRADE_MEMORY_FILE monkeypatched. KELLY_CAP is pinned to strategy_config.
  - AST read-only pin on BOTH scripts: only method='GET', no submit/cancel/replace/close/post/delete/os.replace, and every write-mode open() sits inside `_write_file`. Mutation-checked: an injected open-w, POST, submit_order and os.replace are all caught.
- tests/test_paper_quirk_census.py (13): the detectors (string "0" basis, V carried/sold/state, Q net of FILL and in-kind CFEE with the 1e-6 tolerance, E, A table, R) and every rule branch (adjacency, standing B, Q $50 single, 30-day verdict), plus CLI replay/--last/--out/--evaluate.
- scripts/README.md: 2 inventory rows. Header count is now `34 .py`, the actual row count, because another worker added `meta_calib_nested.py` concurrently. `repo_graph.py` still has no row (pre-existing).
## REAL RUN 1 — phantom audit (`--fetch`, 878 broker orders, coverage from 2026-01-18)
| reason | n | MATCHED | PHANTOM | AMBIG |  | Kelly (mirror) | now | after repair |
|---|---|---|---|---|---|---|---|---|
| broker_stop | 68 | 5 | **63** | 0 |  | crypto sample | 73 (63 phantom) | 10 → Kelly None |
| desync | 12 | 1 | 3 | 8 |  | crypto f / mult | **0.1995 → 1.50x** | None → **1.00x** |
| hard_stop | 15 | 12 | 0 | 3 |  | stock | 0.05 / 0.5x | unchanged |
| trailing / TP / short_cover | 23/2/3 | all | 0 | 0 |  | | | |
- **63 CONFIRMED.** Phantoms run 2026-04-05 → 05-03: LINK 12, SOL 11, BTC/ETH/DOGE/XRP 10 each. The nearest same-side fill is ≥ 7,707 min (5.35 d) away, so the count is 63 at every match window from 5 to 240 min.
  - `estimated` is set on 0 of them. The key is absent on all 123 rows (legacy schema), so `t.get('estimated')` treats them as not estimated and Kelly counts them (trading_utils.py:239).
  - Phantom pnl mean +2.33 % (47 wins / 16 losses). Ledger-implied notional ≈ $697k (approximate: ignores CFEE and broker drift).
  - The 5 matched broker_stop rows are 4 on 03-31 (+9.2 min) and COIN on 04-02.
- **Material consequence (OBJECTIVE, mirror verified against the real function):** the phantoms alone switch crypto Kelly on. Mirroring base_loop.py:2587-2591, crypto Kelly is 0.1995, giving a **1.5x crypto sizing multiplier** while KELLY_SAMPLE_GATE=False (strategy_config.py:724). After the repair only 10 real rows remain (< min 50), so Kelly returns None and the multiplier is 1.0x.
- Repair = 63 ids in w21/repair_real.json.
## REAL RUN 2 — census snapshot 2026-09-27T14:05Z (w21/census/rows.jsonl)
- Fired: B 6/6 (avg_entry_price "0", cost_basis 0; standing, B_new none) and A 6/6. V none (position_state tracks the same 6). Q n/a (first row). E false: equity 123,296.27, cash 93.63, gap/long-mv = 1.0. R none: qty_available 0 on all 6, fully reserved by 6 full-qty stop_limits.
- **#23 asset-id table.** For all six, the position asset_id ≠ the resting stop's asset_id, and the stop's id == the current /v2/assets id:
  - BTC 64bbff51 vs 276e2673
  - DOGE a3ba8ac0 vs 03e005f7
  - ETH 35f33a69 vs a1733398
  - LINK 71a012ba vs faf30512
  - SOL 1cf35270 vs 9226ef75
  - XRP 88a31675 vs 85cbcac6
- So a stop placed on the new id DOES reserve the old-id position (qty_available=0, R silent). Whether a FILL on the new id depletes the old-id position is still UNVERIFIED. The first real stop fill answers it: compare the next census row.
## FOUND-NOT-FIXED (owner items)
- #22 (evidence above): approve the repair file, or KELLY_SAMPLE_GATE=True. Either one removes the phantom 1.5x crypto multiplier. Model-facing sizing, so not shipped.
- The 12 legacy `desync` rows also lack the flag. Today's code always journals desync with estimated=True (base_loop.py:1634), so these rows are counted in Kelly too. Reported under `legacy_unflagged_estimated_by_design`; not part of the repair.
- `broker_stop` is not an exit reason in today's code; today's equivalent is `server_stop` (base_loop.py:926/:1450). Both are in the default audit scope.
- docs/MODULES.md has no per-script row for W4's crypto_quote_staleness_census.py or for fill_venue_slippage_report.py. Following that pattern, I added no MODULES rows (its header count "20 scripts/*.py" is stale anyway). INTEL: tests/README.md rows for the two new test files.
- JUDGMENT calls in the rule:
  - B alerts only for a NEW loss that persists (a standing B is log-only like A); otherwise the known 6/6 state would alert forever.
  - R is log-only (the X12 rule names only V/B/Q/E).
  - Q counts CFEE by created_at. CFEE posts the next day, so a single-row Q residual of one fee is possible; it stays well under $50.
## VERIFIED-CLEAN
- The Kelly selection and formula mirror equals trading_utils (12 cases).
- test_trading_utils.py is unchanged and green. No tests/test_trade_memory*.py pins existed before this.
## TEST RUNS (each via `hwlock.sh heavy`, CUDA_VISIBLE_DEVICES='', TRADER_LOG_DIR=testlogs)
- `$JPY -m py_compile` on all 4 files: ok.
- `$JPY -m pytest tests/<f>.py -q -p no:cacheprovider`: test_trade_memory_phantom_audit 38 passed · test_paper_quirk_census 13 passed · test_trading_utils 9 passed (hygiene clean, production log untouched).
- The census waited ~75 min for the arbiter (MemAvailable < 1.8 GB while hypersearch_v2 ran).
