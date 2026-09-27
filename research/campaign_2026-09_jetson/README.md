# 2026-09-26/27 Jetson test & improvement campaign — report

**Status: overnight run COMPLETE; departments wound down (founder's usage budget). 09:20–09:25: on the founder's explicit instruction the Phase-3 bundle was flipped (5 pins modernised), the paper account was liquidated (cash $122,202), the Phase-3 stock retrain (70 trials) was launched, and the stock-only user service `trader.service` was installed and started (linger on). Nothing is committed — the owner reviews first.** Orchestrated by Fable 5.1 as CEO; all audits, fixes and research
by Opus 5.5 agents; from 00:15 on 2026-09-27 three Fable "generals" ran departments (SIGNAL / ENGINE /
INTEL) autonomously under the charter in the session scratchpad (proof bar, serialized snapshot gate,
hardware arbiter, research mandate, landing protocol for the training path).

This is the first time the current code was run against the production stack, data, journals and broker.

## 0. Where things are
- `CHANGELOG.md` (this dir) — one line per landed change, with its test file (append-only).
- `research_signal.md`, `research_engine.md`, `research_intel.md` — what each department learned
  (cited, dated), plus the pre-registered decision rules for every proposed experiment.
- `signal_evidence_runbook.md` — the 12-step offline evidence sequence adapted to the new stores.
- `objective_session_mask_proposal.md`, `startup_trials_owner_item.md`, `mac_dotenv_fix_runbook.md`,
  `llm_eprocess_params.json` (UNSIGNED pre-registration sheet) — owner-facing proposals.
- Device-side artifacts (scratchpad, this session only): audit reports A–H, hunts G1–G8, fix reports,
  the gate logs, the bot harness (`phase5/harness/`), the live-bot monitor (`phase5/monitor.jsonl`),
  the landing kit, and each department's `ROUND_n.md` / `OWNER_ITEMS.md`.

## 1. Baseline truth (2026-09-26 evening)
- Jetson env needs `LD_PRELOAD` of the conda libstdc++ (run_pipeline already sets it; documented in
  memory + CLAUDE.md). Full suite runs in ~2 min here.
- Device was stale: bots down since 2026-05-07; stores ended February on the old 51/53-col schema
  with yfinance :30 bars and Yahoo volume units; April LSTM-only models with no manifest/LGB legs,
  replaying negative gross of costs; no service installed; `bidask` missing; an untracked C
  extension (`indicators_c*.so`) that SIGABRTs the interpreter on frames shorter than the window.
- Initial suite: 3919 passed / 5 failed (C ext, Mac-recorded golden fingerprint, untracked residue, bidask).

## 2. Audits (seven Opus agents + sentiment PIT audit) — headline findings
Feature kernel: C ext heap overflow (ASan) + refcount leak, numerically identical to numba → archived.
Serving: April artifacts serve LSTM-only silently; a full rebuild is mandatory. Data: stock harvest
lost its newest ~3 months (SIP end not clamped), stores contaminated → clean rebuild. Measurement:
16 instruments run; verdicts void on May journals (schema drift). Ops: unit picked the wrong
interpreter and hid the GPU from training; OOM policy; no alerting. GUI: dead orders view, sort
recursion, clipping. LLM: Gemini-only, stale pricing/budget tables, Claude-5 request rules wrong.
Sentiment: confirmed one-day look-ahead leak in Daily_Sentiment (both stores; April models trained on it).

## 3. Fix round + review round (13 fix agents, 1 reviewer, docs sweep) → gates
All landed with failing-before/passing-after tests; suite 3919/5 → 4116/0 → 4237/0 → 4249/0 → 4564/0 …
(see `CHANGELOG.md`). Model-facing items applied INSIDE the clean rebuild (gotcha-#2 event): SIP
end-clamp, sentiment date repair, OI zero-print masking, TB labels stamped after row filters,
sidecar reload tz fix, OBJECTIVE_LONG_ONLY=True (CEO decision, runbook Phase-3 owner-optional).

## 4. The clean rebuild
- Archives pre-synced (funding / OI / short-flow); Finnhub stock sentiment gap filled (42k articles).
- Crypto store: 263,889 rows × 71 cols; stock store: 1,536,356 rows × 110 cols (92 names, 100% Alpaca
  bars, Eff_Spread_Pct from bidask, CS_* panel ranks, TB labels re-derived exactly).
- Training: 6-trial functional pass → all negative, nothing saved (honest ratchet). 40-trial crypto
  search (long-only objective, random-search phase — PRUNE_STARTUP_TRIALS=60), 01:18–06:01 with one
  stop/resume for SIGNAL's landing: 44 trials in the study, 43 negative, one at +0.03; its holdout
  produced 3 trades on 31,662 rows → `Model NOT saved: insufficient_n` (DSR gate fail-closed);
  meta-label and the policy gate then ran their no-champion paths cleanly. No crypto model exists.
  DECOMP-1 verdict on 16 trials: 44/48 fold Sharpes < 0; scoring arithmetic not the cause; every
  searched threshold below the live 1.2 % admission floor → trades lose after cost; more trials of the
  same design won't help. Stock chain (bundle OFF, 25 trials, FAILED_TRIAL_PRUNE on; started 06:29 on SIGNAL's landed
  tree; the new `[FLAGS]` banner proves the run's flag state): finished 09:06 — 25 trials, best +0.024;
  holdout 20 trades (n_eff 14), Sharpe 0.13, DSR 0.058 < 0.60 → `Model NOT saved: failed holdout gate`
  (status ok, fail-closed on DSR); meta-label and the policy gate ran their no-champion paths. No stock
  model exists either. The new `cum_holdout_gates` counter and per-fold decomposition attributes worked. DECOMP-2 on the first 11
  completed trials (definitive, with the new per-fold gross/net/cost attributes): no trial beats a
  zero-skill ranker by > 1 SE; the random-ranker Sharpe predicts the observed ordering (ρ 0.65, p .03);
  fold 1 is ≤ 0 in 11/11 trials; the study's "best" is the 0.0 no-trade sentinel (a config that took
  0/0/6 trades), which cannot be saved. Unlike crypto, 9/11 stock thresholds clear the live floor and
  raw-target drift exceeds cost — the sign is lost through the −0.5·std term and the bear-regime penalty.
  **The night's scientific result:** on clean, leak-free data the current LSTM + objective design shows
  no measurable selection skill in either book; the diagnosis is precise (levers: a minimum-trade prune,
  a drift/benchmark-relative score, the penalty form, regime/entry conditioning, cost-anchored
  thresholds) and every lever is either staged behind a default-OFF flag or written up for the owner.
- Paper bots LIVE since 03:09 (combined mode, exits-only via halt flag, no model → fail-closed):
  behaved exactly as the ENGINE harness predicted (six zero-basis positions, GTC stops, 5 % trail).
  **09:05 — switched to `run_bots.py --stock-only` on the founder's direction** (halt flag kept; the
  crypto loop is no longer running; its GTC stop-limits on the six inherited positions persist at the broker).

## 5. Departments (rounds 1–7)
Volume so far (uncommitted, on this device): 74 CHANGELOG lines, 89 modified tracked files, 88 new files
(65 of them new test files), three research notebooks with ~60 cited 2025–26 sources and pre-registered
decision rules. Every landing carried failing-before/passing-after tests; rounds were gated on the
serialized snapshot gate (box-wide green at 02:38 = 5351 passed / 0 failed; later gates were reddened
only by other departments' in-flight files until the gate was changed to test a tree snapshot — the
final CEO certification gate is recorded in §7).
Highlights — SIGNAL: the seconds→hours n_eff bug in the trainer's holdout (with a 0.9 GB transient),
the trainer flags banner, failed-trial pruning, per-fold score decomposition (DECOMP-1 verdict above),
the Phase-3 bundle verified end to end on a slice, BARS_PER_YEAR measured (stock store is
extended-hours: 3827 bars/yr vs the 1638 constant), a data-quality wick filter, a session-mask
objective proposal. ENGINE: zero-basis position valuation and stop protection, the keeper-stop
day-TIF bug, a fake-broker replay harness driving both loops, restart back-off replayed on the April
crash history (1,616 → 12 restarts), fail-closed quote timestamps, breaker attribution and halt
semantics behind flags, a 686-fill slippage study (cost model's crypto spread far below realised),
a user-level systemd unit path needing no sudo. INTEL: repo-hygiene and production-log guards in the
suite, one-command evidence reads, LLM scorecard size audit (production test over-rejects; the
report-only IM estimator is on nominal), an anytime-valid spend ledger design, content-aware
freshness and alarm hierarchy in the console, the quantified lexicon-fix proposal, docs cite pass.

## 6. Owner decisions — the morning brief
The consolidated, evidence-linked lists are `session_artifacts/generals/engine/OWNER_ITEMS.md` (24
items, §A–D), `session_artifacts/generals/intel/OWNER_ITEMS.md` (12 decisions + 3 parked money asks)
and SIGNAL's items in `session_artifacts/generals/signal/ROUND_3.md` + `startup_trials_owner_item.md`
+ `objective_session_mask_proposal.md`. The five that gate the next 24 hours:

1. **Phase-3 bundle retrain** — DONE 09:20 on the founder's explicit permission (flags True, pins modernised,
   legacy studies archived, 70-trial stock retrain launched 09:21; its gate result will be appended to §4).
   Original text kept for the record: (HYPERSEARCH_V3 + OBJECTIVE_V3 + TRAINING_REPAIRS_V1; OBJECTIVE_LONG_ONLY
   is already True). Evidence: DECOMP-1 (trades lose after cost; thresholds below the admission floor)
   and SIGNAL's end-to-end verification of the ON paths. Blocked for agents by the permission system
   (five tests pin the defaults False: `tests/test_c26_T1.py:335-336`, `tests/test_r2c_lgb_refit.py:171`,
   `tests/test_r2c_blend_coherence.py:294`, `tests/test_r2c_training_repairs.py:210`). Options: allow
   the pin edits → SIGNAL lands flips + FLAGS rows in ~10 min; or flip the three constants + five pins
   by hand; then `session_artifacts/landing/train_phase3_note.md` gives the exact launch, the expected
   `[FLAGS]` banner and the F1 caution (V3's cost-anchored threshold floor may leave 0 certificate
   trades on crypto — still informative). Study reset = move both study DBs + adaptive_state files.
2. **Paper account:** DONE 09:21 — stop orders cancelled and all six positions closed via the API on the
   founder's instruction; cash $122,202, buying power $488k, no positions. Original text: six inherited
   crypto positions (~$122k, avg_entry 0, $94 cash). Bots now protect
   them with 5 % trails and will sell on those or on the account breaker; no entries are possible
   without cash. Reset the paper account (clean start) or accept the inherited book. ENGINE W19 adds: all six sit on
   an Alpaca asset id none of our 1,320 orders ever used (likely the 2026-08-13 account event when
   equity printed == cash) and it is UNVERIFIED whether a sell on the current id reduces them — i.e.
   the resting stops may not protect these positions at all (owner #23). A reset removes the question.
3. **Service without sudo:** DONE 09:22 — user unit installed + enabled, linger on, drop-in override
   `~/.config/systemd/user/trader.service.d/override.conf` sets `--stock-only`; halt flag removed. `bash scripts/setup_jetson_system.sh --user --print-unit` → `--user` →
   `loginctl enable-linger kyle` → `systemctl --user start trader` (ENGINE O10). Never also start bots by
   hand/GUI once the unit runs. Stop the harness bots first (`session_artifacts/harness/stop_bots.sh`).
4. **Alerting:** no Telegram/webhook configured — every breaker/crash/halt alert has been silent
   (notify.py:259). One-minute fix in `.env` (TRADER_TELEGRAM_* or TRADER_WEBHOOK_URL).
5. **Model-facing findings awaiting a ruling** (all flag-gated or written up, nothing shipped): crypto
   flat spread 0.10 % vs realised 26–34 bps taker round trips (ENGINE §A + kill-list ask #1);
   BARS_PER_YEAR_MEASURED (stock store is extended-hours); OBJECTIVE_SESSION_MASK; WICK_PRINT_FILTER
   (96 bad-print bars; ENGINE parity precondition); KW_SCORER_V2 lexicon bundle; IM as the LLM
   keep/kill verdict carrier; CRYPTO_QUOTE_MAX_AGE_SEC ≥300 s; RESTART_STOP_ANCHOR_DESIRED;
   BREAKER_SERVER_FILL_ATTRIB; HALT_CANCELS_WORKING_BUYS; TRADER_BOT_RESTART_BACKOFF;
   PRUNE_STARTUP_TRIALS=60 (a 40-trial run is pure random search); the 0.0 no-trade sentinel.

**Founder direction (2026-09-27 08:55, pending confirmation): stop spending effort on the crypto book;
concentrate on stocks.** Supported by DECOMP-1 (43/44 crypto trials negative, zero-skill ordering) and
ENGINE §A (realised crypto taker round trips 26–34 bps vs the 10 bps model). Mechanics: `run_pipeline.py
--stock-only` / `run_bots.py --stock-only` already exist (nothing is deleted; crypto stays available);
crypto-specific owner items (spread census, funding/OI features, crypto de-risk state) go ON HOLD; the
six inherited crypto positions are the one loose end (a paper reset clears them; their GTC stops persist
at the broker if kept).

Standing facts: nothing in this campaign is committed; `git status` shows ~90 modified and ~90 new
files, all documented in `CHANGELOG.md`; the dev-Mac baseline (`tests/baseline_failures.txt`) is
untouched and the Mac now needs the atomic dotenv-leak procedure in `mac_dotenv_fix_runbook.md`.

## 7. Certification gate (CEO)
Snapshot of the whole tree at 07:19:18 (stock search + paper bots running alongside), full suite in the
snapshot: **5652 passed, 2 failed, 11 skipped, 64 xfailed** (9 m 41 s under load). The two failures are
one ENGINE test file still in flight (`tests/test_engine_r5_rows_llm.py`, W17 stock buy-row keys) and
were routed back to that department; nothing else in the tree is red. Log: session scratchpad
`gates/ceo-cert_20260927_071918.log`. For comparison the campaign started at 3919 passed / 5 failed.
