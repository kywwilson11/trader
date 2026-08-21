# Commit-message drafts — 2026-08 campaign (owner commits after review)

The campaign's edits are interleaved across ~60 files in one working tree (packet diffs in the
session scratchpad conflate earlier waves when taken per-file), so **one commit is the honest
unit**. Primary draft below; an optional two-commit split follows if you prefer separating tests.

## Primary draft (single commit)

```
feat: 2026-08 comprehensive campaign — gate honesty, engine safety, cost truth, LLM economics

Ten gated waves (understand -> web research -> fix -> build -> adversarial
self-hunt), every wave serialized behind scripts/ab_check.sh; suite grew
2769 -> 3401 passed with zero regressions vs the 23-name dev-Mac baseline.

Promotion honesty: calendar-concurrency effective-n + cumulative-trials
deflation + noisy ratchet (PROMOTION_GATE_V2); the weekly gate can target
the CHALLENGER with hold-not-rollback semantics (GATE_TARGETS_CHALLENGER);
shadow DM v2 = per-timestamp collapse + Ibragimov-Muller cluster t + two
scheduled looks (TRADER_SHADOW_DM_V2, false-promote ~9-13% vs 16-39%);
hypersearch final full-window refit, LGB-before-gate, fitted+shrunk blend
weight, holdout certifies the deployed blend+q10 (HYPERSEARCH_V3), per-
ticker objective reset + cost-anchored threshold range (OBJECTIVE_V3).

Meta/calibration: staged+guarded meta publish (refusal keeps the previous
triple live), isotonic tie-pooling + logit-Platt + real embargo
(CALIBRATION_V2), OOF primary persistence killing the in-sample leak
(META_OOF_PRED), admission-parity training population
(META_REPLAY_POLICY_PARITY).

Live-engine safety (direct): prediction fan-out timeout + pool rebuild
rate-limit, REST timeouts, per-book flatten, partial-fill tracking, maker
abort-on-unknown + deterministic rung ids, confirm-only liquidation, LLM
outage backoff + stale-veto expiry, peak-equity ordering, TP-leg confirmed
exits (Kelly un-censoring), emergency zero-size, stop-classification
journaling.

De-risk stack (DERISK_STACK_V2): regime family aggregates by MIN with one
VIX read + hysteresis, modal state = 1.0, crypto regime input = BTC
trailing-RV state machine, kill-listed pseudo-CAPE/HMM excluded, single
vol-target scope, deposit-outlier exclusion; v2 composition shadow-
journaled on every entry while OFF.

Cost & data truth (dark flags + census): quote-first crypto spread tiers,
minute-bar stock EDGE, vol-scale impact re-base, D40 floor-fill fix,
cost-regime features wiring, raw-OHLCV sidecars (head-creep kill),
yfinance venue slice + Src provenance, closed-bar discipline
(TRADER_CLOSED_BARS_V2 + always-on replay flips), repo-wide pct_change
pins + pandas<3 bound.

LLM economics: llm_eval rebuilt on timestamp-clustered DK errors with
bar-stepped stock horizons + effective-n power gates + spend ledger
(false-keep 15% -> 5% in null sims); Anthropic prompt caching + pricing
registry; free-endpoint qualification harness (scripts/llm_qualify.py);
dedup-cache prep.

New instruments: Stage-0 predictions dump + hourly MTM equity (unblocks
wave-9 activation chain), sizing co-fire report, learned sentiment
lexicon (supervised SESTM/word-power, dark), meta learning-curve harness,
stationary-bootstrap Sharpe p-value, daily-bars cache restoring 13
constant-at-inference stock features + HAR-RV daily feed
(TRADER_DAILY_FEATURE_RESTORE / TRADER_HAR_DAILY_FEED), promotion ledger,
journal gzip rotation, EOD digest, run_bots ops thread.

Verification stack: ab_check hardened (sanity floor, watchdog, flaky
triage, PY_COLORS=0), baseline shrunk to 23 names with import-skip
retrofit, Jetson-parity CI leg (lightgbm + pinned numpy/pandas +
alpaca-py), GUI decision-truth panels + freshness strip, beta_ledger
additive corrected/clean/lagged keys.

Docs: research/campaign_2026-08/{01_state_map,02_research,
03_jetson_runbook,04_commit_messages}.md; KILL_LIST pending-asks appendix;
CLAUDE.md + AGENT_CONTEXT refreshed. Activation is owner-driven per the
runbook; every model-facing change ships default-OFF.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
```

## Optional split (if you prefer two commits)

1. `feat: 2026-08 campaign — production code, flags, instruments, docs` (everything except tests/)
2. `test: 2026-08 campaign — ~630 new tests across 30 test_c26_*/v3 modules + stub modernizations`

Note: the split is cosmetic; ab_check gates the pair only when applied together.
