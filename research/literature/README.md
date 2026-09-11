# research/literature/ — the external-literature rounds · FROZEN

Three rounds of "what does the published research actually say, and does any of it apply to *this*
system?". Each round was screened against `../KILL_LIST.md` before anything was proposed, and each
contributed its own rejections back to that list.

The system every round is judged against: LSTM + LightGBM blend on hourly bars, 6 large-cap
cryptos + ~45 mostly-liquid US stocks, **long-only**, retail costs (~50–60 bps round-trip crypto,
~15–25 bps stocks), Jetson 8 GB in production, honest validation (purged CV, Deflated Sharpe,
CSCV-PBO, shadow A/B).

**Frozen.** Dated records; their code citations are working-tree snapshots from the day each was
written.

---

## Round 1a — `econ_research_2026-07.json` (2026-07-15)

The economics sweep. Six themed literature searchers (Nobel lineages; ML / factor zoo; volatility
and tails; momentum / reversal / seasonality; crypto; sizing, execution and costs) produced 50 raw
findings, which an Opus compiler deduped, filtered against the wave-2..9 kill lists, and mapped to
applicability; a Fable pass then ordered the result and tagged each survivor `mac_now` or
`jetson_later`. Output: **6 survivors** (led by honest effective-trials accounting for the DSR
deflation), **16 items already implemented** here, and a **`killed_overlaps` array of 10** ideas
refused because the system had already killed them — PEAD, CAPE/valuation mean-reversion and
friends. That array is source key **`econ-07`** in the consolidated kill list.

## Round 1b — `nobel_modern_research_2026-07.md` (2026-07-15)

The Nobel/modern-finance digest, and the companion to the JSON above. Three searchers returned 45
findings; one exact duplicate merged, leaving **44 unique**, each graded by evidence quality and
ranked by *evidence grade × fit-to-this-system*, with WebFetch spot-checks on the load-bearing and
the off-looking claims (including a future-dated arXiv paper that turned out to be real, and whose
numbers were *worse* than the finding claimed). Organized as laureate foundations, modern research,
proven strategies, and a top-10 actionable list. The overall verdict was **integrity before new
alpha**. Its Section 1 and Section 3 SKIP entries are source key **`nobel-07`** in the kill list.

## Round 3 — `nobel_modern_research_2026-08.md` (2026-08-22)

The gaps-only round: laureate/nominee-tier economics not yet touched, 2025–26 economic research
outside the ML-architecture ground that round 2 covered, and strategy families not yet screened.
(Round 2 is the ML frontier — `../campaign_2026-08/05_frontier_research.md`.) It opens with a
verification-discipline section to read before trusting any number, and re-screens everything
against the kill list including its PENDING OWNER ASKS appendix.

Its most immediately useful part is **Section 0 — six verified code findings** surfaced while
reading the repo for the literature work, each checked line-by-line:

| ID | Finding |
|---|---|
| D1 | Backtest/live macro stand-down parity gap |
| D2 | `macro_calendar.py` has no NFP — the stand-down list is FOMC + CPI only |
| D3 | Vol-target sizing splits daily vol flat across the day |
| D4 | The LLM spend ledger never nets benefit against cost (the Grossman-Stiglitz gap) |
| D5 | `_DV30` horizon mismatch latent in the dark-impact path |
| D6 | The stablecoin peg halt is not cascade protection |

The rest is sectioned as laureate foundations, modern economics 2025–26, a strategy-family screen,
ranked top actionables with ship classes, honest negatives + proposed kill-list additions, and a
quarantine list of items that must not be acted on until one verification pass clears or kills
them. It explicitly does **not** displace the 12-step Jetson experiment sequence in
`../campaign_2026-08/06_signal_model_plan.md`.
