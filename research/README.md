# Research wave archive

Full multi-agent research outputs (findings, adversarial verdicts, build
plans). The executive summaries live in the assistant memory; these are
the complete artifacts.

| Wave | File | Focus | Shipped in |
|------|------|-------|-----------|
| 1 | wave1_eval.json | Full codebase evaluation (validation, costs, Jetson, LLM) | phases 0-4 overhaul |
| 2 | wave2_research.json | 9 domains: crypto microstructure, events, labels/meta, portfolio, lifecycle, execution, LLM, ops, red team | tiers 1-2 |
| 3 | wave3_research.json | Selection/timing: cross-sectional ranking, calendar, volume, vol structure, internals, price patterns, ownership | 91b926b, 7d5c566 |
| 4 | wave4_research.json | Chart patterns, TA survivors, leading indicators, ML structure (raw journal extraction — synthesis was hand-built after a network outage) | 6113b72 |
| 5 | wave5_research.json | High-conviction sizing, shorts, options — measurement-first (Stage-0 instrumentation) | c1a8792, d62a955 |
| 6 | wave6_research.json | Integrity/cost/validation: effective-n DSR, CSCV-PBO, per-name EDGE cost, uniqueness weights | 8389f37 |
| 7 | wave7_research.json | Execution timing, shorts, options, carry: entry tactics + IOC, borrow cost, offline short kernel | 0a16a0b |
| 8 | wave8_research.json | Activation of orphaned wave-5/6/7 code + integrity/perf (default-off flags) | 19d1572 |
| 9 | wave9_research.json | MAKE-MONEY dependency chain: meta-label calibration → promotion gate → breadth → edge-sizing | e7e2bed |

## 2026-07 (post-wave) artifacts

| File | What it is |
|------|-----------|
| module_review_2026-07.json | Full 69-module/600-fn review: 90-item owner decision queue (render with `/decision-queue`) |
| econ_research_2026-07.json | Economics/trading-literature sweep: 6 survivors after kill-list filtering, 10 kill-overlaps refused |
| nobel_modern_research_2026-07.md | Nobel/modern-finance research digest: 44 graded findings; verdict = integrity > new-alpha |
| KILL_LIST.md | **Canonical consolidated kill list** (all waves + reviews). Check before proposing any strategy/feature |
| AGENT_CONTEXT.md | Canonical subagent brief — spawned agents read this instead of re-typed context |

Red-team kill lists in each wave are as valuable as the survivors:
they are the do-NOT-build list — consolidated in `KILL_LIST.md`.
