# research/waves/ — the nine numbered research waves (2026-06) · FROZEN

The complete multi-agent research artifacts for waves 1–9: every finding, the adversarial
red-team verdicts on each one, the survivor rankings, and the build plans that came out of them.
The executive summaries live in the out-of-repo assistant memory (one file per wave); **these
JSONs are the full record**.

**Frozen.** Nothing here is edited after its wave closed. The `file.py:NNN` citations inside are
snapshots of the working tree on the day each wave ran — later waves moved those lines, so read
them as evidence of what was true then, not as current pointers. Waves also cite each other by the
flat pre-reorg path these files used to have (a bare `research/` prefix, no `waves/` segment);
those in-document citations were deliberately left untouched.

| Wave | File | Focus | Shipped in |
|------|------|-------|-----------|
| 1 | `wave1_eval.json` | Full codebase evaluation (validation, costs, Jetson, LLM) | phases 0–4 overhaul |
| 2 | `wave2_research.json` | 9 domains: crypto microstructure, events, labels/meta, portfolio, lifecycle, execution, LLM, ops, red team | tiers 1–2 |
| 3 | `wave3_research.json` | Selection/timing: cross-sectional ranking, calendar, volume, vol structure, internals, price patterns, ownership | `91b926b`, `7d5c566` (2026-06-11) |
| 4 | `wave4_research.json` | Chart patterns, TA survivors, leading indicators, ML structure (raw journal extraction — synthesis was hand-built after a network outage) | `6113b72` (2026-06-12) |
| 5 | `wave5_research.json` | High-conviction sizing, shorts, options — measurement-first (Stage-0 instrumentation) | `c1a8792`, `d62a955` (2026-06-12) |
| 6 | `wave6_research.json` | Integrity/cost/validation: effective-n DSR, CSCV-PBO, per-name EDGE cost, uniqueness weights | `8389f37` (2026-06-18) |
| 7 | `wave7_research.json` | Execution timing, shorts, options, carry: entry tactics + IOC, borrow cost, offline short kernel | `0a16a0b` (2026-06-18) |
| 8 | `wave8_research.json` | Activation of orphaned wave-5/6/7 code + integrity/perf (default-off flags) | `19d1572` (2026-06-18) |
| 9 | `wave9_research.json` | MAKE-MONEY dependency chain: meta-label calibration → promotion gate → breadth → edge-sizing | `e7e2bed` (2026-06-18) |

Hashes re-verified against `git log` on 2026-09-08. The same table appears in `../README.md`;
both describe a frozen archive, so they cannot drift apart.

## What a wave JSON contains

Roughly: the seed questions handed to each researcher, the raw findings, an adversarial pass that
either upholds a finding with a `reason` or kills it with a `fatal_flaw`, a `revised_impact` where
the red team downgraded rather than killed, and a tiered build plan tagged `mac_now` /
`jetson_later`. The kill sections are the point as much as the survivors.

## Before proposing anything from a wave

Read `../KILL_LIST.md` first. It consolidates every KILLED / REJECTED / refuted verdict from these
nine files (source-tagged `wave-2` … `wave-7`) plus the 2026-07 reviews and the literature rounds.
Items leave that list only by explicit owner decision.
