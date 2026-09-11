# research/ — the project's research record

Everything this system knows *because it was investigated*, rather than because it was coded.
Multi-agent research outputs, adversarial verdicts, module reviews, campaign plans, and the
consolidated do-not-rebuild list.

**Layout rule:** the top level holds the **live, canonical, agent-facing** files; every **dated
record lives in a dated subdirectory and is frozen**. Frozen means: read it, cite it, do not
"fix" it — its `file.py:NNN` citations are snapshots of the working tree on the day it was
written, and later waves moved those lines.

For the architecture/system map (what each module is and how a trade happens), see `docs/MAP.md`.
This directory answers "what did we learn and what did we rule out", not "how does it work".

---

## Canonical files (stay flat — do not move these)

| File | Role | Why it stays flat |
|---|---|---|
| `KILL_LIST.md` | **The consolidated do-not-rebuild list** across every wave, review, and literature round | Cited from production code comments and tests, from three workflow-JS prompt sites, and from `.claude/` skills/agents. Every research or build agent MUST read it (including its **PENDING OWNER ASKS** appendix) before proposing anything. An entry leaves this list only by explicit owner decision. |
| `AGENT_CONTEXT.md` | **The canonical subagent brief** | Orchestrator prompts say "read `research/AGENT_CONTEXT.md` first" instead of re-typing context; named by `.claude/agents/` and `.claude/workflows/`. An unbounded set of out-of-repo prompts point at this exact path. |
| `module_review_2026-07.json` | **Live owner-decision queue** — the 2026-07 review's 90 open items (1 P0 / 21 P1 / 68 P2 across 41 modules) | It is *tool input*, not an archive: `/decision-queue` (`.claude/skills/decision-queue/render.py`) reads it by default, and two workflow prompts load it. Queue items are owner decisions — do **not** auto-fix them. |
| `README.md` | this index | — |

## Subdirectories

| Directory | What it holds | Status |
|---|---|---|
| `waves/` | The nine numbered research waves (2026-06): `wave1_eval.json`, `wave{2..9}_research.json` | frozen |
| `reviews_2026-07/` | The 2026-07 review corpus: the module-improve-v3 panel batches, the GUI review, the panel campaign plan | frozen |
| `literature/` | The three literature rounds: 2026-07 economics sweep, 2026-07 Nobel/modern digest, 2026-08 Nobel/modern round 3 | frozen |
| `campaign_2026-08/` | The 2026-08 comprehensive campaign, phases 01–08 — including `03_jetson_runbook.md`, **the activation sequence** | frozen research; the runbook + `06` §4 + `08` IA-4 are the live activation reference |
| `cleanup_2026-09/` | The 2026-09-08 cleanup/map pass: what moved, the verified fix ledger (`fix_ledger.json`), the owner list, the draft commit message | pass record; the durable output is `docs/` |

Each subdirectory has its own README with the per-file detail.

---

## The wave archive (2026-06)

Full multi-agent research outputs — findings, adversarial verdicts, build plans. These are the
complete artifacts; the executive summaries live in the out-of-repo assistant memory
(`~/.claude/projects/-Users-kywwilson-Desktop-Projects-trader/memory/`), and the durable
architecture content is being migrated into `docs/MAP.md`.

| Wave | File | Focus | Shipped in |
|------|------|-------|-----------|
| 1 | `waves/wave1_eval.json` | Full codebase evaluation (validation, costs, Jetson, LLM) | phases 0–4 overhaul |
| 2 | `waves/wave2_research.json` | 9 domains: crypto microstructure, events, labels/meta, portfolio, lifecycle, execution, LLM, ops, red team | tiers 1–2 |
| 3 | `waves/wave3_research.json` | Selection/timing: cross-sectional ranking, calendar, volume, vol structure, internals, price patterns, ownership | `91b926b`, `7d5c566` |
| 4 | `waves/wave4_research.json` | Chart patterns, TA survivors, leading indicators, ML structure (raw journal extraction — synthesis was hand-built after a network outage) | `6113b72` |
| 5 | `waves/wave5_research.json` | High-conviction sizing, shorts, options — measurement-first (Stage-0 instrumentation) | `c1a8792`, `d62a955` |
| 6 | `waves/wave6_research.json` | Integrity/cost/validation: effective-n DSR, CSCV-PBO, per-name EDGE cost, uniqueness weights | `8389f37` |
| 7 | `waves/wave7_research.json` | Execution timing, shorts, options, carry: entry tactics + IOC, borrow cost, offline short kernel | `0a16a0b` |
| 8 | `waves/wave8_research.json` | Activation of orphaned wave-5/6/7 code + integrity/perf (default-off flags) | `19d1572` |
| 9 | `waves/wave9_research.json` | MAKE-MONEY dependency chain: meta-label calibration → promotion gate → breadth → edge-sizing | `e7e2bed` |

All nine hashes re-verified against `git log` on 2026-09-08.

## Post-wave records (2026-07 → 2026-08)

| Record | Where | One line |
|---|---|---|
| 2026-07 econ sweep | `literature/econ_research_2026-07.json` | 6 survivors after kill-list filtering, 10 kill-overlaps refused |
| 2026-07 Nobel/modern digest | `literature/nobel_modern_research_2026-07.md` | 44 graded findings; verdict = integrity > new alpha |
| 2026-07 module review | `module_review_2026-07.json` (flat, live) | 69 modules / 600 fns; the 90-item owner decision queue |
| 2026-07 panel reviews | `reviews_2026-07/` | module-improve-v3 batches 0 + A + B1–B4, the GUI review, the campaign plan (CLOSED at B4) |
| 2026-08 campaign | `campaign_2026-08/` | 10 waves, phases 01–08; default-OFF flags (inventory: `docs/FLAGS.md`); the Jetson activation runbook |
| 2026-08 Nobel/modern round 3 | `literature/nobel_modern_research_2026-08.md` | gaps-only round; 6 verified code findings (D1–D6) |

---

## Kill lists are as valuable as the survivors

Every wave's red-team section, every review's rejections, and every literature round's SKIP
entries are the **do-NOT-build list**. They are consolidated — with original source tags
preserved — in `KILL_LIST.md`. The per-wave and per-review files remain the detailed record of
*why* something was killed; `KILL_LIST.md` is the fast pre-flight check.

A compelling new paper, or a re-reading of an old one, is grounds to **ask**, not to quietly
rebuild a killed idea.
