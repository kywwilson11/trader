# .claude/ — Claude Code assets (committed on purpose)

This directory is **tracked**. `.gitignore` records why: skills / workflows / settings / hooks are
committed so every Claude session behaves the same on the dev Mac and on the Jetson. Only the
machine-local settings file is ignored.

| Path | Git status | Note |
|---|---|---|
| `settings.json`, `hooks/**`, `skills/**`, `workflows/*.js`, `workflows/*.example.json` | tracked | the portable asset set |
| `settings.local.json` | ignored (`.gitignore`) | per-machine read-only allowances only |
| `RESUME.md` | excluded locally via `.git/info/exclude` — **not** `.gitignore` | tool-managed checkpoint pointer; stale, harmless. Leave it alone; the exclude entry is intentional and does not travel to the Jetson |
| `agents/fable-high.md` | untracked as of 2026-09-08 | see § Agents |

## settings.json

Two blocks:

- **`permissions.allow`** (11 entries): `Bash(git stash push:*)`, `Bash(git stash pop)`,
  `Bash(git stash pop:*)`, `Bash(git stash list:*)`, `Bash(sort:*)`, `Bash(diff:*)`,
  `Bash(comm:*)`, `Bash(wc:*)`, `Bash(head:*)`, `Bash(tail:*)`,
  `Bash(python3 .claude/skills/decision-queue/render.py:*)`.
- **`hooks.PostToolUse`**: matcher `Edit|Write` → `${CLAUDE_PROJECT_DIR}/.claude/hooks/py-compile-gate.sh`,
  timeout 10 s, statusMessage "Syntax-checking edited Python file".

**Open conflict (owner decision, not fixed here):** three of the pre-approved rules are exactly
`git stash push` / `git stash pop`, while `research/AGENT_CONTEXT.md` rule 7 tells every agent
*never* to `git stash` a shared tree ("reconstruct baselines from `git show HEAD:<file>` instead"),
and `skills/regression-ab/SKILL.md` still documents the stash A/B as its fallback method. So the
harness pre-approves the one operation the agent brief forbids, on a tree the brief itself
describes as concurrently written. Three-way: settings ↔ skill ↔ AGENT_CONTEXT. Resolving it means
either dropping the stash rules from `settings.json` or relaxing rule 7 — an owner call.

## hooks/

`py-compile-gate.sh` is a one-line `exec python3 "$(dirname "$0")/py_compile_gate.py"`.
`py_compile_gate.py` reads the Claude Code hook JSON from stdin, takes `tool_input.file_path` (or
`.path`), returns 0 for anything that is not `.py`, then `compile(source, path, "exec")`
**in memory** — it writes no `.pyc` and imports nothing, so it is safe for the heavy-dep modules
this Mac cannot import. Exit **2** with a stderr message on `SyntaxError`/`ValueError` (blocking,
message goes back to Claude); exit **0** otherwise. Every unexpected condition fails **open**.
It mirrors CI's `py_compile` stage.

## skills/

| Skill | One line | Reads |
|---|---|---|
| `regression-ab/SKILL.md` | Canonical zero-regression check: primary `bash scripts/ab_check.sh` (names, not counts); fallback the git-stash A/B with explicit preconditions + recovery | `tests/baseline_failures.txt`, `scripts/ab_check.sh`, `CLAUDE.md` |
| `improve/SKILL.md` | Operating manual for `group-improve-v2`: 4 phases (Design+Scout → Sonnet implement → Fable verify → Haiku commit msg), args contract, never-relax rules | `workflows/groups-2026-07.example.json`, `research/module_review_2026-07.json` |
| `panel-improve/SKILL.md` | Operating manual for `module-improve-v3`: the 7-phase panel table, config contract, agent budget `modules × (reviewers + 4) + 2` | `workflows/modules-v3.example.json` (template), `research/KILL_LIST.md`, `research/module_review_2026-07.json` |
| `decision-queue/SKILL.md` + `render.py` | Renders the 90-item 2026-07 owner-decision queue; `render.py` is stdlib-only (Mac-safe), flags `--ledger --severity --module --full` | `research/module_review_2026-07.json` (key `decision_queue_p0_p2_not_autofixed`) |

`regression-ab/SKILL.md` carries a stale orientation count from 2026-07-15. Suite numbers have one
home — `CLAUDE.md` § Running tests; treat any count elsewhere as advisory.

## workflows/

| File | What it is |
|---|---|
| `group-improve-v2.js` | The `/improve` workflow script. Tier A → Fable designer, tier B → Opus, tier C → one agent designs+implements with a Sonnet verifier; one guided repair round; 3 workers; final Haiku agent writes a commit message. `REPO` is hard-coded (overridable via `args.repo`). |
| `module-improve-v3.js` | The `/panel-improve` workflow script. N independent Opus reviewers per module → Fable spec → Sonnet implement → Fable harden → one serialized `bash scripts/ab_check.sh` gate with one guided repair. Defaults: reviewers 3 (clamped 2–5), workers 2, `BUDGET_RESERVE` 80 000 output tokens. `CFG_PATH` defaults to `.claude/workflows/modules-v3.run.json`. |
| `groups-2026-07.example.json` | Template — the completed 12-group 2026-07 campaign config. |
| `modules-v3.example.json` | Template — a 5-module measurement-layer campaign. |

**`modules-v3.run.json` was moved to `archive/claude_workflow_runs/modules-v3.run.json`**
(2026-09-08). It held panel batch B5 (`model_v2` / `model_lgb` / `blend_fit`, 5 reviewers,
report `research/module_improve_v3_batchB5.md`) — a batch that **never ran**: the report does not
exist, none of its test files exist, and its `tree_note` froze a 2026-08 tree description that is
no longer true. Being tracked, it would have been picked up as the default "working config" by the
next `/panel-improve` run, with stale seeds and a false tree note. Its absence is safe:
`module-improve-v3.js` defaults `CFG_PATH` to that path and returns `{error: 'missing modules
config'}` with guidance when it cannot load it — the workflow fails loudly, never silently runs the
wrong thing. To start a new panel batch, copy `modules-v3.example.json` to `modules-v3.run.json`
and edit it; keep the template intact.

## agents/

`agents/fable-high.md` — frontmatter `name: fable-high`, `model: fable`, `effort: high`. Body is a
compact repo brief (read `CLAUDE.md` + `research/AGENT_CONTEXT.md` first; the Mac's missing-dep
list; train/serve parity, default-OFF flags, fail-closed live paths, never commit/push, one writer
per tree, check `research/KILL_LIST.md`; verify with `py_compile` then `bash scripts/ab_check.sh`
judged by failure NAMES). Use it for judgment-heavy work that does not need the orchestrator's
context: `subagent_type: "fable-high"`.

Two operational notes: agent definitions are discovered **at session start**, so a newly added
`agents/*.md` is not usable until the next session; and this file is the only `.claude/` asset that
is untracked — given the stated commit policy for the rest of the directory, tracking it is an
owner decision.
