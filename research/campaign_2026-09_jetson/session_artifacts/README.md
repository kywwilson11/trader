# Session artifacts of the 2026-09-26/27 Jetson campaign (copied from the session scratchpad)

Everything here was produced on the Jetson during the overnight campaign and copied verbatim so it
outlives the session's scratchpad. Paths inside these files that mention `/tmp/claude-…/scratchpad`
refer to the original location; the same files are here under the same relative names.

| Dir | What |
|---|---|
| `reports/` | Phase-2 audits A–H, fix-round reports FIX_*, the independent review, IMPL_* implementer reports, HARNESS.md |
| `hunt/` | the "indisputable improvement" hunts G1–G8 (findings with proofs + judgment-call appendices) |
| `generals/<dept>/` | each Fable general's ROUND_n.md and OWNER_ITEMS.md (the owner's decision lists) |
| `landing/` | SIGNAL landing notes (train_phase3_note.md — the one-command Phase-3 launch once the owner clears the pins) and PROOF.md files |
| `harness/` | the paper-bot observation harness (start/stop/monitor/summarize) — scratchpad tooling, not a repo module |
| `gates/` | the serialized snapshot gate + hardware arbiter scripts and every gate's one-line result |
| `CHARTER.md`, `CAMPAIGN_BRIEF.md`, `INDISPUTABLE_BRIEF.md` | the rules the agents worked under |
| `train_crypto_40trials.log` | the crypto search log (44 trials, holdout insufficient_n) |
