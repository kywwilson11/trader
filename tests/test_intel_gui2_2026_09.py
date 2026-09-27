"""INTEL W8 (2026-09-27): content-aware report freshness + evidence-readiness
panel (gui.py Models tab, pure helpers in chart_core.py).

PySide6-free: chart_core is imported directly (numpy + stdlib); the gui.py
methods are exec'd from the gui.py AST against stubs (the
tests/test_g8_fixes_2026_09.py pattern).

- artifact_validity: every VOID/valid state per producer flag
  (decision_report stale / api_available / stale_reason / quality.representative,
  llm_eval + llm_advisor no_data / insufficient_power, execution empty window /
  no fills, beta joint.underpowered), missing / unreadable / non-dict.
- freshness_state: missing / aged / void / fresh; mtime ageing wins.
- the D-run stub fixtures (W2 run1 root files, byte-copied below): false-fresh
  0/3 (the pre-edit strip read all three as fresh).
- newest_evidence_summary / evidence_readiness_rows / evidence_run_age_s on
  W2's real run1 summary.json (readiness block copied verbatim).
- gui.py wiring: strip text, panel placeholder, rows, (mtime,size) parse gate.
"""
import ast
import html
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import chart_core  # noqa: E402

SRC = (REPO / "gui.py").read_text()
TREE = ast.parse(SRC)


def _node(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name!r} not found in gui.py")


def _assign(name):
    for node in TREE.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            return node.value
    raise AssertionError(f"module-level {name!r} not found in gui.py")


# ---- the D-run stub fixtures: W2 run1 repo-root files, verbatim -----------
DECISION_STALE = {"generated": "2026-09-27T00:26:34.671844-05:00", "days": 30,
                  "api_available": None, "stale": True}
_LLM_META = {"generated_at": "2026-09-27T00:26:35.446727-05:00", "days": 30,
             "asset_filter": "stock", "veto_threshold": 0.15,
             "min_power_n": 60, "min_power_t0": 120, "min_effective_n": 20}
LLM_EVAL_NODATA = {"meta": _LLM_META, "n": 0, "verdict": "no_data",
                   "reason": "no_journal_entries"}
LLM_ADVISOR_NODATA = {"meta": dict(_LLM_META, asset_filter=None,
                                   generated_at="2026-09-27T00:26:35.815931-05:00"),
                      "n": 0, "verdict": "no_data",
                      "reason": "no_journal_entries"}
EXECUTION_EMPTY = {"generated_at": "2026-09-27T00:26:36.238059",
                   "window_days": 30}
# W2 run1 summary.json (generated_at, days, readiness, exit status), verbatim.
RUN1_SUMMARY = json.loads(r"""{
 "generated_at": "2026-09-27T05:26:36.655977+00:00",
 "days": 30,
 "beta_days": 90,
 "readiness": [
  {
   "read": "beta_ledger",
   "rule": "n_obs_used >= 60 daily equity obs",
   "source": "provisional, evidence_reads default (runbook 03:41-43 names the keys, not a count)",
   "observed": "n_obs_used=87",
   "verdict": "READY"
  },
  {
   "read": "decision_report",
   "rule": "quality.priced >= 30",
   "source": "provisional, evidence_reads default",
   "observed": "priced=0",
   "verdict": "NO DATA (stale report (api_available=None))"
  },
  {
   "read": "llm_eval",
   "rule": "n >= 60 AND n_clusters >= 120 AND effective_n >= 20",
   "source": "runbook 03:48 (>=120 distinct hourly t0 clusters, n_eff >= 20 = keep/kill-LLM-spend read); n >= 60 = llm_eval.MIN_POWER_N (llm_eval.py:73-77)",
   "observed": "n=0, n_clusters=n/a, effective_n=n/a",
   "verdict": "NO DATA (no_data stub (no_journal_entries))"
  },
  {
   "read": "llm_eval_stock",
   "rule": "n >= 60 AND n_clusters >= 120 AND effective_n >= 20",
   "source": "runbook 03:48 (>=120 distinct hourly t0 clusters, n_eff >= 20 = keep/kill-LLM-spend read); n >= 60 = llm_eval.MIN_POWER_N (llm_eval.py:73-77)",
   "observed": "n=0, n_clusters=n/a, effective_n=n/a",
   "verdict": "NO DATA (no_data stub (no_journal_entries))"
  },
  {
   "read": "llm_advisor",
   "rule": "n >= 60 AND n_clusters >= 120 AND effective_n >= 20",
   "source": "runbook 03:48 (>=120 distinct hourly t0 clusters, n_eff >= 20 = keep/kill-LLM-spend read); n >= 60 = llm_eval.MIN_POWER_N (llm_eval.py:73-77)",
   "observed": "n=0, n_clusters=n/a, effective_n=n/a",
   "verdict": "NO DATA (no_data stub (no_journal_entries))"
  },
  {
   "read": "execution_report",
   "rule": "buys carrying slippage_bps >= 30",
   "source": "provisional, evidence_reads default",
   "observed": "n_buys_with_slippage=0",
   "verdict": "NO DATA (have n_buys_with_slippage=0)"
  },
  {
   "read": "sizing_cofire",
   "rule": "buy rows with sizing >= 30 (>= 1 for any read)",
   "source": "provisional, evidence_reads default (co-fire matrix; runbook 03:54-55 gives none)",
   "observed": "n_buy_rows=0",
   "verdict": "NO DATA (have n_buy_rows=0)"
  },
  {
   "read": "reliability",
   "rule": "calib_holdout.json present with >= 1 holdout row",
   "source": "scripts/reliability_report.py:7 (input), no power floor",
   "observed": "-",
   "verdict": "SKIPPED (no calib_holdout.json in repo root (hand-made Jetson dump))"
  },
  {
   "read": "ic_by_name",
   "rule": "Stage-0 dump present with >= 1 row",
   "source": "runbook 03:56-60 (dump lands on the next weekly backtest)",
   "observed": "-",
   "verdict": "SKIPPED (no stage0_preds.json (lands on the next weekly backtest))"
  },
  {
   "read": "rank_gradient",
   "rule": "Stage-0 dump present with >= 1 row",
   "source": "runbook 03:56-60 (dump lands on the next weekly backtest)",
   "observed": "-",
   "verdict": "SKIPPED (no stage0_preds.json (lands on the next weekly backtest))"
  },
  {
   "read": "ic_by_name_stock",
   "rule": "Stage-0 dump present with >= 1 row",
   "source": "runbook 03:56-60 (dump lands on the next weekly backtest)",
   "observed": "-",
   "verdict": "SKIPPED (no stock_stage0_preds.json (lands on the next weekly backtest))"
  },
  {
   "read": "rank_gradient_stock",
   "rule": "Stage-0 dump present with >= 1 row",
   "source": "runbook 03:56-60 (dump lands on the next weekly backtest)",
   "observed": "-",
   "verdict": "SKIPPED (no stock_stage0_preds.json (lands on the next weekly backtest))"
  }
 ],
 "failed_steps": [],
 "exit_code": 0
}""")


def _write(p, doc):
    p.write_text(json.dumps(doc, indent=2))
    return p


# ---------------------------------------------------------------------------
# artifact_validity — every state
# ---------------------------------------------------------------------------
V, VOID = "valid", "void"       # literals: a missing API fails per test


def test_state_constants():
    assert (chart_core.VALID, chart_core.VOID) == (V, VOID)


@pytest.mark.parametrize("doc, reason", [
    (DECISION_STALE, "no journal rows"),
    (dict(DECISION_STALE, api_available=False), "no API to price counterfactuals"),
    ({"stale": True, "stale_reason": "analysis error: gates",
      "api_available": True}, "analysis error: gates"),
    ({"stale": True}, "stale report"),
    ({"quality": {"priced": 2, "unpriced_rate": 0.988,
                  "representative": False}},
     "not representative (priced 2, unpriced 99%)"),
])
def test_decision_report_void(doc, reason):
    assert chart_core.artifact_validity(doc, "decision_report") == {
        "state": VOID, "reason": reason}


def test_decision_report_valid():
    doc = {"gates": {}, "quality": {"priced": 40, "representative": True}}
    assert chart_core.artifact_validity(doc, "decision_report")["state"] == V


@pytest.mark.parametrize("kind", ["llm_eval", "llm_advisor"])
def test_llm_no_data_void(kind):
    r = chart_core.artifact_validity(LLM_EVAL_NODATA, kind)
    assert r == {"state": VOID, "reason": "no data: no_journal_entries"}


def test_llm_eval_insufficient_power_top_level_verdict():
    doc = {"n": 70, "verdict": "insufficient_power (n_clusters=40 < 120 ...)",
           "incremental": {"n": 70, "n_clusters": 40, "effective_n_hint": 6.1,
                           "insufficient_power": True}}
    r = chart_core.artifact_validity(doc, "llm_eval")
    assert r["state"] == VOID
    assert r["reason"] == "insufficient power (n=70, clusters=40, n_eff=6.1)"


def test_llm_advisor_insufficient_power_nested_only():
    # the advisor report has no top-level verdict (llm_eval.py report block)
    doc = {"n": 12, "incremental": {"n": 12, "insufficient_power": True}}
    r = chart_core.artifact_validity(doc, "llm_advisor")
    assert r["state"] == VOID and r["reason"].startswith("insufficient power (n=12")


def test_llm_valid():
    doc = {"n": 300, "verdict": "LLM largely ECHOES the ML pred",
           "incremental": {"n": 300, "n_clusters": 200, "effective_n_hint": 25}}
    assert chart_core.artifact_validity(doc, "llm_eval")["state"] == V


def test_execution_states():
    assert chart_core.artifact_validity(EXECUTION_EMPTY, "execution") == {
        "state": VOID, "reason": "no journal rows"}
    no_fills = dict(EXECUTION_EMPTY, crypto_maker_share=0.4)
    assert chart_core.artifact_validity(no_fills, "execution") == {
        "state": VOID, "reason": "no fills with slippage"}
    ok = dict(EXECUTION_EMPTY, overall_mean_bps=3.1)
    assert chart_core.artifact_validity(ok, "execution")["state"] == V


def test_beta_states():
    under = {"joint": {"n_obs": 25, "obs_per_param": 5.0, "underpowered": True}}
    assert chart_core.artifact_validity(under, "beta") == {
        "state": VOID, "reason": "underpowered (5.0 obs/param)"}
    ok = {"joint": {"n_obs": 87, "obs_per_param": 17.4, "underpowered": False}}
    assert chart_core.artifact_validity(ok, "beta")["state"] == V


def test_unknown_kind_is_valid_and_never_opened(tmp_path):
    # shadow / drift / ledgers have no content contract: never read
    assert chart_core.artifact_validity(tmp_path / "absent.json",
                                        None)["state"] == V
    assert chart_core.artifact_validity({"stale": True}, "shadow")["state"] == V


def test_path_states(tmp_path):
    assert chart_core.artifact_validity(tmp_path / "nope.json",
                                        "llm_eval")["state"] == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert chart_core.artifact_validity(bad, "llm_eval") == {
        "state": VOID, "reason": "unreadable JSON"}
    arr = _write(tmp_path / "arr.json", [1, 2])
    assert chart_core.artifact_validity(arr, "beta") == {
        "state": VOID, "reason": "not a JSON object"}
    ok = _write(tmp_path / "ok.json", {"overall_mean_bps": 1.0})
    assert chart_core.artifact_validity(str(ok), "execution")["state"] == V


# ---------------------------------------------------------------------------
# freshness_state + the D-run false-fresh rate
# ---------------------------------------------------------------------------
def test_freshness_state_matrix():
    fs = chart_core.freshness_state
    void = {"state": VOID, "reason": "x"}
    assert fs({"exists": False, "stale": True}, void) == "missing"
    assert fs({"exists": True, "stale": True}, void) == "aged"   # mtime wins
    assert fs({"exists": True, "stale": False}, void) == VOID
    assert fs({"exists": True, "stale": False}, {"state": V}) == "fresh"
    assert fs({"exists": True, "stale": False}, None) == "fresh"


def test_drun_stub_fixtures_false_fresh_rate_is_zero(tmp_path):
    """The brief's proof: fresh-by-mtime D-run stubs are 0/3 'fresh' (the
    mtime-only strip read 3/3)."""
    items = [("decision_report", _write(tmp_path / "decision_report.json",
                                        DECISION_STALE), 7 * 86400),
             ("llm_eval", _write(tmp_path / "llm_eval_report.json",
                                 LLM_EVAL_NODATA), 7 * 86400),
             ("llm_advisor", _write(tmp_path / "llm_advisor_report.json",
                                    LLM_ADVISOR_NODATA), 7 * 86400)]
    rows = chart_core.artifact_freshness(items)
    assert [r["stale"] for r in rows] == [False] * 3   # mtime alone: fresh
    states = [chart_core.freshness_state(r, chart_core.artifact_validity(
        r["path"], r["name"])) for r in rows]
    assert states == [VOID] * 3
    assert sum(s == "fresh" for s in states) == 0


# ---------------------------------------------------------------------------
# evidence-readiness helpers
# ---------------------------------------------------------------------------
def test_newest_evidence_summary(tmp_path):
    base = tmp_path / "logs" / "evidence_reads"
    assert chart_core.newest_evidence_summary(base) is None      # absent
    base.mkdir(parents=True)
    assert chart_core.newest_evidence_summary(base) is None      # empty
    (base / "20260101T000000Z").mkdir()                          # no summary
    (base / "stray.json").write_text("{}")                       # not a dir
    assert chart_core.newest_evidence_summary(base) is None
    a = base / "20260927T052636Z"; a.mkdir()
    b = base / "20260927T052700Z"; b.mkdir()
    pa = _write(a / "summary.json", RUN1_SUMMARY)
    pb = _write(b / "summary.json", RUN1_SUMMARY)
    now = time.time()
    os.utime(pa, (now, now)); os.utime(pb, (now - 7200, now - 7200))
    assert chart_core.newest_evidence_summary(base) == str(pa)   # by mtime
    os.utime(pb, (now, now))
    assert chart_core.newest_evidence_summary(str(base)) == str(pb)  # tie: name
    assert chart_core.newest_evidence_summary(None) is None


def test_readiness_rows_on_w2_run1_summary(tmp_path):
    p = _write(tmp_path / "summary.json", RUN1_SUMMARY)
    rows = chart_core.evidence_readiness_rows(json.loads(p.read_text()))
    assert len(rows) == len(RUN1_SUMMARY["readiness"]) == 12
    by = {r[0]: r for r in rows}
    assert by["beta_ledger"][:5] == ("beta_ledger", "READY", "READY",
                                     "obs 87/60", "provisional")
    assert by["llm_eval"][1:5] == (
        "NO DATA", "NO DATA (no_data stub (no_journal_entries))",
        "n 0/60 · clusters –/120 · n_eff –/20", "documented")
    assert by["decision_report"][3] == "priced 0/30"
    assert by["sizing_cofire"][3] == "buy rows 0/30"   # '(>= 1 ...)' ignored
    assert by["reliability"][1:4] == (
        "SKIPPED", by["reliability"][2], "—")
    assert by["reliability"][4] == "documented"
    assert all(len(r) == 6 for r in rows)


def test_readiness_rows_other_verdicts_and_garbage():
    s = {"readiness": [
        {"read": "decision_report", "rule": "quality.priced >= 30",
         "source": "provisional, evidence_reads default",
         "observed": "priced=2", "verdict": "NOT YET (priced 2 < 30)"},
        {"read": "x", "rule": "n >= 5", "source": "doc", "observed": "-",
         "verdict": "FAILED (exit 3)"},
        {"read": "y", "rule": "n >= 5", "source": "doc", "observed": "-",
         "verdict": "PARSE FAILED (missing n)"},
        {"read": "z", "rule": "no threshold", "source": "doc",
         "observed": "n=4", "verdict": "weird"},
        "not-a-dict"]}
    rows = chart_core.evidence_readiness_rows(s)
    assert [r[1] for r in rows] == ["NOT YET", "FAILED", "PARSE FAILED",
                                    "UNKNOWN"]
    assert rows[0][3] == "priced 2/30"
    assert rows[3][3] == "n=4"                  # threshold mismatch: raw text
    assert chart_core.evidence_readiness_rows(None) == []
    assert chart_core.evidence_readiness_rows({"readiness": None}) == []


def test_evidence_run_age():
    now = 1790486796.655977 + 3600          # generated_at + 1h
    assert chart_core.evidence_run_age_s(RUN1_SUMMARY, now=now) ==         pytest.approx(3600, abs=1)
    assert chart_core.evidence_run_age_s({}, mtime=now - 60, now=now) == 60
    assert chart_core.evidence_run_age_s({"generated_at": "garbage"},
                                         now=now) is None


# ---------------------------------------------------------------------------
# gui.py wiring (AST-exec against stubs)
# ---------------------------------------------------------------------------
class _Color:
    def __init__(self, n):
        self._n = n

    def name(self):
        return self._n


class _Label:
    def __init__(self):
        self.text, self.tip, self.n_set = None, None, 0

    def setText(self, t):
        self.text, self.n_set = t, self.n_set + 1

    def setToolTip(self, t):
        self.tip = t


def _gui_ns(tmp_path, json_mod=json):
    T = {k: _Color(v) for k, v in {"green": "#00ff00", "yellow": "#ffff00",
                                   "red": "#ff0000", "white": "#ffffff",
                                   "muted": "#888888"}.items()}
    ns = {"T": T, "html": html, "os": os, "json": json_mod, "time": time,
          "chart_core": chart_core,
          "REPORT_VALIDITY_KINDS": ast.literal_eval(
              _assign("REPORT_VALIDITY_KINDS")),
          "EVIDENCE_READS_DIR": tmp_path / "logs" / "evidence_reads",
          "META_REFUSED_FILES": {"Crypto": tmp_path / "none1.json"}}
    # get_source_segment drops only the FIRST line's 4-space method indent
    src = "class Win:\n" + "\n".join(
        "    " + ast.get_source_segment(SRC, _node(n)) for n in (
            "_report_validity", "_refresh_evidence_panel",
            "_refresh_reports_freshness"))
    exec(compile(src, "gui.py::W8", "exec"), ns)
    win = ns["Win"]()
    win._reports_fresh_label = _Label()
    win._evidence_label = _Label()
    return ns, win


def test_gui_constants_consistent():
    kinds = ast.literal_eval(_assign("REPORT_VALIDITY_KINDS"))
    labels = {elt.elts[0].value for elt in _assign("REPORT_FRESHNESS_ITEMS").elts}
    assert set(kinds) <= labels
    assert set(kinds.values()) <= set(chart_core.VALIDITY_KINDS)
    src = ast.get_source_segment(SRC, _assign("EVIDENCE_READS_DIR"))
    assert '"logs"' in src and '"evidence_reads"' in src


def test_gui_models_timer_calls_evidence_panel():
    body = ast.get_source_segment(SRC, _node("_refresh_models_tab"))
    assert "self._refresh_evidence_panel()" in body
    assert "self._refresh_reports_freshness()" in body


def test_gui_freshness_strip_shows_void(tmp_path):
    ns, win = _gui_ns(tmp_path)
    ns["REPORT_FRESHNESS_ITEMS"] = [
        ("decision_report", _write(tmp_path / "d.json", DECISION_STALE), 7 * 86400),
        ("llm_eval", _write(tmp_path / "l.json", LLM_EVAL_NODATA), 7 * 86400),
        ("beta", _write(tmp_path / "b.json", {"joint": {"underpowered": False}}),
         7 * 86400),
        ("shadow crypto", tmp_path / "absent.json", 2 * 86400)]
    win._refresh_reports_freshness()
    t = win._reports_fresh_label.text
    assert "decision_report: <span style='color:#888888'>0s · <b>VOID</b> "            "(no journal rows)</span>" in t
    assert "<b>VOID</b> (no data: no_journal_entries)" in t
    assert "beta: <span style='color:#00ff00'>0s</span>" in t
    assert "shadow crypto: —" in t


def test_gui_validity_parse_gate(tmp_path):
    ns, win = _gui_ns(tmp_path)
    p = _write(tmp_path / "l.json", LLM_EVAL_NODATA)
    calls = []
    real = chart_core.artifact_validity
    ns["chart_core"] = types.SimpleNamespace(
        artifact_validity=lambda *a: calls.append(a) or real(*a))
    row = {"name": "llm_eval", "path": str(p), "exists": True}
    assert win._report_validity(row)["state"] == VOID
    assert win._report_validity(row)["state"] == VOID
    assert len(calls) == 1                       # unchanged file: cached
    _write(p, {"n": 300, "verdict": "ok", "incremental": {}, "pad": "x" * 10})
    assert win._report_validity(row)["state"] == V
    assert len(calls) == 2
    assert win._report_validity({"name": "shadow crypto", "path": str(p),
                                 "exists": True}) is None


def test_gui_evidence_panel_placeholder(tmp_path):
    ns, win = _gui_ns(tmp_path)
    win._refresh_evidence_panel()
    assert "no run yet" in win._evidence_label.text
    assert "scripts/evidence_reads.py" in win._evidence_label.text


def test_gui_evidence_panel_rows_and_stat_gate(tmp_path):
    n_load = []

    class _J:
        def __getattr__(self, k):
            return getattr(json, k)

        def load(self, f):
            n_load.append(1)
            return json.load(f)

    ns, win = _gui_ns(tmp_path, json_mod=_J())
    run = ns["EVIDENCE_READS_DIR"] / "20260927T052636Z"
    run.mkdir(parents=True)
    p = _write(run / "summary.json", RUN1_SUMMARY)
    win._refresh_evidence_panel()
    t = win._evidence_label.text
    assert "run 20260927T052636Z" in t and "no ETA projection" in t
    assert t.count("<tr>") == 1 + 12                     # header + 12 reads
    assert "<b>[READY]</b>" in t and "obs 87/60" in t
    assert "n 0/60 · clusters –/120 · n_eff –/20" in t
    assert "provisional" in t and "documented" in t
    assert "llm_eval: runbook 03:48" in win._evidence_label.tip
    win._refresh_evidence_panel()
    assert len(n_load) == 1                              # unchanged: no re-parse
    doc = dict(RUN1_SUMMARY, readiness=RUN1_SUMMARY["readiness"][:2])
    _write(p, doc)
    win._refresh_evidence_panel()
    assert len(n_load) == 2 and win._evidence_label.text.count("<tr>") == 3
    p.write_text("{broken")
    win._refresh_evidence_panel()
    assert "unreadable summary.json" in win._evidence_label.text
