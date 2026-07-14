"""GUI source contracts (U1-U5): pure source inspection, PySide6-free.

gui.py cannot be imported on the dev Mac (no PySide6), so these tests parse
the source text/AST and assert the shape of the Models-tab upgrades:

- U4: one PIPELINE_STALE_SEC constant, used at all three staleness sites,
  with no leftover literal-600/120 staleness comparisons in those methods
  (the 120 vs 600 disagreement made the retrain button enabled-but-dead).
- U3: _refresh_models_tab renders a stale-status warning between 120s and
  PIPELINE_STALE_SEC, plus an "updated Ns ago" note while running.
- U1: _refresh_models_tab reads "phase_results" and badges the
  'gate_failed_rolled_back' outcome (a gate rollback was invisible before).
- U2: the final-scores block distinguishes key-present-but-None ("search
  failed") from key-absent (blank) via "in pinfo" membership checks.
- U5: a Reports group launches the three measurement-only scripts through
  the engine subprocess pattern (_engine_python / _engine_env).
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUI_PATH = REPO / "gui.py"
SRC = GUI_PATH.read_text()
TREE = ast.parse(SRC)
SRC_LINES = SRC.splitlines()


def _method_source(name):
    """Source text of a function/method by name, wherever it's nested."""
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return "\n".join(SRC_LINES[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"method {name!r} not found in gui.py")


# ---------------------------------------------------------------------------
# U4 — unified staleness constant
# ---------------------------------------------------------------------------

class TestU4StalenessConstant:
    def test_defined_once_at_module_level(self):
        defs = [n for n in TREE.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PIPELINE_STALE_SEC"
                        for t in n.targets)]
        assert len(defs) == 1, "PIPELINE_STALE_SEC must be defined exactly once"
        assert defs[0].value.value == 600

    def test_referenced_at_three_or_more_sites(self):
        # References = occurrences beyond the single definition line
        refs = len(re.findall(r"\bPIPELINE_STALE_SEC\b", SRC))
        assert refs >= 4, f"expected definition + >=3 uses, found {refs} total"

    @pytest.mark.parametrize("method", [
        "_refresh_models_tab", "_trigger_retrain", "_is_pipeline_running"])
    def test_each_staleness_site_uses_constant(self, method):
        body = _method_source(method)
        assert "PIPELINE_STALE_SEC" in body, \
            f"{method} must compare age against PIPELINE_STALE_SEC"

    @pytest.mark.parametrize("method", [
        "_refresh_models_tab", "_trigger_retrain", "_is_pipeline_running"])
    def test_no_literal_staleness_comparisons_remain(self, method):
        body = _method_source(method)
        # No `age < 600` / `age < 120` style literal comparisons left behind.
        assert not re.search(r"age\s*<\s*(600|120)\b", body), \
            f"{method} still compares age against a literal 600/120"


# ---------------------------------------------------------------------------
# U3 — staleness banner
# ---------------------------------------------------------------------------

class TestU3StalenessBanner:
    def test_stale_banner_window(self):
        body = _method_source("_refresh_models_tab")
        assert re.search(r"120\s*<\s*age\s*<\s*PIPELINE_STALE_SEC", body), \
            "banner must trigger for 120 < age < PIPELINE_STALE_SEC"

    def test_stale_banner_text_and_color(self):
        body = _method_source("_refresh_models_tab")
        assert "stale" in body
        assert "⚠" in body, "banner should carry the warning glyph"
        # yellow per the file's palette idiom
        assert re.search(r"T\[['\"]yellow['\"]\].*stale|stale.*T\[['\"]yellow['\"]\]",
                         body, re.S)

    def test_updated_ago_shown_while_running(self):
        body = _method_source("_refresh_models_tab")
        assert "updated {age:.0f}s ago" in body


# ---------------------------------------------------------------------------
# U1 — phase_results rendering
# ---------------------------------------------------------------------------

class TestU1PhaseResults:
    def test_reads_phase_results(self):
        body = _method_source("_refresh_models_tab")
        assert '"phase_results"' in body or "'phase_results'" in body

    def test_handles_gate_rollback_outcome(self):
        body = _method_source("_refresh_models_tab")
        assert "gate_failed_rolled_back" in body
        assert "rolled back" in body

    def test_handles_ok_and_failed_badges(self):
        body = _method_source("_refresh_models_tab")
        assert "✓" in body  # ok
        assert "✗" in body  # failed

    def test_attempts_annotation(self):
        body = _method_source("_refresh_models_tab")
        assert "attempts" in body
        assert re.search(r"attempts\s*>\s*1", body)

    def test_label_built_with_word_wrap(self):
        body = _method_source("_build_models_tab")
        assert "_pipeline_phase_results" in body
        assert re.search(
            r"_pipeline_phase_results\.setWordWrap\(True\)", body)

    def test_empty_dict_clears_label(self):
        body = _method_source("_refresh_models_tab")
        assert re.search(
            r'_pipeline_phase_results\.setText\(\s*[\'"][\'"]\s*\)', body), \
            "empty phase_results must clear the label"


# ---------------------------------------------------------------------------
# U2 — tri-state final scores
# ---------------------------------------------------------------------------

class TestU2TriStateScores:
    def test_membership_checks(self):
        body = _method_source("_refresh_models_tab")
        assert '"crypto_final_score" in pinfo' in body
        assert '"stock_final_score" in pinfo' in body

    def test_search_failed_rendered_in_red(self):
        body = _method_source("_refresh_models_tab")
        assert "search failed" in body
        # rendered with the palette's red
        idx = body.index("search failed")
        window = body[max(0, idx - 200):idx + 50]
        assert "T['red']" in window or 'T["red"]' in window


# ---------------------------------------------------------------------------
# U5 — Reports group
# ---------------------------------------------------------------------------

class TestU5Reports:
    def test_reports_group_and_buttons(self):
        body = _method_source("_build_models_tab")
        assert 'QGroupBox("Reports")' in body
        assert "Decision Report" in body
        assert "Beta Ledger" in body
        assert "Indicator Lead/Lag" in body

    def test_report_commands(self):
        body = _method_source("_build_models_tab")
        assert '"decision_report.py", "--days", "30"' in body
        assert '"beta_ledger.py", "--days", "90"' in body
        assert ('"indicator_leadlag.py", "--data", '
                '"crypto_training_data.parquet"') in body

    def test_run_report_uses_engine_pattern(self):
        body = _method_source("_run_report_clicked")
        assert "_engine_python()" in body
        assert "_engine_env()" in body
        assert "Popen" in body
        assert "subprocess.STDOUT" in body
        # never PIPE: nothing drains it, a chatty child would hang
        assert "subprocess.PIPE" not in body

    def test_poll_method_wired(self):
        body = _method_source("_run_report_clicked")
        assert "_check_report_run" in body
        # poll method exists and re-enables buttons + shows dialog
        poll = _method_source("_check_report_run")
        assert "poll()" in poll
        assert "setEnabled(True)" in poll
        assert "_show_report_dialog" in poll

    def test_report_dialog_shape(self):
        body = _method_source("_show_report_dialog")
        assert "QDialog" in body
        assert "QPlainTextEdit" in body
        assert "setReadOnly(True)" in body
        assert "900" in body and "600" in body
