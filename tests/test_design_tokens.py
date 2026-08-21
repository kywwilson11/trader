"""design_tokens.py contract tests — pure stdlib, no PySide6 anywhere.

gui.py cannot be imported on the dev Mac (no PySide6), so THEMES is parsed
straight out of gui.py's source via `ast`, the same technique
tests/test_gui_contracts.py uses for its Models-tab contracts. Each theme's
QColor(r, g, b) literals become plain (r, g, b) tuples — design_tokens.py
never needs to know what a "color" actually is (that's the point of a pure
renaming module), so a tuple stand-in exercises `resolve_colors` exactly the
same way a real QColor would: verbatim passthrough, checked with `is`.
"""
import ast
from pathlib import Path

import pytest

import design_tokens as dt

REPO = Path(__file__).resolve().parent.parent
GUI_PATH = REPO / "gui.py"
SRC = GUI_PATH.read_text()
TREE = ast.parse(SRC)

EXPECTED_SOURCE_KEYS = {
    "green", "red", "yellow", "white", "muted",
    "bg_dark", "bg_card", "bg_table", "accent",
    "bg_header", "bg_border", "bg_hover", "bg_log",
}


def _parse_themes():
    """{theme_name: {role_name: (r, g, b)}} straight from gui.py's THEMES
    dict literal — no import, no PySide6, no execution of gui.py at all."""
    themes_node = None
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "THEMES" for t in node.targets
        ):
            themes_node = node.value
            break
    assert themes_node is not None, "THEMES assignment not found in gui.py"
    assert isinstance(themes_node, ast.Dict), "THEMES is not a dict literal"

    themes = {}
    for name_node, body_node in zip(themes_node.keys, themes_node.values):
        assert isinstance(body_node, ast.Dict), f"{name_node.value!r} theme body is not a dict literal"
        roles = {}
        for role_node, color_node in zip(body_node.keys, body_node.values):
            assert isinstance(color_node, ast.Call), (
                f"{name_node.value!r}.{role_node.value!r} is not a QColor(...) call"
            )
            roles[role_node.value] = tuple(arg.value for arg in color_node.args)
        themes[name_node.value] = roles
    return themes


THEMES = _parse_themes()


# ---------------------------------------------------------------------------
# (b) coverage sanity — parsing actually found the real THEMES dict
# ---------------------------------------------------------------------------
class TestThemeDiscovery:
    def test_at_least_ten_themes_found(self):
        assert len(THEMES) >= 10, f"only found {len(THEMES)} themes: {sorted(THEMES)}"

    @pytest.mark.parametrize("theme_name", sorted(_parse_themes()))
    def test_theme_has_exactly_the_13_expected_keys(self, theme_name):
        assert set(THEMES[theme_name]) == EXPECTED_SOURCE_KEYS


# ---------------------------------------------------------------------------
# (a) every theme resolves with zero missing keys + verbatim value identity
# ---------------------------------------------------------------------------
class TestResolveColorsVerbatim:
    @pytest.mark.parametrize("theme_name", sorted(THEMES))
    def test_resolves_without_error(self, theme_name):
        theme = THEMES[theme_name]
        resolved = dt.resolve_colors(theme)
        assert isinstance(resolved, dict)

    @pytest.mark.parametrize("theme_name", sorted(THEMES))
    def test_every_mapped_token_is_verbatim_source_value(self, theme_name):
        """Same object back out, not a copy/re-derivation — `is`, not `==`."""
        theme = THEMES[theme_name]
        resolved = dt.resolve_colors(theme)

        assert resolved["bg"]["base"] is theme["bg_dark"]
        assert resolved["bg"]["raised"] is theme["bg_card"]
        assert resolved["bg"]["overlay"] is theme["bg_card"]
        assert resolved["bg"]["inset"] is theme["bg_table"]
        assert resolved["text"]["hi"] is theme["white"]
        assert resolved["text"]["mid"] is theme["muted"]
        assert resolved["accent"] is theme["accent"]
        assert resolved["pnl"]["up"] is theme["green"]
        assert resolved["pnl"]["down"] is theme["red"]
        assert resolved["warn"] is theme["yellow"]

    @pytest.mark.parametrize("theme_name", sorted(THEMES))
    def test_raw_passthrough_is_the_same_dict(self, theme_name):
        theme = THEMES[theme_name]
        resolved = dt.resolve_colors(theme)
        assert resolved["_raw"] is theme

    @pytest.mark.parametrize("theme_name", sorted(THEMES))
    def test_string_compare_against_source_literal_matches_too(self, theme_name):
        """Belt-and-suspenders: value-equality against the literal tuples
        parsed straight from gui.py's source text, independent of the `is`
        check above."""
        theme = THEMES[theme_name]
        resolved = dt.resolve_colors(theme)
        mapping = {
            ("bg", "base"): "bg_dark",
            ("bg", "raised"): "bg_card",
            ("bg", "overlay"): "bg_card",
            ("bg", "inset"): "bg_table",
            ("text", "hi"): "white",
            ("text", "mid"): "muted",
            ("pnl", "up"): "green",
            ("pnl", "down"): "red",
        }
        for (group, role), source_key in mapping.items():
            assert resolved[group][role] == theme[source_key]
        assert resolved["accent"] == theme["accent"]
        assert resolved["warn"] == theme["yellow"]


class TestResolveColorsFallback:
    def test_missing_keys_default_instead_of_raising(self):
        resolved = dt.resolve_colors({})  # nothing present -> every key missing
        assert resolved["bg"]["base"] == dt.SOURCE_DEFAULTS["bg_dark"]
        assert resolved["bg"]["raised"] == dt.SOURCE_DEFAULTS["bg_card"]
        assert resolved["bg"]["overlay"] == dt.SOURCE_DEFAULTS["bg_card"]
        assert resolved["bg"]["inset"] == dt.SOURCE_DEFAULTS["bg_table"]
        assert resolved["text"]["hi"] == dt.SOURCE_DEFAULTS["white"]
        assert resolved["text"]["mid"] == dt.SOURCE_DEFAULTS["muted"]
        assert resolved["accent"] == dt.SOURCE_DEFAULTS["accent"]
        assert resolved["pnl"]["up"] == dt.SOURCE_DEFAULTS["green"]
        assert resolved["pnl"]["down"] == dt.SOURCE_DEFAULTS["red"]
        assert resolved["warn"] == dt.SOURCE_DEFAULTS["yellow"]

    def test_partial_theme_only_defaults_the_missing_key(self):
        theme = {"bg_dark": "sentinel-value"}
        resolved = dt.resolve_colors(theme)
        assert resolved["bg"]["base"] == "sentinel-value"
        assert resolved["bg"]["raised"] == dt.SOURCE_DEFAULTS["bg_card"]

    def test_none_theme_does_not_raise(self):
        resolved = dt.resolve_colors(None)
        assert resolved["bg"]["base"] == dt.SOURCE_DEFAULTS["bg_dark"]


# ---------------------------------------------------------------------------
# (c) TYPE/SPACE/RADIUS structure sanity
# ---------------------------------------------------------------------------
class TestScales:
    def test_type_roles_match_spec(self):
        assert dt.TYPE == {
            "display": (24, 700),
            "heading": (15, 600),
            "body": (13, 500),
            "small": (11, 600),
            "tiny": (10, 500),
        }

    @pytest.mark.parametrize("role", ["display", "heading", "body", "small", "tiny"])
    def test_type_values_are_px_weight_int_pairs(self, role):
        px, weight = dt.TYPE[role]
        assert isinstance(px, int) and isinstance(weight, int)
        assert px > 0 and weight > 0

    def test_space_scale_matches_spec(self):
        assert dt.SPACE == {"s1": 4, "s2": 8, "s3": 12, "s4": 16, "s5": 24}

    def test_space_scale_is_strictly_increasing(self):
        ordered = [dt.SPACE[k] for k in ("s1", "s2", "s3", "s4", "s5")]
        assert ordered == sorted(ordered)
        assert len(set(ordered)) == len(ordered)

    def test_radius_scale_matches_spec(self):
        assert dt.RADIUS == {"control": 4, "input": 6, "card": 8, "panel": 10}

    def test_font_family_constants(self):
        assert dt.NUMERIC_FAMILY == "IBM Plex Mono"
        assert dt.UI_FAMILY == "Inter"
        assert dt.FALLBACKS == ["Segoe UI", "Roboto", "DejaVu Sans", "sans-serif"]


# ---------------------------------------------------------------------------
# (d) qss fragments contain the families and no hex literals
# ---------------------------------------------------------------------------
class TestQssBuilders:
    def test_font_qss_contains_ui_family_and_fallbacks(self):
        qss = dt.font_qss()
        assert dt.UI_FAMILY in qss
        for fam in dt.FALLBACKS:
            assert fam in qss

    def test_font_qss_has_no_hex_literals(self):
        assert "#" not in dt.font_qss()

    def test_font_qss_is_a_declaration(self):
        qss = dt.font_qss()
        assert "font-family" in qss
        assert qss.strip().endswith(";")

    def test_numeric_qss_contains_numeric_family(self):
        qss = dt.numeric_qss()
        assert dt.NUMERIC_FAMILY in qss

    def test_numeric_qss_has_no_hex_literals(self):
        assert "#" not in dt.numeric_qss()

    def test_numeric_qss_defaults(self):
        qss = dt.numeric_qss()
        assert "13px" in qss
        assert "600" in qss

    def test_numeric_qss_honors_custom_params(self):
        qss = dt.numeric_qss(px=11, weight=700)
        assert "11px" in qss
        assert "700" in qss
        assert "13px" not in qss
