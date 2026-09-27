"""INTEL Scout D (2026-09) — WCAG 2.x contrast MEASUREMENT across every gui.py theme.

Measurement-only: this file changes no production code. It pins what is true
TODAY so a future palette fix is forced to update it:

* every (theme, pair) that passes today asserts its WCAG floor;
* every (theme, pair) that fails today is `xfail(strict=True)` with the measured
  ratio in the reason — fixing the colour makes it XPASS, which strict mode turns
  into a FAILURE, forcing whoever fixed it to delete the entry from KNOWN_FAIL.
* a theme ADDED to gui.THEMES (e.g. the proposed "Ops") is picked up
  automatically with no xfail marks, i.e. it must pass every pair on arrival.

Spec: WCAG 2.2 SC 1.4.3 (text >= 4.5:1; large text >= 3:1), SC 1.4.11 (non-text UI
components / graphical objects >= 3:1 against adjacent colours), SC 1.4.1 (colour
is never the only carrier). Relative luminance per the WCAG 2.x definition
(sRGB linearisation threshold 0.03928, coefficients .2126/.7152/.0722), contrast
= (L1 + 0.05) / (L2 + 0.05). https://www.w3.org/TR/WCAG22/ and
https://www.w3.org/WAI/GL/wiki/Relative_luminance . The repo's own helper
(chart_core.contrast_ratio) is used for every measurement; `_wcag_contrast`
below is an independent re-implementation of the spec that cross-checks it.

gui.py is never imported (PySide6 lives only in the base env): THEMES is parsed
out of the source with `ast`, the same technique tests/test_design_tokens.py uses.
All chip font sizes are 11-12 px (gui.py `_chip_style`, the regime-chip QSS in
`_restyle`), i.e. NOT "large-scale" text, so every text pair uses the 4.5:1 floor.

Run `python tests/test_intel_contrast_2026_09.py` (or pytest -s) to print the matrix.
"""
import ast
import colorsys
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:   # allow `python tests/<this file>` (pytest's rootdir already covers it)
    sys.path.insert(0, str(REPO))

import chart_core as cc  # noqa: E402
_TREE = ast.parse((REPO / "gui.py").read_text())

TEXT_MIN = 4.5      # WCAG 2.2 SC 1.4.3 (normal-size text)
NONTEXT_MIN = 3.0   # WCAG 2.2 SC 1.4.11


def _parse_themes():
    """{theme: {key: '#rrggbb'}} from gui.py's THEMES dict literal (no import)."""
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "THEMES" for t in node.targets):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                out[k.value] = {
                    rk.value: "#%02x%02x%02x" % tuple(a.value for a in rv.args[:3])
                    for rk, rv in zip(v.keys, v.values)}
            return out
    raise AssertionError("THEMES not found in gui.py")


THEMES = _parse_themes()

# (pair_id, foreground key or tuple-of-keys [best-of = visual boundary], background key, floor, widget)
PAIRS = [
    ("text.hi/base", "white", "bg_dark", TEXT_MIN, "body text on window (apply_theme QMainWindow)"),
    ("text.hi/raised", "white", "bg_card", TEXT_MIN, "body text on cards/tooltips"),
    ("text.hi/inset", "white", "bg_table", TEXT_MIN, "table/input text"),
    ("text.mid/base", "muted", "bg_dark", TEXT_MIN, "muted labels on window (status bar, DD '—')"),
    ("text.mid/raised", "muted", "bg_card", TEXT_MIN, "card titles; heartbeat 'off-hours' chip"),
    ("text.mid/inset", "muted", "bg_table", TEXT_MIN, "muted text in tables/inputs"),
    ("accent/base", "accent", "bg_dark", TEXT_MIN, "QGroupBox titles, accent text"),
    ("warn/base", "yellow", "bg_dark", TEXT_MIN, "warn text (alerts 'stale', event flags)"),
    ("danger/base", "red", "bg_dark", TEXT_MIN, "halt banner / flatten echo / DD badge"),
    ("ok/base", "green", "bg_dark", TEXT_MIN, "DD badge 'at peak', phase badge check"),
    ("chip.hb_ok", "green", "bg_card", TEXT_MIN, "_refresh_heartbeats ok -> _chip_style"),
    ("chip.hb_stale", "red", "bg_card", TEXT_MIN, "_refresh_heartbeats stale -> _chip_style"),
    ("chip.shadow", "yellow", "bg_card", TEXT_MIN, "Settings shadow-mode chip (default ON)"),
    ("chip.regime", "accent", "bg_header", TEXT_MIN, "Markets regime chips (_restyle chip_style)"),
    ("chip.mode", "bg_dark", "accent", TEXT_MIN, "Cockpit PAPER mode chip (_refresh_cockpit_banner)"),
    ("nt.hb_chip_edge", ("bg_card", "bg_border"), "bg_dark", NONTEXT_MIN, "heartbeat chip fill/border vs window"),
    ("nt.regime_edge", ("bg_header", "bg_border"), "bg_dark", NONTEXT_MIN, "regime chip fill/border vs window"),
    ("nt.mode_fill", "accent", "bg_dark", NONTEXT_MIN, "mode chip fill vs window"),
    ("nt.risk_ok_bar", "green", "bg_header", NONTEXT_MIN, "risk gauge <70% chunk vs QProgressBar groove"),
]
PAIR_BY_ID = {p[0]: p for p in PAIRS}

# Measured 2026-09-27 on the Jetson (chart_core.contrast_ratio, 2 dp). 46 of 228.
KNOWN_FAIL = {
    ("Black Metal", "text.mid/base"): 2.96, ("Two-Face", "text.mid/base"): 4.49,
    ("Black Metal", "text.mid/raised"): 2.80, ("Salander", "text.mid/raised"): 4.29,
    ("Two-Face", "text.mid/raised"): 3.95, ("Black Metal", "text.mid/inset"): 2.87,
    ("Paper", "text.mid/inset"): 4.44, ("Salander", "text.mid/inset"): 4.45,
    ("Two-Face", "text.mid/inset"): 4.18, ("Paper", "warn/base"): 2.70,
    ("Black Metal", "danger/base"): 3.05, ("Dark", "danger/base"): 4.15,
    ("Terminal", "danger/base"): 4.37, ("Two-Face", "danger/base"): 4.21,
    ("Paper", "ok/base"): 3.64, ("Paper", "chip.hb_ok"): 4.09,
    ("Black Metal", "chip.hb_stale"): 2.89, ("Dark", "chip.hb_stale"): 3.49,
    ("Terminal", "chip.hb_stale"): 4.02, ("Two-Face", "chip.hb_stale"): 3.69,
    ("Paper", "chip.shadow"): 3.04, ("Harley Quinn", "chip.regime"): 4.35,
}
for _pid in ("nt.hb_chip_edge", "nt.regime_edge"):
    for _t, _v in {"Batman": 1.74, "Black Metal": 1.38, "Bubblegum Goth": 1.69, "Dark": 1.90,
                   "Harley Quinn": 1.69, "Joker": 1.67, "Money": 1.91, "Paper": 1.45,
                   "Salander": 1.52, "Space": 1.58, "Terminal": 1.43, "Two-Face": 2.05}.items():
        KNOWN_FAIL[(_t, _pid)] = _v


def measure(theme, pair_id):
    _, fg, bg, _, _ = PAIR_BY_ID[pair_id]
    b = theme[bg]
    keys = fg if isinstance(fg, tuple) else (fg,)
    return max(cc.contrast_ratio(theme[k], b) for k in keys)


# --------------------------------------------------------------------------
# independent WCAG 2.x re-implementation (cross-check of chart_core)
# --------------------------------------------------------------------------
def _wcag_lum(hex_color):
    h = hex_color.lstrip("#")
    ch = []
    for i in (0, 2, 4):
        s = int(h[i:i + 2], 16) / 255.0
        ch.append(s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def _wcag_contrast(a, b):
    la, lb = sorted((_wcag_lum(a), _wcag_lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_chart_core_contrast_is_wcag_exact():
    assert cc.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert cc.contrast_ratio("#777777", "#ffffff") == pytest.approx(4.478, abs=1e-3)
    colours = sorted({c for th in THEMES.values() for c in th.values()})
    for a in colours[::3]:
        for b in colours[1::5]:
            assert cc.contrast_ratio(a, b) == pytest.approx(_wcag_contrast(a, b), rel=1e-12)


def test_known_fail_table_is_not_stale():
    for (theme, pid), ratio in KNOWN_FAIL.items():
        assert theme in THEMES and pid in PAIR_BY_ID, (theme, pid)
        assert measure(THEMES[theme], pid) == pytest.approx(ratio, abs=0.006), (theme, pid)


def _params():
    out = []
    for theme in sorted(THEMES):
        for pid, _, _, floor, _ in PAIRS:
            marks = ()
            if (theme, pid) in KNOWN_FAIL:
                marks = pytest.mark.xfail(strict=True, reason=(
                    f"{KNOWN_FAIL[(theme, pid)]:.2f} < {floor}:1 — owner item Scout D"))
            out.append(pytest.param(theme, pid, marks=marks, id=f"{theme}|{pid}"))
    return out


@pytest.mark.parametrize("theme,pair_id", _params())
def test_pair_meets_wcag_floor(theme, pair_id):
    floor = PAIR_BY_ID[pair_id][3]
    assert measure(THEMES[theme], pair_id) >= floor


def render_matrix(themes=None):
    themes = themes or THEMES
    names = sorted(themes)
    lines = ["pair".ljust(16) + "".join(n[:8].rjust(9) for n in names)]
    for pid, _, _, floor, _ in PAIRS:
        cells = []
        for n in names:
            r = measure(themes[n], pid)
            cells.append(("%.2f" % r + ("*" if r < floor else " ")).rjust(9))
        lines.append(pid.ljust(16) + "".join(cells))
    return "\n".join(lines)


def test_print_matrix(capsys):
    text = render_matrix()
    with capsys.disabled():
        print("\n" + text + "\n(* = below floor: text 4.5:1, nt.* 3.0:1)")
    assert len(text.splitlines()) == len(PAIRS) + 1


# --------------------------------------------------------------------------
# Proposed ISA-101 "Ops" theme (Scout D) — NOT in gui.py; pinned here so the
# pre-registered acceptance rule is executable before anyone adopts it.
# --------------------------------------------------------------------------
OPS_PROPOSAL = {
    "bg_dark": "#1f2124", "bg_card": "#292c30", "bg_table": "#24272a",
    "bg_header": "#313439", "bg_border": "#80868f", "bg_hover": "#363a3f",
    "bg_log": "#1a1c1f", "white": "#e3e5e8", "muted": "#a3a8b0",
    "accent": "#aeb6c2", "green": "#93b59f",
    "red": "#ff5c5c",      # alarm P1 ONLY (halt, flatten, order-error, stale in-hours)
    "yellow": "#f2b233",   # alarm P2 / warn ONLY
}
# roles the 13-key schema cannot express today (owner ask: new tokens)
OPS_EXTRA = {"ok_nominal": "#a3a8b0", "loss": "#b98e8e"}
NOMINAL_KEYS = ("bg_dark", "bg_card", "bg_table", "bg_header", "bg_border", "bg_hover",
                "bg_log", "white", "muted", "accent", "green")
NOMINAL_MAX_SAT = 0.25   # HSV saturation; ISA-101 level-3 "normal values" ~15%
ALARM_MIN_SAT = 0.50


def _sat(h):
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)[1]


class TestOpsProposal:
    def test_same_13_keys_as_gui_themes(self):
        assert set(OPS_PROPOSAL) == set(next(iter(THEMES.values())))

    @pytest.mark.parametrize("pair_id", [p[0] for p in PAIRS])
    def test_every_pair_passes(self, pair_id):
        assert measure(OPS_PROPOSAL, pair_id) >= PAIR_BY_ID[pair_id][3]

    @pytest.mark.parametrize("role", sorted(OPS_EXTRA))
    @pytest.mark.parametrize("bg", ["bg_dark", "bg_card", "bg_table"])
    def test_extra_roles_are_readable_text(self, role, bg):
        assert cc.contrast_ratio(OPS_EXTRA[role], OPS_PROPOSAL[bg]) >= TEXT_MIN

    def test_nominal_colours_are_unsaturated(self):
        nominal = [OPS_PROPOSAL[k] for k in NOMINAL_KEYS] + list(OPS_EXTRA.values())
        assert max(_sat(c) for c in nominal) <= NOMINAL_MAX_SAT

    def test_alarm_colours_saturated_and_reserved(self):
        alarms = {OPS_PROPOSAL["red"], OPS_PROPOSAL["yellow"]}
        assert min(_sat(c) for c in alarms) >= ALARM_MIN_SAT
        nominal = {OPS_PROPOSAL[k] for k in NOMINAL_KEYS} | set(OPS_EXTRA.values())
        assert not (alarms & nominal)

    def test_profit_loss_luminance_gap_cvd(self):
        # same 0.08 gap chart_core.separate_luminance enforces for charts
        gap = abs(cc.rel_luminance(OPS_PROPOSAL["green"]) - cc.rel_luminance(OPS_EXTRA["loss"]))
        assert gap >= 0.08


if __name__ == "__main__":
    print(render_matrix())
    print("\nOps proposal:\n" + render_matrix({"Ops*": OPS_PROPOSAL}))
    for k, v in {**OPS_PROPOSAL, **OPS_EXTRA}.items():
        print(f"{k:11s} {v}  sat={_sat(v):.2f}  vs base {cc.contrast_ratio(v, OPS_PROPOSAL['bg_dark']):.2f}"
              f"  vs raised {cc.contrast_ratio(v, OPS_PROPOSAL['bg_card']):.2f}")
