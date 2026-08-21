"""Design tokens — Phase 4.2 groundwork for the gui.py theming migration.

PURE module (stdlib only — no PySide6, no `import gui`) that gives gui.py's
existing color / spacing / type primitives semantic names, per the "Design
language spec" in `research/gui_review_2026-07.md` (S8, Group F):

    color tokens bg.base/raised/overlay/inset, text.hi/mid, accent,
    pnl.up/down, warn (ONE semantic set meant to feed both widgets AND
    charts via chart_core.derive_chart_palette); type scale 24/700,
    15/600, 13/500, 11/600 (+ a mono-tabular numeric role); spacing grid
    4-8-12-16-24; radii 4/6/8/10.

Migration contract (read before touching gui.py's apply_theme)
----------------------------------------------------------------
* Phase A (THIS module) is a byte-compatible RENAMING, nothing else.
  `resolve_colors()` must reproduce today's THEMES values with zero
  transformation — same objects, never copies or re-derivations. Once
  gui.py's apply_theme is rewritten to read e.g. `tokens["bg"]["base"]`
  instead of `T["bg_dark"]`, the rendered pixel must be bit-identical to
  today. TYPE/SPACE/RADIUS/*_qss() are additive (no such named scale exists
  in gui.py today) but their *values* come from the spec above, not from
  reverse-engineering current pixel measurements.
* Phase B (later, NOT this module) is where per-theme tuning happens: e.g.
  giving bg.overlay its own color instead of aliasing bg.raised (see below),
  adding a restrained "terminal pro" dark theme + one paper-light theme
  (gui_review_2026-07.md S8/S11), softening the high-chroma outliers. Only
  change the values in this file once Phase B is explicitly underway.

The 13 real per-theme keys (read from gui.py's THEMES dict, lines ~300-456,
verified 2026-07-21 — do not trust this comment over the source if they
drift): green, red, yellow, white, muted, bg_dark, bg_card, bg_table,
accent, bg_header, bg_border, bg_hover, bg_log. All 10 theme entries
(Batman, Joker, Harley Quinn, Two-Face, Salander, Black Metal, Bubblegum
Goth, Dark, Space, Money) carry exactly these 13 keys as QColor(r, g, b)
instances (no per-theme alpha channel).

Only 10 semantic tokens are in scope for Phase A (do not invent more —
`accent`+soft/glow and `info` from the review's aspirational spec are
deliberately deferred):
    bg.base, bg.raised, bg.overlay, bg.inset,
    text.hi, text.mid,
    accent,
    pnl.up, pnl.down,
    warn
That leaves bg_header, bg_border, bg_hover unmapped by name — control-button
chrome, a border/stroke color, and a hover *state* respectively, none of
which is a static background layer — plus bg_log (see below). All four stay
reachable through the `_raw` passthrough (see `resolve_colors`).

Two mapping calls are not 1:1-obvious from the key names alone; both were
made by reading actual usage in gui.py's `apply_theme`/`_restyle` (not by
guessing from the key name), and both are reversible in Phase B:

* `bg.overlay` -> `bg_card`, the same source as `bg.raised`. Today's palette
  has NO distinct overlay surface: QToolTip (apply_theme:1839,1862), the
  QComboBox popup list (1913), and QMessageBox (1990) all key off bg_card
  already. Aliasing the two tokens to one source is the byte-compatible-
  honest answer for Phase A; Phase B may give overlay its own value.
* `bg.inset` -> `bg_table`, not `bg_log`. bg_table is the color that
  actually styles every recessed input/data well in the app today:
  QLineEdit/QComboBox/QSpinBox backgrounds (1904, 2337, 2355, 2366),
  QTableWidget body (2230), and QPlainTextEdit panels (2394) — the
  representative "inset" role. bg_log's only live use is QPalette.Base
  (1837), a Qt fallback background for *unstyled* text-entry widgets; since
  QPlainTextEdit is styled explicitly with bg_table (2394), bg_log likely
  has little to no visible effect in the running app. It is a real,
  single-purpose color that simply didn't win one of the four named slots —
  still available at `_raw["bg_log"]` if a future token wants it.

Nothing in this module imports gui.py or PySide6, so it is fully importable
and testable on the dev Mac (see tests/test_design_tokens.py, which parses
gui.py's THEMES via `ast` instead of importing it).
"""

# ---------------------------------------------------------------------------
# 1) Type scale + font families
# ---------------------------------------------------------------------------
# role -> (pixel size, weight)
TYPE = {
    "display": (24, 700),
    "heading": (15, 600),
    "body":    (13, 500),
    "small":   (11, 600),
    "tiny":    (10, 500),
}

NUMERIC_FAMILY = "IBM Plex Mono"   # tabular numerals for prices/P&L — see numeric_qss()
UI_FAMILY = "Inter"                # everything else
FALLBACKS = ["Segoe UI", "Roboto", "DejaVu Sans", "sans-serif"]  # if the bundled fonts/ files fail to load

# ---------------------------------------------------------------------------
# 2) Spacing + radius grids
# ---------------------------------------------------------------------------
SPACE = {"s1": 4, "s2": 8, "s3": 12, "s4": 16, "s5": 24}
RADIUS = {"control": 4, "input": 6, "card": 8, "panel": 10}

# ---------------------------------------------------------------------------
# 3) Color token resolution
# ---------------------------------------------------------------------------
# Dark-theme-derived fallback for a source key that is entirely absent from
# a given theme dict (never triggered by any of today's 10 themes — they all
# carry all 13 keys — this only guards a future theme that ships incomplete).
# Values are gui.py THEMES["Dark"][<key>].name() computed by hand, as plain
# lowercase hex strings: this module never constructs a QColor.
SOURCE_DEFAULTS = {
    "green":     "#00c853",
    "red":       "#ff4444",
    "yellow":    "#ffc107",
    "white":     "#dcdcdc",
    "muted":     "#a0a0a0",
    "bg_dark":   "#2b2b2b",
    "bg_card":   "#373737",
    "bg_table":  "#323232",
    "accent":    "#64b5f6",
    "bg_header": "#3a3a3a",
    "bg_border": "#555555",
    "bg_hover":  "#454545",
    "bg_log":    "#1e1e1e",
}

# semantic path -> source THEMES key. Grouped tokens (bg.*, text.*, pnl.*)
# are nested dicts in resolve_colors()'s return value; the two ungrouped
# names (accent, warn) are returned as flat top-level keys.
_GROUPED_TOKEN_MAP = {
    "bg":   {"base": "bg_dark", "raised": "bg_card", "overlay": "bg_card", "inset": "bg_table"},
    "text": {"hi": "white", "mid": "muted"},
    "pnl":  {"up": "green", "down": "red"},
}
_FLAT_TOKEN_MAP = {"accent": "accent", "warn": "yellow"}


def resolve_colors(theme: dict) -> dict:
    """Pure renaming of a gui.py THEMES entry (e.g. `THEMES["Batman"]`) into
    semantic tokens. Every mapped value is the SAME object the input theme
    dict carries under its real key — no copying, no re-wrapping, no color
    math — so this is safe to call every repaint with zero drift risk.

    Missing/unknown keys never raise KeyError: a key absent from `theme`
    falls back to `SOURCE_DEFAULTS` (Dark-theme-derived hex strings), which
    is the one case where the returned value is not verbatim from the
    input, because there is no input value to be verbatim about.

    Returns:
        {
          "bg":   {"base": ..., "raised": ..., "overlay": ..., "inset": ...},
          "text": {"hi": ..., "mid": ...},
          "accent": ...,
          "pnl":  {"up": ..., "down": ...},
          "warn": ...,
          "_raw": theme,   # the original dict, verbatim, for migration fallback
        }
    """
    theme = theme or {}

    def _get(source_key):
        return theme[source_key] if source_key in theme else SOURCE_DEFAULTS[source_key]

    out = {}
    for group, roles in _GROUPED_TOKEN_MAP.items():
        out[group] = {role: _get(src) for role, src in roles.items()}
    for role, src in _FLAT_TOKEN_MAP.items():
        out[role] = _get(src)
    out["_raw"] = theme
    return out


# ---------------------------------------------------------------------------
# 4) QSS string builders
# ---------------------------------------------------------------------------
def _family_stack(*families):
    """Quote real font names; leave bare generic CSS keywords (sans-serif,
    serif, monospace) unquoted, matching gui.py's existing font-family QSS
    convention (line ~1857)."""
    generic = {"sans-serif", "serif", "monospace", "cursive", "fantasy"}
    parts = [fam if fam in generic else f'"{fam}"' for fam in families]
    return ", ".join(parts)


def font_qss() -> str:
    """App-wide font-family QSS fragment (UI_FAMILY then FALLBACKS). The gui
    migration composes this into the QWidget rule inside apply_theme."""
    return f"font-family: {_family_stack(UI_FAMILY, *FALLBACKS)};"


def numeric_qss(px: int = 13, weight: int = 600) -> str:
    """QSS fragment for the numeric role (prices/P&L — tabular-by-
    construction IBM Plex Mono, per fonts/README.md). The gui migration
    composes this into a `[numeric="true"]` attribute selector."""
    return f"font-family: {_family_stack(NUMERIC_FAMILY)}; font-size: {px}px; font-weight: {weight};"
