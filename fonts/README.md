# Bundled fonts

Loaded at GUI startup via `QFontDatabase.addApplicationFont` (see gui.py) so the
Jetson renders real typography instead of the DejaVu Sans fallback.

| File | Family | Use | Source |
|---|---|---|---|
| Inter-Variable.ttf | Inter (variable, opsz/wght) | UI text | google/fonts `ofl/inter` |
| IBMPlexMono-Regular.ttf | IBM Plex Mono | numeric role (prices/P&L — tabular by construction) | google/fonts `ofl/ibmplexmono` |
| IBMPlexMono-SemiBold.ttf | IBM Plex Mono SemiBold | emphasized numerics (card values) | google/fonts `ofl/ibmplexmono` |

All three are licensed under the SIL Open Font License 1.1 (OFL). If a file is
missing at runtime the GUI must degrade gracefully to system fonts.
