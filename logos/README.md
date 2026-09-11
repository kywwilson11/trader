# logos/ — GUI theme artwork

Per-theme logo art for the PySide6 dashboard (`gui.py`), plus the app icon and the root README's
header image. Nothing here is used by the trading path — the Jetson runs headless in production.

## What the code actually loads

`gui.generate_theme_logo(theme_name, size=80)` resolves a theme's pixmap in this order:

1. `logos/96/<file>` — the pre-scaled asset (~13–27 KB),
2. `logos/<file>` — the full-res original (up to 9 MB),
3. an SVG rendering,
4. a transparent placeholder.

Results are memoized in `_LOGO_CACHE` keyed by `(theme_name, size)`. **In normal operation the
multi-megabyte PNGs are never decoded** — step 1 always hits for the nine themes whose file exists.
That `96/` set is the fix from `research/reviews_2026-07/gui_review_2026-07.md`, which recorded the
original problem: 62 MB of full-res PNGs decoded uncached to draw an 80 px icon on an 8 GB box.

`gui.THEMES` defines **12** themes; `gui._THEME_IMAGES` maps 10 of them to a PNG. Terminal and
Paper have no entry and fall through to their SVGs.

| Theme | File | Full-size bytes | `logos/96/` bytes |
|---|---|---:|---:|
| Batman | `batman.png` | 7,942,761 | 17,097 |
| Joker | `joker.png` | 3,268,172 | 20,272 |
| Harley Quinn | `harley_quinn.png` | 6,367,248 | 17,209 |
| Two-Face | `two_face.png` | 7,326,477 | 23,097 |
| Black Metal | `black_metal.png` | 9,081,896 | 19,468 |
| Bubblegum Goth | `bubblegum_goth.png` | 4,104,084 | 19,455 |
| Dark | `night.png` | 6,095,198 | 13,144 |
| Space | `space.png` | 3,971,246 | 19,625 |
| Money | `money.png` | 6,069,883 | 22,476 |
| Salander | `salander.png` | **absent** | — |
| Terminal, Paper | *(no `_THEME_IMAGES` entry)* | — | — |
| *(not a theme)* | `circuit_bull.png` | 8,234,047 | 26,800 |
| | **total** | **62,461,012** | **198,643** |

`salander.png` is referenced by `_THEME_IMAGES` but does not exist on disk. This is not a crash:
the loader's step-3 SVG fallback covers it, and the docstring states themes with no art must never
raise. Restoring the art or dropping the entry is an owner decision.

## The two sites that still load the 8.2 MB file

`logos/96/circuit_bull.png` (26,800 bytes) exists, is committed, and is referenced by **nothing**.
Both consumers of that logo take the full-size original:

- **App icon** — `gui.py`'s `__main__` builds `BASE_DIR / "logos" / "circuit_bull.png"` directly and
  hands it to `QIcon`, with no `96/` preference (unlike `generate_theme_logo`).
- **Root README header** — `README.md` line 2, `<img src="logos/circuit_bull.png" alt="Trader"
  width="200">`. GitHub serves the whole 8.2 MB file to render a 200 px image.

## Size

`logos/` is 62,659,655 tracked bytes (62.7 MB) across 20 files — **≈88% of the repository's tracked
bytes**. The `96/` set is 198,643 bytes; the ratio between the two sets is ≈314×.

**Nothing here has been deleted or moved.** Shrinking the repo is an owner decision, and it would
only help future clones in part: git history already contains the full-size blobs, so removing them
from the working tree does not reclaim `.git` unless history is rewritten. The cheap,
history-preserving wins available if the owner wants them are pointing the app icon and the README
header at `logos/96/circuit_bull.png`.

`.gitignore` carries a `logos/old/` rule; that directory does not exist. Stale but harmless.
