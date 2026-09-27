#!/usr/bin/env bash
# Build docs/book/Trader_Book.pdf from every chapter (NN_*.md, ordered by filename).
#
#   bash docs/book/build_pdf.sh             # wait for Part One (.part1_done), then build
#   bash docs/book/build_pdf.sh --no-wait   # build now with whatever chapters exist (drafts)
#
# Steps: md2html.py (stdlib only) converts all chapters into ONE HTML file with a table of
# contents; LibreOffice renders it to PDF (`soffice --headless --convert-to pdf`, run under
# `nice -n 10`, with a private throwaway profile so it never collides with another soffice).
# Idempotent: intermediate files live in a temp dir that is removed on exit, and the only
# file written in this directory is Trader_Book.pdf (replaced atomically on success).
set -euo pipefail

BOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PDF="$BOOK_DIR/Trader_Book.pdf"
WAIT=1
POLL_SEC=60
MAX_WAIT_SEC=$((90 * 60))
MIN_PAGES="${MIN_PAGES:-60}"

for arg in "$@"; do
  case "$arg" in
    --no-wait) WAIT=0 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$WAIT" == 1 && ! -e "$BOOK_DIR/.part1_done" ]]; then
  echo "[book] waiting for $BOOK_DIR/.part1_done (poll ${POLL_SEC}s, max $((MAX_WAIT_SEC / 60)) min)"
  waited=0
  while [[ ! -e "$BOOK_DIR/.part1_done" ]]; do
    if (( waited >= MAX_WAIT_SEC )); then
      echo "[book] gave up waiting for Part One; rerun later or use --no-wait" >&2
      exit 3
    fi
    sleep "$POLL_SEC"
    waited=$((waited + POLL_SEC))
  done
  echo "[book] Part One marked done after ${waited}s"
fi

shopt -s nullglob
chapters=("$BOOK_DIR"/[0-9][0-9]_*.md)
shopt -u nullglob
if (( ${#chapters[@]} == 0 )); then
  echo "[book] no chapters found in $BOOK_DIR" >&2
  exit 1
fi
# Glob order is locale-dependent; sort explicitly by filename.
IFS=$'\n' chapters=($(printf '%s\n' "${chapters[@]}" | LC_ALL=C sort)); unset IFS
echo "[book] ${#chapters[@]} chapters:"
printf '        %s\n' "${chapters[@]##*/}"

PY="$(command -v python3)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/trader_book.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

"$PY" "$BOOK_DIR/md2html.py" --title "The Trader Book" \
  --subtitle "An autonomous paper-trading system, explained for its founder" \
  -o "$TMP/Trader_Book.html" "${chapters[@]}"

# Load the HTML into regular Writer (not Writer/Web) so page styles and page breaks apply.
nice -n 10 soffice -env:UserInstallation="file://$TMP/lo_profile" --headless \
  --infilter="HTML (StarWriter)" --convert-to pdf:writer_pdf_Export \
  --outdir "$TMP" "$TMP/Trader_Book.html" >"$TMP/soffice.log" 2>&1 || {
    cat "$TMP/soffice.log" >&2; echo "[book] soffice failed" >&2; exit 1; }

if [[ ! -s "$TMP/Trader_Book.pdf" ]]; then
  cat "$TMP/soffice.log" >&2
  echo "[book] soffice produced no PDF" >&2
  exit 1
fi

pages="$("$PY" - "$TMP/Trader_Book.pdf" <<'EOF'
import re, sys
data = open(sys.argv[1], 'rb').read()
print(len(re.findall(rb'/Type\s*/Page(?![a-zA-Z])', data)))
EOF
)"
mv -f "$TMP/Trader_Book.pdf" "$OUT_PDF"
echo "[book] wrote $OUT_PDF ($pages pages)"
if (( pages <= MIN_PAGES )); then
  echo "[book] WARNING: page count $pages is not above $MIN_PAGES" >&2
  exit 4
fi
