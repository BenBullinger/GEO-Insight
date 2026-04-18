#!/usr/bin/env bash
# Export the slide deck to presentation/slides.pdf.
#
# Uses the locally-running reveal.js server on :8000 and Chrome headless to
# render ?print-pdf. If no server is running on :8000, we launch a temporary
# one, render, then shut it down.
#
# Usage:
#   ./presentation/export_pdf.sh

set -euo pipefail

cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -x "$CHROME" ]; then
    echo "Error: Google Chrome not found at $CHROME"
    exit 1
fi

# Ensure a server is on :8000.
started_server=0
if ! curl -s -o /dev/null "http://localhost:8000/"; then
    echo "→ Starting temporary http.server on :8000…"
    python3 -m http.server 8000 >/dev/null 2>&1 &
    SERVER_PID=$!
    started_server=1
    trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
    # wait for the port to come up
    for _ in 1 2 3 4 5 6 7 8; do
        curl -s -o /dev/null "http://localhost:8000/" && break
        sleep 0.5
    done
fi

# Kill any orphan headless Chrome processes from a prior run that raced.
pkill -9 -f "chrome-export-slides" 2>/dev/null || true
sleep 1

PROFILE="$(mktemp -d -t chrome-export-slides)"
OUT="slides.pdf"
rm -f "$OUT"

echo "→ Rendering ?print-pdf with Chrome headless…"
"$CHROME" \
    --headless=new \
    --disable-gpu \
    --no-sandbox \
    --hide-scrollbars \
    --no-pdf-header-footer \
    --virtual-time-budget=45000 \
    --run-all-compositor-stages-before-draw \
    --user-data-dir="$PROFILE" \
    --print-to-pdf="$OUT" \
    "http://localhost:8000/?print-pdf" \
    2>/dev/null

if [ -s "$OUT" ] && [ "$(stat -f%z "$OUT")" -gt 100000 ]; then
    pages=$(/usr/bin/mdls -name kMDItemNumberOfPages -raw "$OUT" 2>/dev/null || echo "?")
    size=$(stat -f%z "$OUT")
    echo "✓ Wrote $OUT ($size bytes, $pages pages)"
else
    echo "✗ Export produced an empty or invalid PDF. Check that the server is serving the deck correctly."
    exit 1
fi

rm -rf "$PROFILE"
