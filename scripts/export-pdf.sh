#!/usr/bin/env bash
#
# Export the running reveal.js deck to PDF via headless Chrome.
# Requires presentation to be reachable at http://localhost:8000
# (run ./run.sh first).
#
# Output:  presentation/slides.pdf
#
set -euo pipefail

cd "$(dirname "$0")/.."

URL="http://localhost:8000/?print-pdf"
OUT="presentation/slides.pdf"

# Locate Chrome/Chromium on common macOS / Linux paths
CHROME=""
for candidate in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
    "$(command -v google-chrome 2>/dev/null || true)" \
    "$(command -v chromium 2>/dev/null || true)" \
    "$(command -v chromium-browser 2>/dev/null || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        CHROME="$candidate"
        break
    fi
done

if [ -z "$CHROME" ]; then
    echo "✗ No Chrome/Chromium found. Install Chrome or set CHROME env var." >&2
    exit 1
fi

# Verify server is up
if ! curl -s -o /dev/null -w "%{http_code}" "$URL" | grep -q "^200$"; then
    echo "✗ Presentation not reachable at http://localhost:8000 — run ./run.sh first." >&2
    exit 1
fi

echo "→ Using $CHROME"
echo "→ Exporting $URL → $OUT"

# Use a throwaway --user-data-dir every run. Without this, Chrome reuses the
# default profile, which has caused spurious blank PDFs when the profile was
# left in a bad state (e.g. cancelled previous run, cached empty page).
TMP_PROFILE=$(mktemp -d -t chrome-pdf-export)
trap 'rm -rf "$TMP_PROFILE"' EXIT

"$CHROME" \
    --headless=new \
    --disable-gpu \
    --no-sandbox \
    --hide-scrollbars \
    --no-pdf-header-footer \
    --virtual-time-budget=45000 \
    --run-all-compositor-stages-before-draw \
    --window-size=1280,800 \
    --user-data-dir="$TMP_PROFILE" \
    --print-to-pdf-no-header \
    --print-to-pdf="$OUT" \
    "$URL" 2>&1 | grep -v "^$" | head -20

if [ -f "$OUT" ]; then
    echo "✓ Wrote $OUT ($(du -h "$OUT" | cut -f1))"
else
    echo "✗ PDF not produced — check Chrome output." >&2
    exit 1
fi
