#!/usr/bin/env bash
# Re-render landing/methodology/figures/fig_ontology_v2.png from render_ontology.html.
# Uses headless Chrome to screenshot the funnel, then crops the blank footer
# with Pillow so the PNG is exactly as wide as the funnel and tight vertically.
set -euo pipefail

cd "$(dirname "$0")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML="$PWD/scripts/render_ontology.html"
OUT="$PWD/landing/methodology/figures/fig_ontology_v2.png"

"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1240,470 \
  --default-background-color=FFFFFF \
  --force-device-scale-factor=2 \
  --screenshot="$OUT" \
  "file://$HTML"

dashboard/.venv/bin/python - <<PY
from PIL import Image
im = Image.open("$OUT")
im.crop((0, 0, 2480, 720)).save("$OUT")
print("Wrote $OUT at", im.size)
PY
