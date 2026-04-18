#!/usr/bin/env bash
#
# GEO-Insight — launch all local surfaces with one command.
#
# Brings up the reveal.js presentation (http://localhost:8000) and the Streamlit
# data-landscape dashboard (http://localhost:8501), and points you at the
# proposal PDF. Idempotent: downloads source data and creates the dashboard
# venv on first run if missing, then just starts servers on subsequent runs.
#
# Usage:
#     ./run.sh
#
# Stop:
#     Ctrl-C  (both servers shut down cleanly)
#

set -euo pipefail

cd "$(dirname "$0")"

PORT_PRES=8000
PORT_DASH=8501

# ─── 1. Source data ──────────────────────────────────────────────────────────
if ! python3 Data/download.py --check >/dev/null 2>&1; then
    echo "→ Downloading source data (~270 MB, one-time, ~2–3 min)…"
    python3 Data/download.py
fi

# ─── 2. Dashboard venv ───────────────────────────────────────────────────────
if [ ! -x dashboard/.venv/bin/streamlit ]; then
    echo "→ Setting up dashboard virtualenv (one-time)…"
    python3 -m venv dashboard/.venv
    dashboard/.venv/bin/pip install --quiet --upgrade pip
    dashboard/.venv/bin/pip install --quiet -r dashboard/requirements.txt
fi

# Ensure Streamlit's first-run email prompt is pre-silenced
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# ─── 3. Port checks ──────────────────────────────────────────────────────────
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
if port_busy "$PORT_PRES"; then
    echo "✗ Port $PORT_PRES already in use. Stop the process using it or edit run.sh." >&2
    exit 1
fi
if port_busy "$PORT_DASH"; then
    echo "✗ Port $PORT_DASH already in use. Stop the process using it or edit run.sh." >&2
    exit 1
fi

# ─── 4. Launch servers ───────────────────────────────────────────────────────
LOG_DIR="$(mktemp -d)"
echo "→ Logs: $LOG_DIR"

python3 -m http.server "$PORT_PRES" --directory presentation \
    >"$LOG_DIR/presentation.log" 2>&1 &
PRES_PID=$!

dashboard/.venv/bin/streamlit run dashboard/app.py \
    --server.port="$PORT_DASH" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    >"$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!

# ─── 5. Shutdown handling ────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "→ Shutting down…"
    kill "$PRES_PID" "$DASH_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "  Done."
    exit 0
}
trap cleanup INT TERM

# ─── 6. Wait for readiness, report, optionally open browser ──────────────────
sleep 2

cat <<EOF

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   Presentation   http://localhost:$PORT_PRES                           │
  │   Dashboard      http://localhost:$PORT_DASH                           │
  │   Proposal PDF   proposal/proposal.pdf                           │
  │                                                                  │
  │   Logs:          $LOG_DIR
  │                                                                  │
  │   Ctrl-C to stop.                                                │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

EOF

# Open URLs in the default browser on macOS (harmless on Linux; 'open'
# may not exist elsewhere — failures are swallowed).
if command -v open >/dev/null 2>&1; then
    ( sleep 1 && open "http://localhost:$PORT_PRES" >/dev/null 2>&1 || true ) &
    ( sleep 2 && open "http://localhost:$PORT_DASH" >/dev/null 2>&1 || true ) &
fi

wait
