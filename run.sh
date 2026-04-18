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

PORT_LAND=7777
PORT_PRES=8000
PORT_DASH=8501
PORT_ANAL=8502

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

# ─── 2b. Third-party: INFORM Severity (global monthly severity panel) ───────
# ~130 MB of monthly xlsx → one consolidated CSV. Skipped if already present.
if [ ! -f Data/Third-Party/DRMKC-INFORM/inform_severity_long.csv ]; then
    echo "→ Downloading INFORM Severity snapshots (one-time, ~3–5 min, polite rate-limited)…"
    python3 Data/Third-Party/DRMKC-INFORM/download.py
    echo "→ Consolidating INFORM snapshots into long-format CSVs…"
    dashboard/.venv/bin/python Data/Third-Party/DRMKC-INFORM/consolidate.py
    dashboard/.venv/bin/python Data/Third-Party/DRMKC-INFORM/consolidate_indicators.py
fi
# Back-fill the sub-indicator CSV on installs that predate it.
if [ ! -f Data/Third-Party/DRMKC-INFORM/inform_indicators_long.csv ] && \
   [ -d Data/Third-Party/DRMKC-INFORM/snapshots ]; then
    echo "→ Extracting INFORM sub-indicators (one-time)…"
    dashboard/.venv/bin/python Data/Third-Party/DRMKC-INFORM/consolidate_indicators.py
fi

# Ensure Streamlit's first-run email prompt is pre-silenced
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# ─── 3. Port checks ──────────────────────────────────────────────────────────
# If a port is still held (e.g. stale server from a prior ./run.sh that
# wasn't Ctrl-C'd cleanly), release it. SIGTERM first; SIGKILL if the
# process doesn't exit within 3 s.
free_port() {
    local port=$1
    local pid
    pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
    [ -z "$pid" ] && return 0
    local cmd
    cmd=$(ps -p "$pid" -o comm= 2>/dev/null | xargs)
    echo "→ Port $port held by pid $pid ($cmd) — releasing…"
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3; do
        sleep 1
        pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
        [ -z "$pid" ] && return 0
    done
    echo "  ↳ still held; sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
}

for p in "$PORT_LAND" "$PORT_PRES" "$PORT_DASH" "$PORT_ANAL"; do
    free_port "$p"
done

# ─── 4. Launch servers ───────────────────────────────────────────────────────
LOG_DIR="$(mktemp -d)"
echo "→ Logs: $LOG_DIR"

python3 -m http.server "$PORT_LAND" --directory landing \
    >"$LOG_DIR/landing.log" 2>&1 &
LAND_PID=$!

python3 -m http.server "$PORT_PRES" --directory presentation \
    >"$LOG_DIR/presentation.log" 2>&1 &
PRES_PID=$!

dashboard/.venv/bin/streamlit run dashboard/app.py \
    --server.port="$PORT_DASH" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    >"$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!

dashboard/.venv/bin/streamlit run analysis/app.py \
    --server.port="$PORT_ANAL" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    >"$LOG_DIR/analysis.log" 2>&1 &
ANAL_PID=$!

# ─── 5. Shutdown handling ────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "→ Shutting down…"
    kill "$LAND_PID" "$PRES_PID" "$DASH_PID" "$ANAL_PID" 2>/dev/null || true
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
  │   ▸ Landing          http://localhost:$PORT_LAND  (start here)         │
  │     Presentation     http://localhost:$PORT_PRES                       │
  │     Data exploration http://localhost:$PORT_DASH                       │
  │     Analysis (ML)    http://localhost:$PORT_ANAL                       │
  │     Proposal PDF     proposal/proposal.pdf                       │
  │                                                                  │
  │   Logs:          $LOG_DIR
  │                                                                  │
  │   Ctrl-C to stop.                                                │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

EOF

# Open landing page in the default browser on macOS (others can open
# manually). Individual surface URLs are one click away from landing.
if command -v open >/dev/null 2>&1; then
    ( sleep 1 && open "http://localhost:$PORT_LAND" >/dev/null 2>&1 || true ) &
fi

wait
