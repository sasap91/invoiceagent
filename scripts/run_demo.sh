#!/usr/bin/env bash
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_PYTHON="${DEMO_PYTHON:-${DEMO_ROOT}/.venv-model/bin/python}"
DEMO_PORT="${PORT:-8501}"
PUBLIC_MODE="${1:-local}"

if [[ ! -x "${DEMO_PYTHON}" ]]; then
  echo "Missing demo environment: ${DEMO_PYTHON}"
  echo "Create it with: python3.12 -m venv .venv-model && .venv-model/bin/pip install -e '.[demo,model]'"
  exit 1
fi

cd "${DEMO_ROOT}"
"${DEMO_PYTHON}" -m streamlit run procure_app.py \
  --server.headless=true \
  --server.address=127.0.0.1 \
  --server.port="${DEMO_PORT}" \
  --browser.gatherUsageStats=false &
DEMO_APP_PID=$!

cleanup() {
  kill "${DEMO_APP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in {1..40}; do
  if curl --fail --silent "http://127.0.0.1:${DEMO_PORT}/_stcore/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${DEMO_APP_PID}" 2>/dev/null; then
    wait "${DEMO_APP_PID}"
    exit 1
  fi
  sleep 0.25
done

if ! curl --fail --silent "http://127.0.0.1:${DEMO_PORT}/_stcore/health" >/dev/null; then
  echo "Streamlit did not become healthy on port ${DEMO_PORT}."
  exit 1
fi

echo "Local demo: http://127.0.0.1:${DEMO_PORT}"
if [[ "${PUBLIC_MODE}" == "--public" ]]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared is required for --public mode."
    exit 1
  fi
  echo "Starting an ephemeral public Cloudflare URL; keep this terminal open."
  cloudflared tunnel --url "http://127.0.0.1:${DEMO_PORT}"
else
  wait "${DEMO_APP_PID}"
fi
