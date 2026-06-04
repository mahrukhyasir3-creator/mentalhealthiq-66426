#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ ! -f ".venv/bin/activate" ]; then
  echo ".venv is missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

. .venv/bin/activate

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

if [ -f ".env" ]; then
  ENV_API_HOST="$(sed -n 's/^API_HOST=//p' .env | tail -n 1)"
  ENV_API_PORT="$(sed -n 's/^API_PORT=//p' .env | tail -n 1)"
  [ -n "$ENV_API_HOST" ] && API_HOST="$ENV_API_HOST"
  [ -n "$ENV_API_PORT" ] && API_PORT="$ENV_API_PORT"
fi

echo "Starting API at http://localhost:$API_PORT"
python -m uvicorn mentalhealthiq.api:app --reload --host "$API_HOST" --port "$API_PORT"
