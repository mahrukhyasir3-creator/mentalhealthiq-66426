#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ ! -f ".venv/bin/activate" ]; then
  echo ".venv is missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Starting MongoDB and Mongo Express with Docker Compose..."
  docker compose up -d mongodb mongo-express
else
  echo "Docker Compose is not available or not running. Continuing without MongoDB."
fi

echo
echo "MentalHealthIQ URLs:"
echo "API:           http://localhost:8000"
echo "Health:        http://localhost:8000/health"
echo "Frontend:      http://localhost:5500"
echo "Mongo Express: http://localhost:8081"
echo
echo "Starting API and frontend. Press Ctrl+C to stop both."

./scripts/run-api.sh &
API_PID=$!
./scripts/run-frontend.sh &
FRONTEND_PID=$!

trap 'kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true' INT TERM EXIT
wait
