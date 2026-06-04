#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ ! -f ".venv/bin/activate" ]; then
  echo ".venv is missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

. .venv/bin/activate

echo "Starting frontend at http://localhost:5500"
python -m http.server 5500 -d frontend
