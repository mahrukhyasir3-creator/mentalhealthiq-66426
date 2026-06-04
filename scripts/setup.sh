#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

echo "Setting up MentalHealthIQ..."

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python is not available on PATH. Install Python 3.11+ and try again." >&2
  exit 1
fi

echo "Found $($PYTHON --version)"

if [ ! -d ".venv" ]; then
  echo "Creating .venv..."
  "$PYTHON" -m venv .venv
fi

. .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements.txt..."
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example..."
  cp .env.example .env
else
  echo ".env already exists; leaving it unchanged."
fi

mkdir -p data/raw data/processed data/models data/fairness_reports

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "1. Place demographic.csv and questionnaire.csv in data/raw"
echo "2. Run: ./scripts/bootstrap.sh"
echo "3. Run: ./scripts/run-all.sh"
