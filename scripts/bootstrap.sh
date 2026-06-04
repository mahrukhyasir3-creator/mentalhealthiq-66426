#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ ! -f ".venv/bin/activate" ]; then
  echo ".venv is missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

if [ ! -f "data/raw/demographic.csv" ] || [ ! -f "data/raw/questionnaire.csv" ]; then
  echo "Raw data files are missing. Place demographic.csv and questionnaire.csv in data/raw, then run this script again." >&2
  exit 1
fi

. .venv/bin/activate

echo "Bootstrapping preprocessing, model training, and fairness artifacts..."
python scripts/bootstrap_ml.py
