"""Bootstrap MentalHealthIQ ML artifacts from real NHANES CSV files."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mentalhealthiq.fairness import FAIRNESS_REPORT_PATH, generate_fairness_report
from mentalhealthiq.model import MODEL_PATH, train_and_save
from mentalhealthiq.preprocess import (
    DEMOGRAPHIC_PATH,
    PROCESSED_DIR,
    QUESTIONNAIRE_PATH,
    preprocess_pipeline,
)


def main() -> int:
    """Create artifact folders and run the full ML bootstrap flow."""

    folders = [
        PROJECT_ROOT / "data" / "raw",
        PROCESSED_DIR,
        PROJECT_ROOT / "data" / "models",
        PROJECT_ROOT / "data" / "fairness_reports",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    if not DEMOGRAPHIC_PATH.exists() or not QUESTIONNAIRE_PATH.exists():
        print("Please place demographic.csv and questionnaire.csv inside data/raw/")
        return 1

    preprocess_pipeline()
    train_and_save()
    generate_fairness_report()

    generated_paths = [
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "test.csv",
        PROCESSED_DIR / "train_raw.csv",
        PROCESSED_DIR / "test_raw.csv",
        PROCESSED_DIR / "preprocessor.joblib",
        MODEL_PATH,
        FAIRNESS_REPORT_PATH,
    ]

    print("Generated artifacts:")
    for path in generated_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
