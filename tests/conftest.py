from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mentalhealthiq.fairness import generate_fairness_report
from mentalhealthiq.model import train_and_save
from mentalhealthiq.preprocess import PHQ_COLUMNS, preprocess_pipeline


def _phq_rows() -> list[list[int]]:
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 1, 1],
        [3, 3, 3, 3, 3, 3, 2, 1, 1],
    ]


@pytest.fixture()
def raw_nhanes_files(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    demographic_records = []
    questionnaire_records = []
    seqn = 1000

    for repeat in range(4):
        for class_index, phq_values in enumerate(_phq_rows()):
            seqn += 1
            demographic_records.append(
                {
                    "SEQN": seqn,
                    "RIDAGEYR": [17, 24, 37, 52, 68][class_index],
                    "RIAGENDR": 1 + ((repeat + class_index) % 2),
                    "RIDRETH1": 1 + (class_index % 5),
                    "INDHHIN2": 1 + (repeat % 4),
                    "DMDEDUC2": 1 + (class_index % 5),
                    "DMDMARTL": 1 + (repeat % 6),
                }
            )
            questionnaire_records.append({"SEQN": seqn, **dict(zip(PHQ_COLUMNS, phq_values))})

    seqn += 1
    demographic_records.append(
        {
            "SEQN": seqn,
            "RIDAGEYR": 40,
            "RIAGENDR": 1,
            "RIDRETH1": 3,
            "INDHHIN2": 2,
            "DMDEDUC2": 4,
            "DMDMARTL": 1,
        }
    )
    questionnaire_records.append({"SEQN": seqn, **dict(zip(PHQ_COLUMNS, [9, 0, 0, 0, 0, 0, 0, 0, 0]))})

    demographic_path = raw_dir / "demographic.csv"
    questionnaire_path = raw_dir / "questionnaire.csv"
    pd.DataFrame(demographic_records).to_csv(demographic_path, index=False)
    pd.DataFrame(questionnaire_records).to_csv(questionnaire_path, index=False)

    return {
        "raw_dir": raw_dir,
        "demographic_path": demographic_path,
        "questionnaire_path": questionnaire_path,
    }


@pytest.fixture()
def ml_artifacts(tmp_path: Path, raw_nhanes_files: dict[str, Path]) -> dict[str, Path]:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "models"
    fairness_dir = tmp_path / "fairness_reports"

    preprocess_pipeline(
        demographic_path=raw_nhanes_files["demographic_path"],
        questionnaire_path=raw_nhanes_files["questionnaire_path"],
        output_dir=processed_dir,
        test_size=0.25,
        random_state=42,
    )
    train_and_save(
        train_path=processed_dir / "train.csv",
        test_path=processed_dir / "test.csv",
        output_dir=model_dir,
        random_state=42,
    )
    generate_fairness_report(
        test_raw_path=processed_dir / "test_raw.csv",
        preprocessor_path=processed_dir / "preprocessor.joblib",
        model_path=model_dir / "model.joblib",
        output_path=fairness_dir / "fairness_report.csv",
    )

    return {
        "processed_dir": processed_dir,
        "model_dir": model_dir,
        "fairness_dir": fairness_dir,
        "model_path": model_dir / "model.joblib",
        "preprocessor_path": processed_dir / "preprocessor.joblib",
        "test_raw_path": processed_dir / "test_raw.csv",
        "fairness_report_path": fairness_dir / "fairness_report.csv",
    }
