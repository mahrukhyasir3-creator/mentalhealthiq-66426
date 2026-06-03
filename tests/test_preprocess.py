from pathlib import Path

import pandas as pd

from mentalhealthiq.preprocess import PHQ_COLUMNS, preprocess_pipeline


def test_preprocess_pipeline_creates_processed_files(raw_nhanes_files: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        demographic_path=raw_nhanes_files["demographic_path"],
        questionnaire_path=raw_nhanes_files["questionnaire_path"],
        output_dir=output_dir,
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert preprocessor.feature_names

    for filename in ["train.csv", "test.csv", "train_raw.csv", "test_raw.csv", "preprocessor.joblib"]:
        assert (output_dir / filename).exists()


def test_preprocess_removes_invalid_phq_values_and_creates_labels(
    raw_nhanes_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "processed"
    preprocess_pipeline(
        demographic_path=raw_nhanes_files["demographic_path"],
        questionnaire_path=raw_nhanes_files["questionnaire_path"],
        output_dir=output_dir,
        test_size=0.25,
        random_state=42,
    )

    raw_processed = pd.concat(
        [
            pd.read_csv(output_dir / "train_raw.csv"),
            pd.read_csv(output_dir / "test_raw.csv"),
        ],
        ignore_index=True,
    )

    assert len(raw_processed) == 20
    assert not raw_processed[PHQ_COLUMNS].isin([7, 9, 77, 99]).any().any()
    assert set(raw_processed["SEVERITY"]) == {
        "Minimal",
        "Mild",
        "Moderate",
        "Moderately Severe",
        "Severe",
    }
    assert "PHQ9_TOTAL" in raw_processed.columns
