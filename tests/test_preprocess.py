from pathlib import Path

import pandas as pd
import pytest

from mentalhealthiq.preprocess import (
    PHQ_COLUMNS,
    clean_demographic_columns,
    create_age_groups,
    create_phq9_total,
    create_severity_labels,
    load_raw_data,
    preprocess_pipeline,
    validate_columns,
    validate_stratified_split,
)


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


def test_phq9_total_uses_clean_valid_values_only() -> None:
    rows = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 1],
        [7, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    df = pd.DataFrame([dict(zip(PHQ_COLUMNS, row)) for row in rows])

    scored = create_phq9_total(df)

    assert scored["PHQ9_TOTAL"].tolist()[:5] == [0, 9, 18, 27, 13]
    assert scored["PHQ9_TOTAL"].isna().tolist()[5:] == [True, True]
    assert pd.isna(scored.loc[5, "DPQ010"])
    assert pd.isna(scored.loc[6, "DPQ010"])


def test_severity_boundary_mapping_is_exact() -> None:
    totals = [0, 4, 5, 9, 10, 14, 15, 19, 20, 27]
    expected = [
        "Minimal",
        "Minimal",
        "Mild",
        "Mild",
        "Moderate",
        "Moderate",
        "Moderately Severe",
        "Moderately Severe",
        "Severe",
        "Severe",
    ]
    labeled = create_severity_labels(pd.DataFrame({"PHQ9_TOTAL": totals}))

    assert labeled["SEVERITY"].tolist() == expected


def test_age_group_boundaries_are_exact() -> None:
    df = pd.DataFrame({"RIDAGEYR": [17, 18, 29, 30, 44, 45, 59, 60]})

    grouped = create_age_groups(df)

    assert grouped["AGE_GROUP"].tolist() == [
        "Under 18",
        "18-29",
        "18-29",
        "30-44",
        "30-44",
        "45-59",
        "45-59",
        "60+",
    ]


def test_validate_columns_reports_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(pd.DataFrame({"SEQN": [1]}), ["SEQN", "RIDAGEYR"], "demographic.csv")


def test_clean_demographic_columns_nulls_nhanes_missing_codes() -> None:
    df = pd.DataFrame(
        {
            "SEQN": [1],
            "RIDAGEYR": [130],
            "RIAGENDR": [9],
            "RIDRETH1": [7],
            "INDHHIN2": [99],
            "DMDEDUC2": [9],
            "DMDMARTL": [77],
        }
    )

    cleaned = clean_demographic_columns(df)

    assert cleaned.drop(columns=["SEQN"]).isna().all().all()


def test_load_raw_data_rejects_duplicate_seqn(tmp_path: Path) -> None:
    demographic_path = tmp_path / "demographic.csv"
    questionnaire_path = tmp_path / "questionnaire.csv"

    pd.DataFrame(
        [
            {
                "SEQN": 1,
                "RIDAGEYR": 25,
                "RIAGENDR": 1,
                "RIDRETH1": 3,
                "INDHHIN2": 4,
                "DMDEDUC2": 5,
                "DMDMARTL": 1,
            },
            {
                "SEQN": 1,
                "RIDAGEYR": 30,
                "RIAGENDR": 2,
                "RIDRETH1": 4,
                "INDHHIN2": 5,
                "DMDEDUC2": 4,
                "DMDMARTL": 5,
            },
        ]
    ).to_csv(demographic_path, index=False)
    pd.DataFrame([{"SEQN": 1, **dict.fromkeys(PHQ_COLUMNS, 0)}]).to_csv(questionnaire_path, index=False)

    with pytest.raises(ValueError, match="duplicate SEQN"):
        load_raw_data(demographic_path, questionnaire_path)


def test_validate_stratified_split_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        validate_stratified_split(pd.Series(["Minimal", "Minimal"]), test_size=1.0)


def test_validate_stratified_split_rejects_too_large_test_size() -> None:
    y = pd.Series(["Minimal", "Minimal", "Mild", "Mild", "Severe", "Severe"])

    with pytest.raises(ValueError, match="test_size is too large"):
        validate_stratified_split(y, test_size=0.9)
