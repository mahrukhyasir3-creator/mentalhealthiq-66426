from pathlib import Path

import pandas as pd

from mentalhealthiq.fairness import evaluate_fairness, generate_fairness_report


def test_fairness_report_can_be_generated_from_test_artifacts(ml_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output_path = tmp_path / "fairness_report.csv"

    report_df = generate_fairness_report(
        test_raw_path=ml_artifacts["test_raw_path"],
        preprocessor_path=ml_artifacts["preprocessor_path"],
        model_path=ml_artifacts["model_path"],
        output_path=output_path,
    )

    assert output_path.exists()
    assert set(report_df["group_column"]) == {"RIAGENDR", "AGE_GROUP", "RIDRETH1", "INDHHIN2"}
    assert {
        "group_column",
        "group_value",
        "sample_size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "selection_rate",
    }.issubset(report_df.columns)


def test_evaluate_fairness_returns_grouped_reports(ml_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "fairness_reports"
    grouped_reports, disparities = evaluate_fairness(
        model_path=ml_artifacts["model_path"],
        raw_test_path=ml_artifacts["test_raw_path"],
        preprocessor_path=ml_artifacts["preprocessor_path"],
        output_dir=output_dir,
    )

    assert output_dir.joinpath("fairness_report.csv").exists()
    assert set(grouped_reports) == {"RIAGENDR", "AGE_GROUP", "RIDRETH1", "INDHHIN2"}
    assert isinstance(disparities, dict)
    assert all(isinstance(group_df, pd.DataFrame) for group_df in grouped_reports.values())
