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
        "group_label",
        "sample_size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "selection_rate",
        "risk_percentage",
        "high_risk_false_negative_rate",
        "high_risk_selection_rate",
        "overall_accuracy",
        "overall_f1",
        "accuracy_gap",
        "f1_gap",
        "false_negative_rate_gap",
        "selection_rate_gap",
        "fairness_flag",
        "notes",
    }.issubset(report_df.columns)


def test_fairness_report_flags_low_sample_size(ml_artifacts: dict[str, Path], tmp_path: Path) -> None:
    report_df = generate_fairness_report(
        test_raw_path=ml_artifacts["test_raw_path"],
        preprocessor_path=ml_artifacts["preprocessor_path"],
        model_path=ml_artifacts["model_path"],
        output_path=tmp_path / "fairness_report.csv",
    )

    assert "Low Sample Size" in set(report_df["fairness_flag"])
    low_sample_rows = report_df[report_df["sample_size"] < report_df["min_group_size"]]
    assert not low_sample_rows.empty
    assert set(low_sample_rows["fairness_flag"]) == {"Low Sample Size"}


def test_fairness_report_handles_unknown_demographic_values(
    ml_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    raw_test_df = pd.read_csv(ml_artifacts["test_raw_path"])
    raw_test_df.loc[raw_test_df.index[0], "RIAGENDR"] = 9
    raw_test_df.loc[raw_test_df.index[1], "RIDRETH1"] = 99
    raw_test_df.loc[raw_test_df.index[2], "INDHHIN2"] = 123
    raw_test_df.loc[raw_test_df.index[3], "AGE_GROUP"] = pd.NA

    modified_test_path = tmp_path / "test_raw_unknown.csv"
    raw_test_df.to_csv(modified_test_path, index=False)

    report_df = generate_fairness_report(
        test_raw_path=modified_test_path,
        preprocessor_path=ml_artifacts["preprocessor_path"],
        model_path=ml_artifacts["model_path"],
        output_path=tmp_path / "fairness_report.csv",
    )

    assert not report_df.empty
    assert report_df["group_label"].fillna("").str.len().gt(0).all()
    assert report_df["group_label"].str.contains("Unknown / Other", regex=False).any()


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
