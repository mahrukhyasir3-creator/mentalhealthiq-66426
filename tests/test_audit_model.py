from pathlib import Path

from scripts.audit_model import audit_model


def test_audit_model_writes_expected_report(
    raw_nhanes_files: dict[str, Path],
    ml_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "model_audit.json"

    audit = audit_model(
        demographic_path=raw_nhanes_files["demographic_path"],
        questionnaire_path=raw_nhanes_files["questionnaire_path"],
        processed_dir=ml_artifacts["processed_dir"],
        model_path=ml_artifacts["model_path"],
        preprocessor_path=ml_artifacts["preprocessor_path"],
        fairness_report_path=ml_artifacts["fairness_report_path"],
        output_path=output_path,
    )

    assert output_path.exists()
    assert audit["raw_csv_usage"]["demographic_rows"] == 21
    assert audit["raw_csv_usage"]["questionnaire_rows"] == 21
    assert audit["raw_csv_usage"]["merged_valid_rows"] == 20
    assert audit["phq_cleaning"]["invalid_values_not_counted_in_total"] is True
    assert audit["train_test_split"]["duplicate_seqn_count"] == 0
    assert audit["features"]["leakage_caveat"]
    assert len(audit["sample_prediction_checks"]) == 6
    assert {"phq9_total", "expected_severity", "predicted_severity"}.issubset(
        audit["sample_prediction_checks"][0]
    )
