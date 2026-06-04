"""Audit MentalHealthIQ data, preprocessing, model, and fairness artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mentalhealthiq.api import (  # noqa: E402
    PredictionInput,
    calculate_phq9_total,
    create_input_dataframe,
    phq9_severity_from_total,
    recommendation_for_severity,
    risk_band_from_severity,
)
from mentalhealthiq.fairness import FAIRNESS_REPORT_PATH, FAIRNESS_GROUPS  # noqa: E402
from mentalhealthiq.model import MODEL_PATH, DepthModel  # noqa: E402
from mentalhealthiq.preprocess import (  # noqa: E402
    DEMOGRAPHIC_COLUMNS,
    DEMOGRAPHIC_PATH,
    FEATURE_COLUMNS,
    PHQ_COLUMNS,
    PREPROCESSOR_PATH,
    PROCESSED_DIR,
    QUESTIONNAIRE_COLUMNS,
    QUESTIONNAIRE_PATH,
    RAW_OUTPUT_COLUMNS,
    TARGET_COLUMN,
    DepthPreprocessor,
    clean_demographic_columns,
    create_age_groups,
    create_phq9_total,
    create_severity_labels,
    handle_missing_values,
    load_raw_data,
    validate_columns,
)


MODEL_AUDIT_PATH = PROJECT_ROOT / "data" / "models" / "model_audit.json"
PROCESSED_ARTIFACTS = {
    "train": PROCESSED_DIR / "train.csv",
    "test": PROCESSED_DIR / "test.csv",
    "train_raw": PROCESSED_DIR / "train_raw.csv",
    "test_raw": PROCESSED_DIR / "test_raw.csv",
    "preprocessor": PREPROCESSOR_PATH,
    "model": MODEL_PATH,
    "fairness_report": FAIRNESS_REPORT_PATH,
}
SEVERITY_BINS = {
    "Minimal": [0, 4],
    "Mild": [5, 9],
    "Moderate": [10, 14],
    "Moderately Severe": [15, 19],
    "Severe": [20, 27],
}
LEAKAGE_CAVEAT = (
    "Current model uses PHQ-9 answers as features while SEVERITY is derived from PHQ9_TOTAL. "
    "High accuracy is expected because the model can learn the PHQ-9 scoring rule. This is "
    "appropriate for demo screening/classification, not independent clinical prediction."
)
API_PRIMARY_SEVERITY_NOTE = (
    "API predicted_severity now follows PHQ-9 scoring boundaries as the primary screening "
    "result. model_predicted_severity is retained as the supporting trained model estimate."
)


def display_path(path: Path) -> str:
    """Return a project-relative path when possible for portable reports."""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_info(paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Return existence, modified times, sizes, and hashes for artifacts."""

    info: Dict[str, Dict[str, Any]] = {}
    for name, path in paths.items():
        path = Path(path)
        if not path.exists():
            info[name] = {"path": display_path(path), "exists": False}
            continue
        stat = path.stat()
        info[name] = {
            "path": display_path(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256_file(path),
        }
    return info


def severity_from_total(total: int) -> str:
    """Return PHQ-9 severity from a total score."""

    for label, (lower, upper) in SEVERITY_BINS.items():
        if lower <= total <= upper:
            return label
    raise ValueError(f"PHQ-9 total must be between 0 and 27. Received: {total}")


def phq_values_for_total(total: int) -> Dict[str, int]:
    """Create a valid PHQ-9 answer pattern with the requested total."""

    if not 0 <= total <= 27:
        raise ValueError(f"PHQ-9 total must be between 0 and 27. Received: {total}")
    remaining = total
    values = []
    for _ in PHQ_COLUMNS:
        value = min(3, remaining)
        values.append(value)
        remaining -= value
    return dict(zip(PHQ_COLUMNS, values))


def distribution(series: pd.Series) -> Dict[str, int]:
    """Return string-keyed value counts."""

    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def validate_required_columns(demographic_df: pd.DataFrame, questionnaire_df: pd.DataFrame) -> None:
    """Validate raw input columns with helpful errors."""

    validate_columns(demographic_df, DEMOGRAPHIC_COLUMNS, "demographic.csv")
    validate_columns(questionnaire_df, QUESTIONNAIRE_COLUMNS, "questionnaire.csv")


def build_clean_dataset(demographic_path: Path, questionnaire_path: Path) -> pd.DataFrame:
    """Run the same deterministic cleaning steps used before splitting."""

    merged_df = load_raw_data(demographic_path=demographic_path, questionnaire_path=questionnaire_path)
    cleaned_df = clean_demographic_columns(merged_df)
    cleaned_df = create_phq9_total(cleaned_df)
    cleaned_df = create_severity_labels(cleaned_df)
    cleaned_df = create_age_groups(cleaned_df)
    return handle_missing_values(cleaned_df)


def phq_cleaning_summary(questionnaire_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize valid and invalid PHQ-9 source values."""

    phq_numeric = questionnaire_df[PHQ_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_mask = phq_numeric.isin([0, 1, 2, 3])
    invalid_cells = phq_numeric.where(~valid_mask).to_numpy().ravel()
    invalid_values = sorted({str(value) for value in invalid_cells if not pd.isna(value)})
    complete_valid_rows = int(valid_mask.all(axis=1).sum())
    return {
        "valid_answer_values": [0, 1, 2, 3],
        "invalid_or_missing_cell_count": int((~valid_mask).sum().sum()),
        "complete_valid_phq_rows": complete_valid_rows,
        "observed_invalid_values": invalid_values,
        "invalid_values_not_counted_in_total": True,
    }


def sample_prediction_checks(
    preprocessor: DepthPreprocessor,
    model: DepthModel,
    totals: Iterable[int] = (0, 5, 10, 15, 20, 27),
) -> List[Dict[str, Any]]:
    """Run model predictions for deterministic PHQ-9 total examples."""

    checks = []
    for total in totals:
        payload_data = {
            "RIDAGEYR": 35,
            "RIAGENDR": 2,
            "RIDRETH1": 3,
            "INDHHIN2": 5,
            "DMDEDUC2": 4,
            "DMDMARTL": 1,
            **phq_values_for_total(total),
        }
        payload = PredictionInput(**payload_data)
        features = preprocessor.transform(create_input_dataframe(payload))
        prediction_encoded = model.predict(features)
        model_predicted_severity = str(model.label_encoder.inverse_transform(prediction_encoded)[0])
        probability_values = model.predict_proba(features)[0] if hasattr(model.classifier, "predict_proba") else []
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.label_encoder.classes_, probability_values)
        }
        manual_severity = severity_from_total(total)
        rule_based_severity = phq9_severity_from_total(total)
        model_agreement = model_predicted_severity == rule_based_severity
        risk_score = float(max(probability_values)) if len(probability_values) else 0.0
        checks.append(
            {
                "phq9_total": calculate_phq9_total(payload),
                "expected_severity": manual_severity,
                "manual_severity": manual_severity,
                "rule_based_severity": rule_based_severity,
                "predicted_severity": rule_based_severity,
                "model_predicted_severity": model_predicted_severity,
                "model_agreement": model_agreement,
                "matches_expected": rule_based_severity == manual_severity,
                "risk_band": risk_band_from_severity(rule_based_severity),
                "risk_percentage": round(risk_score * 100, 1),
                "recommendation": recommendation_for_severity(rule_based_severity),
                "probabilities": probabilities,
            }
        )
    return checks


def audit_model(
    demographic_path: Path = DEMOGRAPHIC_PATH,
    questionnaire_path: Path = QUESTIONNAIRE_PATH,
    processed_dir: Path = PROCESSED_DIR,
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    fairness_report_path: Path = FAIRNESS_REPORT_PATH,
    output_path: Path = MODEL_AUDIT_PATH,
) -> Dict[str, Any]:
    """Audit raw data, artifacts, metrics, and sample predictions."""

    demographic_path = Path(demographic_path)
    questionnaire_path = Path(questionnaire_path)
    processed_dir = Path(processed_dir)
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    fairness_report_path = Path(fairness_report_path)
    output_path = Path(output_path)

    if not demographic_path.exists() or not questionnaire_path.exists():
        raise FileNotFoundError(
            "Raw CSVs are missing. Expected data/raw/demographic.csv and data/raw/questionnaire.csv."
        )

    demographic_df = pd.read_csv(demographic_path)
    questionnaire_df = pd.read_csv(questionnaire_path)
    validate_required_columns(demographic_df, questionnaire_df)
    cleaned_df = build_clean_dataset(demographic_path, questionnaire_path)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_raw_path = processed_dir / "train_raw.csv"
    test_raw_path = processed_dir / "test_raw.csv"
    for path in [train_path, test_path, train_raw_path, test_raw_path, model_path, preprocessor_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required generated artifact is missing: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_raw_df = pd.read_csv(train_raw_path)
    test_raw_df = pd.read_csv(test_raw_path)
    model = DepthModel.load(model_path)
    preprocessor = DepthPreprocessor.load(preprocessor_path)

    duplicate_seqn_across_split = []
    if "SEQN" in train_raw_df.columns and "SEQN" in test_raw_df.columns:
        duplicate_seqn_across_split = sorted(
            set(train_raw_df["SEQN"].astype(str)).intersection(set(test_raw_df["SEQN"].astype(str)))
        )

    X_test = test_df[[column for column in test_df.columns if column != TARGET_COLUMN]].to_numpy(dtype=float)
    y_pred = model.predict(X_test)
    y_pred_labels = model.label_encoder.inverse_transform(y_pred)
    metrics = model.metrics or {}
    test_distribution = distribution(test_df[TARGET_COLUMN].astype(str))
    prediction_distribution = distribution(pd.Series(y_pred_labels))
    sample_checks = sample_prediction_checks(preprocessor, model)
    sample_mismatch_count = sum(1 for check in sample_checks if not check["model_agreement"])

    fairness_summary: Dict[str, Any] = {
        "path": display_path(fairness_report_path),
        "exists": fairness_report_path.exists(),
    }
    if fairness_report_path.exists():
        fairness_df = pd.read_csv(fairness_report_path)
        fairness_summary.update(
            {
                "row_count": int(len(fairness_df)),
                "groups": sorted(fairness_df["group_column"].dropna().astype(str).unique().tolist())
                if "group_column" in fairness_df
                else [],
                "low_sample_size_rows": int((fairness_df.get("fairness_flag") == "Low Sample Size").sum())
                if "fairness_flag" in fairness_df
                else 0,
                "required_groups_present": sorted(FAIRNESS_GROUPS),
            }
        )

    artifacts = {
        "train": train_path,
        "test": test_path,
        "train_raw": train_raw_path,
        "test_raw": test_raw_path,
        "preprocessor": preprocessor_path,
        "model": model_path,
        "fairness_report": fairness_report_path,
        "raw_demographic": demographic_path,
        "raw_questionnaire": questionnaire_path,
    }
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_csv_usage": {
            "demographic_path": display_path(demographic_path),
            "questionnaire_path": display_path(questionnaire_path),
            "demographic_rows": int(len(demographic_df)),
            "questionnaire_rows": int(len(questionnaire_df)),
            "merged_valid_rows": int(len(cleaned_df)),
            "required_demographic_columns": DEMOGRAPHIC_COLUMNS,
            "required_questionnaire_columns": QUESTIONNAIRE_COLUMNS,
        },
        "phq_cleaning": phq_cleaning_summary(questionnaire_df),
        "phq9_total_and_severity": {
            "severity_bins": SEVERITY_BINS,
            "valid_total_range": [0, 27],
            "processed_totals_match_cleaned_phq_sum": bool(
                (cleaned_df["PHQ9_TOTAL"] == cleaned_df[PHQ_COLUMNS].sum(axis=1)).all()
            ),
            "severity_distribution": distribution(cleaned_df[TARGET_COLUMN].astype(str)),
        },
        "train_test_split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_raw_rows": int(len(train_raw_df)),
            "test_raw_rows": int(len(test_raw_df)),
            "train_label_distribution": distribution(train_df[TARGET_COLUMN].astype(str)),
            "test_label_distribution": test_distribution,
            "duplicate_seqn_across_train_test": duplicate_seqn_across_split[:20],
            "duplicate_seqn_count": len(duplicate_seqn_across_split),
            "random_state": 42,
            "stratified_split_expected": True,
        },
        "features": {
            "feature_columns": FEATURE_COLUMNS,
            "raw_output_columns": RAW_OUTPUT_COLUMNS,
            "phq_columns_used_as_features": PHQ_COLUMNS,
            "leakage_caveat": LEAKAGE_CAVEAT,
            "api_primary_severity_note": API_PRIMARY_SEVERITY_NOTE,
        },
        "model": {
            "model_type": model.model_type,
            "best_model_name": metrics.get("best_model_name", model.model_type),
            "trained_at": metrics.get("trained_at"),
            "train_size": metrics.get("train_size"),
            "test_size": metrics.get("test_size"),
            "feature_names": model.feature_names,
            "metrics": metrics,
            "test_label_distribution": test_distribution,
            "prediction_label_distribution": prediction_distribution,
            "accuracy_suspiciously_high": bool(float(metrics.get("accuracy", 0.0)) >= 0.95),
            "accuracy_note": LEAKAGE_CAVEAT,
        },
        "sample_prediction_summary": {
            "boundary_sample_count": len(sample_checks),
            "model_mismatch_count": sample_mismatch_count,
            "model_agreement_count": len(sample_checks) - sample_mismatch_count,
            "api_primary_severity_note": API_PRIMARY_SEVERITY_NOTE,
        },
        "sample_prediction_checks": sample_checks,
        "fairness_report": fairness_summary,
        "artifacts": artifact_info(artifacts),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> int:
    """Run the audit and print a short summary."""

    try:
        audit = audit_model()
    except Exception as exc:
        print(f"Model audit failed: {exc}", file=sys.stderr)
        return 1

    raw = audit["raw_csv_usage"]
    split = audit["train_test_split"]
    model = audit["model"]
    print("MentalHealthIQ model audit complete.")
    print(f"- Raw demographic rows: {raw['demographic_rows']}")
    print(f"- Raw questionnaire rows: {raw['questionnaire_rows']}")
    print(f"- Valid merged PHQ-9 rows: {raw['merged_valid_rows']}")
    print(f"- Train rows: {split['train_rows']}")
    print(f"- Test rows: {split['test_rows']}")
    print(f"- Best model: {model['best_model_name']}")
    print(f"- Accuracy: {model['metrics'].get('accuracy', 'N/A')}")
    print(f"- Weighted F1: {model['metrics'].get('f1', 'N/A')}")
    print(f"- Audit report: {MODEL_AUDIT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- Caveat: {LEAKAGE_CAVEAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
