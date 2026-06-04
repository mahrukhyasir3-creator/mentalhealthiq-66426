"""Generate fairness reports from real MentalHealthIQ artifacts.

This is a PHQ-9 screening fairness analysis. Since severity is derived from
PHQ-9 responses, it is not a complete clinical bias audit or diagnosis review.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from mentalhealthiq.model import DepthModel, MODEL_PATH
from mentalhealthiq.preprocess import (
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    DepthPreprocessor,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TEST_RAW_PATH = PROCESSED_DIR / "test_raw.csv"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
FAIRNESS_DIR = PROJECT_ROOT / "data" / "fairness_reports"
FAIRNESS_REPORT_PATH = FAIRNESS_DIR / "fairness_report.csv"
FAIRNESS_GROUPS = ["RIAGENDR", "AGE_GROUP", "RIDRETH1", "INDHHIN2"]
MIN_GROUP_SIZE = 20

LOW_RISK_LABELS = {"Minimal", "Mild"}
HIGH_RISK_LABELS = {"Moderate", "Moderately Severe", "Severe"}

GROUP_LABELS = {
    "RIAGENDR": {
        1: "Male",
        2: "Female",
    },
    "RIDRETH1": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race / Multi-Racial",
    },
    "INDHHIN2": {
        1: "$0 to $4,999",
        2: "$5,000 to $9,999",
        3: "$10,000 to $14,999",
        4: "$15,000 to $19,999",
        5: "$20,000 to $24,999",
        6: "$25,000 to $34,999",
        7: "$35,000 to $44,999",
        8: "$45,000 to $54,999",
        9: "$55,000 to $64,999",
        10: "$65,000 to $74,999",
        12: "$20,000 and over",
        13: "Under $20,000",
        14: "$75,000 to $99,999",
        15: "$100,000 and over",
        77: "Refused",
        99: "Unknown",
    },
}


def _multiclass_error_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> Tuple[float, float]:
    """Calculate aggregate multiclass false-positive and false-negative rates."""

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    false_positive_rate = float(fp.sum() / (fp.sum() + tn.sum())) if (fp.sum() + tn.sum()) else 0.0
    false_negative_rate = float(fn.sum() / (fn.sum() + tp.sum())) if (fn.sum() + tp.sum()) else 0.0
    return false_positive_rate, false_negative_rate


def _selection_rate(y_pred_labels: np.ndarray) -> float:
    """Selection rate is the share predicted above Minimal severity."""

    if len(y_pred_labels) == 0:
        return 0.0
    return float((y_pred_labels != "Minimal").sum() / len(y_pred_labels))


def _high_risk_mask(labels: np.ndarray) -> np.ndarray:
    """Return True for Moderate or higher PHQ-9 severity."""

    return np.isin(labels.astype(str), list(HIGH_RISK_LABELS))


def _binary_high_risk_metrics(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> Tuple[float, float]:
    """Calculate high-risk false-negative and selection rates."""

    if len(y_true_labels) == 0:
        return 0.0, 0.0

    y_true_high = _high_risk_mask(y_true_labels)
    y_pred_high = _high_risk_mask(y_pred_labels)
    false_negative_count = int((y_true_high & ~y_pred_high).sum())
    actual_high_count = int(y_true_high.sum())
    high_risk_false_negative_rate = (
        float(false_negative_count / actual_high_count)
        if actual_high_count
        else 0.0
    )
    high_risk_selection_rate = float(y_pred_high.sum() / len(y_pred_high))
    return high_risk_false_negative_rate, high_risk_selection_rate


def _group_label(group_column: str, group_value: object) -> str:
    """Return a readable demographic label while keeping unknown values safe."""

    if pd.isna(group_value):
        return "Missing"
    if group_column == "AGE_GROUP":
        return str(group_value)

    try:
        lookup_value = int(float(group_value))
    except (TypeError, ValueError):
        lookup_value = group_value

    return GROUP_LABELS.get(group_column, {}).get(lookup_value, f"Unknown / Other ({group_value})")


def _fairness_flag(sample_size: int, false_negative_rate_gap: float, f1_gap: float, accuracy_gap: float) -> str:
    """Classify a group-level fairness finding for review."""

    if sample_size < MIN_GROUP_SIZE:
        return "Low Sample Size"
    if false_negative_rate_gap > 0.10:
        return "Warning"
    if f1_gap < -0.10 or accuracy_gap < -0.10:
        return "Review"
    return "OK"


def _fairness_note(flag: str, group_label: str, sample_size: int) -> str:
    """Return a short human-readable explanation for a fairness row."""

    if flag == "Low Sample Size":
        return f"{group_label} has fewer than {MIN_GROUP_SIZE} test rows; interpret with caution."
    if flag == "Warning":
        return f"{group_label} has a higher false-negative gap than the overall test set."
    if flag == "Review":
        return f"{group_label} has lower model performance than the overall test set."
    return f"{group_label} is within the configured review thresholds."


def _overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    labels: list[int],
) -> Dict[str, float]:
    """Calculate overall metrics used as group comparison baselines."""

    _, false_negative_rate = _multiclass_error_rates(y_true, y_pred, labels)
    high_risk_false_negative_rate, high_risk_selection_rate = _binary_high_risk_metrics(
        y_true_labels,
        y_pred_labels,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "false_negative_rate": false_negative_rate,
        "selection_rate": _selection_rate(y_pred_labels),
        "high_risk_false_negative_rate": high_risk_false_negative_rate,
        "high_risk_selection_rate": high_risk_selection_rate,
    }


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    group_values: pd.Series,
    group_column: str,
    labels: list[int],
    overall: Dict[str, float],
) -> pd.DataFrame:
    """Calculate requested fairness metrics for one grouping column."""

    rows = []
    for group_value in sorted(group_values.dropna().unique(), key=lambda value: str(value)):
        mask = group_values == group_value
        y_true_group = y_true[mask.to_numpy()]
        y_pred_group = y_pred[mask.to_numpy()]
        y_true_labels_group = y_true_labels[mask.to_numpy()]
        y_pred_labels_group = y_pred_labels[mask.to_numpy()]

        if len(y_true_group) == 0:
            continue

        false_positive_rate, false_negative_rate = _multiclass_error_rates(
            y_true_group,
            y_pred_group,
            labels,
        )

        selection_rate = _selection_rate(y_pred_labels_group)
        high_risk_false_negative_rate, high_risk_selection_rate = _binary_high_risk_metrics(
            y_true_labels_group,
            y_pred_labels_group,
        )
        accuracy = float(accuracy_score(y_true_group, y_pred_group))
        f1 = float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0))
        accuracy_gap = accuracy - overall["accuracy"]
        f1_gap = f1 - overall["f1"]
        false_negative_rate_gap = false_negative_rate - overall["false_negative_rate"]
        selection_rate_gap = selection_rate - overall["selection_rate"]
        high_risk_false_negative_rate_gap = (
            high_risk_false_negative_rate - overall["high_risk_false_negative_rate"]
        )
        high_risk_selection_rate_gap = high_risk_selection_rate - overall["high_risk_selection_rate"]
        sample_size = int(len(y_true_group))
        group_label = _group_label(group_column, group_value)
        fairness_flag = _fairness_flag(sample_size, false_negative_rate_gap, f1_gap, accuracy_gap)

        rows.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "group_label": group_label,
                "sample_size": sample_size,
                "min_group_size": MIN_GROUP_SIZE,
                "accuracy": accuracy,
                "precision": float(
                    precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)
                ),
                "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                "f1": f1,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "selection_rate": selection_rate,
                "risk_percentage": float(selection_rate * 100),
                "high_risk_false_negative_rate": high_risk_false_negative_rate,
                "high_risk_selection_rate": high_risk_selection_rate,
                "overall_accuracy": overall["accuracy"],
                "overall_f1": overall["f1"],
                "accuracy_gap": accuracy_gap,
                "f1_gap": f1_gap,
                "false_negative_rate_gap": false_negative_rate_gap,
                "selection_rate_gap": selection_rate_gap,
                "high_risk_false_negative_rate_gap": high_risk_false_negative_rate_gap,
                "high_risk_selection_rate_gap": high_risk_selection_rate_gap,
                "fairness_flag": fairness_flag,
                "notes": _fairness_note(fairness_flag, group_label, sample_size),
            }
        )

    return pd.DataFrame(rows)


def generate_fairness_report(
    test_raw_path: Path = TEST_RAW_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    model_path: Path = MODEL_PATH,
    output_path: Path = FAIRNESS_REPORT_PATH,
) -> pd.DataFrame:
    """Generate and save the fairness report CSV."""

    for path in [test_raw_path, preprocessor_path, model_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required fairness artifact is missing: {path}")

    raw_test_df = pd.read_csv(test_raw_path)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN, *FAIRNESS_GROUPS] if column not in raw_test_df]
    if missing_columns:
        raise ValueError(f"test_raw.csv is missing required columns: {missing_columns}")

    preprocessor = DepthPreprocessor.load(preprocessor_path)
    depth_model = DepthModel.load(model_path)

    X_test = preprocessor.transform(raw_test_df[FEATURE_COLUMNS])
    y_true = depth_model.label_encoder.transform(raw_test_df[TARGET_COLUMN].astype(str))
    y_pred = depth_model.predict(X_test)
    y_true_labels = raw_test_df[TARGET_COLUMN].astype(str).to_numpy()
    y_pred_labels = depth_model.label_encoder.inverse_transform(y_pred)
    labels = list(range(len(depth_model.classes_)))
    overall = _overall_metrics(y_true, y_pred, y_true_labels, y_pred_labels, labels)

    reports = [
        calculate_group_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_true_labels=y_true_labels,
            y_pred_labels=y_pred_labels,
            group_values=raw_test_df[group],
            group_column=group,
            labels=labels,
            overall=overall,
        )
        for group in FAIRNESS_GROUPS
    ]

    report_df = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    numeric_columns = report_df.select_dtypes(include="number").columns
    report_df[numeric_columns] = report_df[numeric_columns].round(4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    logger.info("Fairness report saved to %s", output_path)
    return report_df


def evaluate_fairness(
    model_path: Path = MODEL_PATH,
    raw_test_path: Path = TEST_RAW_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    output_dir: Path = FAIRNESS_DIR,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """Compatibility wrapper returning grouped reports and simple disparities."""

    report_path = Path(output_dir) / "fairness_report.csv"
    report_df = generate_fairness_report(
        test_raw_path=raw_test_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        output_path=report_path,
    )

    grouped_reports = {
        group: report_df[report_df["group_column"] == group].reset_index(drop=True)
        for group in FAIRNESS_GROUPS
    }
    disparities: Dict[str, Dict[str, float]] = {}
    for group, group_df in grouped_reports.items():
        if len(group_df) < 2:
            continue
        disparities[group] = {
            "accuracy_disparity": float(group_df["accuracy"].max() - group_df["accuracy"].min()),
            "fpr_disparity": float(
                group_df["false_positive_rate"].max() - group_df["false_positive_rate"].min()
            ),
            "fnr_disparity": float(
                group_df["false_negative_rate"].max() - group_df["false_negative_rate"].min()
            ),
            "selection_rate_disparity": float(group_df["selection_rate"].max() - group_df["selection_rate"].min()),
        }
    return grouped_reports, disparities


def main() -> int:
    """Run fairness report generation from the command line."""

    try:
        generate_fairness_report()
        print("Fairness report complete. Next: python -m uvicorn mentalhealthiq.api:app --reload --port 8000")
        return 0
    except Exception as exc:
        logger.error("Fairness report generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
