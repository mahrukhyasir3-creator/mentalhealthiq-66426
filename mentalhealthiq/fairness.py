"""Generate fairness reports from real MentalHealthIQ artifacts."""

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


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_labels: np.ndarray,
    group_values: pd.Series,
    group_column: str,
    labels: list[int],
) -> pd.DataFrame:
    """Calculate requested fairness metrics for one grouping column."""

    rows = []
    for group_value in sorted(group_values.dropna().unique(), key=lambda value: str(value)):
        mask = group_values == group_value
        y_true_group = y_true[mask.to_numpy()]
        y_pred_group = y_pred[mask.to_numpy()]
        y_pred_labels_group = y_pred_labels[mask.to_numpy()]

        if len(y_true_group) == 0:
            continue

        false_positive_rate, false_negative_rate = _multiclass_error_rates(
            y_true_group,
            y_pred_group,
            labels,
        )

        selection_rate = _selection_rate(y_pred_labels_group)
        rows.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "sample_size": int(len(y_true_group)),
                "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                "precision": float(
                    precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)
                ),
                "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "selection_rate": selection_rate,
                "risk_percentage": float(selection_rate * 100),
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
    y_pred_labels = depth_model.label_encoder.inverse_transform(y_pred)
    labels = list(range(len(depth_model.classes_)))

    reports = [
        calculate_group_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_labels=y_pred_labels,
            group_values=raw_test_df[group],
            group_column=group,
            labels=labels,
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
