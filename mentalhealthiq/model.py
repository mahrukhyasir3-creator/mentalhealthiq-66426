"""Train and load MentalHealthIQ depression severity models."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from mentalhealthiq.preprocess import PROJECT_ROOT, TARGET_COLUMN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"


class DepthModel:
    """Wrapper around a trained classifier and its label encoder."""

    def __init__(
        self,
        classifier: Any,
        label_encoder: LabelEncoder,
        feature_names: list[str],
        model_type: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.classifier = classifier
        self.model = classifier
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.model_type = model_type
        self.metrics = metrics or {}

    @property
    def classes_(self) -> np.ndarray:
        """Return the original severity class labels."""

        return self.label_encoder.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict encoded severity labels."""

        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        if not hasattr(self.classifier, "predict_proba"):
            raise NotImplementedError(f"{self.model_type} does not support predict_proba.")
        return self.classifier.predict_proba(X)

    def save(self, model_path: Path) -> None:
        """Save the wrapped model."""

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "metrics": self.metrics,
            },
            model_path,
        )
        logger.info("Model saved to %s", model_path)

    @staticmethod
    def load(model_path: Path) -> "DepthModel":
        """Load a saved DepthModel wrapper."""

        asset = joblib.load(model_path)
        classifier = asset.get("classifier", asset.get("model"))
        if classifier is None:
            raise ValueError(f"Model artifact does not contain a classifier: {model_path}")
        return DepthModel(
            classifier=classifier,
            label_encoder=asset["label_encoder"],
            feature_names=asset.get("feature_names", []),
            model_type=asset.get("model_type", "unknown"),
            metrics=asset.get("metrics", {}),
        )


def _load_xgb_classifier_class() -> Optional[Any]:
    """Return XGBClassifier if xgboost is installed, otherwise None."""

    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError:
        logger.warning("xgboost is not installed; skipping XGBoost training.")
        return None
    return XGBClassifier


def _apply_smote_if_available(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE after train/test split when imbalanced-learn is available."""

    try:
        from imblearn.over_sampling import SMOTE
    except ModuleNotFoundError:
        logger.warning("imbalanced-learn is not installed; training without SMOTE.")
        return X_train, y_train

    _, counts = np.unique(y_train, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) else 0
    if min_class_count < 2:
        logger.warning("Skipping SMOTE because at least one class has fewer than 2 training rows.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_class_count - 1))
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info("SMOTE resampled training rows from %s to %s.", len(X_train), len(X_resampled))
    return X_resampled, y_resampled


def _load_processed_data(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load transformed train/test CSV artifacts."""

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed train/test files are missing. Run `python -m mentalhealthiq.preprocess` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for path, df in [(train_path, train_df), (test_path, test_df)]:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{path} is missing {TARGET_COLUMN}.")

    feature_names = [column for column in train_df.columns if column != TARGET_COLUMN]
    X_train = train_df[feature_names]
    y_train = train_df[TARGET_COLUMN].astype(str)
    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COLUMN].astype(str)
    return X_train, X_test, y_train, y_test, feature_names


def _train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> LogisticRegression:
    """Train the logistic regression baseline."""

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def _train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> RandomForestClassifier:
    """Train the random forest baseline."""

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    random_state: int,
) -> Optional[Any]:
    """Train XGBoost when the dependency is installed."""

    XGBClassifier = _load_xgb_classifier_class()
    if XGBClassifier is None:
        return None

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _evaluate_classifier(
    classifier: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
    train_size: int,
) -> Dict[str, Any]:
    """Compute the required evaluation metrics for one classifier."""

    y_pred = classifier.predict(X_test)
    class_labels = label_encoder.classes_.tolist()
    encoded_labels = list(range(len(class_labels)))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=encoded_labels,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=encoded_labels).tolist(),
        "best_model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": int(train_size),
        "test_size": int(len(y_test)),
    }


def train_and_save(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    output_dir: Path = MODELS_DIR,
    random_state: int = 42,
) -> Tuple[DepthModel, Dict[str, Dict[str, Any]]]:
    """Train candidate models, choose the best weighted F1, and save DepthModel."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_test_df, y_train_labels, y_test_labels, feature_names = _load_processed_data(
        Path(train_path),
        Path(test_path),
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    X_train_resampled, y_train_resampled = _apply_smote_if_available(
        X_train,
        y_train,
        random_state,
    )

    candidates: Dict[str, Any] = {
        "Logistic Regression": _train_logistic_regression(
            X_train_resampled,
            y_train_resampled,
            random_state,
        ),
        "Random Forest": _train_random_forest(
            X_train_resampled,
            y_train_resampled,
            random_state,
        ),
    }

    xgboost_model = _train_xgboost(
        X_train_resampled,
        y_train_resampled,
        len(label_encoder.classes_),
        random_state,
    )
    if xgboost_model is not None:
        candidates["XGBoost"] = xgboost_model

    all_metrics = {
        name: _evaluate_classifier(
            classifier=model,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            model_name=name,
            train_size=len(X_train_resampled),
        )
        for name, model in candidates.items()
    }

    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]["f1"])
    best_model = candidates[best_model_name]
    best_metrics = all_metrics[best_model_name]
    best_metrics["best_model_name"] = best_model_name

    depth_model = DepthModel(
        classifier=best_model,
        label_encoder=label_encoder,
        feature_names=feature_names,
        model_type=best_model_name,
        metrics=best_metrics,
    )
    depth_model.save(output_dir / "model.joblib")
    logger.info("Best model: %s (weighted F1 %.4f)", best_model_name, best_metrics["f1"])
    return depth_model, all_metrics


def main() -> int:
    """Run model training from the command line."""

    try:
        train_and_save()
        print("Model training complete. Next: python -m mentalhealthiq.fairness")
        return 0
    except Exception as exc:
        logger.error("Model training failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
