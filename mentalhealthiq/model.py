"""
Phase 3: Model Training and Evaluation Module

Trains baseline (Logistic Regression, Random Forest) and final (XGBoost) models
with SMOTE for class imbalance handling. Compares models and automatically
selects the best performer. Saves model and preprocessor for inference.

Classes:
    - DepthModel: Wrapper for depression severity prediction model

Functions:
    - train_logistic_regression: Baseline logistic regression
    - train_random_forest: Baseline random forest classifier
    - train_xgboost: Final XGBoost classifier
    - apply_smote: SMOTE oversampling for training data only
    - compare_models: Compare model performance across metrics
    - train_and_save: Full training pipeline
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthModel:
    """
    Wrapper for depression severity prediction model.

    Encapsulates the trained classifier, label encoder, and provides
    unified inference and evaluation interface.

    Attributes:
        model: Fitted sklearn classifier (XGBoost, RandomForest, or LogisticRegression)
        label_encoder: LabelEncoder for severity labels
        feature_names: Names of input features
        model_type: Type of model ('xgboost', 'rf', or 'lr')
        metrics: Dictionary of evaluation metrics
    """

    def __init__(self, model: Any, label_encoder: LabelEncoder, feature_names: list, model_type: str = 'xgboost'):
        """
        Initialize DepthModel.

        Args:
            model: Fitted sklearn classifier
            label_encoder: Fitted LabelEncoder for target classes
            feature_names: List of feature names used in training
            model_type: Type of model for logging
        """
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.model_type = model_type
        self.metrics: Dict[str, float] = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict depression severity.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Predicted severity labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Probability estimates for each class
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"Model {self.model_type} does not support predict_proba")
        return self.model.predict_proba(X)

    def save(self, model_path: Path) -> None:
        """
        Save model assets to disk.

        Args:
            model_path: Path to save model archive
        """
        asset = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics
        }
        joblib.dump(asset, model_path)
        logger.info(f"Model saved to {model_path}")

    @staticmethod
    def load(model_path: Path) -> 'DepthModel':
        """
        Load model assets from disk.

        Args:
            model_path: Path to saved model archive

        Returns:
            Loaded DepthModel instance
        """
        asset = joblib.load(model_path)
        depth_model = DepthModel(
            model=asset['model'],
            label_encoder=asset['label_encoder'],
            feature_names=asset['feature_names'],
            model_type=asset.get('model_type', 'unknown')
        )
        depth_model.metrics = asset.get('metrics', {})
        logger.info(f"Model loaded from {model_path}")
        return depth_model


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to training data only.

    SMOTE creates synthetic minority class samples to balance class distribution.
    Applied only to training data to prevent data leakage.

    Args:
        X_train: Training features
        y_train: Training labels (numeric)
        random_state: Random seed

    Returns:
        Tuple of (X_train_resampled, y_train_resampled)
    """
    logger.info("Applying SMOTE oversampling to training data...")

    class_dist_before = pd.Series(y_train).value_counts().sort_index()
    logger.debug(f"Before SMOTE:\n{class_dist_before}")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    class_dist_after = pd.Series(y_train_resampled).value_counts().sort_index()
    logger.info(f"After SMOTE:\n{class_dist_after}")
    logger.info(f"Training samples: {len(X_train)} → {len(X_train_resampled)}")

    return X_train_resampled, y_train_resampled


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression baseline model.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained LogisticRegression model
    """
    logger.info("\nTraining Logistic Regression (baseline)...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    logger.info("✓ Logistic Regression trained")

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest baseline model.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("\nTraining Random Forest (baseline)...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    logger.info("✓ Random Forest trained")

    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> XGBClassifier:
    """
    Train XGBoost final model.

    XGBoost provides superior performance for tabular data with proper
    handling of feature importance, regularization, and class imbalance.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained XGBClassifier model
    """
    logger.info("\nTraining XGBoost (final model)...")

    # Calculate class weights for imbalanced data
    unique_classes = np.unique(y_train)
    class_weights = len(y_train) / (len(unique_classes) * np.bincount(y_train))

    model = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        min_child_weight=1,
        gamma=0.5,  # Minimum loss reduction for split
        objective='multi:softmax',
        num_class=len(unique_classes),
        random_state=random_state,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbose=0
    )

    model.fit(X_train, y_train)
    logger.info("✓ XGBoost trained")

    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Computes accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging

    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[str, Dict[str, float]]:
    """
    Compare multiple models and select best performer.

    Selection based on F1-score (balanced metric for imbalanced data).

    Args:
        models: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (best_model_name, all_metrics_dict)
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    all_metrics = {}
    best_f1 = -1
    best_model_name = None

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics[name] = metrics

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name

    # Summary table
    logger.info("\n" + "-" * 70)
    logger.info("SUMMARY (by F1-Score):")
    logger.info("-" * 70)

    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {name:20s} | Acc: {metrics['accuracy']:.4f} | "
                   f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
                   f"F1: {metrics['f1']:.4f}")

    logger.info("=" * 70)
    logger.info(f"✓ BEST MODEL: {best_model_name} (F1: {best_f1:.4f})")
    logger.info("=" * 70)

    return best_model_name, all_metrics


def train_and_save(
    train_path: Path = Path('data/processed/train.csv'),
    test_path: Path = Path('data/processed/test.csv'),
    preprocessor_path: Path = Path('data/processed/preprocessor.joblib'),
    output_dir: Path = Path('data/models'),
    random_state: int = 42
) -> Tuple[DepthModel, Dict[str, float]]:
    """
    Full training pipeline: load → SMOTE → train multiple models → compare → save.

    Args:
        train_path: Path to processed training data
        test_path: Path to processed test data
        preprocessor_path: Path to fitted preprocessor
        output_dir: Directory to save trained model
        random_state: Random seed

    Returns:
        Tuple of (best_depth_model, all_metrics)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("MODEL TRAINING PIPELINE START")
    logger.info("=" * 70)

    # Step 1: Load preprocessed data
    logger.info("\n[STEP 1] Loading processed data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('SEVERITY', axis=1).values
    y_train_labels = train_df['SEVERITY'].values

    X_test = test_df.drop('SEVERITY', axis=1).values
    y_test_labels = test_df['SEVERITY'].values

    # Step 2: Encode labels
    logger.info("\n[STEP 2] Encoding severity labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    classes = label_encoder.classes_
    logger.info(f"Classes: {list(classes)}")

    # Step 3: Apply SMOTE (training only)
    logger.info("\n[STEP 3] Applying SMOTE to training data...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state)

    # Step 4: Train models
    logger.info("\n[STEP 4] Training models...")
    models = {
        'Logistic Regression': train_logistic_regression(X_train_resampled, y_train_resampled, random_state),
        'Random Forest': train_random_forest(X_train_resampled, y_train_resampled, random_state),
        'XGBoost': train_xgboost(X_train_resampled, y_train_resampled, random_state),
    }

    # Step 5: Compare models
    logger.info("\n[STEP 5] Comparing models...")
    best_model_name, all_metrics = compare_models(models, X_test, y_test)

    # Step 6: Wrap best model
    best_model = models[best_model_name]
    feature_names = train_df.drop('SEVERITY', axis=1).columns.tolist()

    depth_model = DepthModel(
        model=best_model,
        label_encoder=label_encoder,
        feature_names=feature_names,
        model_type=best_model_name.lower()
    )
    depth_model.metrics = all_metrics[best_model_name]

    # Step 7: Save model
    logger.info("\n[STEP 7] Saving model...")
    model_path = output_dir / 'model.joblib'
    depth_model.save(model_path)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test Accuracy: {depth_model.metrics['accuracy']:.4f}")
    logger.info(f"Test F1-Score: {depth_model.metrics['f1']:.4f}")
    logger.info(f"Model saved to: {model_path}")

    return depth_model, all_metrics


def main():
    """Run model training pipeline."""
    try:
        depth_model, all_metrics = train_and_save()

        print("\n" + "=" * 70)
        print("NEXT STEP: Phase 4 - Fairness Analysis (fairness.py)")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
