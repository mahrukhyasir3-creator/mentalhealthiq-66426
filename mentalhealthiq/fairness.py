"""
Phase 4: Fairness Evaluation Module

Evaluates model fairness across demographic groups (gender, age, race, income).
Computes accuracy, false positive rate (FPR), false negative rate (FNR),
and selection rate for each group. Generates fairness report CSV.

Classes:
    - FairnessEvaluator: Evaluate model fairness across groups

Functions:
    - calculate_group_metrics: Calculate metrics for a demographic group
    - generate_fairness_report: Generate comprehensive fairness report
    - export_fairness_report: Save report to CSV
    - evaluate_fairness: Full fairness evaluation pipeline
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
)
from mentalhealthiq.model import DepthModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FairnessEvaluator:
    """
    Evaluator for model fairness across demographic groups.

    Computes disparities in model performance across protected attributes
    (gender, age, race, income). Uses metrics like accuracy, FPR, FNR,
    and selection rate to identify potential bias.

    Attributes:
        demographic_groups: Dict of {group_name: group_mapping}
        fairness_report: DataFrame with fairness metrics by group
        disparities: Dict of disparities between groups
    """

    def __init__(self):
        """Initialize FairnessEvaluator."""
        self.demographic_groups: Dict[str, Dict] = {}
        self.fairness_report: Optional[pd.DataFrame] = None
        self.disparities: Dict[str, Any] = {}

        # Define group mappings
        self._define_groups()

    def _define_groups(self) -> None:
        """Define demographic group mappings."""
        self.demographic_groups = {
            'Gender': {
                1: 'Male',
                2: 'Female'
            },
            'Age_Group': {
                '18-25': 'Young Adults (18-25)',
                '26-40': 'Adults (26-40)',
                '41-55': 'Middle-aged (41-55)',
                '56-70': 'Mature Adults (56-70)',
                '71+': 'Older Adults (71+)'
            },
            'Race': {
                1: 'Mexican American',
                2: 'Other Hispanic',
                3: 'Non-Hispanic White',
                4: 'Non-Hispanic Black',
                5: 'Other Race'
            },
            'Income': {
                1: '<$20k',
                2: '$20-44.9k',
                3: '$45-74.9k',
                4: '$75k+'
            }
        }

    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_values: np.ndarray,
        group_name: str,
        group_mapping: Dict
    ) -> pd.DataFrame:
        """
        Calculate fairness metrics for each group in a demographic.

        Metrics:
            - Count: Number of samples
            - Accuracy: Correct predictions / total
            - Precision: TP / (TP + FP)
            - Recall: TP / (TP + FN)
            - F1-Score: Harmonic mean of precision and recall
            - FPR: FP / (FP + TN) - false positive rate
            - FNR: FN / (FN + TP) - false negative rate
            - Selection_Rate: Positive predictions / total

        Args:
            y_true: True labels (numeric)
            y_pred: Predicted labels (numeric)
            group_values: Group assignment for each sample
            group_name: Name of demographic (e.g., 'Gender')
            group_mapping: Dict mapping group codes to names

        Returns:
            DataFrame with metrics by group
        """
        results = []
        unique_groups = np.unique(group_values)

        logger.info(f"\nCalculating metrics for {group_name}...")

        for group_val in sorted(unique_groups):
            mask = group_values == group_val
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            if len(y_true_group) == 0:
                continue

            # Get group name
            if isinstance(group_val, str):
                group_label = group_mapping.get(group_val, str(group_val))
            else:
                group_label = group_mapping.get(group_val, str(group_val))

            # Basic metrics
            accuracy = accuracy_score(y_true_group, y_pred_group)
            precision = precision_score(y_true_group, y_pred_group, average='weighted', zero_division=0)
            recall = recall_score(y_true_group, y_pred_group, average='weighted', zero_division=0)
            f1 = f1_score(y_true_group, y_pred_group, average='weighted', zero_division=0)

            # For multiclass, compute aggregate FPR and FNR from confusion matrix
            cm = confusion_matrix(y_true_group, y_pred_group)
            fp_total = cm.sum(axis=0) - np.diag(cm)
            fn_total = cm.sum(axis=1) - np.diag(cm)
            tn_total = cm.sum() - (fp_total.sum() + fn_total.sum() + np.diag(cm).sum())

            fpr = fp_total.sum() / (fp_total.sum() + tn_total) if (fp_total.sum() + tn_total) > 0 else 0
            fnr = fn_total.sum() / (fn_total.sum() + np.diag(cm).sum()) if (fn_total.sum() + np.diag(cm).sum()) > 0 else 0

            # Selection rate (proportion of positive predictions)
            selection_rate = (y_pred_group > 0).sum() / len(y_pred_group)

            results.append({
                f'{group_name}': group_label,
                'Count': len(y_true_group),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'FPR': fpr,
                'FNR': fnr,
                'Selection_Rate': selection_rate
            })

            logger.debug(f"  {group_label}: n={len(y_true_group)}, acc={accuracy:.4f}, "
                        f"fpr={fpr:.4f}, fnr={fnr:.4f}")

        df_group = pd.DataFrame(results)
        logger.info(f"  ✓ {group_name} metrics calculated for {len(results)} groups")

        return df_group

    def evaluate_all_groups(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X_features: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate fairness across all demographic groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            X_features: Feature dataframe with demographic columns

        Returns:
            Dictionary of {group_name: metrics_dataframe}
        """
        all_metrics = {}

        # Gender
        if 'RIAGENDR' in X_features.columns:
            metrics_gender = self.calculate_group_metrics(
                y_true, y_pred, X_features['RIAGENDR'].values,
                'Gender', self.demographic_groups['Gender']
            )
            all_metrics['Gender'] = metrics_gender

        # Age Group
        if 'AGE_GROUP' in X_features.columns:
            metrics_age = self.calculate_group_metrics(
                y_true, y_pred, X_features['AGE_GROUP'].values,
                'Age_Group', self.demographic_groups['Age_Group']
            )
            all_metrics['Age_Group'] = metrics_age

        # Race
        if 'RIDRETH1' in X_features.columns:
            metrics_race = self.calculate_group_metrics(
                y_true, y_pred, X_features['RIDRETH1'].values,
                'Race', self.demographic_groups['Race']
            )
            all_metrics['Race'] = metrics_race

        # Income
        if 'INDHHIN2' in X_features.columns:
            metrics_income = self.calculate_group_metrics(
                y_true, y_pred, X_features['INDHHIN2'].values,
                'Income', self.demographic_groups['Income']
            )
            all_metrics['Income'] = metrics_income

        return all_metrics

    def calculate_disparities(
        self,
        all_metrics: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Calculate performance disparities between groups.

        Disparity metric: max_metric - min_metric for each demographic.
        High disparities indicate potential bias.

        Args:
            all_metrics: Dictionary of metrics by demographic group

        Returns:
            Dictionary of disparities
        """
        disparities = {}

        for demographic, metrics_df in all_metrics.items():
            if len(metrics_df) < 2:
                continue

            disp = {
                'Accuracy_Disparity': metrics_df['Accuracy'].max() - metrics_df['Accuracy'].min(),
                'FPR_Disparity': metrics_df['FPR'].max() - metrics_df['FPR'].min(),
                'FNR_Disparity': metrics_df['FNR'].max() - metrics_df['FNR'].min(),
                'Selection_Rate_Disparity': metrics_df['Selection_Rate'].max() - metrics_df['Selection_Rate'].min(),
            }

            disparities[demographic] = disp

            logger.info(f"\n{demographic} Disparities:")
            logger.info(f"  Accuracy Disparity:       {disp['Accuracy_Disparity']:.4f}")
            logger.info(f"  FPR Disparity:            {disp['FPR_Disparity']:.4f}")
            logger.info(f"  FNR Disparity:            {disp['FNR_Disparity']:.4f}")
            logger.info(f"  Selection Rate Disparity: {disp['Selection_Rate_Disparity']:.4f}")

        return disparities

    def generate_report(
        self,
        all_metrics: Dict[str, pd.DataFrame],
        disparities: Dict[str, Dict],
        output_path: Path
    ) -> None:
        """
        Generate comprehensive fairness report.

        Args:
            all_metrics: Metrics by demographic
            disparities: Calculated disparities
            output_path: Path to save CSV report
        """
        reports = []

        # Add metrics sections
        for demographic, metrics_df in all_metrics.items():
            metrics_df_copy = metrics_df.copy()
            metrics_df_copy.insert(0, 'Demographic', demographic)
            reports.append(metrics_df_copy)

        # Combine all reports
        combined_report = pd.concat(reports, ignore_index=True)

        # Round numeric columns
        numeric_cols = combined_report.select_dtypes(include='number').columns
        combined_report[numeric_cols] = combined_report[numeric_cols].round(4)

        # Save to CSV
        combined_report.to_csv(output_path, index=False)
        logger.info(f"\n✓ Fairness report saved to {output_path}")

        # Display summary
        logger.info("\n" + "=" * 70)
        logger.info("FAIRNESS REPORT SUMMARY")
        logger.info("=" * 70)
        logger.info(combined_report.to_string())

        # Display disparities
        logger.info("\n" + "=" * 70)
        logger.info("DISPARITIES SUMMARY")
        logger.info("=" * 70)
        for demographic, disp in disparities.items():
            logger.info(f"\n{demographic}:")
            for metric, value in disp.items():
                threshold = 0.05
                status = "⚠️  HIGH" if value > threshold else "✓ OK"
                logger.info(f"  {metric}: {value:.4f} {status}")

    def save(self, filepath: Path) -> None:
        """Save fairness report to disk."""
        if self.fairness_report is not None:
            self.fairness_report.to_csv(filepath, index=False)
            logger.info(f"Fairness report saved to {filepath}")


def load_test_data(
    test_path: Path = Path('data/processed/test.csv'),
    raw_test_path: Path = Path('data/processed/test_raw.csv')
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load processed and raw test data for fairness evaluation.

    Args:
        test_path: Path to processed test data
        raw_test_path: Path to raw processed test data with demographics

    Returns:
        Tuple of (processed_test_df, raw_test_df, y_test)
    """
    logger.info("Loading test data...")

    test_df = pd.read_csv(test_path)
    raw_test_df = pd.read_csv(raw_test_path)
    y_test = test_df['SEVERITY'].values

    logger.info(f"Loaded {len(test_df)} test samples")
    return test_df, raw_test_df, y_test


def evaluate_fairness(
    model_path: Path = Path('data/models/model.joblib'),
    test_path: Path = Path('data/processed/test.csv'),
    raw_test_path: Path = Path('data/processed/test_raw.csv'),
    output_dir: Path = Path('data/fairness_reports')
) -> Tuple[Dict, Dict]:
    """
    Full fairness evaluation pipeline.

    Steps:
        1. Load model and test data
        2. Generate predictions
        3. Load demographic information
        4. Calculate fairness metrics by group
        5. Calculate disparities
        6. Generate and save fairness report

    Args:
        model_path: Path to trained model
        test_path: Path to processed test data
        raw_data_path: Path to raw NHANES data
        output_dir: Directory to save fairness reports
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("FAIRNESS EVALUATION PIPELINE START")
    logger.info("=" * 70)

    # Step 1: Load model
    logger.info("\n[STEP 1] Loading trained model...")
    depth_model = DepthModel.load(model_path)
    logger.info(f"✓ Loaded {depth_model.model_type} model")

    # Step 2: Load test data
    logger.info("\n[STEP 2] Loading test data...")
    test_df, raw_test_df, y_test_labels = load_test_data(test_path, raw_test_path)

    # Encode true labels to numeric values for evaluation
    y_test_encoded = depth_model.label_encoder.transform(y_test_labels)

    # Get predictions using numeric labels
    X_test = test_df.drop('SEVERITY', axis=1).values
    y_pred_encoded = depth_model.predict(X_test)

    logger.info(f"✓ Generated predictions for {len(X_test)} test samples")

    # Step 3: Load demographic data from raw test set
    logger.info("\n[STEP 3] Loading demographic data...")
    X_demographics = raw_test_df[['RIAGENDR', 'AGE_GROUP', 'RIDRETH1', 'INDHHIN2']]
    logger.info(f"✓ Extracted demographics for {len(X_demographics)} test samples")

    # Step 4: Initialize evaluator and calculate metrics
    logger.info("\n[STEP 4] Evaluating fairness by demographic group...")
    evaluator = FairnessEvaluator()

    all_metrics = evaluator.evaluate_all_groups(
        y_test_encoded, y_pred_encoded, X_demographics
    )

    # Step 5: Calculate disparities
    logger.info("\n[STEP 5] Calculating disparities...")
    disparities = evaluator.calculate_disparities(all_metrics)

    # Step 6: Generate and save report
    logger.info("\n[STEP 6] Generating fairness report...")
    report_path = output_dir / 'fairness_report.csv'
    evaluator.generate_report(all_metrics, disparities, report_path)

    logger.info("\n" + "=" * 70)
    logger.info("FAIRNESS EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report saved to: {report_path}")

    return all_metrics, disparities


def main():
    """Run fairness evaluation pipeline."""
    try:
        all_metrics, disparities = evaluate_fairness()

        print("\n" + "=" * 70)
        print("NEXT STEP: Phase 5 - FastAPI (api.py)")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Fairness evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
