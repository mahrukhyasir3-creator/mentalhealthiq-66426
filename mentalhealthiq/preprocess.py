"""
Phase 2: Data Preprocessing Module

Loads NHANES PHQ-9 depression data, handles missing values, creates severity labels,
encodes categorical variables, performs stratified train/test split, and saves
processed datasets with a fitted preprocessor for inference.

Functions:
    - load_raw_data: Load NHANES CSV files and merge by SEQN
    - create_phq9_total: Calculate PHQ-9 total score
    - create_severity_labels: Map PHQ-9 scores to depression severity levels
    - create_age_groups: Bin continuous age into categorical groups
    - preprocess_pipeline: Full preprocessing pipeline
    - save_preprocessor: Serialize preprocessor with joblib
    - load_preprocessor: Deserialize preprocessor from joblib
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthPreprocessor:
    """
    Preprocessor for NHANES PHQ-9 depression data.

    Handles data loading, missing value imputation, feature engineering,
    categorical encoding, numerical scaling, and dataset splitting.

    Attributes:
        preprocessor: sklearn ColumnTransformer with encoding/scaling pipeline
        label_encoder: LabelEncoder for depression severity labels
        phq_columns: List of PHQ-9 question column names
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        feature_names: Final feature names after preprocessing
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.preprocessor: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.phq_columns: list = [
            'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050',
            'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090'
        ]
        self.numeric_features: list = []
        self.categorical_features: list = []
        self.feature_names: list = []
        self.n_features_in_: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DepthPreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            X: Feature dataframe with numeric and categorical columns
            y: Target labels (not used for fitting, for sklearn compatibility)

        Returns:
            self: Fitted preprocessor instance
        """
        logger.info("Fitting preprocessor on training data")

        # Separate numeric and categorical features
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.debug(f"Numeric features: {self.numeric_features}")
        logger.debug(f"Categorical features: {self.categorical_features}")

        # Create preprocessing pipeline
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            drop='if_binary'
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose=False,
            remainder='drop'
        )

        # Fit preprocessor
        self.preprocessor.fit(X)

        # Get feature names after transformation
        self._set_feature_names()

        self.n_features_in_ = X.shape[1]
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names)}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Feature dataframe

        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        logger.debug(f"Transforming {len(X)} samples")
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _set_feature_names(self) -> None:
        """Extract feature names from ColumnTransformer."""
        feature_names = []

        # Numeric features
        for col in self.numeric_features:
            feature_names.append(col)

        # Categorical features (one-hot encoded)
        cat_transformer = self.preprocessor.named_transformers_['cat']
        cat_names = cat_transformer.get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_names)

        self.feature_names = feature_names
        logger.debug(f"Feature names set: {len(feature_names)} total")

    def save(self, filepath: Path) -> None:
        """Save fitted preprocessor to disk."""
        asset = {
            'preprocessor': self.preprocessor,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'n_features_in_': self.n_features_in_,
        }
        joblib.dump(asset, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> 'DepthPreprocessor':
        """Load fitted preprocessor from disk."""
        asset = joblib.load(filepath)
        preprocessor = DepthPreprocessor()
        preprocessor.preprocessor = asset['preprocessor']
        preprocessor.numeric_features = asset.get('numeric_features', [])
        preprocessor.categorical_features = asset.get('categorical_features', [])
        preprocessor.feature_names = asset.get('feature_names', [])
        preprocessor.n_features_in_ = asset.get('n_features_in_', None)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load NHANES raw data.

    In production, this would load separate demographics, depression survey,
    and education files, then merge on SEQN. For demo, loads synthetic data.

    Args:
        data_path: Path to raw data CSV file

    Returns:
        Merged dataframe with all NHANES columns

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If SEQN column missing
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading raw data from {data_path}")
    df = pd.read_csv(data_path)

    if 'SEQN' not in df.columns:
        raise ValueError("SEQN column not found in data")

    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def create_phq9_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PHQ-9 total score from individual items.

    PHQ-9 total = sum of DPQ010 through DPQ090.
    If any required item is missing, PHQ9_TOTAL = NaN.

    Args:
        df: DataFrame with PHQ-9 columns

    Returns:
        DataFrame with added PHQ9_TOTAL column
    """
    phq_columns = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050',
                   'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090']

    # Verify all PHQ columns present
    missing_cols = [col for col in phq_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing PHQ-9 columns: {missing_cols}")

    # Calculate total (only if at least 7 items answered)
    phq_subset = df[phq_columns].copy()
    valid_count = phq_subset.notna().sum(axis=1)

    df_copy = df.copy()
    df_copy['PHQ9_TOTAL'] = df_copy[phq_columns].sum(axis=1)
    df_copy.loc[valid_count < 7, 'PHQ9_TOTAL'] = np.nan

    n_valid = df_copy['PHQ9_TOTAL'].notna().sum()
    logger.info(f"Created PHQ9_TOTAL: {n_valid} valid scores out of {len(df)}")

    return df_copy


def create_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create depression severity labels from PHQ-9 total score.

    Severity levels (based on PHQ-9 score):
        - Minimal: 0-4
        - Mild: 5-9
        - Moderate: 10-14
        - Moderately Severe: 15-19
        - Severe: 20-27

    Args:
        df: DataFrame with PHQ9_TOTAL column

    Returns:
        DataFrame with added SEVERITY label column
    """
    if 'PHQ9_TOTAL' not in df.columns:
        raise ValueError("PHQ9_TOTAL column not found. Call create_phq9_total() first.")

    df_copy = df.copy()

    # Create severity labels using pd.cut
    bins = [-1, 4, 9, 14, 19, 27]
    labels = ['Minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe']

    df_copy['SEVERITY'] = pd.cut(
        df_copy['PHQ9_TOTAL'],
        bins=bins,
        labels=labels,
        right=True
    )

    severity_counts = df_copy['SEVERITY'].value_counts().sort_index()
    logger.info(f"Severity labels created:\n{severity_counts}")

    return df_copy


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age group categories from continuous age.

    Age groups:
        - 18-25: Young Adults
        - 26-40: Adults
        - 41-55: Middle-aged
        - 56-70: Mature Adults
        - 71+: Older Adults

    Args:
        df: DataFrame with RIDAGEYR column

    Returns:
        DataFrame with added AGE_GROUP column
    """
    if 'RIDAGEYR' not in df.columns:
        raise ValueError("RIDAGEYR column not found.")

    df_copy = df.copy()

    bins = [18, 25, 40, 55, 70, 100]
    labels = ['18-25', '26-40', '41-55', '56-70', '71+']

    df_copy['AGE_GROUP'] = pd.cut(
        df_copy['RIDAGEYR'],
        bins=bins,
        labels=labels,
        right=False
    )

    logger.debug(f"Age groups created:\n{df_copy['AGE_GROUP'].value_counts().sort_index()}")

    return df_copy


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input dataframe
        strategy: 'drop' to remove rows with missing values,
                 'median' to impute numeric with median,
                 'mode' to impute categorical with mode

    Returns:
        DataFrame with missing values handled

    Raises:
        ValueError: If unknown strategy provided
    """
    if strategy == 'drop':
        n_before = len(df)
        df_clean = df.dropna(subset=['PHQ9_TOTAL', 'SEVERITY'])
        n_after = len(df_clean)
        logger.info(f"Dropped {n_before - n_after} rows with missing target. Remaining: {n_after}")
        return df_clean

    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_copy = df.copy()
        for col in numeric_cols:
            if df_copy[col].isna().any():
                median_val = df_copy[col].median()
                df_copy[col].fillna(median_val, inplace=True)
        logger.info(f"Missing numeric values imputed with median")
        return df_copy

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features for modeling.

    Features selected:
        - Numeric: RIDAGEYR
        - Categorical: RIAGENDR, RIDRETH1, INDHHIN2, DMDEDUC2, DMDMARTL, AGE_GROUP
        - PHQ-9 items: DPQ010-DPQ090

    Target:
        - SEVERITY

    Args:
        df: Preprocessed dataframe

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    feature_cols = [
        'RIDAGEYR',  # Age (numeric)
        'RIAGENDR',  # Gender (categorical)
        'RIDRETH1',  # Race/Ethnicity (categorical)
        'INDHHIN2',  # Income (categorical)
        'DMDEDUC2',  # Education (categorical)
        'DMDMARTL',  # Marital Status (categorical)
        'AGE_GROUP',  # Age group (categorical)
        'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050',
        'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090',  # PHQ-9 items (numeric)
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    y = df['SEVERITY'].copy()

    logger.info(f"Selected {X.shape[1]} features, target has {len(y.unique())} classes")

    return X, y


def preprocess_pipeline(
    data_path: Path,
    test_size: float = 0.3,
    random_state: int = 42,
    output_dir: Path = Path('data/processed')
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DepthPreprocessor]:
    """
    Full preprocessing pipeline: load → clean → feature engineer → split → encode.

    Steps:
        1. Load raw NHANES data
        2. Create PHQ9_TOTAL and SEVERITY labels
        3. Create age groups
        4. Handle missing values
        5. Select features
        6. Stratified train/test split
        7. Fit preprocessor on training data
        8. Transform both train and test data
        9. Save preprocessor and datasets

    Args:
        data_path: Path to raw NHANES CSV file
        test_size: Test set proportion (default: 0.3)
        random_state: Random seed for reproducibility
        output_dir: Directory to save processed data and preprocessor

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("=" * 70)

    # Step 1: Load
    logger.info("\n[STEP 1] Loading raw data...")
    df = load_raw_data(data_path)

    # Step 2: Feature engineering
    logger.info("\n[STEP 2] Creating derived features...")
    df = create_phq9_total(df)
    df = create_severity_labels(df)
    df = create_age_groups(df)

    # Step 3: Handle missing values
    logger.info("\n[STEP 3] Handling missing values...")
    df = handle_missing_values(df, strategy='drop')

    # Step 4: Select features and target for model training
    logger.info("\n[STEP 4] Selecting features...")
    X, y = select_features(df)

    # Step 5: Stratified train/test split on full dataframe so we retain raw demographics
    logger.info("\n[STEP 5] Stratified train/test split...")
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, y_train = select_features(df_train)
    X_test, y_test = select_features(df_test)

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Train class distribution:\n{y_train.value_counts()}")
    logger.info(f"Test class distribution:\n{y_test.value_counts()}")

    # Step 6: Fit preprocessor on training data
    logger.info("\n[STEP 6] Fitting preprocessor...")
    preprocessor = DepthPreprocessor()
    preprocessor.fit(X_train)

    # Step 7: Transform both splits
    logger.info("\n[STEP 7] Transforming data...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Step 8: Save
    logger.info("\n[STEP 8] Saving processed data...")
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    train_raw_path = output_dir / 'train_raw.csv'
    test_raw_path = output_dir / 'test_raw.csv'
    preprocessor_path = output_dir / 'preprocessor.joblib'

    # Save transformed dataframes for inspection
    train_df = pd.DataFrame(
        X_train_transformed,
        columns=preprocessor.feature_names
    )
    train_df['SEVERITY'] = y_train.values
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame(
        X_test_transformed,
        columns=preprocessor.feature_names
    )
    test_df['SEVERITY'] = y_test.values
    test_df.to_csv(test_path, index=False)

    # Save raw train/test sets for fairness and tracing
    raw_cols = ['SEQN'] if 'SEQN' in df.columns else []
    raw_train_df = df_train[raw_cols + [
        'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'INDHHIN2', 'DMDEDUC2',
        'DMDMARTL', 'AGE_GROUP', 'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040',
        'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'SEVERITY'
    ]].copy()
    raw_test_df = df_test[raw_cols + [
        'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'INDHHIN2', 'DMDEDUC2',
        'DMDMARTL', 'AGE_GROUP', 'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040',
        'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'SEVERITY'
    ]].copy()
    raw_train_df.to_csv(train_raw_path, index=False)
    raw_test_df.to_csv(test_raw_path, index=False)

    # Save preprocessor
    preprocessor.save(preprocessor_path)

    logger.info(f"✓ Train set: {train_path} ({train_df.shape})")
    logger.info(f"✓ Test set: {test_path} ({test_df.shape})")
    logger.info(f"✓ Raw train set: {train_raw_path} ({raw_train_df.shape})")
    logger.info(f"✓ Raw test set: {test_raw_path} ({raw_test_df.shape})")
    logger.info(f"✓ Preprocessor: {preprocessor_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Features: {len(preprocessor.feature_names)}")
    logger.info(f"Train shape: {X_train_transformed.shape}")
    logger.info(f"Test shape: {X_test_transformed.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


def main():
    """Run preprocessing pipeline."""
    import sys

    try:
        data_path = Path('data/sample_synthetic_nhanes.csv')
        output_dir = Path('data/processed')

        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
            data_path=data_path,
            test_size=0.3,
            random_state=42,
            output_dir=output_dir
        )

        print("\n" + "=" * 70)
        print("NEXT STEP: Phase 3 - Modeling (model.py)")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
