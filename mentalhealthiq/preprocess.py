"""Preprocess real NHANES demographic and PHQ-9 questionnaire data.

The current demo classifies PHQ-9 severity from the PHQ-9 answers themselves.
It is a screening workflow, not an independent clinical diagnosis model.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEMOGRAPHIC_PATH = RAW_DIR / "demographic.csv"
QUESTIONNAIRE_PATH = RAW_DIR / "questionnaire.csv"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"

PHQ_COLUMNS = [
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
]

DEMOGRAPHIC_COLUMNS = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "INDHHIN2",
    "DMDEDUC2",
    "DMDMARTL",
]

QUESTIONNAIRE_COLUMNS = ["SEQN", *PHQ_COLUMNS]

NHANES_MISSING_CODES = {
    "RIAGENDR": {7, 9},
    "RIDRETH1": {7, 9},
    "INDHHIN2": {77, 99},
    "DMDEDUC2": {7, 9},
    "DMDMARTL": {77, 99},
}

NUMERIC_FEATURES = ["RIDAGEYR", *PHQ_COLUMNS]
CATEGORICAL_FEATURES = [
    "RIAGENDR",
    "RIDRETH1",
    "INDHHIN2",
    "DMDEDUC2",
    "DMDMARTL",
    "AGE_GROUP",
]

FEATURE_COLUMNS = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "INDHHIN2",
    "DMDEDUC2",
    "DMDMARTL",
    "AGE_GROUP",
    *PHQ_COLUMNS,
]

TARGET_COLUMN = "SEVERITY"

RAW_OUTPUT_COLUMNS = [
    "SEQN",
    *FEATURE_COLUMNS,
    "PHQ9_TOTAL",
    TARGET_COLUMN,
]


def _one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across supported sklearn versions."""

    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


class DepthPreprocessor:
    """Saved preprocessing wrapper used by training, fairness, and inference."""

    def __init__(self) -> None:
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features = NUMERIC_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        self.feature_names: list[str] = []
        self.n_features_in_: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DepthPreprocessor":
        """Fit the sklearn ColumnTransformer on training features."""

        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns for preprocessing: {missing}")

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _one_hot_encoder()),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

        self.preprocessor.fit(X[FEATURE_COLUMNS])
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self._set_feature_names()
        logger.info("Preprocessor fitted with %s output features.", len(self.feature_names))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw feature columns into model-ready numeric features."""

        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted.")
        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns for transform: {missing}")
        return self.preprocessor.transform(X[FEATURE_COLUMNS])

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform feature columns."""

        self.fit(X, y)
        return self.transform(X)

    def _set_feature_names(self) -> None:
        """Extract transformed feature names from the ColumnTransformer."""

        if self.preprocessor is None:
            self.feature_names = []
            return

        try:
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        except Exception:
            cat_pipeline = self.preprocessor.named_transformers_["cat"]
            encoder = cat_pipeline.named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(self.categorical_features).tolist()
            self.feature_names = [*self.numeric_features, *cat_names]

    def save(self, filepath: Path) -> None:
        """Save fitted preprocessing assets."""

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "feature_names": self.feature_names,
                "n_features_in_": self.n_features_in_,
            },
            filepath,
        )
        logger.info("Preprocessor saved to %s", filepath)

    @staticmethod
    def load(filepath: Path) -> "DepthPreprocessor":
        """Load fitted preprocessing assets."""

        asset = joblib.load(filepath)
        preprocessor = DepthPreprocessor()
        preprocessor.preprocessor = asset["preprocessor"]
        preprocessor.numeric_features = asset.get("numeric_features", NUMERIC_FEATURES.copy())
        preprocessor.categorical_features = asset.get("categorical_features", CATEGORICAL_FEATURES.copy())
        preprocessor.feature_names = asset.get("feature_names", [])
        preprocessor.n_features_in_ = asset.get("n_features_in_")
        return preprocessor


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Validate required columns are present in a dataframe."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def validate_unique_key(df: pd.DataFrame, key_column: str, dataset_name: str) -> None:
    """Validate merge keys are unique before joining source datasets."""

    duplicates = df[key_column][df[key_column].duplicated()].dropna().unique()
    if len(duplicates):
        sample = duplicates[:5].tolist()
        raise ValueError(f"{dataset_name} has duplicate {key_column} values: {sample}")


def load_raw_data(
    demographic_path: Path = DEMOGRAPHIC_PATH,
    questionnaire_path: Path = QUESTIONNAIRE_PATH,
) -> pd.DataFrame:
    """Load and merge real NHANES demographic and questionnaire CSV files."""

    demographic_path = Path(demographic_path)
    questionnaire_path = Path(questionnaire_path)

    if not demographic_path.exists() or not questionnaire_path.exists():
        raise FileNotFoundError(
            "Please place demographic.csv and questionnaire.csv inside data/raw/"
        )

    demographic_df = pd.read_csv(demographic_path)
    questionnaire_df = pd.read_csv(questionnaire_path)

    validate_columns(demographic_df, DEMOGRAPHIC_COLUMNS, str(demographic_path))
    validate_columns(questionnaire_df, QUESTIONNAIRE_COLUMNS, str(questionnaire_path))
    validate_unique_key(demographic_df, "SEQN", str(demographic_path))
    validate_unique_key(questionnaire_df, "SEQN", str(questionnaire_path))

    merged_df = demographic_df[DEMOGRAPHIC_COLUMNS].merge(
        questionnaire_df[QUESTIONNAIRE_COLUMNS],
        on="SEQN",
        how="inner",
    )
    logger.info(
        "Merged raw dataset has %s rows (%s demographic-only rows, %s questionnaire-only rows skipped).",
        len(merged_df),
        len(demographic_df) - len(merged_df),
        len(questionnaire_df) - len(merged_df),
    )
    return merged_df


def clean_demographic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert demographic columns to numeric values and null NHANES missing codes."""

    cleaned = df.copy()
    for column in DEMOGRAPHIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column in NHANES_MISSING_CODES:
            cleaned[column] = cleaned[column].where(
                ~cleaned[column].isin(NHANES_MISSING_CODES[column]),
                np.nan,
            )
    cleaned["RIDAGEYR"] = cleaned["RIDAGEYR"].where(cleaned["RIDAGEYR"].between(0, 120), np.nan)
    return cleaned


def create_phq9_total(df: pd.DataFrame) -> pd.DataFrame:
    """Clean PHQ-9 item columns and create PHQ9_TOTAL."""

    validate_columns(df, PHQ_COLUMNS, "merged dataset")
    cleaned = df.copy()

    for column in PHQ_COLUMNS:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.where(values.isin([0, 1, 2, 3]), np.nan)

    cleaned["PHQ9_TOTAL"] = cleaned[PHQ_COLUMNS].sum(axis=1, min_count=len(PHQ_COLUMNS))
    logger.info("PHQ9_TOTAL created for %s complete rows.", cleaned["PHQ9_TOTAL"].notna().sum())
    return cleaned


def create_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create SEVERITY labels from PHQ9_TOTAL."""

    if "PHQ9_TOTAL" not in df.columns:
        raise ValueError("PHQ9_TOTAL column is required before severity labeling.")

    labeled = df.copy()
    bins = [-1, 4, 9, 14, 19, 27]
    labels = ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]
    labeled[TARGET_COLUMN] = pd.cut(labeled["PHQ9_TOTAL"], bins=bins, labels=labels)
    labeled[TARGET_COLUMN] = labeled[TARGET_COLUMN].astype("object")
    return labeled


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create AGE_GROUP from RIDAGEYR."""

    if "RIDAGEYR" not in df.columns:
        raise ValueError("RIDAGEYR column is required before age grouping.")

    grouped = df.copy()
    age = pd.to_numeric(grouped["RIDAGEYR"], errors="coerce")
    grouped["AGE_GROUP"] = pd.Series(index=grouped.index, dtype="object")
    grouped.loc[age < 18, "AGE_GROUP"] = "Under 18"
    grouped.loc[age.between(18, 29, inclusive="both"), "AGE_GROUP"] = "18-29"
    grouped.loc[age.between(30, 44, inclusive="both"), "AGE_GROUP"] = "30-44"
    grouped.loc[age.between(45, 59, inclusive="both"), "AGE_GROUP"] = "45-59"
    grouped.loc[age >= 60, "AGE_GROUP"] = "60+"
    return grouped


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with invalid PHQ-9 answers or missing target labels."""

    before = len(df)
    cleaned = df.dropna(subset=[*PHQ_COLUMNS, "PHQ9_TOTAL", TARGET_COLUMN]).copy()
    logger.info("Dropped %s rows with invalid or incomplete PHQ-9 data.", before - len(cleaned))
    return cleaned


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select final model features and target."""

    validate_columns(df, FEATURE_COLUMNS, "preprocessed dataset")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} column is required.")
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(str).copy()


def validate_stratified_split(y: pd.Series, test_size: float) -> None:
    """Validate the target distribution can support stratified splitting."""

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1. Received: {test_size}")

    class_counts = y.value_counts()
    if class_counts.empty:
        raise ValueError("No target labels available for splitting.")
    if (class_counts < 2).any():
        raise ValueError(
            "Each severity class needs at least two rows for stratified splitting. "
            f"Counts: {class_counts.to_dict()}"
        )

    requested_test_count = int(np.ceil(len(y) * test_size))
    if requested_test_count < len(class_counts):
        raise ValueError(
            "test_size is too small for stratification across all severity classes. "
            f"Need at least {len(class_counts)} test rows."
        )

    requested_train_count = len(y) - requested_test_count
    if requested_train_count < len(class_counts):
        raise ValueError(
            "test_size is too large for stratification across all severity classes. "
            f"Need at least {len(class_counts)} training rows."
        )


def _transformed_dataframe(
    transformed: np.ndarray,
    target: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a CSV-ready transformed dataframe with SEVERITY appended."""

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature width does not match feature names: "
            f"{transformed.shape[1]} != {len(feature_names)}"
        )

    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    transformed_df[TARGET_COLUMN] = target.to_numpy()
    return transformed_df


def preprocess_pipeline(
    demographic_path: Path = DEMOGRAPHIC_PATH,
    questionnaire_path: Path = QUESTIONNAIRE_PATH,
    output_dir: Path = PROCESSED_DIR,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DepthPreprocessor]:
    """Run real NHANES preprocessing and save model-ready artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(demographic_path=demographic_path, questionnaire_path=questionnaire_path)
    df = clean_demographic_columns(df)
    df = create_phq9_total(df)
    df = create_severity_labels(df)
    df = create_age_groups(df)
    df = handle_missing_values(df)

    X, y = select_features(df)
    validate_stratified_split(y, test_size)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, y_train = select_features(train_df)
    X_test, y_test = select_features(test_df)

    preprocessor = DepthPreprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)

    train_transformed_df = _transformed_dataframe(
        X_train_transformed,
        y_train,
        preprocessor.feature_names,
    )
    test_transformed_df = _transformed_dataframe(
        X_test_transformed,
        y_test,
        preprocessor.feature_names,
    )

    train_transformed_df.to_csv(output_dir / "train.csv", index=False)
    test_transformed_df.to_csv(output_dir / "test.csv", index=False)
    train_df[RAW_OUTPUT_COLUMNS].to_csv(output_dir / "train_raw.csv", index=False)
    test_df[RAW_OUTPUT_COLUMNS].to_csv(output_dir / "test_raw.csv", index=False)
    preprocessor.save(output_dir / "preprocessor.joblib")

    logger.info("Saved processed artifacts to %s", output_dir)
    return X_train, X_test, y_train, y_test, preprocessor


def main() -> int:
    """Run preprocessing from the command line."""

    try:
        preprocess_pipeline()
        print("Preprocessing complete. Next: python -m mentalhealthiq.model")
        return 0
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
