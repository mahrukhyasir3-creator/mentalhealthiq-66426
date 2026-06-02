import tempfile
from pathlib import Path

from mentalhealthiq.preprocess import preprocess_pipeline


def test_preprocess_pipeline_creates_processed_files() -> None:
    raw_data_path = Path('data/sample_synthetic_nhanes.csv')
    assert raw_data_path.exists(), 'Sample synthetic dataset is required for preprocess tests'

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
            data_path=raw_data_path,
            test_size=0.2,
            random_state=42,
            output_dir=output_dir
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert preprocessor.feature_names

        assert (output_dir / 'train.csv').exists()
        assert (output_dir / 'test.csv').exists()
        assert (output_dir / 'train_raw.csv').exists()
        assert (output_dir / 'test_raw.csv').exists()
        assert (output_dir / 'preprocessor.joblib').exists()
