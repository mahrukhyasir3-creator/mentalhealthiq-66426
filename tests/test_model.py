from pathlib import Path

import pandas as pd

from mentalhealthiq.model import DepthModel


def test_model_exposes_depth_model_class() -> None:
    assert hasattr(DepthModel, "predict")
    assert hasattr(DepthModel, "predict_proba")
    assert isinstance(DepthModel.classes_, property)


def test_model_loads_and_predicts(ml_artifacts: dict[str, Path]) -> None:
    model = DepthModel.load(ml_artifacts["model_path"])
    test_df = pd.read_csv(ml_artifacts["processed_dir"] / "test.csv")

    X_test = test_df.drop("SEVERITY", axis=1).to_numpy(dtype=float)
    predictions = model.predict(X_test)
    severity = model.label_encoder.inverse_transform(predictions)[0]

    assert severity in model.classes_
    assert 0.0 <= float(model.predict_proba(X_test)[0].max()) <= 1.0
    assert model.metrics["best_model_name"] == model.model_type
