from pathlib import Path

import pandas as pd

from mentalhealthiq.model import DepthModel
from mentalhealthiq.preprocess import DepthPreprocessor


def test_model_loads_and_predicts() -> None:
    model_path = Path('data/models/model.joblib')
    preprocessor_path = Path('data/processed/preprocessor.joblib')
    raw_test_path = Path('data/processed/test_raw.csv')

    assert model_path.exists(), 'Trained model file is required for model tests'
    assert preprocessor_path.exists(), 'Saved preprocessor file is required for model tests'
    assert raw_test_path.exists(), 'Raw test dataset is required for model tests'

    model = DepthModel.load(model_path)
    preprocessor = DepthPreprocessor.load(preprocessor_path)

    raw_df = pd.read_csv(raw_test_path)
    sample = raw_df.iloc[0]
    input_df = pd.DataFrame([
        {
            'RIDAGEYR': int(sample['RIDAGEYR']),
            'RIAGENDR': int(sample['RIAGENDR']),
            'RIDRETH1': int(sample['RIDRETH1']),
            'INDHHIN2': int(sample['INDHHIN2']),
            'DMDEDUC2': int(sample['DMDEDUC2']),
            'DMDMARTL': int(sample['DMDMARTL']),
            'AGE_GROUP': sample['AGE_GROUP'],
            'DPQ010': sample['DPQ010'],
            'DPQ020': sample['DPQ020'],
            'DPQ030': sample['DPQ030'],
            'DPQ040': sample['DPQ040'],
            'DPQ050': sample['DPQ050'],
            'DPQ060': sample['DPQ060'],
            'DPQ070': sample['DPQ070'],
            'DPQ080': sample['DPQ080'],
            'DPQ090': sample['DPQ090'],
        }
    ])

    features = preprocessor.transform(input_df)
    predictions = model.predict(features)
    severity = model.label_encoder.inverse_transform(predictions)[0]

    assert severity in model.label_encoder.classes_
    assert 0.0 <= float(model.predict_proba(features)[0].max()) <= 1.0
