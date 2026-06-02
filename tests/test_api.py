import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mentalhealthiq.api import app


def _sample_prediction_payload() -> dict:
    raw_test_path = Path('data/processed/test_raw.csv')
    assert raw_test_path.exists(), 'Raw test dataset is required for API tests'

    raw_df = pd.read_csv(raw_test_path)
    sample = raw_df.iloc[0]

    return {
        'RIDAGEYR': int(sample['RIDAGEYR']),
        'RIAGENDR': int(sample['RIAGENDR']),
        'RIDRETH1': int(sample['RIDRETH1']),
        'INDHHIN2': int(sample['INDHHIN2']),
        'DMDEDUC2': int(sample['DMDEDUC2']),
        'DMDMARTL': int(sample['DMDMARTL']),
        'AGE_GROUP': sample['AGE_GROUP'],
        'DPQ010': int(sample['DPQ010']),
        'DPQ020': int(sample['DPQ020']),
        'DPQ030': int(sample['DPQ030']),
        'DPQ040': int(sample['DPQ040']),
        'DPQ050': int(sample['DPQ050']),
        'DPQ060': int(sample['DPQ060']),
        'DPQ070': int(sample['DPQ070']),
        'DPQ080': int(sample['DPQ080']),
        'DPQ090': int(sample['DPQ090']),
    }


@pytest.fixture(autouse=True)
def disable_mongodb(monkeypatch):
    monkeypatch.delenv('MONGODB_URI', raising=False)
    monkeypatch.delenv('MONGODB_DATABASE', raising=False)
    monkeypatch.delenv('MONGODB_COLLECTION', raising=False)
    return None


def test_health_check_endpoint() -> None:
    client = TestClient(app)
    response = client.get('/health')

    assert response.status_code == 200
    json_data = response.json()
    assert json_data['status'] == 'healthy'
    assert 'model_path' in json_data


def test_predict_endpoint() -> None:
    client = TestClient(app)
    payload = _sample_prediction_payload()
    response = client.post('/predict', json=payload)

    assert response.status_code == 200
    json_data = response.json()
    assert 'severity' in json_data
    assert 'risk_score' in json_data
    assert 'warning' in json_data
    assert json_data['predictions_saved'] is False


def test_fairness_report_endpoint() -> None:
    client = TestClient(app)
    response = client.get('/fairness-report')

    assert response.status_code == 200
    json_data = response.json()
    assert 'report_path' in json_data
    assert isinstance(json_data['records'], list)
    assert len(json_data['records']) > 0
