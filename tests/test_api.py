from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import mentalhealthiq.api as api


def _sample_prediction_payload(raw_test_path: Path) -> dict:
    raw_df = pd.read_csv(raw_test_path)
    sample = raw_df.iloc[0]

    return {
        "RIDAGEYR": int(sample["RIDAGEYR"]),
        "RIAGENDR": int(sample["RIAGENDR"]),
        "RIDRETH1": int(sample["RIDRETH1"]),
        "INDHHIN2": int(sample["INDHHIN2"]),
        "DMDEDUC2": int(sample["DMDEDUC2"]),
        "DMDMARTL": int(sample["DMDMARTL"]),
        "DPQ010": int(sample["DPQ010"]),
        "DPQ020": int(sample["DPQ020"]),
        "DPQ030": int(sample["DPQ030"]),
        "DPQ040": int(sample["DPQ040"]),
        "DPQ050": int(sample["DPQ050"]),
        "DPQ060": int(sample["DPQ060"]),
        "DPQ070": int(sample["DPQ070"]),
        "DPQ080": int(sample["DPQ080"]),
        "DPQ090": int(sample["DPQ090"]),
    }


def _prediction_payload_for_total(total: int) -> dict:
    values = []
    remaining = total
    for _ in range(9):
        value = min(3, remaining)
        values.append(value)
        remaining -= value

    return {
        "RIDAGEYR": 35,
        "RIAGENDR": 2,
        "RIDRETH1": 3,
        "INDHHIN2": 5,
        "DMDEDUC2": 4,
        "DMDMARTL": 1,
        "DPQ010": values[0],
        "DPQ020": values[1],
        "DPQ030": values[2],
        "DPQ040": values[3],
        "DPQ050": values[4],
        "DPQ060": values[5],
        "DPQ070": values[6],
        "DPQ080": values[7],
        "DPQ090": values[8],
    }


@pytest.fixture(autouse=True)
def disable_mongodb(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)
    monkeypatch.delenv("MONGODB_COLLECTION", raising=False)
    return None


def test_api_imports_without_crashing() -> None:
    assert api.app.title == "MentalHealthIQ Prediction API"


def test_health_check_endpoint() -> None:
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["api"] == "ready"
    assert json_data["model"] in {"ready", "missing"}
    assert json_data["preprocessor"] in {"ready", "missing"}
    assert json_data["mongo"] in {"ready", "error"}


def test_predictions_returns_empty_list_when_mongodb_not_configured() -> None:
    original_get_predictions = api.mongo_db.get_predictions
    api.mongo_db.get_predictions = lambda limit=100: []
    client = TestClient(api.app)
    try:
        response = client.get("/predictions?limit=100")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        api.mongo_db.get_predictions = original_get_predictions


def test_predict_endpoint_uses_temp_artifacts(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MODEL_PATH", ml_artifacts["model_path"])
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", ml_artifacts["preprocessor_path"])
    api.app.state.inference = None

    with TestClient(api.app) as client:
        response = client.post("/predict", json=_sample_prediction_payload(ml_artifacts["test_raw_path"]))

    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_severity" in json_data
    assert "risk_score" in json_data
    assert "risk_band" in json_data
    assert "phq9_total" in json_data
    assert "recommendation" in json_data
    assert "fairness_flag" in json_data
    assert "explanation" in json_data
    assert "warning" in json_data
    assert "probabilities" in json_data
    assert "rule_based_severity" in json_data
    assert "model_predicted_severity" in json_data
    assert "model_agreement" in json_data
    assert "timestamp" in json_data
    assert json_data["predictions_saved"] is False


@pytest.mark.parametrize(
    ("total", "expected_severity"),
    [
        (0, "Minimal"),
        (5, "Mild"),
        (10, "Moderate"),
        (15, "Moderately Severe"),
        (20, "Severe"),
        (27, "Severe"),
    ],
)
def test_predict_endpoint_uses_phq9_rule_severity_as_primary_result(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    total: int,
    expected_severity: str,
) -> None:
    monkeypatch.setattr(api, "MODEL_PATH", ml_artifacts["model_path"])
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", ml_artifacts["preprocessor_path"])
    api.app.state.inference = None

    with TestClient(api.app) as client:
        response = client.post("/predict", json=_prediction_payload_for_total(total))

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["phq9_total"] == total
    assert json_data["rule_based_severity"] == expected_severity
    assert json_data["predicted_severity"] == expected_severity
    assert json_data["severity"] == expected_severity
    assert isinstance(json_data["model_predicted_severity"], str)
    assert isinstance(json_data["model_agreement"], bool)
    assert "probabilities" in json_data


def test_predict_auto_assigns_patient_id(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MODEL_PATH", ml_artifacts["model_path"])
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", ml_artifacts["preprocessor_path"])
    api.app.state.inference = None

    payload = _sample_prediction_payload(ml_artifacts["test_raw_path"])
    payload["patient_name"] = "Ayesha Khan"

    with TestClient(api.app) as client:
        first_response = client.post("/predict", json=payload)
        second_response = client.post("/predict", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_json = first_response.json()
    second_json = second_response.json()
    assert first_json["patient_id"].startswith("MHQ-AYESHA-KHAN-")
    assert first_json["patient_id"] == second_json["patient_id"]


def test_predict_and_save_uses_auto_patient_id_for_history(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_documents = []

    def fake_save_prediction(document: dict) -> str:
        saved_documents.append(document)
        return f"saved-{len(saved_documents)}"

    monkeypatch.setattr(api, "MODEL_PATH", ml_artifacts["model_path"])
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", ml_artifacts["preprocessor_path"])
    monkeypatch.setattr(api.mongo_db, "save_prediction", fake_save_prediction)
    api.app.state.inference = None

    payload = _sample_prediction_payload(ml_artifacts["test_raw_path"])
    payload["patient_name"] = "Bilal Ahmed"

    with TestClient(api.app) as client:
        first_response = client.post("/predict-and-save", json=payload)
        second_response = client.post("/predict-and-save", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert len(saved_documents) == 2
    assert saved_documents[0]["patient_id"] == saved_documents[1]["patient_id"]
    assert saved_documents[0]["input_data"]["patient_id"] == saved_documents[0]["patient_id"]


def test_predict_and_save_returns_clear_error_when_mongodb_unavailable(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MODEL_PATH", ml_artifacts["model_path"])
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", ml_artifacts["preprocessor_path"])
    monkeypatch.setattr(api.mongo_db, "save_prediction", lambda document: None)
    api.app.state.inference = None

    with TestClient(api.app) as client:
        response = client.post("/predict-and-save", json=_sample_prediction_payload(ml_artifacts["test_raw_path"]))

    assert response.status_code == 503
    assert "MongoDB is not available" in response.json()["detail"]


def test_fairness_report_endpoint_uses_temp_artifacts(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "FAIRNESS_REPORT_PATH", ml_artifacts["fairness_report_path"])

    client = TestClient(api.app)
    response = client.get("/fairness-report")

    assert response.status_code == 200
    json_data = response.json()
    assert "report_path" in json_data
    assert isinstance(json_data["records"], list)
    assert len(json_data["records"]) > 0
    assert "group_column" in json_data["records"][0]


def test_dashboard_patient_and_fairness_summary_endpoints(
    ml_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "FAIRNESS_REPORT_PATH", ml_artifacts["fairness_report_path"])
    monkeypatch.setattr(api.mongo_db, "get_predictions", lambda limit=500: [])
    monkeypatch.setattr(api.mongo_db, "get_patient_history", lambda patient_id: [])
    monkeypatch.setattr(
        api.mongo_db,
        "get_patient_comparison",
        lambda patient_id: {
            "patient_id": patient_id,
            "has_comparison": False,
            "has_enough_history": False,
            "status": "insufficient_history",
            "summary": "At least two saved visits are needed for comparison.",
        },
    )
    monkeypatch.setattr(api.mongo_db, "get_stats", lambda: {"total_predictions": 0})

    client = TestClient(api.app)

    stats_response = client.get("/stats")
    history_response = client.get("/patients/P-1001/history")
    comparison_response = client.get("/patients/P-1001/comparison")
    fairness_summary_response = client.get("/fairness-summary")

    assert stats_response.status_code == 200
    assert stats_response.json()["total_predictions"] == 0
    assert history_response.status_code == 200
    assert history_response.json() == []
    assert comparison_response.status_code == 200
    assert comparison_response.json()["has_comparison"] is False
    assert fairness_summary_response.status_code == 200
    assert "bias_detected" in fairness_summary_response.json()
