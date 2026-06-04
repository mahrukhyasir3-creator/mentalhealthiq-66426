"""FastAPI API for MentalHealthIQ depression severity prediction."""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint

from mentalhealthiq.db import mongo_db
from mentalhealthiq.fairness import FAIRNESS_REPORT_PATH
from mentalhealthiq.model import DepthModel, MODEL_PATH
from mentalhealthiq.preprocess import FEATURE_COLUMNS, PREPROCESSOR_PATH, DepthPreprocessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def derive_age_group(age: int) -> str:
    """Derive the configured AGE_GROUP label from RIDAGEYR."""

    if age < 18:
        return "Under 18"
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


def _stable_patient_hash(value: str) -> str:
    """Return a short deterministic hash shared with the browser implementation."""

    hash_value = 0
    for character in value:
        hash_value = ((hash_value << 5) - hash_value + ord(character)) & 0xFFFFFFFF
    return base36(hash_value).upper().rjust(6, "0")[-6:]


def base36(value: int) -> str:
    """Convert a non-negative integer to base36."""

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def generate_patient_id(payload: "PredictionInput") -> str:
    """Generate a stable clinic ID when the frontend/user does not provide one."""

    patient_name = (payload.patient_name or "Patient").strip()
    name_slug = re.sub(r"[^A-Za-z0-9]+", "-", patient_name).strip("-").upper() or "PATIENT"
    name_slug = name_slug[:18]
    seed = f"{name_slug}|{payload.RIDAGEYR}|{payload.RIAGENDR}"
    return f"MHQ-{name_slug}-{_stable_patient_hash(seed)}"


def normalize_patient_identity(payload: "PredictionInput") -> Dict[str, str]:
    """Return the patient ID/name used consistently for prediction and history."""

    patient_id = (payload.patient_id or "").strip() or generate_patient_id(payload)
    patient_name = (payload.patient_name or "").strip() or "Unnamed patient"
    return {"patient_id": patient_id, "patient_name": patient_name}


class PredictionInput(BaseModel):
    """Prediction request body."""

    patient_id: Optional[str] = Field(default=None, description="Clinic patient identifier")
    patient_name: Optional[str] = Field(default=None, description="Patient full name")
    visit_date: Optional[str] = Field(default=None, description="Visit date")
    visit_time: Optional[str] = Field(default=None, description="Visit time")
    doctor_notes: Optional[str] = Field(default=None, description="Doctor or clinic notes")

    RIDAGEYR: conint(ge=0, le=120) = Field(..., description="Age in years")
    RIAGENDR: conint(ge=0) = Field(..., description="Gender code")
    RIDRETH1: conint(ge=0) = Field(..., description="Race/ethnicity code")
    INDHHIN2: conint(ge=0) = Field(..., description="Household income code")
    DMDEDUC2: conint(ge=0) = Field(..., description="Education code")
    DMDMARTL: conint(ge=0) = Field(..., description="Marital status code")
    AGE_GROUP: Optional[str] = Field(default=None, description="Age group category")

    DPQ010: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 1")
    DPQ020: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 2")
    DPQ030: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 3")
    DPQ040: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 4")
    DPQ050: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 5")
    DPQ060: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 6")
    DPQ070: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 7")
    DPQ080: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 8")
    DPQ090: conint(ge=0, le=3) = Field(..., description="PHQ-9 item 9")


class PredictionResponse(BaseModel):
    """Prediction API response."""

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    visit_date: Optional[str] = None
    visit_time: Optional[str] = None
    doctor_notes: Optional[str] = None
    predicted_severity: str
    rule_based_severity: str
    model_predicted_severity: str
    model_agreement: bool
    model_note: Optional[str] = None
    severity: str
    risk_score: float
    risk_percentage: float
    risk_band: str
    phq9_total: int
    recommendation: str
    warning: str
    fairness_flag: bool
    explanation: str
    probabilities: Dict[str, float]
    timestamp: str
    model_type: str
    saved: bool = False
    prediction_id: Optional[str] = None
    error: Optional[str] = None
    predictions_saved: bool = False
    inserted_id: Optional[str] = None


class FairnessReportResponse(BaseModel):
    """Fairness report API response."""

    report_path: str
    records: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str


def load_inference_assets() -> Dict[str, Any]:
    """Load trained model and preprocessor artifacts."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    return {
        "model": DepthModel.load(MODEL_PATH),
        "preprocessor": DepthPreprocessor.load(PREPROCESSOR_PATH),
    }


def create_input_dataframe(payload: PredictionInput) -> pd.DataFrame:
    """Convert request payload into model input DataFrame."""

    payload_dict = payload.model_dump()
    if not payload_dict.get("AGE_GROUP"):
        payload_dict["AGE_GROUP"] = derive_age_group(payload.RIDAGEYR)
    return pd.DataFrame([payload_dict], columns=FEATURE_COLUMNS)


def calculate_phq9_total(payload: PredictionInput) -> int:
    """Calculate PHQ-9 total score from all 9 answers."""

    return int(
        payload.DPQ010
        + payload.DPQ020
        + payload.DPQ030
        + payload.DPQ040
        + payload.DPQ050
        + payload.DPQ060
        + payload.DPQ070
        + payload.DPQ080
        + payload.DPQ090
    )


def phq9_severity_from_total(total: int) -> str:
    """Map PHQ-9 total score to the standard screening severity category."""

    if total <= 4:
        return "Minimal"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    if total <= 19:
        return "Moderately Severe"
    return "Severe"


def risk_band_from_severity(severity: str) -> str:
    """Convert severity to a simple risk band."""

    if severity in {"Minimal", "Mild"}:
        return "Low"
    if severity == "Moderate":
        return "Medium"
    return "High"


def load_fairness_report() -> List[Dict[str, Any]]:
    """Load fairness report from CSV."""

    if not FAIRNESS_REPORT_PATH.exists():
        raise FileNotFoundError(f"Fairness report not found: {FAIRNESS_REPORT_PATH}")
    return pd.read_csv(FAIRNESS_REPORT_PATH).to_dict(orient="records")


def summarize_fairness_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create simple frontend-ready fairness summary from report rows."""

    summary: Dict[str, Any] = {
        "highest_risk_age_group": None,
        "lowest_risk_age_group": None,
        "highest_risk_gender": None,
        "lowest_risk_gender": None,
        "highest_risk_race": None,
        "fairness_warning": "Fairness report is not available yet.",
        "bias_detected": False,
        "groups": {},
    }
    if not records:
        return summary

    report_df = pd.DataFrame(records)
    if "risk_percentage" not in report_df.columns:
        report_df["risk_percentage"] = report_df.get("selection_rate", 0) * 100

    for group_column, group_df in report_df.groupby("group_column"):
        group_records = group_df.sort_values("risk_percentage", ascending=False).to_dict(orient="records")
        summary["groups"][group_column] = group_records
        if group_records:
            highest = group_records[0]
            lowest = group_records[-1]
            if group_column == "AGE_GROUP":
                summary["highest_risk_age_group"] = highest["group_value"]
                summary["lowest_risk_age_group"] = lowest["group_value"]
            elif group_column == "RIAGENDR":
                summary["highest_risk_gender"] = highest["group_value"]
                summary["lowest_risk_gender"] = lowest["group_value"]
            elif group_column == "RIDRETH1":
                summary["highest_risk_race"] = highest["group_value"]

    disparities = []
    for _, group_df in report_df.groupby("group_column"):
        if len(group_df) < 2:
            continue
        disparities.append(float(group_df["risk_percentage"].max() - group_df["risk_percentage"].min()))
    max_disparity = max(disparities) if disparities else 0.0
    summary["bias_detected"] = max_disparity >= 15.0
    summary["fairness_warning"] = (
        "Some groups show meaningfully different risk rates. Review fairness details before using this result."
        if summary["bias_detected"]
        else "No large risk-rate gap detected in the current fairness report."
    )
    return summary


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application startup and shutdown."""

    try:
        app_instance.state.inference = load_inference_assets()
        app_instance.state.inference_error = None
        logger.info("Inference assets loaded successfully.")
    except Exception as exc:
        app_instance.state.inference = None
        app_instance.state.inference_error = str(exc)
        logger.warning("Inference assets not loaded: %s", exc)

    app_instance.state.mongo_status = "ready" if mongo_db.connect() else "error"
    if app_instance.state.mongo_status == "ready":
        logger.info("MongoDB repository configured successfully.")
    else:
        logger.warning("MongoDB repository unavailable. Using safe empty history responses.")

    yield

    mongo_db.close()
    logger.info("MongoDB connection closed.")


app = FastAPI(
    title="MentalHealthIQ Prediction API",
    description=(
        "PHQ-9 depression severity screening API using real NHANES-style data. "
        "Outputs are screening support, not an independent clinical diagnosis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_inference_assets() -> Dict[str, Any]:
    """Return loaded inference assets or raise service error."""

    inference_assets = getattr(app.state, "inference", None)
    if inference_assets is not None:
        return inference_assets

    try:
        inference_assets = load_inference_assets()
        app.state.inference = inference_assets
        app.state.inference_error = None
        return inference_assets
    except Exception as exc:
        app.state.inference_error = str(exc)
        raise HTTPException(
            status_code=503,
            detail=f"Model/preprocessor not available: {exc}",
        ) from exc


def prediction_warning(severity: str) -> str:
    """Map predicted severity to user-facing follow-up guidance."""

    if severity in {"Minimal", "Mild"}:
        return "Low warning: monitor symptoms and seek support if symptoms persist."
    if severity == "Moderate":
        return "Follow-up suggested: consider contacting a qualified mental health professional."
    return "Urgent follow-up advised: contact a qualified mental health professional promptly."


def recommendation_for_severity(severity: str) -> str:
    """Return doctor-friendly next-step recommendation."""

    if severity == "Minimal":
        return "Continue routine wellness monitoring."
    if severity == "Mild":
        return "Schedule follow-up if symptoms continue or increase."
    if severity == "Moderate":
        return "Mental health follow-up is suggested."
    return "Immediate mental health follow-up is advised."


def explanation_for_result(
    severity: str,
    phq9_total: int,
    risk_band: str,
    model_severity: Optional[str] = None,
) -> str:
    """Return a simple explanation suitable for clinic staff."""

    explanation = (
        f"The PHQ-9 score is {phq9_total}, which falls in the {severity} range. "
        "The primary screening result follows PHQ-9 scoring boundaries."
    )
    if model_severity:
        explanation += f" The trained model estimate is {model_severity}."
    explanation += f" The screening risk band is {risk_band.lower()}."
    return explanation


def patient_fairness_flag(payload: PredictionInput, risk_band: str) -> bool:
    """Flag patients in higher-risk groups when fairness report shows a large gap."""

    try:
        summary = summarize_fairness_records(load_fairness_report())
    except Exception:
        return risk_band == "High"

    age_group = payload.AGE_GROUP or derive_age_group(payload.RIDAGEYR)
    highest_age = summary.get("highest_risk_age_group")
    return bool(summary.get("bias_detected") and (risk_band == "High" or str(age_group) == str(highest_age)))


def predict_payload(payload: PredictionInput) -> Dict[str, Any]:
    """Generate prediction for request payload."""

    inference_assets = ensure_inference_assets()
    model: DepthModel = inference_assets["model"]
    preprocessor: DepthPreprocessor = inference_assets["preprocessor"]
    identity = normalize_patient_identity(payload)

    features = preprocessor.transform(create_input_dataframe(payload))
    prediction_encoded = model.predict(features)
    model_predicted_severity = str(model.label_encoder.inverse_transform(prediction_encoded)[0])
    phq9_total = calculate_phq9_total(payload)
    rule_based_severity = phq9_severity_from_total(phq9_total)
    predicted_severity = rule_based_severity
    model_agreement = model_predicted_severity == rule_based_severity
    risk_band = risk_band_from_severity(predicted_severity)
    model_note = (
        "Model estimate differs from PHQ-9 scoring; clinical screening should follow the PHQ-9 score."
        if not model_agreement
        else "Model estimate agrees with PHQ-9 scoring."
    )

    probabilities: Dict[str, float] = {}
    risk_score = 0.0
    try:
        probability_values = model.predict_proba(features)[0]
        probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.label_encoder.classes_, probability_values)
        }
        risk_score = round(float(max(probability_values)), 4)
    except Exception as exc:
        logger.warning("Probability prediction failed: %s", exc)

    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "patient_id": identity["patient_id"],
        "patient_name": identity["patient_name"],
        "visit_date": payload.visit_date,
        "visit_time": payload.visit_time,
        "doctor_notes": payload.doctor_notes,
        "predicted_severity": predicted_severity,
        "rule_based_severity": rule_based_severity,
        "model_predicted_severity": model_predicted_severity,
        "model_agreement": model_agreement,
        "model_note": model_note,
        "severity": predicted_severity,
        "risk_score": risk_score,
        "risk_percentage": round(risk_score * 100, 1),
        "risk_band": risk_band,
        "phq9_total": phq9_total,
        "recommendation": recommendation_for_severity(predicted_severity),
        "warning": prediction_warning(predicted_severity),
        "fairness_flag": patient_fairness_flag(payload, risk_band),
        "explanation": explanation_for_result(
            predicted_severity,
            phq9_total,
            risk_band,
            model_predicted_severity,
        ),
        "probabilities": probabilities,
        "timestamp": timestamp,
        "model_type": getattr(model, "model_type", "unknown"),
    }


def compare_patient_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the latest patient visit with the previous one."""

    if len(records) < 2:
        return {
            "has_comparison": False,
            "message": "At least two saved visits are needed for comparison.",
            "records": records,
        }

    current = records[0]
    previous = records[1]
    current_score = int(current.get("phq9_total") or current.get("input_data", {}).get("PHQ9_TOTAL") or 0)
    previous_score = int(previous.get("phq9_total") or previous.get("input_data", {}).get("PHQ9_TOTAL") or 0)
    delta = current_score - previous_score

    if delta < 0:
        status = "Improved"
        message = f"Patient condition improved by {abs(delta)} PHQ-9 points."
    elif delta > 0:
        status = "Worsened"
        message = f"Patient condition worsened by {delta} PHQ-9 points."
    else:
        status = "No major change"
        message = "Patient PHQ-9 score did not change."

    return {
        "has_comparison": True,
        "status": status,
        "message": message,
        "change": delta,
        "current": current,
        "previous": previous,
        "trend": list(reversed(records)),
    }


def build_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build dashboard stats from saved prediction records."""

    total = len(predictions)
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    severity_counts: Dict[str, int] = {}
    risk_sum = 0.0

    for item in predictions:
        risk_band = item.get("risk_band") or risk_band_from_severity(item.get("severity", "Minimal"))
        risk_counts[risk_band] = risk_counts.get(risk_band, 0) + 1
        severity = item.get("predicted_severity") or item.get("severity") or "Unknown"
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        risk_sum += float(item.get("risk_score") or 0)

    most_common_severity = max(severity_counts, key=severity_counts.get) if severity_counts else "N/A"
    latest_prediction_time = predictions[0].get("timestamp") if predictions else None

    return {
        "total_predictions": total,
        "high_risk_patients": risk_counts.get("High", 0),
        "medium_risk_patients": risk_counts.get("Medium", 0),
        "low_risk_patients": risk_counts.get("Low", 0),
        "average_risk_score": round(risk_sum / total, 4) if total else 0.0,
        "most_common_severity": most_common_severity,
        "latest_prediction_time": latest_prediction_time,
        "risk_band_counts": risk_counts,
        "severity_counts": severity_counts,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""

    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Return service health without requiring artifacts to exist."""

    mongo_status = mongo_db.health_status()
    app.state.mongo_status = mongo_status
    return {
        "api": "ready",
        "model": "ready" if MODEL_PATH.exists() else "missing",
        "preprocessor": "ready" if PREPROCESSOR_PATH.exists() else "missing",
        "fairness_report": "ready" if FAIRNESS_REPORT_PATH.exists() else "missing",
        "mongo": mongo_status,
        "inference_error": getattr(app.state, "inference_error", None),
    }


@app.post("/predict", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}})
async def predict(payload: PredictionInput) -> PredictionResponse:
    """Predict depression severity without saving result."""

    return PredictionResponse(**predict_payload(payload))


@app.post("/predict-and-save", response_model=PredictionResponse, responses={503: {"model": ErrorResponse}})
async def predict_and_save(payload: PredictionInput) -> PredictionResponse:
    """Predict depression severity and save result to MongoDB."""

    prediction = predict_payload(payload)
    payload_dict = payload.model_dump()
    payload_dict["patient_id"] = prediction.get("patient_id")
    payload_dict["patient_name"] = prediction.get("patient_name")
    if not payload_dict.get("AGE_GROUP"):
        payload_dict["AGE_GROUP"] = derive_age_group(payload.RIDAGEYR)

    now = datetime.now(timezone.utc)
    document = {
        **prediction,
        "input_data": payload_dict,
        "patient_id": prediction.get("patient_id"),
        "patient_name": prediction.get("patient_name"),
        "phq9_total": prediction.get("phq9_total"),
        "severity": prediction["predicted_severity"],
        "predicted_severity": prediction["predicted_severity"],
        "risk_score": prediction["risk_score"],
        "risk_band": prediction.get("risk_band"),
        "recommendation": prediction.get("recommendation"),
        "warning": prediction["warning"],
        "probabilities": prediction.get("probabilities", {}),
        "fairness_flag": bool(prediction.get("fairness_flag")),
        "visit_date": prediction.get("visit_date"),
        "visit_time": prediction.get("visit_time"),
        "doctor_notes": prediction.get("doctor_notes"),
        "timestamp": now,
        "created_at": now,
    }
    inserted_id = mongo_db.save_prediction(document)
    if inserted_id is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available. Start local MongoDB or check MONGODB_URI.",
        )

    return PredictionResponse(
        **prediction,
        saved=True,
        prediction_id=inserted_id,
        predictions_saved=True,
        inserted_id=inserted_id,
    )


@app.get("/predictions", response_model=List[Dict[str, Any]])
async def get_predictions(limit: int = Query(default=100, ge=1, le=500)) -> List[Dict[str, Any]]:
    """Return latest prediction history, or an empty list when MongoDB is unavailable."""

    return mongo_db.get_predictions(limit=limit)


@app.get("/patients/{patient_id}/history", response_model=List[Dict[str, Any]])
async def get_patient_history(
    patient_id: str = ApiPath(..., min_length=1),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Return saved history for one patient, or an empty list when MongoDB is unavailable."""

    return mongo_db.get_patient_history(patient_id=patient_id)[:limit]


@app.get("/patients/{patient_id}/comparison", response_model=Dict[str, Any])
async def get_patient_comparison(patient_id: str = ApiPath(..., min_length=1)) -> Dict[str, Any]:
    """Compare latest saved patient visit with the previous saved visit."""

    return mongo_db.get_patient_comparison(patient_id)


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats() -> Dict[str, Any]:
    """Return dashboard statistics from saved predictions."""

    health = await health_check()
    return {
        **mongo_db.get_stats(),
        "model_status": health["model"],
        "mongo_status": health["mongo"],
        "prediction_history_available": health["mongo"] == "ready",
    }


@app.get("/fairness-report", response_model=FairnessReportResponse)
async def get_fairness_report() -> FairnessReportResponse:
    """Return fairness report records."""

    try:
        records = load_fairness_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FairnessReportResponse(report_path=str(FAIRNESS_REPORT_PATH), records=records)


@app.get("/fairness-summary", response_model=Dict[str, Any])
async def get_fairness_summary() -> Dict[str, Any]:
    """Return simplified fairness report data for frontend cards."""

    try:
        records = load_fairness_report()
    except FileNotFoundError:
        records = []
    return summarize_fairness_records(records)


@app.get("/metrics", response_model=Dict[str, Any], responses={503: {"model": ErrorResponse}})
async def get_metrics() -> Dict[str, Any]:
    """Return model evaluation metrics if available."""

    model: DepthModel = ensure_inference_assets()["model"]
    metrics = getattr(model, "metrics", {}) or {}
    return {
        "metrics": metrics,
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }
