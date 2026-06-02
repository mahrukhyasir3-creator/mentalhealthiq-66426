"""FastAPI API for MentalHealthIQ depression severity prediction."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint

from mentalhealthiq.db import (
    PredictionDocument,
    PredictionRepository,
    load_repository_from_env,
)
from mentalhealthiq.model import DepthModel
from mentalhealthiq.preprocess import DepthPreprocessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


MODEL_PATH = Path("data/models/model.joblib")
PREPROCESSOR_PATH = Path("data/processed/preprocessor.joblib")
FAIRNESS_REPORT_PATH = Path("data/fairness_reports/fairness_report.csv")


class PredictionInput(BaseModel):
    """Prediction request body."""

    RIDAGEYR: conint(ge=18, le=100) = Field(..., description="Age in years")
    RIAGENDR: conint(ge=0) = Field(..., description="Gender code")
    RIDRETH1: conint(ge=0) = Field(..., description="Race/ethnicity code")
    INDHHIN2: conint(ge=0) = Field(..., description="Household income code")
    DMDEDUC2: conint(ge=0) = Field(..., description="Education code")
    DMDMARTL: conint(ge=0) = Field(..., description="Marital status code")
    AGE_GROUP: str = Field(..., description="Age group category")

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

    severity: str
    risk_score: float
    warning: str
    model_type: str
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
    """Load trained model and preprocessor."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    model = DepthModel.load(MODEL_PATH)
    preprocessor = DepthPreprocessor.load(PREPROCESSOR_PATH)

    return {
        "model": model,
        "preprocessor": preprocessor,
    }


def create_input_dataframe(payload: PredictionInput) -> pd.DataFrame:
    """Convert request payload into model input DataFrame."""

    feature_order = [
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "INDHHIN2",
        "DMDEDUC2",
        "DMDMARTL",
        "AGE_GROUP",
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

    return pd.DataFrame([payload.model_dump()], columns=feature_order)


def load_fairness_report() -> List[Dict[str, Any]]:
    """Load fairness report from CSV."""

    if not FAIRNESS_REPORT_PATH.exists():
        raise FileNotFoundError(f"Fairness report not found: {FAIRNESS_REPORT_PATH}")

    report_df = pd.read_csv(FAIRNESS_REPORT_PATH)
    return report_df.to_dict(orient="records")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application startup and shutdown."""

    try:
        app_instance.state.inference = load_inference_assets()
        logger.info("Inference assets loaded successfully.")
    except Exception as exc:
        app_instance.state.inference = None
        logger.warning("Inference assets not loaded: %s", exc)

    try:
        app_instance.state.repository = load_repository_from_env()
        logger.info("MongoDB repository configured successfully.")
    except Exception as exc:
        app_instance.state.repository = None
        logger.warning("MongoDB repository disabled: %s", exc)

    yield

    repository = getattr(app_instance.state, "repository", None)
    if repository is not None and hasattr(repository, "client"):
        repository.client.close()
        logger.info("MongoDB connection closed.")


app = FastAPI(
    title="MentalHealthIQ Prediction API",
    description=(
        "Depression severity risk-screening API with prediction history, "
        "MongoDB persistence, and fairness report retrieval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_inference_assets() -> Dict[str, Any]:
    """Return loaded inference assets or raise service error."""

    inference_assets = getattr(app.state, "inference", None)

    if inference_assets is None:
        try:
            inference_assets = load_inference_assets()
            app.state.inference = inference_assets
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model/preprocessor not available: {exc}",
            ) from exc

    return inference_assets


def get_repository() -> PredictionRepository:
    """Return MongoDB repository or raise service error."""

    repository = getattr(app.state, "repository", None)

    if repository is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB repository is not configured.",
        )

    return repository


def predict_payload(payload: PredictionInput) -> Dict[str, Any]:
    """Generate prediction for request payload."""

    inference_assets = ensure_inference_assets()

    model: DepthModel = inference_assets["model"]
    preprocessor: DepthPreprocessor = inference_assets["preprocessor"]

    input_df = create_input_dataframe(payload)
    features = preprocessor.transform(input_df)

    prediction_labels = model.predict(features)
    severity = model.label_encoder.inverse_transform(prediction_labels)[0]

    risk_score = 0.0

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
            risk_score = float(max(probabilities))
        except Exception as exc:
            logger.warning("Probability prediction failed: %s", exc)

    return {
        "severity": str(severity),
        "risk_score": round(risk_score, 4),
        "warning": "This prediction is not a medical diagnosis. Consult a qualified professional.",
        "model_type": getattr(model, "model_type", "unknown"),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""

    logger.exception("Unhandled API error: %s", exc)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Return service health."""

    inference_ready = getattr(app.state, "inference", None) is not None
    mongo_ready = getattr(app.state, "repository", None) is not None

    return {
        "status": "healthy",
        "inference_ready": inference_ready,
        "mongo_ready": mongo_ready,
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict(payload: PredictionInput) -> PredictionResponse:
    """Predict depression severity without saving result."""

    prediction = predict_payload(payload)
    return PredictionResponse(**prediction)


@app.post(
    "/predict-and-save",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_and_save(payload: PredictionInput) -> PredictionResponse:
    """Predict depression severity and save result to MongoDB."""

    prediction = predict_payload(payload)
    repository = get_repository()

    document = PredictionDocument(
        input_data=payload.model_dump(),
        severity=prediction["severity"],
        risk_score=prediction["risk_score"],
        warning=prediction["warning"],
        timestamp=datetime.utcnow(),
    )

    inserted_id = repository.insert_prediction(document)

    return PredictionResponse(
        **prediction,
        predictions_saved=True,
        inserted_id=inserted_id,
    )


@app.get(
    "/predictions",
    response_model=List[Dict[str, Any]],
    responses={503: {"model": ErrorResponse}},
)
async def get_predictions(
    limit: int = Query(default=100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Return latest prediction history."""

    repository = get_repository()
    return repository.fetch_predictions(limit=limit)


@app.get("/fairness-report", response_model=FairnessReportResponse)
async def get_fairness_report() -> FairnessReportResponse:
    """Return fairness report records."""

    try:
        records = load_fairness_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FairnessReportResponse(
        report_path=str(FAIRNESS_REPORT_PATH),
        records=records,
    )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics() -> Dict[str, Any]:
    """Return model evaluation metrics if available."""

    inference_assets = ensure_inference_assets()
    model: DepthModel = inference_assets["model"]

    metrics = getattr(model, "metrics", {}) or {}

    return {
        "metrics": metrics,
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }
