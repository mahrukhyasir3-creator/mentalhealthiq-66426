"""Local-first MongoDB helpers for MentalHealthIQ prediction history."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


logger = logging.getLogger(__name__)

DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_DATABASE = "mentalhealthiq"
DEFAULT_COLLECTION = "predictions"


def _load_environment() -> None:
    """Load .env from the current project when present."""

    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _serialize_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB-only values into JSON-safe values."""

    serialized = {}
    for key, value in document.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, dict):
            serialized[key] = _serialize_document(value)
        elif isinstance(value, list):
            serialized[key] = [
                _serialize_document(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


class MongoDBHelper:
    """Small MongoDB facade used by FastAPI endpoints and health checks."""

    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        _load_environment()
        self.uri = uri or os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or DEFAULT_MONGODB_URI
        self.database_name = database_name or os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE)
        self.collection_name = collection_name or os.getenv("MONGODB_COLLECTION", DEFAULT_COLLECTION)
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to MongoDB and create indexes. Returns False on failure."""

        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            database = self.client[self.database_name]
            self.collection = database[self.collection_name]
            self._create_indexes()
            self.connected = True
            logger.info(
                "Connected to MongoDB collection '%s.%s' at %s",
                self.database_name,
                self.collection_name,
                self.uri,
            )
            return True
        except (PyMongoError, ServerSelectionTimeoutError, OSError) as exc:
            self.connected = False
            self.collection = None
            if self.client is not None:
                self.client.close()
            self.client = None
            logger.warning("MongoDB connection failed: %s", exc)
            return False

    def _create_indexes(self) -> None:
        """Create indexes used by history and dashboard queries."""

        if self.collection is None:
            return

        self.collection.create_index([("patient_id", ASCENDING)])
        self.collection.create_index([("created_at", DESCENDING)])
        self.collection.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])

    def _ensure_connected(self) -> bool:
        """Return True if a working connection exists, reconnecting if needed."""

        if self.client is not None and self.collection is not None:
            try:
                self.client.admin.command("ping")
                self.connected = True
                return True
            except PyMongoError:
                self.connected = False

        return self.connect()

    def health_status(self) -> str:
        """Return 'ready' when ping succeeds, otherwise 'error'."""

        return "ready" if self._ensure_connected() else "error"

    def save_prediction(self, prediction: Dict[str, Any]) -> Optional[str]:
        """Insert a prediction document and return its ID, or None if unavailable."""

        if not self._ensure_connected() or self.collection is None:
            return None

        document = dict(prediction)
        now = _utc_now()
        document.setdefault("created_at", now)
        document.setdefault("timestamp", now)

        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as exc:
            logger.warning("Failed to save prediction: %s", exc)
            return None

    def get_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return latest predictions sorted by created_at descending."""

        if not self._ensure_connected() or self.collection is None:
            return []

        try:
            cursor = (
                self.collection
                .find({})
                .sort("created_at", DESCENDING)
                .limit(int(limit))
            )
            return [_serialize_document(document) for document in cursor]
        except PyMongoError as exc:
            logger.warning("Failed to fetch predictions: %s", exc)
            return []

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Return one patient's saved visits sorted newest first."""

        if not patient_id or not self._ensure_connected() or self.collection is None:
            return []

        try:
            cursor = (
                self.collection
                .find({"patient_id": patient_id})
                .sort("created_at", DESCENDING)
            )
            return [_serialize_document(document) for document in cursor]
        except PyMongoError as exc:
            logger.warning("Failed to fetch patient history: %s", exc)
            return []

    def get_patient_comparison(self, patient_id: str) -> Dict[str, Any]:
        """Compare latest and previous prediction for a patient."""

        history = self.get_patient_history(patient_id)
        if len(history) < 2:
            return {
                "patient_id": patient_id,
                "has_enough_history": False,
                "has_comparison": False,
                "latest": history[0] if history else None,
                "previous": None,
                "phq9_change": 0,
                "risk_score_change": 0.0,
                "status": "insufficient_history",
                "summary": "At least two saved visits are needed for comparison.",
                "records": history,
            }

        latest = history[0]
        previous = history[1]
        latest_phq = int(latest.get("phq9_total") or 0)
        previous_phq = int(previous.get("phq9_total") or 0)
        phq9_change = latest_phq - previous_phq
        risk_score_change = round(float(latest.get("risk_score") or 0) - float(previous.get("risk_score") or 0), 4)

        if phq9_change < 0:
            status = "improved"
            summary = f"Patient improved by {abs(phq9_change)} PHQ-9 points since last visit."
        elif phq9_change > 0:
            status = "worsened"
            summary = f"Patient worsened by {phq9_change} PHQ-9 points since last visit."
        else:
            status = "no_change"
            summary = "No major PHQ-9 score change since last visit."

        return {
            "patient_id": patient_id,
            "has_enough_history": True,
            "has_comparison": True,
            "latest": latest,
            "current": latest,
            "previous": previous,
            "phq9_change": phq9_change,
            "risk_score_change": risk_score_change,
            "status": status,
            "summary": summary,
            "message": summary,
            "change": phq9_change,
            "trend": list(reversed(history)),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return dashboard stats from saved predictions."""

        predictions = self.get_predictions(limit=500)
        total = len(predictions)

        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        severity_distribution: Dict[str, int] = {}
        risk_sum = 0.0
        phq_sum = 0.0

        for prediction in predictions:
            risk_band = prediction.get("risk_band") or "Low"
            risk_counts[risk_band] = risk_counts.get(risk_band, 0) + 1
            severity = prediction.get("predicted_severity") or prediction.get("severity") or "Unknown"
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            risk_sum += float(prediction.get("risk_score") or 0)
            phq_sum += float(prediction.get("phq9_total") or 0)

        latest_prediction_time = predictions[0].get("created_at") if predictions else None

        return {
            "total_predictions": total,
            "high_risk_count": risk_counts.get("High", 0),
            "medium_risk_count": risk_counts.get("Medium", 0),
            "low_risk_count": risk_counts.get("Low", 0),
            "high_risk_patients": risk_counts.get("High", 0),
            "medium_risk_patients": risk_counts.get("Medium", 0),
            "low_risk_patients": risk_counts.get("Low", 0),
            "average_risk_score": round(risk_sum / total, 4) if total else 0.0,
            "average_phq9_total": round(phq_sum / total, 2) if total else 0.0,
            "latest_prediction_time": latest_prediction_time,
            "severity_distribution": severity_distribution,
            "severity_counts": severity_distribution,
            "risk_band_distribution": risk_counts,
            "risk_band_counts": risk_counts,
        }

    def close(self) -> None:
        """Close the MongoDB client."""

        if self.client is not None:
            self.client.close()
        self.client = None
        self.collection = None
        self.connected = False


mongo_db = MongoDBHelper()
