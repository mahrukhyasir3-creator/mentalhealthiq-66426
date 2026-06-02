"""
MongoDB prediction repository and connection helpers.

This module implements a repository pattern for saving and retrieving prediction
history in MongoDB Atlas. It loads environment configuration using python-dotenv.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


logger = logging.getLogger(__name__)


@dataclass
class PredictionDocument:
    """Prediction document schema for MongoDB storage."""
    input_data: Dict[str, Any]
    severity: str
    risk_score: float
    warning: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_data': self.input_data,
            'severity': self.severity,
            'risk_score': float(self.risk_score),
            'warning': self.warning,
            'timestamp': self.timestamp.isoformat() + 'Z'
        }


class PredictionRepository:
    """Repository for MongoDB prediction persistence."""

    def __init__(
        self,
        mongo_uri: str,
        database_name: str = 'mentalhealthiq',
        collection_name: str = 'predictions'
    ):
        """Initialize repository with MongoDB connection settings."""
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self) -> None:
        """Connect to MongoDB Atlas and resolve the predictions collection."""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            db = self.client[self.database_name]
            self.collection = db[self.collection_name]
            logger.info("Connected to MongoDB Atlas collection '%s.%s'", self.database_name, self.collection_name)
        except PyMongoError as exc:
            logger.error("Failed to connect to MongoDB: %s", exc)
            raise

    def insert_prediction(self, document: PredictionDocument) -> str:
        """Insert a prediction document and return its inserted ID."""
        if self.collection is None:
            raise RuntimeError("MongoDB collection is not initialized")

        payload = document.to_dict()
        result = self.collection.insert_one(payload)
        logger.info("Saved prediction document with id=%s", result.inserted_id)
        return str(result.inserted_id)

    def fetch_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch latest prediction documents ordered by timestamp."""
        if self.collection is None:
            raise RuntimeError("MongoDB collection is not initialized")

        cursor = self.collection.find({}, projection={'_id': False}).sort('timestamp', -1).limit(limit)
        return [doc for doc in cursor]


def load_repository_from_env() -> PredictionRepository:
    """Create PredictionRepository using environment configuration."""
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    from os import getenv

    mongo_uri = getenv('MONGODB_URI')
    database_name = getenv('MONGODB_DATABASE', 'mentalhealthiq')
    collection_name = getenv('MONGODB_COLLECTION', 'predictions')

    if not mongo_uri:
        raise ValueError('MONGODB_URI environment variable is required for MongoDB connection')

    return PredictionRepository(
        mongo_uri=mongo_uri,
        database_name=database_name,
        collection_name=collection_name
    )
