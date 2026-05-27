"""
Fraud Detector — Inference
Wraps the trained Isolation Forest model to return fraud probability
and a boolean fraud flag for each transaction.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pkl"

_artifact: Optional[dict] = None


def _load_model() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
        log.info("Fraud detector loaded from %s", MODEL_PATH)
    return _artifact


def get_training_data() -> np.ndarray:
    """Return raw training features for drift reference."""
    return _load_model()["training_data"]


def detect_fraud(transaction_features: dict) -> dict:
    """
    Score a single transaction.

    Args:
        transaction_features: Dict mapping feature names (V1–V28, Amount)
                              to numeric values.

    Returns:
        dict with fraud_probability (0–1), is_fraud boolean, and anomaly score.
    """
    artifact = _load_model()
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    # Build feature vector in correct column order
    X = np.array(
        [transaction_features.get(col, 0.0) for col in feature_cols],
        dtype=float,
    ).reshape(1, -1)

    X_scaled = scaler.transform(X)

    # score_samples: more negative = more anomalous
    raw_score = float(model.score_samples(X_scaled)[0])
    # Normalise: invert and clip to [0,1]
    # Typical IF scores range from ~-0.7 (anomaly) to ~0.1 (normal)
    fraud_probability = float(np.clip((-raw_score - 0.0) / 0.7, 0.0, 1.0))
    is_fraud = bool(model.predict(X_scaled)[0] == -1)

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": is_fraud,
        "anomaly_score": round(raw_score, 4),
        "model": "fraud_detector_isolation_forest_v1",
    }
