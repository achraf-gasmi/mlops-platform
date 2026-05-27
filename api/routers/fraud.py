"""
Fraud Detection Router
POST /detect-fraud — scores a transaction with A/B testing + drift monitoring.
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml_platform.ab_tester import ABTester
from ml_platform.drift_detector import DriftDetector
from ml_platform.monitoring import monitor

log = logging.getLogger(__name__)
router = APIRouter()

_predict_module = None


def _get_predict():
    global _predict_module
    if _predict_module is None:
        from models.fraud_detector import predict as mod
        _predict_module = mod
    return _predict_module


# ---------------------------------------------------------------------------
# Platform singletons
# ---------------------------------------------------------------------------
ab_tester = ABTester(experiment_name="fraud_detector_v1_vs_v2", traffic_split=0.2)

_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
drift_detector = DriftDetector(feature_names=_FEATURE_NAMES[:5])  # monitor first 5 PCA dims + Amount
_drift_initialised = False


def _ensure_drift_reference() -> None:
    global _drift_initialised
    if not _drift_initialised:
        try:
            training_data = _get_predict().get_training_data()
            # Monitor a subset of features to keep drift report concise
            drift_detector.set_reference(training_data[:, :5])
            _drift_initialised = True
        except Exception as exc:
            log.warning("Could not set drift reference: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FraudRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    # V1–V28: PCA-transformed credit card features (anonymised)
    V1: float = 0.0;  V2: float = 0.0;  V3: float = 0.0;  V4: float = 0.0
    V5: float = 0.0;  V6: float = 0.0;  V7: float = 0.0;  V8: float = 0.0
    V9: float = 0.0;  V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(0.0, ge=0.0, description="Transaction amount in EUR")

    def to_feature_dict(self) -> dict:
        return {
            **{f"V{i}": getattr(self, f"V{i}") for i in range(1, 29)},
            "Amount": self.Amount,
        }


class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    anomaly_score: float
    model_version: str
    latency_ms: float
    drift_detected: bool
    model: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=FraudResponse)
def detect_fraud(request: FraudRequest) -> FraudResponse:
    """
    Score a credit card transaction for fraud.

    - Returns fraud_probability (0–1) and boolean is_fraud flag.
    - 20% of traffic routed to model_v2 for challenger testing.
    - Drift detection on PCA features V1–V5.
    """
    _ensure_drift_reference()
    predict = _get_predict()

    features = request.to_feature_dict()

    # Drift detection (first 5 PCA components as proxy)
    drift_detected = False
    try:
        X_new = np.array(
            [[features[f"V{i}"] for i in range(1, 6)]], dtype=float
        )
        reports = drift_detector.detect(X_new)
        summary = drift_detector.summary(reports)
        drift_detected = summary["drift_detected"]
        monitor.update_drift("fraud_detector", summary)
    except Exception as exc:
        log.warning("Drift detection failed: %s", exc)

    # A/B test
    try:
        result = ab_tester.run(
            request_id=request.transaction_id,
            model_v1=lambda **kw: predict.detect_fraud(**kw),
            model_v2=lambda **kw: predict.detect_fraud(**kw),
            transaction_features=features,
        )
    except Exception as exc:
        log.error("Fraud detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    monitor.record_request("fraud_detector", latency_ms=result.latency_ms)
    monitor.update_ab_stats("fraud_detector", ab_tester.get_stats())

    payload = result.prediction
    return FraudResponse(
        transaction_id=request.transaction_id,
        fraud_probability=payload["fraud_probability"],
        is_fraud=payload["is_fraud"],
        anomaly_score=payload["anomaly_score"],
        model_version=result.model_version,
        latency_ms=round(result.latency_ms, 2),
        drift_detected=drift_detected,
        model=payload.get("model", "fraud_detector_isolation_forest_v1"),
    )
