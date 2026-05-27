"""
Ticket Classification Router
POST /classify-ticket — categorises a support ticket with A/B testing + drift monitoring.
"""

import logging
import hashlib

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml_platform.ab_tester import ABTester
from ml_platform.drift_detector import DriftDetector
from ml_platform.monitoring import monitor

log = logging.getLogger(__name__)
router = APIRouter()

_predict_module = None


def _get_predict():
    global _predict_module
    if _predict_module is None:
        from models.ticket_classifier import predict as mod
        _predict_module = mod
    return _predict_module


# ---------------------------------------------------------------------------
# Platform singletons
# ---------------------------------------------------------------------------
ab_tester = ABTester(experiment_name="ticket_classifier_v1_vs_v2", traffic_split=0.2)

# For text drift, we monitor the hash distribution of vocabulary features
# (a lightweight proxy — real text drift requires embedding comparison)
drift_detector = DriftDetector(feature_names=["text_length", "word_count"])
_drift_initialised = False


def _text_features(text: str) -> list[float]:
    """Extract lightweight numeric features from text for drift detection."""
    return [float(len(text)), float(len(text.split()))]


def _ensure_drift_reference() -> None:
    global _drift_initialised
    if not _drift_initialised:
        try:
            training_texts = _get_predict().get_training_texts()
            X_ref = np.array([_text_features(t) for t in training_texts], dtype=float)
            drift_detector.set_reference(X_ref)
            _drift_initialised = True
        except Exception as exc:
            log.warning("Could not set drift reference: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ..., min_length=5, max_length=2000,
        description="Raw customer support message"
    )
    ticket_id: str = Field(
        default="anon",
        description="Optional ticket ID for A/B routing consistency"
    )

    @field_validator("ticket_text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float
    drift_detected: bool
    model: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=TicketResponse)
def classify_ticket(request: TicketRequest) -> TicketResponse:
    """
    Classify a customer support ticket into: billing | delivery | quality | returns | other.

    - 20% of traffic routed to model_v2 (challenger).
    - Text-length drift detection as a lightweight proxy for distribution shift.
    """
    _ensure_drift_reference()
    predict = _get_predict()

    # Drift detection
    drift_detected = False
    try:
        X_new = np.array([_text_features(request.ticket_text)], dtype=float)
        reports = drift_detector.detect(X_new)
        summary = drift_detector.summary(reports)
        drift_detected = summary["drift_detected"]
        monitor.update_drift("ticket_classifier", summary)
    except Exception as exc:
        log.warning("Drift detection failed: %s", exc)

    # A/B test
    try:
        result = ab_tester.run(
            request_id=request.ticket_id,
            model_v1=lambda **kw: predict.classify(**kw),
            model_v2=lambda **kw: predict.classify(**kw),
            ticket_text=request.ticket_text,
        )
    except Exception as exc:
        log.error("Ticket classification failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    monitor.record_request("ticket_classifier", latency_ms=result.latency_ms)
    monitor.update_ab_stats("ticket_classifier", ab_tester.get_stats())

    payload = result.prediction
    return TicketResponse(
        ticket_id=request.ticket_id,
        category=payload["category"],
        confidence=payload["confidence"],
        probabilities=payload["probabilities"],
        model_version=result.model_version,
        latency_ms=round(result.latency_ms, 2),
        drift_detected=drift_detected,
        model=payload.get("model", "ticket_classifier_tfidf_lr_v1"),
    )
