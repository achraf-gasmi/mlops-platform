"""
Recommendation Router
POST /recommend — returns top-5 product suggestions with A/B testing + drift monitoring.
"""

import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml_platform.ab_tester import ABTester
from ml_platform.drift_detector import DriftDetector
from ml_platform.monitoring import monitor

log = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy-load model to avoid import errors if model.pkl doesn't exist yet
# ---------------------------------------------------------------------------
_predict_module = None

def _get_predict():
    global _predict_module
    if _predict_module is None:
        from models.recommendation import predict as mod
        _predict_module = mod
    return _predict_module


# ---------------------------------------------------------------------------
# Platform singletons (one per router)
# ---------------------------------------------------------------------------
ab_tester = ABTester(experiment_name="recommendation_v1_vs_v2", traffic_split=0.2)
drift_detector = DriftDetector(
    feature_names=["purchase_count", "avg_cart_position", "total_orders"]
)
_drift_initialised = False


def _ensure_drift_reference() -> None:
    """Lazily initialise drift reference from training data."""
    global _drift_initialised
    if not _drift_initialised:
        try:
            training_data = _get_predict().get_training_data()
            drift_detector.set_reference(training_data)
            _drift_initialised = True
        except Exception as exc:
            log.warning("Could not set drift reference: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PurchaseItem(BaseModel):
    product_id: int = Field(..., description="Product identifier")
    purchase_count: int = Field(..., ge=1, description="Times purchased")
    avg_cart_position: float = Field(..., ge=1.0, description="Avg position in cart")
    total_orders: int = Field(..., ge=1, description="Customer total order count")


class RecommendRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    purchase_history: list[PurchaseItem] = Field(
        ..., min_length=1, description="Recent purchase items"
    )
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations")

    @field_validator("customer_id")
    @classmethod
    def customer_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_id must not be blank")
        return v.strip()


class RecommendedProduct(BaseModel):
    product_id: int
    reorder_probability: float
    rank: int


class RecommendResponse(BaseModel):
    customer_id: str
    recommended_products: list[RecommendedProduct]
    model_version: str
    latency_ms: float
    drift_detected: bool
    model: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Generate personalised product recommendations.

    - Routes 80% of traffic to model_v1, 20% to model_v2.
    - Runs KS + PSI drift detection on incoming features.
    - Records metrics to the platform monitor.
    """
    _ensure_drift_reference()
    predict = _get_predict()
    history_dicts = [item.model_dump() for item in request.purchase_history]

    # Drift detection on request features
    drift_detected = False
    try:
        X_new = np.array([
            [h["purchase_count"], h["avg_cart_position"], h["total_orders"]]
            for h in history_dicts
        ], dtype=float)
        reports = drift_detector.detect(X_new)
        summary = drift_detector.summary(reports)
        drift_detected = summary["drift_detected"]
        monitor.update_drift("recommendation", summary)
    except Exception as exc:
        log.warning("Drift detection failed: %s", exc)

    # A/B test between v1 and v2 (v2 is a stub returning same model here;
    # swap out predict.recommend_v2 when a real challenger is trained)
    try:
        result = ab_tester.run(
            request_id=request.customer_id,
            model_v1=lambda **kw: predict.recommend(**kw),
            model_v2=lambda **kw: predict.recommend(**kw),
            customer_id=request.customer_id,
            purchase_history=history_dicts,
            top_n=request.top_n,
        )
    except Exception as exc:
        log.error("Recommendation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    monitor.record_request("recommendation", latency_ms=result.latency_ms)
    monitor.update_ab_stats("recommendation", ab_tester.get_stats())

    payload = result.prediction
    return RecommendResponse(
        customer_id=payload["customer_id"],
        recommended_products=payload["recommended_products"],
        model_version=result.model_version,
        latency_ms=round(result.latency_ms, 2),
        drift_detected=drift_detected,
        model=payload.get("model", "recommendation_xgboost_v1"),
    )
