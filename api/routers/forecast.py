"""
Demand Forecast Router
POST /forecast-demand — generates 7-day demand forecast with A/B testing + drift monitoring.
"""

import logging

import numpy as np
import pandas as pd
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
        from models.demand_forecast import predict as mod
        _predict_module = mod
    return _predict_module


# ---------------------------------------------------------------------------
# Platform singletons
# ---------------------------------------------------------------------------
ab_tester = ABTester(experiment_name="demand_forecast_v1_vs_v2", traffic_split=0.2)
drift_detector = DriftDetector(feature_names=["sales_value"])
_drift_initialised = False


def _ensure_drift_reference() -> None:
    global _drift_initialised
    if not _drift_initialised:
        try:
            training_df = _get_predict().get_training_series()
            sales = training_df["y"].values.reshape(-1, 1)
            drift_detector.set_reference(sales)
            _drift_initialised = True
        except Exception as exc:
            log.warning("Could not set drift reference: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SalesRecord(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    sales: float = Field(..., ge=0.0, description="Observed sales units")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            pd.to_datetime(v)
        except Exception:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD.")
        return v


class ForecastRequest(BaseModel):
    product_id: str = Field(..., description="Product or SKU identifier")
    historical_sales: list[SalesRecord] = Field(
        default=[],
        description="Historical daily sales. If empty, the global model is used."
    )
    horizon_days: int = Field(7, ge=1, le=90, description="Days to forecast ahead")

    @field_validator("product_id")
    @classmethod
    def product_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("product_id must not be blank")
        return v.strip()


class DailyForecast(BaseModel):
    date: str
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    product_id: str
    horizon_days: int
    daily_forecast: list[DailyForecast]
    total_predicted_demand: float
    avg_daily_demand: float
    model_version: str
    latency_ms: float
    drift_detected: bool
    model: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=ForecastResponse)
def forecast_demand(request: ForecastRequest) -> ForecastResponse:
    """
    Forecast daily demand for a product over the next N days.

    - Uses Facebook Prophet under the hood.
    - If historical_sales provided, fine-tunes on that product's series.
    - Otherwise falls back to the globally-trained aggregate model.
    """
    _ensure_drift_reference()
    predict = _get_predict()

    history_dicts = [r.model_dump() for r in request.historical_sales]

    # Drift detection on sales values
    drift_detected = False
    if request.historical_sales:
        try:
            sales_values = np.array(
                [[r.sales] for r in request.historical_sales], dtype=float
            )
            reports = drift_detector.detect(sales_values)
            summary = drift_detector.summary(reports)
            drift_detected = summary["drift_detected"]
            monitor.update_drift("demand_forecast", summary)
        except Exception as exc:
            log.warning("Drift detection failed: %s", exc)

    # A/B test
    try:
        result = ab_tester.run(
            request_id=request.product_id,
            model_v1=lambda **kw: predict.forecast(**kw),
            model_v2=lambda **kw: predict.forecast(**kw),
            product_id=request.product_id,
            historical_sales=history_dicts,
            horizon_days=request.horizon_days,
        )
    except Exception as exc:
        log.error("Demand forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    monitor.record_request("demand_forecast", latency_ms=result.latency_ms)
    monitor.update_ab_stats("demand_forecast", ab_tester.get_stats())

    payload = result.prediction
    return ForecastResponse(
        product_id=payload["product_id"],
        horizon_days=payload["horizon_days"],
        daily_forecast=payload["daily_forecast"],
        total_predicted_demand=payload["total_predicted_demand"],
        avg_daily_demand=payload["avg_daily_demand"],
        model_version=result.model_version,
        latency_ms=round(result.latency_ms, 2),
        drift_detected=drift_detected,
        model=payload.get("model", "demand_forecast_prophet_v1"),
    )
