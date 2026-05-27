"""
Demand Forecaster — Inference
Loads the trained Prophet model and generates a 7-day demand forecast.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
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
        log.info("Demand forecast model loaded from %s", MODEL_PATH)
    return _artifact


def get_training_series() -> pd.DataFrame:
    """Return training DataFrame (ds, y) for drift reference."""
    return _load_model()["training_data"]


def forecast(
    product_id: str,
    historical_sales: list[dict],
    horizon_days: int = 7,
) -> dict:
    """
    Forecast demand for the next `horizon_days` days.

    Args:
        product_id:       Product identifier (used for logging).
        historical_sales: List of dicts with 'date' (YYYY-MM-DD) and 'sales'
                          keys. If empty, uses the globally trained model.
        horizon_days:     Number of future days to forecast (default 7).

    Returns:
        dict with daily_forecast list and summary statistics.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: pip install prophet")

    artifact = _load_model()
    model = artifact["model"]

    # If product-specific history provided, fine-tune a fresh model
    if historical_sales:
        df = pd.DataFrame(historical_sales)
        df = df.rename(columns={"date": "ds", "sales": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)

        # Use the global model's changepoints as a warm start; re-fit on product data
        product_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        product_model.fit(df)
        future = product_model.make_future_dataframe(periods=horizon_days)
        forecast_df = product_model.predict(future).tail(horizon_days)
    else:
        # Fall back to global model trained on all-store aggregate
        future = model.make_future_dataframe(periods=horizon_days)
        forecast_df = model.predict(future).tail(horizon_days)

    daily = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted_demand": max(0.0, round(float(row["yhat"]), 1)),
            "lower_bound": max(0.0, round(float(row["yhat_lower"]), 1)),
            "upper_bound": max(0.0, round(float(row["yhat_upper"]), 1)),
        }
        for _, row in forecast_df.iterrows()
    ]

    total = sum(d["predicted_demand"] for d in daily)
    return {
        "product_id": product_id,
        "horizon_days": horizon_days,
        "daily_forecast": daily,
        "total_predicted_demand": round(total, 1),
        "avg_daily_demand": round(total / horizon_days, 1),
        "model": "demand_forecast_prophet_v1",
    }
