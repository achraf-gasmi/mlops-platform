"""
Demand Forecaster — Training Script
Dataset : Store Sales Time Series (data/store_sales/)
Model   : Facebook Prophet (additive time-series model)

Run:
    python models/demand_forecast/train.py
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "store_sales"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate_data() -> pd.DataFrame:
    train_path = DATA_DIR / "train.csv"
    if train_path.exists():
        log.info("Loading store sales data from %s", train_path)
        df = pd.read_csv(train_path, parse_dates=["date"])
        # Kaggle store-sales dataset has columns: id, date, store_nbr, family, sales, onpromotion
        date_col = "date" if "date" in df.columns else df.columns[1]
        sales_col = "sales" if "sales" in df.columns else df.columns[4]
        agg = df.groupby(date_col)[sales_col].sum().reset_index()
        agg.columns = ["ds", "y"]
        return agg

    log.warning("Store sales data not found — generating synthetic time series")
    return _synthetic_series()


def _synthetic_series() -> pd.DataFrame:
    """Generate 2 years of daily sales with weekly + yearly seasonality."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=730, freq="D")
    t = np.arange(len(dates))

    # Trend + seasonality + noise
    trend = 1000 + t * 0.3
    weekly = 100 * np.sin(2 * np.pi * t / 7)
    yearly = 200 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 50, len(t))

    sales = np.clip(trend + weekly + yearly + noise, a_min=0, a_max=None)
    return pd.DataFrame({"ds": dates, "y": sales})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> object:
    """Fit one Prophet model on the full time series."""
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Facebook Prophet is required. Install with: pip install prophet"
        )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )

    with mlflow.start_run(run_name="demand_forecast_v1"):
        mlflow.log_params({
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
        })

        # Train on all but last 30 days; validate on holdout
        split_date = df["ds"].max() - pd.Timedelta(days=30)
        train_df = df[df["ds"] <= split_date]
        val_df = df[df["ds"] > split_date]

        model.fit(train_df)

        if len(val_df) > 0:
            future = model.make_future_dataframe(periods=len(val_df))
            forecast = model.predict(future)
            val_forecast = forecast[forecast["ds"].isin(val_df["ds"])]["yhat"]
            mae = float(np.mean(np.abs(val_df["y"].values - val_forecast.values)))
            mlflow.log_metric("mae", round(mae, 2))
            log.info("Validation MAE: %.2f", mae)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_experiment("demand_forecast")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data()
    log.info("Time series: %d rows, from %s to %s", len(df), df["ds"].min().date(), df["ds"].max().date())

    model = train(df)

    artifact = {
        "model": model,
        "training_data": df,   # full series for drift / retraining reference
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
