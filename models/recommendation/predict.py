"""
Recommendation System — Inference
Loads the trained XGBoost model and returns the top-N product suggestions
for a given customer based on their purchase history features.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Module-level cache — loaded once on first call
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
        log.info("Recommendation model loaded from %s", MODEL_PATH)
    return _artifact


def get_training_data() -> np.ndarray:
    """Return training feature matrix — used by drift detector."""
    return _load_model()["training_data"]


def recommend(
    customer_id: str,
    purchase_history: list[dict],
    top_n: int = 5,
) -> dict:
    """
    Generate top-N product recommendations.

    Args:
        customer_id:      Identifier for logging / A/B routing.
        purchase_history: List of dicts with keys:
                          product_id, purchase_count, avg_cart_position,
                          total_orders
        top_n:            Number of products to recommend.

    Returns:
        dict with recommended_products list and model metadata.
    """
    artifact = _load_model()
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    if not purchase_history:
        return {
            "customer_id": customer_id,
            "recommended_products": [],
            "message": "No purchase history provided",
        }

    df = pd.DataFrame(purchase_history)

    # Ensure all required columns are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].values.astype(float)

    # Predict reorder probability for each product
    probs = model.predict_proba(X)[:, 1]
    df["reorder_prob"] = probs

    # Sort and return top-N
    top = df.nlargest(top_n, "reorder_prob")

    recommendations = [
        {
            "product_id": int(row.get("product_id", rank)),
            "reorder_probability": round(float(row["reorder_prob"]), 4),
            "rank": rank + 1,
        }
        for rank, (_, row) in enumerate(top.iterrows())
    ]

    return {
        "customer_id": customer_id,
        "recommended_products": recommendations,
        "model": "recommendation_xgboost_v1",
    }
