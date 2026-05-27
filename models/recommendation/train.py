"""
Recommendation System — Training Script
Dataset : Instacart Market Basket (data/instacart/)
Model   : XGBoost binary classifier per product
          (predicts reorder probability for user×product pairs)

Run:
    python models/recommendation/train.py
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "instacart"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate_data() -> pd.DataFrame:
    """
    Load Instacart CSVs if present; otherwise generate synthetic data
    so the training script is always runnable for demo purposes.
    """
    orders_path = DATA_DIR / "orders.csv"
    products_path = DATA_DIR / "order_products__prior.csv"

    # Accept both canonical and this project's renamed file
    op_path = DATA_DIR / "all_order_products.csv"
    if not products_path.exists() and op_path.exists():
        products_path = op_path

    if orders_path.exists() and products_path.exists():
        log.info("Loading Instacart data from %s", DATA_DIR)
        orders = pd.read_csv(orders_path)
        op = pd.read_csv(products_path)
        # Support both dataset layouts
        merge_cols = ["order_id", "user_id"]
        if "order_number" in orders.columns:
            merge_cols.append("order_number")
        df = op.merge(orders[merge_cols], on="order_id")
        if "order_number" not in df.columns:
            # Derive a proxy order sequence from timestamp if available
            if "timestamp" in orders.columns:
                orders["order_number"] = (
                    orders.sort_values("timestamp")
                    .groupby("user_id")
                    .cumcount() + 1
                )
                df = op.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")
            else:
                df["order_number"] = 1
        if "reordered" not in df.columns:
            df["reordered"] = 0   # no reorder signal → treat all as first purchase
        return _build_features(df)

    log.warning("Instacart data not found — generating synthetic dataset")
    return _synthetic_data(n_users=500, n_products=200, n_rows=20_000)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw order lines into one row per user×product pair."""
    feat = (
        df.groupby(["user_id", "product_id"])
        .agg(
            purchase_count=("add_to_cart_order", "count"),
            avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    # Label: product purchased more than once → reordered
    if "reordered" in df.columns:
        reordered_sum = (
            df.groupby(["user_id", "product_id"])["reordered"].sum().reset_index()
        )
        reordered_sum.columns = ["user_id", "product_id", "reordered_sum"]
        feat = feat.merge(reordered_sum, on=["user_id", "product_id"])
        feat["label"] = (feat["reordered_sum"] > 0).astype(int)
    else:
        # Derive from purchase count: bought >1 time → likely reorder
        feat["label"] = (feat["purchase_count"] > 1).astype(int)

    # User-level features
    user_stats = (
        df.groupby("user_id")
        .agg(total_orders=("order_number", "max"))
        .reset_index()
    )
    feat = feat.merge(user_stats, on="user_id")
    return feat


def _synthetic_data(n_users: int, n_products: int, n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    user_ids = rng.integers(1, n_users + 1, n_rows)
    product_ids = rng.integers(1, n_products + 1, n_rows)
    purchase_count = rng.integers(1, 20, n_rows)
    avg_cart_position = rng.uniform(1, 15, n_rows)
    total_orders = rng.integers(5, 50, n_rows)
    # Reorder probability loosely correlated with purchase_count
    prob = purchase_count / purchase_count.max()
    label = rng.binomial(1, prob)

    return pd.DataFrame({
        "user_id": user_ids,
        "product_id": product_ids,
        "purchase_count": purchase_count,
        "avg_cart_position": avg_cart_position,
        "total_orders": total_orders,
        "label": label,
    })


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FEATURE_COLS = ["purchase_count", "avg_cart_position", "total_orders"]


def train(df: pd.DataFrame) -> XGBClassifier:
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="recommendation_v1"):
        mlflow.log_params(model.get_params())
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        mlflow.log_metric("roc_auc", auc)
        log.info("Validation ROC-AUC: %.4f", auc)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_experiment("recommendation_system")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data()
    log.info("Dataset shape: %s", df.shape)

    model = train(df)

    # Save model + metadata
    artifact = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "training_data": df[FEATURE_COLS].values,  # kept for drift detection
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
