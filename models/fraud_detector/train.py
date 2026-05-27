"""
Fraud Detector — Training Script
Dataset : Credit Card Fraud (data/creditcard/)
Model   : Isolation Forest (unsupervised anomaly detection)

Run:
    python models/fraud_detector/train.py
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "creditcard"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.pkl"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "creditcard.csv"
    if csv_path.exists():
        log.info("Loading credit card data from %s", csv_path)
        return pd.read_csv(csv_path)

    log.warning("Credit card data not found — generating synthetic dataset")
    return _synthetic_data(n_normal=10_000, n_fraud=200)


def _synthetic_data(n_normal: int, n_fraud: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_features = 28  # V1–V28

    # Normal transactions: Gaussian around zero (PCA-transformed features)
    normal = rng.standard_normal((n_normal, n_features))
    normal_amount = rng.exponential(scale=80, size=(n_normal, 1))
    normal_label = np.zeros(n_normal)

    # Fraudulent transactions: shifted distribution with higher amounts
    fraud = rng.standard_normal((n_fraud, n_features)) * 2 + 1.5
    fraud_amount = rng.exponential(scale=300, size=(n_fraud, 1))
    fraud_label = np.ones(n_fraud)

    X = np.vstack([normal, fraud])
    amounts = np.vstack([normal_amount, fraud_amount])
    y = np.concatenate([normal_label, fraud_label])

    cols = {f"V{i+1}": X[:, i] for i in range(n_features)}
    cols["Amount"] = amounts[:, 0]
    cols["Class"] = y
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS].values
    y = df["Class"].values if "Class" in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train only on non-fraud (or all data — Isolation Forest handles outliers)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,   # expected ~2% fraud rate
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="fraud_detector_v1"):
        mlflow.log_params({
            "n_estimators": 200,
            "contamination": 0.02,
        })

        model.fit(X_scaled)

        # Isolation Forest returns -1 (anomaly) / 1 (normal)
        raw_preds = model.predict(X_scaled)
        is_fraud_pred = (raw_preds == -1).astype(int)

        if y is not None:
            # Scores are more negative for anomalies → invert for probability
            scores = -model.score_samples(X_scaled)
            # Normalise to [0,1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            auc = roc_auc_score(y, scores)
            p, r, f1, _ = precision_recall_fscore_support(
                y, is_fraud_pred, average="binary", zero_division=0
            )
            mlflow.log_metrics({
                "roc_auc": round(auc, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            })
            log.info("ROC-AUC: %.4f  Precision: %.4f  Recall: %.4f  F1: %.4f", auc, p, r, f1)

    return {"model": model, "scaler": scaler}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_experiment("fraud_detector")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data()
    log.info("Dataset shape: %s", df.shape)
    if "Class" in df.columns:
        log.info("Fraud rate: %.2f%%", df["Class"].mean() * 100)

    artifact = train(df)
    artifact["feature_cols"] = FEATURE_COLS
    artifact["training_data"] = df[FEATURE_COLS].values  # for drift detection

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
