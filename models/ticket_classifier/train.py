"""
LLM Ticket Classifier — Training Script
Dataset : Customer Support Tickets (data/support_tickets/)
Model   : TF-IDF + Logistic Regression pipeline

Categories: billing | delivery | quality | returns | other

Run:
    python models/ticket_classifier/train.py
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "support_tickets"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.pkl"

CATEGORIES = ["billing", "delivery", "quality", "returns", "other"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate_data() -> pd.DataFrame:
    # Accept either the canonical name or the Kaggle download name
    csv_path = DATA_DIR / "tickets.csv"
    alt_path  = DATA_DIR / "customer_support_tickets.csv"
    found = csv_path if csv_path.exists() else (alt_path if alt_path.exists() else None)

    if found:
        log.info("Loading ticket data from %s", found)
        df = pd.read_csv(found)
        # Map Kaggle columns → canonical (text, category)
        # Map Kaggle columns → (text, category)
        # Use both Ticket Description and Subject for richer text signal
        if "Ticket Description" in df.columns:
            # Combine subject + description for richer signal
            subject_col = "Ticket Subject" if "Ticket Subject" in df.columns else None
            if subject_col:
                df["text"] = (
                    df[subject_col].fillna("") + " " + df["Ticket Description"].fillna("")
                ).str.strip()
            else:
                df["text"] = df["Ticket Description"].fillna("")

            type_col  = "Ticket Type"    if "Ticket Type"    in df.columns else None
            subj_col  = "Ticket Subject" if "Ticket Subject" in df.columns else None

            if type_col:
                # Map Ticket Type → our 5 canonical categories
                label_map = {
                    "Billing inquiry":      "billing",
                    "Technical issue":      "other",
                    "Product inquiry":      "quality",
                    "Refund request":       "returns",
                    "Cancellation request": "returns",
                }
                df["category"] = df[type_col].map(label_map).fillna("other")
            elif subj_col:
                # Derive from subject keywords
                def _map_subject(s: str) -> str:
                    s = str(s).lower()
                    if any(w in s for w in ["bill", "charg", "payment", "invoice"]): return "billing"
                    if any(w in s for w in ["deliver", "ship", "arrival", "late"]): return "delivery"
                    if any(w in s for w in ["quality", "damage", "defect", "broken", "rotten"]): return "quality"
                    if any(w in s for w in ["return", "refund", "cancel"]): return "returns"
                    return "other"
                df["category"] = df[subj_col].apply(_map_subject)
            else:
                df["category"] = "other"

            return df[["text", "category"]].dropna()
        return df

    log.warning("Ticket data not found — generating synthetic dataset")
    return _synthetic_data(n=4000)


_TEMPLATES = {
    "billing": [
        "I was charged twice for my order last week",
        "My invoice shows an incorrect amount",
        "I need a refund for the overcharge on my account",
        "Why was my credit card billed again?",
        "Payment issue with my subscription",
        "I cannot see my billing history",
        "Duplicate charge appeared on my statement",
    ],
    "delivery": [
        "My order has not arrived yet",
        "The delivery was late by two days",
        "Package was left in the rain and is damaged",
        "Wrong address was used for my delivery",
        "Delivery driver did not ring the bell",
        "My groceries were delivered to the wrong house",
        "Order marked as delivered but I never received it",
    ],
    "quality": [
        "The vegetables I received were rotten",
        "Meat product was expired when delivered",
        "Yogurt container was open and spilled",
        "Product quality was very poor",
        "I found a foreign object in my food",
        "Items were crushed during delivery",
        "Wrong product substituted without notice",
    ],
    "returns": [
        "I want to return the damaged bread I received",
        "How do I initiate a return for my order?",
        "I need to send back the wrong item",
        "Can I get a replacement for the broken eggs?",
        "Return policy for fresh produce",
        "I was sent the wrong brand and want a refund",
        "How long does a return take to process?",
    ],
    "other": [
        "I want to change my delivery slot",
        "Can I add items to my existing order?",
        "How do I cancel my subscription?",
        "I cannot log in to my account",
        "Please update my delivery address",
        "I would like to leave feedback about the app",
        "How do I add a new payment method?",
    ],
}


def _synthetic_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    texts, labels = [], []
    per_class = n // len(CATEGORIES)
    for cat, templates in _TEMPLATES.items():
        for _ in range(per_class):
            base = rng.choice(templates)
            # Light augmentation: append random filler
            filler = rng.choice([
                "", " Please help.", " Urgent!", " Thank you.",
                " This is unacceptable.", " I am very disappointed.",
            ])
            texts.append(base + filler)
            labels.append(cat)
    return pd.DataFrame({"text": texts, "category": labels})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> Pipeline:
    X = df["text"].values
    y = df["category"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    with mlflow.start_run(run_name="ticket_classifier_v1"):
        mlflow.log_params({
            "tfidf_max_features": 20_000,
            "tfidf_ngram_range": "(1,2)",
            "clf_C": 5.0,
        })

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        report = classification_report(y_val, y_pred, output_dict=True)
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])

        log.info("Accuracy: %.4f", report["accuracy"])
        log.info("Macro F1: %.4f", report["macro avg"]["f1-score"])
        log.info("\n%s", classification_report(y_val, y_pred))

    return pipeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_experiment("ticket_classifier")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data()
    log.info("Dataset shape: %s  Categories: %s", df.shape, df["category"].value_counts().to_dict())

    model = train(df)

    artifact = {
        "model": model,
        "categories": CATEGORIES,
        "training_texts": df["text"].tolist(),  # kept for drift reference
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
