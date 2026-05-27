"""
Ticket Classifier — Inference
Classifies a customer support ticket into one of 5 categories
and returns a confidence score.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

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
        log.info("Ticket classifier loaded from %s", MODEL_PATH)
    return _artifact


def get_training_texts() -> list[str]:
    """Return training texts for drift reference."""
    return _load_model()["training_texts"]


def classify(ticket_text: str) -> dict:
    """
    Classify a support ticket.

    Args:
        ticket_text: Raw customer support message.

    Returns:
        dict with category, confidence, and per-class probabilities.
    """
    if not ticket_text or not ticket_text.strip():
        return {
            "category": "other",
            "confidence": 0.0,
            "probabilities": {},
            "model": "ticket_classifier_tfidf_lr_v1",
        }

    artifact = _load_model()
    pipeline = artifact["model"]
    categories = artifact["categories"]

    probs = pipeline.predict_proba([ticket_text])[0]
    class_labels = pipeline.classes_

    pred_idx = int(np.argmax(probs))
    category = class_labels[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(class_labels, probs)
        },
        "model": "ticket_classifier_tfidf_lr_v1",
    }
