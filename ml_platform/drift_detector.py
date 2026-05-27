"""
Data Drift Detection module for the Picnic ML Platform.

Uses two complementary methods:
  - Kolmogorov-Smirnov (KS) test  — catches distributional shifts
  - Population Stability Index (PSI) — measures population drift over time

Both are industry standards for production ML monitoring.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of drift detection for one feature."""
    feature_name: str
    ks_statistic: float
    ks_p_value: float
    psi_score: float
    is_drifted: bool            # True if either test flags drift
    drift_method: str           # which method triggered the flag


class DriftDetector:
    """
    Monitors incoming inference data against a reference distribution
    captured at training time.

    Usage:
        detector = DriftDetector(feature_names=["age", "amount"])
        detector.set_reference(X_train)          # call once after training
        report = detector.detect(X_new)          # call on each batch
    """

    # Thresholds — tune to your risk tolerance
    KS_P_VALUE_THRESHOLD = 0.05   # below this → statistically significant drift
    PSI_WARNING_THRESHOLD = 0.1   # mild drift
    PSI_CRITICAL_THRESHOLD = 0.2  # severe drift (action required)

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
        n_bins: int = 10,
    ):
        """
        Args:
            feature_names: Column labels for logging. If None, uses col indices.
            n_bins:        Number of histogram bins for PSI calculation.
        """
        self.feature_names = feature_names or []
        self.n_bins = n_bins
        self._reference: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, X: np.ndarray) -> None:
        """
        Store reference distribution from the training set.
        Must be called before detect().
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._reference = X
        # Pre-compute per-feature bin edges for PSI (avoids recomputation)
        self._bin_edges: list[np.ndarray] = []
        for col in range(X.shape[1]):
            edges = np.histogram_bin_edges(X[:, col], bins=self.n_bins)
            self._bin_edges.append(edges)
        logger.info("Drift reference set: %d samples, %d features", *X.shape)

    def detect(self, X_new: np.ndarray) -> list[DriftReport]:
        """
        Compare X_new against the stored reference.

        Returns a DriftReport per feature; raises if reference not set.
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect()")

        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        n_features = self._reference.shape[1]
        reports: list[DriftReport] = []

        for col in range(n_features):
            name = (
                self.feature_names[col] if col < len(self.feature_names)
                else f"feature_{col}"
            )
            ref_col = self._reference[:, col]
            new_col = X_new[:, col]

            ks_stat, ks_p = self._ks_test(ref_col, new_col)
            psi = self._psi(ref_col, new_col, self._bin_edges[col])

            ks_drift = ks_p < self.KS_P_VALUE_THRESHOLD
            psi_drift = psi > self.PSI_WARNING_THRESHOLD
            is_drifted = ks_drift or psi_drift

            method = "none"
            if ks_drift and psi_drift:
                method = "ks+psi"
            elif ks_drift:
                method = "ks"
            elif psi_drift:
                method = "psi"

            if is_drifted:
                logger.warning(
                    "Drift detected in '%s': KS p=%.4f, PSI=%.4f",
                    name, ks_p, psi,
                )

            reports.append(DriftReport(
                feature_name=name,
                ks_statistic=round(float(ks_stat), 4),
                ks_p_value=round(float(ks_p), 4),
                psi_score=round(float(psi), 4),
                is_drifted=is_drifted,
                drift_method=method,
            ))

        return reports

    def summary(self, reports: list[DriftReport]) -> dict:
        """Collapse per-feature reports into a single platform-level dict."""
        drifted = [r for r in reports if r.is_drifted]
        return {
            "total_features": len(reports),
            "drifted_features": len(drifted),
            "drift_detected": len(drifted) > 0,
            "avg_psi": round(
                float(np.mean([r.psi_score for r in reports])), 4
            ) if reports else 0.0,
            "features": [
                {
                    "name": r.feature_name,
                    "psi": r.psi_score,
                    "ks_p_value": r.ks_p_value,
                    "drifted": r.is_drifted,
                    "method": r.drift_method,
                }
                for r in reports
            ],
        }

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_test(
        reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        """Two-sample KS test. Returns (statistic, p_value)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.ks_2samp(reference, current)
        return result.statistic, result.pvalue

    def _psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bin_edges: np.ndarray,
    ) -> float:
        """
        Population Stability Index.
        PSI = Σ (actual% − expected%) × ln(actual% / expected%)
        """
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalise to proportions; add small epsilon to avoid log(0)
        eps = 1e-8
        ref_pct = ref_counts / (ref_counts.sum() + eps) + eps
        cur_pct = cur_counts / (cur_counts.sum() + eps) + eps

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)   # clamp numerical negatives to 0
