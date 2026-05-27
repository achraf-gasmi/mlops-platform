"""
Monitoring module for the Picnic ML Platform.

Aggregates drift reports and A/B test statistics from all models
into a single /metrics payload served by the FastAPI layer.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Runtime metrics for a single model."""
    model_name: str
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    last_drift_report: Optional[dict] = None
    last_ab_stats: Optional[dict] = None
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_requests, 2)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_errors / self.total_requests, 4)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": self.error_rate,
            "last_drift_report": self.last_drift_report,
            "last_ab_stats": self.last_ab_stats,
            "last_updated": self.last_updated,
        }


class PlatformMonitor:
    """
    Central registry that each router writes to.

    Usage:
        monitor = PlatformMonitor()                    # singleton at app startup
        monitor.record_request("fraud_detector", latency_ms=12.3)
        monitor.update_drift("fraud_detector", drift_summary)
        metrics = monitor.get_all_metrics()
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelMetrics] = {}

    # ------------------------------------------------------------------
    # Write path — called from routers
    # ------------------------------------------------------------------

    def record_request(
        self,
        model_name: str,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        m = self._get_or_create(model_name)
        m.total_requests += 1
        m.total_latency_ms += latency_ms
        if error:
            m.total_errors += 1
        m.last_updated = time.time()

    def update_drift(self, model_name: str, drift_summary: dict) -> None:
        """Store the latest drift summary from DriftDetector.summary()."""
        m = self._get_or_create(model_name)
        m.last_drift_report = drift_summary
        m.last_updated = time.time()

    def update_ab_stats(self, model_name: str, ab_stats: dict) -> None:
        """Store the latest A/B stats from ABTester.get_stats()."""
        m = self._get_or_create(model_name)
        m.last_ab_stats = ab_stats
        m.last_updated = time.time()

    # ------------------------------------------------------------------
    # Read path — called by /metrics endpoint
    # ------------------------------------------------------------------

    def get_all_metrics(self) -> dict[str, Any]:
        return {
            "platform": "picnic-ml-platform",
            "models": {
                name: m.to_dict() for name, m in self._models.items()
            },
        }

    def get_model_metrics(self, model_name: str) -> Optional[dict]:
        m = self._models.get(model_name)
        return m.to_dict() if m else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, model_name: str) -> ModelMetrics:
        if model_name not in self._models:
            self._models[model_name] = ModelMetrics(model_name=model_name)
        return self._models[model_name]


# Module-level singleton — imported by routers
monitor = PlatformMonitor()
