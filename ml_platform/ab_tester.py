"""
A/B Testing module for the Picnic ML Platform.

Provides traffic splitting between model versions with
statistical tracking of experiment results.
"""

import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Holds the outcome of a single A/B test call."""
    model_version: str          # "v1" or "v2"
    prediction: Any             # raw model output
    latency_ms: float           # inference time in milliseconds
    experiment_name: str        # name of the experiment


@dataclass
class ExperimentStats:
    """Aggregated statistics for one model version."""
    calls: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls > 0 else 0.0


class ABTester:
    """
    Routes inference requests between two model versions using
    deterministic hashing so the same request_id always lands on
    the same variant — crucial for reproducible experiments.

    Usage:
        tester = ABTester(experiment_name="rec_v1_vs_v2", traffic_split=0.2)
        result = tester.run(request_id="user_123", model_v1=fn1, model_v2=fn2, features=data)
    """

    def __init__(self, experiment_name: str, traffic_split: float = 0.2):
        """
        Args:
            experiment_name: Unique label for this experiment.
            traffic_split:   Fraction of traffic routed to model_v2 (0.0–1.0).
        """
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")

        self.experiment_name = experiment_name
        self.traffic_split = traffic_split
        self._stats: dict[str, ExperimentStats] = {
            "v1": ExperimentStats(),
            "v2": ExperimentStats(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        request_id: str,
        model_v1: Callable[..., Any],
        model_v2: Callable[..., Any],
        **kwargs: Any,
    ) -> ExperimentResult:
        """
        Route the request to v1 or v2, record stats, and return the result.

        Args:
            request_id: Stable identifier (user_id, transaction_id, etc.)
            model_v1:   Callable for the control model.
            model_v2:   Callable for the treatment model.
            **kwargs:   Forwarded to whichever model is called.
        """
        version = self._assign_version(request_id)
        model_fn = model_v2 if version == "v2" else model_v1
        stats = self._stats[version]

        start = time.perf_counter()
        try:
            prediction = model_fn(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            stats.calls += 1
            stats.total_latency_ms += latency_ms
            logger.debug(
                "[%s] request=%s version=%s latency=%.1fms",
                self.experiment_name, request_id, version, latency_ms,
            )
            return ExperimentResult(
                model_version=version,
                prediction=prediction,
                latency_ms=latency_ms,
                experiment_name=self.experiment_name,
            )
        except Exception as exc:
            stats.errors += 1
            logger.error(
                "[%s] Error in %s: %s", self.experiment_name, version, exc
            )
            raise

    def get_stats(self) -> dict[str, dict]:
        """Return a JSON-serialisable summary of experiment results."""
        return {
            version: {
                "calls": s.calls,
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "errors": s.errors,
                "traffic_pct": (
                    self.traffic_split * 100 if version == "v2"
                    else (1 - self.traffic_split) * 100
                ),
            }
            for version, s in self._stats.items()
        }

    def reset_stats(self) -> None:
        """Reset counters — useful between experiment phases."""
        for s in self._stats.values():
            s.calls = 0
            s.total_latency_ms = 0.0
            s.errors = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_version(self, request_id: str) -> str:
        """
        Deterministically assign v1 or v2 using a hash of the request_id.
        The same request_id always gets the same variant within an experiment.
        """
        seed = f"{self.experiment_name}:{request_id}"
        digest = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
        bucket = (digest % 1000) / 1000.0          # 0.000 – 0.999
        return "v2" if bucket < self.traffic_split else "v1"
