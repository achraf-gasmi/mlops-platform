"""
Picnic ML Platform — FastAPI Application Entry Point

Mounts all model routers and exposes platform-level endpoints:
  GET  /health   — liveness probe
  GET  /metrics  — drift scores + A/B test results for all models
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ml_platform.monitoring import monitor
from api.routers import recommendation, tickets, fraud, forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Picnic ML Platform starting up…")
    yield
    log.info("Picnic ML Platform shutting down.")


app = FastAPI(
    title="Picnic ML Platform",
    description=(
        "Production ML platform replicating Picnic Technologies' core use cases. "
        "Includes recommendation, ticket classification, fraud detection, and demand forecasting — "
        "each wrapped with A/B testing and drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Mount model routers
# ---------------------------------------------------------------------------
app.include_router(recommendation.router, prefix="/recommend", tags=["Recommendation"])
app.include_router(tickets.router, prefix="/classify-ticket", tags=["Tickets"])
app.include_router(fraud.router, prefix="/detect-fraud", tags=["Fraud"])
app.include_router(forecast.router, prefix="/forecast-demand", tags=["Forecast"])


# ---------------------------------------------------------------------------
# Platform-level endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Platform"])
def health_check() -> dict:
    """Liveness probe — returns platform uptime and status."""
    return {
        "status": "healthy",
        "platform": "picnic-ml-platform",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.get("/metrics", tags=["Platform"])
def metrics() -> JSONResponse:
    """
    Aggregated drift scores and A/B test statistics for all models.
    Useful for dashboards and alerting.
    """
    return JSONResponse(content=monitor.get_all_metrics())
