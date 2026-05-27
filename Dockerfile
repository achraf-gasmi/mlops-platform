# ─────────────────────────────────────────────────────────────────────────────
# Picnic ML Platform — Production Dockerfile
# Base: Python 3.11 slim (smaller attack surface than full image)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies needed by Prophet / scipy / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# Install Python dependencies (cached layer)
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Copy source code
# ─────────────────────────────────────────────────────────────────────────────
COPY ml_platform/ ml_platform/
COPY models/   models/
COPY api/      api/
COPY data/     data/

# Create __init__.py files so Python treats directories as packages
RUN touch ml_platform/__init__.py \
    && touch models/__init__.py \
    && touch models/recommendation/__init__.py \
    && touch models/ticket_classifier/__init__.py \
    && touch models/fraud_detector/__init__.py \
    && touch models/demand_forecast/__init__.py \
    && touch api/__init__.py

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration
# ─────────────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# FastAPI port
EXPOSE 8000

# Health check — Docker will restart unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn — single worker per container; scale via docker-compose replicas
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
