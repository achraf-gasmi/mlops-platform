# Picnic ML Platform

A production-grade machine learning platform that replicates the core ML use cases at [Picnic Technologies](https://picnic.app) — Europe's fastest-growing online grocery. Built as a platform engineering exercise: every model is wrapped with A/B testing, drift monitoring, FastAPI inference endpoints, MLflow experiment tracking, and Docker containerisation.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Client / Consumers                          │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ HTTP
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FastAPI Inference Server (:8000)                 │
│                                                                     │
│   POST /recommend       POST /classify-ticket                       │
│   POST /detect-fraud    POST /forecast-demand                       │
│   GET  /health          GET  /metrics                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Platform Layer                             │  │
│  │                                                              │  │
│  │  ┌─────────────────┐   ┌─────────────────┐                  │  │
│  │  │   ABTester      │   │  DriftDetector  │                  │  │
│  │  │  (traffic split)│   │  (KS + PSI)     │                  │  │
│  │  └────────┬────────┘   └────────┬────────┘                  │  │
│  │           │                     │                            │  │
│  │           └──────────┬──────────┘                            │  │
│  │                      ▼                                       │  │
│  │             PlatformMonitor → /metrics                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐  │
│  │Recommendation│ │   Ticket     │ │    Fraud     │ │ Demand   │  │
│  │  XGBoost     │ │  TF-IDF +LR  │ │  Isolation   │ │ Prophet  │  │
│  │              │ │              │ │  Forest      │ │          │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘  │
└────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │   MLflow     │  │  S3 Bucket   │  │  Monitoring  │
     │  (:5000)     │  │  (artifacts) │  │  (:9090)     │
     └──────────────┘  └──────────────┘  └──────────────┘
```

---

## ML Models

### 1. Recommendation System
| | |
|---|---|
| **Dataset** | Instacart Market Basket |
| **Model** | XGBoost binary classifier (reorder probability per user×product pair) |
| **Input** | `customer_id`, `purchase_history` (product_id, purchase_count, avg_cart_position, total_orders) |
| **Output** | Top-N products ranked by predicted reorder probability |
| **Endpoint** | `POST /recommend` |

### 2. LLM Ticket Classifier
| | |
|---|---|
| **Dataset** | Customer Support Tickets |
| **Model** | TF-IDF (bigrams, 20k features) + Logistic Regression |
| **Input** | `ticket_text` (raw customer message) |
| **Output** | `category` (billing/delivery/quality/returns/other) + `confidence` score |
| **Endpoint** | `POST /classify-ticket` |

### 3. Fraud Detector
| | |
|---|---|
| **Dataset** | Credit Card Fraud (Kaggle) |
| **Model** | Isolation Forest (unsupervised anomaly detection) |
| **Input** | Transaction features V1–V28 (PCA-anonymised) + `Amount` |
| **Output** | `fraud_probability` (0–1), `is_fraud` boolean, `anomaly_score` |
| **Endpoint** | `POST /detect-fraud` |

### 4. Demand Forecaster
| | |
|---|---|
| **Dataset** | Store Sales Time Series |
| **Model** | Facebook Prophet (additive decomposition with weekly + yearly seasonality) |
| **Input** | `product_id`, `historical_sales` (date + sales pairs) |
| **Output** | 7-day daily demand forecast with confidence intervals |
| **Endpoint** | `POST /forecast-demand` |

---

## Platform Features

### A/B Testing
Every model endpoint routes a configurable percentage of traffic (default 20%) to a challenger model version. Routing is deterministic — the same `request_id` always lands on the same variant — ensuring reproducible experiments. Stats available at `GET /metrics`.

```python
ab_tester = ABTester(experiment_name="fraud_v1_vs_v2", traffic_split=0.2)
result = ab_tester.run(request_id="txn_123", model_v1=model_fn, model_v2=challenger_fn)
```

### Drift Detection
Every batch of inference requests is compared against the training distribution using:
- **Kolmogorov-Smirnov test** — detects distributional shifts (p < 0.05 → drift)
- **Population Stability Index (PSI)** — PSI > 0.1 → mild drift; > 0.2 → severe drift

```python
detector = DriftDetector(feature_names=["amount", "V1", "V2"])
detector.set_reference(X_train)
reports = detector.detect(X_new)
```

### MLflow Experiment Tracking
Each `train.py` logs parameters and metrics to MLflow automatically. View runs at `http://localhost:5000`.

### Monitoring
`GET /metrics` returns per-model request counts, average latency, error rates, latest drift reports, and A/B test statistics in a single JSON payload.

---

## Running Locally with Docker

### Prerequisites
- Docker Desktop installed and running
- 4GB+ RAM available

### Start all services

```bash
git clone https://github.com/YOUR_ORG/picnic-ml-platform.git
cd picnic-ml-platform

# Train all models first (generates model.pkl files)
pip install -r requirements.txt
python models/recommendation/train.py
python models/ticket_classifier/train.py
python models/fraud_detector/train.py
python models/demand_forecast/train.py

# Start the full stack
docker compose up --build
```

Services available:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Metrics**: http://localhost:9090/metrics

### Quick API test

```bash
# Health check
curl http://localhost:8000/health

# Ticket classification
curl -X POST http://localhost:8000/classify-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "My order was delivered to the wrong address", "ticket_id": "t001"}'

# Fraud detection
curl -X POST http://localhost:8000/detect-fraud \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_001", "Amount": 2500.00, "V1": -3.5, "V2": 2.1}'

# Demand forecast
curl -X POST http://localhost:8000/forecast-demand \
  -H "Content-Type: application/json" \
  -d '{"product_id": "banana_001", "historical_sales": [], "horizon_days": 7}'

# Platform metrics
curl http://localhost:8000/metrics
```

---

## Training Models Independently

Each model is self-contained and can be trained without the rest of the platform:

```bash
# From the project root:
python models/recommendation/train.py
python models/ticket_classifier/train.py
python models/fraud_detector/train.py
python models/demand_forecast/train.py
```

If the real datasets are not present in `data/`, each script falls back to a synthetic dataset so training always works out of the box.

---

## Adding Real Datasets

| Model | Dataset | Source |
|---|---|---|
| Recommendation | Instacart Market Basket 2017 | [Kaggle](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data) → extract to `data/instacart/` |
| Ticket Classifier | Customer Support Tickets | Any CSV with `text` + `category` columns → `data/support_tickets/tickets.csv` |
| Fraud Detector | Credit Card Fraud Detection | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → `data/creditcard/creditcard.csv` |
| Demand Forecast | Store Sales - Time Series | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) → `data/store_sales/train.csv` |

---

## Terraform Deployment (AWS)

```bash
cd terraform/

# 1. Set your credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# 2. Create a key pair in the AWS console, then:
terraform init
terraform apply -var="key_pair_name=my-key"

# 3. Get the API URL
terraform output api_url
```

Destroys cleanly with `terraform destroy`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.111, Uvicorn, Pydantic v2 |
| **ML** | scikit-learn, XGBoost, Facebook Prophet, Isolation Forest |
| **Math** | NumPy, Pandas, SciPy |
| **Tracking** | MLflow 2.13 |
| **Container** | Docker, docker-compose v2 |
| **IaC** | Terraform ≥ 1.6, AWS provider ~5.0 |
| **Cloud** | AWS EC2 (t3.medium), S3, IAM |
| **Python** | 3.11 |

---

## Project Structure

```
picnic-ml-platform/
├── api/
│   ├── main.py                  # FastAPI app, /health, /metrics
│   └── routers/
│       ├── recommendation.py    # POST /recommend
│       ├── tickets.py           # POST /classify-ticket
│       ├── fraud.py             # POST /detect-fraud
│       └── forecast.py          # POST /forecast-demand
├── models/
│   ├── recommendation/          # XGBoost reorder model
│   ├── ticket_classifier/       # TF-IDF + LR pipeline
│   ├── fraud_detector/          # Isolation Forest
│   └── demand_forecast/         # Facebook Prophet
├── platform/
│   ├── ab_tester.py             # Traffic splitting (deterministic hashing)
│   ├── drift_detector.py        # KS test + PSI drift detection
│   └── monitoring.py            # Metrics aggregator
├── data/                        # Dataset directories (gitignored)
├── terraform/                   # AWS infrastructure as code
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
