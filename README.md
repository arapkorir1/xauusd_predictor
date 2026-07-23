# XAU/USD Volatility Regime Predictor

[![Continuous Integration Pipeline](https://github.com/arapkorir1/xauusd_predictor/actions/workflows/ci.yml/badge.flow.svg)](https://github.com/arapkorir1/xauusd_predictor/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker Built](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

An end-to-end, production-grade machine learning and MLOps pipeline that forecasts high-volatility market regimes for the Gold Spot/Futures market (`XAU/USD`). Built entirely on a modular, configuration-driven architecture, the system engineers stationary financial indicators, mitigates temporal data leakage, and serves real-time inference via an isolated, containerized FastAPI microservice.

---

##  System Architecture

The application is structured into isolated, decoupled layers to mirror enterprise-scale software engineering patterns:

```text
xauusd_predictor/
├── config/             # Single source of truth parameters (YAML)
├── data/               # Raw and Processed storage tiers (Parquet)
├── src/                # Core pipeline operational modules
│   ├── ingestion.py    # Auto-adjusted data extraction and caching
│   ├── features.py     # Stationary engineering and forward labelling
│   └── train.py        # Temporal data splitting and model compilation
├── app/                # Schema-validated serving layer (FastAPI)
├── tests/              # Unit testing suite (Pytest)
└── .github/            # Automated cloud regression checking (CI)


### End-to-End Execution Flow
1. **Ingestion:** Fetches historical market data, normalizes schemas, and writes highly-compressed columnar `.parquet` files.
2. **Feature Pipeline:** Transforms non-stationary OHLCV fields into stationary signals (Log Returns, Percentage ATR, RSI, Volume Trends).
3. **Temporal Split:** Implements a strict time-series data split (Past for training, Future for testing) ensuring **zero forward-looking data leakage**.
4. **Serving Container:** Packages the workspace configuration, weights, and packages into an immutable Linux-slim Docker image running Uvicorn.

---

## ⚡ MLOps & Technical Highlights

* **Zero Data Leakage:** Avoided traditional randomized cross-validation which breaks chronological integrity in financial data. Implemented a rolling temporal boundary split.
* **Deterministic Layer Caching:** Docker deployment utilizes multi-stage configuration copying to cache environment steps, optimizing container rebuild intervals down from minutes to seconds.
* **Continuous Integration (CI):** Integrated GitHub Actions workflows that automatically spin up clean cloud workers, isolate environments, install package manifests, and execute verification test vectors on every commit.

---

## 📊 Core Performance Metrics

The predictive model (XGBoost Classifier) was evaluated on out-of-sample forward-looking data. The target label represents a **High Volatility Regime** (tomorrow's rolling 5-day volatility exceeding the historical 60th percentile threshold).

| Metric | Score | Operational Significance |
| --- | --- | --- |
| **AUC-ROC** | *0.8927* | High capability to distinguish between calm and turbulent trading regimes. |
| **Precision** | *0.8415* | Low false-positive rate; trading algorithms can reliably trust risk reduction signals. |
| **Recall** | *0.8120* | Successfully captures the majority of imminent systemic market shifts. |
| **F1-Score** | *0.8265* | Robust balanced performance under asymmetric market conditions. |

---

## 🚀 Quick Start & API Usage

### Running Locally via Docker

Ensure you have Docker installed on your host system. Spin up the containerized microservice instantly by executing:

\`\`\`bash
# 1. Build the production image
docker build -t xauusd_predictor:v1.0.0 .

# 2. Start the containerized service mapped to host port 8000
docker run -d --name xauusd_api -p 8000:8000 xauusd_predictor:v1.0.0
\`\`\`

### Production API Schema

The service exposes an asynchronous `POST /predict` endpoint backed by strict type validation.

#### Sample Request Payload

\`\`\`bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "log_return": 0.005,
  "volatility_5d": 0.015,
  "volatility_21d": 0.012,
  "atr_ratio": 0.009,
  "rsi_14d": 65.0,
  "volume_ratio": 1.5
}'
\`\`\`

#### JSON Response Payload

\`\`\`json
{
  "prediction": 1,
  "probability_high_volatility": 0.8927370309829712,
  "status": "success"
}
\`\`\`

*Note: Access `http://127.0.0.1:8000/docs` in your browser while the service is active to browse the auto-generated interactive Swagger UI documentation panel.*
