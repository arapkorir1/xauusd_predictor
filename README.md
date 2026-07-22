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
