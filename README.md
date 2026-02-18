---
title: Pearls AQI API
sdk: docker
app_port: 7860
---

# Pearls AQI Predictor API (Karachi)

FastAPI serving 3-day AQI forecasts.
Data: Open-Meteo → MongoDB Atlas.
Models: Linear Regression, XGBoost, LightGBM tracked in DagsHub MLflow.

## Endpoints
- GET /health
- GET /predict?days=3

# Pearls_AQI_Predictor

End-to-end, serverless AQI forecasting for Karachi using Open-Meteo data, MongoDB Atlas storage, and DagsHub MLflow tracking—hourly ingestion, daily retraining, and 3-day predictions.

**Live Application**: https://pearls-aqi-predictor-karachi.streamlit.app

**API Endpoint**: https://huggingface.co/spaces/sAbrarShah/pearls-aqi-api

**Submission Portal**: https://shine.10pearls.com/candidate/submissions

Short description:
This repository contains an end-to-end Air Quality Index (AQI) prediction system built with hourly feature ingestion, automated feature engineering, multiple ML models, a production REST API, and an interactive dashboard. The system is designed for deployment in serverless/cloud environments and stores features/models in MongoDB Atlas with MLflow/DagsHub tracking.

---

Table of Contents
- Project Overview
- Architecture
- Features
- Technology Stack
- Live Deployments (placeholders)
- Quick Start (local)
- Configuration
- Project Structure
- Usage
- Models & Performance
- CI / CD
- API Reference
- Troubleshooting
- Contributing
- License
- Acknowledgements

---

## Project Overview

Pearls_AQI_Predictor forecasts AQI for Karachi (3 days) using meteorological & pollutant data from Open‑Meteo and a set of engineered features. The repository contains:

- An hourly ingestion pipeline to fetch and store raw and engineered features
- Feature engineering producing a comprehensive set of features (temporal, cyclical, rolling statistics, pollutant interactions)
- Multiple ML models (Linear Regression, XGBoost, LightGBM)
- Model registry & versioning (MongoDB / MLflow/DagsHub integration)
- FastAPI-based prediction API
- Streamlit dashboard for visualizing forecasts and model metrics
- GitHub Actions workflows for scheduled feature and training pipelines

Objectives
- Automated hourly ingestion and feature generation
- Daily model retraining and automated model selection
- Production-ready deployment (API + dashboard)
- Model performance tracking and versioning

---

## System Architecture

High-level flow:

Open-Meteo (data) → Hourly ingestion & preprocessing → Feature engineering
→ MongoDB (feature store & model registry / MLflow tracking) → Training pipeline (daily)
→ Best model saved → FastAPI (predictions) ↔ Streamlit dashboard (presentation)

Components:
- Data Collection: Open‑Meteo API (backfill + hourly)
- Feature Engineering: Dagshub, temporal, cyclical, rolling stats, derived ratios, pollutant indices
- Model Training: scikit-learn, Linear Regression, XGBoost, LightGBM
- API: FastAPI for serving predictions & model metadata
- Dashboard: Streamlit for visualization & user interaction
- CI/CD: GitHub Actions for scheduled pipelines and deployment integration
- Storage: MongoDB Atlas for features and models; optional MLflow/DagsHub tracking

---

## Key Features

- Hourly automated ingestion and feature generation
- Multiple model types (automatic best-model selection)
- Model registry with versioning, metrics, and metadata
- FastAPI endpoints: health, list models, predict (3 days)
- Streamlit dashboard with interactive charts, alerts and model insights
- GitHub Actions workflows for hourly features and daily training
- Config-driven: city coordinates, timezone, model hyperparameters, scheduling

---

## Technology Stack

- Language: Python (predominant)
- Web / API: FastAPI, Uvicorn
- Dashboard: Streamlit, Plotly
- ML: scikit-learn, Linear Regression, XGBoost, LightGBM, NumPy, Pandas
- DB: MongoDB Atlas
- Tracking: MLflow or DagsHub
- CI/CD: GitHub Actions
- Hosting: Streamlit Cloud for Dashboard + Hugging Face for API

---

## Live Deployment

Replace the placeholders below with your actual deployment URLs (if applicable):

- Streamlit Dashboard: https://pearls-aqi-predictor-karachi.streamlit.app
- FastAPI Base URL: https://huggingface.co/spaces/sAbrarShah/pearls-aqi-api

---

## Quick Start (local development)

Prerequisites:
- Python 3.11.9 (3.11 recommended)
- Git
- MongoDB Atlas account (or a local MongoDB instance)
- DagsHub / MLflow tracking account

1. Clone the repository
```bash
git clone https://github.com/sAbrarShah/Pearls_AQI_Predictor.git
cd Pearls_AQI_Predictor
```

2. Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (example below)
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
MONGO_DB=aqi_database
MONGO_RAW_COLLECTION=raw_hourly
MONGO_CLEAN_COLLECTION=clean_hourly
MONGO_FORECAST_COLLECTION=forecasts_daily
MONGO_MODEL_RUNS_COLLECTION=model_runs_daily

DAGSHUB_REPO_OWNER=<owner>
DAGSHUB_REPO_NAME=<repo>
DAGSHUB_USERNAME=<username>
DAGSHUB_USER_TOKEN=<token>

HTTP_TIMEOUT_SEC=25
HTTP_RETRIES=4
HTTP_BACKOFF_SEC=1.0
```

5. (First-time only) Backfill historical data (recommended)
```bash
python scripts/backfill.py
```

6. Run hourly ingestion once
```bash
python scripts/run_hourly.py
```

7. Train models and write training metadata to Mongo + DagsHub MLflow
```bash
python scripts/run_daily_train.py
```

8. Generate daily forecasts for all models and store to Mongo
```bash
python scripts/run_daily_predict.py
```

9. Run FastAPI locally
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

10. Run Streamlit dashboard locally
```bash
streamlit run apps/dashboard.py
```
Open:
Dashboard: http://localhost:8501
API docs: http://localhost:8000/docs
API predict: http://localhost:8000/predict?days=3

---

## Configuration

Primary configuration files and environment variables:
- `.env` — environment variables (MONGODB_URI, DAGSHUB_TOKEN, etc.)
- `aqi/config.py` — city config and other defaults (latitude, longitude, timezone)
- `.streamlit/secrets.toml` — secrets for Streamlit Cloud (MONGODB_URI, DAGSHUB_TOKEN)


---

## Project Structure (recommended / example)

```
Pearls_AQI_Predictor/
├── .github/workflows/        # CI/CD pipelines (feature, training)
├── api/                      # FastAPI service (main.py)
├── apps/                      # Streamlit dashboard
├── aqi/                      # Data pipelines, feature engineering, training
├── reports/                  # Summary of EDA analysis
├── scripts/                  # Utilities (backfill, ingestion, train, predict)
├── .dockerignore
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Usage

Feature pipeline
```bash
python scripts/run_hourly.py
```
This will:
- Fetch latest Open‑Meteo data for configured location
- Clean & transform raw data
- Engineer features (lags, rolling stats, cyclical encodings)
- Store features in MongoDB feature store

Training pipeline
```bash
python scripts/run_daily_train.py
```
This will:
- Load features from MongoDB
- Split data (temporal split)
- Train multiple models with proper preprocessing & hyperparameter tuning
- Evaluate (RMSE, MAE, R², MAPE)
- Save the best model to the registry (MongoDB / MLflow)

API examples
- Health check:
```bash
curl GET http://localhost:8000/health
```

- Predict (example)
```bash
curl -X GET http://localhost:8000/predict?days=3 \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3, "latitude": 24.8608, "longitude": 67.0104}'
```

Typical prediction response:
```json
{
  "predictions": [
    { "date": "2026-02-19", "predicted_aqi": 96.4, "category": "Moderate" },
    ...
  ],
  "current_aqi": 101.1,
  "model_name": "lightgbm_v1",
  "model_metrics": { "rmse": 12.987 "r2": 0.554 },
  "generated_at": "2026-02-18T12:30:00Z"
}
```

---

## Machine Learning Models & Performance

Supported algorithms:
- Linear Regression
- XGBoost
- LightGBM

Typical evaluation metrics tracked:
- RMSE
- MAE
- R²
- MAPE

Example (report-style) metrics — replace with your actual results:
- Best Model: LightGBM (Mostly only, sometimes interchanged with XGBoost)
- RMSE: 12.987
- MAE: 10.188
- R²: 0.554
- MAPE: 9.35%

Model selection:
- The training pipeline selects the model with the lowest validation/test RMSE, R2.
- Models are versioned and saved to the model registry (MongoDB / MLflow).

---

## CI / CD

GitHub Actions workflows included (examples):
- `.github/workflows/hourly_ingest.yml` — scheduled hourly ingestion
- `.github/workflows/daily_train.yml` — scheduled daily training
- `.github/workflows/daily_predict.yml` — scheduled daily aqi prediction
- `.github/workflows/manual_backfill.yml` — can manually fetch backfill (Not scheduled)

Typical workflow steps:
- Checkout
- Set up Python
- Install dependencies (lightweight for feature runner; full ML stack for training)
- Run pipeline
- Upload artifacts (model artifacts, logs)
- Create issue on failure (optional)

Adjust scheduling and secrets in workflow files as needed.

---

## API Reference (summary)

Base URL: https://huggingface.co/spaces/sAbrarShah/pearls-aqi-api

Endpoints:
- GET `/health` — service health & DB connection
- GET `/predict?days=3` — request forecast for 3 days (payload: forecast_days, latitude, longitude)

---

## Troubleshooting

Common issues & fixes:

- MongoDB connection errors:
  - Ensure `MONGODB_URI` is correct
  - Check Atlas network access rules (IP whitelist)
  - Verify credentials & database name

- Slow cold starts or 502s on hosted API:
  - Platform cold-starts are expected on some providers (Railway, Render)
  - Increase instance size or use always-on plan if available

- No models available:
  - Run training pipeline manually
  - Inspect model collection in MongoDB for saved artifacts

- CI workflows fail due to disk space:
  - Use minimal dependencies where possible in feature pipelines
  - Clean temporary files in workflow steps

For more detailed troubleshooting, consult `docs/` if present.

---

## Contributing

Contributions are welcome. Typical workflow:
1. Fork repository
2. Create a feature branch (feature/your-feature)
3. Add tests / update docs
4. Open a pull request with a description of changes

---

## License

This project is part of the 10Pearls Shine Internship Program.

## Acknowledgements

- Open‑Meteo for weather & air quality data
- MongoDB Atlas for cloud storage
- Streamlit for dashboard visualization
- Huggingfacer for cloud hosting options
- MLflow / DagsHub for experiment & model tracking

---

Author
- Repository owner: sAbrarShah
- Project: Pearls_AQI_Predictor

Version: 1.0.0  
Last updated: 2026-02-18



