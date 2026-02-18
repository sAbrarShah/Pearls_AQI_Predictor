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