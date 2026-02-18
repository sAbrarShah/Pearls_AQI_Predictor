from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.config import MONGO_DB, MONGO_URI
from aqi.dagshub_mlflow import init_dagshub_mlflow
from aqi.data import load_latest_clean
from aqi.features import DEFAULT_LAGS, DEFAULT_ROLLS, make_latest_feature_row
from aqi.mongo import get_collection

EXPERIMENT_NAME = "aqi_karachi"
MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs_daily")

app = FastAPI(title="Pearls AQI Predictor API", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    init_dagshub_mlflow(EXPERIMENT_NAME)


def _load_latest_training_doc() -> dict[str, Any]:
    col = get_collection(MONGO_URI, MONGO_DB, MODEL_RUNS_COLLECTION)
    items = list(col.find({}, {"_id": 0}).sort("run_date", -1).limit(1))
    if not items:
        raise RuntimeError("model_runs_daily is empty")
    return items[0]


def _joblib_paths_for(model_name: str) -> list[str]:
    if model_name == "aqi_linear_reg":
        return ["local/latest_linear_reg.joblib", "local/latest_model.joblib"]
    if model_name == "aqi_xgboost":
        return ["local/latest_xgboost.joblib"]
    if model_name == "aqi_lightgbm":
        return ["local/latest_lightgbm.joblib"]
    raise RuntimeError(f"Unknown model_name: {model_name}")


def _load_bundle_from_run(run_id: str, model_name: str) -> dict[str, Any]:
    last_err = None
    for ap in _joblib_paths_for(model_name):
        try:
            p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ap)
            return joblib.load(p)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download joblib bundle for {model_name} run_id={run_id}: {last_err}")


def _predict_best(days: int = 3) -> dict[str, Any]:
    td = _load_latest_training_doc()
    best = td.get("best", {}) or {}
    model_name = str(best.get("model_name", "")).strip()
    run_id = str(best.get("run_id", "")).strip()
    if not model_name or not run_id:
        raise RuntimeError("Latest training doc missing best.model_name or best.run_id")

    bundle = _load_bundle_from_run(run_id, model_name)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df_recent = load_latest_clean(hours=80)
    if df_recent.empty:
        raise RuntimeError("clean_hourly is empty")

    X1, base_ts = make_latest_feature_row(df_recent, feature_cols, lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
    yhat = np.asarray(model.predict(X1), dtype=float).reshape(-1)[:days]
    yhat = np.clip(yhat, 0.0, 500.0)

    base_date = pd.to_datetime(base_ts, utc=True).date()
    preds = [{"date": (base_date + pd.Timedelta(days=d)).isoformat(), "aqi_pred": float(yhat[d - 1])} for d in range(1, days + 1)]

    now = datetime.now(timezone.utc)
    return {
        "status": "ok",
        "run_at": now.isoformat(),
        "run_date": now.date().isoformat(),
        "base_timestamp": pd.to_datetime(base_ts, utc=True).to_pydatetime().isoformat(),
        "model_name": model_name,
        "mlflow_run_id": run_id,
        "days_ahead": days,
        "definition": "daily_avg_next_24h_window",
        "predictions": preds,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    td = _load_latest_training_doc()
    best = td.get("best", {}) or {}
    return {"status": "ok", "latest_run_date": td.get("run_date"), "best_model": best.get("model_name")}


@app.get("/predict")
def predict(days: int = Query(default=3, ge=1, le=3)) -> dict[str, Any]:
    return _predict_best(days=days)