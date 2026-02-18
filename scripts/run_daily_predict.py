from __future__ import annotations

import json
import logging
import sys
import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.dagshub_mlflow import init_dagshub_mlflow
from aqi.config import MONGO_DB, MONGO_URI
from aqi.mongo import get_collection
from aqi.data import load_latest_clean
from aqi.features import make_latest_feature_row, DEFAULT_LAGS, DEFAULT_ROLLS

from aqi.models.linear_reg import forecast_3days as forecast_linear
from aqi.models.linear_reg import store_daily_forecast as store_linear

from aqi.models.xgboost import forecast_3days as forecast_xgb
from aqi.models.xgboost import store_daily_forecast as store_xgb

from aqi.models.lightgbm import forecast_3days as forecast_lgbm
from aqi.models.lightgbm import store_daily_forecast as store_lgbm


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_daily_predict")

MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs_daily")
FORECAST_COLLECTION = os.getenv("MONGO_FORECAST_COLLECTION", "forecasts_daily")
EXPERIMENT_NAME = "aqi_karachi"
DAYS_AHEAD = 3


def _alerts_from_predictions(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(preds, list):
        return out

    for p in preds:
        try:
            a = float(p.get("aqi_pred", 0.0))
        except Exception:
            continue

        if a >= 301:
            out.append({"date": p.get("date"), "aqi_pred": a, "level": "Hazardous", "threshold": 301})
        elif a >= 201:
            out.append({"date": p.get("date"), "aqi_pred": a, "level": "Very Unhealthy", "threshold": 201})
        elif a >= 151:
            out.append({"date": p.get("date"), "aqi_pred": a, "level": "Unhealthy", "threshold": 151})

    out.sort(key=lambda x: float(x.get("aqi_pred", 0.0)), reverse=True)
    return out


def _load_latest_training_doc() -> dict[str, Any]:
    col = get_collection(MONGO_URI, MONGO_DB, MODEL_RUNS_COLLECTION)
    doc = col.find({}, {"_id": 0}).sort("run_date", -1).limit(1)
    items = list(doc)
    if not items:
        raise RuntimeError("No training metadata found in Mongo (model_runs_daily is empty).")
    return items[0]


def _joblib_paths_for(model_name: str) -> list[str]:
    if model_name == "aqi_linear_reg":
        # support both new + old filename
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
            continue
    raise RuntimeError(f"Failed to download joblib bundle for {model_name} run_id={run_id}: {last_err}")


def _forecast_from_run(run_id: str, model_name: str) -> dict[str, Any]:
    bundle = _load_bundle_from_run(run_id, model_name)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df_recent = load_latest_clean(hours=80)
    if df_recent.empty:
        raise RuntimeError("clean_hourly is empty")

    X1, base_ts = make_latest_feature_row(df_recent, feature_cols, lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
    yhat = np.asarray(model.predict(X1), dtype=float).reshape(-1)[:DAYS_AHEAD]
    yhat = np.clip(yhat, 0.0, 500.0)

    base_date = pd.to_datetime(base_ts, utc=True).date()
    preds = [
        {"date": (base_date + pd.Timedelta(days=d)).isoformat(), "aqi_pred": float(yhat[d - 1])}
        for d in range(1, DAYS_AHEAD + 1)
    ]

    run_at = datetime.now(timezone.utc)
    return {
        "run_at": run_at,
        "run_date": run_at.date().isoformat(),
        "base_timestamp": base_ts.to_pydatetime(),
        "model_name": model_name,
        "days_ahead": DAYS_AHEAD,
        "definition": "daily_avg_next_24h_window",
        "predictions": preds,
        "mlflow_run_id": run_id,
        "predictions": preds,
        "alerts": _alerts_from_predictions(preds),
        "mlflow_run_id": run_id,
    }


def _print_forecast(doc: dict[str, Any]) -> None:
    model = doc.get("model_name", "unknown")
    base_ts = doc.get("base_timestamp", None)
    run_date = doc.get("run_date", None)
    best = bool(doc.get("best", False))

    log.info("PREDICT | model=%s best=%s run_date=%s base_timestamp=%s", model, best, run_date, base_ts)

    preds = doc.get("predictions", [])
    for i, p in enumerate(preds, start=1):
        d = p.get("date")
        aqi = p.get("aqi_pred")
        log.info("  Day+%d | date=%s | aqi_pred=%.3f", i, d, float(aqi))

    alerts = doc.get("alerts", [])
    if isinstance(alerts, list) and alerts:
        for a in alerts:
            log.info(
                "  ALERT | date=%s level=%s aqi_pred=%.3f threshold=%s",
                a.get("date"),
                a.get("level"),
                float(a.get("aqi_pred", 0.0)),
                a.get("threshold"),
            )


def main() -> None:
    init_dagshub_mlflow(EXPERIMENT_NAME)

    td = _load_latest_training_doc()
    best_name = str(td.get("best", {}).get("model_name", ""))

    models: list[tuple[str, Any]] = [
        ("aqi_linear_reg", store_linear),
        ("aqi_xgboost", store_xgb),
        ("aqi_lightgbm", store_lgbm),
    ]

    runs = {m.get("model_name"): m.get("run_id") for m in td.get("models", [])}

    for name, store_fn in models:
        run_id = runs.get(name)
        if not run_id:
            raise RuntimeError(f"Missing run_id for model {name} in model_runs_daily")

        doc = _forecast_from_run(str(run_id), name)
        doc["best"] = (name == best_name)

        _print_forecast(doc)

        res = store_fn(doc, FORECAST_COLLECTION)
        log.info(
            "FORECAST stored | model=%s best=%s run_date=%s upserted=%d",
            doc.get("model_name"),
            doc.get("best", False),
            doc.get("run_date"),
            int(res.get("upserted", 0)),
        )


if __name__ == "__main__":
    main()