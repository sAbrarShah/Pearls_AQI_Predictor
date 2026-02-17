from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aqi.config import MONGO_DB, MONGO_URI
from aqi.dagshub_mlflow import init_dagshub_mlflow
from aqi.data import load_clean_hourly, load_latest_clean
from aqi.features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLS,
    make_latest_feature_row,
    make_supervised_daily_avg,
)
from aqi.mongo import get_collection

MODEL_NAME = "aqi_linear_reg"
EXPERIMENT_NAME = "aqi_karachi"
ARTIFACTS_DIR = "models"

DAYS_AHEAD = 3
WINDOW_HOURS = 24
TEST_DAYS = 14


def _time_split(ts: pd.Series, test_days: int) -> np.ndarray:
    max_ts = pd.to_datetime(ts.max(), utc=True)
    cutoff = max_ts - timedelta(days=test_days)
    return (pd.to_datetime(ts, utc=True) > cutoff).to_numpy()


def train_and_register() -> dict[str, Any]:
    init_dagshub_mlflow(EXPERIMENT_NAME)

    df = load_clean_hourly()
    if df.empty:
        raise RuntimeError("clean_hourly is empty")

    X, y, feature_cols, ts = make_supervised_daily_avg(
        df,
        days_ahead=DAYS_AHEAD,
        window_hours=WINDOW_HOURS,
        lags=DEFAULT_LAGS,
        rolls=DEFAULT_ROLLS,
    )
    if len(X) < 200:
        raise RuntimeError(f"Not enough supervised rows: {len(X)}")

    is_test = _time_split(ts, TEST_DAYS)
    X_train, y_train = X.loc[~is_test], y[~is_test]
    X_test, y_test = X.loc[is_test], y[is_test]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(LinearRegression())),
        ]
    )
    model.feature_columns_ = feature_cols  # type: ignore[attr-defined]

    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("days_ahead", DAYS_AHEAD)
        mlflow.log_param("window_hours", WINDOW_HOURS)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("lags", ",".join(map(str, DEFAULT_LAGS)))
        mlflow.log_param("rolls", ",".join(map(str, DEFAULT_ROLLS)))
        mlflow.log_param("n_train", int(len(X_train)))
        mlflow.log_param("n_test", int(len(X_test)))

        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yhat_train = model.predict(X_train)
    
        y_true_test = y_test.reshape(-1)
        y_pred_test = yhat_test.reshape(-1)
        y_true_train = y_train.reshape(-1)
        y_pred_train = yhat_train.reshape(-1)

        RMSE = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
        MAE = float(mean_absolute_error(y_true_test, y_pred_test))

        # Keep "R²" as Day+1 R² (tomorrow), because it's the most interpretable headline
        R2_day1 = float(r2_score(y_test[:, 0], yhat_test[:, 0]))

        # MAPE (%), ignore zero/near-zero targets
        denom = np.where(np.abs(y_true_test) < 1e-6, np.nan, y_true_test)
        MAPE = float(np.nanmean(np.abs((y_true_test - y_pred_test) / denom)) * 100.0)
        if np.isnan(MAPE):
            MAPE = 0.0

        RMSE_train = float(np.sqrt(mean_squared_error(y_true_train, y_pred_train)))
        OverfitGap = float(RMSE - RMSE_train)  # positive => overfitting

        mlflow.log_metric("RMSE", RMSE)
        mlflow.log_metric("MAE", MAE)
        mlflow.log_metric("R²", R2_day1)
        mlflow.log_metric("MAPE", MAPE)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        local_path = os.path.join(ARTIFACTS_DIR, "latest_linear_reg.joblib")
        joblib.dump({"model": model, "feature_cols": feature_cols, "model_name": MODEL_NAME}, local_path)
        mlflow.log_artifact(local_path, artifact_path="local")

        input_example = X_train.iloc[:5].copy().astype("float64")
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=input_example,
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
            )

        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]

    return {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "RMSE": RMSE,
        "MAE": MAE,
        "R²": R2_day1,
        "MAPE": MAPE,
        "Overfitting Gap": OverfitGap,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def forecast_3days() -> dict[str, Any]:
    p = os.path.join(ARTIFACTS_DIR, "latest_linear_reg.joblib")
    obj = joblib.load(p)
    model = obj["model"]
    feature_cols = obj["feature_cols"]

    # Need enough recent hours to build lag/rolling features
    df_recent = load_latest_clean(hours=80)
    if df_recent.empty:
        raise RuntimeError("clean_hourly is empty")

    X1, base_ts = make_latest_feature_row(df_recent, feature_cols, lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
    yhat = np.asarray(model.predict(X1)).reshape(-1)[:DAYS_AHEAD]
    yhat = np.clip(yhat, 0.0, 500.0)

    base_date = pd.to_datetime(base_ts, utc=True).date()
    preds = []
    for d in range(1, DAYS_AHEAD + 1):
        preds.append(
            {
                "date": (base_date + pd.Timedelta(days=d)).isoformat(),
                "aqi_pred": float(yhat[d - 1]),
            }
        )

    run_at = pd.Timestamp.utcnow().to_pydatetime()
    return {
        "run_at": run_at,
        "run_date": run_at.date().isoformat(),
        "base_timestamp": base_ts.to_pydatetime(),
        "model_name": MODEL_NAME,
        "days_ahead": DAYS_AHEAD,
        "definition": "daily_avg_next_24h_window",
        "predictions": preds,
    }


def store_daily_forecast(doc: dict[str, Any], collection_name: str) -> dict[str, int]:
    col = get_collection(MONGO_URI, MONGO_DB, collection_name)
    
    try:
        col.drop_index("uniq_run_date")
    except Exception:
        pass
    
    col.create_index([("run_date", 1), ("model_name", 1)], unique=True, name="uniq_run_date_model")

    res = col.update_one(
        {"run_date": doc["run_date"], "model_name": doc["model_name"]},
        {"$set": doc},
        upsert=True,
    )
    return {
        "matched": int(res.matched_count),
        "modified": int(res.modified_count),
        "upserted": int(res.upserted_id is not None),
    }