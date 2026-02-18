from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aqi.config import MONGO_DB, MONGO_URI
from aqi.dagshub_mlflow import init_dagshub_mlflow
from aqi.data import load_clean_hourly, load_latest_clean
from aqi.features import DEFAULT_LAGS, DEFAULT_ROLLS, make_latest_feature_row, make_supervised_daily_avg
from aqi.mongo import get_collection

from xgboost import XGBRegressor

MODEL_NAME = "aqi_xgboost"
EXPERIMENT_NAME = "aqi_karachi"
ARTIFACTS_DIR = "models"

DAYS_AHEAD = 3
WINDOW_HOURS = 24
TEST_DAYS = 14
VAL_DAYS = 7

# Tweaked to reduce overfit + try to push R² up (not guaranteed)
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 12000,
    "learning_rate": 0.02,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_lambda": 12.0,
    "reg_alpha": 1.0,
    "gamma": 0.05,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 300,
}


class MultiHorizonModel:
    def __init__(self, models: list[Any]) -> None:
        self.models = models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.column_stack(preds)


def _time_split(ts: pd.Series, days: int) -> np.ndarray:
    max_ts = pd.to_datetime(ts.max(), utc=True)
    cutoff = max_ts - timedelta(days=days)
    return (pd.to_datetime(ts, utc=True) > cutoff).to_numpy()


def train_and_register() -> dict[str, Any]:
    # hard reset any dangling run
    if mlflow.active_run() is not None:
        mlflow.end_run()

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
    X_tr_full, y_tr_full, ts_tr_full = X.loc[~is_test], y[~is_test], ts.loc[~is_test]
    X_test, y_test = X.loc[is_test], y[is_test]

    is_val = _time_split(ts_tr_full, VAL_DAYS)
    X_train, y_train = X_tr_full.loc[~is_val], y_tr_full[~is_val]
    X_val, y_val = X_tr_full.loc[is_val], y_tr_full[is_val]

    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("days_ahead", DAYS_AHEAD)
        mlflow.log_param("window_hours", WINDOW_HOURS)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("val_days", VAL_DAYS)
        mlflow.log_param("lags", ",".join(map(str, DEFAULT_LAGS)))
        mlflow.log_param("rolls", ",".join(map(str, DEFAULT_ROLLS)))
        mlflow.log_param("n_train", int(len(X_train)))
        mlflow.log_param("n_val", int(len(X_val)))
        mlflow.log_param("n_test", int(len(X_test)))

        # IMPORTANT: params logged INSIDE the run
        for k, v in XGB_PARAMS.items():
            mlflow.log_param(f"xgb_{k}", v)

        models: list[Any] = []
        for h in range(DAYS_AHEAD):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(
                X_train,
                y_train[:, h],
                eval_set=[(X_val, y_val[:, h])],
                verbose=False,
            )
            models.append(m)
            bi = int(getattr(m, "best_iteration", 0) or 0)
            mlflow.log_param(f"best_iteration_day{h+1}", bi)

        model = MultiHorizonModel(models)
        model.feature_columns_ = feature_cols

        yhat_test = model.predict(X_test)

        y_true_test = y_test.reshape(-1)
        y_pred_test = np.asarray(yhat_test, dtype=float).reshape(-1)

        RMSE = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
        MAE = float(mean_absolute_error(y_true_test, y_pred_test))
        R2_day1 = float(r2_score(y_test[:, 0], yhat_test[:, 0]))

        denom = np.where(np.abs(y_true_test) < 1e-6, np.nan, y_true_test)
        MAPE = float(np.nanmean(np.abs((y_true_test - y_pred_test) / denom)) * 100.0)
        if np.isnan(MAPE):
            MAPE = 0.0

        mlflow.log_metric("RMSE", RMSE)
        mlflow.log_metric("MAE", MAE)
        mlflow.log_metric("R²", R2_day1)
        mlflow.log_metric("MAPE", MAPE)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        local_path = os.path.join(ARTIFACTS_DIR, "latest_xgboost.joblib")
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

        run_id = mlflow.active_run().info.run_id

    return {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "RMSE": RMSE,
        "MAE": MAE,
        "R²": R2_day1,
        "MAPE": MAPE,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }


def forecast_3days() -> dict[str, Any]:
    p = os.path.join(ARTIFACTS_DIR, "latest_xgboost.joblib")
    if not os.path.exists(p):
        raise RuntimeError(f"Missing XGBoost artifact: {p}")

    obj = joblib.load(p)
    model = obj["model"]
    feature_cols = obj["feature_cols"]

    df_recent = load_latest_clean(hours=80)
    if df_recent.empty:
        raise RuntimeError("clean_hourly is empty")

    X1, base_ts = make_latest_feature_row(df_recent, feature_cols, lags=DEFAULT_LAGS, rolls=DEFAULT_ROLLS)
    yhat = np.asarray(model.predict(X1), dtype=float).reshape(-1)[:DAYS_AHEAD]
    yhat = np.clip(yhat, 0.0, 500.0)

    base_date = pd.to_datetime(base_ts, utc=True).date()
    preds = [{"date": (base_date + pd.Timedelta(days=d)).isoformat(), "aqi_pred": float(yhat[d - 1])} for d in range(1, DAYS_AHEAD + 1)]

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
    return {"matched": int(res.matched_count), "modified": int(res.modified_count), "upserted": int(res.upserted_id is not None)}