from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VARS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "wind_speed_10m",
]

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24]
DEFAULT_ROLLS = [3, 6, 12, 24]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)

    out = df.copy()
    out["hour"] = ts.dt.hour.astype("float64")
    out["dow"] = ts.dt.dayofweek.astype("float64")
    out["month"] = ts.dt.month.astype("float64")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    lags: list[int] = DEFAULT_LAGS,
    rolls: list[int] = DEFAULT_ROLLS,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d = d.sort_values("timestamp").reset_index(drop=True)

    for c in FEATURE_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = _add_time_features(d)

    for v in FEATURE_VARS:
        for lag in lags:
            d[f"{v}_lag{lag}"] = d[v].shift(lag)
        for w in rolls:
            d[f"{v}_roll{w}"] = d[v].rolling(window=w, min_periods=w).mean()

    d["aqi_delta_1h"] = d["us_aqi"] - d["us_aqi"].shift(1)
    d["aqi_delta_3h"] = d["us_aqi"] - d["us_aqi"].shift(3)
    d["aqi_delta_24h"] = d["us_aqi"] - d["us_aqi"].shift(24)

    return d


def make_supervised_daily_avg(
    df: pd.DataFrame,
    days_ahead: int = 3,
    window_hours: int = 24,
    lags: list[int] = DEFAULT_LAGS,
    rolls: list[int] = DEFAULT_ROLLS,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.Series]:
    """
    Predict 1 value per day (daily average AQI).
    y_day1 = mean AQI over next 24h
    y_day2 = mean AQI over hours 25-48
    y_day3 = mean AQI over hours 49-72
    """
    f = build_feature_frame(df, lags=lags, rolls=rolls)
    if f.empty:
        return pd.DataFrame(), np.empty((0, days_ahead)), [], pd.Series(dtype="datetime64[ns, UTC]")

    s = pd.to_numeric(f["us_aqi"], errors="coerce")

    y_cols = []
    for d in range(1, days_ahead + 1):
        end_shift = -(d * window_hours)
        y_d = s.shift(end_shift).rolling(window_hours, min_periods=window_hours).mean()
        y_cols.append(y_d.rename(f"y_day{d}"))

    y_df = pd.concat(y_cols, axis=1)

    feature_cols = [c for c in f.columns if c != "timestamp"]
    X = f[feature_cols]
    ts = f["timestamp"]

    mask = X.notna().all(axis=1) & y_df.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y_df.loc[mask].to_numpy(dtype=float)
    ts = ts.loc[mask].reset_index(drop=True)

    return X, y, feature_cols, ts


def make_latest_feature_row(
    df_recent: pd.DataFrame,
    feature_cols: list[str],
    lags: list[int] = DEFAULT_LAGS,
    rolls: list[int] = DEFAULT_ROLLS,
    max_lookback_rows: int = 48,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    f = build_feature_frame(df_recent, lags=lags, rolls=rolls)
    if f.empty:
        raise RuntimeError("No data to build features")

    for c in feature_cols:
        if c not in f.columns:
            f[c] = np.nan

    tail = f.tail(max_lookback_rows).copy()
    X = tail[feature_cols]

    valid = ~X.isna().any(axis=1)
    if not bool(valid.any()):
        raise RuntimeError(
            "No complete feature row available in recent window "
            "(missing values after lags/rolls). Increase history or relax features."
        )

    idx = valid[valid].index[-1]
    row = f.loc[[idx]].copy()
    base_ts = pd.to_datetime(row["timestamp"].iloc[0], utc=True)

    return row[feature_cols], base_ts