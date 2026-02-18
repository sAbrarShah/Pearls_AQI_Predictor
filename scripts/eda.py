from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.data import load_clean_hourly

# Keep in sync with aqi.data.FIELDS (minus timestamp)
FIELDS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "wind_speed_10m",
]

AQI_BANDS = [
    ("Good", 0, 50),
    ("Moderate", 51, 100),
    ("Unhealthy (SG)", 101, 150),
    ("Unhealthy", 151, 200),
    ("Very Unhealthy", 201, 300),
    ("Hazardous", 301, 500),
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _aqi_category(aqi: float | int | None) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Missing"
    a = float(aqi)
    for name, lo, hi in AQI_BANDS:
        if lo <= a <= hi:
            return name
    if a < 0:
        return "Good"
    return "Hazardous"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _coverage_and_gaps(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "min_ts": None,
            "max_ts": None,
            "rows": 0,
            "expected_hours": 0,
            "missing_hours": 0,
            "duplicate_timestamps": 0,
        }

    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    min_ts = ts.iloc[0]
    max_ts = ts.iloc[-1]

    # duplicates (should be 0 due to uniq index)
    dup = int(ts.duplicated().sum())

    # gaps
    diffs = ts.diff().dropna()
    gap_hours = diffs[diffs > pd.Timedelta(hours=1)]
    # missing hours = sum((diff_hours - 1)) across gaps
    missing = int(np.sum([(d / pd.Timedelta(hours=1)) - 1 for d in gap_hours]).round()) if not gap_hours.empty else 0

    expected = int(((max_ts - min_ts) / pd.Timedelta(hours=1)) + 1)

    return {
        "min_ts": str(min_ts),
        "max_ts": str(max_ts),
        "rows": int(len(df)),
        "expected_hours": expected,
        "missing_hours": missing,
        "duplicate_timestamps": dup,
    }


def _missingness_table(df: pd.DataFrame, title: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["field", "missing_pct", "non_missing_pct", "count"])

    out = []
    for c in ["timestamp"] + FIELDS:
        if c not in df.columns:
            continue
        miss = float(df[c].isna().mean() * 100.0)
        out.append(
            {
                "field": c,
                "missing_pct": miss,
                "non_missing_pct": 100.0 - miss,
                "count": int(df[c].notna().sum()),
            }
        )
    t = pd.DataFrame(out).sort_values(["missing_pct", "field"], ascending=[False, True]).reset_index(drop=True)
    t.attrs["title"] = title
    return t


def _daily_persistence_baseline(df: pd.DataFrame, days_eval: int = 60) -> dict[str, Any]:
    """
    Baseline on DAILY average AQI:
      pred(tomorrow) = today
    Evaluate over last `days_eval` days (or fewer if not available).
    """
    if df.empty:
        return {"n_days": 0, "rmse": None, "mae": None}

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()

    daily = d["us_aqi"].resample("D").mean().dropna()
    if len(daily) < 10:
        return {"n_days": int(len(daily)), "rmse": None, "mae": None}

    # Align y_true (day t) with y_pred (day t-1)
    y_true = daily.iloc[1:]
    y_pred = daily.shift(1).iloc[1:]

    # Evaluate last N
    if len(y_true) > days_eval:
        y_true = y_true.iloc[-days_eval:]
        y_pred = y_pred.iloc[-days_eval:]

    return {
        "n_days": int(len(y_true)),
        "rmse": _rmse(y_true.to_numpy(), y_pred.to_numpy()),
        "mae": _mae(y_true.to_numpy(), y_pred.to_numpy()),
    }


def _save_df(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    # light markdown for README/paste
    try:
        (out_dir / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        pass


def _save_fig(fig: go.Figure, out_dir: Path, name: str) -> None:
    # HTML export (no kaleido needed)
    fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")


def main() -> None:
    out_dir = ROOT / "reports" / "eda"
    _ensure_dir(out_dir)

    # Load all available clean hourly data
    df = load_clean_hourly()
    if df.empty:
        raise RuntimeError("clean_hourly is empty. Run backfill + hourly ingest first.")

    # Enforce types
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- Core summary ----------
    cov = _coverage_and_gaps(df)
    cov_df = pd.DataFrame([cov])
    _save_df(cov_df, out_dir, "coverage_summary")

    # Missingness overall + last 30 days
    miss_all = _missingness_table(df, "missingness_all")
    _save_df(miss_all, out_dir, "missingness_all")

    end_ts = pd.to_datetime(df["timestamp"].max(), utc=True)
    start_30 = end_ts - pd.Timedelta(days=30)
    df30 = df[df["timestamp"] >= start_30].copy()
    miss_30 = _missingness_table(df30, "missingness_last_30_days")
    _save_df(miss_30, out_dir, "missingness_last_30_days")

    # AQI categories
    df_cat = df[["timestamp", "us_aqi"]].copy()
    df_cat["aqi_category"] = df_cat["us_aqi"].map(_aqi_category)
    cat_counts = (
        df_cat["aqi_category"].value_counts(dropna=False).rename_axis("category").reset_index(name="count")
    )
    _save_df(cat_counts, out_dir, "aqi_category_counts")

    # ---------- Plots ----------
    # 1) AQI histogram
    fig_hist = px.histogram(
        df.dropna(subset=["us_aqi"]),
        x="us_aqi",
        nbins=60,
        title="AQI distribution (US AQI)",
    )
    fig_hist.update_layout(xaxis_title="us_aqi", yaxis_title="count")
    _save_fig(fig_hist, out_dir, "aqi_histogram")

    # 2) AQI last 14 days time series + 24h rolling mean
    start_14 = end_ts - pd.Timedelta(days=14)
    df14 = df[df["timestamp"] >= start_14].copy()
    df14["aqi_roll24"] = df14["us_aqi"].rolling(window=24, min_periods=24).mean()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df14["timestamp"], y=df14["us_aqi"], mode="lines", name="us_aqi"))
    fig_ts.add_trace(go.Scatter(x=df14["timestamp"], y=df14["aqi_roll24"], mode="lines", name="24h rolling mean"))
    fig_ts.update_layout(title="AQI time series (last 14 days)", xaxis_title="timestamp (UTC)", yaxis_title="us_aqi")
    _save_fig(fig_ts, out_dir, "aqi_timeseries_last_14_days")

    # 3) Monthly AQI boxplot
    dmonth = df.dropna(subset=["us_aqi"]).copy()
    dmonth["month"] = dmonth["timestamp"].dt.to_period("M").astype(str)
    # keep last 12 months for readability if there’s a lot
    month_order = sorted(dmonth["month"].unique())[-12:]
    dmonth = dmonth[dmonth["month"].isin(month_order)]
    fig_box = px.box(dmonth, x="month", y="us_aqi", title="AQI by month (last 12 months shown)")
    fig_box.update_layout(xaxis_title="month", yaxis_title="us_aqi")
    _save_fig(fig_box, out_dir, "aqi_box_by_month")

    # 4) Correlation heatmap (numeric only)
    corr_cols = [c for c in FIELDS if c in df.columns]
    corr_df = df[corr_cols].corr(numeric_only=True)
    corr_df_out = corr_df.reset_index().rename(columns={"index": "feature"})
    _save_df(corr_df_out, out_dir, "correlation_matrix_table")

    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        title="Correlation heatmap (features)",
    )
    _save_fig(fig_corr, out_dir, "correlation_heatmap")

    # 5) Key scatters vs AQI (no statsmodels dependency)
    for xcol in ["pm2_5", "pm10", "wind_speed_10m", "ozone", "nitrogen_dioxide", "sulphur_dioxide"]:
        if xcol not in df.columns:
            continue
        dsc = df.dropna(subset=["us_aqi", xcol]).copy()
        if len(dsc) < 50:
            continue
        fig_sc = px.scatter(
            dsc,
            x=xcol,
            y="us_aqi",
            title=f"AQI vs {xcol}",
        )
        fig_sc.update_layout(xaxis_title=xcol, yaxis_title="us_aqi")
        _save_fig(fig_sc, out_dir, f"scatter_aqi_vs_{xcol}")

    # ---------- Baseline ----------
    baseline = _daily_persistence_baseline(df, days_eval=60)
    baseline_df = pd.DataFrame([baseline])
    _save_df(baseline_df, out_dir, "daily_persistence_baseline")

    # ---------- Summary markdown ----------
    summary_lines = []
    summary_lines.append("# EDA Summary")
    summary_lines.append("")
    summary_lines.append("## Coverage")
    summary_lines.append(f"- min_ts: {cov.get('min_ts')}")
    summary_lines.append(f"- max_ts: {cov.get('max_ts')}")
    summary_lines.append(f"- rows: {cov.get('rows')}")
    summary_lines.append(f"- expected_hours: {cov.get('expected_hours')}")
    summary_lines.append(f"- missing_hours (gaps): {cov.get('missing_hours')}")
    summary_lines.append(f"- duplicate_timestamps: {cov.get('duplicate_timestamps')}")
    summary_lines.append("")
    summary_lines.append("## Missingness (overall)")
    summary_lines.append("- See missingness_all.csv / missingness_all.md")
    summary_lines.append("")
    summary_lines.append("## Missingness (last 30 days)")
    summary_lines.append("- See missingness_last_30_days.csv / missingness_last_30_days.md")
    summary_lines.append("")
    summary_lines.append("## Baseline")
    summary_lines.append(f"- Daily persistence baseline over last {baseline.get('n_days')} days:")
    summary_lines.append(f"  - RMSE: {baseline.get('rmse')}")
    summary_lines.append(f"  - MAE:  {baseline.get('mae')}")
    summary_lines.append("")
    summary_lines.append("## Plots (HTML)")
    summary_lines.append("- aqi_histogram.html")
    summary_lines.append("- aqi_timeseries_last_14_days.html")
    summary_lines.append("- aqi_box_by_month.html")
    summary_lines.append("- correlation_heatmap.html")
    summary_lines.append("- scatter_aqi_vs_*.html")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[OK] EDA written to: {out_dir}")
    print("[OK] Open reports/eda/SUMMARY.md and the .html plots.")


if __name__ == "__main__":
    main()