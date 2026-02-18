from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _hydrate_env_from_streamlit_secrets() -> None:
    # Safe: do nothing locally if no secrets.toml exists
    try:
        if hasattr(st.secrets, "to_dict"):
            s = st.secrets.to_dict()
        else:
            s = {k: st.secrets[k] for k in st.secrets}
    except Exception:
        return

    for k in ["MONGO_URI", "MONGO_DB", "MONGO_FORECAST_COLLECTION", "MONGO_MODEL_RUNS_COLLECTION"]:
        if not os.getenv(k) and k in s:
            os.environ[k] = str(s[k])


_hydrate_env_from_streamlit_secrets()

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB = os.getenv("MONGO_DB", "aqi_database").strip() or "aqi_database"
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI. Set it in .env (local) or Streamlit/GitHub secrets (deploy).")


FORECAST_COLLECTION = os.getenv("MONGO_FORECAST_COLLECTION", "forecasts_daily").strip() or "forecasts_daily"
MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs_daily").strip() or "model_runs_daily"

AQI_BANDS = [
    ("Good", 0, 50, "#00E400"),
    ("Moderate", 51, 100, "#FFFF00"),
    ("Unhealthy (SG)", 101, 150, "#FF7E00"),
    ("Unhealthy", 151, 200, "#FF0000"),
    ("Very Unhealthy", 201, 300, "#8F3F97"),
    ("Hazardous", 301, 500, "#7E0023"),
]


def _pretty_model_name(model_name: str) -> str:
    m = (model_name or "").lower()
    if "lightgbm" in m:
        return "LightGBM"
    if "xgboost" in m:
        return "XGBoost"
    if "linear" in m:
        return "Linear Regression"
    return model_name or "Model"


def aqi_category(aqi: float) -> str:
    a = float(aqi)
    for name, lo, hi, _ in AQI_BANDS:
        if lo <= a <= hi:
            return name
    if a < 0:
        return "Good"
    return "Hazardous"


def aqi_color(aqi: float) -> str:
    a = float(aqi)
    for _, lo, hi, color in AQI_BANDS:
        if lo <= a <= hi:
            return color
    if a < 0:
        return AQI_BANDS[0][3]
    return AQI_BANDS[-1][3]


@st.cache_resource
def _mongo_client(uri: str) -> MongoClient:
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=60000,
        retryWrites=True,
        maxPoolSize=5,
    )


def _col(db: str, name: str):
    return _mongo_client(MONGO_URI)[db][name]


@st.cache_data(ttl=15)
def load_latest_training_doc(db: str, colname: str) -> dict[str, Any]:
    c = _col(db, colname)
    doc = list(c.find({}, {"_id": 0}).sort("run_date", -1).limit(1))
    if not doc:
        raise RuntimeError(f"No training docs found in {colname}. Run scripts/run_daily_train.py first.")
    return doc[0]


@st.cache_data(ttl=15)
def load_latest_best_forecast(db: str, colname: str) -> dict[str, Any]:
    c = _col(db, colname)

    doc = list(c.find({"best": True}, {"_id": 0}).sort("run_date", -1).limit(1))
    if doc:
        return doc[0]

    doc = list(c.find({}, {"_id": 0}).sort("run_date", -1).limit(1))
    if doc:
        return doc[0]

    raise RuntimeError(f"No forecast docs found in {colname}. Run scripts/run_daily_predict.py first.")


def gauge_figure(value: float, title: str) -> go.Figure:
    steps = [{"range": [lo, hi], "color": color} for _, lo, hi, color in AQI_BANDS]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": " AQI", "font": {"size": 26}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {
                    "range": [0, 500],
                    "tickmode": "array",
                    "tickvals": [0, 50, 100, 150, 200, 300, 500],
                    "tickfont": {"size": 11},
                    "tickwidth": 0,
                    "ticklen": 0,
                },
                "bar": {"color": "rgba(255,255,255,0.88)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": steps,
                "threshold": {"line": {"color": "rgba(255,255,255,0.85)", "width": 4}, "value": float(value)},
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=34, r=18, t=44, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _css() -> None:
    st.markdown(
        """
        <style>
          .block-container{
            padding-top: 0.9rem;
            padding-bottom: 1.1rem;
            padding-left: 1.0rem;
            padding-right: 1.0rem;
            max-width: 1800px;
          }
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header { visibility: hidden; }

          .kicker{opacity:.78; font-size:14px; margin-top:-6px; margin-bottom:10px;}
          .pill-wrap{width:100%; text-align:center; margin-top:8px;}
          .pill{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            padding:6px 12px;
            border-radius:999px;
            border:2px solid rgba(255,255,255,0.22);
            background:rgba(255,255,255,0.06);
            font-weight:800;
            font-size:0.92rem;
            line-height:1;
            min-width: 130px;
          }

          .legend-row{display:flex; align-items:center; gap:10px; margin:10px 0;}
          .swatch{width:14px; height:14px; border-radius:4px; border:1px solid rgba(255,255,255,0.18);}
          .legend-name{font-weight:700;}
          .legend-range{opacity:.75;}
          .reason{padding:10px 12px; border-radius:14px; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.05);}

          .metrics-table{
            width: 100% !important;
            table-layout: fixed;
            border-collapse: collapse;
            font-size: 14px;
          }
          .metrics-table th, .metrics-table td{
            text-align: center !important;
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.10);
            white-space: nowrap;
          }
          .metrics-table th{
            font-weight: 800;
            opacity: 0.90;
          }
          .metrics-table tr:last-child td{
            border-bottom: none;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def range_legend() -> None:
    st.markdown("### AQI ranges")
    for name, lo, hi, color in AQI_BANDS:
        st.markdown(
            f"""
            <div class="legend-row">
              <div class="swatch" style="background:{color};"></div>
              <div class="legend-name">{name}</div>
              <div class="legend-range">({lo}–{hi})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _score_key(m: dict[str, Any]) -> tuple[float, float, float]:
    rmse = float(m.get("RMSE", 1e18))
    mae = float(m.get("MAE", 1e18))
    r2 = float(m.get("R²", -1e18))
    return (rmse, mae, -r2)


def best_reason(best: dict[str, Any], all_models: list[dict[str, Any]]) -> str:
    best_rmse = float(best.get("RMSE", 0.0))
    best_mae = float(best.get("MAE", 0.0))
    best_r2 = float(best.get("R²", 0.0))

    others = [m for m in all_models if str(m.get("model_name")) != str(best.get("model_name"))]
    if not others:
        return "Chosen because it has the best overall validation metrics."

    runner_up = sorted(others, key=_score_key)[0]
    ru_rmse = float(runner_up.get("RMSE", 0.0))
    ru_mae = float(runner_up.get("MAE", 0.0))
    ru_r2 = float(runner_up.get("R²", 0.0))

    d_rmse = ru_rmse - best_rmse
    d_mae = ru_mae - best_mae
    d_r2 = best_r2 - ru_r2

    parts = []
    if d_rmse > 0:
        parts.append(f"lower RMSE by {d_rmse:.3f}")
    if d_mae > 0:
        parts.append(f"lower MAE by {d_mae:.3f}")
    if d_r2 > 0:
        parts.append(f"higher R² by {d_r2:.3f}")

    if not parts:
        return "Chosen because it wins on the combined ranking (RMSE, MAE, R²)."

    return "Chosen because it has " + ", ".join(parts) + " vs the next best model."


# ---------- UI ----------
st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
_css()

train_doc = load_latest_training_doc(MONGO_DB, MODEL_RUNS_COLLECTION)
forecast_doc = load_latest_best_forecast(MONGO_DB, FORECAST_COLLECTION)

models = train_doc.get("models", [])
best = train_doc.get("best", {})

best_model_name = str(best.get("model_name", forecast_doc.get("model_name", "unknown")))
pretty_best = _pretty_model_name(best_model_name)

st.title("Pearls AQI Predictor — Karachi")
st.markdown(f"<div class='kicker'>Using {pretty_best} to predict next 3-day forecast</div>", unsafe_allow_html=True)

preds = forecast_doc.get("predictions", [])
if not isinstance(preds, list) or len(preds) < 3:
    st.error("Best forecast must contain predictions[3].")
    st.stop()

g1, g2, g3, leg = st.columns([1, 1, 1, 0.95], gap="large", vertical_alignment="top")
gcols = [g1, g2, g3]

for i in range(3):
    p = preds[i]
    date_str = str(p.get("date", ""))
    aqi_val = float(p.get("aqi_pred", 0.0))

    cat = aqi_category(aqi_val)
    colr = aqi_color(aqi_val)

    with gcols[i]:
        st.plotly_chart(gauge_figure(aqi_val, date_str), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div class='pill-wrap'><span class='pill' style='border-color:{colr};'>{cat}</span></div>",
            unsafe_allow_html=True,
        )

with leg:
    range_legend()

st.markdown("---")

# Metrics table + reason
st.subheader("Model comparison (latest training run)")

valid_models = [m for m in (models if isinstance(models, list) else []) if isinstance(m, dict) and m.get("model_name")]

rows = []
for m in valid_models:
    rows.append(
        {
            "Model": _pretty_model_name(str(m.get("model_name", ""))),
            "RMSE": float(m.get("RMSE", float("nan"))),
            "MAE": float(m.get("MAE", float("nan"))),
            "R²": float(m.get("R²", float("nan"))),
            "MAPE%": float(m.get("MAPE", float("nan"))),
            "Best": "Yes" if str(m.get("model_name")) == best_model_name else "",
        }
    )

df = pd.DataFrame(rows)

left, right = st.columns([4, 2], gap="large", vertical_alignment="top")

with left:
    if df.empty:
        st.info("No metrics found in model_runs_daily document.")
    else:
        display_df = df.sort_values(["Best", "RMSE"], ascending=[False, True]).reset_index(drop=True)

        fmt = display_df.copy()
        for c in ["RMSE", "MAE", "R²"]:
            fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        fmt["MAPE%"] = fmt["MAPE%"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

        st.markdown(fmt.to_html(index=False, escape=False, classes="metrics-table"), unsafe_allow_html=True)

with right:
    if isinstance(best, dict) and best:
        why = best_reason(best, valid_models)
        st.markdown(f"<div class='reason'><b>Why this model:</b><br>{why}</div>", unsafe_allow_html=True)