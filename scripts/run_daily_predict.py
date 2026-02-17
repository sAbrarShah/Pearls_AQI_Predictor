from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.models.linear_reg import forecast_3days as forecast_linear
from aqi.models.linear_reg import store_daily_forecast as store_linear

from aqi.models.xgboost import forecast_3days as forecast_xgb
from aqi.models.xgboost import store_daily_forecast as store_xgb

from aqi.models.lightgbm import forecast_3days as forecast_lgbm
from aqi.models.lightgbm import store_daily_forecast as store_lgbm


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_daily_predict")

BEST_PATH = ROOT / "models" / "best_model.json"
FORECAST_COLLECTION = "forecasts_daily"


def _load_best_model_name() -> str | None:
    if not BEST_PATH.exists():
        return None
    try:
        obj = json.loads(BEST_PATH.read_text(encoding="utf-8"))
        name = obj.get("model_name")
        return str(name) if name else None
    except Exception:
        return None


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


def main() -> None:
    best_name = _load_best_model_name()

    models: list[tuple[str, Any, Any]] = [
        ("aqi_linear_reg", forecast_linear, store_linear),
        ("aqi_xgboost", forecast_xgb, store_xgb),
        ("aqi_lightgbm", forecast_lgbm, store_lgbm),
    ]

    # fallback if best_model.json missing/corrupt
    if best_name is None:
        best_name = "aqi_lightgbm"

    for name, forecast_fn, store_fn in models:
        doc = forecast_fn()  # must contain predictions[3]
        doc["best"] = (doc.get("model_name") == best_name)

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