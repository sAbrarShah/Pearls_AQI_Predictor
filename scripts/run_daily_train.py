from __future__ import annotations

import json
import logging
import sys
import mlflow
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.models.linear_reg import train_and_register as train_linear
from aqi.models.xgboost import train_and_register as train_xgb
from aqi.models.lightgbm import train_and_register as train_lgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_daily_train")

BEST_PATH = ROOT / "models" / "best_model.json"

def _end_run_if_any() -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run()

def _score_key(out: dict[str, Any]) -> tuple[float, float, float]:
    # Lower RMSE better, lower MAE better, higher R² better
    rmse = float(out.get("RMSE", 1e18))
    mae = float(out.get("MAE", 1e18))
    r2 = float(out.get("R²", -1e18))
    return (rmse, mae, -r2)

def _jsonable(x: Any) -> Any:
    # make numpy/pandas scalars JSON-safe + keep dict/list structures
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        try:
            return x.item()
        except Exception:
            pass
    return x


def main() -> None:
    results: list[dict[str, Any]] = []

    _end_run_if_any()


    for fn in (train_linear, train_xgb, train_lgbm):
        _end_run_if_any()
        
        try:
            out = fn()
            results.append(out)
            log.info(
                "TRAIN done | model=%s run_id=%s RMSE=%.3f MAE=%.3f R²=%.3f MAPE=%.3f",
                out["model_name"],
                out["run_id"],
                out["RMSE"],
                out["MAE"],
                out["R²"],
                out["MAPE"],
            )
        except Exception as e:
            log.exception("TRAIN failed | err=%s", e)
        finally:
            mlflow.end_run()
            
    if not results:
        raise RuntimeError("All model trainings failed; no best model can be selected.")

    best = sorted(results, key=_score_key)[0]
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PATH.write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")

    log.info(
        "BEST model=%s | RMSE=%.3f MAE=%.3f R²=%.3f",
        best["model_name"],
        best["RMSE"],
        best["MAE"],
        best["R²"],
    )


if __name__ == "__main__":
    main()