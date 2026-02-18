from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _topk_from_scores(scores: np.ndarray, feature_names: list[str], top_k: int) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": feature_names[i], "mean_abs_shap": float(scores[i])} for i in idx]


def compute_shap_top_features_tree(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    import shap  # heavy import; used only in training

    Xs = X.astype("float64")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv, dtype=float)
    scores = np.mean(np.abs(sv), axis=0)
    return _topk_from_scores(scores, feature_names, top_k)


def compute_top_features_linear_fallback(
    coef: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    coef = np.asarray(coef, dtype=float).reshape(-1)
    scores = np.abs(coef)
    return _topk_from_scores(scores, feature_names, top_k)


def log_top_features_mlflow(top_features: list[dict[str, Any]], artifact_name: str = "top_features_day1.json") -> str:
    import mlflow

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / artifact_name
        p.write_text(json.dumps(top_features, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(p), artifact_path="explainability")
    return f"explainability/{artifact_name}"