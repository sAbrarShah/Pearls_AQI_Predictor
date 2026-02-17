from __future__ import annotations

import os
from pathlib import Path

import mlflow


def init_dagshub_mlflow(experiment_name: str) -> None:
    """
    Avoid dagshub.init() to prevent httpx TLS handshake timeouts.
    Configure MLflow directly via tracking URI + basic auth env vars.
    DagsHub tracking URI format: https://dagshub.com/<owner>/<repo>.mlflow
    """
    owner = os.getenv("DAGSHUB_REPO_OWNER", "").strip()
    repo = os.getenv("DAGSHUB_REPO_NAME", "").strip()
    user = os.getenv("DAGSHUB_USERNAME", "").strip()
    token = (
        os.getenv("DAGSHUB_USER_TOKEN", "").strip()
        or os.getenv("DAGSHUB_TOKEN", "").strip()
        or os.getenv("DAGSHUB_PAT", "").strip()
    )

    if owner and repo and user and token:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_REGISTRY_URI"] = uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")  # seconds

        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
    else:
        # Local fallback: still trains + saves models/latest_model.joblib
        local = Path("mlruns").resolve()
        uri = f"file:///{local.as_posix()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)

    mlflow.set_experiment(experiment_name)