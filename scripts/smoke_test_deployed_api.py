from __future__ import annotations

import json
import sys
from typing import Any

import requests


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        _fail('Usage: python scripts/smoke_test_deployed_api.py "<BASE_URL>"\nExample: python scripts/smoke_test_deployed_api.py "https://your-api.onrender.com"')

    base_url = sys.argv[1].rstrip("/")
    health_url = f"{base_url}/health"
    predict_url = f"{base_url}/predict"

    print("=" * 70)
    print("DEPLOYED API SMOKE TEST: /health + /predict")
    print("=" * 70)
    print(f"API: {base_url}")

    # health
    try:
        r = requests.get(health_url, timeout=30)
    except Exception as e:
        _fail(f"/health request error: {e}")

    if r.status_code != 200:
        _fail(f"/health HTTP {r.status_code} | body={r.text[:400]}")

    try:
        health = r.json()
    except Exception:
        _fail(f"/health returned non-JSON: {r.text[:400]}")

    print(f"[OK] /health: {health}")

    # predict
    payload: dict[str, Any] = {"city": "Karachi", "days": 3, "model": "best"}  # keep consistent with your API
    headers = {"Content-Type": "application/json"}

    try:
        r2 = requests.post(predict_url, headers=headers, data=json.dumps(payload), timeout=60)
    except Exception as e:
        _fail(f"/predict request error: {e}")

    if r2.status_code != 200:
        _fail(f"/predict HTTP {r2.status_code} | body={r2.text[:800]}")

    try:
        out = r2.json()
    except Exception:
        _fail(f"/predict returned non-JSON: {r2.text[:800]}")

    # validate shape
    preds = out.get("predictions")
    if not isinstance(preds, list) or len(preds) < 1:
        _fail(f"/predict missing predictions list | keys={list(out.keys())}")

    print(f"[OK] /predict keys={list(out.keys())}")
    print("[OK] first prediction:", preds[0])

    print("=" * 70)
    print("PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()