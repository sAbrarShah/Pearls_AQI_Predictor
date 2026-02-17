from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

AIR_VARS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
]

WEATHER_VARS = ["wind_speed_10m"]


def _parse_time_utc(iso: str) -> datetime:
    # timezone=GMT => timestamp string without offset; treat as UTC
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _zip_hourly(hourly: dict[str, Any]) -> list[dict[str, Any]]:
    times = hourly.get("time", [])
    out: list[dict[str, Any]] = []
    for i, t in enumerate(times):
        row: dict[str, Any] = {"timestamp": _parse_time_utc(t)}
        for k, arr in hourly.items():
            if k == "time":
                continue
            row[k] = arr[i] if i < len(arr) else None
        out.append(row)
    return out


def _merge_on_timestamp(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[datetime, dict[str, Any]] = {}
    for r in a:
        ts = r["timestamp"]
        by_ts.setdefault(ts, {"timestamp": ts}).update({k: v for k, v in r.items() if k != "timestamp"})
    for r in b:
        ts = r["timestamp"]
        by_ts.setdefault(ts, {"timestamp": ts}).update({k: v for k, v in r.items() if k != "timestamp"})
    return [by_ts[k] for k in sorted(by_ts.keys())]


class OpenMeteoClient:
    def __init__(self, timeout_sec: int = 25, retries: int = 4, backoff_sec: float = 1.0):
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.backoff_sec = backoff_sec
        self.session = requests.Session()

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_sec)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff_sec * (2 ** (attempt - 1)))
        raise RuntimeError(f"Open-Meteo request failed after {self.retries} attempts: {last_err}")

    def fetch_recent_hourly(self, lat: float, lon: float, past_hours: int, forecast_hours: int) -> list[dict[str, Any]]:
        air_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(AIR_VARS),
            "timezone": "GMT",
            "timeformat": "iso8601",
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "domains": "auto",
        }
        wx_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "GMT",
            "timeformat": "iso8601",
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
        }

        air = _zip_hourly(self._get_json(AIR_URL, air_params).get("hourly", {}))
        wx = _zip_hourly(self._get_json(WEATHER_FORECAST_URL, wx_params).get("hourly", {}))
        return _merge_on_timestamp(air, wx)

    def fetch_range_hourly(self, lat: float, lon: float, start_date: str, end_date: str) -> list[dict[str, Any]]:
        air_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(AIR_VARS),
            "timezone": "GMT",
            "timeformat": "iso8601",
            "domains": "auto",
        }
        wx_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "GMT",
            "timeformat": "iso8601",
        }

        air = _zip_hourly(self._get_json(AIR_URL, air_params).get("hourly", {}))
        wx = _zip_hourly(self._get_json(WEATHER_ARCHIVE_URL, wx_params).get("hourly", {}))
        return _merge_on_timestamp(air, wx)