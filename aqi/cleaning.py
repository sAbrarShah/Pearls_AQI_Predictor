from __future__ import annotations

from typing import Any
from datetime import datetime, timezone


CORE_FIELDS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "wind_speed_10m",
]

# Caps derived from Open-Meteo US AQI threshold table (μg/m³) + sane wind cap (km/h).
CAPS = {
    "us_aqi": 500.0,          # allow >100; just block garbage
    "pm2_5": 800.0,
    "pm10": 1200.0,
    "nitrogen_dioxide": 1000.0,
    "ozone": 800.0,
    "sulphur_dioxide": 1250.0,
    "wind_speed_10m": 200.0,         # km/h default
}


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _is_valid_ts(ts: Any) -> bool:
    return isinstance(ts, datetime)


def clean_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    - Keeps only timestamp + CORE_FIELDS + minimal metadata if present.
    - Converts numeric fields to float.
    - Negative values -> None.
    - Values above caps -> None.
    - De-dupes by timestamp (last wins).
    - Drops rows with invalid timestamp or with all air fields missing.
    """
    stats = {
        "input_rows": len(rows),
        "dropped_bad_timestamp": 0,
        "dropped_all_air_missing": 0,
        "nulled_out_of_range": 0,
        "output_rows": 0,
        "deduped": 0,
    }

    by_ts: dict[datetime, dict[str, Any]] = {}

    for r in rows:
        ts = r.get("timestamp")
        if not _is_valid_ts(ts):
            stats["dropped_bad_timestamp"] += 1
            continue

        # force tz-aware UTC
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        out: dict[str, Any] = {"timestamp": ts}

        # keep minimal metadata if present
        for k in ("city", "country", "source"):
            if k in r:
                out[k] = r.get(k)

        # numeric fields
        for f in CORE_FIELDS:
            v = _to_float(r.get(f))
            if v is None:
                out[f] = None
                continue
            if v < 0:
                out[f] = None
                stats["nulled_out_of_range"] += 1
                continue
            cap = CAPS.get(f)
            if cap is not None and v > cap:
                out[f] = None
                stats["nulled_out_of_range"] += 1
                continue
            out[f] = v

        # Drop if all air fields missing (keep wind optional)
        air_fields = ["us_aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
        if all(out.get(f) is None for f in air_fields):
            stats["dropped_all_air_missing"] += 1
            continue

        by_ts[ts] = out

    stats["deduped"] = stats["input_rows"] - stats["dropped_bad_timestamp"] - stats["dropped_all_air_missing"] - len(by_ts)

    cleaned = [by_ts[k] for k in sorted(by_ts.keys())]
    stats["output_rows"] = len(cleaned)
    return cleaned, stats