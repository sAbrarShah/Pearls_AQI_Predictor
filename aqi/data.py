from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from aqi.config import MONGO_DB, MONGO_URI, MONGO_CLEAN_COLLECTION
from aqi.mongo import get_collection, ensure_timestamp_indexes


FIELDS = [
    "timestamp",
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "wind_speed_10m",
]


def load_clean_hourly(
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    col = get_collection(MONGO_URI, MONGO_DB, MONGO_CLEAN_COLLECTION)
    ensure_timestamp_indexes(col)

    q: dict[str, Any] = {}
    if start_ts or end_ts:
        q["timestamp"] = {}
        if start_ts:
            q["timestamp"]["$gte"] = start_ts
        if end_ts:
            q["timestamp"]["$lte"] = end_ts

    proj = {k: 1 for k in FIELDS}
    proj["_id"] = 0

    cursor = col.find(q, proj).sort("timestamp", 1)
    if limit:
        cursor = cursor.limit(limit)

    rows = list(cursor)
    if not rows:
        return pd.DataFrame(columns=FIELDS)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_latest_clean(hours: int = 72) -> pd.DataFrame:
    col = get_collection(MONGO_URI, MONGO_DB, MONGO_CLEAN_COLLECTION)
    ensure_timestamp_indexes(col)

    proj = {k: 1 for k in FIELDS}
    proj["_id"] = 0

    rows = list(col.find({}, proj).sort("timestamp", -1).limit(hours))
    if not rows:
        return pd.DataFrame(columns=FIELDS)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df