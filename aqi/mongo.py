from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import NetworkTimeout, AutoReconnect, ConnectionFailure


def get_collection(mongo_uri: str, db_name: str, collection_name: str) -> Collection:
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=20000,
        socketTimeoutMS=180000,   # 3 minutes
        retryWrites=True,
        maxPoolSize=5,
    )
    return client[db_name][collection_name]


def ensure_timestamp_indexes(col: Collection) -> None:
    col.create_index([("timestamp", 1)], unique=True, name="uniq_timestamp")
    col.create_index([("timestamp", -1)], name="idx_ts_desc")


def _bulk_write_retry(col: Collection, ops: list[UpdateOne], retries: int = 5, backoff: float = 1.0):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return col.bulk_write(ops, ordered=False)
        except (NetworkTimeout, AutoReconnect, ConnectionFailure) as e:
            last = e
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise last # unreachable


def upsert_by_timestamp(col: Collection, rows: list[dict[str, Any]], batch_size: int = 200) -> dict[str, int]:
    if not rows:
        return {"matched": 0, "upserted": 0, "modified": 0}

    # de-dupe input by timestamp
    by_ts = {}
    for r in rows:
        ts = r.get("timestamp")
        if ts is not None:
            by_ts[ts] = r

    items = list(by_ts.items())
    if not items:
        return {"matched": 0, "upserted": 0, "modified": 0}

    now = datetime.now(timezone.utc)
    matched = modified = upserted = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        ops = []
        for ts, r in batch:
            doc = dict(r)
            doc["ingested_at"] = now
            ops.append(UpdateOne({"timestamp": ts}, {"$set": doc}, upsert=True))

        res = _bulk_write_retry(col, ops, retries=5, backoff=1.0)
        matched += int(res.matched_count)
        modified += int(res.modified_count)
        upserted += int(len(res.upserted_ids) if res.upserted_ids else 0)

    return {"matched": matched, "modified": modified, "upserted": upserted}