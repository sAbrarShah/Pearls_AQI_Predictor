from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.config import (
    CITY, COUNTRY, LAT, LON,
    HTTP_BACKOFF_SEC, HTTP_RETRIES, HTTP_TIMEOUT_SEC,
    MONGO_DB, MONGO_RAW_COLLECTION, MONGO_CLEAN_COLLECTION, MONGO_URI,
    SOURCE_NAME,
)
from aqi.mongo import ensure_timestamp_indexes, get_collection, upsert_by_timestamp
from aqi.openmeteo import OpenMeteoClient
from aqi.cleaning import clean_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("backfill")


def main() -> None:
    backfill_days = int(os.getenv("BACKFILL_DAYS", "180"))
    chunk_days = int(os.getenv("BACKFILL_CHUNK_DAYS", "14"))
    sleep_sec = float(os.getenv("BACKFILL_SLEEP_SEC", "0.25"))

    end_day = (datetime.now(timezone.utc).date() - timedelta(days=1))
    start_day = end_day - timedelta(days=backfill_days - 1)

    log.info("Backfill Karachi | %s -> %s (%s days)", start_day, end_day, backfill_days)

    client = OpenMeteoClient(
        timeout_sec=HTTP_TIMEOUT_SEC,
        retries=HTTP_RETRIES,
        backoff_sec=HTTP_BACKOFF_SEC,
    )

    raw_col = get_collection(MONGO_URI, MONGO_DB, MONGO_RAW_COLLECTION)
    clean_col = get_collection(MONGO_URI, MONGO_DB, MONGO_CLEAN_COLLECTION)
    ensure_timestamp_indexes(raw_col)
    ensure_timestamp_indexes(clean_col)

    chunk_start = start_day
    total_raw_upserted = 0
    total_clean_upserted = 0

    while chunk_start <= end_day:
        chunk_end = min(end_day, chunk_start + timedelta(days=chunk_days - 1))
        start_date = chunk_start.isoformat()
        end_date = chunk_end.isoformat()

        log.info("Chunk: %s -> %s", start_date, end_date)
        rows = client.fetch_range_hourly(LAT, LON, start_date, end_date)

        for r in rows:
            r["city"] = CITY
            r["country"] = COUNTRY
            r["source"] = SOURCE_NAME

        raw_res = upsert_by_timestamp(raw_col, rows)
        total_raw_upserted += raw_res["upserted"]

        cleaned, stats = clean_rows(rows)
        clean_res = upsert_by_timestamp(clean_col, cleaned)
        total_clean_upserted += clean_res["upserted"]

        log.info("RAW upsert | rows=%s matched=%s modified=%s upserted=%s",
                 len(rows), raw_res["matched"], raw_res["modified"], raw_res["upserted"])
        log.info("CLEAN stats | in=%s out=%s nulled=%s dropped_ts=%s dropped_air=%s",
                 stats["input_rows"], stats["output_rows"], stats["nulled_out_of_range"],
                 stats["dropped_bad_timestamp"], stats["dropped_all_air_missing"])
        log.info("CLEAN upsert | matched=%s modified=%s upserted=%s",
                 clean_res["matched"], clean_res["modified"], clean_res["upserted"])

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(sleep_sec)

    log.info("Backfill done | raw_upserted=%s clean_upserted=%s", total_raw_upserted, total_clean_upserted)


if __name__ == "__main__":
    main()