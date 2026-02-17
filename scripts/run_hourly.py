from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqi.config import (
    CITY, COUNTRY, LAT, LON,
    HTTP_BACKOFF_SEC, HTTP_RETRIES, HTTP_TIMEOUT_SEC,
    MONGO_DB, MONGO_RAW_COLLECTION, MONGO_CLEAN_COLLECTION, MONGO_URI,
    PAST_HOURS, FORECAST_HOURS, SOURCE_NAME,
)
from aqi.mongo import ensure_timestamp_indexes, get_collection, upsert_by_timestamp
from aqi.openmeteo import OpenMeteoClient
from aqi.cleaning import clean_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_hourly")


def main() -> None:
    client = OpenMeteoClient(
        timeout_sec=HTTP_TIMEOUT_SEC,
        retries=HTTP_RETRIES,
        backoff_sec=HTTP_BACKOFF_SEC,
    )

    log.info("Fetching hourly Karachi data (past_hours=%s)", PAST_HOURS)
    rows = client.fetch_recent_hourly(LAT, LON, PAST_HOURS, FORECAST_HOURS)

    for r in rows:
        r["city"] = CITY
        r["country"] = COUNTRY
        r["source"] = SOURCE_NAME

    raw_col = get_collection(MONGO_URI, MONGO_DB, MONGO_RAW_COLLECTION)
    clean_col = get_collection(MONGO_URI, MONGO_DB, MONGO_CLEAN_COLLECTION)

    ensure_timestamp_indexes(raw_col)
    ensure_timestamp_indexes(clean_col)

    raw_res = upsert_by_timestamp(raw_col, rows)

    cleaned, stats = clean_rows(rows)
    clean_res = upsert_by_timestamp(clean_col, cleaned)

    log.info("RAW upsert | matched=%s modified=%s upserted=%s", raw_res["matched"], raw_res["modified"], raw_res["upserted"])
    log.info("CLEAN stats | in=%s out=%s nulled=%s dropped_ts=%s dropped_air=%s",
             stats["input_rows"], stats["output_rows"], stats["nulled_out_of_range"],
             stats["dropped_bad_timestamp"], stats["dropped_all_air_missing"])
    log.info("CLEAN upsert | matched=%s modified=%s upserted=%s", clean_res["matched"], clean_res["modified"], clean_res["upserted"])


if __name__ == "__main__":
    main()