from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# Fixed scope: Karachi only
CITY = "Karachi"
COUNTRY = "Pakistan"
LAT = 24.8607
LON = 67.0011

# Hourly fetch window
PAST_HOURS = 6
FORECAST_HOURS = 0

# HTTP
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "25"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "4"))
HTTP_BACKOFF_SEC = float(os.getenv("HTTP_BACKOFF_SEC", "1.0"))

# Mongo
MONGO_URI = os.getenv("MONGO_URI", "").strip()
if not MONGO_URI:
    raise ValueError("Missing required env var: MONGO_URI")

MONGO_DB = os.getenv("MONGO_DB", "aqi_predictor").strip()
MONGO_RAW_COLLECTION = os.getenv("MONGO_RAW_COLLECTION", "raw_hourly").strip()
MONGO_CLEAN_COLLECTION = os.getenv("MONGO_CLEAN_COLLECTION", "clean_hourly").strip()

SOURCE_NAME = "open-meteo"