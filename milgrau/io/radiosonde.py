"""Radiosonde retrieval and caching utilities for MILGRAU."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
from tenacity import retry, stop_after_attempt, wait_exponential

from milgrau.io.weather import return_none_on_failure


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=return_none_on_failure,
)
def fetch_wyoming_radiosonde(
    measurement_dt_utc: datetime,
    station_id: str,
    logger: logging.Logger,
    cache_dir: str = "01-data/wyoming_cache",
) -> Optional[pd.DataFrame]:
    """Fetch Wyoming radiosonde data and cache the cleaned table locally."""
    hour_utc = measurement_dt_utc.hour
    if 0 <= hour_utc <= 8:
        target_dt = measurement_dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elif 9 <= hour_utc <= 20:
        target_dt = measurement_dt_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        target_dt = (measurement_dt_utc + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    cache_path = Path.cwd() / cache_dir / year / month
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename

    if cache_file.exists():
        logger.info(f"  -> [RADIOSONDE] Cached sounding found: {cache_filename}. Skipping download.")
        return pd.read_csv(cache_file)

    logger.info(
        f"  -> [RADIOSONDE] Fetching {target_dt.strftime('%Y-%m-%d %H:%M')}Z "
        f"for station {station_id} via Siphon..."
    )

    df_raw = WyomingUpperAir.request_data(target_dt, station_id)
    df = df_raw.drop_duplicates(subset=["height"], keep="first").sort_values("height")
    df.to_csv(cache_file, index=False)

    metadata_file = cache_file.with_suffix(".json")
    metadata = {
        "station_id": station_id,
        "target_datetime_utc": target_dt.isoformat(),
        "download_datetime_utc": datetime.utcnow().isoformat(),
        "source": "University of Wyoming Upper Air via Siphon",
        "csv_file": cache_file.name,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("  -> [OK] Radiosonde data successfully fetched and cached!")
    return df
