"""Surface weather retrieval and caching utilities."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential


def return_none_on_failure(retry_state):
    """Tenacity callback used by API functions to fail gracefully."""
    return None


def _weather_cache_file(dt_utc: datetime, lat: float, lon: float, cache_dir: str) -> Path:
    """Build the cache filename for one day of Open-Meteo hourly data."""
    date_str = dt_utc.strftime("%Y-%m-%d")
    year = dt_utc.strftime("%Y")
    month = dt_utc.strftime("%m")
    lat_tag = f"{lat:.4f}".replace("-", "m").replace(".", "p")
    lon_tag = f"{lon:.4f}".replace("-", "m").replace(".", "p")
    cache_path = Path.cwd() / cache_dir / year / month
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"openmeteo_{lat_tag}_{lon_tag}_{date_str}.json"


def _extract_surface_weather_from_payload(payload: dict, target_time: str) -> Optional[dict]:
    """Extract one hourly record from an Open-Meteo archive payload."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if target_time not in times:
        return None
    idx = times.index(target_time)
    return {
        "temperature_c": hourly["temperature_2m"][idx],
        "pressure_hpa": hourly["surface_pressure"][idx],
        "relative_humidity_percent": hourly["relative_humidity_2m"][idx],
        "cloud_cover_percent": hourly["cloud_cover"][idx],
        "wind_speed_kmh": hourly["wind_speed_10m"][idx],
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=return_none_on_failure,
)
def fetch_surface_weather(
    dt_utc: datetime,
    lat: float,
    lon: float,
    logger: Optional[logging.Logger] = None,
    cache_dir: str = "01-data/openmeteo_cache",
) -> Optional[dict]:
    """Fetch historical surface weather from Open-Meteo Archive API."""
    target_time = dt_utc.strftime("%Y-%m-%dT%H:00")
    target_date = dt_utc.strftime("%Y-%m-%d")
    cache_file = _weather_cache_file(dt_utc, lat, lon, cache_dir)

    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            weather = _extract_surface_weather_from_payload(payload, target_time)
            if weather is not None:
                if logger:
                    logger.info(f"  -> [OPEN-METEO] Cached surface weather found: {cache_file.name}")
                return weather
        except Exception as exc:
            if logger:
                logger.warning(f"  -> [OPEN-METEO] Could not read cache {cache_file}: {exc}")

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={target_date}&end_date={target_date}&"
        "hourly=temperature_2m,surface_pressure,relative_humidity_2m,cloud_cover,wind_speed_10m"
    )
    if logger:
        logger.info(f"  -> [OPEN-METEO] Fetching surface weather for {target_time}...")

    req = urllib.request.Request(url, headers={"User-Agent": "SPU-Lidar"})
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _extract_surface_weather_from_payload(payload, target_time)
