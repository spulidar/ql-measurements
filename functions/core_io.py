"""
MILGRAU - Core Input/Output Module

Utilities shared by the MILGRAU processing levels: configuration loading,
logging, directory handling, Licel file discovery/parsing, measurement inventory
construction, and external meteorological data retrieval.

Design rule
-----------
This module contains reusable infrastructure and I/O primitives. Scientific
orchestration decisions that are specific to one processing level should remain
in that level's script, for example LIBIDS, LIPANCORA or LEBEAR.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import urllib.request
import yaml

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mode
from typing import Optional, Tuple

import netCDF4 as nc  # Kept for compatibility with existing imports.
import numpy as np
import pandas as pd

from siphon.simplewebservice.wyoming import WyomingUpperAir
from tenacity import retry, stop_after_attempt, wait_exponential

from functions.physics_utils import classify_period, get_night_date


# =============================================================================
# Configuration and logging
# =============================================================================


def _validate_config_minimum(config: dict) -> None:
    """Perform a lightweight schema check for the YAML configuration."""
    required_sections = ("directories", "processing", "physics", "hardware")
    missing = [section for section in required_sections if section not in config]

    if missing:
        raise KeyError(
            "Configuration file is missing required section(s): "
            + ", ".join(missing)
        )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and minimally validate the MILGRAU YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML configuration: {exc}") from exc

    if config is None:
        raise RuntimeError(f"Configuration file is empty: {config_path}")

    _validate_config_minimum(config)
    return config


def setup_logger(module_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a standardized logger for MILGRAU modules.

    Existing handlers from the same logger name are cleared to avoid duplicate
    messages in notebooks, IDE sessions, or repeated script calls.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        f"{module_name}_run_{datetime.now().strftime('%Y%m%d')}.log",
    )

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def ensure_directories(*directories: str | Path) -> None:
    """Create one or more directories if they do not already exist."""
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.error(f"Failed to create directory {directory}: {exc}")


# =============================================================================
# Raw file discovery and sanitization
# =============================================================================


def _processing_option(config: Optional[dict], key: str, default):
    """Return an optional processing setting from config."""
    if not config:
        return default
    return config.get("processing", {}).get(key, default)


def _quarantine_file(
    path: Path,
    quarantine_root: Path,
    logger: Optional[logging.Logger],
) -> None:
    """Move a spurious file to a quarantine folder instead of deleting it."""
    ensure_directories(quarantine_root)

    destination = quarantine_root / path.name
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = quarantine_root / f"{path.stem}_{stamp}{path.suffix}"

    shutil.move(str(path), str(destination))

    if logger:
        logger.info(f"  -> Spurious file quarantined: {path.name} -> {destination}")


def scan_raw_files(
    datadir_name: str,
    logger: Optional[logging.Logger] = None,
    config: Optional[dict] = None,
) -> Tuple[list[str], list[str]]:
    """
    Scan the raw-data tree for Licel files and classify dark-current files.

    Spurious auxiliary files are no longer deleted by default. They can be moved
    to a quarantine folder, ignored, or deleted depending on YAML keys under
    ``processing``.
    """
    filepath: list[str] = []
    meas_type: list[str] = []

    raw_root = Path(datadir_name)
    if not raw_root.exists():
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    spurious_extensions = tuple(
        ext.lower()
        for ext in _processing_option(
            config,
            "spurious_extensions",
            [".dat", ".dpp", ".zip"],
        )
    )
    quarantine_spurious = bool(_processing_option(config, "quarantine_spurious_files", True))
    delete_spurious = bool(_processing_option(config, "delete_spurious_files", False))
    quarantine_dir = Path(
        _processing_option(config, "quarantine_dir", str(raw_root / "_quarantine"))
    )

    for dirpath, dirnames, files in os.walk(raw_root):
        dirnames.sort()
        files.sort()

        # Do not recursively process the quarantine folder.
        try:
            quarantine_resolved = quarantine_dir.resolve()
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if Path(dirpath, dirname).resolve() != quarantine_resolved
            ]
        except Exception:
            pass

        for file_name in files:
            full_path = Path(dirpath) / file_name
            suffix = full_path.suffix.lower()

            if suffix in spurious_extensions:
                try:
                    if delete_spurious:
                        full_path.unlink()
                        if logger:
                            logger.info(f"  -> Spurious file deleted: {full_path.name}")
                    elif quarantine_spurious:
                        _quarantine_file(full_path, quarantine_dir, logger)
                    else:
                        if logger:
                            logger.debug(f"  -> Spurious file ignored: {full_path.name}")
                except Exception as exc:
                    if logger:
                        logger.warning(f"Could not handle spurious file {full_path}: {exc}")
                continue

            filepath.append(str(full_path))
            if "dark" in str(full_path).lower():
                meas_type.append("dark_current")
            else:
                meas_type.append("measurements")

    return filepath, meas_type


# =============================================================================
# Licel header and payload parsing
# =============================================================================


def _extract_licel_datetimes(line2: str) -> Tuple[datetime, datetime]:
    """
    Extract start/stop datetimes from the second Licel header line.

    Preferred method: regular expression over explicit DD/MM/YYYY HH:MM:SS
    tokens. Fallback: the fixed slices used by the original SPU-Lidar parser.
    """
    matches = re.findall(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}", line2)

    if len(matches) >= 2:
        start_time_str, stop_time_str = matches[0], matches[1]
    else:
        start_time_str = line2[8:27].strip()
        stop_time_str = line2[28:47].strip()

    start_time_utc = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
    stop_time_utc = datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S")
    return start_time_utc, stop_time_utc


def read_licel_header(
    filepath: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[datetime], Optional[datetime], Optional[float], Optional[int], Optional[int]]:
    """
    Read global metadata from a Licel binary header.

    Invalid files return five None values so the inventory builder can drop them
    safely.
    """
    try:
        with open(filepath, "rb") as file:
            _ = file.readline()
            line2 = file.readline().decode("utf-8", errors="ignore").strip()
            line3 = file.readline().decode("utf-8", errors="ignore").strip()

        start_time_utc, stop_time_utc = _extract_licel_datetimes(line2)
        duration = (stop_time_utc - start_time_utc).total_seconds()

        parts = line3.split()
        n_shots = int(parts[2])
        laser_freq = int(parts[3])

        if n_shots <= 0:
            raise ValueError(f"invalid number of laser shots: {n_shots}")

        return start_time_utc, stop_time_utc, duration, n_shots, laser_freq

    except Exception as exc:
        if logger:
            logger.warning(f"  -> Invalid Licel header skipped: {filepath} ({exc})")
        return None, None, None, None, None


def _parse_single_licel_file(filepath: str) -> dict:
    """
    Read one SPU-Lidar Licel binary file into physical channel arrays.

    Photon-counting channels are returned as raw accumulated counts. Analog
    channels are converted to mV per shot using ADC metadata from the header.
    """
    with open(filepath, "rb") as file:
        _ = file.readline()
        _ = file.readline()
        line3 = file.readline().decode("utf-8", errors="ignore").strip()

        parts3 = line3.split()
        n_shots = int(parts3[2])
        laser_freq = int(parts3[3])
        num_channels = int(parts3[4])

        if n_shots <= 0:
            raise ValueError(f"Invalid number of laser shots in {filepath}: {n_shots}")

        channels_meta = []
        for _ in range(num_channels):
            ch_line = file.readline().decode("utf-8", errors="ignore").strip()
            parts = ch_line.split()

            active = int(parts[0])
            is_photon_counting = bool(int(parts[1]))
            num_points = int(parts[3])

            raw_wl = parts[7]
            clean_wl = re.sub(r"[^0-9]", "", raw_wl.split(".")[0]).lstrip("0")
            if not clean_wl:
                clean_wl = "0"

            ch_name = f"{clean_wl}.{'PC' if is_photon_counting else 'AN'}"

            adc_bits = int(parts[12]) if len(parts) > 12 else 12
            if adc_bits == 0 and not is_photon_counting:
                adc_bits = 12

            adc_range_v = float(parts[14]) if len(parts) > 14 else 0.5
            adc_range_mv = adc_range_v * 1000.0

            channels_meta.append(
                {
                    "name": ch_name,
                    "active": active,
                    "is_pc": is_photon_counting,
                    "points": num_points,
                    "adc_range": adc_range_mv,
                    "adc_bits": adc_bits,
                }
            )

        file.readline()
        binary_payload = np.fromfile(file, dtype=np.int32)

    expected_payload_points = sum(ch["points"] for ch in channels_meta if ch["active"] != 0)
    if binary_payload.size < expected_payload_points:
        raise ValueError(
            f"Binary payload too short in {filepath}: "
            f"expected {expected_payload_points}, found {binary_payload.size}"
        )

    data_dict: dict[str, np.ndarray] = {}
    cursor = 0

    for ch in channels_meta:
        if ch["active"] == 0:
            continue

        ch_name = ch["name"]
        if ch_name in data_dict:
            raise ValueError(f"Duplicated active channel name in {filepath}: {ch_name}")

        raw_int_array = binary_payload[cursor : cursor + ch["points"]]
        cursor += ch["points"]

        if ch["is_pc"]:
            physical_array = raw_int_array.astype(np.float64)
        else:
            adc_factor = ch["adc_range"] / (2 ** ch["adc_bits"])
            physical_array = (raw_int_array.astype(np.float64) / n_shots) * adc_factor

        data_dict[ch_name] = physical_array

    return {
        "data": data_dict,
        "shots": n_shots,
        "laser_freq": laser_freq,
        "channels_meta": channels_meta,
    }


# =============================================================================
# Measurement inventory and group parsing
# =============================================================================


def _effective_timezone(config: dict) -> str:
    """Return the site timezone from YAML, with São Paulo as fallback."""
    return (
        config.get("site", {}).get("timezone")
        or config.get("location", {}).get("timezone")
        or "America/Sao_Paulo"
    )


def build_measurement_inventory(
    raw_dir: str,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Build the Level-0 processing inventory from raw files.

    Responsibilities: scan raw files, parse basic Licel headers, convert UTC
    timestamps to local station time, assign MILGRAU measurement IDs, associate
    orphan dark-current files with nearby measurements, and skip already
    processed groups in incremental mode.
    """
    logger.info("Building raw data inventory...")

    file_paths, file_types = scan_raw_files(raw_dir, logger=logger, config=config)
    if not file_paths:
        return pd.DataFrame()

    records = []
    for filepath, file_type in zip(file_paths, file_types):
        start_time_utc, stop_time, duration, n_shots, laser_freq = read_licel_header(
            filepath,
            logger=logger,
        )
        if start_time_utc is None:
            continue

        records.append(
            {
                "filepath": filepath,
                "meas_type": file_type,
                "start_time_utc": start_time_utc,
                "stop_time": stop_time,
                "nshots": n_shots,
                "duration": duration,
                "laser_freq": laser_freq,
            }
        )

    df_raw = pd.DataFrame.from_records(records)
    if df_raw.empty:
        return df_raw

    timezone = _effective_timezone(config)
    df_raw["start_time_utc"] = pd.to_datetime(df_raw["start_time_utc"]).dt.tz_localize("UTC")
    df_raw["start_time_local"] = df_raw["start_time_utc"].dt.tz_convert(timezone)

    df_raw["meas_id"] = (
        df_raw["start_time_local"].apply(get_night_date).dt.strftime("%Y%m%d")
        + df_raw["start_time_local"].apply(classify_period)
    )

    valid_mids = df_raw.loc[df_raw["meas_type"] == "measurements", "meas_id"].unique()
    orphan_mids = set(df_raw["meas_id"].unique()) - set(valid_mids)

    if orphan_mids:
        logger.info(
            f"   -> Reassigning orphaned Dark Current groups: {orphan_mids} "
            "to nearest measurements..."
        )
        valid_df = df_raw[df_raw["meas_id"].isin(valid_mids)].copy()
        max_hours = config.get("processing", {}).get("dark_current_max_association_hours")

        if not valid_df.empty:
            for idx, row in df_raw[df_raw["meas_id"].isin(orphan_mids)].iterrows():
                time_diffs = abs(valid_df["start_time_utc"] - row["start_time_utc"])
                closest_idx = time_diffs.idxmin()
                closest_diff_h = time_diffs.loc[closest_idx].total_seconds() / 3600.0

                if max_hours is not None and closest_diff_h > float(max_hours):
                    logger.warning(
                        "   -> Orphan dark current not reassigned; nearest measurement "
                        f"is {closest_diff_h:.2f} h away: {row['filepath']}"
                    )
                    continue

                df_raw.at[idx, "meas_id"] = valid_df.at[closest_idx, "meas_id"]
        else:
            df_raw = df_raw[~df_raw["meas_id"].isin(orphan_mids)]

    if config.get("processing", {}).get("incremental", True):
        netcdf_dir = os.path.join(os.getcwd(), config["directories"]["processed_data"])
        existing_mids = []

        for mid in df_raw["meas_id"].unique():
            save_id = f"{mid[:8]}sa{mid[8:]}"
            expected_path = os.path.join(
                netcdf_dir,
                mid[:4],
                mid[4:6],
                save_id,
                f"{save_id}.nc",
            )
            if os.path.exists(expected_path):
                existing_mids.append(mid)

        df_raw = df_raw[~df_raw["meas_id"].isin(existing_mids)]
        logger.info(f"Incremental mode: Skipped {len(existing_mids)} already processed groups.")

    return df_raw.reset_index(drop=True)


def parse_licel_group(filepaths: list[str], logger: logging.Logger) -> dict:
    """
    Parse multiple Licel files into time x range tensors.

    Files are processed chronologically. The function enforces that every file in
    a group has the same active channel set, point counts, and laser frequency.
    Incompatible files are skipped and reported to the logger.
    """
    logger.info(f"    -> Parsing {len(filepaths)} raw binary files...")

    time_series: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    global_shots: list[int] = []

    baseline_channels: Optional[tuple[str, ...]] = None
    baseline_points: Optional[dict[str, int]] = None
    baseline_laser_freq: Optional[int] = None

    for filepath in sorted(filepaths):
        try:
            parsed = _parse_single_licel_file(filepath)
            data = parsed["data"]
            channels = tuple(sorted(data.keys()))
            points = {ch_name: int(array.shape[0]) for ch_name, array in data.items()}
            laser_freq = int(parsed["laser_freq"])

            if baseline_channels is None:
                baseline_channels = channels
                baseline_points = points
                baseline_laser_freq = laser_freq
            else:
                if channels != baseline_channels:
                    logger.warning(
                        f"    -> Skipping incompatible file {os.path.basename(filepath)}: "
                        f"channel set {channels} differs from baseline {baseline_channels}."
                    )
                    continue

                if points != baseline_points:
                    logger.warning(
                        f"    -> Skipping incompatible file {os.path.basename(filepath)}: "
                        "range-bin count differs from baseline."
                    )
                    continue

                if laser_freq != baseline_laser_freq:
                    logger.warning(
                        f"    -> Skipping incompatible file {os.path.basename(filepath)}: "
                        f"laser frequency {laser_freq} differs from baseline {baseline_laser_freq}."
                    )
                    continue

            global_shots.append(int(parsed["shots"]))
            for ch_name, array in data.items():
                time_series[ch_name].append(array)

        except Exception as exc:
            logger.warning(f"    -> Failed to read {os.path.basename(filepath)}: {exc}")
            continue

    tensor_dict: dict[str, np.ndarray] = {}
    for ch_name, list_of_arrays in time_series.items():
        tensor_dict[ch_name] = np.vstack(list_of_arrays)

    accumulated_shots = int(mode(global_shots)) if global_shots else 0
    return {
        "tensors": tensor_dict,
        "shots": accumulated_shots,
        "channels": list(tensor_dict.keys()),
    }


# =============================================================================
# External meteorological data
# =============================================================================


def return_none_on_failure(retry_state):
    """Tenacity callback used by API functions to fail gracefully."""
    return None


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


def _weather_cache_file(
    dt_utc: datetime,
    lat: float,
    lon: float,
    cache_dir: str,
) -> Path:
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
    """
    Fetch historical surface weather from Open-Meteo Archive API.

    A daily JSON cache avoids repeated API calls during reprocessing. The return
    dictionary is compatible with LIBIDS global NetCDF attributes.
    """
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
