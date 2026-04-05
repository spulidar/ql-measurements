"""
MILGRAU Suite - Core Input/Output Module
Handles configuration loading, robust logging setup, directory management,
and raw Licel binary file scanning/parsing.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import logging
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from siphon.simplewebservice.wyoming import WyomingUpperAir
import urllib.request
import json
from typing import Dict, Optional

def load_config(config_path="config.yaml"):
    """
    Loads the master configuration YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML configuration: {exc}")

def setup_logger(module_name, log_dir="logs"):
    """
    Sets up a standardized logger for the MILGRAU suite.
    Prevents duplicate logging handlers if called multiple times.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{module_name}_run_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates during interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger

def ensure_directories(*directories):
    """
    Safely creates multiple directories if they do not exist.
    """
    for d in directories:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except OSError as e:
                logging.error(f"Failed to create directory {d}: {e}")

def scan_raw_files(datadir_name, logger=None):
    """
    Scans the directory for valid Licel binary files.
    Cleans up spurious files (.dat, .dpp, .zip) and flags dark current measurements.
    """
    filepath = []
    meas_type = []

    if not os.path.exists(datadir_name):
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            full_path = os.path.join(dirpath, file)
            
            # Clean up spurious files directly
            if file.endswith((".dat", ".dpp", ".zip")):
                try:
                    os.remove(full_path)
                    if logger:
                        logger.debug(f"Spurious file deleted: {file}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not delete file {file}: {e}")
            else:
                filepath.append(full_path)
                
                # Check if it's a dark current measurement based purely on path string
                if "dark" in full_path.lower():
                    meas_type.append("dark_current")
                else:
                    meas_type.append("measurements")
                    
    return filepath, meas_type

def read_licel_header(filepath):
    """
    Reads the binary file header to extract vital metadata.
    Returns UTC times, duration, number of shots, and laser frequency.
    """
    try:
        with open(filepath, "rb") as f:
            _ = f.readline().decode("utf-8")
            lines = [f.readline().decode("utf-8") for _ in range(3)]
        
        start_time_str = lines[0][10:29].strip()
        stop_time_str = lines[0][30:49].strip()
        n_shots = int(lines[1][16:21])
        laser_freq = int(lines[1][22:27])
        
        start_time_utc = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
        stop_time_utc = datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S")
        duration = (stop_time_utc - start_time_utc).total_seconds()
        
        return start_time_utc, stop_time_utc, duration, n_shots, laser_freq
    except Exception as e:
        return None, None, None, None, None


def fetch_wyoming_radiosonde(measurement_dt_utc, station_id, logger, cache_dir="01-data/wyoming_cache"):
    """
    Fetches Wyoming radiosonde data (Pressure, Altitude, Temperature) 
    using the Siphon library, bypassing manual HTML parsing.
    Uses a local cache to prevent redundant downloads and timeouts.
    Returns a clean Pandas DataFrame or None if the server fails.
    """
    # Determine the closest sounding time (00Z or 12Z)
    hour_utc = measurement_dt_utc.hour
    if 0 <= hour_utc <= 8:
        target_dt = measurement_dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elif 9 <= hour_utc <= 20:
        target_dt = measurement_dt_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    else: # 21 to 23 belongs to the next day's 00Z sounding
        target_dt = (measurement_dt_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
    year = target_dt.strftime('%Y')
    month = target_dt.strftime('%m')
    
    # Setup local cache path based on the sounding time
    cache_path = Path(os.getcwd()) / cache_dir / year / month
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename
    
    # Check if we already have this specific sounding in our local cache
    if cache_file.exists():
        logger.info(f"  -> [RADIOSONDE] Cached sounding found: {cache_filename}. Skipping download.")
        return pd.read_csv(cache_file)

    logger.info(f"  -> [RADIOSONDE] Fetching {target_dt.strftime('%Y-%m-%d %H:%M')}Z for station {station_id} via Siphon...")
    
    try:
        # Fetch data using Siphon (Zero HTML parsing required!)
        df_raw = WyomingUpperAir.request_data(target_dt, station_id)
        df = df_raw.drop_duplicates(subset=['height'], keep='first').sort_values('height')
        df.to_csv(cache_file, index=False)
        logger.info("  -> [OK] Radiosonde data successfully fetched and cached!")
        
        return df
        
    except Exception as e:
        logger.warning(f"  -> [RADIOSONDE ERROR] Failed to fetch data from Wyoming via Siphon: {e}")
        return None

def fetch_surface_weather(dt_utc: datetime, lat: float, lon: float) -> Optional[Dict[str, float]]:
    """
    Fetches historical surface weather from the Open-Meteo Archive API.
    Returns a dictionary with temperature, pressure, humidity, cloud cover, and wind.
    """
    try:
        # Format time to ISO 8601 hourly format required by Open-Meteo
        target_time = dt_utc.strftime("%Y-%m-%dT%H:00")
        target_date = dt_utc.strftime("%Y-%m-%d")
        
        # ERA5 Archive API URL with extended meteorological variables
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={target_date}&end_date={target_date}&"
            f"hourly=temperature_2m,surface_pressure,relative_humidity_2m,cloud_cover,wind_speed_10m"
        )
        
        req = urllib.request.Request(url, headers={'User-Agent': 'MILGRAU-Lidar-Bot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        times = data.get('hourly', {}).get('time', [])
        
        if target_time in times:
            idx = times.index(target_time)
            return {
                'temperature_c': data['hourly']['temperature_2m'][idx],
                'pressure_hpa': data['hourly']['surface_pressure'][idx],
                'relative_humidity_percent': data['hourly']['relative_humidity_2m'][idx],
                'cloud_cover_percent': data['hourly']['cloud_cover'][idx],
                'wind_speed_kmh': data['hourly']['wind_speed_10m'][idx]
            }
            
        return None
    except Exception:
        # Fails silently to allow fallback values in the main pipeline
        return None
