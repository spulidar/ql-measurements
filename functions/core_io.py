"""
MILGRAU - Core Input/Output Module
Handles configuration loading, robust logging setup, directory management,
and raw Licel binary file scanning/parsing.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

import os, re, logging, yaml

import pandas as pd
import numpy as np
import netCDF4 as nc

from collections import defaultdict
from typing import Dict, Optional

from statistics import mode
from datetime import datetime, timedelta

from pathlib import Path
from siphon.simplewebservice.wyoming import WyomingUpperAir
import urllib.request, json

# Import MILGRAU core functions 
from functions.physics_utils import classify_period, get_night_date


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

def read_licel_header(filepath: str):
    """
    Reads the global metadata from the binary file header.
    """
    try:
        with open(filepath, "rb") as f:
            _ = f.readline()  # Line 1: filename, discard
            line2 = f.readline().decode("utf-8", errors='ignore').strip()
            line3 = f.readline().decode("utf-8", errors='ignore').strip()
        
        # Line 2 format: "Location StartDate StartTime StopDate StopTime Altitude Lon Lat Zenith"
        # We slice the strictly formatted date-time block (19 chars: DD/MM/YYYY HH:MM:SS)
        start_time_str = line2[8:27].strip()
        stop_time_str = line2[28:47].strip()
        
        start_time_utc = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
        stop_time_utc = datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S")
        duration = (stop_time_utc - start_time_utc).total_seconds()
        
        # Line 3 format dynamically split: e.g. "0000000 0100 0003001 0100 12"
        parts = line3.split()
        n_shots = int(parts[2])     # 0003001 -> 3001
        laser_freq = int(parts[3])  # 0100 -> 100
        
        return start_time_utc, stop_time_utc, duration, n_shots, laser_freq
    except Exception as e:
        return None, None, None, None, None

def _parse_single_licel_file(filepath: str) -> dict:
    """
    Reads a single Licel binary file, fully customized for SPU-Lidar hardware.
    """
    with open(filepath, 'rb') as f:
        _ = f.readline() # Line 1
        _ = f.readline() # Line 2
        line3 = f.readline().decode('utf-8', errors='ignore').strip()
        
        parts3 = line3.split()
        n_shots = int(parts3[2])
        laser_freq = int(parts3[3])
        num_channels = int(parts3[4])
        
        channels_meta = []
        for _ in range(num_channels):
            ch_line = f.readline().decode('utf-8', errors='ignore').strip()
            parts = ch_line.split()
            
            # Base properties
            active = int(parts[0])
            is_photon_counting = bool(int(parts[1]))
            num_points = int(parts[3])
            
            # String formatting for wavelength (e.g., '01064.o' -> '1064')
            raw_wl = parts[7]
            clean_wl = re.sub(r'[^0-9]', '', raw_wl.split('.')[0]).lstrip('0')
            if not clean_wl: clean_wl = "0"
            
            ch_name = f"{clean_wl}.{'PC' if is_photon_counting else 'AN'}"
            
            # Hardware specific constants from header
            adc_bits = int(parts[12]) if len(parts) > 12 else 12
            if adc_bits == 0 and not is_photon_counting:
                adc_bits = 12 # Safe fallback if missing
                
            # Range 
            adc_range_v = float(parts[14]) if len(parts) > 14 else 0.5
            adc_range_mv = adc_range_v * 1000.0 
            
            channels_meta.append({
                'name': ch_name,
                'active': active,
                'is_pc': is_photon_counting,
                'points': num_points,
                'adc_range': adc_range_mv,
                'adc_bits': adc_bits
            })
            
        # Skip empty line and read binary contiguous memory
        f.readline() 
        binary_payload = np.fromfile(f, dtype=np.int32)
        
    # Math & Physics Tensor Construction
    data_dict = {}
    cursor = 0
    
    for ch in channels_meta:
        if ch['active'] == 0:
            continue
            
        raw_int_array = binary_payload[cursor : cursor + ch['points']]
        cursor += ch['points']
        
        if ch['is_pc']:
            # Photon Counting: Raw Accumulated counts (Float64 for later Dead-Time correction)
            physical_array = raw_int_array.astype(np.float64)
        else:
            # Analog Lidar Equation: mV = (Raw / Shots) * (Range_mV / 2^Bits)
            adc_factor = ch['adc_range'] / (2 ** ch['adc_bits'])
            physical_array = (raw_int_array.astype(np.float64) / n_shots) * adc_factor
            
        data_dict[ch['name']] = physical_array

    return {
        'data': data_dict,
        'shots': n_shots,
        'laser_freq': laser_freq,
        'channels_meta': channels_meta
    }

def build_measurement_inventory(raw_dir: str, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Scans directories, parses basic headers, converts timezones, and filters existing NetCDFs.
    """
    logger.info("Building raw data inventory...")
    file_paths, file_types = scan_raw_files(raw_dir, logger)
    if not file_paths:
        return pd.DataFrame()

    results = [read_licel_header(f) for f in file_paths]
    start_times_utc, stop_times, durations, nshots_list, laser_freqs = zip(*results)

    df_raw = pd.DataFrame({
        "filepath": file_paths, 
        "meas_type": file_types, 
        "start_time_utc": start_times_utc, 
        "stop_time": stop_times, 
        "nshots": nshots_list, 
        "duration": durations, 
        "laser_freq": laser_freqs,
    }).dropna(subset=['start_time_utc'])

    if df_raw.empty:
        return df_raw

    # Timezone alignments 
    df_raw['start_time_utc'] = pd.to_datetime(df_raw['start_time_utc']).dt.tz_localize('UTC')
    df_raw['start_time_local'] = df_raw['start_time_utc'].dt.tz_convert('America/Sao_Paulo')
    
    # Generate unique ID 
    df_raw['meas_id'] = (
        df_raw['start_time_local'].apply(get_night_date).dt.strftime('%Y%m%d') + 
        df_raw['start_time_local'].apply(classify_period)
    )

    # Identifies real measurements
    valid_mids = df_raw[df_raw['meas_type'] == 'measurements']['meas_id'].unique()
    
    # Identifies orphans (only dark current measurements in time period)
    orphan_mids = set(df_raw['meas_id'].unique()) - set(valid_mids)
    
    if orphan_mids:
        logger.info(f"   -> Reassigning orphaned Dark Current groups: {orphan_mids} to nearest measurements...")
        valid_df = df_raw[df_raw['meas_id'].isin(valid_mids)].copy()
        
        if not valid_df.empty:
            for idx, row in df_raw[df_raw['meas_id'].isin(orphan_mids)].iterrows():
                time_diffs = abs(valid_df['start_time_utc'] - row['start_time_utc'])
                # Closest real measurement ID to adopt orphan DC
                closest_idx = time_diffs.idxmin()
                df_raw.at[idx, 'meas_id'] = valid_df.at[closest_idx, 'meas_id']
        else:
            df_raw = df_raw[~df_raw['meas_id'].isin(orphan_mids)]

    # Incremental processing filter
    if config['processing'].get('incremental', True):
        netcdf_dir = os.path.join(os.getcwd(), config['directories']['processed_data'])
        existing_mids = []
        for mid in df_raw["meas_id"].unique():
            save_id = f"{mid[:8]}sa{mid[8:]}"
            expected_path = os.path.join(netcdf_dir, mid[:4], mid[4:6], save_id, f"{save_id}.nc")
            if os.path.exists(expected_path):
                existing_mids.append(mid)
                
        df_raw = df_raw[~df_raw["meas_id"].isin(existing_mids)]
        logger.info(f"Incremental mode: Skipped {len(existing_mids)} already processed groups.")

    return df_raw

def parse_licel_group(filepaths: list, logger: logging.Logger) -> dict:
    """
    Orchestrates the reading of multiple Licel files into Time x Range matrices.
    """
    logger.info(f"    -> Parsing {len(filepaths)} raw binary files...")
    
    time_series = defaultdict(list)
    global_shots = []
    
    # Process each file chronologically
    for fp in sorted(filepaths):
        try:
            parsed = _parse_single_licel_file(fp)
            global_shots.append(parsed['shots'])
            
            for ch_name, array in parsed['data'].items():
                time_series[ch_name].append(array)
        except Exception as e:
            logger.warning(f"    -> Failed to read {os.path.basename(fp)}: {e}")
            continue

    # Stack into 2D Tensors (Time x Points)
    tensor_dict = {}
    for ch_name, list_of_arrays in time_series.items():
        tensor_dict[ch_name] = np.vstack(list_of_arrays)

    # We assume mode of shots for the NetCDF global metadata
    accumulated_shots = int(mode(global_shots)) if global_shots else 0

    return {
        'tensors': tensor_dict,
        'shots': accumulated_shots,
        'channels': list(tensor_dict.keys())
    }
   
def fetch_wyoming_radiosonde(measurement_dt_utc, station_id, logger, cache_dir="01-data/wyoming_cache"):
    """
    Fetches Wyoming radiosonde data (Pressure, Altitude, Temperature) 
    using the Siphon library.
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
        # Fetch data 
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
        return None
