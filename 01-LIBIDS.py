"""
MILGRAU - Level 0: LIdar BInary Data Standardized (LIBIDS)
Reads raw Licel binary data, sanitizes spurious files, classifies measurement
periods (UTC to Local Time), and converts valid data into SCC compliant NetCDFs.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

import os
import traceback
import logging
import pandas as pd
import numpy as np
import netCDF4 as nc
from statistics import mode

# Import MILGRAU core functions 
from functions.core_io import (
    load_config, 
    setup_logger, 
    ensure_directories, 
    build_measurement_inventory,
    fetch_surface_weather,
    parse_licel_group,
)

def filter_laser_shots(df_raw: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Evaluates laser shots quality and consistency per measurement period.
    Rejects files with 0 shots or deviations greater than 0.2% of the expected mode.
    """
    logger.info("Evaluating laser shots quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            expected_shots = mode(group["nshots"])
            # Filter criteria: 0 shots or deviation greater than 0.2% of expected
            bad_condition = (group["nshots"] == 0) | (abs(group["nshots"] - expected_shots) >= 2e-3 * expected_shots)
            
            bad_group = group.loc[bad_condition]
            good_group = group.loc[~bad_condition]
            
            total_files = len(group)
            bad_files = len(bad_group)
            loss_percent = (bad_files / total_files) * 100 if total_files > 0 else 0
            
            if bad_files > 0:
                log_msg = f"  -> [{meas_id}] QA Report: {bad_files}/{total_files} files rejected ({loss_percent:.1f}% loss)."
                if loss_percent > 10.0:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)
            else:
                logger.info(f"  -> [{meas_id}] QA Report: 100% data retention. No files rejected.")

            if not good_group.empty:
                good_groups.append(good_group)
                
        except Exception as e:
            logger.warning(f"  -> [{meas_id}] Error evaluating quality: {e}")
            
    return pd.concat(good_groups).reset_index(drop=True) if good_groups else pd.DataFrame()

def build_netcdf(netcdf_path: str, save_id: str, period: str, lidar_data: dict, group_df: pd.DataFrame, weather_data: dict, config: dict, logger: logging.Logger):
    """
    Generates an SCC-compliant NetCDF from numpy matrices.
    Injects meteorological data globally and appends Dark Current (Background_Profile) if available.
    """
    try:
        tensors = lidar_data['tensors']
        channels = lidar_data['channels']
        
        # Hardware ID Routing (Day/Night)
        system_mode = 'night' if period == 'nt' else 'day'
        hardware_map = config['hardware'].get('name_to_id', {}).get(system_mode, {})
        
        # Time Vectors
        min_start_utc = pd.to_datetime(group_df['start_time_utc']).min()
        max_stop_utc = pd.to_datetime(group_df['stop_time']).max()
        df_meas = group_df[group_df["meas_type"] == "measurements"]
        start_times_epoch = (df_meas['start_time_utc'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
        
        with nc.Dataset(netcdf_path, 'w', format='NETCDF4') as ds:
            # Global Attributes 
            ds.Measurement_ID = save_id
            ds.System = "SPU-Lidar"
            ds.Latitude_degrees_north = float(config['physics'].get('latitude', -23.561))
            ds.Longitude_degrees_east = float(config['physics'].get('longitude', -46.735))
            ds.Accumulated_Shots = lidar_data.get('shots')
            
            ds.RawData_Start_Date = min_start_utc.strftime("%Y%m%d")
            ds.RawData_Start_Time_UT = min_start_utc.strftime("%H%M%S")
            ds.RawData_Stop_Time_UT = max_stop_utc.strftime("%H%M%S")

            # Thermodynamics injected globally for Rayleigh calculation later
            ds.Temperature_C = weather_data.get('temperature_c', 25.0)
            ds.Pressure_hPa = weather_data.get('pressure_hpa', 940.0)
            ds.CloudCover_percent = weather_data.get('cloud_cover_percent')
            
            # Dimensions
            num_times = len(start_times_epoch)
            num_channels = len(channels)
            num_points = next(iter(tensors.values())).shape[1] 
            
            ds.createDimension('time', num_times)
            ds.createDimension('channels', num_channels)
            ds.createDimension('points', num_points)
            
            raw_data_start = ds.createVariable('Raw_Data_Start_Time', 'f8', ('time',))
            raw_data_start.units = "seconds since 1970-01-01 00:00:00"
            raw_data_start[:] = start_times_epoch.values
            raw_lidar_data = ds.createVariable('Raw_Lidar_Data', 'f8', ('time', 'channels', 'points'), zlib=True)

            # SCC Mandatory Configuration Variables
            channel_ids = ds.createVariable('channel_ID', 'i4', ('channels',))
            range_res = ds.createVariable('Raw_Data_Range_Resolution', 'f8', ('channels',))
            bg_low = ds.createVariable('Background_Low', 'f8', ('channels',))
            bg_high = ds.createVariable('Background_High', 'f8', ('channels',))
            channel_names = ds.createVariable('channel_string', str, ('channels',))
            
            # Fill Core Variables
            # Stack dictionaries into a 3D Tensor array (Time x Channels x Points)
            stacked_tensor = np.zeros((num_times, num_channels, num_points), dtype=np.float64)
            
            for i, ch_name in enumerate(channels):
                stacked_tensor[:, i, :] = tensors[ch_name]
                channel_names[i] = ch_name
                
                # Dynamic mapping of the SCC ID
                if ch_name not in hardware_map:
                    logger.warning(f"  -> Channel {ch_name} missing in config for {system_mode} mode. Using default 9999.")
                channel_ids[i] = hardware_map.get(ch_name, 9999)
                
                # Hardware and Background configuration
                range_res[i] = float(config['physics'].get('vertical_resolution_m', 7.5))
                bg_low[i] = float(config['physics'].get('bg_start', 29000))
                bg_high[i] = float(config['physics'].get('bg_stop', 29999))
                
            raw_lidar_data[:] = stacked_tensor
            
            # Dark Current / Background Profile Injection
            # If the group contains dark current files, we parse them and append to the NetCDF
            df_dc = group_df[group_df["meas_type"] == "dark_current"]
            
            if not df_dc.empty:
                
                dc_files = df_dc["filepath"].tolist()
                dc_data = parse_licel_group(dc_files, logger)
                
                if dc_data['tensors']:
                    num_time_bck = next(iter(dc_data['tensors'].values())).shape[0]
                    ds.createDimension('time_bck', num_time_bck)
                    
                    bck_prof = ds.createVariable('Background_Profile', 'f8', ('time_bck', 'channels', 'points'), zlib=True)
                    
                    stacked_dc = np.zeros((num_time_bck, num_channels, num_points), dtype=np.float64)
                    
                    for i, ch_name in enumerate(channels):
                        if ch_name in dc_data['tensors']:
                            stacked_dc[:, i, :] = dc_data['tensors'][ch_name]
                            
                    bck_prof[:] = stacked_dc
                    logger.info(f"  -> Successfully injected Dark Current matrix ({num_time_bck} profiles).")
                else:
                    logger.warning("  -> Dark current files found but parsing failed. NetCDF will lack Background_Profile.")
                    
    except Exception as e:
        raise RuntimeError(f"Failed to build NetCDF: {e}")
    
def process_level_0(config: dict, logger: logging.Logger):
    """
    Main orchestrator for Level 0 processing.
    """
    # Build the Data Inventory (Scanning, UTC conversion, ID generation)
    raw_dir = os.path.join(os.getcwd(), config['directories']['raw_data'])
    df_raw = build_measurement_inventory(raw_dir, config, logger)
    
    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        return

    # Quality Control: Filter out inconsistent laser shots
    df_good = filter_laser_shots(df_raw, logger)
    if df_good.empty:
        logger.warning("=== No data survived quality control. Exiting. ===")
        return

    # Process each measurement period
    out_base_dir = os.path.join(os.getcwd(), config['directories']['processed_data'])
    success_count = 0
    total_groups = len(df_good["meas_id"].unique())

    for meas_id, group_df in df_good.groupby("meas_id"):
        # Format standardized SCC save ID (e.g., 20230815sa12)
        save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
        year_str, month_str = save_id[:4], save_id[4:6]
        
        out_dir = os.path.join(out_base_dir, year_str, month_str, save_id)
        netcdf_path = os.path.join(out_dir, f"{save_id}.nc")

        logger.info(f"Processing group [{save_id}]...")

        try:
            # Fetch surface weather for Rayleigh molecular calibration
            lat = float(config['physics'].get('latitude', -23.561))
            lon = float(config['physics'].get('longitude', -46.735))
            dt_utc_mean = group_df['start_time_utc'].iloc[len(group_df) // 2].to_pydatetime()
            
            weather_data = fetch_surface_weather(dt_utc_mean, lat, lon)
            if not weather_data:
                logger.warning(f"  -> [{save_id}] Weather API failed. Using fallback physics standard atmosphere.")
                weather_data = {
                    'temperature_c': float(config['physics'].get('default_surface_temp_c', 25.0)),
                    'pressure_hpa': float(config['physics'].get('default_surface_pressure_hpa', 940.0)),
                    'relative_humidity_percent': 50.0,
                    'cloud_cover_percent': 0.0,
                    'wind_speed_kmh': 0.0
                }

            # Parse raw binary files into numpy tensors
            df_meas = group_df[group_df["meas_type"] == "measurements"]
            files_meas = df_meas["filepath"].tolist()

            if not files_meas:
                logger.warning(f"  -> [{save_id}] No measurement files found. Skipping.")
                continue
                
            # Extracts raw arrays: Time x Range x Channels (Analog & Photon Counting)
            lidar_data_tensors = parse_licel_group(files_meas, logger)

            # Build SCC-compliant NetCDF 
            ensure_directories(out_dir)
            build_netcdf(
                netcdf_path=netcdf_path,
                save_id=save_id,
                period=meas_id[8:],          
                lidar_data=lidar_data_tensors,
                group_df=group_df,          
                weather_data=weather_data,
                config=config,
                logger=logger
            )
            
            logger.info(f"  -> [OK] NetCDF successfully generated: {netcdf_path}")
            success_count += 1

        except Exception as e:
            logger.error(f"  -> [ERROR] Fatal error converting {save_id}:\n{traceback.format_exc()}")

    logger.info(f"=== Processed {success_count}/{total_groups} groups. LIBIDS finished! ===")

if __name__ == "__main__":
    config_dict = load_config()
    main_logger = setup_logger("LIBIDS", config_dict['directories']['log_dir'])
    main_logger.info("=== Starting MILGRAU LIBIDS processing (Level 0) ===")
    
    process_level_0(config_dict, main_logger)