"""
MILGRAU Suite - Level 0: LIdar BInary Data Standardized (LIBIDS)
Reads raw Licel binary data, sanitizes spurious files, classifies measurement
periods (UTC to Local Time), and converts valid data into SCC compliant NetCDFs.
"""

import os
import traceback
import logging
import pandas as pd

# Import MILGRAU core functions
from functions.core_io import (
    load_config, setup_logger, ensure_directories, scan_raw_files, read_licel_header,
    fetch_surface_weather, filter_laser_shots, append_attributes_to_nc
)
from functions.physics_utils import classify_period, get_night_date

# Import SCC specific libraries
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import msp_netcdf_parameters_system484, msp_netcdf_parameters_system565

logging.getLogger('atmospheric_lidar').setLevel(logging.ERROR)

class LidarMeasurement484(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system484

class LidarMeasurement565(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system565

def process_single_netcdf(args):
    meas_id, group_df, config = args
    save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
    year_str, month_str = save_id[:4], save_id[4:6]
    
    out_dir = os.path.join(os.getcwd(), config['directories']['processed_data'], year_str, month_str, save_id)
    netcdf_path = os.path.join(out_dir, f"{save_id}.nc")

    files_meas = group_df[group_df["meas_type"] == "measurements"]["filepath"].tolist()
    files_meas_dc = group_df[group_df["meas_type"] != "measurements"]["filepath"].tolist()

    if not files_meas: return f"[FAILED] No measurement files found for {save_id}"

    try:
        MeasurementClass = LidarMeasurement565 if meas_id[8:] in ["am", "pm"] else LidarMeasurement484
        my_measurement = MeasurementClass(files_meas)
        if files_meas_dc: my_measurement.dark_measurement = MeasurementClass(files_meas_dc)
            
        my_measurement.info["Measurement_ID"] = save_id
        
        # Hardware & Physics settings from mode
        my_measurement.info["Accumulated_Shots"] = str(group_df["nshots"].mode()[0])
        my_measurement.info["Laser_Frequency_Hz"] = str(group_df["laser_freq"].mode()[0])
        my_measurement.info["Measurements_Duration_s"] = str(group_df["duration"].mode()[0])

        # Weather / Metadata
        lat = float(config['physics'].get('latitude', -23.561))
        lon = float(config['physics'].get('longitude', -46.735))
        dt_utc_mean = group_df['start_time_utc'].iloc[len(group_df) // 2].to_pydatetime()
        weather_data = fetch_surface_weather(dt_utc_mean, lat, lon)
        
        if weather_data:
            my_measurement.info["Temperature_C"] = str(round(weather_data['temperature_c'], 1))
            my_measurement.info["Pressure_hPa"] = str(round(weather_data['pressure_hpa'], 1))
        else:
            my_measurement.info["Temperature_C"] = str(config['physics'].get('default_surface_temp_c', 25.0))
            my_measurement.info["Pressure_hPa"] = str(config['physics'].get('default_surface_pressure_hpa', 940.0))

        # Save Standard SCC
        ensure_directories(out_dir)
        my_measurement.save_as_SCC_netcdf(netcdf_path)
        
        # Inject MILGRAU Custom Global Attributes
        append_attributes_to_nc(netcdf_path, weather_data, my_measurement, logger)
                
        return f"[OK] NetCDF successfully saved: {year_str}/{month_str}/{save_id}.nc"
        
    except Exception as e:
        return f"[ERROR] Fatal error converting {save_id}.\n{traceback.format_exc()}"

if __name__ == "__main__":
    config = load_config()
    logger = setup_logger("LIBIDS", config['directories']['log_dir'])
    logger.info("=== Starting LIBIDS processing (Level 0) ===")
    
    raw_dir = os.path.join(os.getcwd(), config['directories']['raw_data'])
    file_paths, file_types = scan_raw_files(raw_dir, logger)
    if not file_paths: exit()

    results = [read_licel_header(f) for f in file_paths]
    start_times_utc, stop_times, durations, nshots_list, laser_freqs = zip(*results)

    df_raw = pd.DataFrame({
        "filepath": file_paths, "meas_type": file_types, "start_time_utc": start_times_utc, 
        "stop_time": stop_times, "nshots": nshots_list, "duration": durations, "laser_freq": laser_freqs,
    }).dropna(subset=['start_time_utc'])

    df_raw['start_time_utc'] = pd.to_datetime(df_raw['start_time_utc']).dt.tz_localize('UTC')
    df_raw['start_time_local'] = df_raw['start_time_utc'].dt.tz_convert('America/Sao_Paulo')
    df_raw['meas_id'] = df_raw['start_time_local'].apply(get_night_date).dt.strftime('%Y%m%d') + df_raw['start_time_local'].apply(classify_period)

    if config['processing']['incremental']:
        netcdf_dir = os.path.join(os.getcwd(), config['directories']['processed_data'])
        existing_mids = [mid for mid in df_raw["meas_id"].unique() if os.path.exists(os.path.join(netcdf_dir, mid[:4], mid[4:6], f"{mid[:8]}sa{mid[8:]}", f"{mid[:8]}sa{mid[8:]}.nc"))]
        df_raw = df_raw[~df_raw["meas_id"].isin(existing_mids)]

    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        exit()

    # Filter bad files and report
    df_good = filter_laser_shots(df_raw, logger)

    if not df_good.empty:
        process_args = [(meas_id, group, config) for meas_id, group in df_good.groupby("meas_id")]
        success_count = sum(1 for args in process_args if "[OK]" in (res := process_single_netcdf(args)) and logger.info(res) is None or logger.error(res))
        logger.info(f"=== Processed {success_count}/{len(process_args)} groups. ===")
    else:
        logger.warning("No data survived quality control.")