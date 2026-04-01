"""
MILGRAU Suite - Level 0: LIdar BInary Data Standardized (LIBIDS)
Reads raw Licel binary data, sanitizes spurious files, classifies measurement
periods (UTC to Local Time), and converts valid data into SCC NetCDF format.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import pandas as pd
from statistics import mode
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories, scan_raw_files, read_licel_header
from functions.physics_utils import classify_period, get_night_date

# Import SCC specific libraries
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import (
    msp_netcdf_parameters_system484,
    msp_netcdf_parameters_system565,
)

# ==========================================
# GLOBAL CLASS DEFINITIONS 
# ==========================================
class LidarMeasurement_484(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system484

class LidarMeasurement_565(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system565

# ==========================================
# WORKER FUNCTION (MULTIPROCESSING)
# ==========================================
def process_single_netcdf(args):
    """Worker function to convert a grouped pandas dataframe into an SCC NetCDF."""
    meas_id, group_df, config_dirs = args
    
    date_str = meas_id[:8]
    period = meas_id[8:]
    save_id = f"{date_str}sa{period}"
    year_str, month_str = save_id[:4], save_id[4:6]
    
    out_dir = os.path.join(os.getcwd(), config_dirs['processed_data'], year_str, month_str, save_id)
    netcdf_path = os.path.join(out_dir, f"{save_id}.nc")

    files_meas = group_df[group_df["meas_type"] == "measurements"]["filepath"].tolist()
    files_meas_dc = group_df[group_df["meas_type"] != "measurements"]["filepath"].tolist()

    if not files_meas:
        return f"[FAILED] No measurement files for {save_id}"

    try:
        # Instantiate the correct SCC class based on time of day
        MeasurementClass = LidarMeasurement_565 if period in ["am", "pm"] else LidarMeasurement_484
        my_measurement = MeasurementClass(files_meas)
        
        # Inject Dark Current if it exists
        if files_meas_dc:
            my_measurement.dark_measurement = MeasurementClass(files_meas_dc)
            
        # Hardcoded parameters required by SCC but missing in binary headers
        my_measurement.info["Measurement_ID"] = save_id
        my_measurement.info["Temperature"] = "25"
        my_measurement.info["Pressure"] = "940"
        
        # Determine expected baseline from the mode of the dataset
        duration = mode(group_df["duration"])
        freq = mode(group_df["laser_freq"])
        my_measurement.info["Accumulated_Shots"] = str(int(duration * freq))
        my_measurement.info["Laser_Frequency"] = str(freq)
        my_measurement.info["Measurement_Duration"] = str(duration)

        ensure_directories(out_dir)
        my_measurement.save_as_SCC_netcdf(netcdf_path)
        
        return f"[OK] NetCDF successfully saved: {year_str}/{month_str}/{save_id}.nc"
        
    except Exception as e:
        return f"[ERROR] Fatal error converting {save_id}. Error: {e}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # 1. Load Configuration & Setup Logger
    config = load_config()
    logger = setup_logger("LIBIDS", config['directories']['log_dir'])
    logger.info("=== Starting LIBIDS processing (Level 0) ===")
    
    root_dir = os.getcwd()
    raw_dir = os.path.join(root_dir, config['directories']['raw_data'])
    netcdf_dir = os.path.join(root_dir, config['directories']['processed_data'])
    
    # 2. Scan and Sanitize Input Directory
    file_paths, file_types = scan_raw_files(raw_dir, logger)
    if not file_paths:
        logger.warning(f"No valid files found in {raw_dir}. Exiting.")
        exit()

    # 3. Read Headers (Parallel I/O)
    logger.info(f"Reading headers of {len(file_paths)} files...")
    with ThreadPoolExecutor(max_workers=config['processing']['max_workers_io']) as executor:
        results = list(executor.map(read_licel_header, file_paths))

    start_times_utc, stop_times, durations, nshots_list, laser_freqs = zip(*results)

    df = pd.DataFrame({
        "filepath": file_paths, "meas_type": file_types,
        "start_time_utc": start_times_utc, "stop_time": stop_times,
        "nshots": nshots_list, "duration": durations, "laser_freq": laser_freqs,
    }).dropna(subset=['start_time_utc'])

    if df.empty:
        logger.warning("All headers failed to read. Exiting.")
        exit()

    # 4. Timezone & Atmospheric Period Intelligence
    logger.info("Applying timezone conversions (UTC -> Local) and classifying periods...")
    df['start_time_utc'] = pd.to_datetime(df['start_time_utc']).dt.tz_localize('UTC')
    df['start_time_local'] = df['start_time_utc'].dt.tz_convert('America/Sao_Paulo')
    
    df['flag_period'] = df['start_time_local'].apply(classify_period)
    df['meas_id'] = df['start_time_local'].apply(get_night_date).dt.strftime('%Y%m%d') + df['flag_period']

    # 5. Incremental Filter
    if config['processing']['incremental']:
        logger.info("Applying early incremental filter...")
        
        def needs_processing(meas_id):
            save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
            expected_path = os.path.join(netcdf_dir, save_id[:4], save_id[4:6], f"{save_id}.nc")
            return not os.path.exists(expected_path)

        valid_meas_ids = [mid for mid in df["meas_id"].unique() if needs_processing(mid)]
        skipped = len(df["meas_id"].unique()) - len(valid_meas_ids)
        
        if skipped > 0:
            logger.info(f"[SKIPPED] {skipped} measurement periods already exist as NetCDF.")
            
        df = df[df["meas_id"].isin(valid_meas_ids)]

    if df.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        exit()

    # 6. Quality Control (Laser Shots consistency)
    logger.info("Evaluating laser shots quality and consistency...")
    df_good_list, df_bad_list = [], []

    for meas_id, group in df.groupby("meas_id"):
        try:
            expected_shots = mode(group["duration"]) * mode(group["laser_freq"])
            bad_cond = (group["nshots"] == 0) | (abs(group["nshots"] - expected_shots) >= 2e-3 * expected_shots)
                
            df_bad_list.append(group.loc[bad_cond])
            df_good_list.append(group.loc[~bad_cond])
        except Exception as e:
            logger.warning(f"Error checking file condition in group {meas_id}: {e}")
            df_bad_list.append(group)

    df_good = pd.concat(df_good_list).reset_index(drop=True) if df_good_list else pd.DataFrame()
    bad_files = sum(len(g) for g in df_bad_list)
    total_files = len(df)

    if total_files > 0:
        loss_percent = (bad_files / total_files) * 100
        logger.info(f"Quality Report: {bad_files} bad files rejected ({loss_percent:.2f}% loss).")
        if loss_percent > 10:
            logger.warning("High data loss rate detected (>10%). Check hardware or atmospheric conditions.")

    # 7. NetCDF SCC Conversion (Parallel CPU)
    if not df_good.empty:
        logger.info(f"Starting NetCDF SCC conversion with {config['processing']['max_workers_cpu']} CPU processes...")
        
        # Package arguments for multiprocessing
        process_args = [(meas_id, group, config['directories']) for meas_id, group in df_good.groupby("meas_id")]
        
        with ProcessPoolExecutor(max_workers=config['processing']['max_workers_cpu']) as executor:
            for result in executor.map(process_single_netcdf, process_args):
                if "[OK]" in result:
                    logger.info(result)
                else:
                    logger.error(result)
                
        logger.info("=== LIBIDS processing finished successfully! ===")
    else:
        logger.warning("No data with sufficient quality survived for NetCDF conversion.")