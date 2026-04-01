"""
MILGRAU Suite - Level 1 Visualization (LIRACOS)
Orchestrates the rendering of Lidar Range Corrected Signal (RCS) colormaps 
and mean profiles. It reads Level 1 NetCDF files and offloads the plotting 
tasks to the core visualization utilities.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories
from functions.viz_utils import plot_quicklook, plot_global_mean_rcs

# ==========================================
# WORKER FUNCTION (MULTIPROCESSING)
# ==========================================
def process_single_nc(args):
    nc_file, config, root_dir = args
    from functions.core_io import setup_logger
    logger = setup_logger("LIRACOS", config['directories']['log_dir'])
    
    try:
        file_name_prefix = os.path.basename(nc_file).replace('_level1_rcs.nc', '')
        base_folder = os.path.dirname(nc_file)
        
        output_folder = os.path.join(base_folder, "quicklooks")
        ensure_directories(output_folder)

        check_file = os.path.join(output_folder, f'GlobalMeanRCS_{file_name_prefix}.webp')
        if config['processing']['incremental'] and os.path.exists(check_file):
            return f"[SKIPPED] Plots already exist for: {file_name_prefix}"

        logger.info(f"[{file_name_prefix}] Loading Level 1 data and adjusting spatial/time axes...")
        ds = xr.open_dataset(nc_file)
        ds.load()

        num_bins = ds.sizes['range']
        vert_res = config['physics']['vertical_resolution_m']
        new_altitude_km = np.arange(0, num_bins * vert_res, vert_res) / 1000.0
        
        ds = ds.assign_coords(altitude=("range", new_altitude_km))
        ds = ds.swap_dims({'range': 'altitude'})

        try:
            dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
            dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"
            dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
            dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")
            
            if dt_end < dt_in: dt_end += timedelta(days=1)
            time_array = pd.date_range(start=dt_in, end=dt_end, periods=ds.sizes['time'])
            ds = ds.assign_coords(time=time_array)
        except Exception as e:
            logger.warning(f"[{file_name_prefix}] Could not build precise time array: {e}")

        channels_to_plot = config['visualization']['channels_to_plot']
        altitude_ranges = config['visualization']['altitude_ranges_km']

        logger.info(f"[{file_name_prefix}] Rendering colormaps and mean profiles for {len(channels_to_plot)} channels...")
        for channel_name in channels_to_plot:
            if channel_name in ds.channel.values:
                rc_signal = ds['Range_Corrected_Signal'].sel(channel=channel_name)
                rc_error = ds['Range_Corrected_Signal_Error'].sel(channel=channel_name)

                for max_altitude in altitude_ranges:
                    sig_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
                    err_slice = rc_error.where(rc_error['altitude'] <= max_altitude, drop=True)

                    plot_quicklook(
                        sig_slice, err_slice, max_altitude, channel_name, 
                        ds, output_folder, file_name_prefix, config, root_dir
                    )

        logger.info(f"[{file_name_prefix}] Generating Global Mean RCS comparative profile...")
        plot_global_mean_rcs(ds, output_folder, file_name_prefix, config, root_dir)

        ds.close()
        return f"[OK] All plots generated for {file_name_prefix}"

    except Exception as e:
        return f"[FAILED] Error plotting {os.path.basename(nc_file)}: {e}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    config = load_config()
    logger = setup_logger("LIRACOS", config['directories']['log_dir'])
    logger.info("=== Starting LIRACOS rendering (Visualization) ===")
    
    root_dir = os.getcwd()
    base_data_folder = os.path.join(root_dir, config['directories']['processed_data'])
    nc_files = sorted(Path(base_data_folder).rglob("*_level1_rcs.nc"))

    if not nc_files:
        logger.warning(f"No Level 1 NetCDF data found in '{base_data_folder}'. Exiting.")
        exit()

    modo = "Incremental" if config['processing']['incremental'] else "Rewriting"
    max_workers = config['processing']['max_workers_cpu']
    logger.info(f"Found {len(nc_files)} Level 1 files. Mode: {modo}. Workers: {max_workers}")

    process_args = [(str(f), config, root_dir) for f in nc_files]
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_single_nc, process_args):
            if "[OK]" in result or "[SKIPPED]" in result:
                logger.info(result)
                success_count += 1
            else:
                logger.error(result)

    if success_count == len(nc_files): logger.info("=== LIRACOS processing finished successfully for all files! ===")
    elif success_count > 0: logger.warning(f"=== LIRACOS finished. {success_count}/{len(nc_files)} successful. Check errors. ===")
    else: logger.error("=== LIRACOS failed completely. No plots were generated. ===")