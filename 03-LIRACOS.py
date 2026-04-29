"""
MILGRAU - LIRACOS - LIdar RAnge COrrected Signal: Level 1 Visualization
This script provides tools to handle range corrected signal graphics and RCS maps
(quicklooks). It reads Level 1 NetCDF files, calculates time-averaged profiles,
and plots shaded error bands (1-sigma).

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo
"""

import os
import traceback
import xarray as xr
import gc
from matplotlib import pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from functions.core_io import load_config, setup_logger, ensure_directories
from functions.viz_utils import plot_quicklook, plot_global_mean_rcs

def process_single_nc(args):
    nc_file, config, root_dir = args
    
    try:
        file_name_prefix = os.path.basename(nc_file).replace('_level1_rcs.nc', '')
        base_folder = os.path.dirname(nc_file)
        
        output_folder = os.path.join(base_folder, "quicklooks")
        ensure_directories(output_folder)

        check_file = os.path.join(output_folder, f'GlobalMeanRCS_{file_name_prefix}.webp')
        if config['processing']['incremental'] and os.path.exists(check_file):
            return f"[SKIPPED] Plots already exist for: {file_name_prefix}"

        logger.info(f"[{file_name_prefix}] Loading Level 1 data and preparing axes...")
        
        with xr.open_dataset(nc_file) as ds:
            ds.load()
            
            # Convert Altitude from meters to kilometers for visualization
            ds = ds.assign_coords(altitude=ds['altitude'] / 1000.0)
            ds['altitude'].attrs['units'] = 'km'

            # --- EXTRACT METADATA (PBL & TROPOPAUSE) ---
            pbl_da = ds['PBL_Height_km'] if 'PBL_Height_km' in ds else None
            cpt_km = float(ds.attrs.get('tropopause_cpt_km', -999.0))
            lrt_km = float(ds.attrs.get('tropopause_lrt_km', -999.0))

            channels_to_plot = config['visualization']['channels_to_plot']
            altitude_ranges = config['visualization']['altitude_ranges_km']

            logger.info(f"[{file_name_prefix}] Rendering colormaps and mean profiles for {len(channels_to_plot)} channels...")
            for channel_name in channels_to_plot:
                if channel_name in ds.channel.values:
                    rc_signal = ds['Range_Corrected_Signal'].sel(channel=channel_name)
                    rc_error = ds['Range_Corrected_Signal_Error'].sel(channel=channel_name)

                    for max_altitude in altitude_ranges:
                        sig_slice = rc_signal.sel(altitude=slice(0, max_altitude))
                        err_slice = rc_error.sel(altitude=slice(0, max_altitude))

                        # Call the viz_utils factory injecting the metadata
                        plot_quicklook(
                            sig_slice, err_slice, max_altitude, channel_name, 
                            ds, output_folder, file_name_prefix, config, root_dir,
                            pbl_da=pbl_da, cpt_km=cpt_km, lrt_km=lrt_km
                        )
                        
                        del sig_slice, err_slice
                        plt.close('all')
                        gc.collect()

            logger.info(f"[{file_name_prefix}] Generating Global Mean RCS comparative profile...")
            plot_global_mean_rcs(ds, output_folder, file_name_prefix, config, root_dir)

        plt.close('all')
        gc.collect()

        return f"[OK] All plots generated for {file_name_prefix}"

    except Exception as e:
        error_details = traceback.format_exc()
        return f"[FAILED] Error plotting {os.path.basename(nc_file)}:\n{error_details}"

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

    logger.info(f"Found {len(nc_files)} Level 1 files.")

    for nc_file in nc_files:
        result = process_single_nc((str(nc_file), config, root_dir))
        if "[OK]" in result or "[SKIPPED]" in result:
            logger.info(result)
        else:
            logger.error(result)

    logger.info("=== LIRACOS processing finished! ===")