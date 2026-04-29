"""
MILGRAU - Level 1: LIdar Pre-ANalysis CORrection Algorithm (LIPANCORA)
Reads Level 0 SCC NetCDF files, applies physical corrections (Deadtime, Dark Current, 
Bin-shift, Sky Background), and dynamically propagates statistical uncertainties.
Enriches the NetCDF with PBL height, Radiosonde data, and Tropopause calculations.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import traceback

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories, fetch_wyoming_radiosonde
from functions.physics_utils import apply_instrumental_corrections, calculate_pbl_height_gradient, calculate_tropopause_heights


def load_and_prepare_level0(nc_path, logger):
    """Stage 1: Ingest NetCDF and standardize coordinates."""
    try:
        ds = xr.open_dataset(nc_path)
        ds.load() 

        # Convert Unix epoch to Datetime64
        time_dt = pd.to_datetime(ds['Raw_Data_Start_Time'].values, unit='s')
        ds = ds.assign_coords(time=time_dt)
        
        # Convert bin index to altitude in meters
        channel_strings = ds['channel_string'].values.astype(str)
        dz = float(ds['Raw_Data_Range_Resolution'].values[0])
        z_arr = np.arange(ds.sizes['points']) * dz
        
        ds = ds.rename({'points': 'altitude', 'channels': 'channel'})
        ds = ds.assign_coords(altitude=z_arr, channel=channel_strings)
        
        logger.info(f"  -> [STAGE 1] Level 0 ingestion successful.")
        return ds, z_arr
    
    except Exception as e:
        logger.error(f"  -> [STAGE 1] Failed to ingest Level 0: {e}")
        raise

def apply_all_physical_corrections(ds, z_arr, config, logger):
    """Stage 2: Execute physics engine for each channel with error propagation."""
    channels_config = config['physics']['channels']
    c_speed = config['physics']['speed_of_light']
    dz = z_arr[1] - z_arr[0]
    bin_time_us = (2 * dz / c_speed) * 1e6
    shots = float(ds.attrs.get('Accumulated_Shots'))
    
    rcs_datasets = []
    logger.info(f"  -> [STAGE 2] Running instrumental corrections")
    
    for ch_idx, ch_name in enumerate(ds.channel.values):
        try:
            sig = ds['Raw_Lidar_Data'].isel(channel=ch_idx)
            bg_mask = (ds['altitude'] >= float(ds['Background_Low'].isel(channel=ch_idx))) & \
                      (ds['altitude'] <= float(ds['Background_High'].isel(channel=ch_idx)))
            
            dt, shift, bg_offset = channels_config.get(ch_name, [0.0, 0, 0.0])
            is_photon = "pc" in ch_name.lower() or "ph" in ch_name.lower()

            # Background Profile (Dark Current) retrieval
            dc_prof, dc_err = None, None
            if "Background_Profile" in ds:
                dc_data = ds["Background_Profile"].isel(channel=ch_idx)
                dc_prof = dc_data.mean(dim="time_bck").rename({'altitude': 'range'})
                dc_err = dc_data.std(dim="time_bck").rename({'altitude': 'range'}) / np.sqrt(ds.sizes.get("time_bck", 1))

            sig_renamed = sig.rename({'altitude': 'range'})
            bg_mask_renamed = bg_mask.rename({'altitude': 'range'})
            z_da = xr.DataArray(z_arr, dims=["range"])
            
            # physics_utils corrections
            _, _, rcs_c, err_rcs_c = apply_instrumental_corrections(
                sig_renamed, z_da, shots, bin_time_us, dt, shift, bg_offset, is_photon, bg_mask_renamed, dc_prof, dc_err
            )
            
            ch_ds = xr.Dataset({
                'Range_Corrected_Signal': rcs_c.rename({'range': 'altitude'}).assign_coords(channel=ch_name).astype(np.float32),
                'Range_Corrected_Signal_Error': err_rcs_c.rename({'range': 'altitude'}).assign_coords(channel=ch_name).astype(np.float32)
            })
            rcs_datasets.append(ch_ds)
        except Exception as e:
            logger.warning(f"  -> [STAGE 2] Channel {ch_name} failed: {e}")

    if not rcs_datasets:
        raise RuntimeError("All channels failed during instrumental correction.")
        
    return xr.concat(rcs_datasets, dim='channel')

def estimate_pbl_timeseries(final_ds, z_arr, logger):
    """Stage 3: Gradient-based PBL height detection per profile."""
    try:
        # Priority: 532nm Analog 
        pbl_channel = next((ch for ch in final_ds.channel.values if "an" in ch.lower() and "532" in ch), final_ds.channel.values[0])
        rcs_matrix = final_ds['Range_Corrected_Signal'].sel(channel=pbl_channel).values
        
        logger.info(f"  -> [STAGE 3] Tracking PBL evolution using {pbl_channel}...")
        pbl_h = [calculate_pbl_height_gradient(rcs_matrix[t, :], z_arr) for t in range(rcs_matrix.shape[0])]
        
        final_ds["PBL_Height_km"] = xr.DataArray(pbl_h, dims=["time"])
        final_ds["PBL_Height_km"].attrs = {"units": "km", "method": "Gradient", "reference_channel": pbl_channel}
        return final_ds
    except Exception as e:
        logger.warning(f"  -> [STAGE 3] PBL tracking failed: {e}")
        return final_ds

def integrate_thermodynamics(final_ds, config, logger):
    """Stage 4: WMO Sounding integration and Tropopause detection."""
    try:
        dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time)//2])
        station_id = config.get('location', {}).get('station_id', '83779')
        
        df_radio = fetch_wyoming_radiosonde(dt_utc, station_id, logger)
        if df_radio is not None and not df_radio.empty:
            final_ds = final_ds.assign_coords(radiosonde_altitude=("radiosonde_altitude", df_radio['height'].values))
            final_ds["Radiosonde_Temperature_K"] = (("radiosonde_altitude",), df_radio['temperature'].values + 273.15)
            
            cpt, lrt = calculate_tropopause_heights(df_radio)
            final_ds.attrs.update({"tropopause_cpt_km": cpt or -999.0, "tropopause_lrt_km": lrt or -999.0})
            logger.info(f"  -> [STAGE 4] Sounding integrated. CPT: {cpt:.2f}km | LRT: {lrt:.2f}km")
        return final_ds
    except Exception as e:
        logger.warning(f"  -> [STAGE 4] Thermodynamics stage incomplete: {e}")
        return final_ds

def process_single_file(args):
    nc_path, config = args
    try:
        stem = Path(nc_path).stem
        save_path = Path(os.getcwd()) / config['directories']['processed_data'] / stem[:4] / stem[4:6] / stem / f"{stem}_level1_rcs.nc"

        logger.info(f"[{stem}] Initializing...")
        
        ds_raw, z_arr = load_and_prepare_level0(nc_path, logger)
        final_ds = apply_all_physical_corrections(ds_raw, z_arr, config, logger)
        final_ds = estimate_pbl_timeseries(final_ds, z_arr, logger)
        final_ds = integrate_thermodynamics(final_ds, config, logger)
        
        # Meta-data persistence
        final_ds.attrs.update(ds_raw.attrs)
        final_ds.attrs["Processing_level"] = "Level 1: PC->MHz, DeadTime, Dark Current, Shift, Background, RCS, Errors, PBL, Tropopause"
        
        # Compressed Export
        ensure_directories(os.path.dirname(save_path))
        final_ds.to_netcdf(save_path, encoding={v: {'zlib': True, 'complevel': 4} for v in final_ds.data_vars})
        
        return f"[OK] {stem} Level 1 generated successfully."
    except Exception:
        return f"[FAILED] {nc_path} execution halted:\n{traceback.format_exc()}"

if __name__ == "__main__":
    conf = load_config()
    logger = setup_logger("LIPANCORA", conf['directories']['log_dir'])
    logger.info("=== Starting MILGRAU LIPANCORA (Level 1) ===")
    
    in_dir = os.path.join(os.getcwd(), conf['directories']['processed_data'])
    files = [f for f in sorted(Path(in_dir).rglob("*.nc")) if "level" not in f.name]
    
    if not files:
        logger.warning("No Level 0 files found for processing.")
    else:
        for f in files: 
            logger.info(process_single_file((str(f), conf)))

    logger.info("=== LIPANCORA finished processing all files. ===")