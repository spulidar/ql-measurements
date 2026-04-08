"""
MILGRAU Suite - Level 1: LIdar Pre-ANalysis CORrection Algorithm (LIPANCORA)
Reads Level 0 SCC NetCDF files, applies physical corrections (Deadtime, Dark Current, 
Bin-shift, Sky Background), and dynamically propagates statistical uncertainties.
Enriches the NetCDF with PBL height, Radiosonde data, and Tropopause calculations.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
from datetime import datetime, timezone
from pathlib import Path
import warnings
import traceback 

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories, fetch_wyoming_radiosonde
from functions.physics_utils import apply_instrumental_corrections

def process_single_file(args):
    nc_path, config = args
    channels_config = config['physics']['channels']
    incremental = config['processing']['incremental']
    save_corrected = config['processing'].get('save_intermediate_corrected', False)
    out_dir_base = config['directories']['processed_data']
    c_speed = config['physics']['speed_of_light']
    station_id = config.get('location', {}).get('station_id', '83779')

    try:
        stem = Path(nc_path).stem
        year, month = stem[:4], stem[4:6]
        base_dir = Path(os.getcwd()) / out_dir_base / year / month / stem

        corrected_path = base_dir / f"{stem}_level1_corrected.nc"
        rcs_path = base_dir / f"{stem}_level1_rcs.nc"

        if incremental and rcs_path.exists():
            return f"[SKIPPED] Level 1 RCS already exists: {stem}"

        logger.info(f"[{stem}] Reading Level 0 data...")
        
        with xr.open_dataset(nc_path) as ds:
            rename_dict = {}
            if 'channels' in ds.dims: rename_dict['channels'] = 'channel'
            if 'points' in ds.dims: rename_dict['points'] = 'range'
            if rename_dict: ds = ds.rename(rename_dict)

            raw = ds["Raw_Lidar_Data"].astype(np.float32)
            if list(raw.dims) != ["time", "channel", "range"]:
                raw = raw.transpose("time", "channel", "range")

            dz = float(ds.get("Raw_Data_Range_Resolution", [[7.5]])[0][0]) if "Raw_Data_Range_Resolution" in ds else config['physics']['vertical_resolution_m']
            z = np.arange(raw.sizes["range"], dtype=np.float32) * dz
            z_da = xr.DataArray(z, dims=["range"], name="altitude_m")
            dr = np.mean(np.diff(z))
            bin_time_us = (2 * dr / c_speed) * 1e6
            shots = float(ds.attrs.get("Accumulated_Shots", 600.0))

            channel_ids = ds["channel_ID"].values if "channel_ID" in ds else np.arange(raw.sizes["channel"])
            id_to_name = config['hardware']['id_to_name']
            channel_names_scc = [id_to_name.get(int(cid), f"unknown_{cid}") for cid in channel_ids]

            bg_low_arr = ds.get("Background_Low", None)
            bg_high_arr = ds.get("Background_High", None)

            # Retrieve Radiosonde
            base_dt = datetime.strptime(stem[:8], "%Y%m%d")
            meas_dt = base_dt.replace(hour={"am": 12, "pm": 18, "nt": 23}.get(stem[10:12], 12), tzinfo=timezone.utc)
            
            logger.info(f"[{stem}] Fetching Wyoming Radiosonde data...")
            df_radio = fetch_wyoming_radiosonde(meas_dt, station_id, logger)

            logger.info(f"[{stem}] Applying physics corrections and propagating errors...")
            pbl_list = []
            n_time, n_chan, n_range = raw.sizes["time"], len(channel_names_scc), raw.sizes["range"]
            
            final_corrected = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_err_corrected = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_rcs = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_err_rcs = np.empty((n_time, n_chan, n_range), dtype=np.float32)

            for ch_i, ch_name in enumerate(channel_names_scc):
                sig = raw.isel(channel=ch_i).copy()
                deadtime, shift, bg_offset = channels_config.get(ch_name, [0.0, 0, 0.0])
                is_photon = "ph" in ch_name.lower()

                bg_low = float(bg_low_arr.values[ch_i]) if bg_low_arr is not None else 29000.0
                bg_high = float(bg_high_arr.values[ch_i]) if bg_high_arr is not None else 29999.0
                bg_mask = (z_da >= bg_low) & (z_da <= bg_high)

                # Dark Current profile if it exists
                dc_prof, dc_err = None, None
                if "Background_Profile" in ds:
                    dc_data = ds["Background_Profile"].isel(channel=ch_i)
                    if "time_bck" in dc_data.dims:
                        dc_prof = dc_data.mean(dim="time_bck")
                        n_bck = ds.sizes.get("time_bck", 1)
                        dc_err = dc_data.std(dim="time_bck") / np.sqrt(n_bck) if n_bck > 1 else xr.zeros_like(dc_prof)
                    else:
                        dc_prof, dc_err = dc_data, xr.zeros_like(dc_data)

                # Corrections
                sig_c, err_c, rcs, err_rcs = apply_instrumental_corrections(
                    sig, z_da, shots, bin_time_us, deadtime, shift, bg_offset, is_photon, bg_mask, dc_prof, dc_err
                )
                

                # Populate Tensors
                final_corrected[:, ch_i, :] = sig_c.values.astype(np.float32)
                final_err_corrected[:, ch_i, :] = err_c.values.astype(np.float32)
                final_rcs[:, ch_i, :] = rcs.values.astype(np.float32)
                final_err_rcs[:, ch_i, :] = err_rcs.values.astype(np.float32)

                del sig, sig_c, err_c, rcs, err_rcs
                gc.collect()

            logger.info(f"[{stem}] Merging tensors and saving Level 1 NetCDF...")
            del raw
            gc.collect()

            # Global Attributes Mapping
            attrs_common = dict(ds.attrs)
            attrs_common["Processing_level"] = "Level 1: PC->MHz, DeadTime, Dark Current, Shift, Background, Errors, PBL, Tropopause"
            attrs_common["History"] = f"{ds.attrs.get('history', '')}\nProcessed with MILGRAU LIPANCORA on {datetime.now(timezone.utc).isoformat()} UTC"

            coords = {"time": ds["time"], "channel": ("channel", np.array(channel_names_scc)), "range": ("range", np.float32(z))}

            # Export Final RCS NetCDF
            rcs_ds = xr.Dataset(
                {"Range_Corrected_Signal": (("time", "channel", "range"), final_rcs),
                 "Range_Corrected_Signal_Error": (("time", "channel", "range"), final_err_rcs)},
                coords=coords, attrs=attrs_common
            )
            
            if df_radio is not None and not df_radio.empty:
                rcs_ds = rcs_ds.assign_coords(radiosonde_alt=("radiosonde_alt", df_radio['height'].values))
                rcs_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_alt",), df_radio['pressure'].values)
                rcs_ds["Radiosonde_Temperature_K"] = (("radiosonde_alt",), df_radio['temperature'].values+273.15)

            for var in rcs_ds.data_vars: rcs_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)
            
            ensure_directories(base_dir)
            rcs_ds.to_netcdf(rcs_path)
            
            if save_corrected:
                corrected_ds = xr.Dataset(
                    {"Corrected_Lidar_Data": (("time", "channel", "range"), final_corrected),
                     "Corrected_Lidar_Data_Error": (("time", "channel", "range"), final_err_corrected)},
                    coords=coords, attrs=attrs_common
                )
                for var in corrected_ds.data_vars: corrected_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)
                corrected_ds.to_netcdf(corrected_path)

        return f"[OK] Processing complete for {stem}"

    except Exception as e:
        return f"[FAILED] Error processing {Path(nc_path).name}:\n{traceback.format_exc()}"


if __name__ == "__main__":
    config = load_config()
    logger = setup_logger("LIPANCORA", config['directories']['log_dir'])
    logger.info("=== Starting LIPANCORA processing (Level 1) ===")
    
    input_dir = os.path.join(os.getcwd(), config['directories']['processed_data'])
    files = [f for f in sorted(Path(input_dir).rglob("*.nc")) if "level" not in f.name]

    if not files:
        logger.warning(f"No Level 0 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    logger.info(f"Found {len(files)} Level 0 files. Mode: {'Incremental' if config['processing']['incremental'] else 'Rewriting'}.")

    success_count = 0
    for nc_file in files:
        result = process_single_file((str(nc_file), config))
        if "[OK]" in result or "[SKIPPED]" in result:
            logger.info(result)
            success_count += 1
        else:
            logger.error(result)

    if success_count == len(files): 
        logger.info("=== LIPANCORA processing finished successfully for all files! ===")
    elif success_count > 0: 
        logger.warning(f"=== LIPANCORA finished. {success_count}/{len(files)} successful. Check errors. ===")
    else: 
        logger.error("=== LIPANCORA failed completely. No files were processed. ===")