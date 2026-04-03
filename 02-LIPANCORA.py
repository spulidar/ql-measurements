"""
MILGRAU Suite - Level 1: LIdar Pre-ANalysis CORrection Algorithm (LIPANCORA)
Reads Level 0 SCC NetCDF files, applies physical corrections (Deadtime, Dark Current, 
Bin-shift, Sky Background), and dynamically propagates statistical uncertainties.
Enriches the NetCDF with PBL height, Radiosonde data, and Tropopause calculations.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import gc
from datetime import datetime, timezone
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories, fetch_wyoming_radiosonde
from functions.physics_utils import calculate_pbl_height_gradient, calculate_tropopause_heights

# ==========================================
# SEQUENTIAL WORKER FUNCTION
# ==========================================
def process_single_file(args):
    nc_path, config = args
    channels_config = config['physics']['channels']
    incremental = config['processing']['incremental']
    out_dir_base = config['directories']['processed_data']
    c_speed = config['physics']['speed_of_light']
    
    # We use a station_id from config or fallback to Campo de Marte (83779)
    station_id = config.get('location', {}).get('station_id', '83779')
    
    logger = setup_logger("LIPANCORA", config['directories']['log_dir'])

    try:
        stem = Path(nc_path).stem
        year, month = stem[:4], stem[4:6]
        base_dir = Path(os.getcwd()) / out_dir_base / year / month / stem

        corrected_path = base_dir / f"{stem}_level1_corrected.nc"
        rcs_path = base_dir / f"{stem}_level1_rcs.nc"

        if incremental and corrected_path.exists() and rcs_path.exists():
            return f"[SKIPPED] Level 1 already exists: {stem}"

        logger.info(f"[{stem}] Reading Level 0 data...")
        
        # Use Context Manager to strictly protect RAM and avoid memory leaks
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

            bg_low_arr = ds["Background_Low"].values if "Background_Low" in ds else None
            bg_high_arr = ds["Background_High"].values if "Background_High" in ds else None

            # --- FETCH RADIOSONDE ---
            # Extract precise measurement date directly from the filename (stem)
            # Expected format: YYYYMMDDsaXX (e.g., 20240606sant)
            date_str = stem[:8]
            period = stem[10:12]
            base_dt = datetime.strptime(date_str, "%Y%m%d")
            
            # Assign a representative UTC hour based on the measurement period
            if period == "am":
                meas_dt = base_dt.replace(hour=12, tzinfo=timezone.utc) # ~09:00 Local
            elif period == "pm":
                meas_dt = base_dt.replace(hour=18, tzinfo=timezone.utc) # ~15:00 Local
            elif period == "nt":
                meas_dt = base_dt.replace(hour=23, tzinfo=timezone.utc) # ~20:00 Local (triggers 00Z next day)
            else:
                meas_dt = base_dt.replace(hour=12, tzinfo=timezone.utc)
            
            logger.info(f"[{stem}] Fetching Wyoming Radiosonde data...")
            df_radio = fetch_wyoming_radiosonde(meas_dt, station_id, logger)

            pbl_list = []

            logger.info(f"[{stem}] Applying physics corrections and propagating errors for {len(channel_names_scc)} channels...")

            n_time = raw.sizes["time"]
            n_chan = len(channel_names_scc)
            n_range = raw.sizes["range"]
            
            final_corrected = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_err_corrected = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_rcs = np.empty((n_time, n_chan, n_range), dtype=np.float32)
            final_err_rcs = np.empty((n_time, n_chan, n_range), dtype=np.float32)

            for ch_i, ch_name in enumerate(channel_names_scc):
                sig = raw.isel(channel=ch_i).copy()
                deadtime, shift, bg_offset = channels_config.get(ch_name, [0.0, 0, 0.0])
                is_photon = "ph" in ch_name.lower()

                bg_low = float(bg_low_arr[ch_i]) if bg_low_arr is not None else 29000.0
                bg_high = float(bg_high_arr[ch_i]) if bg_high_arr is not None else 29999.0
                bg_mask = (z_da >= bg_low) & (z_da <= bg_high)

                is_elastic = any(wl in ch_name for wl in ["355", "532", "1064"])
                
                if not is_photon:
                    if np.nanmax(sig) > 1000: sig = sig / (shots * bin_time_us)
                    N_photons = xr.where(sig * shots * bin_time_us > 0, sig * shots * bin_time_us, 0)
                    err_raw = np.sqrt(N_photons) / (shots * bin_time_us)

                    if deadtime > 0:
                        denom = xr.where(1.0 - (sig * deadtime) <= 1e-6, np.nan, 1.0 - (sig * deadtime))
                        sig_dt = sig / denom
                        err_dt = err_raw / (denom**2) 
                    else:
                        sig_dt, err_dt = sig, err_raw
                else:
                    sig_dt = sig
                    err_dt = xr.ones_like(sig) * sig.where(bg_mask).std(dim="range")

                if "Background_Profile" in ds:
                    dc_data = ds["Background_Profile"].isel(channel=ch_i)
                    if "time_bck" in dc_data.dims:
                        dc_prof = dc_data.mean(dim="time_bck")
                        n_bck = ds.sizes.get("time_bck", 1)
                        dc_err = dc_data.std(dim="time_bck") / np.sqrt(n_bck) if n_bck > 1 else xr.zeros_like(dc_prof)
                    else:
                        dc_prof, dc_err = dc_data, xr.zeros_like(dc_data)
                    
                    sig_dt = sig_dt - dc_prof
                    err_dt = np.sqrt(err_dt**2 + dc_err**2) 

                sig_shift = sig_dt.shift(range=shift, fill_value=np.nan)
                err_shift = err_dt.shift(range=shift, fill_value=np.nan)

                bg_mean = sig_shift.where(bg_mask).mean(dim="range") - bg_offset
                err_bg_mean = sig_shift.where(bg_mask).std(dim="range") / np.sqrt(bg_mask.sum().values)

                sig_c = sig_shift - bg_mean
                err_c = np.sqrt(err_shift**2 + err_bg_mean**2)

                rcs = sig_c * (z_da**2)
                err_rcs = err_c * (z_da**2)
                
                # --- DETECT PBL ON ANALOG CHANNELS ---
                if not is_photon and is_elastic:
                    pbl_km = calculate_pbl_height_gradient(rcs.mean(dim="time").values, z, min_search_m=500.0, max_search_m=4000.0)
                    if not np.isnan(pbl_km):
                        pbl_list.append(pbl_km)
                        logger.info(f"  -> [{stem}] PBL gradient detected at {pbl_km:.2f} km for analog channel {ch_name}.")


                final_corrected[:, ch_i, :] = sig_c.values.astype(np.float32)
                final_err_corrected[:, ch_i, :] = err_c.values.astype(np.float32)
                final_rcs[:, ch_i, :] = rcs.values.astype(np.float32)
                final_err_rcs[:, ch_i, :] = err_rcs.values.astype(np.float32)

                del sig, sig_dt, err_dt, sig_shift, err_shift, sig_c, err_c, rcs, err_rcs
                gc.collect()

            logger.info(f"[{stem}] Merging tensors and saving Level 1 NetCDF...")
            
            del raw
            gc.collect()

            attrs_common = dict(ds.attrs)
            attrs_common["processing_level"] = "Level 1: PC->MHz, DT, DC, Shift, Background, Error Propagation, PBL, Tropopause"
            attrs_common["history"] = f"{ds.attrs.get('history', '')}\nProcessed with MILGRAU LIPANCORA on {datetime.now(timezone.utc).isoformat()} UTC"

            # --- INJECT PBL & TROPOPAUSE METADATA ---
            # Calculate ensemble mean for PBL
            mean_pbl_km = np.nanmean(pbl_list) if pbl_list else np.nan
            if not np.isnan(mean_pbl_km):
                logger.info(f"  -> [{stem}] Final Analog Ensemble PBL Height: {mean_pbl_km:.2f} km")
                attrs_common["pbl_height_km"] = float(mean_pbl_km)
            else:
                attrs_common["pbl_height_km"] = -999.0
                
            # Calculate Tropopause (CPT and LRT)
            try:
                cpt_km, lrt_km = calculate_tropopause_heights(df_radio)
                
                if not np.isnan(cpt_km):
                    logger.info(f"  -> [{stem}] Cold Point Tropopause (CPT) found at {cpt_km:.2f} km")
                    attrs_common["tropopause_cpt_km"] = float(cpt_km)
                else:
                    attrs_common["tropopause_cpt_km"] = -999.0
                    
                if not np.isnan(lrt_km):
                    logger.info(f"  -> [{stem}] WMO Lapse Rate Tropopause (LRT) found at {lrt_km:.2f} km")
                    attrs_common["tropopause_lrt_km"] = float(lrt_km)
                else:
                    attrs_common["tropopause_lrt_km"] = -999.0
            except Exception as e:
                logger.warning(f"  -> [{stem}] Tropopause metadata calculation failed: {e}")
                attrs_common["tropopause_cpt_km"] = -999.0
                attrs_common["tropopause_lrt_km"] = -999.0

            coords = {"time": ds["time"], "channel": ("channel", np.array(channel_names_scc)), "range": ("range", np.float32(z))}

            corrected_ds = xr.Dataset(
                {"Corrected_Lidar_Data": (("time", "channel", "range"), final_corrected),
                 "Corrected_Lidar_Data_Error": (("time", "channel", "range"), final_err_corrected)},
                coords=coords, attrs=attrs_common
            )

            rcs_ds = xr.Dataset(
                {"Range_Corrected_Signal": (("time", "channel", "range"), final_rcs),
                 "Range_Corrected_Signal_Error": (("time", "channel", "range"), final_err_rcs)},
                coords=coords, attrs=attrs_common
            )
            
            # --- INJECT RADIOSONDE INTO RCS DATASET ---
            if df_radio is not None and not df_radio.empty:
                rcs_ds = rcs_ds.assign_coords(radiosonde_alt=("radiosonde_alt", df_radio['height'].values))
                rcs_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_alt",), df_radio['pressure'].values)
                rcs_ds["Radiosonde_Temperature_K"] = (("radiosonde_alt",), df_radio['temperature'].values+273.15)

            for var in corrected_ds.data_vars: corrected_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)
            for var in rcs_ds.data_vars: rcs_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)

            ensure_directories(base_dir)
            corrected_ds.to_netcdf(corrected_path)
            rcs_ds.to_netcdf(rcs_path)

        return f"[OK] Processing complete for {stem}"

    except Exception as e:
        return f"[FAILED] Error processing {Path(nc_path).name}: {e}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    config = load_config()
    logger = setup_logger("LIPANCORA", config['directories']['log_dir'])
    logger.info("=== Starting LIPANCORA processing (Level 1) ===")
    
    input_dir = os.path.join(os.getcwd(), config['directories']['processed_data'])
    all_nc_files = sorted(Path(input_dir).rglob("*.nc"))
    # Match only raw Level 0 files (they don't have 'level' in the filename)
    files = [f for f in all_nc_files if "level" not in f.name]

    if not files:
        logger.warning(f"No Level 0 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    modo = "Incremental" if config['processing']['incremental'] else "Rewriting"
    logger.info(f"Found {len(files)} Level 0 files. Mode: {modo}. Execution: Sequential")

    process_args = [(str(f), config) for f in files]
    
    success_count = 0

    for args in process_args:
        result = process_single_file(args)
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
