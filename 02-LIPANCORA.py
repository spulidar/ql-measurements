"""
MILGRAU Suite - Level 1: LIdar Pre-ANalysis CORrection Algorithm (LIPANCORA)
Reads Level 0 SCC NetCDF files, applies physical corrections (Deadtime, Dark Current, 
Bin-shift, Sky Background), and dynamically propagates statistical uncertainties 
(Poisson & Background StdDev) to output Level 1 data.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories

# ==========================================
# WORKER FUNCTION (MULTIPROCESSING)
# ==========================================
def process_single_file(args):
    nc_path, config = args
    channels_config = config['physics']['channels']
    incremental = config['processing']['incremental']
    out_dir_base = config['directories']['processed_data']
    c_speed = config['physics']['speed_of_light']
    
    from functions.core_io import setup_logger
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
        ds = xr.open_dataset(nc_path)
        ds.load()

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

        corrected_list, rcs_list = [], []
        err_corrected_list, err_rcs_list = [], []

        logger.info(f"[{stem}] Applying physics corrections and propagating errors for {len(channel_names_scc)} channels...")

        for ch_i, ch_name in enumerate(channel_names_scc):
            sig = raw.isel(channel=ch_i).copy()
            deadtime, shift, bg_offset = channels_config.get(ch_name, [0.0, 0, 0.0])
            is_photon = "ph" in ch_name.lower()

            bg_low = float(bg_low_arr[ch_i]) if bg_low_arr is not None else 29000.0
            bg_high = float(bg_high_arr[ch_i]) if bg_high_arr is not None else 29999.0
            bg_mask = (z_da >= bg_low) & (z_da <= bg_high)

            if is_photon:
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

            corrected_list.append(sig_c.assign_coords(channel=ch_name))
            err_corrected_list.append(err_c.assign_coords(channel=ch_name))
            rcs_list.append(rcs.assign_coords(channel=ch_name))
            err_rcs_list.append(err_rcs.assign_coords(channel=ch_name))

        logger.info(f"[{stem}] Merging tensors and saving Level 1 NetCDF...")
        
        final_corrected = xr.concat(corrected_list, dim="channel").transpose("time", "channel", "range")
        final_err_corrected = xr.concat(err_corrected_list, dim="channel").transpose("time", "channel", "range")
        final_rcs = xr.concat(rcs_list, dim="channel").transpose("time", "channel", "range")
        final_err_rcs = xr.concat(err_rcs_list, dim="channel").transpose("time", "channel", "range")

        attrs_common = dict(ds.attrs)
        attrs_common["processing_level"] = "Level 1: PC->MHz, Deadtime, Dark Current, Bin-Shift, Sky Background, Error Propagation"
        attrs_common["history"] = f"{ds.attrs.get('history', '')}\nProcessed with MILGRAU LIPANCORA on {datetime.now(timezone.utc).isoformat()} UTC"

        coords = {"time": ds["time"], "channel": ("channel", np.array(channel_names_scc)), "range": ("range", np.float32(z))}

        corrected_ds = xr.Dataset(
            {"Corrected_Lidar_Data": (("time", "channel", "range"), final_corrected.values),
             "Corrected_Lidar_Data_Error": (("time", "channel", "range"), final_err_corrected.values)},
            coords=coords, attrs=attrs_common
        )

        rcs_ds = xr.Dataset(
            {"Range_Corrected_Signal": (("time", "channel", "range"), final_rcs.values),
             "Range_Corrected_Signal_Error": (("time", "channel", "range"), final_err_rcs.values)},
            coords=coords, attrs=attrs_common
        )

        for var in corrected_ds.data_vars: corrected_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)
        for var in rcs_ds.data_vars: rcs_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)

        ensure_directories(base_dir)
        corrected_ds.to_netcdf(corrected_path)
        rcs_ds.to_netcdf(rcs_path)

        ds.close()
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
    files = [f for f in all_nc_files if "level" not in f.name]

    if not files:
        logger.warning(f"No Level 0 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    modo = "Incremental" if config['processing']['incremental'] else "Rewriting"
    max_workers = config['processing']['max_workers_cpu']
    logger.info(f"Found {len(files)} Level 0 files. Mode: {modo}. Workers: {max_workers}")

    process_args = [(str(f), config) for f in files]
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_single_file, process_args):
            if "[OK]" in result or "[SKIPPED]" in result:
                logger.info(result)
                success_count += 1
            else:
                logger.error(result)

    if success_count == len(files): logger.info("=== LIPANCORA processing finished successfully for all files! ===")
    elif success_count > 0: logger.warning(f"=== LIPANCORA finished. {success_count}/{len(files)} successful. Check errors. ===")
    else: logger.error("=== LIPANCORA failed completely. No files were processed. ===")