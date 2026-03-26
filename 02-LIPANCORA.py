"""
LIdar Pre-ANalysis CORrection Algorithm - LIPANCORA
This script reads standardized Level 0 SCC NetCDF files, applies physical
corrections, and dynamically propagates statistical uncertainties (Poisson &
Background StdDev) to output Level 1 data (Corrected Lidar Data and RCS).

Corrections & Error Propagation:
    - Deadtime correction (Derivative propagation)
    - Dark Current subtraction (Quadrature sum)
    - Bin-shift correction (Spatial array shift)
    - Sky Background calculation and subtraction (Quadrature sum)
    - Range corrected signal (Geometric scaling)
    - Real-time Signal-to-Noise Ratio (SNR) Sanity Checks

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

# Suppress expected xarray/numpy warnings for division by zero or all-NaN slices
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"LIPANCORA_run_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LIPANCORA")


# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = False
MAX_WORKERS = 4  # Heavy math operations benefit from multiprocessing

INPUT_DIR = "03-netcdf_data"
OUTPUT_DIR = "05-data_level1"

# Channels deadtime correction (µs), bin-shift and bg-cheating (MHz / mV)
CHANNELS = {
    "00355.o_ph": [0.0020, -2, 0.0005],
    "00355.o_an": [0.0000,  8, 0.0000],
    "00387.o_ph": [0.0000, -2, 0.0000],
    "00387.o_an": [0.0000,  8, 0.0000],
    "00408.o_ph": [0.0000, -2, 0.0000],
    "00408.o_an": [0.0000,  8, 0.0000],
    "00530.o_ph": [0.0000, -2, 0.0015],
    "00530.o_an": [0.0000,  7, 0.0000],
    "00532.o_ph": [0.0035, -3, 0.0015],
    "00532.o_an": [0.0000,  6, 0.0000],
    "01064.o_ph": [0.0000,  0, 0.0000],
    "01064.o_an": [0.0000,  1, 0.0000],
}

ID_TO_NAME = {
    934: "01064.o_an", 935: "01064.o_ph",
    722: "00532.o_an", 1593: "00532.o_an", 716: "00532.o_ph",
    1558: "00530.o_an", 1595: "00530.o_an", 1557: "00530.o_ph",
    737: "00355.o_an", 1594: "00355.o_an", 736: "00355.o_ph",
    749: "00387.o_an", 1596: "00387.o_an", 748: "00387.o_ph",
    1446: "00408.o_an", 1447: "00408.o_ph",
}

def process_single_file(nc_path):
    try:
        stem = Path(nc_path).stem
        year = stem[:4]
        month = stem[4:6]
        base_dir = Path(OUTPUT_DIR) / year / month / stem

        corrected_path = base_dir / f"{stem}_level1_corrected.nc"
        rcs_path = base_dir / f"{stem}_level1_rcs.nc"

        if INCREMENTAL_PROCESSING and corrected_path.exists() and rcs_path.exists():
            logger.info(f"[SKIPPED] Level 1 already exists for: {stem}")
            return

        logger.debug(f"Opening Level 0 file: {stem}")
        ds = xr.open_dataset(nc_path)
        ds.load()

        # Standardize dimension names
        rename_dict = {}
        if 'channels' in ds.dims: rename_dict['channels'] = 'channel'
        if 'points' in ds.dims: rename_dict['points'] = 'range'
        if rename_dict: ds = ds.rename(rename_dict)

        raw = ds["Raw_Lidar_Data"].astype(np.float32)
        if list(raw.dims) != ["time", "channel", "range"]:
            raw = raw.transpose("time", "channel", "range")

        dz = float(ds.get("Raw_Data_Range_Resolution", [[7.5]])[0][0]) if "Raw_Data_Range_Resolution" in ds else 7.5
        z = np.arange(raw.sizes["range"], dtype=np.float32) * dz
        z_da = xr.DataArray(z, dims=["range"], name="altitude_m")
        dr = np.mean(np.diff(z))

        # Hardware & Timing constants
        c = 2.99792458e8
        bin_time_us = (2 * dr / c) * 1e6
        shots = float(ds.attrs.get("Accumulated_Shots", 600.0))

        channel_ids = ds["channel_ID"].values if "channel_ID" in ds else np.arange(raw.sizes["channel"])
        channel_names_scc = [ID_TO_NAME.get(int(cid), f"unknown_{cid}") for cid in channel_ids]

        bg_low_arr = ds["Background_Low"].values if "Background_Low" in ds else None
        bg_high_arr = ds["Background_High"].values if "Background_High" in ds else None

        # Output containers
        corrected_list, rcs_list = [], []
        err_corrected_list, err_rcs_list = [], []

        logger.info(f"Processing physics and propagating errors for {stem}...")

        # ==========================================
        # VECTORIZED PHYSICS & ERROR PROPAGATION
        # ==========================================
        for ch_i, ch_name in enumerate(channel_names_scc):
            sig = raw.isel(channel=ch_i).copy()
            deadtime, shift, bg_offset = CHANNELS.get(ch_name, (0.0, 0, 0.0))
            is_photon = "ph" in ch_name.lower()

            bg_low = float(bg_low_arr[ch_i]) if bg_low_arr is not None else 29000.0
            bg_high = float(bg_high_arr[ch_i]) if bg_high_arr is not None else 29999.0
            bg_mask = (z_da >= bg_low) & (z_da <= bg_high)

            # --- 1. Raw Uncertainty Setup ---
            if is_photon:
                if np.nanmax(sig) > 1000:  # Convert counts to MHz if needed
                    sig = sig / (shots * bin_time_us)

                # Poisson Statistics: N_photons = MHz * shots * bin_time
                N_photons = sig * shots * bin_time_us
                N_photons = xr.where(N_photons > 0, N_photons, 0) # Safety against negative noise
                err_raw = np.sqrt(N_photons) / (shots * bin_time_us)

                # --- 2. Deadtime Correction (PC Only) ---
                if deadtime > 0:
                    denom = 1.0 - (sig * deadtime)
                    denom = xr.where(denom <= 1e-6, np.nan, denom) # Prevent division by zero
                    sig_dt = sig / denom
                    # Derivative of deadtime equation
                    err_dt = err_raw / (denom**2)
                else:
                    sig_dt, err_dt = sig, err_raw
            else:
                # Analog signal - Base error is the std deviation of the electronic noise
                sig_dt = sig
                bg_std = sig.where(bg_mask).std(dim="range")
                err_dt = xr.ones_like(sig) * bg_std # Broadcast across altitude

            # --- 2.5 Dark Current Subtraction ---
            if "Background_Profile" in ds:
                dc_data = ds["Background_Profile"].isel(channel=ch_i)
                if "time_bck" in dc_data.dims:
                    dc_prof = dc_data.mean(dim="time_bck")
                    n_bck = ds.sizes.get("time_bck", 1)
                    dc_err = dc_data.std(dim="time_bck") / np.sqrt(n_bck) if n_bck > 1 else xr.zeros_like(dc_prof)
                else:
                    dc_prof = dc_data
                    dc_err = xr.zeros_like(dc_prof)
                
                sig_dt = sig_dt - dc_prof
                err_dt = np.sqrt(err_dt**2 + dc_err**2)

            # --- 3. Bin Shift ---
            sig_shift = sig_dt.shift(range=shift, fill_value=np.nan)
            err_shift = err_dt.shift(range=shift, fill_value=np.nan)

            # --- 4. Sky Background Subtraction ---
            bg_mean = sig_shift.where(bg_mask).mean(dim="range") - bg_offset
            bg_std_shift = sig_shift.where(bg_mask).std(dim="range")
            N_bg_bins = bg_mask.sum().values

            sig_c = sig_shift - bg_mean

            # Error propagation: Add signal error and background error in quadrature
            err_bg_mean = bg_std_shift / np.sqrt(N_bg_bins)
            err_c = np.sqrt(err_shift**2 + err_bg_mean**2)

            # --- 5. Range Corrected Signal (RCS) ---
            rcs = sig_c * (z_da**2)
            err_rcs = err_c * (z_da**2)

            # --- 6. SANITY CHECKS (SNR) ---
            snr = sig_c / err_c

            # Check 1: Boundary Layer SNR (High expected)
            snr_pbl = snr.where((z_da >= 1000) & (z_da <= 2000)).mean().values
            if not np.isnan(snr_pbl) and snr_pbl < 5:
                logger.warning(f"[{stem} | {ch_name}] Low SNR at PBL ({snr_pbl:.2f}). Error might be overestimated or signal is too weak.")

            # Check 2: Background SNR (Should be ~1 or lower)
            snr_bg_check = snr.where(bg_mask).mean().values
            if not np.isnan(snr_bg_check) and abs(snr_bg_check) > 3:
                logger.warning(f"[{stem} | {ch_name}] High SNR at Background ({snr_bg_check:.2f}). Noise might not be properly characterized.")

            # Check 3: Deadtime explosion (Fixed logic)
            if is_photon and deadtime > 0:
                explosion_count = np.isnan(denom).sum().values
                if explosion_count > 0:
                    logger.debug(f"[{stem} | {ch_name}] Deadtime explosion masked {explosion_count} bins at near-range.")

            # Append to channel lists
            corrected_list.append(sig_c.assign_coords(channel=ch_name))
            err_corrected_list.append(err_c.assign_coords(channel=ch_name))
            rcs_list.append(rcs.assign_coords(channel=ch_name))
            err_rcs_list.append(err_rcs.assign_coords(channel=ch_name))

        # ==========================================
        # MERGE AND NETCDF CONFIGURATION
        # ==========================================
        # Concatenate and force the original dimension order: (time, channel, range)
        final_corrected = xr.concat(corrected_list, dim="channel").transpose("time", "channel", "range")
        final_err_corrected = xr.concat(err_corrected_list, dim="channel").transpose("time", "channel", "range")
        final_rcs = xr.concat(rcs_list, dim="channel").transpose("time", "channel", "range")
        final_err_rcs = xr.concat(err_rcs_list, dim="channel").transpose("time", "channel", "range")

        attrs_common = dict(ds.attrs)

        attrs_common["processing_level"] = "Level 1: PC->MHz, Deadtime, Dark Current, Bin-Shift, Sky Background, Error Propagation"
        attrs_common["history"] = f"{ds.attrs.get('history', '')}\nProcessed with LIPANCORA on {datetime.now(timezone.utc).isoformat()} UTC"

        # Construct final datasets
        coords = {"time": ds["time"], "channel": ("channel", np.array(channel_names_scc)), "range": ("range", np.float32(z))}

        corrected_ds = xr.Dataset(
            {
                "Corrected_Lidar_Data": (("time", "channel", "range"), final_corrected.values),
                "Corrected_Lidar_Data_Error": (("time", "channel", "range"), final_err_corrected.values)
            },
            coords=coords, attrs=attrs_common
        )

        rcs_ds = xr.Dataset(
            {
                "Range_Corrected_Signal": (("time", "channel", "range"), final_rcs.values),
                "Range_Corrected_Signal_Error": (("time", "channel", "range"), final_err_rcs.values)
            },
            coords=coords, attrs=attrs_common
        )

        # Optimize storage types
        for var in corrected_ds.data_vars:
            corrected_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)
        for var in rcs_ds.data_vars:
            rcs_ds[var].encoding.update(dtype="float32", _FillValue=np.nan)

        os.makedirs(base_dir, exist_ok=True)
        corrected_ds.to_netcdf(corrected_path)
        rcs_ds.to_netcdf(rcs_path)

        ds.close()
        logger.info(f"[OK] Level 1 saved successfully: {stem}")
        return True

    except Exception as e:
        logger.error(f"[FAILED] Error processing file {Path(nc_path).name}.", exc_info=True)
        return False

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    logger.info("=== Starting LIPANCORA processing ===")
    files = sorted(Path(INPUT_DIR).rglob("*.nc"))

    if not files:
        logger.warning(f"No raw NetCDF data found in '{INPUT_DIR}'. Exiting.")
    else:
        modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
        logger.info(f"Found {len(files)} Level 0 files. Mode: {modo}. Workers: {MAX_WORKERS}")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # list() collects the True/False returns from all processed files
            results = list(executor.map(process_single_file, files))

        # Evaluate the collective results
        if all(results):
            logger.info("=== LIPANCORA processing finished successfully for all files! ===")
        elif any(results):
            logger.warning("=== LIPANCORA finished, but some files failed. Check the logs. ===")
        else:
            logger.error("=== LIPANCORA failed completely. No files were processed. ===")
