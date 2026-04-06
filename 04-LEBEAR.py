"""
MILGRAU Suite - Level 2: Lidar Elastic Backscatter and Extinction Analysis Routine (LEBEAR)
Performs optical inversion (KFS) using metadata and sounding data retrieved from Level 1.
Optimized for sequential processing, rigorous error propagation (Monte Carlo), 
and strict physical calibration (Molecular & Gluing).

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import gc
import traceback
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import warnings

# Import Physics & Viz Functions
from functions.viz_utils import plot_gluing_qa, plot_molecular_qa, plot_kfs_results
from functions.physics_utils import (
    calculate_molecular_profile, 
    slide_glue_signals, 
    kfs_inversion_monte_carlo, 
    find_optimal_reference_altitude
)

# Suppress expected warnings for math on NaN slices and div by zero during fallback
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# SEQUENTIAL WORKER FUNCTION
# ==========================================
def process_level2_file(args):
    nc_path, config, root_dir = args
    
    try:
        stem = Path(nc_path).stem.replace("_level1_rcs", "")
        year_str, month_str = stem[:4], stem[4:6]
        
        out_dir_base = config['directories']['processed_data']
        base_dir = Path(root_dir) / out_dir_base / year_str / month_str / stem
        
        level2_path = base_dir / f"{stem}_level2_aerosol.nc"
        corrected_path = base_dir / f"{stem}_level1_corrected.nc"

        if config['processing']['incremental'] and level2_path.exists():
            return f"[SKIPPED] Level 2 already exists: {stem}"

        logger.info(f"[{stem}] Initiating Level 2 Optical Inversion pipeline...")
        
        # 1. Load Data (Context Manager ensures safe closure)
        with xr.open_dataset(nc_path) as ds_rcs:
            
            # Determine coordinate names dynamically
            alt_coord = 'altitude' if 'altitude' in ds_rcs.coords else 'range'
            alt_m = ds_rcs[alt_coord].values.astype(np.float64)
            # Standardize internal processing to kilometers to prevent float overflow
            alt_km = alt_m / 1000.0 
            
            # Geometry vector for fallbacks (avoiding division by zero at the ground)
            r_squared = np.where(alt_m > 0, alt_m**2, 1e-6)

            # Extract available channels
            available_channels = ds_rcs.channel.values.tolist()
            
            # Build Atmospheric Profiles (Interpolate sounding to Lidar grid)
            logger.info(f"  -> [{stem}] Interpolating thermodynamic profiles...")
            if "Radiosonde_Temperature_K" in ds_rcs and "Radiosonde_Pressure_hPa" in ds_rcs:
                snd_alt = ds_rcs["radiosonde_alt"].values
                snd_temp = ds_rcs["Radiosonde_Temperature_K"].values
                snd_press = ds_rcs["Radiosonde_Pressure_hPa"].values
                
                temp_profile = np.interp(alt_m, snd_alt, snd_temp)
                press_profile = np.interp(alt_m, snd_alt, snd_press)
            else:
                # Standard Atmosphere Fallback (15C and 1013.25hPa at sea level)
                temp_profile = 288.15 - (6.5 * alt_km)
                press_profile = 1013.25 * ((1 - (0.0065 * alt_m / 288.15)) ** 5.2561)
                logger.warning(f"  -> [{stem}] Radiosonde missing. Applied US Standard Atmosphere.")

            # Prepare Level 2 Output Arrays
            wavelengths_to_invert = config.get('inversion', {}).get('wavelengths', [355, 532, 1064])
            n_alt = len(alt_km)
            n_wl = len(wavelengths_to_invert)
            
            out_beta = np.full((n_wl, n_alt), np.nan, dtype=np.float32)
            out_beta_err = np.full((n_wl, n_alt), np.nan, dtype=np.float32)
            out_alpha = np.full((n_wl, n_alt), np.nan, dtype=np.float32)
            out_alpha_err = np.full((n_wl, n_alt), np.nan, dtype=np.float32)

            # Check if Pure Signal Corrected file exists for gluing
            ds_corr = xr.open_dataset(corrected_path) if corrected_path.exists() else None
            if ds_corr:
                logger.info(f"  -> [{stem}] Found _corrected.nc. Using pure signal for Gluing.")
            else:
                logger.info(f"  -> [{stem}] _corrected.nc not found. Using dynamical geometric fallback (RCS/r²) for Gluing.")

            # 2. Iterate over target inversion wavelengths
            for wl_idx, wl in enumerate(wavelengths_to_invert):
                str_wl = str(wl)
                
                # Identify hardware channels for the current wavelength
                ch_an = next((c for c in available_channels if str_wl in c and "an" in c.lower()), None)
                ch_pc = next((c for c in available_channels if str_wl in c and "ph" in c.lower()), None)
                
                if not ch_an and not ch_pc:
                    logger.warning(f"  -> [{stem}] Missing data for {wl}nm. Skipping.")
                    continue
                
                logger.info(f"[{stem}] Processing {wl}nm Elastic Channel...")

                # ==========================================
                # STEP A: HARDWARE GLUING (Analog + PC)
                # ==========================================
                if ch_an and ch_pc:
                    # Extract Pure Signals
                    if ds_corr:
                        pure_an = ds_corr['Corrected_Lidar_Data'].sel(channel=ch_an).mean(dim='time').values
                        pure_pc = ds_corr['Corrected_Lidar_Data'].sel(channel=ch_pc).mean(dim='time').values
                    else:
                        # FALLBACK: Destruct the geometry to obtain pseudo-pure signals
                        rcs_an = ds_rcs['Range_Corrected_Signal'].sel(channel=ch_an).mean(dim='time').values
                        rcs_pc = ds_rcs['Range_Corrected_Signal'].sel(channel=ch_pc).mean(dim='time').values
                        pure_an = rcs_an / r_squared
                        pure_pc = rcs_pc / r_squared

                    # Perform robust sliding correlation on Pure Signals
                    glued_pure, best_idx, slope, intercept = slide_glue_signals(pure_an, pure_pc, alt_km)
                    
                    # Reconstruct the Glued RCS dynamically
                    target_rcs = glued_pure * r_squared
                    
                    qa_save_dir = base_dir / "qa_plots"
                    plot_gluing_qa(alt_km, pure_an*r_squared, pure_pc*r_squared, target_rcs, best_idx, 150, config, f"{wl}nm", ds_rcs, root_dir, qa_save_dir, stem)

                elif ch_an:
                    target_rcs = ds_rcs['Range_Corrected_Signal'].sel(channel=ch_an).mean(dim='time').values
                    logger.info(f"  -> [{stem}] Only Analog available for {wl}nm. Bypassing gluing.")
                else:
                    target_rcs = ds_rcs['Range_Corrected_Signal'].sel(channel=ch_pc).mean(dim='time').values
                    logger.info(f"  -> [{stem}] Only PC available for {wl}nm. Bypassing gluing.")

                # ==========================================
                # STEP B: MOLECULAR CALIBRATION (Rayleigh)
                # ==========================================
                beta_mol, ext_mol = calculate_molecular_profile(temp_profile, press_profile, wl)
                
                # Find optimal aerosol-free calibration zone (Slope and Variance weighted)
                ref_idx = find_optimal_reference_altitude(target_rcs, beta_mol, alt_km, min_alt=5.0, max_alt=15.0)
                
                simulated_mol = beta_mol * (target_rcs[ref_idx] / beta_mol[ref_idx])
                plot_molecular_qa(alt_km, target_rcs, simulated_mol, alt_km[max(0, ref_idx-25)], alt_km[min(len(alt_km)-1, ref_idx+25)], config, f"{wl}nm", ds_rcs, root_dir, base_dir / "qa_plots", stem)
                
                logger.info(f"  -> [{stem}] Rayleigh calibration point established at {alt_km[ref_idx]:.2f} km.")

                # ==========================================
                # STEP C: OPTICAL INVERSION (KFS Monte Carlo)
                # ==========================================
                # Fetch Monthly Lidar Ratio Configuration dynamically
                try:
                    lr_base = config['inversion']['lidar_ratios'][str_wl][month_str]
                except KeyError:
                    lr_base = 50.0 # Standard fallback for aerosols if config is missing
                    logger.warning(f"  -> [{stem}] Missing LR config for {wl}nm in month {month_str}. Using 50.0 sr.")

                lr_std = config['inversion'].get('monte_carlo_lr_std', 10.0)
                n_mc = config['inversion'].get('monte_carlo_iterations', 100)

                b_mean, b_std, a_mean, a_std = kfs_inversion_monte_carlo(target_rcs, alt_km, beta_mol, lr_base, lr_std, ref_idx, n_mc)
                
                plot_kfs_results(alt_km, b_mean, b_std, a_mean, a_std, config, f"{wl}nm", base_dir / "qa_plots", stem, ds_rcs, root_dir)

                out_beta[wl_idx, :] = b_mean.astype(np.float32)
                out_beta_err[wl_idx, :] = b_std.astype(np.float32)
                out_alpha[wl_idx, :] = a_mean.astype(np.float32)
                out_alpha_err[wl_idx, :] = a_std.astype(np.float32)

            # Clean up intermediate arrays
            if ds_corr: ds_corr.close()
            del r_squared, temp_profile, press_profile
            gc.collect()

            # ==========================================
            # STEP D: NETCDF LEVEL 2 CREATION
            # ==========================================
            logger.info(f"[{stem}] Assembling Level 2 NetCDF structure...")
            
            coords = {
                "wavelength": ("wavelength", np.array(wavelengths_to_invert, dtype=np.int32)),
                "altitude": ("altitude", np.float32(alt_m))
            }
            
            ds_l2 = xr.Dataset(
                {
                    "Aerosol_Backscatter": (("wavelength", "altitude"), out_beta),
                    "Aerosol_Backscatter_Error": (("wavelength", "altitude"), out_beta_err),
                    "Aerosol_Extinction": (("wavelength", "altitude"), out_alpha),
                    "Aerosol_Extinction_Error": (("wavelength", "altitude"), out_alpha_err)
                },
                coords=coords
            )

            # Inherit global attributes and add L2 specific metadata
            ds_l2.attrs = ds_rcs.attrs.copy()
            ds_l2.attrs["processing_level"] = "Level 2: Optical Properties (KFS Inversion, Gluing, Monte Carlo Errors)"
            ds_l2.attrs["history"] = f"{ds_rcs.attrs.get('history', '')}\nProcessed with MILGRAU LEBEAR on {datetime.now(timezone.utc).isoformat()} UTC"
            
            # Assign specific CF-compliant units to variables
            ds_l2["Aerosol_Backscatter"].attrs["units"] = "m^-1 sr^-1"
            ds_l2["Aerosol_Extinction"].attrs["units"] = "m^-1"
            ds_l2["altitude"].attrs["units"] = "m"
            ds_l2["wavelength"].attrs["units"] = "nm"

            for var in ds_l2.data_vars: 
                ds_l2[var].encoding.update(dtype="float32", _FillValue=np.nan)
            
            ds_l2.to_netcdf(level2_path)
            
        return f"[OK] Level 2 processing complete for {stem}"

    except Exception as e:
        error_details = traceback.format_exc()
        return f"[FAILED] Error processing Level 2 for {Path(nc_path).name}:\n{error_details}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    from functions.core_io import load_config, setup_logger
    
    config = load_config()
    logger = setup_logger("LEBEAR", config['directories']['log_dir'])
    logger.info("=== Starting LEBEAR processing (Level 2 Optical Inversion) ===")
    
    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, config['directories']['processed_data'])
    
    files = sorted(Path(input_dir).rglob("*_level1_rcs.nc"))

    if not files:
        logger.warning(f"No Level 1 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    interactive_qa = config.get('processing', {}).get('interactive_qa', True)
    logger.info(f"Found {len(files)} Level 1 files. (Interactive QA: {interactive_qa})")

    success_count = 0
    for f in files:
        result = process_level2_file((str(f), config, root_dir))
        if "[OK]" in result or "[SKIPPED]" in result:
            logger.info(result)
            success_count += 1
        else:
            logger.error(result)
            
    if success_count == len(files):
        logger.info("=== LEBEAR Level 2 finished successfully for all files! ===")
    else:
        logger.warning(f"=== LEBEAR finished with errors. Processed {success_count}/{len(files)} files. ===")
