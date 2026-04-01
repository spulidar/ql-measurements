"""
MILGRAU Suite - Level 2: Lidar Elastic Backscatter and Extinction Analysis Routine (LEBEAR)
Reads Level 1 RCS NetCDF files, fetches atmospheric sounding data, glues analog 
and photon-counting signals, calculates Rayleigh molecular profiles, and runs 
vectorized KFS inversion with Monte Carlo error propagation.

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

# Import Physics & Viz Functions
from functions.viz_utils import plot_gluing_qa, plot_molecular_qa, plot_kfs_results
from functions.physics_utils import (
    calculate_molecular_profile, 
    slide_glue_signals, 
    kfs_inversion_monte_carlo, 
    find_optimal_reference_altitude,
    calculate_tropopause_heights,
    calculate_pbl_height_gradient
)


# Suppress expected warnings for math on NaN slices
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# WORKER FUNCTION (MULTIPROCESSING)
# ==========================================
def process_level2_file(args):
    nc_path, config, root_dir = args
    
    # Safely initialize logger for multiprocessing (Windows spawn safe)
    from functions.core_io import setup_logger, ensure_directories, fetch_wyoming_radiosonde
    logger = setup_logger("LEBEAR", config['directories']['log_dir'])
    
    try:
        stem = Path(nc_path).stem.replace('_level1_rcs', '')
        year, month = stem[:4], stem[4:6]
        base_dir = Path(os.getcwd()) / config['directories']['processed_data'] / year / month / stem
        out_dir = Path(os.getcwd()) / config['directories']['processed_data'] / year / month / stem 
        ensure_directories(out_dir)

        level2_path = out_dir / f"{stem}_level2_optical.nc"
        if config['processing']['incremental'] and level2_path.exists():
            return f"[SKIPPED] Level 2 already exists: {stem}"

        logger.info(f"[{stem}] Loading Level 1 RCS data...")
        with xr.open_dataset(nc_path) as ds:
            ds.load()
        
        # 1. Extract Time and Altitude
        alt_m = ds['range'].values if 'range' in ds.coords else ds['altitude_m'].values
        alt_km = alt_m / 1000.0
        bin_m = float(np.mean(np.diff(alt_m)))
        
        try:
            dt_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
            meas_dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except Exception:
            meas_dt = datetime.utcnow()
            
        # 2. Fetch Radiosonde Data
        station_id = config['radiosonde']['station_id']
        df_radio = fetch_wyoming_radiosonde(meas_dt, station_id, logger)
        if df_radio is None and not config['radiosonde']['fallback_to_standard']:
            return f"[FAILED] Radiosonde failed and fallback is disabled for {stem}"
            
        if df_radio is None:
            logger.info(f"[{stem}] Using US Standard Atmosphere 1976 as fallback.")

        channels_present = ds.channel.values
        wavelengths = ["355", "532", "1064"]
        
        results_beta_mean, results_beta_std = [], []
        results_ext_mean, results_ext_std = [], []
        final_channels = []
        results_pbl = []

        logger.info(f"[{stem}] Starting Optical Inversion Pipeline...")

        for wl in wavelengths:
            # Find Analog and PC channels for this wavelength
            ch_an = next((c for c in channels_present if wl in c and "an" in c.lower()), None)
            ch_pc = next((c for c in channels_present if wl in c and "ph" in c.lower()), None)
            
            if not ch_an and not ch_pc:
                continue # Wavelength not present
                
            logger.info(f"[{stem}] Processing {wl} nm...")
            
            # --- A. GLUING (If both channels exist) ---
            rcs_an = ds['Range_Corrected_Signal'].sel(channel=ch_an).mean(dim='time').values if ch_an else None
            rcs_pc = ds['Range_Corrected_Signal'].sel(channel=ch_pc).mean(dim='time').values if ch_pc else None
            
            glued_rcs = None
            if wl == "1064":
                logger.info(f"[{stem}] Bypassing gluing for 1064 nm (Infrared uses Analog-only standard).")
                glued_rcs = rcs_an if rcs_an is not None else rcs_pc
            elif rcs_an is not None and rcs_pc is not None:
                logger.info(f"[{stem}] Gluing Analog and PC for {wl} nm...")
                g_conf = config['inversion']['gluing']
                
                glued_rcs, best_idx, _ = slide_glue_signals(
                    rcs_an, rcs_pc, 
                    g_conf['window_length_bins'], 
                    g_conf['correlation_threshold'], 
                    g_conf.get('search_min_idx', 200),
                    g_conf.get('search_max_idx', 2000)
                )
                
                if best_idx > 0:
                    plot_gluing_qa(alt_km, rcs_an, rcs_pc, glued_rcs, best_idx, g_conf['window_length_bins'], config, wl, ds, root_dir, os.path.join(out_dir,'level2-plots'), stem)
                else:
                    logger.warning(f"[{stem}] Gluing failed for {wl} nm. Falling back to Analog only.")
                    glued_rcs = rcs_an
            else:
                glued_rcs = rcs_an if rcs_an is not None else rcs_pc
            
            # --- A.1 PBL DETECTION ---
            # Estimate PBL height using the glued signal for this specific wavelength
            pbl_km = calculate_pbl_height_gradient(glued_rcs, alt_m)
            if not np.isnan(pbl_km):
                results_pbl.append(pbl_km)
                logger.info(f"  -> [{stem}] PBL gradient detected at {pbl_km:.2f} km for {wl} nm.")

            # Error propagation placeholder (simplified for mean profile)
            if ch_an:
                rcs_err = ds['Range_Corrected_Signal_Error'].sel(channel=ch_an).mean(dim='time').values
            else:
                rcs_err = ds['Range_Corrected_Signal_Error'].sel(channel=ch_pc).mean(dim='time').values

            # --- B. MOLECULAR CALIBRATION ---
            logger.info(f"[{stem}] Calculating Rayleigh Scattering for {wl} nm...")
            beta_mol, alpha_mol = calculate_molecular_profile(alt_m, wl, df_radio)
            
            m_conf = config['inversion']['molecular_fit']
            min_alt_idx = int(m_conf['ref_alt_min_m'] / bin_m)
            max_alt_idx = min(int(m_conf['ref_alt_max_m'] / bin_m), len(glued_rcs)-1)
            
            # Robust scaling: uses the entire safe calibration region (e.g., 5km to 9km)
            calib_rcs = glued_rcs[min_alt_idx:max_alt_idx]
            calib_mol = beta_mol[min_alt_idx:max_alt_idx]
            
            valid_mask = (calib_rcs > 0) & (~np.isnan(calib_rcs))
            if np.sum(valid_mask) > 0:
                scaling_factor = np.nanmean(calib_rcs[valid_mask]) / np.nanmean(calib_mol[valid_mask])
            else:
                logger.warning(f"[{stem}] Invalid signal in calibration region. Fallback to 1.0.")
                scaling_factor = 1.0
                
            simulated_mol = beta_mol * scaling_factor
            
            # Dynamically find the best aerosol-free reference altitude
            # window_size=60 represents a ~450m block assuming 7.5m vertical resolution
            ref_idx = find_optimal_reference_altitude(glued_rcs, simulated_mol, min_alt_idx, max_alt_idx, window_size=60)
            logger.info(f"  -> [{stem}] Optimal KFS reference altitude found at {alt_km[ref_idx]:.2f} km.")
            plot_molecular_qa(alt_km, glued_rcs, simulated_mol, alt_km[min_alt_idx], alt_km[max_alt_idx], config, f"{wl} nm", ds, root_dir, os.path.join(out_dir,'level2-plots'), stem)

            # --- C. KFS MONTE CARLO INVERSION ---
            logger.info(f"[{stem}] Running KFS Monte Carlo Inversion ({config['inversion']['monte_carlo_iterations']} iterations)...")
            lr_aer = float(config['inversion']['lidar_ratios'][wl][month])
            lr_mol = float(m_conf['lidar_ratio_molecular'])
            
            b_mean, b_std, e_mean, e_std = kfs_inversion_monte_carlo(
                glued_rcs, rcs_err, beta_mol, lr_aer, lr_mol, ref_idx, bin_m, 
                iterations=config['inversion']['monte_carlo_iterations']
            )
            
            plot_kfs_results(alt_km, b_mean, b_std, e_mean, e_std, config, f"{wl} nm", os.path.join(out_dir,'level2-plots'), stem, ds, root_dir)
            

            results_beta_mean.append(b_mean)
            results_beta_std.append(b_std)
            results_ext_mean.append(e_mean)
            results_ext_std.append(e_std)
            final_channels.append(f"{wl}nm")

        # --- D. SAVE LEVEL 2 NETCDF ---
        if not final_channels:
            return f"[FAILED] No valid wavelengths processed for {stem}"
            
        logger.info(f"[{stem}] Packaging and saving Level 2 NetCDF...")
        
        out_ds = xr.Dataset(
            {
                "Aerosol_Backscatter": (("channel", "range"), np.array(results_beta_mean)),
                "Aerosol_Backscatter_Error": (("channel", "range"), np.array(results_beta_std)),
                "Aerosol_Extinction": (("channel", "range"), np.array(results_ext_mean)),
                "Aerosol_Extinction_Error": (("channel", "range"), np.array(results_ext_std)),
            },
            coords={
                "channel": final_channels,
                "range": alt_m,
                "altitude_km": ("range", alt_km)
            },
            attrs=ds.attrs
        )
        
        # Inject Radiosonde Data into NetCDF for ultimate reproducibility
        if df_radio is not None:
            out_ds = out_ds.assign_coords(radiosonde_alt=("radiosonde_alt", df_radio['height'].values))
            out_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_alt",), df_radio['pressure'].values)
            out_ds["Radiosonde_Temperature_K"] = (("radiosonde_alt",), df_radio['temperature'].values+273.15)
            
        out_ds.attrs["processing_level"] = "Level 2: Gluing, Rayleigh Molecular, Monte Carlo KFS Inversion"
        
        # ---------------------------------------------------------
        # METADATA: PBL and Tropopause (Global Attributes)
        # ---------------------------------------------------------
        # Calculate the ensemble mean of the PBL heights detected across all wavelengths
        mean_pbl_km = np.nanmean(results_pbl) if results_pbl else np.nan
        out_ds.attrs["pbl_height_km"] = mean_pbl_km
        
        if not np.isnan(mean_pbl_km):
            logger.info(f"  -> [{stem}] Final Ensemble PBL Height: {mean_pbl_km:.2f} km")
        
        # Calculate Tropopause heights based on the fetched radiosonde
        cpt_km, lrt_km = calculate_tropopause_heights(df_radio)
        if not np.isnan(cpt_km):
            logger.info(f"  -> [{stem}] Tropopause Found | CPT: {cpt_km:.2f} km | LRT: {lrt_km:.2f} km")
            
        out_ds.attrs["tropopause_cpt_km"] = cpt_km
        out_ds.attrs["tropopause_lrt_km"] = lrt_km
        
        out_ds.attrs["history"] = f"{ds.attrs.get('history', '')}\nInverted with MILGRAU LEBEAR on {datetime.now(timezone.utc).isoformat()} UTC"

        out_ds.to_netcdf(level2_path)
        ds.close()

        return f"[OK] Level 2 processing complete for {stem}"

    except Exception as e:
        return f"[FAILED] Error processing Level 2 for {Path(nc_path).name}: {e}"

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
    files = sorted(Path(input_dir).rglob("*_rcs.nc"))

    if not files:
        logger.warning(f"No Level 1 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    interactive_qa = config.get('inversion', {}).get('interactive_qa', True)
    modo = "Incremental" if config['processing']['incremental'] else "Rewriting"
    
    logger.info(f"Found {len(files)} Level 1 files. Mode: {modo}. Execução: Sequencial (QA Interactive: {interactive_qa})")

    process_args = [(str(f), config, root_dir) for f in files]
    
    success_count = 0
    # Loop sequencial limpo (Adeus Multiprocessing!)
    for args in process_args:
        result = process_level2_file(args)
        
        if "[OK]" in result or "[SKIPPED]" in result:
            logger.info(result)
            success_count += 1
        else:
            logger.error(result)

    if success_count == len(files): 
        logger.info("=== LEBEAR processing finished successfully for all files! ===")
    elif success_count > 0: 
        logger.warning(f"=== LEBEAR finished. {success_count}/{len(files)} successful. Check errors in log. ===")
    else: 
        logger.error("=== LEBEAR failed completely. No Level 2 files were generated. ===")