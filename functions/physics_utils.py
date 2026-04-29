"""
MILGRAU - Physics & Math Utilities
Contains core mathematical functions for Level 1 (Time, Clouds, Dead-Time,
corrections, IQR background subtraction, Error Propagation)
and Level 2 (Rayleigh Scattering, Signal Gluing, Monte Carlo KFS Inversion).

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumulative_trapezoid

# ==========================================
# Level  1 functions
# ==========================================

def classify_period(local_dt):
    """Classifies measurement period: 'am' (06-11), 'pm' (12-17), 'nt' (18-05)."""
    if 6 <= local_dt.hour < 12: return 'am'
    elif 12 <= local_dt.hour < 18: return 'pm'
    else: return 'nt'

def get_night_date(local_dt):
    """Adjusts the effective date for night measurements past midnight."""
    if local_dt.hour < 6: return local_dt - pd.Timedelta(days=1)
    return local_dt

def apply_instrumental_corrections(sig, z_da, shots, bin_time_us, deadtime, shift, bg_offset, is_photon, bg_mask, dc_prof=None, dc_err=None):
    """
    Applies Level 1 physical instrumental corrections rigorously: 
    Dark Current, Unit Normalization (MHz), Dead-time, Bin-Shift,
    and Sky Background.
    Propagates statistical errors (Poisson/Gaussian) throughout the tensors.
    """
    # -----------------------------------------------------------
    # DARK CURRENT SUBTRACTION (Instrument Noise)
    # -----------------------------------------------------------
    sig_dc = sig.copy()
    err_dc = xr.zeros_like(sig)
    
    if dc_prof is not None:
        sig_dc = sig_dc - dc_prof
        if dc_err is not None:
            err_dc = dc_err

    # -----------------------------------------------------------
    # PHYSICAL ERROR PROPAGATION & UNIT NORMALIZATION
    # -----------------------------------------------------------
    if not is_photon:
        sig_dt = sig_dc.copy()
        if np.nanmax(sig_dt) > 1000: 
            sig_dt = sig_dt / (shots * bin_time_us)
            
        err_bg = sig_dt.where(bg_mask).std(dim="range")
        err_dt = xr.ones_like(sig_dt) * err_bg
        
    else:
        sig_mhz = sig_dc.copy()
        if float(np.nanmax(sig_mhz)) > 150.0: 
            sig_mhz = sig_mhz / (shots * bin_time_us)

        # Poisson statistics for raw photon counting
        N_photons = xr.where(sig_mhz * shots * bin_time_us > 0, sig_mhz * shots * bin_time_us, 0)
        err_raw = np.sqrt(N_photons) / (shots * bin_time_us)

        if deadtime > 0:
            denom = 1.0 - (sig_mhz * deadtime)
            # Saturation cap (5%) prevents negative denominators
            safe_denom = xr.where(denom < 0.05, 0.05, denom)
            sig_dt = sig_mhz / safe_denom
            err_dt = err_raw / (safe_denom**2) 
        else:
            sig_dt, err_dt = sig_mhz, err_raw

    max_sig_val = float(sig_dt.max().values) if float(sig_dt.max().values) > 0 else 0.0

    if shift > 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=max_sig_val)
    elif shift < 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=0.0)
    else:
        sig_shift = sig_dt.copy()

    err_shift = err_dt.shift(range=shift, fill_value=0.0)

    # Sky Background evaluation 
    bg_mean = sig_shift.where(bg_mask).mean(dim="range") - bg_offset
    err_bg_mean = sig_shift.where(bg_mask).std(dim="range") / np.sqrt(bg_mask.sum().values)

    # Final Corrected Signal
    sig_c = sig_shift - bg_mean
    err_c = np.sqrt(err_shift**2 + err_bg_mean**2)

    # Final Range Corrected Signal (RCS)
    rcs = sig_c * (z_da**2)
    err_rcs = err_c * (z_da**2)
    
    return sig_c, err_c, rcs, err_rcs

# ==========================================
# Level 2 functions
# ==========================================

def calculate_pbl_height_gradient(rcs_signal, alt_m, min_search_m=500.0, max_search_m=4000.0, smooth_bins=15):
    """
    Estimates the Planetary Boundary Layer (PBL) height using the Gradient Method.
    The PBL top is identified as the altitude with the strongest negative gradient
    (sharpest drop in aerosol concentration) in the smoothed Range Corrected Signal.
    """
    # Restrict search to typical physical boundary layer altitudes
    valid_idx = np.where((alt_m >= min_search_m) & (alt_m <= max_search_m))[0]
    
    # Ensure we have enough data points in the window to perform smoothing
    if len(valid_idx) < smooth_bins:
        return np.nan
        
    search_alt = alt_m[valid_idx]
    search_rcs = rcs_signal[valid_idx]
    
    # Smooth the signal heavily to prevent noise from creating false gradients
    smoothed_rcs = pd.Series(search_rcs).rolling(window=smooth_bins, center=True, min_periods=1).mean().values
    
    # Calculate the first derivative wrt altitude (dRCS / dz)
    gradient = np.gradient(smoothed_rcs, search_alt)
    
    # Find the strongest negative gradient
    min_grad_idx = np.argmin(gradient)
    
    # Safety check: ensure we actually found a drop-off (negative gradient)
    if gradient[min_grad_idx] >= 0:
        return np.nan 
        
    pbl_m = search_alt[min_grad_idx]
    
    return float(pbl_m / 1000.0)

def calculate_tropopause_heights(df_radiosonde):
    """
    Calculates the Tropopause height (CPT and LRT WMO definitions).
    Uses interpolation to a uniform grid to prevent high-frequency telemetry noise 
    from breaking the WMO lapse rate conditions.
    """
    if df_radiosonde is None or df_radiosonde.empty:
        return np.nan, np.nan
        
    alt_m = df_radiosonde['height'].values
    temp_k = df_radiosonde['temperature'].values+273.15
    
    valid_idx = np.where(alt_m > 5000.0)[0]
    if len(valid_idx) == 0:
        return np.nan, np.nan
        
    search_alt = alt_m[valid_idx]
    search_temp = temp_k[valid_idx]
    
    # Cold Point Tropopause (CPT) - Exact absolute minimum
    cpt_idx = int(np.argmin(search_temp))
    cpt_km = float(search_alt[cpt_idx] / 1000.0)
    
    # Ensure we have enough atmospheric profile to apply the 2km WMO rule
    if len(search_alt) < 2 or (search_alt[-1] - search_alt[0]) < 2000.0:
        return cpt_km, np.nan
        
    # Lapse Rate Tropopause (LRT) - WMO Definition
    # We interpolate to a uniform 100m grid to calculate macroscopic gradients.
    z_grid = np.arange(search_alt[0], search_alt[-1], 100.0)
    t_grid = np.interp(z_grid, search_alt, search_temp)
    
    lrt_km = np.nan
    
    for i in range(len(z_grid) - 1):
        dz_km = 0.1 # 100 meters fixed grid step
        dt_k = t_grid[i+1] - t_grid[i]
        gamma = -dt_k / dz_km
        
        if gamma <= 2.0:
            z_i = z_grid[i]
            t_i = t_grid[i]
            
            # WMO Condition 2: Check all levels within 2 km above
            window_indices = np.where((z_grid > z_i) & (z_grid <= z_i + 2000.0))[0]
            
            if len(window_indices) > 0:
                valid_window = True
                
                for j in window_indices:
                    dz_window_km = (z_grid[j] - z_i) / 1000.0
                    dt_window_k = t_grid[j] - t_i
                    gamma_avg = -dt_window_k / dz_window_km
                    
                    if gamma_avg > 2.0:
                        valid_window = False
                        break # Failed WMO Condition 2, move to next altitude
                        
                if valid_window:
                    lrt_km = float(z_i / 1000.0)
                    break # First level to satisfy all conditions is the LRT
                    
    return cpt_km, lrt_km

def get_standard_atmosphere(altitude_array_m):
    """Generates US Standard Atmosphere 1976 Temperature and Pressure profiles."""
    T0, P0 = 288.15, 1013.25
    L = 0.0065 # Lapse rate (K/m)
    R = 8.3144598
    g = 9.80665
    M = 0.0289644
    
    # Temperature (K)
    temp = T0 - L * altitude_array_m
    temp = np.clip(temp, a_min=216.65, a_max=None) # Isothermal stratosphere approximation
    
    # Pressure (hPa)
    press = P0 * (1 - (L * altitude_array_m) / T0) ** ((g * M) / (R * L))
    
    return press, temp

def calculate_molecular_profile(temp_profile, press_profile, wavelength_nm):
    """
    Calculates the molecular backscatter and extinction profiles based on 
    thermodynamic profiles using the Ideal Gas Law and Rayleigh scattering.
    """
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    
    # Convert pressure from hPa to Pascal (N/m^2)
    press_pa = press_profile * 100.0
    
    # Atmospheric number density (molecules / m^3)
    n_density = press_pa / (k_B * temp_profile)
    
    # Rayleigh scattering cross-section approximation (m^2)
    # Using the standard empirical formula for lidar
    sigma = 5.45e-28 * ((550.0 / wavelength_nm) ** 4)
    
    # Molecular Extinction (alpha) [m^-1]
    alpha_mol = n_density * sigma
    
    # Molecular Lidar Ratio for pure Rayleigh scattering (8 * pi / 3 sr)
    lr_mol = (8.0 * np.pi) / 3.0
    
    # Molecular Backscatter (beta) [m^-1 sr^-1]
    beta_mol = alpha_mol / lr_mol
    
    return beta_mol, alpha_mol

def find_optimal_reference_altitude(rcs, beta_mol, altitude, min_alt=5.0, max_alt=15.0, window_size=50):
    """
    Finds the optimal Rayleigh calibration altitude by locating the region 
    where the measured RCS is most parallel to the simulated molecular profile.
    
    Evaluates both the variance and the linear slope of the RCS/Beta_mol ratio 
    to ensure true parallelism, avoiding falsely low-variance regions that are tilted.
    """
    # Restrict search to the defined upper-troposphere/lower-stratosphere region
    search_mask = (altitude >= min_alt) & (altitude <= max_alt)
    valid_indices = np.where(search_mask)[0]
    
    if len(valid_indices) < window_size:
        # Fallback to the highest valid altitude if the search window is too small
        return len(altitude) - 1 

    best_idx = -1
    min_cost = np.inf
    
    # Compute the ratio profile (must be constant in aerosol-free regions)
    ratio = rcs / beta_mol
    
    search_end = len(valid_indices) - window_size
    
    for i in range(search_end):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + window_size]
        
        window_ratio = ratio[start_idx:end_idx]
        window_alt = altitude[start_idx:end_idx]
        
        # Skip invalid data windows (NaNs, infs, or zero/negative signals)
        if np.any(np.isnan(window_ratio)) or np.any(np.isinf(window_ratio)) or np.any(window_ratio <= 0):
            continue
            
        mean_ratio = np.mean(window_ratio)
        var_ratio = np.var(window_ratio)
        rel_var = var_ratio / (mean_ratio**2) if mean_ratio != 0 else np.inf
        
        # Slope of the ratio
        slope, _ = np.polyfit(window_alt, window_ratio, 1)
        rel_slope = abs(slope) / mean_ratio if mean_ratio != 0 else np.inf
        
        # Combined Cost Function: 
        # Heavily penalize high slopes and moderately penalize variance/noise
        cost = rel_var + (rel_slope * 5.0) 
        
        if cost < min_cost:
            min_cost = cost
            best_idx = start_idx + (window_size // 2)
            
    if best_idx == -1:
        # Fallback if no valid window is found (noisy data or thick clouds)
        # Defaults to the top of the search region
        best_idx = valid_indices[-1]
        
    return best_idx

# ======SIGNAL GLUING (ANALOG + PHOTON)=====

def slide_glue_signals(analog_sig, pc_sig, altitude, window_size=150, min_corr=0.90):
    """
    Performs a sliding window correlation to find the optimal gluing region 
    between Analog and Photon Counting (PC) signals.
    """
    best_idx = -1
    best_corr = -1.0
    best_slope = 1.0
    best_intercept = 0.0
    
    # Ensure standard 1D numpy arrays
    an_vals = np.asarray(analog_sig)
    pc_vals = np.asarray(pc_sig)
    
    search_end = len(an_vals) - window_size
    
    for i in range(search_end):
        an_window = an_vals[i : i + window_size]
        pc_window = pc_vals[i : i + window_size]
        
        # Skip regions with flatlines, saturation limits, or NaNs
        if np.any(np.isnan(an_window)) or np.any(np.isnan(pc_window)): 
            continue
        if np.std(an_window) == 0 or np.std(pc_window) == 0: 
            continue
        
        corr = np.corrcoef(an_window, pc_window)[0, 1]
        
        if corr > best_corr:
            best_corr = corr
            best_idx = i
            
            # Linear regression mapping: PC = slope * Analog + intercept
            slope, intercept = np.polyfit(an_window, pc_window, 1)
            best_slope = slope
            best_intercept = intercept
            
    if best_corr < min_corr:
        # Fallback if the atmospheric condition prevents valid correlation
        return pc_vals, -1, best_slope, best_intercept
        
    # Create the unified glued signal
    glued_sig = np.copy(pc_vals)
    split_point = best_idx + (window_size // 2)
    glued_sig[:split_point] = (an_vals[:split_point] * best_slope) + best_intercept
    
    return glued_sig, split_point, best_slope, best_intercept

# == OPTICAL INVERSION (KFS MONTE CARLO)====

def kfs_inversion_monte_carlo(rcs, altitude, beta_mol, lr_base, lr_std=10.0, ref_idx=-1, n_iterations=100):
    """
    Discrete Fernald (1984) backward integration to retrieve aerosol optical properties.
    Incorporates Monte Carlo perturbations for Lidar Ratio and calibration reference
    to strictly propagate measurement uncertainties.
    
    Includes mathematical singularity protection to prevent integrators from blowing up.
    """
    n_bins = len(rcs)
    dz = np.mean(np.diff(altitude)) * 1000.0  # Spatial resolution strictly in meters
    lr_mol = 8.0 * np.pi / 3.0                # Theoretical Rayleigh Lidar Ratio
    
    beta_aer_sims = np.full((n_iterations, n_bins), np.nan)
    alpha_aer_sims = np.full((n_iterations, n_bins), np.nan)
    
    # Fundamental boundary assumption: Aerosol-free atmosphere at calibration altitude
    beta_total_ref = beta_mol[ref_idx] 
    
    for i in range(n_iterations):
        # 1. Perturb the assumed aerosol Lidar Ratio
        lr_sim = np.random.normal(lr_base, lr_std)
        if lr_sim < 10.0: 
            lr_sim = 10.0  # Physical lower boundary constraint for tropospheric aerosols
            
        # 2. Perturb the molecular calibration reference by 10%
        beta_ref_sim = np.random.normal(beta_total_ref, beta_total_ref * 0.10)
        
        beta_aer = np.full(n_bins, np.nan)
        beta_aer[ref_idx] = max(0.0, beta_ref_sim - beta_mol[ref_idx])
        
        # 3. Iterative Backward Integration 
        for j in range(ref_idx - 1, -1, -1):
            
            # Fernald discrete attenuation term (A_j)
            a_step = (lr_sim - lr_mol) * (beta_mol[j] + beta_mol[j+1]) * dz
            
            # Fernald discrete power term (P_j)
            p_step = rcs[j] * np.exp(a_step)
            
            beta_total_prev = beta_aer[j+1] + beta_mol[j+1]
            
            # Math Domain Protection 1: Signal collapse checks
            if beta_total_prev <= 0 or rcs[j+1] <= 0:
                break 
                
            # Fernald Denominator formulation
            denom = (rcs[j+1] / beta_total_prev) + (lr_sim * (p_step + rcs[j+1]) * dz)
            
            # Math Domain Protection 2: Singularity avoidance
            if denom <= 1e-12:
                break 
                
            beta_aer_step = (p_step / denom) - beta_mol[j]
            
            # Physical constraint: Backscatter cannot be severely negative.
            # Small negative variations are permitted exclusively due to statistical noise.
            if beta_aer_step < -beta_mol[j]: 
                beta_aer_step = -beta_mol[j] * 0.99
                
            beta_aer[j] = beta_aer_step
            
        beta_aer_sims[i, :] = beta_aer
        alpha_aer_sims[i, :] = beta_aer * lr_sim
        
    # Collapse the Monte Carlo matrix into robust statistical vectors
    beta_mean = np.nanmean(beta_aer_sims, axis=0)
    beta_std = np.nanstd(beta_aer_sims, axis=0)
    alpha_mean = np.nanmean(alpha_aer_sims, axis=0)
    alpha_std = np.nanstd(alpha_aer_sims, axis=0)
    
    return beta_mean, beta_std, alpha_mean, alpha_std

