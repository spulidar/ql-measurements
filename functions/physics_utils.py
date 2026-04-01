"""
MILGRAU Suite - Physics & Math Utilities
Contains core mathematical functions for Phase 1 (Time/Clouds) 
and Phase 2 (Rayleigh Scattering, Signal Gluing, Monte Carlo KFS Inversion).

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumulative_trapezoid

# ==========================================
# PHASE 1: TIME & CLOUD MASKS
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

def calculate_dynamic_cloud_threshold(data_array, multiplier=10.0):
    """Dynamic cloud threshold using the Interquartile Range (IQR)."""
    p25 = np.nanpercentile(data_array, 25)
    p75 = np.nanpercentile(data_array, 75)
    return p75 + (multiplier * (p75 - p25))

# ==========================================
# PHASE 2: RAYLEIGH MOLECULAR SCATTERING
# ==========================================

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

def calculate_molecular_profile(altitude_array_m, wavelength_nm, df_radiosonde=None):
    """
    Calculates molecular backscatter (beta) and extinction (alpha) using 
    Bucholtz (1995) theory. Integrates radiosonde data if available.
    """
    # 1. Atmospheric State (Pressure in hPa, Temp in K)
    if df_radiosonde is not None and not df_radiosonde.empty:
        press = np.interp(altitude_array_m, df_radiosonde['height'].values, df_radiosonde['pressure'].values)
        temp = np.interp(altitude_array_m, df_radiosonde['height'].values, df_radiosonde['temperature'].values + 273.15)
    else:
        press, temp = get_standard_atmosphere(altitude_array_m)
        
    # 2. Bucholtz Constants for specific Lidar wavelengths
    # Cross sections in cm^2, converted to m^2
    wl = int(wavelength_nm)
    if wl == 355:
        sigma_ray = 2.923e-27 * 1e-4  
    elif wl == 532:
        sigma_ray = 5.295e-28 * 1e-4
    elif wl == 1064:
        sigma_ray = 3.204e-29 * 1e-4
    else:
        # Generic approximation if wavelength is unusual
        sigma_ray = (3.98e-28 * (550 / wl)**4) * 1e-4
        
    # 3. Number Density (Ideal Gas Law)
    k_B = 1.380649e-23 # Boltzmann constant J/K
    number_density = (press * 100) / (k_B * temp) # molecules / m^3
    
    # 4. Optical Properties
    alpha_mol = number_density * sigma_ray
    lr_mol = 8.0 * np.pi / 3.0
    beta_mol = alpha_mol / lr_mol
    
    return beta_mol, alpha_mol

def find_optimal_reference_altitude(rcs_signal, beta_mol, min_idx, max_idx, window_size=60):
    """
    Finds the most stable, aerosol-free region to be used as the calibration 
    reference for the KFS inversion. It searches for the region where the 
    RCS profile best matches the shape of the molecular profile (lowest variance).
    
    Args:
        rcs_signal (np.array): Range Corrected Signal array.
        beta_mol (np.array): Calculated molecular backscatter array.
        min_idx (int): Minimum altitude index allowed for the search.
        max_idx (int): Maximum altitude index allowed for the search.
        window_size (int): Number of bins to calculate the rolling variance.
                           (e.g., 60 bins * 7.5m = 450 meters search block).
        
    Returns:
        int: The global index of the optimal reference altitude.
    """
    # Ensure we have a valid search space to avoid index crashes
    if (max_idx - min_idx) <= window_size:
        return max_idx 
        
    # Extract the search region allowed by the YAML configuration
    search_rcs = rcs_signal[min_idx:max_idx]
    search_mol = beta_mol[min_idx:max_idx]
    
    # Calculate the ratio (Signal / Molecular)
    # In an aerosol-free zone, this ratio is constant (pure Rayleigh scattering)
    # Adding a small epsilon (1e-12) prevents division by zero warnings
    ratio = search_rcs / (search_mol + 1e-12)
    
    best_idx = max_idx
    min_variance = float('inf')
    
    # Slide the window to find the flattest ratio
    for i in range(len(ratio) - window_size):
        window_ratio = ratio[i : i + window_size]
        
        # Skip windows with NaNs or negative/zero physical values
        if np.any(np.isnan(window_ratio)) or np.any(window_ratio <= 0):
            continue
            
        variance = np.var(window_ratio)
        
        if variance < min_variance:
            min_variance = variance
            # The optimal reference point is the center of the flattest window
            # We add min_idx to map it back to the global array coordinates
            best_idx = min_idx + i + (window_size // 2)
            
    return best_idx

# ==========================================
# PHASE 2: SIGNAL GLUING (ANALOG + PHOTON)
# ==========================================

def slide_glue_signals(lower_sig, upper_sig, window_size, corr_thresh, search_min_idx=200, search_max_idx=2000):
    """
    Robust sliding window to find the best gluing region between Analog and PC.
    Strictly limits the search space to a physical atmospheric window (e.g., 1.5km to 15km).
    """
    best_idx = -1
    best_score = -1.0
    best_multiplier = 1.0
    
    smooth_low = pd.Series(lower_sig).rolling(25, center=True, min_periods=1).mean().values
    smooth_up = pd.Series(upper_sig).rolling(25, center=True, min_periods=1).mean().values
    
    relaxed_corr = corr_thresh * 0.85 
    search_max = min(search_max_idx, len(lower_sig) - window_size)
    
    for i in range(search_min_idx, search_max):
        low_win = smooth_low[i : i+window_size]
        up_win = smooth_up[i : i+window_size]
        
        # Valid data mask to protect correlation math from NaNs
        valid_mask = ~np.isnan(low_win) & ~np.isnan(up_win)
        if np.sum(valid_mask) < (window_size * 0.8): continue # Skip if >20% is NaN
            
        if np.nanmean(up_win[valid_mask]) <= 1e-6 or np.nanmean(low_win[valid_mask]) <= 1e-6: continue
            
        corr = np.corrcoef(low_win[valid_mask], up_win[valid_mask])[0, 1]
        
        if not np.isnan(corr) and corr > relaxed_corr:
            # Calculate scalar on RAW data
            raw_low = lower_sig[i : i+window_size][valid_mask]
            raw_up = upper_sig[i : i+window_size][valid_mask]
            
            multiplier = np.sum(raw_low * raw_up) / max(np.sum(raw_low**2), 1e-12)
            
            if corr > best_score:
                best_score = corr
                best_idx = i
                best_multiplier = multiplier
                
    if best_idx == -1:
        return lower_sig, -1, 1.0 
        
    glued = np.copy(upper_sig)
    glued[:best_idx] = lower_sig[:best_idx] * best_multiplier
    
    fade = np.linspace(1, 0, window_size)
    glued[best_idx:best_idx+window_size] = (
        (lower_sig[best_idx:best_idx+window_size] * best_multiplier * fade) + 
        (upper_sig[best_idx:best_idx+window_size] * (1 - fade))
    )
    
    return glued, best_idx, best_multiplier

# ==========================================
# PHASE 2: KFS MONTE CARLO INVERSION
# ==========================================

def kfs_inversion_monte_carlo(rcs_mean, rcs_error, beta_mol, lr_aerosol, lr_mol, ref_idx, bin_m, iterations=100):
    """
    Fully vectorized Klett-Fernald-Sasano (KFS) backward inversion with Monte Carlo
    error propagation. Resolves N differential equations simultaneously.
    """
    bins = len(rcs_mean)
    
    # 1. Generate Perturbed Signals Matrix (N_iterations, Bins)
    noise = np.random.normal(loc=0.0, scale=1.0, size=(iterations, bins))
    X = rcs_mean + (noise * rcs_error)
    X = np.clip(X, a_min=1e-12, a_max=None) # Prevent negative signal crash
    
    beta_mol_2d = np.tile(beta_mol, (iterations, 1))
    
    # 2. Reference Values at calibration altitude
    # We assume aerosol free at ref_idx, so beta_aer(ref) = 0
    beta_aer_ref = np.zeros(iterations)
    X_ref = np.mean(X[:, ref_idx-5 : ref_idx+5], axis=1) # Smooth reference to avoid noise spikes
    beta_mol_ref = beta_mol[ref_idx]
    
    # 3. Integral components (Trapezoidal rule vectorized backward)
    # T = exp(-2 * integral_R_to_Ref( (Lr_aer - Lr_mol) * beta_mol ))
    int_arg_tau = (lr_aerosol - lr_mol) * beta_mol_2d
    tau_integral = np.zeros_like(X)
    
    # Backward integration from reference
    if ref_idx > 0:
        reversed_arg = int_arg_tau[:, :ref_idx+1][:, ::-1]
        integ_back = cumulative_trapezoid(reversed_arg, dx=bin_m, axis=1, initial=0)
        tau_integral[:, :ref_idx+1] = integ_back[:, ::-1]
        
    tau = np.exp(-2.0 * tau_integral)
    
    # Denominator integral: 2 * Lr_aer * integral_R_to_Ref( X * tau )
    int_arg_den = lr_aerosol * X * tau
    den_integral = np.zeros_like(X)
    
    if ref_idx > 0:
        reversed_den = int_arg_den[:, :ref_idx+1][:, ::-1]
        integ_den_back = cumulative_trapezoid(reversed_den, dx=bin_m, axis=1, initial=0)
        den_integral[:, :ref_idx+1] = integ_den_back[:, ::-1]
        
    # 4. KFS Master Equation
    numerator = X * tau
    denominator = (X_ref / (beta_aer_ref + beta_mol_ref))[:, np.newaxis] + (2.0 * den_integral)
    
    beta_total = numerator / denominator
    beta_aer_matrix = beta_total - beta_mol_2d
    
    # 5. Extract statistics
    beta_aer_mean = np.nanmean(beta_aer_matrix, axis=0)
    beta_aer_std = np.nanstd(beta_aer_matrix, axis=0)
    
    # Cleanup physical impossibilities (negative backscatter)
    beta_aer_mean = np.where(beta_aer_mean < 0, 1e-10, beta_aer_mean)
    
    extinction_mean = beta_aer_mean * lr_aerosol
    extinction_std = beta_aer_std * lr_aerosol
    
    return beta_aer_mean, beta_aer_std, extinction_mean, extinction_std
    

def calculate_pbl_height_gradient(rcs_signal, alt_m, min_search_m=500.0, max_search_m=4000.0, smooth_bins=15):
    """
    Estimates the Planetary Boundary Layer (PBL) height using the Gradient Method.
    The PBL top is identified as the altitude with the strongest negative gradient
    (sharpest drop in aerosol concentration) in the smoothed Range Corrected Signal.
    
    Args:
        rcs_signal (np.array): Range Corrected Signal (glued or single channel).
        alt_m (np.array): Altitude array in meters.
        min_search_m (float): Minimum altitude to search (avoids telescope overlap/surface noise).
        max_search_m (float): Maximum altitude to search (avoids high clouds).
        smooth_bins (int): Window size for smoothing the signal before taking the derivative.
        
    Returns:
        float: Estimated PBL height in kilometers, or np.nan if calculation fails.
    """
    # 1. Restrict search to typical physical boundary layer altitudes
    valid_idx = np.where((alt_m >= min_search_m) & (alt_m <= max_search_m))[0]
    
    # Ensure we have enough data points in the window to perform smoothing
    if len(valid_idx) < smooth_bins:
        return np.nan
        
    search_alt = alt_m[valid_idx]
    search_rcs = rcs_signal[valid_idx]
    
    # 2. Smooth the signal heavily to prevent noise from creating false gradients
    # A rolling mean acts as a low-pass filter, leaving only the macroscopic PBL drop-off
    smoothed_rcs = pd.Series(search_rcs).rolling(window=smooth_bins, center=True, min_periods=1).mean().values
    
    # 3. Calculate the first derivative (gradient) w.r.t altitude (dRCS / dz)
    gradient = np.gradient(smoothed_rcs, search_alt)
    
    # 4. Find the strongest negative gradient
    min_grad_idx = np.argmin(gradient)
    
    # Safety check: ensure we actually found a drop-off (negative gradient)
    if gradient[min_grad_idx] >= 0:
        return np.nan 
        
    pbl_m = search_alt[min_grad_idx]
    
    return float(pbl_m / 1000.0)

def calculate_tropopause_heights(df_radiosonde):
    """
    Calculates the Tropopause height using two distinct atmospheric definitions:
    1. CPT (Cold Point Tropopause): The altitude of the absolute temperature minimum.
    2. LRT (Lapse Rate Tropopause): WMO definition where the lapse rate drops to <= 2 K/km.
    
    Args:
        df_radiosonde (pd.DataFrame): DataFrame containing 'alt' (m) and 'temp' (K).
        
    Returns:
        tuple: (cpt_height_km, lrt_height_km). Returns (np.nan, np.nan) if calculation fails.
    """
    if df_radiosonde is None or df_radiosonde.empty:
        return np.nan, np.nan
        
    alt_m = df_radiosonde['height'].values
    temp_k = df_radiosonde['temperature'].values+273,15
    
    # We restrict the search to altitudes above 5000m (5 km) to strictly avoid 
    # false positives caused by planetary boundary layer (PBL) inversions.
    valid_idx = np.where(alt_m > 5000.0)[0]
    if len(valid_idx) == 0:
        return np.nan, np.nan
        
    search_alt = alt_m[valid_idx]
    search_temp = temp_k[valid_idx]
    
    # ==========================================
    # 1. Cold Point Tropopause (CPT)
    # ==========================================
    cpt_idx = np.argmin(search_temp)
    cpt_km = search_alt[cpt_idx] / 1000.0
    
    # ==========================================
    # 2. Lapse Rate Tropopause (LRT) - WMO Definition
    # ==========================================
    lrt_km = np.nan
    
    for i in range(len(search_alt) - 1):
        # Calculate forward lapse rate (Gamma = -dT/dz) in K/km
        dz_km = (search_alt[i+1] - search_alt[i]) / 1000.0
        dt_k = search_temp[i+1] - search_temp[i]
        
        if dz_km == 0:
            continue
            
        gamma = -dt_k / dz_km
        
        # WMO Condition 1: Lapse rate drops to 2.0 K/km or less
        if gamma <= 2.0:
            z_i = search_alt[i]
            t_i = search_temp[i]
            
            # WMO Condition 2: The average lapse rate between this level and all 
            # higher levels within the next 2 km must not exceed 2.0 K/km.
            window_indices = np.where((search_alt > z_i) & (search_alt <= z_i + 2000.0))[0]
            
            if len(window_indices) > 0:
                valid_window = True
                
                for j in window_indices:
                    dz_window_km = (search_alt[j] - z_i) / 1000.0
                    dt_window_k = search_temp[j] - t_i
                    gamma_avg = -dt_window_k / dz_window_km
                    
                    if gamma_avg > 2.0:
                        valid_window = False
                        break # Fails WMO condition 2, move to the next altitude
                        
                if valid_window:
                    lrt_km = z_i / 1000.0
                    break # First level to satisfy all conditions is the LRT
                    
    return cpt_km, lrt_km