"""
MILGRAU - Physics & Math Utilities

Reusable mathematical and physical routines used by the MILGRAU processing
levels: time classification, Level 1 instrumental corrections, PBL/tropopause
diagnostics, molecular atmosphere calculations, signal gluing and KFS/Fernald
Monte Carlo inversion.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange


# =============================================================================
# Level 0 / time helpers
# =============================================================================


def classify_period(local_dt: Any) -> str:
    """Classify a local timestamp into MILGRAU acquisition periods."""
    if 6 <= local_dt.hour < 12:
        return "am"
    if 12 <= local_dt.hour < 18:
        return "pm"
    return "nt"


def get_night_date(local_dt: Any) -> Any:
    """Return the effective acquisition date for night-time measurements."""
    if local_dt.hour < 6:
        return local_dt - pd.Timedelta(days=1)
    return local_dt


# =============================================================================
# Level 1 / instrumental corrections
# =============================================================================


def _safe_nanmax_xarray(data: xr.DataArray, default: float = 0.0) -> float:
    """Safely compute a finite maximum from an xarray object."""
    try:
        value = float(data.max(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def apply_instrumental_corrections(
    sig: xr.DataArray,
    z_da: xr.DataArray,
    shots: float,
    bin_time_us: float,
    deadtime: float,
    shift: int,
    bg_offset: float,
    is_photon: bool,
    bg_mask: xr.DataArray,
    dc_prof: xr.DataArray | None = None,
    dc_err: xr.DataArray | None = None,
    deadtime_min_denominator: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Apply Level 1 lidar instrumental corrections with uncertainty propagation.

    The correction sequence is dark-current subtraction, photon-counting MHz
    normalization, non-paralyzable dead-time correction, bin-shift correction,
    sky-background subtraction and range correction.
    """
    if shots is None or not np.isfinite(float(shots)) or float(shots) <= 0.0:
        raise ValueError(f"Invalid laser shots value: {shots}")
    if bin_time_us is None or not np.isfinite(float(bin_time_us)) or float(bin_time_us) <= 0.0:
        raise ValueError(f"Invalid bin_time_us value: {bin_time_us}")

    shots = float(shots)
    bin_time_us = float(bin_time_us)
    deadtime = float(deadtime)
    shift = int(shift)
    bg_offset = float(bg_offset)
    deadtime_min_denominator = float(deadtime_min_denominator)

    sig_dc = sig.copy()
    err_dc = xr.zeros_like(sig)

    if dc_prof is not None:
        sig_dc = sig_dc - dc_prof
        if dc_err is not None:
            err_dc = dc_err

    if not is_photon:
        sig_dt = sig_dc.copy()
        if _safe_nanmax_xarray(sig_dt) > 1000.0:
            sig_dt = sig_dt / (shots * bin_time_us)

        err_bg = sig_dt.where(bg_mask).std(dim="range", skipna=True)
        err_dt = xr.ones_like(sig_dt) * err_bg
        if dc_prof is not None and dc_err is not None:
            err_dt = np.sqrt(err_dt**2 + err_dc**2)

    else:
        sig_mhz = sig_dc.copy()
        if _safe_nanmax_xarray(sig_mhz) > 150.0:
            sig_mhz = sig_mhz / (shots * bin_time_us)

        n_photons = xr.where(sig_mhz * shots * bin_time_us > 0.0, sig_mhz * shots * bin_time_us, 0.0)
        err_raw = np.sqrt(n_photons) / (shots * bin_time_us)
        if dc_prof is not None and dc_err is not None:
            err_raw = np.sqrt(err_raw**2 + err_dc**2)

        if deadtime > 0.0:
            denom = 1.0 - (sig_mhz * deadtime)
            safe_denom = xr.where(denom < deadtime_min_denominator, deadtime_min_denominator, denom)
            sig_dt = sig_mhz / safe_denom
            err_dt = err_raw / (safe_denom**2)
        else:
            sig_dt, err_dt = sig_mhz, err_raw

    max_sig_val = _safe_nanmax_xarray(sig_dt, default=0.0)
    if shift > 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=max_sig_val)
    elif shift < 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=0.0)
    else:
        sig_shift = sig_dt.copy()

    err_shift = err_dt.shift(range=shift, fill_value=0.0)

    bg_mean = sig_shift.where(bg_mask).mean(dim="range", skipna=True) - bg_offset
    n_bg = int(bg_mask.sum().values) if hasattr(bg_mask.sum(), "values") else int(bg_mask.sum())
    n_bg = max(n_bg, 1)
    err_bg_mean = sig_shift.where(bg_mask).std(dim="range", skipna=True) / np.sqrt(n_bg)

    sig_c = sig_shift - bg_mean
    err_c = np.sqrt(err_shift**2 + err_bg_mean**2)

    rcs = sig_c * (z_da**2)
    err_rcs = err_c * (z_da**2)
    return sig_c, err_c, rcs, err_rcs


# =============================================================================
# Level 1 / atmospheric diagnostics
# =============================================================================


def calculate_pbl_height_gradient(
    rcs_signal: np.ndarray,
    alt_m: np.ndarray,
    min_search_m: float = 500.0,
    max_search_m: float = 4000.0,
    smooth_bins: int = 15,
) -> float:
    """Estimate PBL height with the strongest negative RCS gradient method."""
    rcs_signal = np.asarray(rcs_signal, dtype=np.float64)
    alt_m = np.asarray(alt_m, dtype=np.float64)

    if rcs_signal.ndim != 1 or alt_m.ndim != 1 or rcs_signal.size != alt_m.size:
        return np.nan

    smooth_bins = max(int(smooth_bins), 3)
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    valid_idx = np.where(
        (alt_m >= float(min_search_m))
        & (alt_m <= float(max_search_m))
        & np.isfinite(alt_m)
        & np.isfinite(rcs_signal)
    )[0]
    if len(valid_idx) < smooth_bins:
        return np.nan

    search_alt = alt_m[valid_idx]
    search_rcs = rcs_signal[valid_idx]
    finite = np.isfinite(search_rcs)
    if finite.sum() < smooth_bins:
        return np.nan

    median_val = np.nanmedian(search_rcs[finite])
    search_rcs = np.where(np.isfinite(search_rcs), search_rcs, median_val)

    window = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    smoothed_rcs = np.convolve(search_rcs, window, mode="same")
    gradient = np.gradient(smoothed_rcs, search_alt)

    edge_trim = smooth_bins // 2
    if len(gradient) > 2 * edge_trim:
        search_region = gradient[edge_trim:-edge_trim]
        min_grad_idx = int(np.argmin(search_region)) + edge_trim
    else:
        min_grad_idx = int(np.argmin(gradient))

    if not np.isfinite(gradient[min_grad_idx]) or gradient[min_grad_idx] >= 0.0:
        return np.nan

    return float(search_alt[min_grad_idx] / 1000.0)


def calculate_tropopause_heights(df_radiosonde: pd.DataFrame | None) -> tuple[float, float]:
    """Calculate Cold Point and WMO-style Lapse Rate Tropopause heights."""
    if df_radiosonde is None or df_radiosonde.empty:
        return np.nan, np.nan

    required_cols = {"height", "temperature"}
    if not required_cols.issubset(df_radiosonde.columns):
        return np.nan, np.nan

    df = (
        df_radiosonde.dropna(subset=["height", "temperature"])
        .drop_duplicates(subset=["height"], keep="first")
        .sort_values("height")
    )
    if df.empty:
        return np.nan, np.nan

    alt_m = df["height"].to_numpy(dtype=np.float64)
    temp_k = df["temperature"].to_numpy(dtype=np.float64) + 273.15

    valid_idx = np.where((alt_m > 5000.0) & np.isfinite(alt_m) & np.isfinite(temp_k))[0]
    if len(valid_idx) == 0:
        return np.nan, np.nan

    search_alt = alt_m[valid_idx]
    search_temp = temp_k[valid_idx]
    cpt_idx = int(np.argmin(search_temp))
    cpt_km = float(search_alt[cpt_idx] / 1000.0)

    if len(search_alt) < 2 or (search_alt[-1] - search_alt[0]) < 2000.0:
        return cpt_km, np.nan

    z_grid = np.arange(search_alt[0], search_alt[-1], 100.0)
    if z_grid.size < 25:
        return cpt_km, np.nan

    t_grid = np.interp(z_grid, search_alt, search_temp)
    lrt_km = np.nan

    for i in range(len(z_grid) - 1):
        gamma = -(t_grid[i + 1] - t_grid[i]) / 0.1
        if gamma <= 2.0:
            z_i = z_grid[i]
            t_i = t_grid[i]
            window_indices = np.where((z_grid > z_i) & (z_grid <= z_i + 2000.0))[0]
            if len(window_indices) == 0:
                continue

            valid_window = True
            for j in window_indices:
                dz_window_km = (z_grid[j] - z_i) / 1000.0
                if dz_window_km <= 0.0:
                    continue
                gamma_avg = -(t_grid[j] - t_i) / dz_window_km
                if gamma_avg > 2.0:
                    valid_window = False
                    break

            if valid_window:
                lrt_km = float(z_i / 1000.0)
                break

    return cpt_km, lrt_km


# =============================================================================
# Level 2 / molecular atmosphere utilities
# =============================================================================


def get_standard_atmosphere(altitude_array_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple US Standard Atmosphere-like pressure/temperature profile."""
    altitude_array_m = np.asarray(altitude_array_m, dtype=np.float64)
    z = np.maximum(altitude_array_m, 0.0)

    t0 = 288.15
    p0 = 1013.25
    lapse = 0.0065
    gas_constant = 8.3144598
    gravity = 9.80665
    molar_mass = 0.0289644

    temp = t0 - lapse * z
    temp = np.clip(temp, a_min=216.65, a_max=None)
    base = np.maximum(1.0 - (lapse * z) / t0, 1e-6)
    press = p0 * base ** ((gravity * molar_mass) / (gas_constant * lapse))
    return press.astype(np.float64), temp.astype(np.float64)


def calculate_molecular_profile(
    temp_profile: np.ndarray,
    press_profile: np.ndarray,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate molecular backscatter and extinction profiles."""
    temp_profile = np.asarray(temp_profile, dtype=np.float64)
    press_profile = np.asarray(press_profile, dtype=np.float64)
    wavelength_nm = float(wavelength_nm)

    if temp_profile.shape != press_profile.shape:
        raise ValueError("Temperature and pressure profiles must have the same shape.")
    if wavelength_nm <= 0.0 or not np.isfinite(wavelength_nm):
        raise ValueError(f"Invalid wavelength_nm: {wavelength_nm}")

    k_b = 1.380649e-23
    lr_mol = (8.0 * np.pi) / 3.0

    temp_safe = np.where((temp_profile > 0.0) & np.isfinite(temp_profile), temp_profile, np.nan)
    press_safe = np.where((press_profile > 0.0) & np.isfinite(press_profile), press_profile, np.nan)

    press_pa = press_safe * 100.0
    n_density = press_pa / (k_b * temp_safe)
    sigma = 5.45e-28 * ((550.0 / wavelength_nm) ** 4)
    alpha_mol = n_density * sigma
    beta_mol = alpha_mol / lr_mol
    return beta_mol.astype(np.float64), alpha_mol.astype(np.float64)


def _resolve_altitude_search_units(
    altitude: np.ndarray,
    min_alt: float,
    max_alt: float,
    altitude_units: Literal["auto", "m", "km"] = "auto",
) -> tuple[np.ndarray, float, float]:
    """Resolve altitude/search bounds to a common unit for reference fitting."""
    altitude = np.asarray(altitude, dtype=np.float64)
    min_alt = float(min_alt)
    max_alt = float(max_alt)

    if altitude_units in {"m", "km"}:
        return altitude, min_alt, max_alt
    if altitude_units != "auto":
        raise ValueError("altitude_units must be 'auto', 'm', or 'km'.")

    alt_max = np.nanmax(altitude)
    if alt_max > 100.0 and max_alt <= 100.0:
        min_alt *= 1000.0
        max_alt *= 1000.0
    if alt_max <= 100.0 and max_alt > 100.0:
        min_alt /= 1000.0
        max_alt /= 1000.0
    return altitude, min_alt, max_alt


def find_optimal_reference_altitude(
    rcs: np.ndarray,
    beta_mol: np.ndarray,
    altitude: np.ndarray,
    min_alt: float = 5.0,
    max_alt: float = 15.0,
    window_size: int = 50,
    altitude_units: Literal["auto", "m", "km"] = "auto",
) -> int:
    """Find the best Rayleigh calibration altitude window."""
    rcs = np.asarray(rcs, dtype=np.float64)
    beta_mol = np.asarray(beta_mol, dtype=np.float64)
    altitude, min_alt, max_alt = _resolve_altitude_search_units(
        altitude,
        min_alt,
        max_alt,
        altitude_units=altitude_units,
    )

    if rcs.ndim != 1 or beta_mol.ndim != 1 or altitude.ndim != 1:
        raise ValueError("rcs, beta_mol and altitude must be 1D arrays.")
    if not (rcs.size == beta_mol.size == altitude.size):
        raise ValueError("rcs, beta_mol and altitude must have the same length.")

    window_size = max(int(window_size), 3)
    search_mask = (altitude >= min_alt) & (altitude <= max_alt)
    valid_indices = np.where(search_mask)[0]
    if len(valid_indices) < window_size:
        return int(valid_indices[-1]) if len(valid_indices) else int(len(altitude) - 1)

    ratio = rcs / beta_mol
    best_idx = -1
    min_cost = np.inf

    for i in range(len(valid_indices) - window_size + 1):
        start_idx = int(valid_indices[i])
        end_idx = int(start_idx + window_size)
        if end_idx > len(ratio):
            continue

        window_ratio = ratio[start_idx:end_idx]
        window_alt = altitude[start_idx:end_idx]
        valid = np.isfinite(window_ratio) & np.isfinite(window_alt) & (window_ratio > 0.0)
        if valid.sum() < max(3, window_size // 2):
            continue

        wr = window_ratio[valid]
        wa = window_alt[valid]
        mean_ratio = np.mean(wr)
        if mean_ratio <= 0.0 or not np.isfinite(mean_ratio):
            continue

        rel_var = np.var(wr) / (mean_ratio**2)
        slope, _ = np.polyfit(wa, wr, 1)
        rel_slope = abs(slope) / mean_ratio
        cost = rel_var + (rel_slope * 5.0)

        if cost < min_cost:
            min_cost = cost
            best_idx = start_idx + (window_size // 2)

    if best_idx == -1:
        best_idx = int(valid_indices[-1])
    return int(best_idx)


# =============================================================================
# Level 2 / signal gluing
# =============================================================================


@njit(fastmath=True)
def _window_stats(values: np.ndarray, start: int, stop: int):
    """Numba helper returning mean, std, min, max and finite-status."""
    n = 0
    s = 0.0
    mn = 1e308
    mx = -1e308
    for k in range(start, stop):
        v = values[k]
        if not np.isfinite(v):
            return 0.0, 0.0, 0.0, 0.0, False
        s += v
        n += 1
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, False
    mean = s / n
    var = 0.0
    for k in range(start, stop):
        d = values[k] - mean
        var += d * d
    var /= n
    return mean, np.sqrt(var), mn, mx, True


@njit(fastmath=True)
def _slide_glue_core(
    an_vals: np.ndarray,
    pc_vals: np.ndarray,
    window_size: int,
    min_corr: float,
    search_min_idx: int,
    search_max_idx: int,
    intercept_threshold: float,
    gaussian_threshold: float,
    minmax_threshold: float,
):
    """Compiled core for analog/photon-counting gluing window selection."""
    n = len(an_vals)
    best_idx = -1
    best_corr = -1.0
    best_slope = 1.0
    best_intercept = 0.0

    if window_size < 3 or n < window_size:
        return best_idx, best_corr, best_slope, best_intercept

    start_search = max(0, search_min_idx)
    end_search = min(n - window_size, search_max_idx)
    if end_search <= start_search:
        start_search = 0
        end_search = n - window_size

    eps = 1e-12
    for i in range(start_search, end_search + 1):
        an_mean, an_std, an_min, an_max, ok_an = _window_stats(an_vals, i, i + window_size)
        pc_mean, pc_std, pc_min, pc_max, ok_pc = _window_stats(pc_vals, i, i + window_size)
        if not ok_an or not ok_pc:
            continue
        if an_std <= eps or pc_std <= eps:
            continue

        cov = 0.0
        for k in range(i, i + window_size):
            cov += (an_vals[k] - an_mean) * (pc_vals[k] - pc_mean)
        cov /= window_size
        corr = cov / (an_std * pc_std)
        if corr < min_corr:
            continue

        slope = cov / (an_std * an_std)
        intercept = pc_mean - slope * an_mean
        rel_intercept = abs(intercept) / (abs(pc_mean) + eps)
        if rel_intercept > intercept_threshold:
            continue

        residual_var = 0.0
        for k in range(i, i + window_size):
            residual = (slope * an_vals[k] + intercept) - pc_vals[k]
            residual_var += residual * residual
        gaussian_score = np.sqrt(residual_var / window_size) / (pc_std + eps)
        if gaussian_score > gaussian_threshold:
            continue

        scaled_min = slope * an_min + intercept
        scaled_max = slope * an_max + intercept
        if scaled_min > scaled_max:
            tmp = scaled_min
            scaled_min = scaled_max
            scaled_max = tmp
        pc_range = abs(pc_max - pc_min) + eps
        minmax_score = max(abs(scaled_min - pc_min), abs(scaled_max - pc_max)) / pc_range
        if minmax_score > minmax_threshold:
            continue

        if corr > best_corr:
            best_corr = corr
            best_idx = i
            best_slope = slope
            best_intercept = intercept

    return best_idx, best_corr, best_slope, best_intercept


def slide_glue_signals(
    analog_sig: np.ndarray,
    pc_sig: np.ndarray,
    altitude: np.ndarray | None = None,
    window_size: int = 150,
    min_corr: float = 0.90,
    search_min_idx: int = 0,
    search_max_idx: int | None = None,
    intercept_threshold: float = 0.5,
    gaussian_threshold: float = 0.1,
    minmax_threshold: float = 0.5,
    return_diagnostics: bool = False,
):
    """Glue analog and photon-counting signals into one dynamic-range profile."""
    an_vals = np.ascontiguousarray(analog_sig, dtype=np.float64)
    pc_vals = np.ascontiguousarray(pc_sig, dtype=np.float64)
    if an_vals.ndim != 1 or pc_vals.ndim != 1 or an_vals.size != pc_vals.size:
        raise ValueError("analog_sig and pc_sig must be 1D arrays with the same length.")

    n = an_vals.size
    window_size = int(window_size)
    search_min_idx = int(search_min_idx)
    search_max_idx = int(n - window_size if search_max_idx is None else search_max_idx)

    best_idx, best_corr, best_slope, best_intercept = _slide_glue_core(
        an_vals,
        pc_vals,
        window_size,
        float(min_corr),
        search_min_idx,
        search_max_idx,
        float(intercept_threshold),
        float(gaussian_threshold),
        float(minmax_threshold),
    )

    if best_idx == -1:
        glued_sig = pc_vals.copy()
        split_point = -1
    else:
        split_point = int(best_idx + (window_size // 2))
        glued_sig = pc_vals.copy()
        glued_sig[:split_point] = (an_vals[:split_point] * best_slope) + best_intercept

    if return_diagnostics:
        diagnostics = {
            "best_idx": int(best_idx),
            "best_corr": float(best_corr),
            "split_point": int(split_point),
            "search_min_idx": int(search_min_idx),
            "search_max_idx": int(search_max_idx),
            "window_size": int(window_size),
        }
        if altitude is not None and split_point >= 0:
            altitude_arr = np.asarray(altitude, dtype=np.float64)
            if altitude_arr.ndim == 1 and split_point < altitude_arr.size:
                diagnostics["split_altitude"] = float(altitude_arr[split_point])
        return glued_sig, split_point, float(best_slope), float(best_intercept), diagnostics

    return glued_sig, split_point, float(best_slope), float(best_intercept)


# =============================================================================
# Level 2 / KFS-Fernald inversion
# =============================================================================


@njit(parallel=True)
def _kfs_fernald_mc_core(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lr_samples: np.ndarray,
    beta_total_ref_samples: np.ndarray,
    rcs_noise: np.ndarray,
    ref_idx: int,
    lr_mol: float,
    min_lidar_ratio: float,
    use_rcs_noise: bool,
    allow_negative_aerosol: bool,
):
    """Numba-compiled Monte Carlo Fernald/Klett-Sasano backward inversion."""
    n_iter = lr_samples.shape[0]
    n_bins = rcs.shape[0]
    beta_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)
    alpha_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)

    for i in prange(n_iter):
        for j in range(n_bins):
            beta_aer_sims[i, j] = np.nan
            alpha_aer_sims[i, j] = np.nan

        lr_aer = lr_samples[i]
        if not np.isfinite(lr_aer):
            continue
        if lr_aer < min_lidar_ratio:
            lr_aer = min_lidar_ratio

        beta_total_ref = beta_total_ref_samples[i]
        beta_mol_ref = beta_mol[ref_idx]
        if (not np.isfinite(beta_total_ref)) or beta_total_ref <= 0.0:
            beta_total_ref = beta_mol_ref
        if beta_total_ref <= 0.0:
            continue

        x_ref = rcs[ref_idx]
        if use_rcs_noise:
            x_ref = x_ref + rcs_error[ref_idx] * rcs_noise[i, ref_idx]
        if (not np.isfinite(x_ref)) or x_ref <= 0.0:
            continue

        beta_aer_ref = beta_total_ref - beta_mol_ref
        if (not allow_negative_aerosol) and beta_aer_ref < 0.0:
            beta_aer_ref = 0.0

        beta_aer_sims[i, ref_idx] = beta_aer_ref
        alpha_aer_sims[i, ref_idx] = lr_aer * beta_aer_ref

        denom0 = x_ref / beta_total_ref
        mol_int = 0.0
        den_int = 0.0
        y_prev = x_ref

        for j in range(ref_idx - 1, -1, -1):
            dz = altitude_m[j + 1] - altitude_m[j]
            if (not np.isfinite(dz)) or dz <= 0.0:
                break

            x_j = rcs[j]
            if use_rcs_noise:
                x_j = x_j + rcs_error[j] * rcs_noise[i, j]
            if (not np.isfinite(x_j)) or x_j <= 0.0:
                x_j = 1e-30

            bm0 = beta_mol[j]
            bm1 = beta_mol[j + 1]
            if (
                (not np.isfinite(bm0))
                or (not np.isfinite(bm1))
                or bm0 <= 0.0
                or bm1 <= 0.0
            ):
                break

            mol_int += 0.5 * (bm0 + bm1) * dz
            exponent = -2.0 * (lr_aer - lr_mol) * mol_int
            if exponent > 700.0:
                exponent = 700.0
            elif exponent < -700.0:
                exponent = -700.0

            y_j = x_j * np.exp(exponent)
            den_int += 0.5 * (y_j + y_prev) * dz
            denom = denom0 + 2.0 * lr_aer * den_int
            if (not np.isfinite(denom)) or denom <= 0.0:
                break

            beta_total_j = y_j / denom
            beta_aer_j = beta_total_j - bm0
            if (not allow_negative_aerosol) and beta_aer_j < 0.0:
                beta_aer_j = 0.0

            beta_aer_sims[i, j] = beta_aer_j
            alpha_aer_sims[i, j] = lr_aer * beta_aer_j
            y_prev = y_j

    return beta_aer_sims, alpha_aer_sims


def kfs_inversion_monte_carlo(
    rcs: np.ndarray,
    altitude: np.ndarray,
    beta_mol: np.ndarray,
    lr_base: float,
    lr_std: float = 10.0,
    ref_idx: int = -1,
    n_iterations: int = 300,
    rcs_error: np.ndarray | None = None,
    beta_ref_relative_std: float = 0.10,
    aerosol_ref_fraction: float = 0.0,
    altitude_units: Literal["auto", "m", "km"] = "auto",
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = False,
    seed: int | None = None,
    return_diagnostics: bool = False,
):
    """Run KFS/Fernald-Sasano inversion with Monte Carlo uncertainty."""
    rcs_arr = np.ascontiguousarray(rcs, dtype=np.float64)
    beta_mol_arr = np.ascontiguousarray(beta_mol, dtype=np.float64)
    altitude_arr = np.ascontiguousarray(altitude, dtype=np.float64)

    if rcs_arr.ndim != 1:
        raise ValueError("kfs_inversion_monte_carlo expects a 1D RCS profile.")
    if beta_mol_arr.shape[0] != rcs_arr.shape[0]:
        raise ValueError("beta_mol must have the same length as rcs.")
    if altitude_arr.shape[0] != rcs_arr.shape[0]:
        raise ValueError("altitude must have the same length as rcs.")

    if altitude_units == "auto":
        altitude_m = altitude_arr * 1000.0 if np.nanmax(altitude_arr) <= 100.0 else altitude_arr.copy()
    elif altitude_units == "km":
        altitude_m = altitude_arr * 1000.0
    elif altitude_units == "m":
        altitude_m = altitude_arr.copy()
    else:
        raise ValueError("altitude_units must be 'auto', 'm', or 'km'.")

    altitude_m = np.ascontiguousarray(altitude_m, dtype=np.float64)
    n_bins = rcs_arr.shape[0]
    if ref_idx < 0:
        ref_idx = n_bins + int(ref_idx)
    ref_idx = int(ref_idx)
    if ref_idx <= 0 or ref_idx >= n_bins:
        raise ValueError("ref_idx must point to a valid altitude bin above ground.")

    use_rcs_noise = rcs_error is not None
    if use_rcs_noise:
        rcs_error_arr = np.ascontiguousarray(rcs_error, dtype=np.float64)
        if rcs_error_arr.shape[0] != n_bins:
            raise ValueError("rcs_error must have the same length as rcs.")
        rcs_error_arr = np.ascontiguousarray(
            np.where(np.isfinite(rcs_error_arr), rcs_error_arr, 0.0),
            dtype=np.float64,
        )
    else:
        rcs_error_arr = np.zeros_like(rcs_arr, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_iterations = int(n_iterations)
    lr_samples = np.ascontiguousarray(
        rng.normal(float(lr_base), float(lr_std), size=n_iterations),
        dtype=np.float64,
    )

    beta_total_ref_mean = beta_mol_arr[ref_idx] * (1.0 + float(aerosol_ref_fraction))
    beta_total_ref_samples = np.ascontiguousarray(
        rng.normal(
            beta_total_ref_mean,
            abs(beta_total_ref_mean) * float(beta_ref_relative_std),
            size=n_iterations,
        ),
        dtype=np.float64,
    )

    if use_rcs_noise:
        rcs_noise = np.ascontiguousarray(rng.standard_normal((n_iterations, n_bins)), dtype=np.float64)
    else:
        rcs_noise = np.empty((1, 1), dtype=np.float64)

    lr_mol = 8.0 * np.pi / 3.0
    beta_sims, alpha_sims = _kfs_fernald_mc_core(
        rcs_arr,
        rcs_error_arr,
        altitude_m,
        beta_mol_arr,
        lr_samples,
        beta_total_ref_samples,
        rcs_noise,
        ref_idx,
        lr_mol,
        float(min_lidar_ratio),
        bool(use_rcs_noise),
        bool(allow_negative_aerosol),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        beta_mean = np.nanmean(beta_sims, axis=0)
        beta_std = np.nanstd(beta_sims, axis=0)
        alpha_mean = np.nanmean(alpha_sims, axis=0)
        alpha_std = np.nanstd(alpha_sims, axis=0)

    if return_diagnostics:
        diagnostics = {
            "beta_aer_sims": beta_sims,
            "alpha_aer_sims": alpha_sims,
            "lr_samples": lr_samples,
            "beta_total_ref_samples": beta_total_ref_samples,
            "ref_idx": int(ref_idx),
            "altitude_m": altitude_m,
            "used_rcs_noise": bool(use_rcs_noise),
        }
        return beta_mean, beta_std, alpha_mean, alpha_std, diagnostics

    return beta_mean, beta_std, alpha_mean, alpha_std
