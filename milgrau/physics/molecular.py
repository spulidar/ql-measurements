"""Molecular/Rayleigh calculations for elastic lidar inversion.

The molecular profile is calculated from pressure and temperature using
Bucholtz-style Rayleigh scattering.  The returned molecular backscatter is the
angular volume-scattering coefficient at 180 degrees, appropriate for elastic
backscatter lidar, while molecular extinction is the total Rayleigh volume
scattering coefficient.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

from milgrau.physics.constants import RAYLEIGH_LIDAR_RATIO_SR

_STANDARD_NUMBER_DENSITY_CM3 = 2.54743e19
_STANDARD_PRESSURE_HPA = 1013.25
_STANDARD_TEMPERATURE_K = 288.15


def depolarization_factor(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return the wavelength-dependent molecular depolarization factor.

    The tabulated values follow the Bates values used by Bucholtz for standard
    dry air.  The factor controls both the King correction and the Rayleigh phase
    function, therefore it directly affects molecular backscatter at 180 degrees.
    """
    wavelength_reference_um = np.concatenate(
        (
            np.arange(0.2, 0.231, 0.005),
            np.arange(0.24, 0.401, 0.01),
            np.arange(0.45, 1.01, 0.05),
        )
    )
    wavelength_reference_nm = wavelength_reference_um * 1000.0
    depol_reference = np.array(
        [
            4.545,
            4.384,
            4.221,
            4.113,
            4.004,
            3.895,
            3.785,
            3.675,
            3.565,
            3.455,
            3.4,
            3.289,
            3.233,
            3.178,
            3.178,
            3.122,
            3.066,
            3.066,
            3.01,
            3.01,
            3.01,
            2.955,
            2.955,
            2.955,
            2.899,
            2.842,
            2.842,
            2.786,
            2.786,
            2.786,
            2.786,
            2.73,
            2.73,
            2.73,
            2.73,
            2.73,
        ],
        dtype=np.float64,
    ) * 1e-2
    return np.interp(wavelength_nm, wavelength_reference_nm, depol_reference)


def refractive_index_standard_air(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return refractive index of standard air for the lidar wavelength."""
    wavelength_um = np.asarray(wavelength_nm, dtype=np.float64) * 1e-3
    n_minus_one = np.where(
        wavelength_um > 0.23,
        (5791817.0 / (238.0185 - (1.0 / wavelength_um) ** 2))
        + (167909.0 / (57.362 - (1.0 / wavelength_um) ** 2)),
        8060.51
        + (2480990.0 / (132.274 - (1.0 / wavelength_um) ** 2))
        + (17455.7 / (39.32957 - (1.0 / wavelength_um) ** 2)),
    )
    return n_minus_one * 1e-8 + 1.0


def king_correction_factor(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return the King correction factor for molecular anisotropy."""
    rho_n = depolarization_factor(wavelength_nm)
    return (6.0 + 3.0 * rho_n) / (6.0 - 7.0 * rho_n)


def scattering_cross_section_bucholtz(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return the total Rayleigh cross section per molecule in cm²."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    wavelength_cm = wavelength_nm * 1e-7
    n_s = refractive_index_standard_air(wavelength_nm)
    f_king = king_correction_factor(wavelength_nm)
    numerator = 24.0 * np.pi**3 * (n_s**2 - 1.0) ** 2
    denominator = wavelength_cm**4 * _STANDARD_NUMBER_DENSITY_CM3**2 * (n_s**2 + 2.0) ** 2
    return numerator * f_king / denominator


def rayleigh_phase_function(scattering_angle_rad: float, wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return the Rayleigh phase function for unpolarized light."""
    rho_n = depolarization_factor(wavelength_nm)
    gamma = rho_n / (2.0 - rho_n)
    return 3.0 * ((1.0 + 3.0 * gamma) + (1.0 - gamma) * np.cos(scattering_angle_rad) ** 2) / (4.0 * (1.0 + 2.0 * gamma))


def volume_scattering_coefficient_bucholtz(
    wavelength_nm: float,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    """Return total molecular extinction/Rayleigh volume scattering in m⁻¹."""
    pressure = np.asarray(pressure_hpa, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    sigma_cm2 = scattering_cross_section_bucholtz(float(wavelength_nm))
    beta_cm1 = _STANDARD_NUMBER_DENSITY_CM3 * sigma_cm2 * pressure * _STANDARD_TEMPERATURE_K / (_STANDARD_PRESSURE_HPA * temperature)
    return beta_cm1 * 100.0


def angular_volume_scattering_coefficient_bucholtz(
    wavelength_nm: float,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    scattering_angle_rad: float = np.pi,
) -> np.ndarray:
    """Return molecular angular backscatter coefficient in m⁻¹ sr⁻¹."""
    alpha_mol = volume_scattering_coefficient_bucholtz(wavelength_nm, pressure_hpa, temperature_k)
    phase = rayleigh_phase_function(scattering_angle_rad, float(wavelength_nm))
    return alpha_mol * phase / (4.0 * np.pi)


def calculate_molecular_profile(
    temp_profile: np.ndarray,
    press_profile: np.ndarray,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate molecular backscatter and extinction profiles.

    Pressure is expected in hPa and temperature in K.  Invalid or non-positive
    thermodynamic values are masked as NaN to avoid generating artificial
    molecular structure in the optical inversion.
    """
    temp_profile = np.asarray(temp_profile, dtype=np.float64)
    press_profile = np.asarray(press_profile, dtype=np.float64)
    wavelength_nm = float(wavelength_nm)

    if temp_profile.shape != press_profile.shape:
        raise ValueError("Temperature and pressure profiles must have the same shape.")
    if wavelength_nm <= 0.0 or not np.isfinite(wavelength_nm):
        raise ValueError(f"Invalid wavelength_nm: {wavelength_nm}")

    temp_safe = np.where((temp_profile > 0.0) & np.isfinite(temp_profile), temp_profile, np.nan)
    press_safe = np.where((press_profile > 0.0) & np.isfinite(press_profile), press_profile, np.nan)
    alpha_mol = volume_scattering_coefficient_bucholtz(wavelength_nm, press_safe, temp_safe)
    beta_mol = angular_volume_scattering_coefficient_bucholtz(wavelength_nm, press_safe, temp_safe, np.pi)
    return beta_mol.astype(np.float64), alpha_mol.astype(np.float64)


def calculate_molecular_two_way_transmission(alpha_mol: np.ndarray, altitude_m: np.ndarray) -> np.ndarray:
    """Calculate molecular two-way transmission from extinction."""
    alpha = np.asarray(alpha_mol, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if alpha.ndim != 1 or altitude.ndim != 1 or alpha.size != altitude.size:
        raise ValueError("alpha_mol and altitude_m must be 1D arrays with the same length.")

    valid_alpha = np.where(np.isfinite(alpha) & (alpha >= 0.0), alpha, 0.0)
    optical_depth = cumulative_trapezoid(valid_alpha, altitude, initial=0.0)
    transmission = np.exp(-2.0 * optical_depth)
    return np.clip(transmission, 0.0, 1.0).astype(np.float64)


def calculate_simulated_molecular_signal(beta_mol: np.ndarray, alpha_mol: np.ndarray, altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a molecular elastic lidar signal shape."""
    beta = np.asarray(beta_mol, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if beta.ndim != 1 or altitude.ndim != 1 or beta.size != altitude.size:
        raise ValueError("beta_mol and altitude_m must be 1D arrays with the same length.")

    transmission = calculate_molecular_two_way_transmission(alpha_mol, altitude)
    positive_altitudes = altitude[altitude > 0.0]
    min_positive_altitude = float(positive_altitudes[0]) if positive_altitudes.size else 1.0
    safe_altitude = np.where(altitude > 0.0, altitude, min_positive_altitude)
    simulated_signal = beta * transmission / (safe_altitude**2)
    return simulated_signal.astype(np.float64), transmission


def linear_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, float, int]:
    """Fit measured signal as a linear function of molecular signal.

    The slope scales the molecular signal onto the lidar signal.  The intercept
    is retained as a diagnostic because a large intercept indicates incomplete
    background correction or contamination in the Rayleigh reference interval.
    """
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    x = simulated[start:stop]
    y = measured[start:stop]
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if valid.sum() < 2:
        return np.nan, np.nan, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    return float(slope), float(intercept), float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())


def robust_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, int]:
    """Estimate Rayleigh calibration factor using a robust reference window."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError("measured_signal, simulated_molecular_signal and altitude_m must be 1D arrays.")
    if not (measured.size == simulated.size == altitude.size):
        raise ValueError("measured_signal, simulated_molecular_signal and altitude_m must have the same length.")

    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    ratio = measured[start:stop] / simulated[start:stop]
    valid = np.isfinite(ratio) & (ratio > 0.0)
    if not np.any(valid):
        return np.nan, float(altitude[start]), float(altitude[stop - 1]), 0
    factor = float(np.nanmedian(ratio[valid]))
    return factor, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())


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
    altitude, min_alt, max_alt = _resolve_altitude_search_units(altitude, min_alt, max_alt, altitude_units=altitude_units)

    if rcs.ndim != 1 or beta_mol.ndim != 1 or altitude.ndim != 1:
        raise ValueError("rcs, beta_mol and altitude must be 1D arrays.")
    if not (rcs.size == beta_mol.size == altitude.size):
        raise ValueError("rcs, beta_mol and altitude must have the same length.")

    window_size = max(int(window_size), 3)
    valid_indices = np.where((altitude >= min_alt) & (altitude <= max_alt))[0]
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
        altitude_span = max(float(np.nanmax(wa) - np.nanmin(wa)), 1.0)
        rel_slope = abs(slope) * altitude_span / mean_ratio
        cost = rel_var + rel_slope
        if cost < min_cost:
            min_cost = cost
            best_idx = start_idx + (window_size // 2)

    if best_idx == -1:
        best_idx = int(valid_indices[-1])
    return int(best_idx)
