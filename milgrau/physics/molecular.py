"""Molecular/Rayleigh calculations for lidar inversion."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

from milgrau.physics.constants import BOLTZMANN_CONSTANT_J_K, RAYLEIGH_LIDAR_RATIO_SR

# Practical Rayleigh scattering cross-section near 550 nm in m².
RAYLEIGH_CROSS_SECTION_550_M2 = 5.45e-31


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

    temp_safe = np.where(
        (temp_profile > 0.0) & np.isfinite(temp_profile), temp_profile, np.nan
    )
    press_safe = np.where(
        (press_profile > 0.0) & np.isfinite(press_profile), press_profile, np.nan
    )
    press_pa = press_safe * 100.0
    n_density = press_pa / (BOLTZMANN_CONSTANT_J_K * temp_safe)

    sigma = RAYLEIGH_CROSS_SECTION_550_M2 * ((550.0 / wavelength_nm) ** 4)
    alpha_mol = n_density * sigma
    beta_mol = alpha_mol / RAYLEIGH_LIDAR_RATIO_SR
    return beta_mol.astype(np.float64), alpha_mol.astype(np.float64)


def calculate_molecular_two_way_transmission(
    alpha_mol: np.ndarray, altitude_m: np.ndarray
) -> np.ndarray:
    """Calculate molecular two-way transmission from extinction.

    The returned factor is ``exp(-2 * integral(alpha_mol dz))`` and is used to
    build a physically consistent molecular elastic signal for Rayleigh fitting.
    """
    alpha = np.asarray(alpha_mol, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if alpha.ndim != 1 or altitude.ndim != 1 or alpha.size != altitude.size:
        raise ValueError(
            "alpha_mol and altitude_m must be 1D arrays with the same length."
        )

    valid_alpha = np.where(np.isfinite(alpha) & (alpha >= 0.0), alpha, 0.0)
    optical_depth = cumulative_trapezoid(valid_alpha, altitude, initial=0.0)
    transmission = np.exp(-2.0 * optical_depth)
    return np.clip(transmission, 0.0, 1.0).astype(np.float64)


def calculate_simulated_molecular_signal(
    beta_mol: np.ndarray,
    alpha_mol: np.ndarray,
    altitude_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a molecular elastic lidar signal shape.

    The simulated molecular signal is proportional to
    ``beta_mol(z) * T_mol(z)^2 / z^2``. The first range bin is protected from
    division by zero using the first positive altitude step.
    """
    beta = np.asarray(beta_mol, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if beta.ndim != 1 or altitude.ndim != 1 or beta.size != altitude.size:
        raise ValueError(
            "beta_mol and altitude_m must be 1D arrays with the same length."
        )

    transmission = calculate_molecular_two_way_transmission(alpha_mol, altitude)
    positive_altitudes = altitude[altitude > 0.0]
    min_positive_altitude = (
        float(positive_altitudes[0]) if positive_altitudes.size else 1.0
    )
    safe_altitude = np.where(altitude > 0.0, altitude, min_positive_altitude)
    simulated_signal = beta * transmission / (safe_altitude**2)
    return simulated_signal.astype(np.float64), transmission


def robust_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, int]:
    """Estimate Rayleigh calibration factor using a robust reference window.

    Returns the median measured/simulated ratio in the reference window, the
    start and stop altitudes of that window, and the number of valid bins used.
    """
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError(
            "measured_signal, simulated_molecular_signal and altitude_m must be 1D arrays."
        )
    if not (measured.size == simulated.size == altitude.size):
        raise ValueError(
            "measured_signal, simulated_molecular_signal and altitude_m must have the same length."
        )

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
        valid = (
            np.isfinite(window_ratio) & np.isfinite(window_alt) & (window_ratio > 0.0)
        )
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
