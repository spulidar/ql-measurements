"""Molecular/Rayleigh calculations for lidar inversion."""

from __future__ import annotations

from typing import Literal

import numpy as np

from milgrau.physics.constants import BOLTZMANN_CONSTANT_J_K, RAYLEIGH_LIDAR_RATIO_SR


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

    temp_safe = np.where((temp_profile > 0.0) & np.isfinite(temp_profile), temp_profile, np.nan)
    press_safe = np.where((press_profile > 0.0) & np.isfinite(press_profile), press_profile, np.nan)
    press_pa = press_safe * 100.0
    n_density = press_pa / (BOLTZMANN_CONSTANT_J_K * temp_safe)

    sigma = 5.45e-28 * ((550.0 / wavelength_nm) ** 4)
    alpha_mol = n_density * sigma
    beta_mol = alpha_mol / RAYLEIGH_LIDAR_RATIO_SR
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
        rel_slope = abs(slope) / mean_ratio
        cost = rel_var + (rel_slope * 5.0)
        if cost < min_cost:
            min_cost = cost
            best_idx = start_idx + (window_size // 2)

    if best_idx == -1:
        best_idx = int(valid_indices[-1])
    return int(best_idx)
