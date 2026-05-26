"""LEBEAR Level 2 optical inversion pipeline.

LEBEAR converts Level 1 Range Corrected Signal products into first-pass Level 2
optical products.  The fundamental retrieval unit is a temporal block.  Signals
are averaged first, then Analog/Photon Counting gluing, Rayleigh calibration,
scattering ratio and KFS/Fernald-Sasano inversion are calculated independently
for each block.  Mean products are averages of valid block products.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.contracts import validate_level1_contract, validate_level2_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.paths import LEVEL2_SUFFIX, level2_output_path, processed_data_root
from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.physics.gluing import slide_glue_signals
from milgrau.physics.kfs import kfs_inversion_monte_carlo
from milgrau.physics.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
    find_optimal_reference_altitude,
    linear_rayleigh_calibration_factor,
)
from milgrau.visualization.level2_qa import plot_all_level2_qa


KFS_BRANCH_INVALID = 0
KFS_BRANCH_BACKWARD_BELOW_REFERENCE = 1
KFS_BRANCH_REFERENCE_WINDOW = 2
KFS_BRANCH_FORWARD_ABOVE_REFERENCE_EXPERIMENTAL = 3


def _incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def discover_level1_files(config: Mapping[str, Any], root_dir: str | Path | None = None) -> list[Path]:
    """Discover Level 1 RCS NetCDF files available for LEBEAR processing."""
    return sorted(processed_data_root(config, root_dir=root_dir).rglob("*_level1_rcs.nc"))


def _get_wavelengths_to_process(config: Mapping[str, Any]) -> list[int]:
    """Return configured wavelengths for Level 2 processing."""
    raw_values = config.get("inversion", {}).get("wavelengths_to_process", [532])
    wavelengths: list[int] = []
    for value in raw_values:
        try:
            wavelength = int(value)
        except (TypeError, ValueError):
            continue
        if wavelength > 0 and wavelength not in wavelengths:
            wavelengths.append(wavelength)
    return wavelengths or [532]


def _infer_channel_pair(ds_l1: xr.Dataset, wavelength_nm: int) -> tuple[str | None, str | None]:
    """Infer Analog and Photon Counting channel names for one wavelength."""
    channels = [str(channel) for channel in ds_l1["channel"].values]
    prefix = f"{int(wavelength_nm)}."
    analog = next((channel for channel in channels if channel.startswith(prefix) and channel.upper().endswith(".AN")), None)
    photon = next(
        (
            channel
            for channel in channels
            if channel.startswith(prefix) and (channel.upper().endswith(".PC") or channel.upper().endswith(".PH"))
        ),
        None,
    )
    return analog, photon


def _get_lidar_ratio(config: Mapping[str, Any], wavelength_nm: int, measurement_time: Any) -> tuple[float, float]:
    """Return monthly aerosol lidar ratio and standard deviation for one wavelength."""
    month = pd.to_datetime(measurement_time).strftime("%m")
    inv_cfg = config.get("inversion", {})
    wavelength_key = str(int(wavelength_nm))
    ratios = inv_cfg.get("lidar_ratios_sr", inv_cfg.get("lidar_ratios", {}))
    lr_base = float(ratios.get(wavelength_key, {}).get(month, 60.0))
    lr_std = float(inv_cfg.get("lidar_ratio_std_sr", {}).get(wavelength_key, 10.0))
    return lr_base, lr_std


def _get_gluing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return gluing configuration with safe defaults."""
    gluing_cfg = config.get("inversion", {}).get("gluing", {}) or {}
    return {
        "window_size": int(gluing_cfg.get("window_length_bins", 150)),
        "min_corr": float(gluing_cfg.get("correlation_threshold", 0.95)),
        "search_min_idx": int(gluing_cfg.get("search_min_idx", 200)),
        "search_max_idx": int(gluing_cfg.get("search_max_idx", 2000)),
        "intercept_threshold": float(gluing_cfg.get("intercept_threshold", 0.5)),
        "gaussian_threshold": float(gluing_cfg.get("gaussian_threshold", 0.1)),
        "minmax_threshold": float(gluing_cfg.get("minmax_threshold", 0.5)),
        "fallback_to_photon_counting": bool(gluing_cfg.get("fallback_to_photon_counting", True)),
    }


def _get_molecular_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return molecular reference configuration with safe defaults."""
    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    return {
        "ref_alt_min_m": float(fit_cfg.get("ref_alt_min_m", 5000.0)),
        "ref_alt_max_m": float(fit_cfg.get("ref_alt_max_m", 25000.0)),
        "ref_window_bins": int(fit_cfg.get("ref_window_bins", 2667)),
        "max_relative_slope": float(fit_cfg.get("max_relative_slope", 0.25)),
        "max_relative_variance": float(fit_cfg.get("max_relative_variance", 0.50)),
        "min_valid_fraction": float(fit_cfg.get("min_valid_fraction", 0.50)),
    }


def _get_kfs_mode(config: Mapping[str, Any]) -> str:
    """Return the configured KFS integration mode."""
    mode = str(config.get("inversion", {}).get("kfs_mode", "two_sided")).strip().lower()
    if mode not in {"backward", "two_sided"}:
        return "backward"
    return mode


def _kfs_mode_description(mode: str) -> str:
    """Return a human-readable description of the KFS mode."""
    if mode == "two_sided":
        return "Backward below reference and forward above reference; above-reference retrieval is experimental and noise-sensitive."
    return "Backward Fernald-Sasano retrieval below the reference altitude."


def _get_block_average_minutes(config: Mapping[str, Any]) -> int:
    """Return temporal block size used by LEBEAR retrievals."""
    inv_cfg = config.get("inversion", {})
    return max(int(inv_cfg.get("block_average_minutes", inv_cfg.get("temporal_average_minutes", 15))), 1)


def _build_kfs_branch(altitude_m: np.ndarray, ref_start_m: float, ref_stop_m: float, mode: str) -> np.ndarray:
    """Build per-altitude validity/branch flags for KFS products."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    branch = np.zeros(altitude.size, dtype=np.int8)
    finite = np.isfinite(altitude)
    branch[finite & (altitude < ref_start_m)] = KFS_BRANCH_BACKWARD_BELOW_REFERENCE
    branch[finite & (altitude >= ref_start_m) & (altitude <= ref_stop_m)] = KFS_BRANCH_REFERENCE_WINDOW
    if mode == "two_sided":
        branch[finite & (altitude > ref_stop_m)] = KFS_BRANCH_FORWARD_ABOVE_REFERENCE_EXPERIMENTAL
    return branch


def _evaluate_rayleigh_reference(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
    fit_config: Mapping[str, Any],
    calibration_factor: float,
) -> dict[str, float | int]:
    """Evaluate whether the selected Rayleigh reference window is acceptable.

    In a clean molecular interval the measured/molecular ratio should be nearly
    flat.  Slope and variance diagnose aerosol contamination, cloud leakage or
    residual background structure inside the reference window.
    """
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    ratio = measured[start:stop] / simulated[start:stop]
    window_altitude = altitude[start:stop]
    valid = np.isfinite(ratio) & np.isfinite(window_altitude) & (ratio > 0.0)
    valid_count = int(valid.sum())
    window_size = max(int(stop - start), 1)
    valid_fraction = float(valid_count / window_size)

    relative_variance = np.inf
    relative_slope = np.inf
    if valid_count >= 3:
        valid_ratio = ratio[valid]
        valid_altitude = window_altitude[valid]
        mean_ratio = float(np.nanmean(valid_ratio))
        if np.isfinite(mean_ratio) and mean_ratio > 0.0:
            relative_variance = float(np.nanvar(valid_ratio) / (mean_ratio**2))
            slope, _ = np.polyfit(valid_altitude, valid_ratio, 1)
            altitude_span = float(np.nanmax(valid_altitude) - np.nanmin(valid_altitude))
            relative_slope = float(abs(slope) * max(altitude_span, 1.0) / mean_ratio)

    max_relative_slope = float(fit_config.get("max_relative_slope", 0.25))
    max_relative_variance = float(fit_config.get("max_relative_variance", 0.50))
    min_valid_fraction = float(fit_config.get("min_valid_fraction", 0.50))
    success = (
        np.isfinite(calibration_factor)
        and calibration_factor > 0.0
        and valid_fraction >= min_valid_fraction
        and np.isfinite(relative_variance)
        and relative_variance <= max_relative_variance
        and np.isfinite(relative_slope)
        and relative_slope <= max_relative_slope
    )
    return {
        "success_flag": int(success),
        "relative_slope": float(relative_slope),
        "relative_variance": float(relative_variance),
        "valid_fraction": float(valid_fraction),
        "max_relative_slope": max_relative_slope,
        "max_relative_variance": max_relative_variance,
        "min_valid_fraction": min_valid_fraction,
    }


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator/denominator where both terms are finite and positive."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=np.float64), where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0))


def _nanmean_or_nan(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a NaN-safe mean without RuntimeWarning for all-NaN slices."""
    arr = np.asarray(matrix, dtype=np.float64)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    total = np.nansum(arr, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def _build_thermodynamic_profile(ds_l1: xr.Dataset, altitude_agl_m: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Build pressure and temperature profiles on the lidar altitude grid."""
    site_cfg = config.get("site", {})
    station_altitude_m = float(site_cfg.get("station_altitude_m", config.get("physics", {}).get("station_altitude_m", 0.0)))
    altitude_asl_m = altitude_agl_m + station_altitude_m

    standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)
    if {"Radiosonde_Temperature_K", "Radiosonde_Pressure_hPa", "radiosonde_altitude"}.issubset(set(ds_l1.variables) | set(ds_l1.coords)):
        radio_alt = np.asarray(ds_l1["radiosonde_altitude"].values, dtype=np.float64)
        radio_temp = np.asarray(ds_l1["Radiosonde_Temperature_K"].values, dtype=np.float64)
        radio_pressure = np.asarray(ds_l1["Radiosonde_Pressure_hPa"].values, dtype=np.float64)
        valid = np.isfinite(radio_alt) & np.isfinite(radio_temp) & np.isfinite(radio_pressure) & (radio_pressure > 0.0) & (radio_temp > 0.0)
        if valid.sum() >= 2:
            order = np.argsort(radio_alt[valid])
            alt_sorted = radio_alt[valid][order]
            temp_sorted = radio_temp[valid][order]
            pressure_sorted = radio_pressure[valid][order]
            temperature = np.interp(altitude_asl_m, alt_sorted, temp_sorted, left=np.nan, right=np.nan)
            pressure = np.interp(altitude_asl_m, alt_sorted, pressure_sorted, left=np.nan, right=np.nan)
            temperature = np.where(np.isfinite(temperature), temperature, standard_temperature)
            pressure = np.where(np.isfinite(pressure), pressure, standard_pressure)
            return pressure.astype(np.float64), temperature.astype(np.float64), "radiosonde_with_standard_fallback"

    return standard_pressure.astype(np.float64), standard_temperature.astype(np.float64), "standard_atmosphere"


def _propagate_glued_error(analog_error: np.ndarray, photon_error: np.ndarray, slope: float, min_bin: int, max_bin: int) -> np.ndarray:
    """Propagate one-sigma uncertainty through analog/PC gluing weights."""
    analog = np.asarray(analog_error, dtype=np.float64)
    photon = np.asarray(photon_error, dtype=np.float64)
    if analog.ndim != 1 or photon.ndim != 1 or analog.size != photon.size:
        raise ValueError("analog_error and photon_error must be 1D arrays with the same length.")

    n_bins = analog.size
    min_bin = max(int(min_bin), 0)
    max_bin = min(int(max_bin), n_bins)
    if max_bin <= min_bin:
        raise ValueError("max_bin must be greater than min_bin for gluing uncertainty propagation.")

    scaled_analog_error = abs(float(slope)) * analog
    glued_error = photon.copy()
    glued_error[:min_bin] = scaled_analog_error[:min_bin]
    gluing_length = max_bin - min_bin
    analog_weights = 1.0 - np.arange(gluing_length, dtype=np.float64) / float(gluing_length)
    photon_weights = 1.0 - analog_weights
    glued_error[min_bin:max_bin] = np.sqrt((analog_weights * scaled_analog_error[min_bin:max_bin]) ** 2 + (photon_weights * photon[min_bin:max_bin]) ** 2)
    return glued_error


def _error_of_mean(error_matrix: np.ndarray) -> np.ndarray:
    """Combine profile one-sigma errors into uncertainty of the temporal mean."""
    valid_count = np.sum(np.isfinite(error_matrix), axis=0)
    valid_count = np.maximum(valid_count, 1)
    return np.sqrt(np.nansum(error_matrix**2, axis=0)) / valid_count


def _block_groups(time_values: np.ndarray, minutes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return block labels and index groups for temporal averaging."""
    times = pd.to_datetime(time_values)
    labels = times.floor(f"{int(minutes)}min")
    unique_labels = pd.Index(labels).unique().sort_values()
    groups = [np.where(labels == label)[0] for label in unique_labels]
    return unique_labels.to_numpy(dtype="datetime64[ns]"), groups


def _mean_by_groups(matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Calculate NaN-safe means for a time x altitude matrix over index groups."""
    return np.stack([_nanmean_or_nan(matrix[group, :], axis=0) for group in groups], axis=0)


def _error_by_groups(error_matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Calculate uncertainty of grouped means from one-sigma profiles."""
    return np.stack([_error_of_mean(error_matrix[group, :]) for group in groups], axis=0)


def _expand_blocks_to_time(block_matrix: np.ndarray, groups: list[np.ndarray], n_time: int) -> np.ndarray:
    """Map block products back to the original time axis for compatibility."""
    block_matrix = np.asarray(block_matrix, dtype=np.float64)
    expanded = np.full((n_time, block_matrix.shape[-1]), np.nan, dtype=np.float64)
    for block_idx, group in enumerate(groups):
        expanded[group, :] = block_matrix[block_idx, :]
    return expanded


def _expand_block_vector_to_time(values: np.ndarray, groups: list[np.ndarray], n_time: int, dtype=np.float64) -> np.ndarray:
    """Map one block diagnostic vector back to the original time axis."""
    values = np.asarray(values)
    expanded = np.full(n_time, np.nan, dtype=np.float64)
    for block_idx, group in enumerate(groups):
        expanded[group] = values[block_idx]
    return expanded.astype(dtype) if np.issubdtype(dtype, np.integer) else expanded.astype(dtype)


def _run_kfs_profile(rcs: np.ndarray, rcs_error: np.ndarray, altitude_m: np.ndarray, beta_mol: np.ndarray, ref_idx: int, lr_base: float, lr_std: float, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run KFS for one RCS profile using the configured inversion options."""
    inv_cfg = config.get("inversion", {})
    return kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=lr_base,
        lr_std=lr_std,
        ref_idx=ref_idx,
        n_iterations=int(inv_cfg.get("monte_carlo_iterations", 300)),
        rcs_error=rcs_error,
        beta_ref_relative_std=float(inv_cfg.get("beta_ref_relative_std", 0.10)),
        aerosol_ref_fraction=float(inv_cfg.get("aerosol_ref_fraction", 0.0)),
        altitude_units="m",
        min_lidar_ratio=float(inv_cfg.get("min_lidar_ratio_sr", 10.0)),
        allow_negative_aerosol=bool(inv_cfg.get("allow_negative_aerosol", False)),
        seed=inv_cfg.get("random_seed"),
        mode=_get_kfs_mode(config),
    )


def _nan_optical_products_like(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return NaN optical products with the same shape as one reference array."""
    nan_arr = np.full_like(np.asarray(reference, dtype=np.float64), np.nan, dtype=np.float64)
    return nan_arr.copy(), nan_arr.copy(), nan_arr.copy(), nan_arr.copy()


def _origin_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, int]:
    """Return multiplicative Rayleigh calibration constrained through the origin.

    Elastic Rayleigh calibration is physically a multiplicative normalization of
    the molecular return.  A free intercept is useful as a diagnostic of residual
    background, but the primary calibration factor should not absorb an additive
    offset.
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
        return np.nan, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())
    factor = float(np.nansum(x[valid] * y[valid]) / np.nansum(x[valid] ** 2))
    return factor, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())


def _valid_block_mean(block_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Average only block products that passed gluing and Rayleigh QA."""
    block_matrix = np.asarray(block_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return _nanmean_or_nan(block_matrix[valid, :], axis=0)
    return np.full(block_matrix.shape[-1], np.nan, dtype=np.float64)


def _valid_block_error(block_error_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Combine block uncertainties using only valid retrieval blocks."""
    block_error_matrix = np.asarray(block_error_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return _error_of_mean(block_error_matrix[valid, :])
    return np.full(block_error_matrix.shape[-1], np.nan, dtype=np.float64)


def _process_wavelength(ds_l1: xr.Dataset, wavelength_nm: int, altitude_m: np.ndarray, config: Mapping[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Process one wavelength using block retrievals."""
    analog_ch, photon_ch = _infer_channel_pair(ds_l1, wavelength_nm)
    if analog_ch is None and photon_ch is None:
        raise ValueError(f"No channel found for wavelength {wavelength_nm} nm.")

    signal_da = ds_l1["range_corrected_signal"]
    error_da = ds_l1["range_corrected_signal_error"]
    n_time = ds_l1.sizes.get("time", 1)
    n_alt = altitude_m.size
    block_minutes = _get_block_average_minutes(config)
    block_time, block_groups = _block_groups(ds_l1["time"].values, block_minutes)
    n_block = len(block_groups)
    gluing_cfg = _get_gluing_config(config)

    if photon_ch is not None:
        photon_signal = signal_da.sel(channel=photon_ch).values.astype(np.float64)
        photon_error = error_da.sel(channel=photon_ch).values.astype(np.float64)
        photon_block = _mean_by_groups(photon_signal, block_groups)
        photon_error_block = _error_by_groups(photon_error, block_groups)
    else:
        photon_block = None
        photon_error_block = None

    if analog_ch is not None:
        analog_signal = signal_da.sel(channel=analog_ch).values.astype(np.float64)
        analog_error = error_da.sel(channel=analog_ch).values.astype(np.float64)
        analog_block = _mean_by_groups(analog_signal, block_groups)
        analog_error_block = _error_by_groups(analog_error, block_groups)
    else:
        analog_block = None
        analog_error_block = None

    glued_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    glued_error_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    gluing_success_block = np.zeros(n_block, dtype=np.int8)
    gluing_fallback_block = np.zeros(n_block, dtype=np.int8)
    gluing_split_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_start_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_stop_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_slope_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_intercept_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_correlation_block = np.full(n_block, np.nan, dtype=np.float64)

    if analog_block is not None and photon_block is not None and analog_error_block is not None and photon_error_block is not None:
        source = "block_mean_analog_photon_glued"
        for block_idx in range(n_block):
            glued_profile, split_point, slope_i, intercept_i, diagnostics = slide_glue_signals(
                analog_sig=analog_block[block_idx, :],
                pc_sig=photon_block[block_idx, :],
                altitude=altitude_m,
                window_size=gluing_cfg["window_size"],
                min_corr=gluing_cfg["min_corr"],
                search_min_idx=gluing_cfg["search_min_idx"],
                search_max_idx=gluing_cfg["search_max_idx"],
                intercept_threshold=gluing_cfg["intercept_threshold"],
                gaussian_threshold=gluing_cfg["gaussian_threshold"],
                minmax_threshold=gluing_cfg["minmax_threshold"],
                return_diagnostics=True,
            )
            gluing_slope_block[block_idx] = slope_i
            gluing_intercept_block[block_idx] = intercept_i
            gluing_correlation_block[block_idx] = float(diagnostics.get("best_corr", np.nan))
            if split_point >= 0:
                min_bin = int(diagnostics.get("min_bin", max(split_point - gluing_cfg["window_size"] // 2, 0)))
                max_bin = int(diagnostics.get("max_bin", min(split_point + gluing_cfg["window_size"] // 2, n_alt)))
                glued_block[block_idx, :] = glued_profile
                glued_error_block[block_idx, :] = _propagate_glued_error(analog_error_block[block_idx, :], photon_error_block[block_idx, :], slope_i, min_bin, max_bin)
                gluing_success_block[block_idx] = 1
                gluing_split_block[block_idx] = float(altitude_m[split_point])
                gluing_start_block[block_idx] = float(altitude_m[min_bin])
                gluing_stop_block[block_idx] = float(altitude_m[max_bin - 1])
            elif gluing_cfg["fallback_to_photon_counting"]:
                glued_block[block_idx, :] = photon_block[block_idx, :]
                glued_error_block[block_idx, :] = photon_error_block[block_idx, :]
                gluing_fallback_block[block_idx] = 1
        if gluing_success_block.sum() == 0 and not gluing_cfg["fallback_to_photon_counting"]:
            raise ValueError(f"{wavelength_nm} nm has no successful block gluing and photon fallback is disabled.")
        logger.info(f"  -> {wavelength_nm} nm block gluing success: {100.0 * gluing_success_block.sum() / max(n_block, 1):.1f}% ({analog_ch} + {photon_ch}); fallback blocks: {int(gluing_fallback_block.sum())}.")
    else:
        fallback_ch = photon_ch or analog_ch
        fallback_block = photon_block if photon_block is not None else analog_block
        fallback_error_block = photon_error_block if photon_error_block is not None else analog_error_block
        if fallback_ch is None or fallback_block is None or fallback_error_block is None:
            raise ValueError(f"No usable channel found for wavelength {wavelength_nm} nm.")
        if not gluing_cfg["fallback_to_photon_counting"]:
            raise ValueError(f"{wavelength_nm} nm cannot perform gluing because only {fallback_ch} is available and photon fallback is disabled.")
        glued_block[:, :] = fallback_block
        glued_error_block[:, :] = fallback_error_block
        gluing_fallback_block[:] = 1
        source = f"block_mean_single_channel_{fallback_ch}"
        logger.warning(f"  -> {wavelength_nm} nm using block single-channel fallback: {fallback_ch}.")

    pressure_hpa, temperature_k, molecular_source = _build_thermodynamic_profile(ds_l1, altitude_m, config)
    beta_mol, alpha_mol = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, molecular_transmission = calculate_simulated_molecular_signal(beta_mol, alpha_mol, altitude_m)
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(altitude_m > 0.0, altitude_m, positive_altitudes[0] if positive_altitudes.size else 1.0)
    simulated_molecular_rcs = simulated_signal * safe_altitude**2

    fit_cfg = _get_molecular_fit_config(config)
    lr_base, lr_std = _get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    kfs_mode = _get_kfs_mode(config)

    rayleigh_success_block = np.zeros(n_block, dtype=np.int8)
    ref_altitude_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_start_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_stop_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_valid_bins_block = np.zeros(n_block, dtype=np.int32)
    ref_relative_slope_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_relative_variance_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_valid_fraction_block = np.full(n_block, np.nan, dtype=np.float64)
    calibration_factor_block = np.full(n_block, np.nan, dtype=np.float64)
    calibration_intercept_block = np.full(n_block, np.nan, dtype=np.float64)
    scaled_molecular_rcs_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    scattering_ratio_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    beta_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    beta_block_std = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    alpha_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    alpha_block_std = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    kfs_branch_block = np.zeros((n_block, n_alt), dtype=np.int8)

    for block_idx in range(n_block):
        if gluing_success_block[block_idx] != 1:
            continue
        ref_idx = find_optimal_reference_altitude(
            rcs=glued_block[block_idx, :],
            beta_mol=simulated_molecular_rcs,
            altitude=altitude_m,
            min_alt=fit_cfg["ref_alt_min_m"],
            max_alt=fit_cfg["ref_alt_max_m"],
            window_size=fit_cfg["ref_window_bins"],
            altitude_units="m",
        )
        factor, ref_start_m, ref_stop_m, ref_valid_bins = _origin_rayleigh_calibration_factor(
            measured_signal=glued_block[block_idx, :],
            simulated_molecular_signal=simulated_molecular_rcs,
            altitude_m=altitude_m,
            reference_center_idx=ref_idx,
            reference_window_bins=fit_cfg["ref_window_bins"],
        )
        _, intercept_diag, _, _, _ = linear_rayleigh_calibration_factor(
            measured_signal=glued_block[block_idx, :],
            simulated_molecular_signal=simulated_molecular_rcs,
            altitude_m=altitude_m,
            reference_center_idx=ref_idx,
            reference_window_bins=fit_cfg["ref_window_bins"],
        )
        qa = _evaluate_rayleigh_reference(glued_block[block_idx, :], simulated_molecular_rcs, altitude_m, ref_idx, fit_cfg["ref_window_bins"], fit_cfg, factor)
        calibration_factor_block[block_idx] = factor
        calibration_intercept_block[block_idx] = intercept_diag
        ref_altitude_block[block_idx] = float(altitude_m[ref_idx])
        ref_start_block[block_idx] = ref_start_m
        ref_stop_block[block_idx] = ref_stop_m
        ref_valid_bins_block[block_idx] = int(ref_valid_bins)
        ref_relative_slope_block[block_idx] = float(qa["relative_slope"])
        ref_relative_variance_block[block_idx] = float(qa["relative_variance"])
        ref_valid_fraction_block[block_idx] = float(qa["valid_fraction"])
        scaled_molecular_rcs_block[block_idx, :] = simulated_molecular_rcs * factor
        scattering_ratio_block[block_idx, :] = _safe_ratio(glued_block[block_idx, :], scaled_molecular_rcs_block[block_idx, :])
        kfs_branch_block[block_idx, :] = _build_kfs_branch(altitude_m, ref_start_m, ref_stop_m, kfs_mode)
        if int(qa["success_flag"]) == 1:
            rayleigh_success_block[block_idx] = 1
            beta_mean, beta_std, alpha_mean, alpha_std = _run_kfs_profile(glued_block[block_idx, :], glued_error_block[block_idx, :], altitude_m, beta_mol, ref_idx, lr_base, lr_std, config)
            beta_block[block_idx, :] = beta_mean
            beta_block_std[block_idx, :] = beta_std
            alpha_block[block_idx, :] = alpha_mean
            alpha_block_std[block_idx, :] = alpha_std

    valid_block = (gluing_success_block == 1) & (rayleigh_success_block == 1)
    if not valid_block.any():
        logger.warning(f"  -> {wavelength_nm} nm has no valid retrieval block. Mean optical products set to NaN.")

    glued_mean = _valid_block_mean(glued_block, valid_block)
    glued_error_mean = _valid_block_error(glued_error_block, valid_block)
    scattering_ratio_mean = _valid_block_mean(scattering_ratio_block, valid_block)
    beta_mean = _valid_block_mean(beta_block, valid_block)
    beta_std_mean = _valid_block_error(beta_block_std, valid_block)
    alpha_mean = _valid_block_mean(alpha_block, valid_block)
    alpha_std_mean = _valid_block_error(alpha_block_std, valid_block)

    if valid_block.any():
        calibration_factor = float(np.nanmedian(calibration_factor_block[valid_block]))
        calibration_intercept = float(np.nanmedian(calibration_intercept_block[valid_block]))
        ref_altitude = float(np.nanmedian(ref_altitude_block[valid_block]))
        ref_start = float(np.nanmedian(ref_start_block[valid_block]))
        ref_stop = float(np.nanmedian(ref_stop_block[valid_block]))
        ref_valid_bins = int(np.nanmedian(ref_valid_bins_block[valid_block]))
        ref_rel_slope = float(np.nanmedian(ref_relative_slope_block[valid_block]))
        ref_rel_var = float(np.nanmedian(ref_relative_variance_block[valid_block]))
        ref_valid_fraction = float(np.nanmedian(ref_valid_fraction_block[valid_block]))
        scaled_molecular_rcs = simulated_molecular_rcs * calibration_factor
        kfs_branch = _build_kfs_branch(altitude_m, ref_start, ref_stop, kfs_mode)
        rayleigh_success = 1
    else:
        calibration_factor = np.nan
        calibration_intercept = np.nan
        ref_altitude = np.nan
        ref_start = np.nan
        ref_stop = np.nan
        ref_valid_bins = 0
        ref_rel_slope = np.nan
        ref_rel_var = np.nan
        ref_valid_fraction = np.nan
        scaled_molecular_rcs = np.full(n_alt, np.nan, dtype=np.float64)
        kfs_branch = np.zeros(n_alt, dtype=np.int8)
        rayleigh_success = 0

    time_glued = _expand_blocks_to_time(glued_block, block_groups, n_time)
    time_glued_error = _expand_blocks_to_time(glued_error_block, block_groups, n_time)
    time_gluing_success = _expand_block_vector_to_time(gluing_success_block, block_groups, n_time, dtype=np.int8)
    time_gluing_fallback = _expand_block_vector_to_time(gluing_fallback_block, block_groups, n_time, dtype=np.int8)
    time_gluing_split = _expand_block_vector_to_time(gluing_split_block, block_groups, n_time)
    time_gluing_start = _expand_block_vector_to_time(gluing_start_block, block_groups, n_time)
    time_gluing_stop = _expand_block_vector_to_time(gluing_stop_block, block_groups, n_time)
    time_gluing_slope = _expand_block_vector_to_time(gluing_slope_block, block_groups, n_time)
    time_gluing_intercept = _expand_block_vector_to_time(gluing_intercept_block, block_groups, n_time)
    time_gluing_correlation = _expand_block_vector_to_time(gluing_correlation_block, block_groups, n_time)

    return {
        "wavelength": wavelength_nm,
        "block_time": block_time,
        "molecular_source": molecular_source,
        "molecular_backscatter": beta_mol,
        "molecular_extinction": alpha_mol,
        "molecular_transmission": molecular_transmission,
        "simulated_molecular_signal": simulated_signal,
        "simulated_molecular_range_corrected_signal": simulated_molecular_rcs,
        "scaled_molecular_range_corrected_signal": scaled_molecular_rcs,
        "scaled_molecular_range_corrected_signal_block": scaled_molecular_rcs_block,
        "glued_range_corrected_signal": time_glued,
        "glued_range_corrected_signal_error": time_glued_error,
        "glued_range_corrected_signal_block": glued_block,
        "glued_range_corrected_signal_error_block": glued_error_block,
        "glued_range_corrected_signal_mean": glued_mean,
        "glued_range_corrected_signal_error_mean": glued_error_mean,
        "scattering_ratio_mean": scattering_ratio_mean,
        "scattering_ratio_block": scattering_ratio_block,
        "aerosol_backscatter": beta_mean,
        "aerosol_backscatter_error": beta_std_mean,
        "aerosol_extinction": alpha_mean,
        "aerosol_extinction_error": alpha_std_mean,
        "aerosol_backscatter_block": beta_block,
        "aerosol_backscatter_error_block": beta_block_std,
        "aerosol_extinction_block": alpha_block,
        "aerosol_extinction_error_block": alpha_block_std,
        "valid_retrieval_block_flag": valid_block.astype(np.int8),
        "rayleigh_reference_altitude_m": ref_altitude,
        "rayleigh_reference_start_altitude_m": ref_start,
        "rayleigh_reference_stop_altitude_m": ref_stop,
        "rayleigh_reference_valid_bins": ref_valid_bins,
        "rayleigh_reference_success_flag": rayleigh_success,
        "rayleigh_reference_relative_slope": ref_rel_slope,
        "rayleigh_reference_relative_variance": ref_rel_var,
        "rayleigh_reference_valid_fraction": ref_valid_fraction,
        "rayleigh_calibration_factor": calibration_factor,
        "rayleigh_calibration_intercept": calibration_intercept,
        "rayleigh_reference_altitude_m_block": ref_altitude_block,
        "rayleigh_reference_start_altitude_m_block": ref_start_block,
        "rayleigh_reference_stop_altitude_m_block": ref_stop_block,
        "rayleigh_reference_valid_bins_block": ref_valid_bins_block,
        "rayleigh_reference_success_flag_block": rayleigh_success_block,
        "rayleigh_reference_relative_slope_block": ref_relative_slope_block,
        "rayleigh_reference_relative_variance_block": ref_relative_variance_block,
        "rayleigh_reference_valid_fraction_block": ref_valid_fraction_block,
        "rayleigh_calibration_factor_block": calibration_factor_block,
        "rayleigh_calibration_intercept_block": calibration_intercept_block,
        "lidar_ratio_assumed_sr": lr_base,
        "lidar_ratio_std_sr": lr_std,
        "kfs_branch": kfs_branch,
        "kfs_branch_block": kfs_branch_block,
        "gluing_success_flag": time_gluing_success,
        "gluing_fallback_flag": time_gluing_fallback,
        "gluing_split_altitude_m": time_gluing_split,
        "gluing_start_altitude_m": time_gluing_start,
        "gluing_stop_altitude_m": time_gluing_stop,
        "gluing_slope": time_gluing_slope,
        "gluing_intercept": time_gluing_intercept,
        "gluing_correlation": time_gluing_correlation,
        "gluing_success_flag_block": gluing_success_block,
        "gluing_fallback_flag_block": gluing_fallback_block,
        "gluing_split_altitude_m_block": gluing_split_block,
        "gluing_start_altitude_m_block": gluing_start_block,
        "gluing_stop_altitude_m_block": gluing_stop_block,
        "gluing_slope_block": gluing_slope_block,
        "gluing_intercept_block": gluing_intercept_block,
        "gluing_correlation_block": gluing_correlation_block,
        "gluing_source": source,
        "analog_channel": analog_ch,
        "photon_channel": photon_ch,
    }


def _build_level2_dataset(ds_l1: xr.Dataset, results: list[dict[str, Any]], altitude_m: np.ndarray, source_file: Path, config: Mapping[str, Any]) -> xr.Dataset:
    """Build an xarray Level 2 dataset from wavelength processing results."""
    wavelengths = np.asarray([result["wavelength"] for result in results], dtype=np.int32)
    time_values = ds_l1["time"].values
    block_time = np.asarray(results[0]["block_time"], dtype="datetime64[ns]")
    coords = {"time": time_values, "block_time": block_time, "wavelength": wavelengths, "altitude": altitude_m}

    def stack(name: str) -> np.ndarray:
        return np.stack([np.asarray(result[name], dtype=np.float64) for result in results], axis=0)

    def stack_time(name: str) -> np.ndarray:
        return np.stack([np.asarray(result[name], dtype=np.float64) for result in results], axis=1)

    def stack_block(name: str) -> np.ndarray:
        return np.stack([np.asarray(result[name], dtype=np.float64) for result in results], axis=1)

    def vector(name: str) -> np.ndarray:
        return np.asarray([result[name] for result in results], dtype=np.float64)

    kfs_mode = _get_kfs_mode(config)
    ds_l2 = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), stack("molecular_backscatter")),
            "molecular_extinction": (("wavelength", "altitude"), stack("molecular_extinction")),
            "molecular_transmission": (("wavelength", "altitude"), stack("molecular_transmission")),
            "simulated_molecular_signal": (("wavelength", "altitude"), stack("simulated_molecular_signal")),
            "simulated_molecular_range_corrected_signal": (("wavelength", "altitude"), stack("simulated_molecular_range_corrected_signal")),
            "scaled_molecular_range_corrected_signal": (("wavelength", "altitude"), stack("scaled_molecular_range_corrected_signal")),
            "scaled_molecular_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block("scaled_molecular_range_corrected_signal_block")),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal")),
            "glued_range_corrected_signal_error": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal_error")),
            "glued_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block("glued_range_corrected_signal_block")),
            "glued_range_corrected_signal_error_block": (("block_time", "wavelength", "altitude"), stack_block("glued_range_corrected_signal_error_block")),
            "glued_range_corrected_signal_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_mean")),
            "glued_range_corrected_signal_error_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_error_mean")),
            "scattering_ratio_mean": (("wavelength", "altitude"), stack("scattering_ratio_mean")),
            "scattering_ratio_block": (("block_time", "wavelength", "altitude"), stack_block("scattering_ratio_block")),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), stack("aerosol_backscatter")),
            "aerosol_backscatter_mean_error": (("wavelength", "altitude"), stack("aerosol_backscatter_error")),
            "aerosol_extinction_mean": (("wavelength", "altitude"), stack("aerosol_extinction")),
            "aerosol_extinction_mean_error": (("wavelength", "altitude"), stack("aerosol_extinction_error")),
            "aerosol_backscatter": (("wavelength", "altitude"), stack("aerosol_backscatter")),
            "aerosol_backscatter_error": (("wavelength", "altitude"), stack("aerosol_backscatter_error")),
            "aerosol_extinction": (("wavelength", "altitude"), stack("aerosol_extinction")),
            "aerosol_extinction_error": (("wavelength", "altitude"), stack("aerosol_extinction_error")),
            "aerosol_backscatter_block": (("block_time", "wavelength", "altitude"), stack_block("aerosol_backscatter_block")),
            "aerosol_backscatter_error_block": (("block_time", "wavelength", "altitude"), stack_block("aerosol_backscatter_error_block")),
            "aerosol_extinction_block": (("block_time", "wavelength", "altitude"), stack_block("aerosol_extinction_block")),
            "aerosol_extinction_error_block": (("block_time", "wavelength", "altitude"), stack_block("aerosol_extinction_error_block")),
            "valid_retrieval_block_flag": (("block_time", "wavelength"), stack_block("valid_retrieval_block_flag").astype(np.int8)),
            "rayleigh_reference_altitude_m": (("wavelength",), vector("rayleigh_reference_altitude_m")),
            "rayleigh_reference_start_altitude_m": (("wavelength",), vector("rayleigh_reference_start_altitude_m")),
            "rayleigh_reference_stop_altitude_m": (("wavelength",), vector("rayleigh_reference_stop_altitude_m")),
            "rayleigh_reference_valid_bins": (("wavelength",), vector("rayleigh_reference_valid_bins")),
            "rayleigh_reference_success_flag": (("wavelength",), vector("rayleigh_reference_success_flag").astype(np.int8)),
            "rayleigh_reference_relative_slope": (("wavelength",), vector("rayleigh_reference_relative_slope")),
            "rayleigh_reference_relative_variance": (("wavelength",), vector("rayleigh_reference_relative_variance")),
            "rayleigh_reference_valid_fraction": (("wavelength",), vector("rayleigh_reference_valid_fraction")),
            "rayleigh_calibration_factor": (("wavelength",), vector("rayleigh_calibration_factor")),
            "rayleigh_calibration_intercept": (("wavelength",), vector("rayleigh_calibration_intercept")),
            "rayleigh_reference_altitude_m_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_altitude_m_block")),
            "rayleigh_reference_start_altitude_m_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_start_altitude_m_block")),
            "rayleigh_reference_stop_altitude_m_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_stop_altitude_m_block")),
            "rayleigh_reference_valid_bins_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_valid_bins_block")),
            "rayleigh_reference_success_flag_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_success_flag_block").astype(np.int8)),
            "rayleigh_reference_relative_slope_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_relative_slope_block")),
            "rayleigh_reference_relative_variance_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_relative_variance_block")),
            "rayleigh_reference_valid_fraction_block": (("block_time", "wavelength"), stack_block("rayleigh_reference_valid_fraction_block")),
            "rayleigh_calibration_factor_block": (("block_time", "wavelength"), stack_block("rayleigh_calibration_factor_block")),
            "rayleigh_calibration_intercept_block": (("block_time", "wavelength"), stack_block("rayleigh_calibration_intercept_block")),
            "lidar_ratio_assumed_sr": (("wavelength",), vector("lidar_ratio_assumed_sr")),
            "lidar_ratio_std_sr": (("wavelength",), vector("lidar_ratio_std_sr")),
            "kfs_branch": (("wavelength", "altitude"), stack("kfs_branch").astype(np.int8)),
            "kfs_branch_block": (("block_time", "wavelength", "altitude"), stack_block("kfs_branch_block").astype(np.int8)),
            "gluing_success_flag": (("time", "wavelength"), stack_time("gluing_success_flag").astype(np.int8)),
            "gluing_fallback_flag": (("time", "wavelength"), stack_time("gluing_fallback_flag").astype(np.int8)),
            "gluing_split_altitude_m": (("time", "wavelength"), stack_time("gluing_split_altitude_m")),
            "gluing_start_altitude_m": (("time", "wavelength"), stack_time("gluing_start_altitude_m")),
            "gluing_stop_altitude_m": (("time", "wavelength"), stack_time("gluing_stop_altitude_m")),
            "gluing_slope": (("time", "wavelength"), stack_time("gluing_slope")),
            "gluing_intercept": (("time", "wavelength"), stack_time("gluing_intercept")),
            "gluing_correlation": (("time", "wavelength"), stack_time("gluing_correlation")),
            "gluing_success_flag_block": (("block_time", "wavelength"), stack_block("gluing_success_flag_block").astype(np.int8)),
            "gluing_fallback_flag_block": (("block_time", "wavelength"), stack_block("gluing_fallback_flag_block").astype(np.int8)),
            "gluing_split_altitude_m_block": (("block_time", "wavelength"), stack_block("gluing_split_altitude_m_block")),
            "gluing_start_altitude_m_block": (("block_time", "wavelength"), stack_block("gluing_start_altitude_m_block")),
            "gluing_stop_altitude_m_block": (("block_time", "wavelength"), stack_block("gluing_stop_altitude_m_block")),
            "gluing_slope_block": (("block_time", "wavelength"), stack_block("gluing_slope_block")),
            "gluing_intercept_block": (("block_time", "wavelength"), stack_block("gluing_intercept_block")),
            "gluing_correlation_block": (("block_time", "wavelength"), stack_block("gluing_correlation_block")),
        },
        coords=coords,
        attrs=dict(ds_l1.attrs),
    )
    ds_l2["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station"})
    ds_l2["wavelength"].attrs.update({"units": "nm"})
    ds_l2["valid_retrieval_block_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "invalid valid", "description": "Block passed both gluing and Rayleigh-reference QA and was used in mean optical products."})
    ds_l2["scattering_ratio_mean"].attrs.update({"units": "1", "description": "Mean of valid block scattering ratios."})
    ds_l2["scattering_ratio_block"].attrs.update({"units": "1", "description": "Block scattering ratio from block-mean glued RCS and block-scaled molecular RCS."})
    ds_l2["rayleigh_calibration_intercept"].attrs.update({"description": "Median intercept from free linear Rayleigh diagnostic fit. The main calibration factor is constrained through the origin."})
    ds_l2["kfs_branch"].attrs.update({"flag_values": "0, 1, 2, 3", "flag_meanings": "invalid backward_below_reference reference_window forward_above_reference_experimental", "description": "KFS/Fernald-Sasano retrieval branch by altitude. Above-reference two-sided retrieval is experimental and noise-sensitive."})
    ds_l2["gluing_success_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    ds_l2["gluing_fallback_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used photon_counting_fallback_used"})
    ds_l2["gluing_success_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    ds_l2["gluing_fallback_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used photon_counting_fallback_used"})
    ds_l2["gluing_start_altitude_m"].attrs.update({"units": "m", "description": "Start altitude of analog/photon-counting fade-in/fade-out gluing window."})
    ds_l2["gluing_stop_altitude_m"].attrs.update({"units": "m", "description": "Stop altitude of analog/photon-counting fade-in/fade-out gluing window."})
    ds_l2["rayleigh_reference_success_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed passed", "description": "Whether at least one block Rayleigh reference passed QA."})
    ds_l2["rayleigh_reference_success_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed passed", "description": "Whether the block Rayleigh reference passed slope, variance, valid-fraction and positive-calibration checks."})
    ds_l2["rayleigh_reference_relative_slope"].attrs.update({"units": "1", "description": "Median relative ratio change across valid block Rayleigh reference windows."})
    ds_l2["rayleigh_reference_relative_variance"].attrs.update({"units": "1", "description": "Median variance of measured/molecular ratio normalized by squared mean ratio."})
    ds_l2["rayleigh_reference_valid_fraction"].attrs.update({"units": "1", "description": "Median fraction of finite positive bins in valid Rayleigh reference windows."})
    fit_cfg = _get_molecular_fit_config(config)
    ds_l2.attrs.update(
        {
            "Processing_level": "Level 2: LEBEAR block-based optical inversion",
            "Pipeline": "MILGRAU/LEBEAR",
            "Input_Level1_File": source_file.name,
            "LEBEAR_Mode": "block_mean_gluing_rayleigh_kfs",
            "LEBEAR_Block_Average_Minutes": _get_block_average_minutes(config),
            "KFS_Mode": kfs_mode,
            "KFS_Mode_Description": _kfs_mode_description(kfs_mode),
            "Molecular_Rayleigh_Method": "Bucholtz-style Rayleigh scattering with angular backscatter at 180 degrees.",
            "Rayleigh_Calibration_Method": "Block-wise multiplicative fit constrained through origin; free intercept retained as a background diagnostic.",
            "Gluing_Error_Propagation": "Weighted one-sigma propagation across fade window: sigma² = w_an²(slope sigma_an)² + w_pc² sigma_pc².",
            "Rayleigh_Reference_Max_Relative_Slope": float(fit_cfg["max_relative_slope"]),
            "Rayleigh_Reference_Max_Relative_Variance": float(fit_cfg["max_relative_variance"]),
            "Rayleigh_Reference_Min_Valid_Fraction": float(fit_cfg["min_valid_fraction"]),
            "Molecular_sources": ";".join(str(result["molecular_source"]) for result in results),
            "Gluing_sources": ";".join(str(result["gluing_source"]) for result in results),
            "Analog_channels": ";".join(str(result["analog_channel"]) for result in results),
            "Photon_channels": ";".join(str(result["photon_channel"]) for result in results),
        }
    )
    return ds_l2


def process_single_level1_file(nc_file: str | Path, config: Mapping[str, Any], logger: logging.Logger) -> str:
    """Process one Level 1 file into a Level 2 optical product."""
    nc_path = Path(nc_file)
    try:
        with xr.open_dataset(nc_path) as ds_l1:
            ds_l1.load()
            validate_level1_contract(ds_l1)
            altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
            if np.nanmax(altitude_m) <= 100.0:
                altitude_m = altitude_m * 1000.0
            wavelengths = _get_wavelengths_to_process(config)
            results = []
            for wavelength in wavelengths:
                try:
                    results.append(_process_wavelength(ds_l1, wavelength, altitude_m, config, logger))
                except Exception as exc:
                    logger.warning(f"  -> Skipping {wavelength} nm in {nc_path.name}: {exc}")
            if not results:
                raise RuntimeError("No wavelength could be processed by LEBEAR.")
            ds_l2 = _build_level2_dataset(ds_l1, results, altitude_m, nc_path, config)
            validate_level2_contract(ds_l2)

        output_path = level2_output_path(nc_path)
        ensure_directories(output_path.parent)
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds_l2.data_vars if ds_l2[var].ndim > 0}
        ds_l2.to_netcdf(output_path, encoding=encoding)
        logger.info(f"  -> [OK] Level 2 NetCDF generated: {output_path}")

        qa_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
        if bool(qa_cfg.get("enabled", True)):
            qa_dir = output_path.parent / "level2_qa"
            with xr.open_dataset(output_path) as ds_saved, xr.open_dataset(nc_path) as ds_l1_saved:
                ds_saved.load()
                ds_l1_saved.load()
                generated = plot_all_level2_qa(ds_l2=ds_saved, output_folder=qa_dir, file_name_prefix=output_path.name.replace(LEVEL2_SUFFIX, ""), config=dict(config), root_dir=Path.cwd(), ds_l1=ds_l1_saved)
            logger.info(f"  -> Generated {len(generated)} Level 2 QA plot(s).")
        return f"[OK] {nc_path.name} Level 2 generated successfully: {output_path}"
    except Exception:
        return f"[FAILED] {nc_path}:\n{traceback.format_exc()}"


def process_level_2(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover Level 1 files and process them into Level 2 products."""
    files = discover_level1_files(config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        return

    incremental = _incremental_enabled(config)
    files_to_process = []
    skipped_count = 0
    for file_path in files:
        output_path = level2_output_path(file_path)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 2 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 1 files require Level 2 processing. Skipped {skipped_count} existing products.")
        return

    logger.info(f"Found {len(files_to_process)} Level 1 files for LEBEAR ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_level1_file(file_path, config, logger))
