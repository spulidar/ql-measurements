"""LEBEAR Level 2 optical inversion pipeline.

LEBEAR converts Level 1 Range Corrected Signal products into first-pass Level 2
optical products. This implementation produces both 15-minute block products
and a full-period mean product, using Analog/Photon Counting gluing, Rayleigh
molecular calibration, and KFS/Fernald-Sasano inversion.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.filesystem import ensure_directories
from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.physics.gluing import slide_glue_signals
from milgrau.physics.kfs import kfs_inversion_monte_carlo
from milgrau.physics.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
    find_optimal_reference_altitude,
    robust_rayleigh_calibration_factor,
)
from milgrau.visualization.level2_qa import plot_all_level2_qa


LEVEL2_SUFFIX = "_level2_optical.nc"
REQUIRED_LEVEL1_VARIABLES = (
    "corrected_signal",
    "corrected_signal_error",
    "range_corrected_signal",
    "range_corrected_signal_error",
)


def discover_level1_files(config: Mapping[str, Any], root_dir: str | Path | None = None) -> list[Path]:
    """Discover Level 1 RCS NetCDF files available for LEBEAR processing."""
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    base_data_folder = root_path / config["directories"]["processed_data"]
    return sorted(base_data_folder.rglob("*_level1_rcs.nc"))


def validate_level1_contract(ds_l1: xr.Dataset) -> None:
    """Validate the Level 1 variables required by LEBEAR processing."""
    missing = [name for name in REQUIRED_LEVEL1_VARIABLES if name not in ds_l1]
    if missing:
        raise KeyError(f"Level 1 file lacks required variable(s): {missing}")
    if "altitude" not in ds_l1.coords:
        raise KeyError("Level 1 file lacks altitude coordinate.")
    if "channel" not in ds_l1.coords:
        raise KeyError("Level 1 file lacks channel coordinate.")


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
        "window_size": int(gluing_cfg.get("window_length_bins", 120)),
        "min_corr": float(gluing_cfg.get("correlation_threshold", 0.90)),
        "search_min_idx": int(gluing_cfg.get("search_min_idx", 80)),
        "search_max_idx": int(gluing_cfg.get("search_max_idx", 500)),
        "intercept_threshold": float(gluing_cfg.get("intercept_threshold", 5.0)),
        "gaussian_threshold": float(gluing_cfg.get("gaussian_threshold", 2.0)),
        "minmax_threshold": float(gluing_cfg.get("minmax_threshold", 2.0)),
        "fallback_to_photon_counting": bool(gluing_cfg.get("fallback_to_photon_counting", True)),
    }


def _get_molecular_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return molecular reference configuration with safe defaults."""
    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    return {
        "ref_alt_min_m": float(fit_cfg.get("ref_alt_min_m", 22000.0)),
        "ref_alt_max_m": float(fit_cfg.get("ref_alt_max_m", 28000.0)),
        "ref_window_bins": int(fit_cfg.get("ref_window_bins", 100)),
    }


def _get_kfs_mode(config: Mapping[str, Any]) -> str:
    """Return the configured KFS integration mode."""
    mode = str(config.get("inversion", {}).get("kfs_mode", "two_sided")).strip().lower()
    if mode not in {"backward", "two_sided"}:
        return "backward"
    return mode


def _get_temporal_average_minutes(config: Mapping[str, Any]) -> int:
    """Return the temporal averaging window in minutes for block products."""
    return max(int(config.get("inversion", {}).get("temporal_average_minutes", 15)), 1)


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


def _glue_wavelength_profiles(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, np.ndarray | str | None]:
    """Glue all time profiles for one wavelength and return signal matrices plus diagnostics."""
    analog_ch, photon_ch = _infer_channel_pair(ds_l1, wavelength_nm)
    if analog_ch is None and photon_ch is None:
        raise ValueError(f"No channel found for wavelength {wavelength_nm} nm.")

    signal_da = ds_l1["range_corrected_signal"]
    error_da = ds_l1["range_corrected_signal_error"]
    n_time = ds_l1.sizes.get("time", 1)
    n_alt = altitude_m.size
    glued = np.full((n_time, n_alt), np.nan, dtype=np.float64)
    glued_error = np.full((n_time, n_alt), np.nan, dtype=np.float64)
    success_flag = np.zeros(n_time, dtype=np.int8)
    split_altitude_m = np.full(n_time, np.nan, dtype=np.float64)
    slope = np.full(n_time, np.nan, dtype=np.float64)
    intercept = np.full(n_time, np.nan, dtype=np.float64)
    correlation = np.full(n_time, np.nan, dtype=np.float64)
    gluing_cfg = _get_gluing_config(config)

    if analog_ch is not None and photon_ch is not None:
        analog_signal = signal_da.sel(channel=analog_ch).values.astype(np.float64)
        photon_signal = signal_da.sel(channel=photon_ch).values.astype(np.float64)
        analog_error = error_da.sel(channel=analog_ch).values.astype(np.float64)
        photon_error = error_da.sel(channel=photon_ch).values.astype(np.float64)

        for time_idx in range(n_time):
            glued_profile, split_point, slope_i, intercept_i, diagnostics = slide_glue_signals(
                analog_sig=analog_signal[time_idx, :],
                pc_sig=photon_signal[time_idx, :],
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
            glued[time_idx, :] = glued_profile
            slope[time_idx] = slope_i
            intercept[time_idx] = intercept_i
            correlation[time_idx] = float(diagnostics.get("best_corr", np.nan))
            if split_point >= 0:
                success_flag[time_idx] = 1
                split_altitude_m[time_idx] = float(altitude_m[split_point])
                glued_error[time_idx, :split_point] = np.abs(slope_i) * analog_error[time_idx, :split_point]
                glued_error[time_idx, split_point:] = photon_error[time_idx, split_point:]
            else:
                glued_error[time_idx, :] = photon_error[time_idx, :]

        logger.info(
            f"  -> {wavelength_nm} nm gluing success: {100.0 * success_flag.sum() / max(n_time, 1):.1f}% ({analog_ch} + {photon_ch})."
        )
        source = "analog_photon_glued"
    else:
        fallback_ch = photon_ch or analog_ch
        glued[:, :] = signal_da.sel(channel=fallback_ch).values.astype(np.float64)
        glued_error[:, :] = error_da.sel(channel=fallback_ch).values.astype(np.float64)
        source = f"single_channel_{fallback_ch}"
        logger.warning(f"  -> {wavelength_nm} nm using single-channel fallback: {fallback_ch}.")

    return {
        "analog_channel": analog_ch,
        "photon_channel": photon_ch,
        "source": source,
        "glued": glued,
        "glued_error": glued_error,
        "success_flag": success_flag,
        "split_altitude_m": split_altitude_m,
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
    }


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
    return np.stack([np.nanmean(matrix[group, :], axis=0) for group in groups], axis=0)


def _error_by_groups(error_matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Calculate uncertainty of grouped means from one-sigma profiles."""
    return np.stack([_error_of_mean(error_matrix[group, :]) for group in groups], axis=0)


def _run_kfs_profile(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    ref_idx: int,
    lr_base: float,
    lr_std: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        mode=str(inv_cfg.get("kfs_mode", "two_sided")),
    )


def _process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Process one wavelength into block and mean Level 2 arrays."""
    gluing = _glue_wavelength_profiles(ds_l1, wavelength_nm, altitude_m, config, logger)
    pressure_hpa, temperature_k, molecular_source = _build_thermodynamic_profile(ds_l1, altitude_m, config)
    beta_mol, alpha_mol = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, molecular_transmission = calculate_simulated_molecular_signal(beta_mol, alpha_mol, altitude_m)
    safe_altitude = np.where(altitude_m > 0.0, altitude_m, altitude_m[altitude_m > 0.0][0])
    simulated_molecular_rcs = simulated_signal * safe_altitude**2

    glued = gluing["glued"]  # type: ignore[assignment]
    glued_error = gluing["glued_error"]  # type: ignore[assignment]
    glued_mean = np.nanmean(glued, axis=0)
    glued_error_mean = _error_of_mean(glued_error)

    block_minutes = _get_temporal_average_minutes(config)
    block_time, block_groups = _block_groups(ds_l1["time"].values, block_minutes)
    glued_block = _mean_by_groups(glued, block_groups)
    glued_error_block = _error_by_groups(glued_error, block_groups)

    fit_cfg = _get_molecular_fit_config(config)
    ref_idx = find_optimal_reference_altitude(
        rcs=glued_mean,
        beta_mol=simulated_molecular_rcs,
        altitude=altitude_m,
        min_alt=fit_cfg["ref_alt_min_m"],
        max_alt=fit_cfg["ref_alt_max_m"],
        window_size=fit_cfg["ref_window_bins"],
        altitude_units="m",
    )
    calibration_factor, ref_start_m, ref_stop_m, ref_valid_bins = robust_rayleigh_calibration_factor(
        measured_signal=glued_mean,
        simulated_molecular_signal=simulated_molecular_rcs,
        altitude_m=altitude_m,
        reference_center_idx=ref_idx,
        reference_window_bins=fit_cfg["ref_window_bins"],
    )

    lr_base, lr_std = _get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    beta_mean, beta_std, alpha_mean, alpha_std = _run_kfs_profile(
        glued_mean,
        glued_error_mean,
        altitude_m,
        beta_mol,
        ref_idx,
        lr_base,
        lr_std,
        config,
    )

    beta_block = []
    beta_block_std = []
    alpha_block = []
    alpha_block_std = []
    for block_idx in range(glued_block.shape[0]):
        b_mean, b_std, a_mean, a_std = _run_kfs_profile(
            glued_block[block_idx, :],
            glued_error_block[block_idx, :],
            altitude_m,
            beta_mol,
            ref_idx,
            lr_base,
            lr_std,
            config,
        )
        beta_block.append(b_mean)
        beta_block_std.append(b_std)
        alpha_block.append(a_mean)
        alpha_block_std.append(a_std)

    return {
        "wavelength": wavelength_nm,
        "block_time": block_time,
        "molecular_source": molecular_source,
        "molecular_backscatter": beta_mol,
        "molecular_extinction": alpha_mol,
        "molecular_transmission": molecular_transmission,
        "simulated_molecular_signal": simulated_signal,
        "simulated_molecular_range_corrected_signal": simulated_molecular_rcs,
        "scaled_molecular_range_corrected_signal": simulated_molecular_rcs * calibration_factor,
        "glued_range_corrected_signal": glued,
        "glued_range_corrected_signal_error": glued_error,
        "glued_range_corrected_signal_block": glued_block,
        "glued_range_corrected_signal_error_block": glued_error_block,
        "glued_range_corrected_signal_mean": glued_mean,
        "glued_range_corrected_signal_error_mean": glued_error_mean,
        "aerosol_backscatter": beta_mean,
        "aerosol_backscatter_error": beta_std,
        "aerosol_extinction": alpha_mean,
        "aerosol_extinction_error": alpha_std,
        "aerosol_backscatter_block": np.stack(beta_block, axis=0),
        "aerosol_backscatter_error_block": np.stack(beta_block_std, axis=0),
        "aerosol_extinction_block": np.stack(alpha_block, axis=0),
        "aerosol_extinction_error_block": np.stack(alpha_block_std, axis=0),
        "rayleigh_reference_altitude_m": float(altitude_m[ref_idx]),
        "rayleigh_reference_start_altitude_m": ref_start_m,
        "rayleigh_reference_stop_altitude_m": ref_stop_m,
        "rayleigh_reference_valid_bins": ref_valid_bins,
        "rayleigh_calibration_factor": calibration_factor,
        "lidar_ratio_assumed_sr": lr_base,
        "lidar_ratio_std_sr": lr_std,
        "gluing_success_flag": gluing["success_flag"],
        "gluing_split_altitude_m": gluing["split_altitude_m"],
        "gluing_slope": gluing["slope"],
        "gluing_intercept": gluing["intercept"],
        "gluing_correlation": gluing["correlation"],
        "gluing_source": gluing["source"],
        "analog_channel": gluing["analog_channel"],
        "photon_channel": gluing["photon_channel"],
    }


def _build_level2_dataset(ds_l1: xr.Dataset, results: list[dict[str, Any]], altitude_m: np.ndarray, source_file: Path) -> xr.Dataset:
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

    ds_l2 = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), stack("molecular_backscatter")),
            "molecular_extinction": (("wavelength", "altitude"), stack("molecular_extinction")),
            "molecular_transmission": (("wavelength", "altitude"), stack("molecular_transmission")),
            "simulated_molecular_signal": (("wavelength", "altitude"), stack("simulated_molecular_signal")),
            "simulated_molecular_range_corrected_signal": (("wavelength", "altitude"), stack("simulated_molecular_range_corrected_signal")),
            "scaled_molecular_range_corrected_signal": (("wavelength", "altitude"), stack("scaled_molecular_range_corrected_signal")),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal")),
            "glued_range_corrected_signal_error": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal_error")),
            "glued_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block("glued_range_corrected_signal_block")),
            "glued_range_corrected_signal_error_block": (("block_time", "wavelength", "altitude"), stack_block("glued_range_corrected_signal_error_block")),
            "glued_range_corrected_signal_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_mean")),
            "glued_range_corrected_signal_error_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_error_mean")),
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
            "rayleigh_reference_altitude_m": (("wavelength",), vector("rayleigh_reference_altitude_m")),
            "rayleigh_reference_start_altitude_m": (("wavelength",), vector("rayleigh_reference_start_altitude_m")),
            "rayleigh_reference_stop_altitude_m": (("wavelength",), vector("rayleigh_reference_stop_altitude_m")),
            "rayleigh_reference_valid_bins": (("wavelength",), vector("rayleigh_reference_valid_bins")),
            "rayleigh_calibration_factor": (("wavelength",), vector("rayleigh_calibration_factor")),
            "lidar_ratio_assumed_sr": (("wavelength",), vector("lidar_ratio_assumed_sr")),
            "lidar_ratio_std_sr": (("wavelength",), vector("lidar_ratio_std_sr")),
            "gluing_success_flag": (("time", "wavelength"), stack_time("gluing_success_flag").astype(np.int8)),
            "gluing_split_altitude_m": (("time", "wavelength"), stack_time("gluing_split_altitude_m")),
            "gluing_slope": (("time", "wavelength"), stack_time("gluing_slope")),
            "gluing_intercept": (("time", "wavelength"), stack_time("gluing_intercept")),
            "gluing_correlation": (("time", "wavelength"), stack_time("gluing_correlation")),
        },
        coords=coords,
        attrs=dict(ds_l1.attrs),
    )
    ds_l2["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station"})
    ds_l2["wavelength"].attrs.update({"units": "nm"})
    ds_l2.attrs.update(
        {
            "Processing_level": "Level 2: LEBEAR block and mean optical inversion",
            "Pipeline": "MILGRAU/LEBEAR",
            "Input_Level1_File": source_file.name,
            "LEBEAR_Mode": "block_and_mean_kfs",
            "KFS_Mode": "configured",
            "Molecular_sources": ";".join(str(result["molecular_source"]) for result in results),
            "Gluing_sources": ";".join(str(result["gluing_source"]) for result in results),
            "Analog_channels": ";".join(str(result["analog_channel"]) for result in results),
            "Photon_channels": ";".join(str(result["photon_channel"]) for result in results),
        }
    )
    return ds_l2


def _level2_output_path(nc_file: Path, config: Mapping[str, Any]) -> Path:
    """Return the Level 2 output path for one Level 1 file."""
    stem = nc_file.name.replace("_level1_rcs.nc", "")
    return nc_file.parent / f"{stem}{LEVEL2_SUFFIX}"


def process_single_level1_file(nc_file: str | Path, config: Mapping[str, Any], logger: logging.Logger) -> str:
    """Process one Level 1 file into an initial Level 2 optical product."""
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
            ds_l2 = _build_level2_dataset(ds_l1, results, altitude_m, nc_path)

        output_path = _level2_output_path(nc_path, config)
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
                generated = plot_all_level2_qa(
                    ds_l2=ds_saved,
                    output_folder=qa_dir,
                    file_name_prefix=output_path.name.replace(LEVEL2_SUFFIX, ""),
                    config=dict(config),
                    root_dir=Path.cwd(),
                    ds_l1=ds_l1_saved,
                )
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
    logger.info(f"Found {len(files)} Level 1 files for LEBEAR.")
    for file_path in files:
        logger.info(process_single_level1_file(file_path, config, logger))
