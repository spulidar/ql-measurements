"""LEBEAR Level 2 optical inversion pipeline.

LEBEAR converts Level 1 Range Corrected Signal products into first-pass Level 2
mean optical products. This initial implementation is intentionally conservative:
for each configured wavelength it performs profile-by-profile Analog/Photon
Counting gluing, computes a molecular profile, selects a Rayleigh reference
region, and runs a mean-profile KFS/Fernald-Sasano inversion with Monte Carlo
uncertainty.
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
from milgrau.physics.molecular import calculate_molecular_profile, find_optimal_reference_altitude
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
        "window_size": int(gluing_cfg.get("window_length_bins", 150)),
        "min_corr": float(gluing_cfg.get("correlation_threshold", 0.95)),
        "search_min_idx": int(gluing_cfg.get("search_min_idx", 400)),
        "search_max_idx": int(gluing_cfg.get("search_max_idx", 1000)),
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
        "ref_alt_max_m": float(fit_cfg.get("ref_alt_max_m", 9000.0)),
        "ref_window_bins": int(fit_cfg.get("ref_window_bins", 50)),
    }


def _build_thermodynamic_profile(ds_l1: xr.Dataset, altitude_agl_m: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Build pressure and temperature profiles on the lidar altitude grid.

    Level 1 altitude is stored as AGL. Radiosonde altitude is usually ASL/MSL, so
    the lidar grid is shifted by station altitude before interpolation.
    """
    site_cfg = config.get("site", {})
    station_altitude_m = float(site_cfg.get("station_altitude_m", config.get("physics", {}).get("station_altitude_m", 0.0)))
    altitude_asl_m = altitude_agl_m + station_altitude_m

    standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)
    if {"Radiosonde_Temperature_K", "Radiosonde_Pressure_hPa", "radiosonde_altitude"}.issubset(
        set(ds_l1.variables) | set(ds_l1.coords)
    ):
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
            result = slide_glue_signals(
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
            glued_profile, split_point, slope_i, intercept_i, diagnostics = result
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
            f"  -> {wavelength_nm} nm gluing success: {100.0 * success_flag.sum() / max(n_time, 1):.1f}% "
            f"({analog_ch} + {photon_ch})."
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


def _calibrate_rayleigh(rcs_mean: np.ndarray, beta_mol: np.ndarray, ref_idx: int) -> float:
    """Estimate the multiplicative factor mapping molecular backscatter to RCS."""
    if not np.isfinite(rcs_mean[ref_idx]) or not np.isfinite(beta_mol[ref_idx]) or beta_mol[ref_idx] <= 0.0:
        return np.nan
    return float(rcs_mean[ref_idx] / beta_mol[ref_idx])


def _process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Process one wavelength into mean-profile Level 2 arrays and diagnostics."""
    gluing = _glue_wavelength_profiles(ds_l1, wavelength_nm, altitude_m, config, logger)
    pressure_hpa, temperature_k, molecular_source = _build_thermodynamic_profile(ds_l1, altitude_m, config)
    beta_mol, alpha_mol = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)

    glued = gluing["glued"]  # type: ignore[assignment]
    glued_error = gluing["glued_error"]  # type: ignore[assignment]
    glued_mean = np.nanmean(glued, axis=0)
    glued_error_mean = _error_of_mean(glued_error)

    fit_cfg = _get_molecular_fit_config(config)
    ref_idx = find_optimal_reference_altitude(
        rcs=glued_mean,
        beta_mol=beta_mol,
        altitude=altitude_m,
        min_alt=fit_cfg["ref_alt_min_m"],
        max_alt=fit_cfg["ref_alt_max_m"],
        window_size=fit_cfg["ref_window_bins"],
        altitude_units="m",
    )
    calibration_factor = _calibrate_rayleigh(glued_mean, beta_mol, ref_idx)
    lr_base, lr_std = _get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    inv_cfg = config.get("inversion", {})
    beta_mean, beta_std, alpha_mean, alpha_std = kfs_inversion_monte_carlo(
        rcs=glued_mean,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=lr_base,
        lr_std=lr_std,
        ref_idx=ref_idx,
        n_iterations=int(inv_cfg.get("monte_carlo_iterations", 300)),
        rcs_error=glued_error_mean,
        beta_ref_relative_std=float(inv_cfg.get("beta_ref_relative_std", 0.10)),
        aerosol_ref_fraction=float(inv_cfg.get("aerosol_ref_fraction", 0.0)),
        altitude_units="m",
        min_lidar_ratio=float(inv_cfg.get("min_lidar_ratio_sr", 10.0)),
        allow_negative_aerosol=bool(inv_cfg.get("allow_negative_aerosol", False)),
        seed=inv_cfg.get("random_seed"),
    )

    return {
        "wavelength": wavelength_nm,
        "molecular_source": molecular_source,
        "molecular_backscatter": beta_mol,
        "molecular_extinction": alpha_mol,
        "glued_range_corrected_signal": glued,
        "glued_range_corrected_signal_error": glued_error,
        "glued_range_corrected_signal_mean": glued_mean,
        "glued_range_corrected_signal_error_mean": glued_error_mean,
        "aerosol_backscatter": beta_mean,
        "aerosol_backscatter_error": beta_std,
        "aerosol_extinction": alpha_mean,
        "aerosol_extinction_error": alpha_std,
        "rayleigh_reference_altitude_m": float(altitude_m[ref_idx]),
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
    coords = {"time": time_values, "wavelength": wavelengths, "altitude": altitude_m}

    def stack(name: str) -> np.ndarray:
        return np.stack([np.asarray(result[name], dtype=np.float64) for result in results], axis=0)

    def stack_time(name: str) -> np.ndarray:
        return np.stack([np.asarray(result[name], dtype=np.float64) for result in results], axis=1)

    def vector(name: str) -> np.ndarray:
        return np.asarray([result[name] for result in results], dtype=np.float64)

    ds_l2 = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), stack("molecular_backscatter")),
            "molecular_extinction": (("wavelength", "altitude"), stack("molecular_extinction")),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal")),
            "glued_range_corrected_signal_error": (("time", "wavelength", "altitude"), stack_time("glued_range_corrected_signal_error")),
            "glued_range_corrected_signal_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_mean")),
            "glued_range_corrected_signal_error_mean": (("wavelength", "altitude"), stack("glued_range_corrected_signal_error_mean")),
            "aerosol_backscatter": (("wavelength", "altitude"), stack("aerosol_backscatter")),
            "aerosol_backscatter_error": (("wavelength", "altitude"), stack("aerosol_backscatter_error")),
            "aerosol_extinction": (("wavelength", "altitude"), stack("aerosol_extinction")),
            "aerosol_extinction_error": (("wavelength", "altitude"), stack("aerosol_extinction_error")),
            "rayleigh_reference_altitude_m": (("wavelength",), vector("rayleigh_reference_altitude_m")),
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
            "Processing_level": "Level 2: LEBEAR mean-profile optical inversion",
            "Pipeline": "MILGRAU/LEBEAR",
            "Input_Level1_File": source_file.name,
            "LEBEAR_Mode": "mean_profile_backward_kfs",
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
