"""LIPANCORA Level 1 pipeline orchestration."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.filesystem import ensure_directories
from milgrau.io.radiosonde import fetch_wyoming_radiosonde
from milgrau.physics.corrections import apply_instrumental_corrections
from milgrau.physics.pbl import calculate_pbl_height_gradient
from milgrau.physics.tropopause import calculate_tropopause_heights


def _incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def level1_output_path(nc_file: str | Path, config: Mapping[str, Any]) -> Path:
    """Return the Level 1 output path for one Level 0 NetCDF file."""
    nc_path = Path(nc_file)
    stem = nc_path.stem
    return Path.cwd() / config["directories"]["processed_data"] / stem[:4] / stem[4:6] / stem / f"{stem}_level1_rcs.nc"


def _finite_or_fill(value: Any, fill_value: float = -999.0) -> float:
    """Convert a numeric value to float, replacing invalid values by a fill value."""
    try:
        value = float(value)
        return value if np.isfinite(value) else float(fill_value)
    except Exception:
        return float(fill_value)


def _get_channel_constant(
    channels_config: Mapping[str, Sequence[float]],
    ch_name: str,
    logger: logging.Logger,
) -> tuple[float, int, float]:
    """Return instrumental constants for one channel."""
    if ch_name not in channels_config:
        logger.warning(
            f"  -> Channel {ch_name} is missing from physics.channels. Using neutral correction constants."
        )
    deadtime, shift, bg_offset = channels_config.get(ch_name, [0.0, 0, 0.0])
    return float(deadtime), int(shift), float(bg_offset)


def load_and_prepare_level0(nc_path: str | Path, logger: logging.Logger) -> tuple[xr.Dataset, np.ndarray]:
    """Load one Level 0 NetCDF file and standardize its coordinates."""
    try:
        ds = xr.open_dataset(nc_path)
        ds.load()
        required_vars = ["Raw_Data_Start_Time", "Raw_Data_Range_Resolution", "channel_string", "Raw_Lidar_Data"]
        missing = [var for var in required_vars if var not in ds]
        if missing:
            raise KeyError(f"Level 0 file is missing required variables: {missing}")

        time_dt = pd.to_datetime(ds["Raw_Data_Start_Time"].values, unit="s")
        ds = ds.assign_coords(time=time_dt)

        dz_values = np.asarray(ds["Raw_Data_Range_Resolution"].values, dtype=float)
        dz_values = dz_values[np.isfinite(dz_values)]
        if dz_values.size == 0:
            raise ValueError("Raw_Data_Range_Resolution contains no finite values.")
        dz = float(dz_values[0])
        if not np.allclose(dz_values, dz, rtol=0.0, atol=1e-6):
            logger.warning(
                "  -> Not all channels have identical range resolution. "
                f"Using the first value: {dz:.6f} m."
            )

        z_arr = np.arange(ds.sizes["points"], dtype=np.float64) * dz
        channel_strings = ds["channel_string"].values.astype(str)
        ds = ds.rename({"points": "altitude", "channels": "channel"})
        ds = ds.assign_coords(altitude=z_arr, channel=channel_strings)
        ds["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station"})
        logger.info(
            f"  -> Level 0 ingestion successful: {ds.sizes.get('time', 0)} profiles, "
            f"{ds.sizes.get('channel', 0)} channels, {ds.sizes.get('altitude', 0)} bins."
        )
        return ds, z_arr
    except Exception as exc:
        logger.error(f"  -> Failed to ingest Level 0 file {nc_path}: {exc}")
        raise


def apply_all_physical_corrections(
    ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Apply Level 1 instrumental corrections to all available channels."""
    channels_config = config.get("physics", {}).get("channels", {})
    c_speed = float(config.get("physics", {}).get("speed_of_light", 299792458.0))
    if len(z_arr) < 2:
        raise ValueError("Altitude grid must contain at least two bins.")
    dz = float(z_arr[1] - z_arr[0])
    if dz <= 0.0 or not np.isfinite(dz):
        raise ValueError(f"Invalid altitude step: {dz}")

    bin_time_us = (2.0 * dz / c_speed) * 1e6
    shots = float(ds.attrs.get("Accumulated_Shots", np.nan))
    if not np.isfinite(shots) or shots <= 0.0:
        raise ValueError(f"Invalid Accumulated_Shots attribute: {shots}")

    z_da = xr.DataArray(z_arr, dims=["range"], attrs={"units": "m"})
    channel_datasets = []
    status_records = []
    logger.info("  -> Running instrumental corrections channel-by-channel...")

    for ch_idx, ch_name in enumerate(ds.channel.values.astype(str)):
        dark_current_used = False
        try:
            sig = ds["Raw_Lidar_Data"].isel(channel=ch_idx)
            bg_low = float(ds["Background_Low"].isel(channel=ch_idx))
            bg_high = float(ds["Background_High"].isel(channel=ch_idx))
            bg_mask = (ds["altitude"] >= bg_low) & (ds["altitude"] <= bg_high)
            if int(bg_mask.sum().values) < 2:
                logger.warning(
                    f"  -> Channel {ch_name}: background mask has fewer than 2 bins ({bg_low:.1f}-{bg_high:.1f} m)."
                )

            deadtime, shift, bg_offset = _get_channel_constant(channels_config, ch_name, logger)
            is_photon = "pc" in ch_name.lower() or "ph" in ch_name.lower()
            dc_prof, dc_err = None, None
            if "Background_Profile" in ds:
                dc_data = ds["Background_Profile"].isel(channel=ch_idx)
                if dc_data.sizes.get("time_bck", 0) > 0:
                    dc_prof = dc_data.mean(dim="time_bck", skipna=True).rename({"altitude": "range"})
                    dc_err = dc_data.std(dim="time_bck", skipna=True).rename({"altitude": "range"}) / np.sqrt(
                        max(ds.sizes.get("time_bck", 1), 1)
                    )
                    dark_current_used = True

            corrected, corrected_error, rcs, rcs_error = apply_instrumental_corrections(
                sig=sig.rename({"altitude": "range"}),
                z_da=z_da,
                shots=shots,
                bin_time_us=bin_time_us,
                deadtime=deadtime,
                shift=shift,
                bg_offset=bg_offset,
                is_photon=is_photon,
                bg_mask=bg_mask.rename({"altitude": "range"}),
                dc_prof=dc_prof,
                dc_err=dc_err,
            )

            ch_ds = xr.Dataset(
                {
                    "corrected_signal": corrected.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "corrected_signal_error": corrected_error.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "range_corrected_signal": rcs.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "range_corrected_signal_error": rcs_error.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                }
            )
            channel_datasets.append(ch_ds)
            status_records.append((ch_name, 1, int(dark_current_used)))
            logger.info(f"  -> Channel {ch_name}: corrected successfully.")
        except Exception as exc:
            status_records.append((ch_name, 0, int(dark_current_used)))
            logger.warning(f"  -> Channel {ch_name} failed during correction: {exc}")

    if not channel_datasets:
        raise RuntimeError("All channels failed during instrumental correction.")

    final_ds = xr.concat(channel_datasets, dim="channel")
    final_ds["corrected_signal"].attrs.update(
        {
            "long_name": "Instrumentally corrected lidar signal",
            "description": "Signal after dark-current, dead-time, bin-shift and background corrections before range correction",
            "units": "channel native corrected units",
        }
    )
    final_ds["corrected_signal_error"].attrs.update(
        {"long_name": "One-sigma uncertainty of instrumentally corrected lidar signal", "units": "channel native corrected units"}
    )
    final_ds["range_corrected_signal"].attrs.update(
        {
            "long_name": "Range Corrected Signal",
            "description": "Instrumentally corrected signal multiplied by range squared",
            "units": "a.u. m^2",
        }
    )
    final_ds["range_corrected_signal_error"].attrs.update(
        {"long_name": "One-sigma uncertainty of Range Corrected Signal", "units": "a.u. m^2"}
    )

    status_map = {name: (ok, dc) for name, ok, dc in status_records}
    final_channels = final_ds.channel.values.astype(str)
    final_ds["channel_correction_success"] = xr.DataArray(
        [status_map.get(ch, (0, 0))[0] for ch in final_channels], dims=["channel"], coords={"channel": final_channels}
    ).astype(np.int8)
    final_ds["dark_current_used"] = xr.DataArray(
        [status_map.get(ch, (0, 0))[1] for ch in final_channels], dims=["channel"], coords={"channel": final_channels}
    ).astype(np.int8)
    final_ds["channel_correction_success"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    final_ds["dark_current_used"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used used"})
    return final_ds


def estimate_pbl_timeseries(
    final_ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Estimate Planetary Boundary Layer height for every time profile."""
    try:
        pbl_channel = next(
            (ch for ch in final_ds.channel.values.astype(str) if "an" in ch.lower() and "532" in ch),
            str(final_ds.channel.values[0]),
        )
        physics_cfg = config.get("physics", {})
        min_search_m = float(physics_cfg.get("pbl_min_search_m", 500.0))
        max_search_m = float(physics_cfg.get("pbl_max_search_m", 4000.0))
        smooth_bins = int(physics_cfg.get("pbl_smooth_bins", 15))
        rcs_matrix = final_ds["range_corrected_signal"].sel(channel=pbl_channel).values
        logger.info(f"  -> Tracking PBL using {pbl_channel} ({min_search_m:.0f}-{max_search_m:.0f} m).")
        pbl_h = [
            calculate_pbl_height_gradient(
                rcs_matrix[t, :], z_arr, min_search_m=min_search_m, max_search_m=max_search_m, smooth_bins=smooth_bins
            )
            for t in range(rcs_matrix.shape[0])
        ]
        final_ds["PBL_Height_km"] = xr.DataArray(pbl_h, dims=["time"], coords={"time": final_ds.time}).astype(np.float32)
        final_ds["PBL_Height_km"].attrs = {
            "units": "km",
            "method": "Gradient method on smoothed RCS",
            "reference_channel": pbl_channel,
            "min_search_m": min_search_m,
            "max_search_m": max_search_m,
            "smooth_bins": smooth_bins,
        }
        return final_ds
    except Exception as exc:
        logger.warning(f"  -> PBL tracking failed: {exc}")
        return final_ds


def integrate_thermodynamics(final_ds: xr.Dataset, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Add radiosonde thermodynamics and WMO tropopause diagnostics to Level 1."""
    try:
        dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time) // 2])
        station_id = str(config.get("radiosonde", {}).get("station_id", config.get("location", {}).get("station_id", "83779")))
        df_radio = fetch_wyoming_radiosonde(dt_utc, station_id, logger)
        if df_radio is None or df_radio.empty:
            logger.warning("  -> Radiosonde unavailable. Level 1 will keep surface-only thermodynamics.")
            final_ds.attrs.update(
                {"radiosonde_station_id": station_id, "radiosonde_available": "false", "tropopause_cpt_km": -999.0, "tropopause_lrt_km": -999.0}
            )
            return final_ds

        required_cols = {"height", "temperature", "pressure"}
        missing = sorted(required_cols - set(df_radio.columns))
        if missing:
            raise KeyError(f"Radiosonde data is missing required columns: {missing}")

        df_radio = df_radio.dropna(subset=["height", "temperature", "pressure"]).drop_duplicates(subset=["height"], keep="first").sort_values("height")
        if df_radio.empty:
            raise ValueError("Radiosonde data became empty after cleaning.")

        final_ds = final_ds.assign_coords(radiosonde_altitude=("radiosonde_altitude", df_radio["height"].values.astype(np.float64)))
        final_ds["radiosonde_altitude"].attrs.update({"units": "m", "long_name": "Radiosonde altitude above mean sea level"})
        final_ds["Radiosonde_Temperature_K"] = (
            ("radiosonde_altitude",), (df_radio["temperature"].values.astype(np.float64) + 273.15).astype(np.float32)
        )
        final_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_altitude",), df_radio["pressure"].values.astype(np.float32))
        final_ds["Radiosonde_Temperature_K"].attrs.update({"units": "K", "long_name": "Radiosonde air temperature", "source": "Wyoming Upper Air sounding"})
        final_ds["Radiosonde_Pressure_hPa"].attrs.update({"units": "hPa", "long_name": "Radiosonde atmospheric pressure", "source": "Wyoming Upper Air sounding"})

        cpt, lrt = calculate_tropopause_heights(df_radio)
        cpt = _finite_or_fill(cpt)
        lrt = _finite_or_fill(lrt)
        final_ds.attrs.update({"radiosonde_station_id": station_id, "radiosonde_available": "true", "tropopause_cpt_km": cpt, "tropopause_lrt_km": lrt})
        logger.info(f"  -> Sounding integrated. CPT: {cpt:.2f} km | LRT: {lrt:.2f} km")
        return final_ds
    except Exception as exc:
        logger.warning(f"  -> Sounding integration incomplete: {exc}")
        final_ds.attrs.update({"radiosonde_available": "false", "tropopause_cpt_km": -999.0, "tropopause_lrt_km": -999.0})
        return final_ds


def process_single_file(args: tuple[str | Path, Mapping[str, Any], logging.Logger]) -> str:
    """Process one Level 0 NetCDF into a Level 1 RCS NetCDF product."""
    nc_path, config, logger = args
    try:
        nc_file = Path(nc_path)
        stem = nc_file.stem
        save_path = level1_output_path(nc_file, config)
        logger.info(f"[{stem}] Initializing Level 1 processing...")
        ds_raw, z_arr = load_and_prepare_level0(nc_file, logger)
        final_ds = apply_all_physical_corrections(ds_raw, z_arr, config, logger)
        final_ds = estimate_pbl_timeseries(final_ds, z_arr, config, logger)
        final_ds = integrate_thermodynamics(final_ds, config, logger)
        final_ds.attrs.update(ds_raw.attrs)
        final_ds.attrs.update(
            {
                "Processing_level": "Level 1: PC->MHz, DeadTime, Dark Current, Bin Shift, Background subtraction, corrected signal, Range Corrected Signal, uncertainty propagation, PBL, Radiosonde, Tropopause",
                "Pipeline": "MILGRAU/LIPANCORA",
                "Input_Level0_File": str(nc_file.name),
                "Altitude_units": "m",
            }
        )
        ensure_directories(save_path.parent)
        encoding = {var: {"zlib": True, "complevel": 4} for var in final_ds.data_vars if final_ds[var].ndim > 0}
        final_ds.to_netcdf(save_path, encoding=encoding)
        return f"[OK] {stem} Level 1 generated successfully: {save_path}"
    except Exception:
        return f"[FAILED] {nc_path} execution halted:\n{traceback.format_exc()}"


def process_level_1(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover and process every Level 0 NetCDF file into Level 1."""
    in_dir = Path.cwd() / config["directories"]["processed_data"]
    files = [f for f in sorted(in_dir.rglob("*.nc")) if "level" not in f.name]
    if not files:
        logger.warning(f"No Level 0 files found in {in_dir}. Exiting.")
        return

    incremental = _incremental_enabled(config)
    files_to_process = []
    skipped_count = 0
    for file_path in files:
        output_path = level1_output_path(file_path, config)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 1 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 0 files require Level 1 processing. Skipped {skipped_count} existing products.")
        return

    logger.info(f"Found {len(files_to_process)} Level 0 files to process ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_file((str(file_path), config, logger)))
