"""
MILGRAU - Level 0: LIdar BInary Data Standardized (LIBIDS)

Reads raw Licel binary data, applies Level-0 quality control, synchronizes data
with metadata, and writes SCC-compliant NetCDF files for downstream processing.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from statistics import StatisticsError, mode

import netCDF4 as nc
import numpy as np
import pandas as pd

from functions.core_io import (
    build_measurement_inventory,
    ensure_directories,
    fetch_surface_weather,
    load_config,
    parse_licel_group,
    setup_logger,
)


def _safe_mode(values) -> float:
    """
    Return the statistical mode with a deterministic fallback.

    Laser-shot counts should be tightly clustered. If the distribution is
    multimodal, the median is a conservative fallback for the expected value.
    """
    try:
        return float(mode(values))
    except StatisticsError:
        return float(np.nanmedian(values))


def filter_laser_shots(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    tolerance_fraction: float = 2e-3,
) -> pd.DataFrame:
    """
    Evaluate laser-shot consistency for each measurement period.

    Measurement files and dark-current files are treated separately. The strict
    consistency criterion is applied to real measurement files. Dark-current
    files are kept unless they have invalid or zero shots, because they may be
    acquired with a different accumulation setup.
    """
    logger.info("Evaluating laser shots quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            df_meas = group[group["meas_type"] == "measurements"].copy()
            df_dc = group[group["meas_type"] == "dark_current"].copy()

            if df_meas.empty:
                logger.warning(f"  -> [{meas_id}] No measurement files found after inventory stage.")
                continue

            expected_shots = _safe_mode(df_meas["nshots"].dropna().values)
            shot_deviation = abs(df_meas["nshots"] - expected_shots)
            bad_meas_condition = (
                df_meas["nshots"].isna()
                | (df_meas["nshots"] <= 0)
                | (shot_deviation >= tolerance_fraction * expected_shots)
            )

            good_meas = df_meas.loc[~bad_meas_condition]
            bad_meas = df_meas.loc[bad_meas_condition]

            if not df_dc.empty:
                good_dc = df_dc.loc[df_dc["nshots"].fillna(0) > 0]
                bad_dc = df_dc.loc[df_dc["nshots"].fillna(0) <= 0]
            else:
                good_dc = df_dc
                bad_dc = df_dc

            total_files = len(group)
            bad_files = len(bad_meas) + len(bad_dc)
            loss_percent = (bad_files / total_files) * 100.0 if total_files > 0 else 0.0

            if bad_files > 0:
                log_msg = (
                    f"  -> [{meas_id}] QA Report: {bad_files}/{total_files} files rejected "
                    f"({loss_percent:.1f}% loss). Measurement rejects: {len(bad_meas)}, "
                    f"dark-current rejects: {len(bad_dc)}."
                )
                if loss_percent > 10.0:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)
            else:
                logger.info(f"  -> [{meas_id}] QA Report: 100% data retention. No files rejected.")

            good_group = pd.concat([good_meas, good_dc], ignore_index=True)
            if not good_group.empty:
                good_groups.append(good_group)

        except Exception as exc:
            logger.warning(f"  -> [{meas_id}] Error evaluating quality: {exc}")

    if not good_groups:
        return pd.DataFrame()

    return pd.concat(good_groups, ignore_index=True)


def _validate_lidar_tensors(tensors: dict, channels: list[str]) -> tuple[int, int]:
    """
    Validate Level-0 tensor consistency before NetCDF export.

    All active channels must be 2D arrays with the same shape: time x range_bin.
    """
    if not tensors:
        raise ValueError("No lidar tensors available for NetCDF export.")
    if not channels:
        raise ValueError("No channel list available for NetCDF export.")

    missing_channels = [ch for ch in channels if ch not in tensors]
    if missing_channels:
        raise ValueError(f"Channels missing from tensor dictionary: {missing_channels}")

    reference_shape = None
    for ch_name in channels:
        tensor = np.asarray(tensors[ch_name])
        if tensor.ndim != 2:
            raise ValueError(f"Tensor for channel {ch_name} must be 2D; got shape {tensor.shape}.")
        if reference_shape is None:
            reference_shape = tensor.shape
        elif tensor.shape != reference_shape:
            raise ValueError(
                f"Inconsistent tensor shape for channel {ch_name}: "
                f"expected {reference_shape}, got {tensor.shape}."
            )

    num_times, num_points = reference_shape
    return int(num_times), int(num_points)


def _build_global_attributes(
    save_id: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
) -> dict:
    """Build global NetCDF attributes for the Level-0 product."""
    min_start_utc = pd.to_datetime(group_df["start_time_utc"]).min()
    max_stop_utc = pd.to_datetime(group_df["stop_time"]).max()
    source_files = sorted(Path(path).name for path in group_df["filepath"].tolist())

    return {
        "Measurement_ID": save_id,
        "System": "SPU-Lidar",
        "Processing_level": "Level 0: Raw Licel to SCC-compliant NetCDF",
        "Pipeline": "MILGRAU",
        "Latitude_degrees_north": float(config["physics"].get("latitude", -23.561)),
        "Longitude_degrees_east": float(config["physics"].get("longitude", -46.735)),
        "Accumulated_Shots": int(lidar_data.get("shots", 0)),
        "RawData_Start_Date": min_start_utc.strftime("%Y%m%d"),
        "RawData_Start_Time_UT": min_start_utc.strftime("%H%M%S"),
        "RawData_Stop_Time_UT": max_stop_utc.strftime("%H%M%S"),
        "Temperature_C": float(
            weather_data.get("temperature_c", config["physics"].get("default_surface_temp_c", 25.0))
        ),
        "Pressure_hPa": float(
            weather_data.get("pressure_hpa", config["physics"].get("default_surface_pressure_hpa", 940.0))
        ),
        "CloudCover_percent": float(weather_data.get("cloud_cover_percent", np.nan)),
        "RelativeHumidity_percent": float(weather_data.get("relative_humidity_percent", np.nan)),
        "WindSpeed_kmh": float(weather_data.get("wind_speed_kmh", np.nan)),
        "Source_File_Count": int(len(source_files)),
        "Source_Files": ";".join(source_files),
    }


def _write_dark_current_profile(
    ds: nc.Dataset,
    group_df: pd.DataFrame,
    channels: list[str],
    num_channels: int,
    num_points: int,
    logger: logging.Logger,
) -> None:
    """Parse and write the optional dark-current matrix into the NetCDF file."""
    df_dc = group_df[group_df["meas_type"] == "dark_current"]
    if df_dc.empty:
        return

    dc_files = df_dc["filepath"].tolist()
    dc_data = parse_licel_group(dc_files, logger)

    if not dc_data.get("tensors"):
        logger.warning("  -> Dark current files found but parsing failed. NetCDF will lack Background_Profile.")
        return

    first_tensor = next(iter(dc_data["tensors"].values()))
    num_time_bck = first_tensor.shape[0]

    ds.createDimension("time_bck", num_time_bck)
    bck_prof = ds.createVariable(
        "Background_Profile",
        "f8",
        ("time_bck", "channels", "points"),
        zlib=True,
    )
    bck_prof.long_name = "Dark-current background profile"
    bck_prof.units = "channel native units"

    stacked_dc = np.zeros((num_time_bck, num_channels, num_points), dtype=np.float64)

    for i, ch_name in enumerate(channels):
        if ch_name not in dc_data["tensors"]:
            logger.warning(f"  -> Dark-current data missing for channel {ch_name}. Filling with zeros.")
            continue

        dc_tensor = dc_data["tensors"][ch_name]
        if dc_tensor.shape[1] != num_points:
            logger.warning(
                f"  -> Dark-current channel {ch_name} has {dc_tensor.shape[1]} bins; "
                f"expected {num_points}. Skipping this dark-current channel."
            )
            continue

        n_copy = min(num_time_bck, dc_tensor.shape[0])
        stacked_dc[:n_copy, i, :] = dc_tensor[:n_copy, :]

    bck_prof[:] = stacked_dc
    logger.info(f"  -> Successfully injected Dark Current matrix ({num_time_bck} profiles).")


def build_netcdf(
    netcdf_path: str,
    save_id: str,
    period: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """
    Generate an SCC-compliant Level-0 NetCDF from parsed Licel tensors.

    This function writes the Level-0 product schema. Lower-level binary parsing
    and inventory logic remain in core_io.
    """
    try:
        tensors = lidar_data["tensors"]
        channels = lidar_data["channels"]

        num_times_tensor, num_points = _validate_lidar_tensors(tensors, channels)
        num_channels = len(channels)

        system_mode = "night" if period == "nt" else "day"
        hardware_map = config["hardware"].get("name_to_id", {}).get(system_mode, {})

        df_meas = group_df[group_df["meas_type"] == "measurements"].copy()
        start_times_epoch = (
            pd.to_datetime(df_meas["start_time_utc"])
            - pd.Timestamp("1970-01-01", tz="UTC")
        ) // pd.Timedelta("1s")

        if len(start_times_epoch) != num_times_tensor:
            n_copy = min(len(start_times_epoch), num_times_tensor)
            logger.warning(
                f"  -> Time axis mismatch for {save_id}: metadata has {len(start_times_epoch)} "
                f"profiles but tensor has {num_times_tensor}. Truncating to {n_copy}."
            )
            start_times_epoch = start_times_epoch.iloc[:n_copy]
            num_times = n_copy
        else:
            num_times = num_times_tensor

        if num_times <= 0:
            raise ValueError("No valid time profiles available after tensor/time-axis validation.")

        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as ds:
            ds.setncatts(
                _build_global_attributes(
                    save_id=save_id,
                    lidar_data=lidar_data,
                    group_df=group_df,
                    weather_data=weather_data,
                    config=config,
                )
            )

            ds.createDimension("time", num_times)
            ds.createDimension("channels", num_channels)
            ds.createDimension("points", num_points)

            raw_data_start = ds.createVariable("Raw_Data_Start_Time", "f8", ("time",))
            raw_data_start.units = "seconds since 1970-01-01 00:00:00"
            raw_data_start.long_name = "Raw data start time"
            raw_data_start[:] = start_times_epoch.values

            raw_lidar_data = ds.createVariable(
                "Raw_Lidar_Data",
                "f8",
                ("time", "channels", "points"),
                zlib=True,
            )
            raw_lidar_data.long_name = "Raw lidar signal"
            raw_lidar_data.units = "counts for PC, mV per shot for analog"

            channel_ids = ds.createVariable("channel_ID", "i4", ("channels",))
            range_res = ds.createVariable("Raw_Data_Range_Resolution", "f8", ("channels",))
            bg_low = ds.createVariable("Background_Low", "f8", ("channels",))
            bg_high = ds.createVariable("Background_High", "f8", ("channels",))
            channel_names = ds.createVariable("channel_string", str, ("channels",))

            range_res.units = "m"
            bg_low.units = "m"
            bg_high.units = "m"

            stacked_tensor = np.zeros((num_times, num_channels, num_points), dtype=np.float64)

            for i, ch_name in enumerate(channels):
                stacked_tensor[:, i, :] = tensors[ch_name][:num_times, :]
                channel_names[i] = ch_name

                if ch_name not in hardware_map:
                    logger.warning(
                        f"  -> Channel {ch_name} missing in config for {system_mode} mode. "
                        "Using default 9999."
                    )

                channel_ids[i] = hardware_map.get(ch_name, 9999)
                range_res[i] = float(config["physics"].get("vertical_resolution_m", 7.5))
                bg_low[i] = float(config["physics"].get("bg_start", 29000))
                bg_high[i] = float(config["physics"].get("bg_stop", 29999))

            raw_lidar_data[:] = stacked_tensor

            _write_dark_current_profile(
                ds=ds,
                group_df=group_df,
                channels=channels,
                num_channels=num_channels,
                num_points=num_points,
                logger=logger,
            )

    except Exception as exc:
        raise RuntimeError(f"Failed to build NetCDF: {exc}") from exc


def process_level_0(config: dict, logger: logging.Logger) -> None:
    """
    Main Level-0 processing orchestrator.

    Processing order:
    1. build raw-data inventory;
    2. filter invalid laser-shot groups;
    3. fetch/cache surface weather;
    4. parse raw Licel measurement files;
    5. write SCC-compliant Level-0 NetCDF.
    """
    raw_dir = Path.cwd() / config["directories"]["raw_data"]
    df_raw = build_measurement_inventory(str(raw_dir), config, logger)

    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        return

    tolerance_fraction = float(config.get("processing", {}).get("laser_shot_tolerance_fraction", 2e-3))
    df_good = filter_laser_shots(df_raw, logger, tolerance_fraction=tolerance_fraction)

    if df_good.empty:
        logger.warning("=== No data survived quality control. Exiting. ===")
        return

    out_base_dir = Path.cwd() / config["directories"]["processed_data"]
    success_count = 0
    total_groups = len(df_good["meas_id"].unique())

    for meas_id, group_df in df_good.groupby("meas_id"):
        save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
        year_str, month_str = save_id[:4], save_id[4:6]

        out_dir = out_base_dir / year_str / month_str / save_id
        netcdf_path = out_dir / f"{save_id}.nc"

        logger.info(f"Processing group [{save_id}]...")

        try:
            lat = float(config["physics"].get("latitude", -23.561))
            lon = float(config["physics"].get("longitude", -46.735))
            dt_utc_mean = group_df["start_time_utc"].iloc[len(group_df) // 2].to_pydatetime()

            weather_data = fetch_surface_weather(dt_utc_mean, lat, lon)

            if not weather_data:
                logger.warning(
                    f"  -> [{save_id}] Weather API/cache failed. "
                    "Using fallback standard surface values."
                )
                weather_data = {
                    "temperature_c": float(config["physics"].get("default_surface_temp_c", 25.0)),
                    "pressure_hpa": float(config["physics"].get("default_surface_pressure_hpa", 940.0)),
                    "relative_humidity_percent": np.nan,
                    "cloud_cover_percent": np.nan,
                    "wind_speed_kmh": np.nan,
                }

            df_meas = group_df[group_df["meas_type"] == "measurements"]
            files_meas = df_meas["filepath"].tolist()

            if not files_meas:
                logger.warning(f"  -> [{save_id}] No measurement files found. Skipping.")
                continue

            lidar_data_tensors = parse_licel_group(files_meas, logger)
            if not lidar_data_tensors.get("tensors"):
                logger.warning(f"  -> [{save_id}] No valid lidar tensors parsed. Skipping.")
                continue

            ensure_directories(out_dir)

            build_netcdf(
                netcdf_path=str(netcdf_path),
                save_id=save_id,
                period=meas_id[8:],
                lidar_data=lidar_data_tensors,
                group_df=group_df,
                weather_data=weather_data,
                config=config,
                logger=logger,
            )

            logger.info(f"  -> [OK] NetCDF successfully generated: {netcdf_path}")
            success_count += 1

        except Exception:
            logger.error(f"  -> [ERROR] Fatal error converting {save_id}:\n{traceback.format_exc()}")

    logger.info(f"=== Processed {success_count}/{total_groups} groups. LIBIDS finished! ===")


if __name__ == "__main__":
    config_dict = load_config()
    main_logger = setup_logger("LIBIDS", config_dict["directories"]["log_dir"])
    main_logger.info("=== Starting MILGRAU LIBIDS processing (Level 0) ===")

    process_level_0(config_dict, main_logger)
