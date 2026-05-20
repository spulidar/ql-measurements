"""NetCDF writing helpers for MILGRAU products."""

from __future__ import annotations

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from milgrau.io.licel import parse_licel_group


def validate_lidar_tensors(tensors: dict, channels: list[str]) -> tuple[int, int]:
    """Validate Level-0 tensor consistency before NetCDF export."""
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
            raise ValueError(f"Inconsistent tensor shape for channel {ch_name}: expected {reference_shape}, got {tensor.shape}.")
    num_times, num_points = reference_shape
    return int(num_times), int(num_points)


def _source_file_names(group_df: pd.DataFrame) -> list[str]:
    """Return sorted source-file names from an inventory group."""
    if "filepath" not in group_df:
        return []
    return sorted(Path(path).name for path in group_df["filepath"].tolist())


def _dark_current_attributes(group_df: pd.DataFrame) -> dict:
    """Build global attributes describing dark-current provenance."""
    df_dc = group_df[group_df["meas_type"] == "dark_current"].copy()
    if df_dc.empty:
        return {
            "Dark_Current_Source_File_Count": 0,
            "Dark_Current_Source_Files": "",
            "Dark_Current_Association_Methods": "none",
            "Dark_Current_Max_Association_Delta_hours": np.nan,
        }

    methods = "unknown"
    if "association_method" in df_dc:
        methods = ";".join(sorted(str(value) for value in df_dc["association_method"].dropna().unique())) or "unknown"

    max_delta = np.nan
    if "dark_current_association_delta_hours" in df_dc:
        delta_values = pd.to_numeric(df_dc["dark_current_association_delta_hours"], errors="coerce")
        if delta_values.notna().any():
            max_delta = float(delta_values.max())

    return {
        "Dark_Current_Source_File_Count": int(len(df_dc)),
        "Dark_Current_Source_Files": ";".join(_source_file_names(df_dc)),
        "Dark_Current_Association_Methods": methods,
        "Dark_Current_Max_Association_Delta_hours": max_delta,
    }


def build_level0_global_attributes(
    save_id: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
) -> dict:
    """Build global NetCDF attributes for the Level-0 product."""
    min_start_utc = pd.to_datetime(group_df["start_time_utc"]).min()
    max_stop_utc = pd.to_datetime(group_df["stop_time"]).max()
    source_files = _source_file_names(group_df)

    attrs = {
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
        "Temperature_C": float(weather_data.get("temperature_c", config["physics"].get("default_surface_temp_c", 25.0))),
        "Pressure_hPa": float(weather_data.get("pressure_hpa", config["physics"].get("default_surface_pressure_hpa", 940.0))),
        "CloudCover_percent": float(weather_data.get("cloud_cover_percent", np.nan)),
        "RelativeHumidity_percent": float(weather_data.get("relative_humidity_percent", np.nan)),
        "WindSpeed_kmh": float(weather_data.get("wind_speed_kmh", np.nan)),
        "Source_File_Count": int(len(source_files)),
        "Source_Files": ";".join(source_files),
    }
    attrs.update(_dark_current_attributes(group_df))
    return attrs


def _write_dark_current_availability(ds: nc.Dataset, availability: np.ndarray) -> None:
    """Write channel-wise dark-current availability flags."""
    var = ds.createVariable("Background_Profile_Available", "i1", ("channels",))
    var.long_name = "Dark-current profile availability by channel"
    var.flag_values = "0, 1"
    var.flag_meanings = "not_available available"
    var[:] = availability.astype(np.int8)


def write_dark_current_profile(
    ds: nc.Dataset,
    group_df: pd.DataFrame,
    channels: list[str],
    num_channels: int,
    num_points: int,
    logger: logging.Logger,
) -> None:
    """Parse and write the optional dark-current matrix and availability flags."""
    availability = np.zeros(num_channels, dtype=np.int8)
    df_dc = group_df[group_df["meas_type"] == "dark_current"]
    if df_dc.empty:
        _write_dark_current_availability(ds, availability)
        return

    dc_files = df_dc["filepath"].tolist()
    dc_data = parse_licel_group(dc_files, logger)
    if not dc_data.get("tensors"):
        logger.warning("  -> Dark current files found but parsing failed. NetCDF will lack Background_Profile.")
        _write_dark_current_availability(ds, availability)
        return

    first_tensor = next(iter(dc_data["tensors"].values()))
    num_time_bck = first_tensor.shape[0]
    ds.createDimension("time_bck", num_time_bck)
    bck_prof = ds.createVariable("Background_Profile", "f8", ("time_bck", "channels", "points"), zlib=True)
    bck_prof.long_name = "Dark-current background profile"
    bck_prof.units = "channel native units"

    stacked_dc = np.full((num_time_bck, num_channels, num_points), np.nan, dtype=np.float64)
    for i, ch_name in enumerate(channels):
        if ch_name not in dc_data["tensors"]:
            logger.warning(f"  -> Dark-current data missing for channel {ch_name}. Filling with NaN and flagging unavailable.")
            continue
        dc_tensor = np.asarray(dc_data["tensors"][ch_name], dtype=np.float64)
        if dc_tensor.ndim != 2 or dc_tensor.shape[1] != num_points:
            logger.warning(
                f"  -> Dark-current channel {ch_name} has shape {dc_tensor.shape}; expected (*, {num_points}). "
                "Filling with NaN and flagging unavailable."
            )
            continue
        n_copy = min(num_time_bck, dc_tensor.shape[0])
        stacked_dc[:n_copy, i, :] = dc_tensor[:n_copy, :]
        availability[i] = 1

    bck_prof[:] = stacked_dc
    _write_dark_current_availability(ds, availability)
    logger.info(f"  -> Successfully injected Dark Current matrix ({num_time_bck} profiles).")


def build_level0_netcdf(
    netcdf_path: str,
    save_id: str,
    period: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """Generate an SCC-compliant Level-0 NetCDF from parsed Licel tensors."""
    try:
        tensors = lidar_data["tensors"]
        channels = lidar_data["channels"]
        num_times_tensor, num_points = validate_lidar_tensors(tensors, channels)
        num_channels = len(channels)

        system_mode = "night" if period == "nt" else "day"
        hardware_map = config["hardware"].get("name_to_id", {}).get(system_mode, {})

        df_meas = group_df[group_df["meas_type"] == "measurements"].copy()
        start_times_epoch = (pd.to_datetime(df_meas["start_time_utc"]) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")

        if len(start_times_epoch) != num_times_tensor:
            n_copy = min(len(start_times_epoch), num_times_tensor)
            logger.warning(
                f"  -> Time axis mismatch for {save_id}: metadata has {len(start_times_epoch)} profiles but tensor has {num_times_tensor}. "
                f"Truncating to {n_copy}."
            )
            start_times_epoch = start_times_epoch.iloc[:n_copy]
            num_times = n_copy
        else:
            num_times = num_times_tensor
        if num_times <= 0:
            raise ValueError("No valid time profiles available after tensor/time-axis validation.")

        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as ds:
            ds.setncatts(build_level0_global_attributes(save_id, lidar_data, group_df, weather_data, config))
            ds.createDimension("time", num_times)
            ds.createDimension("channels", num_channels)
            ds.createDimension("points", num_points)

            raw_data_start = ds.createVariable("Raw_Data_Start_Time", "f8", ("time",))
            raw_data_start.units = "seconds since 1970-01-01 00:00:00"
            raw_data_start.long_name = "Raw data start time"
            raw_data_start[:] = start_times_epoch.values

            raw_lidar_data = ds.createVariable("Raw_Lidar_Data", "f8", ("time", "channels", "points"), zlib=True)
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
                    logger.warning(f"  -> Channel {ch_name} missing in config for {system_mode} mode. Using default 9999.")
                channel_ids[i] = hardware_map.get(ch_name, 9999)
                range_res[i] = float(config["physics"].get("vertical_resolution_m", 7.5))
                bg_low[i] = float(config["physics"].get("background_start_m", config["physics"].get("bg_start", 29000)))
                bg_high[i] = float(config["physics"].get("background_stop_m", config["physics"].get("bg_stop", 29999)))

            raw_lidar_data[:] = stacked_tensor
            write_dark_current_profile(ds, group_df, channels, num_channels, num_points, logger)
    except Exception as exc:
        raise RuntimeError(f"Failed to build NetCDF: {exc}") from exc
