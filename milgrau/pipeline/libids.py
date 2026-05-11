"""LIBIDS Level 0 pipeline orchestration."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from statistics import StatisticsError, mode

import numpy as np
import pandas as pd

from milgrau.io.filesystem import ensure_directories
from milgrau.io.inventory import build_measurement_inventory
from milgrau.io.licel import parse_licel_group
from milgrau.io.netcdf import build_level0_netcdf
from milgrau.io.weather import fetch_surface_weather


def _safe_mode(values) -> float:
    """Return the statistical mode with a median fallback."""
    try:
        return float(mode(values))
    except StatisticsError:
        return float(np.nanmedian(values))


def filter_laser_shots(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    tolerance_fraction: float = 2e-3,
) -> pd.DataFrame:
    """Evaluate laser-shot consistency for each measurement period."""
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


def process_level_0(config: dict, logger: logging.Logger) -> None:
    """Run LIBIDS Level 0 processing from raw Licel files to NetCDF."""
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
        out_dir = out_base_dir / save_id[:4] / save_id[4:6] / save_id
        netcdf_path = out_dir / f"{save_id}.nc"
        logger.info(f"Processing group [{save_id}]...")

        try:
            lat = float(config["physics"].get("latitude", -23.561))
            lon = float(config["physics"].get("longitude", -46.735))
            dt_utc_mean = group_df["start_time_utc"].iloc[len(group_df) // 2].to_pydatetime()
            weather_data = fetch_surface_weather(dt_utc_mean, lat, lon, logger=logger)
            if not weather_data:
                logger.warning(
                    f"  -> [{save_id}] Weather API/cache failed. Using fallback standard surface values."
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
            build_level0_netcdf(
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
