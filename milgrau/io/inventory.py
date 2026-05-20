"""Measurement inventory construction for MILGRAU Level 0 processing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from milgrau.io.filesystem import scan_raw_files
from milgrau.io.licel import read_licel_header
from milgrau.physics.time import classify_period, get_night_date


def _effective_timezone(config: dict) -> str:
    """Return the site timezone from YAML, with São Paulo as fallback."""
    return config.get("site", {}).get("timezone") or config.get("location", {}).get("timezone") or "America/Sao_Paulo"


def _initialize_association_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add dark-current association audit columns to the raw inventory."""
    df_raw = df_raw.copy()
    df_raw["original_meas_id"] = df_raw["meas_id"]
    df_raw["association_method"] = np.where(
        df_raw["meas_type"] == "measurements",
        "measurement",
        "same_period",
    )
    df_raw["dark_current_association_delta_hours"] = np.nan
    return df_raw


def _reassign_orphan_dark_currents(df_raw: pd.DataFrame, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Reassign dark-current-only groups to nearby measurement groups when possible."""
    valid_mids = df_raw.loc[df_raw["meas_type"] == "measurements", "meas_id"].unique()
    orphan_mids = set(df_raw["meas_id"].unique()) - set(valid_mids)

    if not orphan_mids:
        return df_raw

    logger.info(f"   -> Reassigning orphaned Dark Current groups: {orphan_mids} to nearest measurements...")
    valid_df = df_raw[df_raw["meas_id"].isin(valid_mids)].copy()
    max_hours = config.get("processing", {}).get("dark_current_max_association_hours")

    if valid_df.empty:
        logger.warning("   -> No measurement groups available. Dropping orphan dark-current groups from inventory.")
        return df_raw[~df_raw["meas_id"].isin(orphan_mids)].copy()

    orphan_mask = df_raw["meas_id"].isin(orphan_mids)
    for idx, row in df_raw[orphan_mask].iterrows():
        time_diffs = abs(valid_df["start_time_utc"] - row["start_time_utc"])
        closest_idx = time_diffs.idxmin()
        closest_diff_h = time_diffs.loc[closest_idx].total_seconds() / 3600.0

        if max_hours is not None and closest_diff_h > float(max_hours):
            logger.warning(
                "   -> Orphan dark current not reassigned; nearest measurement "
                f"is {closest_diff_h:.2f} h away: {row['filepath']}"
            )
            continue

        df_raw.at[idx, "meas_id"] = valid_df.at[closest_idx, "meas_id"]
        df_raw.at[idx, "association_method"] = "nearest_measurement"
        df_raw.at[idx, "dark_current_association_delta_hours"] = float(closest_diff_h)

    return df_raw


def build_measurement_inventory(
    raw_dir: str,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build the Level-0 processing inventory from raw Licel files.

    Incremental skipping is intentionally handled by the LIBIDS pipeline, not by
    this inventory layer.  The inventory should report all discoverable raw files
    and preserve dark-current association metadata for NetCDF provenance.
    """
    logger.info("Building raw data inventory...")

    file_paths, file_types = scan_raw_files(raw_dir, logger=logger, config=config)
    if not file_paths:
        return pd.DataFrame()

    records = []
    for filepath, file_type in zip(file_paths, file_types):
        start_time_utc, stop_time, duration, n_shots, laser_freq = read_licel_header(filepath, logger=logger)
        if start_time_utc is None:
            continue

        records.append(
            {
                "filepath": filepath,
                "meas_type": file_type,
                "start_time_utc": start_time_utc,
                "stop_time": stop_time,
                "nshots": n_shots,
                "duration": duration,
                "laser_freq": laser_freq,
            }
        )

    df_raw = pd.DataFrame.from_records(records)
    if df_raw.empty:
        return df_raw

    timezone = _effective_timezone(config)
    df_raw["start_time_utc"] = pd.to_datetime(df_raw["start_time_utc"]).dt.tz_localize("UTC")
    df_raw["start_time_local"] = df_raw["start_time_utc"].dt.tz_convert(timezone)
    df_raw["meas_id"] = (
        df_raw["start_time_local"].apply(get_night_date).dt.strftime("%Y%m%d")
        + df_raw["start_time_local"].apply(classify_period)
    )
    df_raw = _initialize_association_columns(df_raw)
    df_raw = _reassign_orphan_dark_currents(df_raw, config, logger)

    return df_raw.reset_index(drop=True)
