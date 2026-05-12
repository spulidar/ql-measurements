"""LEBEAR Level 2 optical inversion pipeline scaffold.

This module defines the Level 2 orchestration boundary and imports the physics
kernels used for signal gluing, Rayleigh molecular calibration and KFS inversion.
The current scaffold validates Level 1 input contracts before the operational
LEBEAR implementation is added.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import xarray as xr

from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.physics.gluing import slide_glue_signals
from milgrau.physics.kfs import kfs_inversion_monte_carlo
from milgrau.physics.molecular import calculate_molecular_profile, find_optimal_reference_altitude


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
    """Validate the Level 1 variables required by future LEBEAR processing."""
    missing = [name for name in REQUIRED_LEVEL1_VARIABLES if name not in ds_l1]
    if missing:
        raise KeyError(f"Level 1 file lacks required variable(s): {missing}")
    if "altitude" not in ds_l1.coords:
        raise KeyError("Level 1 file lacks altitude coordinate.")
    if "channel" not in ds_l1.coords:
        raise KeyError("Level 1 file lacks channel coordinate.")


def process_single_level1_file(
    nc_file: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> str:
    """Placeholder for one-file Level 2 processing.

    The physical kernels are already modularized, but the operational LEBEAR
    product schema and tests should be finalized before writing Level 2 files.
    """
    nc_path = Path(nc_file)
    try:
        with xr.open_dataset(nc_path) as ds_l1:
            ds_l1.load()
            validate_level1_contract(ds_l1)

        logger.info(
            "LEBEAR scaffold validated Level 1 input only. "
            "Full Level 2 inversion will be implemented after synthetic tests."
        )
        return f"[PENDING] {nc_path.name} validated for future LEBEAR processing."
    except Exception as exc:
        return f"[FAILED] {nc_path}: {exc}"


def process_level_2(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover Level 1 files and run the current LEBEAR scaffold."""
    files = discover_level1_files(config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        return
    logger.info(f"Found {len(files)} Level 1 files for LEBEAR.")
    for file_path in files:
        logger.info(process_single_level1_file(file_path, config, logger))
