"""Canonical path builders for MILGRAU products.

This module centralizes file-name and directory conventions so pipeline stages do
not need to duplicate product layout logic.  All functions are intentionally
small and side-effect free; directory creation remains the responsibility of the
calling pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

LEVEL0_SUFFIX = ".nc"
LEVEL1_SUFFIX = "_level1_rcs.nc"
LEVEL2_SUFFIX = "_level2_optical.nc"


def processed_data_root(config: Mapping[str, Any], root_dir: str | Path | None = None) -> Path:
    """Return the configured processed-data root directory."""
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    return root_path / str(config["directories"]["processed_data"])


def measurement_save_id(measurement_id: str) -> str:
    """Return the canonical SCC-style MILGRAU save ID for a measurement group.

    Inventory measurement IDs are expected to use the compact form
    ``YYYYMMDDam``, ``YYYYMMDDpm`` or ``YYYYMMDDnt``.  Product directories and
    Level 0 files use ``YYYYMMDDsa<period>``.
    """
    value = str(measurement_id)
    if len(value) < 10:
        raise ValueError(f"Invalid measurement_id: {measurement_id!r}")
    return f"{value[:8]}sa{value[8:]}"


def measurement_product_dir(
    save_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return the canonical product directory for one measurement save ID."""
    save_id = str(save_id)
    if len(save_id) < 6:
        raise ValueError(f"Invalid save_id: {save_id!r}")
    return processed_data_root(config, root_dir=root_dir) / save_id[:4] / save_id[4:6] / save_id


def level0_output_path(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return the Level 0 NetCDF output path for one inventory measurement ID."""
    save_id = measurement_save_id(measurement_id)
    return measurement_product_dir(save_id, config, root_dir=root_dir) / f"{save_id}{LEVEL0_SUFFIX}"


def level1_output_path(
    level0_file: str | Path,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return the Level 1 RCS NetCDF output path for one Level 0 NetCDF file."""
    stem = Path(level0_file).stem
    return measurement_product_dir(stem, config, root_dir=root_dir) / f"{stem}{LEVEL1_SUFFIX}"


def level2_output_path(level1_file: str | Path) -> Path:
    """Return the Level 2 optical NetCDF output path for one Level 1 NetCDF file."""
    path = Path(level1_file)
    if not path.name.endswith(LEVEL1_SUFFIX):
        raise ValueError(f"Expected a Level 1 file ending with {LEVEL1_SUFFIX}: {path}")
    stem = path.name.removesuffix(LEVEL1_SUFFIX)
    return path.parent / f"{stem}{LEVEL2_SUFFIX}"


def quicklook_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    formatted_channel_name: str,
    max_altitude_km: float,
    output_format: str,
) -> Path:
    """Return the expected Level 1 quicklook image path."""
    safe_channel = str(formatted_channel_name).replace(" ", "_")
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"Quicklook_{file_name_prefix}_{safe_channel}_{float(max_altitude_km):g}km.{suffix}"


def global_mean_rcs_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    output_format: str,
) -> Path:
    """Return the expected global mean RCS image path."""
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"GlobalMeanRCS_{file_name_prefix}.{suffix}"
