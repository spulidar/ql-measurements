"""LIRACOS Level 1 visualization pipeline orchestration."""

from __future__ import annotations

import gc
import logging
import traceback
from pathlib import Path
from typing import Any

import xarray as xr
from matplotlib import pyplot as plt

from milgrau.io.filesystem import ensure_directories
from milgrau.visualization.quicklooks import format_channel_name, plot_global_mean_rcs, plot_quicklook

RCS_VARIABLE = "range_corrected_signal"
RCS_ERROR_VARIABLE = "range_corrected_signal_error"


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert configuration values to bool safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _get_output_format(config: dict[str, Any]) -> str:
    """Return configured plot output format."""
    return str(config.get("visualization", {}).get("output_format", "webp")).lstrip(".").lower()


def _quicklook_output_path(output_folder: Path, file_name_prefix: str, channel_name: str, max_altitude: float, config: dict[str, Any]) -> Path:
    """Return the expected output path for one quicklook plot."""
    pretty_channel = format_channel_name(channel_name)
    output_format = _get_output_format(config)
    return output_folder / f"Quicklook_{file_name_prefix}_{pretty_channel.replace(' ', '_')}_{float(max_altitude):g}km.{output_format}"


def _global_mean_output_path(output_folder: Path, file_name_prefix: str, config: dict[str, Any]) -> Path:
    """Return the expected output path for the global mean RCS plot."""
    return output_folder / f"GlobalMeanRCS_{file_name_prefix}.{_get_output_format(config)}"


def _get_visualization_channels(config: dict[str, Any]) -> list[str]:
    """Return configured channels without duplicates, preserving YAML order."""
    channels = config.get("visualization", {}).get("channels_to_plot", []) or []
    unique_channels: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        channel_name = str(channel)
        if channel_name not in seen:
            unique_channels.append(channel_name)
            seen.add(channel_name)
    return unique_channels


def _get_altitude_ranges_km(config: dict[str, Any]) -> list[float]:
    """Return configured altitude limits in kilometers."""
    raw_ranges = config.get("visualization", {}).get("altitude_ranges_km", [5, 15, 30])
    altitude_ranges: list[float] = []
    for value in raw_ranges:
        try:
            altitude_km = float(value)
        except (TypeError, ValueError):
            continue
        if altitude_km > 0:
            altitude_ranges.append(altitude_km)
    return altitude_ranges or [5.0, 15.0, 30.0]


def _prepare_level1_for_visualization(ds: xr.Dataset) -> xr.Dataset:
    """Return a plotting-ready Level 1 dataset with altitude in kilometers."""
    if "altitude" not in ds.coords:
        raise ValueError("Level 1 dataset does not contain an 'altitude' coordinate.")
    max_altitude = float(ds["altitude"].max().values)
    if max_altitude > 100.0:
        ds = ds.assign_coords(altitude=ds["altitude"] / 1000.0)
    ds["altitude"].attrs["units"] = "km"
    ds["altitude"].attrs["long_name"] = "Altitude above ground level"
    return ds


def _extract_level1_boundaries(ds: xr.Dataset) -> tuple[xr.DataArray | None, float, float]:
    """Extract PBL and tropopause metadata used as plot overlays."""
    pbl_da = ds["PBL_Height_km"] if "PBL_Height_km" in ds else None
    cpt_km = float(ds.attrs.get("tropopause_cpt_km", -999.0))
    lrt_km = float(ds.attrs.get("tropopause_lrt_km", -999.0))
    return pbl_da, cpt_km, lrt_km


def _validate_l1_visualization_contract(ds: xr.Dataset) -> None:
    """Validate the Level 1 variables required by LIRACOS."""
    missing = [name for name in (RCS_VARIABLE, RCS_ERROR_VARIABLE) if name not in ds]
    if missing:
        raise KeyError(f"Level 1 file is missing required RCS variable(s): {missing}")


def process_single_nc(args: tuple[str | Path, dict[str, Any], str | Path, logging.Logger]) -> str:
    """Render all Level 1 quicklooks for one NetCDF file."""
    nc_file_path, config, root_dir, logger = args
    nc_file = Path(nc_file_path)
    root_path = Path(root_dir)
    try:
        file_name_prefix = nc_file.name.replace("_level1_rcs.nc", "")
        output_folder = nc_file.parent / "quicklooks"
        ensure_directories(output_folder)
        incremental = _as_bool(config.get("processing", {}).get("incremental", False))
        generated_count = 0
        skipped_count = 0

        logger.info(f"[{file_name_prefix}] Loading Level 1 data and preparing axes...")
        with xr.open_dataset(nc_file) as ds:
            ds.load()
            _validate_l1_visualization_contract(ds)
            ds = _prepare_level1_for_visualization(ds)
            pbl_da, cpt_km, lrt_km = _extract_level1_boundaries(ds)
            channels_to_plot = _get_visualization_channels(config)
            altitude_ranges = _get_altitude_ranges_km(config)
            available_channels = {str(channel) for channel in ds.channel.values}

            logger.info(f"[{file_name_prefix}] Rendering quicklooks for {len(channels_to_plot)} configured channels.")
            for channel_name in channels_to_plot:
                if channel_name not in available_channels:
                    logger.warning(f"[{file_name_prefix}] Channel not found, skipping: {channel_name}")
                    continue
                rc_signal = ds[RCS_VARIABLE].sel(channel=channel_name)
                rc_error = ds[RCS_ERROR_VARIABLE].sel(channel=channel_name)
                for max_altitude in altitude_ranges:
                    expected_path = _quicklook_output_path(output_folder, file_name_prefix, channel_name, max_altitude, config)
                    if incremental and expected_path.exists():
                        logger.info(f"[SKIPPED] Quicklook already exists: {expected_path.name}")
                        skipped_count += 1
                        continue

                    sig_slice = rc_signal.sel(altitude=slice(0, max_altitude))
                    err_slice = rc_error.sel(altitude=slice(0, max_altitude))
                    if sig_slice.size == 0:
                        logger.warning(
                            f"[{file_name_prefix}] Empty altitude slice for {channel_name} up to {max_altitude} km."
                        )
                        continue
                    plot_quicklook(
                        data_slice=sig_slice,
                        error_slice=err_slice,
                        max_altitude=max_altitude,
                        channel_name=channel_name,
                        ds=ds,
                        output_folder=str(output_folder),
                        file_name_prefix=file_name_prefix,
                        config=config,
                        root_dir=str(root_path),
                        pbl_da=pbl_da,
                        cpt_km=cpt_km,
                        lrt_km=lrt_km,
                    )
                    generated_count += 1
                    del sig_slice, err_slice
                    plt.close("all")
                    gc.collect()

            global_mean_path = _global_mean_output_path(output_folder, file_name_prefix, config)
            if incremental and global_mean_path.exists():
                logger.info(f"[SKIPPED] Global mean RCS already exists: {global_mean_path.name}")
                skipped_count += 1
            else:
                logger.info(f"[{file_name_prefix}] Generating global mean RCS profile...")
                plot_global_mean_rcs(ds, str(output_folder), file_name_prefix, config, str(root_path))
                generated_count += 1

        plt.close("all")
        gc.collect()
        return f"[OK] Plots generated for {file_name_prefix}: generated={generated_count}, skipped={skipped_count}"
    except Exception:
        return f"[FAILED] Error plotting {nc_file.name}:\n{traceback.format_exc()}"


def process_all_level1_files(
    config: dict[str, Any],
    logger: logging.Logger,
    root_dir: str | Path | None = None,
) -> None:
    """Discover and render all Level 1 NetCDF files under processed_data."""
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    base_data_folder = root_path / config["directories"]["processed_data"]
    nc_files = sorted(base_data_folder.rglob("*_level1_rcs.nc"))
    if not nc_files:
        logger.warning(f"No Level 1 NetCDF data found in '{base_data_folder}'. Exiting.")
        return
    logger.info(f"Found {len(nc_files)} Level 1 files.")
    for nc_file in nc_files:
        result = process_single_nc((nc_file, config, root_path, logger))
        if "[OK]" in result or "[SKIPPED]" in result:
            logger.info(result)
        else:
            logger.error(result)
