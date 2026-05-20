"""NetCDF product contract validators for MILGRAU.

The validators in this module check structural requirements shared by multiple
pipeline stages.  They intentionally avoid enforcing every possible metadata
attribute, but they do fail early on missing variables, dimensions and core
coordinates that would make downstream processing scientifically ambiguous.
"""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr


LEVEL0_REQUIRED_VARIABLES = (
    "Raw_Data_Start_Time",
    "Raw_Data_Range_Resolution",
    "channel_string",
    "Raw_Lidar_Data",
)
LEVEL1_REQUIRED_VARIABLES = (
    "corrected_signal",
    "corrected_signal_error",
    "range_corrected_signal",
    "range_corrected_signal_error",
)
LEVEL2_REQUIRED_VARIABLES = (
    "molecular_backscatter",
    "molecular_extinction",
    "glued_range_corrected_signal",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
)


def _missing_names(ds: xr.Dataset, names: Iterable[str]) -> list[str]:
    """Return names that are absent from an xarray Dataset."""
    return [name for name in names if name not in ds]


def _require_variables(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required variables are absent."""
    missing = _missing_names(ds, names)
    if missing:
        raise KeyError(f"{product_name} lacks required variable(s): {missing}")


def _require_coords(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required coordinates are absent."""
    missing = [name for name in names if name not in ds.coords]
    if missing:
        raise KeyError(f"{product_name} lacks required coordinate(s): {missing}")


def _require_dims(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required dimensions are absent."""
    missing = [name for name in names if name not in ds.dims]
    if missing:
        raise KeyError(f"{product_name} lacks required dimension(s): {missing}")


def validate_level0_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 0 structure required by LIPANCORA."""
    _require_variables(ds, LEVEL0_REQUIRED_VARIABLES, "Level 0 file")
    _require_dims(ds, ("time", "channels", "points"), "Level 0 file")
    if ds["Raw_Lidar_Data"].dims != ("time", "channels", "points"):
        raise ValueError("Level 0 Raw_Lidar_Data must have dimensions ('time', 'channels', 'points').")
    if "Background_Profile" in ds and ds["Background_Profile"].dims != ("time_bck", "channels", "points"):
        raise ValueError(
            "Level 0 Background_Profile must have dimensions ('time_bck', 'channels', 'points') when present."
        )


def validate_level1_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 1 structure required by LIRACOS and LEBEAR."""
    _require_variables(ds, LEVEL1_REQUIRED_VARIABLES, "Level 1 file")
    _require_coords(ds, ("time", "channel", "altitude"), "Level 1 file")
    expected_dims = ("time", "channel", "altitude")
    for name in LEVEL1_REQUIRED_VARIABLES:
        if ds[name].dims != expected_dims:
            raise ValueError(f"Level 1 {name} must have dimensions {expected_dims}; got {ds[name].dims}.")
    reference_shape = ds["range_corrected_signal"].shape
    for name in LEVEL1_REQUIRED_VARIABLES:
        if ds[name].shape != reference_shape:
            raise ValueError(f"Level 1 {name} shape does not match range_corrected_signal shape.")


def validate_level2_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 2 optical-product structure."""
    _require_variables(ds, LEVEL2_REQUIRED_VARIABLES, "Level 2 file")
    _require_coords(ds, ("wavelength", "altitude"), "Level 2 file")
    if "glued_range_corrected_signal" in ds:
        expected_dims = ("time", "wavelength", "altitude")
        if ds["glued_range_corrected_signal"].dims != expected_dims:
            raise ValueError(
                "Level 2 glued_range_corrected_signal must have dimensions "
                f"{expected_dims}; got {ds['glued_range_corrected_signal'].dims}."
            )
