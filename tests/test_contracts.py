"""Tests for MILGRAU NetCDF contract validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract, validate_level1_contract, validate_level2_contract


def test_validate_level0_contract_accepts_minimal_dataset() -> None:
    """A minimal Level 0 dataset with the canonical raw tensor should validate."""
    ds = xr.Dataset(
        data_vars={
            "Raw_Data_Start_Time": (("time",), np.array([0.0, 1.0])),
            "Raw_Data_Range_Resolution": (("channels",), np.array([7.5, 7.5])),
            "channel_string": (("channels",), np.array(["532.AN", "532.PC"], dtype=object)),
            "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((2, 2, 4))),
        }
    )

    validate_level0_contract(ds)


def test_validate_level1_contract_rejects_missing_rcs_error() -> None:
    """A Level 1 file missing propagated RCS uncertainty should fail early."""
    time = pd.date_range("2024-01-01", periods=2)
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.arange(4.0)
    shape = (2, 1, 4)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
    )

    with pytest.raises(KeyError):
        validate_level1_contract(ds)


def test_validate_level1_contract_accepts_required_signal_tensors() -> None:
    """The Level 1 contract should accept all four core signal tensors."""
    time = pd.date_range("2024-01-01", periods=2)
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.arange(4.0)
    shape = (2, 1, 4)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
    )

    validate_level1_contract(ds)


def test_validate_level2_contract_accepts_minimal_optical_dataset() -> None:
    """A minimal Level 2 optical product should validate."""
    time = pd.date_range("2024-01-01", periods=2)
    wavelength = np.array([532])
    altitude = np.arange(4.0)
    ds = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), np.ones((1, 4))),
            "molecular_extinction": (("wavelength", "altitude"), np.ones((1, 4))),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), np.ones((2, 1, 4))),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "aerosol_extinction_mean": (("wavelength", "altitude"), np.ones((1, 4))),
        },
        coords={"time": time, "wavelength": wavelength, "altitude": altitude},
    )

    validate_level2_contract(ds)
