"""Tests for LIPANCORA Level 1 correction diagnostics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.physics.corrections import apply_instrumental_corrections
from milgrau.pipeline.lipancora import apply_all_physical_corrections


def test_apply_instrumental_corrections_marks_shifted_bins_as_nan() -> None:
    """Bin-shift introduced samples should be NaN instead of artificial signal values."""
    time = pd.date_range("2024-01-01", periods=2)
    raw = xr.DataArray(
        np.arange(10, dtype=np.float64).reshape(2, 5) + 100.0,
        dims=("time", "range"),
        coords={"time": time, "range": np.arange(5)},
    )
    z_da = xr.DataArray(np.arange(5, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, False, True, True]), dims=["range"])

    corrected, corrected_error, rcs, rcs_error, diagnostics = apply_instrumental_corrections(
        sig=raw,
        z_da=z_da,
        shots=1000.0,
        bin_time_us=0.05,
        deadtime=0.0,
        shift=2,
        bg_offset=0.0,
        is_photon=False,
        bg_mask=bg_mask,
        return_diagnostics=True,
    )

    assert np.all(np.isnan(corrected.isel(range=slice(0, 2)).values))
    assert np.all(np.isnan(rcs.isel(range=slice(0, 2)).values))
    assert np.allclose(diagnostics["bin_shift_invalid_fraction"].values, np.array([0.4, 0.4]))
    assert corrected_error.shape == corrected.shape
    assert rcs_error.shape == rcs.shape


def test_apply_instrumental_corrections_reports_deadtime_clipping() -> None:
    """Photon-counting dead-time denominator clipping should be explicitly reported."""
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(
        np.array([[900.0, 900.0, 10.0, 10.0, 10.0]], dtype=np.float64),
        dims=("time", "range"),
        coords={"time": time, "range": np.arange(5)},
    )
    z_da = xr.DataArray(np.arange(5, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, False, True, True]), dims=["range"])

    *_signals, diagnostics = apply_instrumental_corrections(
        sig=raw,
        z_da=z_da,
        shots=1.0,
        bin_time_us=1.0,
        deadtime=0.0035,
        shift=0,
        bg_offset=0.0,
        is_photon=True,
        bg_mask=bg_mask,
        deadtime_min_denominator=0.05,
        return_diagnostics=True,
    )

    assert diagnostics["deadtime_correction_applied"] is True
    assert np.isclose(float(diagnostics["deadtime_clipping_fraction"].values[0]), 0.4)
    assert np.isclose(diagnostics["deadtime_min_denominator_allowed"], 0.05)


def test_apply_all_physical_corrections_persists_diagnostics() -> None:
    """The LIPANCORA pipeline should persist correction diagnostics in Level 1 output."""
    time = pd.date_range("2024-01-01", periods=2)
    altitude = np.arange(5, dtype=np.float64) * 7.5
    channels = np.array(["532.PC", "532.AN"], dtype=object)
    raw_data = np.ones((2, 2, 5), dtype=np.float64) * 10.0
    raw_data[:, 0, :2] = 900.0

    ds = xr.Dataset(
        data_vars={
            "Raw_Lidar_Data": (("time", "channel", "altitude"), raw_data),
            "Background_Low": (("channel",), np.array([22.5, 22.5])),
            "Background_High": (("channel",), np.array([30.0, 30.0])),
        },
        coords={"time": time, "channel": channels, "altitude": altitude},
        attrs={"Accumulated_Shots": 1},
    )
    config = {
        "physics": {
            "speed_of_light": 299792458.0,
            "channels": {
                "532.PC": [0.0035, 0, 0.0],
                "532.AN": [0.0, 2, 0.0],
            },
        }
    }

    result = apply_all_physical_corrections(ds, altitude, config, logging.getLogger("test"))

    assert "deadtime_clipping_fraction" in result
    assert "deadtime_correction_applied" in result
    assert "bin_shift_invalid_fraction" in result
    assert "bin_shift_bins" in result
    assert int(result["deadtime_correction_applied"].sel(channel="532.PC")) == 1
    assert int(result["deadtime_correction_applied"].sel(channel="532.AN")) == 0
    assert float(result["deadtime_clipping_fraction"].sel(channel="532.PC").max()) > 0.0
    assert np.isclose(float(result["bin_shift_invalid_fraction"].sel(channel="532.AN").max()), 0.4)
    assert np.all(np.isnan(result["corrected_signal"].sel(channel="532.AN").isel(altitude=slice(0, 2)).values))
