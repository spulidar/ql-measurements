"""Tests for LEBEAR gluing uncertainty propagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import level2_output_path
from milgrau.pipeline import lebear


class _ListLogger:
    """Small logger stub used to capture pipeline messages in tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def test_propagate_glued_error_uses_fade_weights() -> None:
    """Glued uncertainty should follow the same linear weights as the signal fade."""
    analog_error = np.ones(10, dtype=np.float64) * 2.0
    photon_error = np.ones(10, dtype=np.float64) * 10.0

    result = lebear._propagate_glued_error(  # noqa: SLF001
        analog_error=analog_error,
        photon_error=photon_error,
        slope=3.0,
        min_bin=2,
        max_bin=6,
    )

    assert np.allclose(result[:2], 6.0)
    assert np.allclose(result[6:], 10.0)
    analog_weights = 1.0 - np.arange(4, dtype=np.float64) / 4.0
    photon_weights = 1.0 - analog_weights
    expected_window = np.sqrt((analog_weights * 6.0) ** 2 + (photon_weights * 10.0) ** 2)
    assert np.allclose(result[2:6], expected_window)
    assert not np.isclose(result[5], result[6])


def _write_level1(path: Path) -> Path:
    """Write a synthetic Level 1 file where analog and PC are linearly related."""
    time = pd.date_range("2024-01-01T00:00:00", periods=2, freq="5min")
    altitude = np.arange(240, dtype=np.float64) * 7.5
    channel = np.array(["532.AN", "532.PC"], dtype=object)
    shape = (time.size, channel.size, altitude.size)

    analog = np.tile(np.linspace(10.0, 100.0, altitude.size), (time.size, 1))
    photon = 2.0 * analog + 5.0
    rcs = np.stack([analog, photon], axis=1).astype(np.float32)
    rcs_error = np.empty(shape, dtype=np.float32)
    rcs_error[:, 0, :] = 3.0
    rcs_error[:, 1, :] = 11.0
    corrected = rcs.copy()
    corrected_error = rcs_error.copy()

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={"Processing_level": "Level 1 synthetic gluing test product", "Altitude_units": "m"},
    )
    ds.to_netcdf(path)
    return path


def _config(tmp_path: Path) -> dict:
    """Return a compact LEBEAR config for gluing uncertainty tests."""
    return {
        "processing": {"incremental": False},
        "directories": {"processed_data": str(tmp_path)},
        "site": {"station_altitude_m": 760.0},
        "inversion": {
            "wavelengths_to_process": [532],
            "kfs_mode": "two_sided",
            "temporal_average_minutes": 15,
            "monte_carlo_iterations": 5,
            "random_seed": 123,
            "molecular_fit": {"ref_alt_min_m": 500.0, "ref_alt_max_m": 1500.0, "ref_window_bins": 20},
            "gluing": {
                "window_length_bins": 20,
                "correlation_threshold": 0.5,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "fallback_to_photon_counting": True,
            },
            "lidar_ratios_sr": {"532": {"01": 60.0}},
            "lidar_ratio_std_sr": {"532": 5.0},
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }


def test_level2_saves_gluing_window_diagnostics(tmp_path: Path) -> None:
    """Level 2 products should persist gluing start/stop diagnostics and weighted errors."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc")
    logger = _ListLogger()

    result = lebear.process_single_level1_file(level1, _config(tmp_path), logger)  # type: ignore[arg-type]

    assert result.startswith("[OK]")
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert "gluing_start_altitude_m" in ds
        assert "gluing_stop_altitude_m" in ds
        assert "Gluing_Error_Propagation" in ds.attrs
        assert np.isfinite(ds["gluing_start_altitude_m"].values).any()
        assert np.isfinite(ds["gluing_stop_altitude_m"].values).any()
        start = float(ds["gluing_start_altitude_m"].isel(time=0, wavelength=0))
        split = float(ds["gluing_split_altitude_m"].isel(time=0, wavelength=0))
        stop = float(ds["gluing_stop_altitude_m"].isel(time=0, wavelength=0))
        assert start < split < stop
        error_profile = ds["glued_range_corrected_signal_error"].isel(time=0, wavelength=0)
        window_error = error_profile.sel(altitude=slice(start, stop)).values
        assert np.isfinite(window_error).all()
        assert np.nanmin(window_error) > 0.0
