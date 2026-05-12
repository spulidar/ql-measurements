"""Synthetic NetCDF contract tests for Level 1 and LEBEAR scaffold inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.pipeline.lebear import process_single_level1_file


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


def _write_synthetic_level1(path: Path) -> Path:
    """Write a tiny Level 1 NetCDF product for contract testing."""
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(0.0, 1500.0, 7.5)
    channel = np.array(["532.AN", "532.PC"], dtype=object)

    shape = (time.size, channel.size, altitude.size)
    base_profile = np.exp(-altitude / 1000.0)
    corrected_signal = np.empty(shape, dtype=np.float32)
    corrected_signal_error = np.empty(shape, dtype=np.float32)
    range_corrected_signal = np.empty(shape, dtype=np.float32)
    range_corrected_signal_error = np.empty(shape, dtype=np.float32)

    for t_idx in range(time.size):
        for c_idx in range(channel.size):
            scale = 1.0 + 0.1 * t_idx + 0.05 * c_idx
            corrected_signal[t_idx, c_idx, :] = scale * base_profile
            corrected_signal_error[t_idx, c_idx, :] = 0.05 * np.abs(corrected_signal[t_idx, c_idx, :])
            range_corrected_signal[t_idx, c_idx, :] = corrected_signal[t_idx, c_idx, :] * altitude**2
            range_corrected_signal_error[t_idx, c_idx, :] = corrected_signal_error[t_idx, c_idx, :] * altitude**2

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected_signal),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_signal_error),
            "range_corrected_signal": (("time", "channel", "altitude"), range_corrected_signal),
            "range_corrected_signal_error": (("time", "channel", "altitude"), range_corrected_signal_error),
            "PBL_Height_km": (("time",), np.array([0.8, 0.9, 1.0], dtype=np.float32)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={
            "Processing_level": "Level 1 synthetic test product",
            "Altitude_units": "m",
            "tropopause_cpt_km": -999.0,
            "tropopause_lrt_km": -999.0,
        },
    )
    ds["corrected_signal"].attrs["units"] = "channel native corrected units"
    ds["corrected_signal_error"].attrs["units"] = "channel native corrected units"
    ds["range_corrected_signal"].attrs["units"] = "a.u. m^2"
    ds["range_corrected_signal_error"].attrs["units"] = "a.u. m^2"
    ds["altitude"].attrs["units"] = "m"
    ds.to_netcdf(path)
    return path


def test_synthetic_level1_contract(tmp_path: Path) -> None:
    """A synthetic Level 1 file should expose the variables expected by LEBEAR."""
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")

    with xr.open_dataset(path) as ds:
        assert "corrected_signal" in ds
        assert "corrected_signal_error" in ds
        assert "range_corrected_signal" in ds
        assert "range_corrected_signal_error" in ds
        assert "altitude" in ds.coords
        assert ds["range_corrected_signal"].dims == ("time", "channel", "altitude")
        assert ds["range_corrected_signal_error"].shape == ds["range_corrected_signal"].shape
        assert float(ds["altitude"].max()) > 100.0


def test_lebear_scaffold_accepts_synthetic_level1(tmp_path: Path) -> None:
    """The current LEBEAR scaffold should validate a minimal Level 1 file."""
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")
    logger = _ListLogger()
    config = {"directories": {"processed_data": str(tmp_path)}}

    result = process_single_level1_file(path, config, logger)  # type: ignore[arg-type]

    assert result.startswith("[PENDING]")
    assert any("validated" in message.lower() for message in logger.messages)
