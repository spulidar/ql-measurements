"""Tests for LEBEAR round-1 architecture and traceability behavior."""

from __future__ import annotations

import logging
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


def _write_level1(path: Path, channels: list[str], n_altitude: int = 240) -> Path:
    """Write a synthetic Level 1 file with smooth finite RCS profiles."""
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(n_altitude, dtype=np.float64) * 7.5
    channel = np.array(channels, dtype=object)
    shape = (time.size, channel.size, altitude.size)

    base = np.exp(-altitude / 900.0) + 0.1
    corrected = np.empty(shape, dtype=np.float32)
    corrected_error = np.empty(shape, dtype=np.float32)
    rcs = np.empty(shape, dtype=np.float32)
    rcs_error = np.empty(shape, dtype=np.float32)
    for time_idx in range(time.size):
        for channel_idx, channel_name in enumerate(channels):
            scale = 1.0 + 0.05 * time_idx + 0.02 * channel_idx
            if channel_name.endswith(".PC"):
                scale *= 1.05
            corrected[time_idx, channel_idx, :] = scale * base
            corrected_error[time_idx, channel_idx, :] = 0.02 * np.abs(corrected[time_idx, channel_idx, :])
            rcs[time_idx, channel_idx, :] = corrected[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2
            rcs_error[time_idx, channel_idx, :] = corrected_error[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={"Processing_level": "Level 1 synthetic test product", "Altitude_units": "m"},
    )
    ds.to_netcdf(path)
    return path


def _config(tmp_path: Path, *, fallback_to_photon_counting: bool = True, kfs_mode: str = "two_sided") -> dict:
    """Return a compact LEBEAR config for synthetic tests."""
    return {
        "processing": {"incremental": True},
        "directories": {"processed_data": str(tmp_path)},
        "site": {"station_altitude_m": 760.0},
        "inversion": {
            "wavelengths_to_process": [532],
            "kfs_mode": kfs_mode,
            "temporal_average_minutes": 15,
            "monte_carlo_iterations": 5,
            "random_seed": 123,
            "molecular_fit": {"ref_alt_min_m": 500.0, "ref_alt_max_m": 1500.0, "ref_window_bins": 20},
            "gluing": {
                "window_length_bins": 20,
                "correlation_threshold": 0.5,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "fallback_to_photon_counting": fallback_to_photon_counting,
            },
            "lidar_ratios_sr": {"532": {"01": 60.0}},
            "lidar_ratio_std_sr": {"532": 5.0},
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }


def test_lebear_saves_real_kfs_mode_and_branch_flags(tmp_path: Path) -> None:
    """Level 2 output should preserve the configured KFS mode and branch traceability."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()

    result = lebear.process_single_level1_file(level1, _config(tmp_path, kfs_mode="two_sided"), logger)  # type: ignore[arg-type]

    assert result.startswith("[OK]")
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert ds.attrs["KFS_Mode"] == "two_sided"
        assert "experimental" in ds.attrs["KFS_Mode_Description"]
        assert "kfs_branch" in ds
        branch_values = set(np.asarray(ds["kfs_branch"].values).ravel().astype(int).tolist())
        assert 1 in branch_values
        assert 2 in branch_values
        assert 3 in branch_values
        assert "gluing_fallback_flag" in ds


def test_gluing_failure_respects_disabled_photon_fallback(tmp_path: Path) -> None:
    """When photon fallback is disabled, failed gluing should make the wavelength fail."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()
    config = _config(tmp_path, fallback_to_photon_counting=False)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1

    result = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]

    assert result.startswith("[FAILED]")
    assert not level2_output_path(level1).exists()
    assert any("fallback is disabled" in message for message in logger.messages)


def test_gluing_failure_uses_photon_fallback_when_enabled(tmp_path: Path) -> None:
    """When photon fallback is enabled, failed gluing should be flagged and processed."""
    level1 = _write_level1(tmp_path / "synthetic_level1_rcs.nc", ["532.AN", "532.PC"])
    logger = _ListLogger()
    config = _config(tmp_path, fallback_to_photon_counting=True)
    config["inversion"]["gluing"]["correlation_threshold"] = 1.1

    result = lebear.process_single_level1_file(level1, config, logger)  # type: ignore[arg-type]

    assert result.startswith("[OK]")
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert int(ds["gluing_success_flag"].sum()) == 0
        assert int(ds["gluing_fallback_flag"].sum()) == ds.sizes["time"]


def test_process_level2_skips_existing_output_when_incremental(tmp_path: Path, monkeypatch) -> None:
    """LEBEAR process_level_2 should skip existing Level 2 files in incremental mode."""
    level1 = _write_level1(tmp_path / "20240101sant_level1_rcs.nc", ["532.AN", "532.PC"])
    output = level2_output_path(level1)
    output.write_text("existing", encoding="utf-8")
    logger = _ListLogger()
    calls = {"count": 0}

    def fake_process_single_level1_file(nc_file: str | Path, config: dict, logger: logging.Logger) -> str:
        calls["count"] += 1
        return "[OK] should not run"

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process_single_level1_file)

    lebear.process_level_2(_config(tmp_path), logger)  # type: ignore[arg-type]

    assert calls["count"] == 0
    assert any("SKIPPED" in message for message in logger.messages)
