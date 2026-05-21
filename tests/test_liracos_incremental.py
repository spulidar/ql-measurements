"""Tests for LIRACOS incremental plotting behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.pipeline import liracos
from milgrau.visualization.quicklooks import _insert_time_gap_markers


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


def _write_level1(path: Path, channels: list[str]) -> Path:
    """Write a tiny Level 1 dataset suitable for LIRACOS tests."""
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(0.0, 1500.0, 7.5)
    channel = np.array(channels, dtype=object)
    shape = (time.size, channel.size, altitude.size)
    profile = np.exp(-altitude / 1000.0).astype(np.float32)
    corrected = np.zeros(shape, dtype=np.float32)
    corrected_error = np.zeros(shape, dtype=np.float32)
    rcs = np.zeros(shape, dtype=np.float32)
    rcs_error = np.zeros(shape, dtype=np.float32)
    for time_idx in range(time.size):
        for channel_idx in range(channel.size):
            scale = 1.0 + 0.1 * time_idx + 0.05 * channel_idx
            corrected[time_idx, channel_idx, :] = scale * profile
            corrected_error[time_idx, channel_idx, :] = 0.05 * corrected[time_idx, channel_idx, :]
            rcs[time_idx, channel_idx, :] = corrected[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2
            rcs_error[time_idx, channel_idx, :] = corrected_error[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
            "PBL_Height_km": (("time",), np.array([0.7, 0.8, 0.9], dtype=np.float32)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={"tropopause_cpt_km": -999.0, "tropopause_lrt_km": -999.0},
    )
    ds.to_netcdf(path)
    return path


def _config(channels: list[str], incremental: bool = True) -> dict:
    """Return a minimal LIRACOS config."""
    return {
        "processing": {"incremental": incremental},
        "directories": {"processed_data": "02-processed_data"},
        "visualization": {
            "output_format": "png",
            "dpi": 60,
            "altitude_ranges_km": [1.0],
            "channels_to_plot": channels,
            "quicklook": {"max_time_gap_minutes": 10, "missing_data_color": "lightgray", "colormap": "viridis"},
        },
    }


def test_time_gap_markers_insert_nan_profiles() -> None:
    """Large temporal gaps should be represented by inserted NaN profiles."""
    times = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:05:00", "2024-01-01T00:40:00"])
    altitude = np.array([0.0, 0.5, 1.0])
    data = xr.DataArray(
        np.ones((3, 3)),
        dims=("time", "altitude"),
        coords={"time": times, "altitude": altitude},
    )

    result = _insert_time_gap_markers(data, _config(["532.AN"]))

    assert result.sizes["time"] == 5
    assert np.isnan(result.isel(time=2).values).all()
    assert np.isnan(result.isel(time=3).values).all()


def test_global_mean_manifest_skips_current_plot(tmp_path: Path, monkeypatch) -> None:
    """Current global mean plots should be skipped when signature matches."""
    level1 = _write_level1(tmp_path / "20240101sant_level1_rcs.nc", ["532.AN"])
    logger = _ListLogger()
    calls = {"quicklook": 0, "global": 0}

    def fake_quicklook(**kwargs):
        calls["quicklook"] += 1
        out_path = Path(kwargs["output_folder"]) / "fake_quicklook.png"
        out_path.write_text("quicklook", encoding="utf-8")
        return out_path

    def fake_global(ds, output_folder, file_name_prefix, config, root_dir):
        calls["global"] += 1
        out_path = Path(output_folder) / f"GlobalMeanRCS_{file_name_prefix}.png"
        out_path.write_text("global", encoding="utf-8")
        return out_path

    monkeypatch.setattr(liracos, "plot_quicklook", fake_quicklook)
    monkeypatch.setattr(liracos, "plot_global_mean_rcs", fake_global)

    first = liracos.process_single_nc((level1, _config(["532.AN"], incremental=True), tmp_path, logger))
    second = liracos.process_single_nc((level1, _config(["532.AN"], incremental=True), tmp_path, logger))

    assert first.startswith("[OK]")
    assert second.startswith("[OK]")
    assert calls["global"] == 1
    assert any("Global mean RCS is current" in message for message in logger.messages)


def test_global_mean_regenerates_when_channel_config_changes(tmp_path: Path, monkeypatch) -> None:
    """Changing configured channels should invalidate the global mean manifest."""
    level1 = _write_level1(tmp_path / "20240101sant_level1_rcs.nc", ["532.AN", "355.AN"])
    logger = _ListLogger()
    calls = {"quicklook": 0, "global": 0}

    def fake_quicklook(**kwargs):
        calls["quicklook"] += 1
        out_path = Path(kwargs["output_folder"]) / f"fake_quicklook_{calls['quicklook']}.png"
        out_path.write_text("quicklook", encoding="utf-8")
        return out_path

    def fake_global(ds, output_folder, file_name_prefix, config, root_dir):
        calls["global"] += 1
        out_path = Path(output_folder) / f"GlobalMeanRCS_{file_name_prefix}.png"
        out_path.write_text(f"global {calls['global']}", encoding="utf-8")
        return out_path

    monkeypatch.setattr(liracos, "plot_quicklook", fake_quicklook)
    monkeypatch.setattr(liracos, "plot_global_mean_rcs", fake_global)

    liracos.process_single_nc((level1, _config(["532.AN"], incremental=True), tmp_path, logger))
    liracos.process_single_nc((level1, _config(["532.AN", "355.AN"], incremental=True), tmp_path, logger))

    assert calls["global"] == 2
