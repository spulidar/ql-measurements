"""Tests for Level 0 NetCDF writing and provenance metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract
from milgrau.io.netcdf import build_level0_netcdf, validate_lidar_tensors


def _config() -> dict:
    """Return a minimal Level 0 writer configuration."""
    return {
        "physics": {
            "latitude": -23.5615,
            "longitude": -46.7383,
            "vertical_resolution_m": 7.5,
            "background_start_m": 29000.0,
            "background_stop_m": 29999.0,
            "default_surface_temp_c": 25.0,
            "default_surface_pressure_hpa": 940.0,
        },
        "hardware": {
            "name_to_id": {
                "day": {"532.AN": 1593, "532.PC": 716},
                "night": {"532.AN": 722, "532.PC": 716},
            }
        },
    }


def _group_df(tmp_path: Path, include_dark_current: bool = True) -> pd.DataFrame:
    """Build a small synthetic inventory group."""
    records = [
        {
            "filepath": str(tmp_path / "meas_0001"),
            "meas_type": "measurements",
            "start_time_utc": pd.Timestamp("2024-01-01T00:00:00Z"),
            "stop_time": pd.Timestamp("2024-01-01T00:05:00Z"),
            "original_meas_id": "20231231nt",
            "association_method": "measurement",
            "dark_current_association_delta_hours": np.nan,
        },
        {
            "filepath": str(tmp_path / "meas_0002"),
            "meas_type": "measurements",
            "start_time_utc": pd.Timestamp("2024-01-01T00:05:00Z"),
            "stop_time": pd.Timestamp("2024-01-01T00:10:00Z"),
            "original_meas_id": "20231231nt",
            "association_method": "measurement",
            "dark_current_association_delta_hours": np.nan,
        },
    ]
    if include_dark_current:
        records.append(
            {
                "filepath": str(tmp_path / "dark_0001"),
                "meas_type": "dark_current",
                "start_time_utc": pd.Timestamp("2023-12-31T23:40:00Z"),
                "stop_time": pd.Timestamp("2023-12-31T23:45:00Z"),
                "original_meas_id": "20231231pm",
                "association_method": "nearest_measurement",
                "dark_current_association_delta_hours": 0.5,
            }
        )
    return pd.DataFrame.from_records(records)


def _lidar_data() -> dict:
    """Return synthetic parsed measurement tensors."""
    return {
        "channels": ["532.AN", "532.PC"],
        "shots": 1200,
        "tensors": {
            "532.AN": np.ones((2, 4), dtype=np.float64),
            "532.PC": np.ones((2, 4), dtype=np.float64) * 2.0,
        },
    }


def test_validate_lidar_tensors_rejects_shape_mismatch() -> None:
    """Level 0 tensors must share one time/range shape across channels."""
    tensors = {"532.AN": np.ones((2, 4)), "532.PC": np.ones((3, 4))}

    with pytest.raises(ValueError):
        validate_lidar_tensors(tensors, ["532.AN", "532.PC"])


def test_build_level0_netcdf_writes_dark_current_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dark-current source metadata and channel availability should be stored."""
    import milgrau.io.netcdf as netcdf_module

    def fake_parse_licel_group(files: list[str], logger: logging.Logger) -> dict:
        return {
            "channels": ["532.AN", "532.PC"],
            "tensors": {
                "532.AN": np.ones((1, 4), dtype=np.float64) * 0.1,
                "532.PC": np.ones((1, 4), dtype=np.float64) * 0.2,
            },
        }

    monkeypatch.setattr(netcdf_module, "parse_licel_group", fake_parse_licel_group)
    output_path = tmp_path / "level0.nc"

    build_level0_netcdf(
        netcdf_path=str(output_path),
        save_id="20240101sant",
        period="nt",
        lidar_data=_lidar_data(),
        group_df=_group_df(tmp_path, include_dark_current=True),
        weather_data={"temperature_c": 23.0, "pressure_hpa": 935.0},
        config=_config(),
        logger=logging.getLogger("test"),
    )

    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert "Background_Profile" in ds
        assert "Background_Profile_Available" in ds
        assert ds.attrs["Dark_Current_Source_File_Count"] == 1
        assert ds.attrs["Dark_Current_Association_Methods"] == "nearest_measurement"
        assert float(ds.attrs["Dark_Current_Max_Association_Delta_hours"]) == 0.5
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 1], dtype=np.int8))


def test_build_level0_netcdf_flags_missing_dark_current_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dark-current channels should be NaN-filled and flagged unavailable."""
    import milgrau.io.netcdf as netcdf_module

    def fake_parse_licel_group(files: list[str], logger: logging.Logger) -> dict:
        return {
            "channels": ["532.AN"],
            "tensors": {"532.AN": np.ones((1, 4), dtype=np.float64) * 0.1},
        }

    monkeypatch.setattr(netcdf_module, "parse_licel_group", fake_parse_licel_group)
    output_path = tmp_path / "level0_missing_dc_channel.nc"

    build_level0_netcdf(
        netcdf_path=str(output_path),
        save_id="20240101sant",
        period="nt",
        lidar_data=_lidar_data(),
        group_df=_group_df(tmp_path, include_dark_current=True),
        weather_data={"temperature_c": 23.0, "pressure_hpa": 935.0},
        config=_config(),
        logger=logging.getLogger("test"),
    )

    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 0], dtype=np.int8))
        assert np.all(np.isnan(ds["Background_Profile"].isel(channels=1).values))


def test_build_level0_netcdf_without_dark_current_writes_unavailable_flags(tmp_path: Path) -> None:
    """Products without dark-current files should still expose availability flags."""
    output_path = tmp_path / "level0_no_dc.nc"

    build_level0_netcdf(
        netcdf_path=str(output_path),
        save_id="20240101sant",
        period="nt",
        lidar_data=_lidar_data(),
        group_df=_group_df(tmp_path, include_dark_current=False),
        weather_data={"temperature_c": 23.0, "pressure_hpa": 935.0},
        config=_config(),
        logger=logging.getLogger("test"),
    )

    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert "Background_Profile" not in ds
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([0, 0], dtype=np.int8))
        assert ds.attrs["Dark_Current_Source_File_Count"] == 0
