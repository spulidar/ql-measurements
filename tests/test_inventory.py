"""Tests for Level 0 inventory construction and dark-current association."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from milgrau.io.inventory import build_measurement_inventory


def test_inventory_reassigns_orphan_dark_current_with_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Orphan dark-current files should be linked to nearby measurement groups."""
    import milgrau.io.inventory as inventory_module

    measurement_path = str(tmp_path / "measurement_file")
    dark_path = str(tmp_path / "dark_file")

    def fake_scan_raw_files(raw_dir: str, logger: logging.Logger, config: dict) -> tuple[list[str], list[str]]:
        return [measurement_path, dark_path], ["measurements", "dark_current"]

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        if filepath == measurement_path:
            return datetime(2024, 1, 1, 12, 0, 0), pd.Timestamp("2024-01-01T12:05:00"), 300.0, 1200, 10.0
        return datetime(2024, 1, 1, 18, 0, 0), pd.Timestamp("2024-01-01T18:05:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)

    config = {
        "site": {"timezone": "America/Sao_Paulo"},
        "processing": {"dark_current_max_association_hours": 12.0},
    }
    df = build_measurement_inventory(str(tmp_path), config, logging.getLogger("test"))

    assert len(df) == 2
    measurement_id = df.loc[df["meas_type"] == "measurements", "meas_id"].iloc[0]
    dark_row = df.loc[df["meas_type"] == "dark_current"].iloc[0]

    assert dark_row["meas_id"] == measurement_id
    assert dark_row["original_meas_id"] != measurement_id
    assert dark_row["association_method"] == "nearest_measurement"
    assert float(dark_row["dark_current_association_delta_hours"]) == 6.0


def test_inventory_keeps_incremental_decision_out_of_inventory(tmp_path: Path, monkeypatch) -> None:
    """Inventory should not drop groups simply because incremental mode is enabled."""
    import milgrau.io.inventory as inventory_module

    measurement_path = str(tmp_path / "measurement_file")

    def fake_scan_raw_files(raw_dir: str, logger: logging.Logger, config: dict) -> tuple[list[str], list[str]]:
        return [measurement_path], ["measurements"]

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        return datetime(2024, 1, 1, 12, 0, 0), pd.Timestamp("2024-01-01T12:05:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)

    config = {
        "site": {"timezone": "America/Sao_Paulo"},
        "processing": {"incremental": True},
        "directories": {"processed_data": str(tmp_path / "processed")},
    }
    df = build_measurement_inventory(str(tmp_path), config, logging.getLogger("test"))

    assert len(df) == 1
    assert df.iloc[0]["meas_type"] == "measurements"
