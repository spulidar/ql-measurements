"""Tests for MILGRAU configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from milgrau.config.loader import load_config
from milgrau.config.schema import validate_config_minimum


def test_load_repository_config() -> None:
    """The repository config.yaml should load and expose required sections."""
    config = load_config("config.yaml")

    assert "directories" in config
    assert "processing" in config
    assert "physics" in config
    assert "hardware" in config
    assert config["physics"]["vertical_resolution_m"] > 0
    assert "532.PC" in config["physics"]["channels"]


def test_minimum_schema_rejects_missing_sections() -> None:
    """The lightweight schema should fail on incomplete configs."""
    with pytest.raises(KeyError):
        validate_config_minimum({"processing": {}, "physics": {}})


def test_load_minimal_valid_config_from_tmp_path(tmp_path: Path) -> None:
    """A minimal valid YAML file should load through the public loader."""
    config_path = tmp_path / "config.yaml"
    config_payload = {
        "directories": {
            "raw_data": "raw",
            "processed_data": "processed",
            "log_dir": "logs",
        },
        "processing": {"incremental": False},
        "physics": {
            "vertical_resolution_m": 7.5,
            "channels": {"532.PC": [0.0035, -3, 0.0015]},
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    config = load_config(config_path)

    assert config["directories"]["raw_data"] == "raw"
    assert config["physics"]["channels"]["532.PC"][0] == 0.0035
