"""Tests for MILGRAU configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from milgrau.config.loader import load_config, normalize_config
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


def test_load_repository_config_injects_legacy_aliases() -> None:
    """Canonical YAML keys should be exposed with compatibility aliases in memory."""
    config = load_config("config.yaml")

    assert config["physics"]["speed_of_light"] == config["physics"]["speed_of_light_m_s"]
    assert config["physics"]["bg_start"] == config["physics"]["background_start_m"]
    assert config["physics"]["bg_stop"] == config["physics"]["background_stop_m"]
    assert config["radiosonde"]["fallback_to_standard"] is config["radiosonde"]["fallback_to_standard_atmosphere"]
    assert config["inversion"]["lidar_ratios"] == config["inversion"]["lidar_ratios_sr"]


def test_minimum_schema_rejects_missing_sections() -> None:
    """The lightweight schema should fail on incomplete configs."""
    with pytest.raises(KeyError):
        validate_config_minimum({"processing": {}, "physics": {}})


def test_schema_rejects_invalid_background_window() -> None:
    """Background stop altitude must be greater than the start altitude."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": True},
        "physics": {
            "vertical_resolution_m": 7.5,
            "background_start_m": 30000.0,
            "background_stop_m": 29000.0,
            "channels": {"532.PC": [0.0035, -3, 0.0015]},
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }

    with pytest.raises(ValueError):
        validate_config_minimum(config)


def test_normalize_config_preserves_existing_aliases() -> None:
    """Explicit legacy aliases should not be overwritten during normalization."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": False},
        "physics": {
            "vertical_resolution_m": 7.5,
            "speed_of_light_m_s": 299792458.0,
            "speed_of_light": 1.0,
            "background_start_m": 29000.0,
            "background_stop_m": 30000.0,
            "channels": {"532.PC": [0.0035, -3, 0.0015]},
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }

    normalized = normalize_config(config)

    assert normalized["physics"]["speed_of_light"] == 1.0
    assert normalized["physics"]["bg_start"] == 29000.0
    assert normalized["physics"]["bg_stop"] == 30000.0


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
