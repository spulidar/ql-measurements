"""Configuration loader for MILGRAU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from milgrau.config.schema import validate_config_minimum


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and minimally validate the MILGRAU YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing YAML configuration: {exc}") from exc

    if config is None:
        raise RuntimeError(f"Configuration file is empty: {path}")

    validate_config_minimum(config)
    return config
