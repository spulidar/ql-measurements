"""Lightweight configuration schema validation for MILGRAU.

The project is still evolving scientifically, so this module intentionally uses
minimal validation instead of a rigid schema. It catches missing sections and
obvious structural problems while keeping experimental Level 2 settings flexible.
"""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_TOP_LEVEL_SECTIONS = ("directories", "processing", "physics", "hardware")


def validate_config_minimum(config: Mapping) -> None:
    """Validate that the minimum required MILGRAU config sections exist."""
    missing = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise KeyError(
            "Configuration file is missing required section(s): "
            + ", ".join(missing)
        )

    directories = config.get("directories", {})
    for key in ("raw_data", "processed_data", "log_dir"):
        if key not in directories:
            raise KeyError(f"Configuration directories.{key} is required.")

    physics = config.get("physics", {})
    if "vertical_resolution_m" not in physics:
        raise KeyError("Configuration physics.vertical_resolution_m is required.")
    if "channels" not in physics:
        raise KeyError("Configuration physics.channels is required.")
