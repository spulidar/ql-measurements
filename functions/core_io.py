"""Compatibility layer for legacy imports from functions.core_io.

New code should import from the milgrau package directly:
- milgrau.config.loader
- milgrau.io.filesystem
- milgrau.io.inventory
- milgrau.io.licel
- milgrau.io.logging_utils
- milgrau.io.radiosonde
- milgrau.io.weather
"""

from __future__ import annotations

from milgrau.config.loader import load_config
from milgrau.io.filesystem import ensure_directories, scan_raw_files
from milgrau.io.inventory import build_measurement_inventory
from milgrau.io.licel import parse_licel_group, parse_single_licel_file, read_licel_header
from milgrau.io.logging_utils import setup_logger
from milgrau.io.radiosonde import fetch_wyoming_radiosonde
from milgrau.io.weather import fetch_surface_weather, return_none_on_failure

# Backward-compatible private name used by older experimental code.
_parse_single_licel_file = parse_single_licel_file

__all__ = [
    "_parse_single_licel_file",
    "build_measurement_inventory",
    "ensure_directories",
    "fetch_surface_weather",
    "fetch_wyoming_radiosonde",
    "load_config",
    "parse_licel_group",
    "read_licel_header",
    "return_none_on_failure",
    "scan_raw_files",
    "setup_logger",
]
