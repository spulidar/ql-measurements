"""Filesystem helpers for MILGRAU raw-data discovery and safe sanitization."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_directories(*directories: str | Path) -> None:
    """Create one or more directories if they do not already exist."""
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.error(f"Failed to create directory {directory}: {exc}")


def _processing_option(config: Optional[dict], key: str, default):
    """Return an optional processing setting from config."""
    if not config:
        return default
    return config.get("processing", {}).get(key, default)


def quarantine_file(path: Path, quarantine_root: Path, logger: Optional[logging.Logger]) -> None:
    """Move a spurious file to a quarantine folder instead of deleting it."""
    ensure_directories(quarantine_root)
    destination = quarantine_root / path.name
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = quarantine_root / f"{path.stem}_{stamp}{path.suffix}"
    shutil.move(str(path), str(destination))
    if logger:
        logger.info(f"  -> Spurious file quarantined: {path.name} -> {destination}")


def scan_raw_files(
    datadir_name: str,
    logger: Optional[logging.Logger] = None,
    config: Optional[dict] = None,
) -> tuple[list[str], list[str]]:
    """Scan the raw-data tree for Licel files and classify dark-current files."""
    filepath: list[str] = []
    meas_type: list[str] = []

    raw_root = Path(datadir_name)
    if not raw_root.exists():
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    spurious_extensions = tuple(
        ext.lower()
        for ext in _processing_option(config, "spurious_extensions", [".dat", ".dpp", ".zip"])
    )
    quarantine_spurious = bool(_processing_option(config, "quarantine_spurious_files", True))
    delete_spurious = bool(_processing_option(config, "delete_spurious_files", False))
    quarantine_dir = Path(_processing_option(config, "quarantine_dir", str(raw_root / "_quarantine")))

    for dirpath, dirnames, files in os.walk(raw_root):
        dirnames.sort()
        files.sort()

        try:
            quarantine_resolved = quarantine_dir.resolve()
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if Path(dirpath, dirname).resolve() != quarantine_resolved
            ]
        except Exception:
            pass

        for file_name in files:
            full_path = Path(dirpath) / file_name
            suffix = full_path.suffix.lower()

            if suffix in spurious_extensions:
                try:
                    if delete_spurious:
                        full_path.unlink()
                        if logger:
                            logger.info(f"  -> Spurious file deleted: {full_path.name}")
                    elif quarantine_spurious:
                        quarantine_file(full_path, quarantine_dir, logger)
                    else:
                        if logger:
                            logger.debug(f"  -> Spurious file ignored: {full_path.name}")
                except Exception as exc:
                    if logger:
                        logger.warning(f"Could not handle spurious file {full_path}: {exc}")
                continue

            filepath.append(str(full_path))
            meas_type.append("dark_current" if "dark" in str(full_path).lower() else "measurements")

    return filepath, meas_type
