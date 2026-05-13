"""Command-line entry point for MILGRAU LEBEAR Level 2 processing."""

from __future__ import annotations

from pathlib import Path

from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.lebear import LEVEL2_SUFFIX, discover_level1_files, process_single_level1_file


def _incremental_enabled(config: dict) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def _level2_output_path(level1_file: Path) -> Path:
    """Return the expected Level 2 output path for one Level 1 file."""
    stem = level1_file.name.replace("_level1_rcs.nc", "")
    return level1_file.parent / f"{stem}{LEVEL2_SUFFIX}"


def main() -> None:
    """Run LEBEAR from the command line."""
    config = load_config()
    logger = setup_logger("LEBEAR", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LEBEAR processing (Level 2) ===")

    files = discover_level1_files(config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        logger.info("=== LEBEAR finished. ===")
        return

    incremental = _incremental_enabled(config)
    files_to_process: list[Path] = []
    skipped_count = 0
    for file_path in files:
        output_path = _level2_output_path(file_path)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 2 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 1 files require Level 2 processing. Skipped {skipped_count} existing products.")
        logger.info("=== LEBEAR finished. ===")
        return

    logger.info(f"Found {len(files_to_process)} Level 1 files to process ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_level1_file(file_path, config, logger))

    logger.info("=== LEBEAR finished. ===")


if __name__ == "__main__":
    main()
