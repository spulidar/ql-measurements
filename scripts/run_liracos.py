"""Command-line entry point for MILGRAU LIRACOS Level 1 visualization."""

from __future__ import annotations

from pathlib import Path

from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.liracos import process_all_level1_files


def main() -> None:
    """Run LIRACOS from the command line."""
    config = load_config()
    logger = setup_logger("LIRACOS", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LIRACOS rendering (Level 1 Visualization) ===")
    process_all_level1_files(config=config, logger=logger, root_dir=Path.cwd())
    logger.info("=== LIRACOS processing finished! ===")


if __name__ == "__main__":
    main()
