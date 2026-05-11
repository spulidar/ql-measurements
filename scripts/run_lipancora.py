"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.lipancora import process_level_1


def main() -> None:
    """Run LIPANCORA from the command line."""
    config = load_config()
    logger = setup_logger("LIPANCORA", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LIPANCORA processing (Level 1) ===")
    process_level_1(config, logger)
    logger.info("=== LIPANCORA finished processing all files. ===")


if __name__ == "__main__":
    main()
