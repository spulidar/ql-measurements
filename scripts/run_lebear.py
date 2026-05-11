"""Command-line entry point for MILGRAU LEBEAR Level 2 processing."""

from __future__ import annotations

from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.lebear import process_level_2


def main() -> None:
    """Run the current LEBEAR scaffold from the command line."""
    config = load_config()
    logger = setup_logger("LEBEAR", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LEBEAR processing (Level 2 scaffold) ===")
    process_level_2(config, logger)
    logger.info("=== LEBEAR scaffold finished. ===")


if __name__ == "__main__":
    main()
