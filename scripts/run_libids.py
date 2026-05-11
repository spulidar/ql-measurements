"""Command-line entry point for MILGRAU LIBIDS Level 0 processing."""

from __future__ import annotations

from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.libids import process_level_0


def main() -> None:
    """Run LIBIDS from the command line."""
    config = load_config()
    logger = setup_logger("LIBIDS", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LIBIDS processing (Level 0) ===")
    process_level_0(config, logger)


if __name__ == "__main__":
    main()
