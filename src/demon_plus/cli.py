"""Small command-line entry point for the first milestone."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version

import structlog
from pydantic import ValidationError

from demon_plus.config import Settings
from demon_plus.logging import configure_logging
from demon_plus.schemas import CheckResult


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration; exit 2 on expected configuration errors."""
    parser = argparse.ArgumentParser(description="Demon Plus local tools")
    parser.add_argument("command", choices=["check"], help="Validate local configuration")
    parser.parse_args(argv)

    configure_logging("INFO")
    logger = structlog.get_logger()
    try:
        settings = Settings()
    except (ValidationError, OSError, UnicodeError):
        # Validation errors may contain input values. Do not serialize them or .env.
        logger.error(
            "configuration_invalid",
            message="Invalid configuration. Check .env against .env.example and retry.",
        )
        return 2

    configure_logging(settings.log_level)
    result = CheckResult(package_version=version("demon-plus"))
    logger.info("configuration_checked", package_version=result.package_version)
    print(result.model_dump_json())
    return 0
