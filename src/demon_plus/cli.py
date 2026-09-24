"""Small command-line entry point for the first milestone."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version

import structlog
from pydantic import ValidationError
from pydantic_settings import SettingsError

from demon_plus.config import Settings
from demon_plus.errors import DataError
from demon_plus.logging import configure_logging
from demon_plus.schemas import ArtifactResult, CheckResult


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration; exit 2 on expected configuration errors."""
    parser = argparse.ArgumentParser(description="Demon Plus local tools")
    parser.add_argument("command", choices=["check", "fetch-data", "baseline"])
    args = parser.parse_args(argv)

    configure_logging("INFO")
    logger = structlog.get_logger()
    try:
        settings = Settings()
    except (ValidationError, SettingsError, OSError, UnicodeError):
        # Validation errors may contain input values. Do not serialize them or .env.
        logger.error(
            "configuration_invalid",
            message="Invalid configuration. Check .env against .env.example and retry.",
        )
        return 2

    configure_logging(settings.log_level)
    if args.command in ("fetch-data", "baseline"):
        try:
            if args.command == "fetch-data":
                from demon_plus.data.download import fetch_olist

                artifact = fetch_olist(settings.data_dir)
            else:
                from demon_plus.evaluation.run import run_baselines

                artifact = run_baselines(settings)
        except DataError as exc:
            logger.error("data_invalid", message=str(exc))
            return 2
        except OSError:
            logger.error(
                "io_error", message="Cannot access data files; check paths and permissions."
            )
            return 2
        logger.info("pipeline_completed", command=args.command)
        print(ArtifactResult(command=args.command, artifact_path=str(artifact)).model_dump_json())
        return 0
    result = CheckResult(package_version=version("demon-plus"))
    logger.info("configuration_checked", package_version=result.package_version)
    print(result.model_dump_json())
    return 0
