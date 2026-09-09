"""Application logs are JSON on stderr; command results use stdout."""

import logging
import sys

import structlog

from demon_plus.config import LogLevel


def configure_logging(level: LogLevel) -> None:
    """Configure our application logger without logging settings or payloads."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
