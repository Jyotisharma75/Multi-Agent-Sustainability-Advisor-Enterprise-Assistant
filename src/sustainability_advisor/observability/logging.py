"""Structured logging.

JSON lines in deployed environments and a readable console format locally.
Correlation identifiers bound through ``observability.context`` are merged into
every event automatically.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog

from sustainability_advisor.config.models import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure the standard library and structlog once per process."""
    level = logging.getLevelNamesMapping().get(config.level.upper(), logging.INFO)
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if config.json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level, stream=sys.stdout, format="%(message)s", force=True)
    for noisy in ("httpx", "azure", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to a component name."""
    return cast(Any, structlog.get_logger(name))
