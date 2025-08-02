"""
Logging utilities for the Memento framework.

This module provides a centralized logging system with proper configuration,
formatters, and handlers for different environments.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import get_settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for console output."""

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"

        # Format the message
        formatted = super().format(record)

        return formatted


def setup_logger(
    name: str = "memento",
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up a logger with proper configuration.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Whether to use JSON formatting

    Returns:
        Configured logger instance
    """
    # Get settings
    settings = get_settings()

    # Determine log level
    if level is None:
        level = settings.log_level

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    if json_format:
        formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - " "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        console_formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "memento") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **context: Any) -> None:
    """
    Log a message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    # Create a custom log record with extra fields
    record = logger.makeRecord(logger.name, getattr(logging, level.upper()), "", 0, message, (), None)

    # Add extra fields
    record.extra_fields = context

    # Log the record
    logger.handle(record)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)
        self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger

    def log_debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        log_with_context(self._logger, "DEBUG", message, **context)

    def log_info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        log_with_context(self._logger, "INFO", message, **context)

    def log_warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        log_with_context(self._logger, "WARNING", message, **context)

    def log_error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        log_with_context(self._logger, "ERROR", message, **context)

    def log_critical(self, message: str, **context: Any) -> None:
        """Log critical message with context."""
        log_with_context(self._logger, "CRITICAL", message, **context)
